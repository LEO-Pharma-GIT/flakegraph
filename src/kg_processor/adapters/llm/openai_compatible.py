# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible chat adapter for graph extraction and report prompts."""

from __future__ import annotations

import httpx

from kg_processor.adapters.llm.openai_common import (
    ChatCompletion,
    chat_completion_from_payload,
    community_summary_result,
    complete_json_object_with_retry,
    complete_structured_with_retry,
    optional_string,
    send_with_http_retry,
)
from kg_processor.application.prompt_registry import (
    community_report_prompt,
    entity_description_merge_prompt,
    prompt_metadata,
)
from kg_processor.domain.consumption import TokenUsage
from kg_processor.ports.llm import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    CommunitySummaryRequest,
    CommunitySummaryResult,
    DescriptionMergeRequest,
    DescriptionMergeResult,
    LlmCapabilities,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)

DEFAULT_CHAT_MAX_TOKENS = 8192


class OpenAICompatibleLlmProvider:
    """Execute FlakeGraph LLM tasks through an OpenAI-compatible chat API.

    This adapter is shared by hosted compatible services, Ollama, and other
    servers that implement the chat-completions contract. It owns transport
    serialization and response parsing while extraction policy stays in the
    application layer.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str | None,
        model: str,
        provider_name: str = "openai_compatible",
        timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
        max_output_tokens: int | None = None,
    ) -> None:
        """Configure the endpoint, default model, credentials, and request timeout.

        The default model is used for graph enrichment calls that do not carry
        their own model field. The timeout is retained for those calls, while
        extraction requests may provide a task-specific timeout explicitly.
        """

        if not model.strip():
            raise ValueError("OpenAI-compatible LLM requires an explicit model")
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider_name = provider_name
        # Extraction requests carry their own timeout. Enrichment calls do not,
        # so retain the same configured value here for description/community work.
        self.timeout_seconds = timeout_seconds
        # An operator-configured ceiling wins over the adapter's own. On a shared
        # fleet it is the primary control over how long interactive work waits,
        # because a queue-jumping request is served once a running one finishes.
        self._max_output_tokens = max_output_tokens
        # One client per adapter: httpx clients are thread-safe and keep their
        # connections alive across the thousands of calls an extraction makes.
        self._client = httpx.Client()

    def close(self) -> None:
        """Release retained keep-alive connections owned by this adapter."""

        self._client.close()

    def capabilities(self) -> LlmCapabilities:
        """Describe only features reliably shared by compatible implementations.

        Strict JSON Schema is part of FlakeGraph's adapter contract, but seed
        behavior varies between Ollama, vLLM, and hosted services and is
        therefore not advertised here.
        """

        # Ollama, vLLM, and hosted OpenAI-compatible APIs all understand JSON
        # Schema, but support for a deterministic seed is not universal.
        return LlmCapabilities(
            supports_seed=False,
            max_output_tokens=self._max_output_tokens or DEFAULT_CHAT_MAX_TOKENS,
        )

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Execute one strict structured task and normalize its JSON payload.

        A malformed transport response is regenerated once under the identical
        schema. The method reports that repair attempt in provider metadata so
        operators can distinguish clean responses from recovered ones.
        """

        capabilities = self.capabilities()
        return complete_structured_with_retry(
            request,
            provider_name=self.provider_name,
            max_output_tokens=capabilities.max_output_tokens,
            supports_seed=capabilities.supports_seed,
            send=lambda messages, response_format, max_tokens, seed: self._chat_content(
                model=request.model,
                timeout_seconds=request.timeout_seconds,
                messages=messages,
                temperature=request.temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                seed=seed,
            ),
        )

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        """Synthesize repeated observations into one canonical entity description.

        The adapter sends the shared merge prompt through the configured default
        model and returns prompt metadata alongside the normalized description.
        """

        prompt = entity_description_merge_prompt(request)
        payload, usage = self._chat(
            model=self.model,
            timeout_seconds=self.timeout_seconds,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        )
        return DescriptionMergeResult(
            description=optional_string(payload.get("description")),
            usage=usage,
            provider_metadata={
                "provider": self.provider_name,
                "model": self.model,
                **prompt_metadata(prompt),
            },
        )

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        """Generate a normalized narrative report for one graph community.

        Optional list and rating fields are coerced conservatively so malformed
        provider extras cannot corrupt otherwise valid community artifacts.
        """

        prompt = community_report_prompt(request)
        payload, usage = self._chat(
            model=self.model,
            timeout_seconds=self.timeout_seconds,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        )
        return community_summary_result(
            payload,
            request,
            usage=usage,
            provider_metadata={
                "provider": self.provider_name,
                "model": self.model,
                **prompt_metadata(prompt),
            },
        )

    def _chat(
        self,
        model: str,
        timeout_seconds: int,
        messages: list[dict[str, str]],
    ) -> tuple[dict[str, object], TokenUsage]:
        return complete_json_object_with_retry(
            lambda: self._chat_content(
                model,
                timeout_seconds,
                messages,
                max_tokens=min(
                    DEFAULT_CHAT_MAX_TOKENS, self._max_output_tokens or DEFAULT_CHAT_MAX_TOKENS
                ),
            ),
            model=model,
        )

    def _chat_content(
        self,
        model: str,
        timeout_seconds: int,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_CHAT_MAX_TOKENS,
        response_format: dict[str, object] | None = None,
        seed: int | None = None,
    ) -> ChatCompletion:
        """Submit one chat-completions request and return its textual message.

        The caller supplies schema, token, and determinism options because this
        low-level transport supports both strict extraction and ordinary JSON
        enrichment prompts.
        """

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        request_payload: dict[str, object] = {
            "model": model,
            "temperature": temperature,
            # Graph extraction asks for structured JSON rather than a short
            # chat answer, so compatible servers need enough room to finish
            # the object before parsing or validation begins.
            "max_tokens": max_tokens,
            "response_format": response_format or {"type": "json_object"},
            "messages": messages,
        }
        if seed is not None:
            request_payload["seed"] = seed
        # Compatible runtimes occasionally require transport-level controls
        # that are not part of FlakeGraph's provider-neutral LLM port. A narrow
        # protected hook lets a specialized adapter add those controls without
        # copying request construction, retries, schemas, or response parsing.
        request_payload.update(self._chat_request_overrides())

        def send() -> httpx.Response:
            """Submit one retryable attempt through the retained connection pool."""

            return self._client.post(
                f"{self.endpoint}/chat/completions",
                headers=headers,
                json=request_payload,
                timeout=timeout_seconds,
            )

        response = send_with_http_retry(send)
        response.raise_for_status()
        return chat_completion_from_payload(response.json())

    def _chat_request_overrides(self) -> dict[str, object]:
        """Return runtime-specific top-level chat-completion request fields.

        The generic compatible adapter needs no additions. Subclasses may add
        nonstandard extension fields, but should not replace the shared model,
        messages, schema, sampling, or token-budget contract.
        """

        return {}
