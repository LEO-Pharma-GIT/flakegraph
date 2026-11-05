# SPDX-License-Identifier: Apache-2.0
"""Ollama adapter using the native chat API for structured graph tasks."""

from __future__ import annotations

from typing import Any

from kg_processor.adapters.llm.openai_common import (
    ChatCompletion,
    ChatResponseError,
    send_with_http_retry,
)
from kg_processor.adapters.llm.openai_compatible import OpenAICompatibleLlmProvider
from kg_processor.domain.consumption import TokenUsage
from kg_processor.ports.llm import DEFAULT_LLM_TIMEOUT_SECONDS


class OllamaLlmProvider(OpenAICompatibleLlmProvider):
    """Run FlakeGraph prompts through Ollama with model thinking disabled.

    Ollama's native endpoint accepts JSON Schema directly and exposes a stable
    ``think`` switch. Disabling thinking reserves the bounded completion budget
    for graph JSON instead of a separate reasoning trace that FlakeGraph neither
    stores nor consumes.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
        api_key: str | None = None,
        context_window_tokens: int = 32_768,
        max_output_tokens: int | None = None,
    ) -> None:
        """Configure an Ollama server URL, model tag, and optional proxy credential.

        Both ``http://host:11434`` and the familiar OpenAI-compatible
        ``http://host:11434/v1`` form are accepted; native requests always target
        ``/api/chat`` on the same server.
        """

        normalized_endpoint = endpoint.rstrip("/")
        if normalized_endpoint.endswith("/v1"):
            normalized_endpoint = normalized_endpoint[:-3]
        super().__init__(
            endpoint=normalized_endpoint,
            api_key=api_key,
            model=model,
            provider_name="ollama",
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )
        self.context_window_tokens = context_window_tokens

    def _chat_content(
        self,
        model: str,
        timeout_seconds: int,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        response_format: dict[str, object] | None = None,
        seed: int | None = None,
    ) -> ChatCompletion:
        """Submit one native non-streaming Ollama request and return message content.

        The inherited application-facing methods continue to own retries, prompt
        metadata, and result normalization. This override translates only the
        transport controls that differ from OpenAI chat completions.
        """

        options: dict[str, object] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": max(self.context_window_tokens, max_tokens * 2),
        }
        if seed is not None:
            options["seed"] = seed
        request_payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": _ollama_format(response_format),
            "options": options,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = send_with_http_retry(
            lambda: self._client.post(
                f"{self.endpoint}/api/chat",
                headers=headers,
                json=request_payload,
                timeout=timeout_seconds,
            )
        )
        response.raise_for_status()
        payload = response.json()
        # Ollama reports counts natively rather than in an OpenAI usage object.
        # A local model bills nothing, but the tokens are what make the avoided
        # cost measurable, so they are recorded exactly as a hosted call's are —
        # including for a reply whose content turns out to be unusable.
        usage = TokenUsage.from_provider_payload(
            {
                "prompt_tokens": payload.get("prompt_eval_count"),
                "completion_tokens": payload.get("eval_count"),
            }
            if isinstance(payload, dict)
            else None
        )
        finish_reason = _done_reason(payload)
        if not isinstance(payload, dict):
            raise ChatResponseError(
                "Ollama chat response must be a JSON object",
                usage=usage,
                finish_reason=finish_reason,
            )
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ChatResponseError(
                "Ollama chat response is missing message",
                usage=usage,
                finish_reason=finish_reason,
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ChatResponseError(
                "Ollama chat response message content must be a non-empty string "
                f"(done_reason={finish_reason or 'unknown'})",
                usage=usage,
                finish_reason=finish_reason,
            )
        return ChatCompletion(content=content, usage=usage, finish_reason=finish_reason)


def _done_reason(payload: object) -> str:
    """Return Ollama's native equivalent of an OpenAI finish reason."""

    if not isinstance(payload, dict):
        return ""
    done_reason = payload.get("done_reason")
    return done_reason if isinstance(done_reason, str) else ""


def _ollama_format(response_format: dict[str, object] | None) -> str | dict[str, Any]:
    """Convert an OpenAI response-format envelope to Ollama's native format value."""

    if response_format and response_format.get("type") == "json_schema":
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict):
            schema = json_schema.get("schema")
            if isinstance(schema, dict):
                return schema
    return "json"
