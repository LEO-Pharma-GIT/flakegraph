# SPDX-License-Identifier: Apache-2.0
"""Snowflake Cortex LLM adapter using the Cortex REST completion endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kg_processor.adapters.llm.openai_common import (
    ChatCompletion,
    ChatResponseError,
    community_summary_result,
    complete_json_object_with_retry,
    complete_structured_with_retry,
    optional_string,
)
from kg_processor.adapters.snowflake import ConnectorFactory, SnowflakeConnectionConfig
from kg_processor.adapters.snowflake_cortex_rest import COMPLETE_PATH, CortexRestClient
from kg_processor.application.prompt_registry import (
    community_report_prompt,
    entity_description_merge_prompt,
    prompt_metadata,
)
from kg_processor.application.structured_output import strict_json_schema
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

# Snowflake's structured-output implementation supports JSON Schema structure,
# enums, and references, but deliberately rejects these validation constraints.
# FlakeGraph removes them only from the transport schema and still validates the
# response against its complete Pydantic contract after the provider call.
_UNSUPPORTED_SNOWFLAKE_SCHEMA_KEYWORDS = {
    "contains",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "maxContains",
    "maximum",
    "maxItems",
    "maxLength",
    "maxProperties",
    "minContains",
    "minimum",
    "minItems",
    "minLength",
    "minProperties",
    "multipleOf",
    "patternProperties",
    "propertyNames",
    "uniqueItems",
}


class SnowflakeCortexLlmProvider:
    """Execute provider-neutral LLM tasks through the Cortex REST completion API.

    ``inference:complete`` is the REST form of ``AI_COMPLETE``: it takes the same
    ``{"type": "json", "schema": ...}`` response format and enforces it for every
    model Cortex serves. The OpenAI-style chat endpoint ignores a response format
    for models outside the Claude and OpenAI families, which would silently drop
    strict output for the default model, so it is not used here.
    """

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        default_model: str,
        connector_factory: ConnectorFactory | None = None,
        timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
        client: CortexRestClient | None = None,
    ) -> None:
        """Configure the credentials, default model, and enrichment request timeout.

        Extraction requests carry their own timeout through the shared port
        contract. Graph enrichment calls do not, so they use this value.
        """

        self.config = config
        self.default_model = default_model
        self.connector_factory = connector_factory
        self.timeout_seconds = timeout_seconds
        self._client = client or CortexRestClient.from_config(config, connector_factory)

    def close(self) -> None:
        """Release retained HTTP connections and any session held for credentials."""

        self._client.close()

    def capabilities(self) -> LlmCapabilities:
        """Describe the Cortex capabilities available to extraction orchestration.

        Native strict output is enabled through Cortex response formats; seed
        support is omitted because Cortex model behavior is model-specific.
        """

        return LlmCapabilities(
            supports_seed=False,
            max_output_tokens=8192,
        )

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Execute a shared strict request and normalize Cortex output metadata.

        Cortex model-output validation failures participate in the same bounded,
        schema-preserving repair loop as other LLM adapters. Permanent Snowflake
        input, access, or configuration errors fail immediately. Model usage from
        the successful attempt remains in provider metadata while only the
        schema-conforming object is returned as the task payload.
        """

        successful_metadata: dict[str, Any] = {}

        def send(
            messages: list[dict[str, str]],
            _response_format: dict[str, object],
            max_tokens: int,
            _seed: int | None,
        ) -> ChatCompletion:
            """Run one Cortex attempt and expose its reply to the shared parser."""

            completion, metadata = self._complete(
                model=request.model,
                prompt="\n\n".join(message["content"] for message in messages),
                temperature=request.temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json",
                    "schema": _snowflake_json_schema(request.json_schema),
                },
                timeout_seconds=request.timeout_seconds,
            )
            successful_metadata.clear()
            successful_metadata.update(metadata)
            return completion

        result = complete_structured_with_retry(
            request,
            provider_name="snowflake_cortex",
            max_output_tokens=self.capabilities().max_output_tokens,
            supports_seed=False,
            send=send,
        )
        return StructuredCompletionResult(
            payload=result.payload,
            usage=result.usage,
            provider_metadata={
                **result.provider_metadata,
                "model": successful_metadata.get("model", request.model),
                "usage": successful_metadata.get("usage"),
            },
        )

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        """Merge entity descriptions with Cortex structured output."""

        prompt = entity_description_merge_prompt(request)
        payload, metadata = self._complete_enrichment_with_retry(
            prompt=f"{prompt.system}\n\n{prompt.user}",
            max_tokens=1024,
            response_format=_description_merge_response_format(),
        )
        return DescriptionMergeResult(
            description=optional_string(payload.get("description")),
            usage=_metadata_usage(metadata),
            provider_metadata={
                "provider": "snowflake_cortex",
                "model": metadata.get("model", self.default_model),
                "usage": metadata.get("usage"),
                **prompt_metadata(prompt),
            },
        )

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        """Generate a community report with Cortex structured output."""

        prompt = community_report_prompt(request)
        payload, metadata = self._complete_enrichment_with_retry(
            prompt=f"{prompt.system}\n\n{prompt.user}",
            max_tokens=2048,
            response_format=_community_response_format(),
        )
        return community_summary_result(
            payload,
            request,
            usage=_metadata_usage(metadata),
            provider_metadata={
                "provider": "snowflake_cortex",
                "model": metadata.get("model", self.default_model),
                "usage": metadata.get("usage"),
                **prompt_metadata(prompt),
            },
        )

    def _complete_enrichment_with_retry(
        self,
        *,
        prompt: str,
        max_tokens: int,
        response_format: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, Any]]:
        """Retry transient Cortex structured-output validation failures."""

        metadata: dict[str, Any] = {}

        def send() -> ChatCompletion:
            completion, attempt_metadata = self._complete(
                model=self.default_model,
                prompt=prompt,
                temperature=0,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout_seconds=self.timeout_seconds,
            )
            metadata.clear()
            metadata.update(attempt_metadata)
            return completion

        payload, usage = complete_json_object_with_retry(send, model=self.default_model)
        return payload, {**metadata, "token_usage": usage}

    def _complete(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, object],
        timeout_seconds: int,
    ) -> tuple[ChatCompletion, dict[str, Any]]:
        """Send one completion and separate the reply from Cortex usage metadata.

        The prompt travels as one user message, which is what ``AI_COMPLETE``
        makes of a string prompt, so the model reads the same thing over REST as
        over SQL. Metadata stays apart from the reply so traces can show
        usage and model details without leaking provider response shapes.
        """

        events = self._client.post(
            COMPLETE_PATH,
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
                "stream": True,
            },
            timeout_seconds=timeout_seconds,
            operation=f"completion with model {model}",
        )
        return _completion_from_events(events)


def _completion_from_events(
    events: list[dict[str, Any]],
) -> tuple[ChatCompletion, dict[str, Any]]:
    """Assemble one reply from streamed deltas or a single non-streamed body.

    Usage is read before the content is validated, because a reply with no
    usable content was still billed. Cortex sends an empty usage object on
    intermediate events and the counts on a later one, so the last non-empty
    usage wins.
    """

    parts: list[str] = []
    raw_usage: Mapping[str, Any] | None = None
    finish_reason = ""
    model: str | None = None
    for event in events:
        if isinstance(event.get("model"), str):
            model = event["model"]
        usage = event.get("usage")
        if isinstance(usage, Mapping) and usage:
            raw_usage = usage
        choice = _first_choice(event)
        if choice is None:
            continue
        reason = choice.get("finish_reason")
        if isinstance(reason, str) and reason:
            finish_reason = reason
        parts.append(_choice_text(choice))
    usage_payload = dict(raw_usage) if raw_usage is not None else None
    token_usage = TokenUsage.from_provider_payload(usage_payload)
    content = "".join(parts)
    if not content.strip():
        raise ChatResponseError(
            "Snowflake Cortex completion returned no content "
            f"(finish_reason={finish_reason or 'unknown'})",
            usage=token_usage,
            finish_reason=finish_reason,
        )
    metadata: dict[str, Any] = {"usage": usage_payload}
    if model is not None:
        metadata["model"] = model
    return (
        ChatCompletion(content=content, usage=token_usage, finish_reason=finish_reason),
        metadata,
    )


def _first_choice(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    choices = event.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        return choices[0]
    return None


def _choice_text(choice: Mapping[str, Any]) -> str:
    """Read the text of one choice, streamed as a delta or returned whole."""

    for key in ("delta", "message"):
        part = choice.get(key)
        if isinstance(part, Mapping):
            content = part.get("content")
            if isinstance(content, str):
                return content
            content_list = part.get("content_list")
            if isinstance(content_list, list):
                return "".join(
                    item["text"]
                    for item in content_list
                    if isinstance(item, Mapping) and isinstance(item.get("text"), str)
                )
    messages = choice.get("messages")
    return messages if isinstance(messages, str) else ""


def _description_merge_response_format() -> dict[str, object]:
    return {
        "type": "json",
        "schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
            },
            "required": ["description"],
        },
    }


def _community_response_format() -> dict[str, object]:
    return {
        "type": "json",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "rating": {"type": "number"},
                "rating_explanation": {"type": "string"},
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "explanation": {"type": "string"},
                        },
                        "required": ["summary", "explanation"],
                    },
                },
                "suggested_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "title",
                "summary",
                "rating",
                "rating_explanation",
                "findings",
                "suggested_questions",
            ],
        },
    }


def _snowflake_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Create a strict schema containing only constraints Snowflake supports.

    The shared schema is copied before normalization because it is also used for
    local validation, cache fingerprints, and other provider adapters. This
    provider-specific projection prevents Snowflake limitations from weakening
    those contracts globally.
    """

    normalized = strict_json_schema(schema)
    _remove_unsupported_snowflake_constraints(normalized)
    return normalized


def _remove_unsupported_snowflake_constraints(value: object) -> None:
    """Recursively remove documented unsupported Cortex schema keywords in place."""

    if isinstance(value, dict):
        schema_maps = ("properties", "$defs", "definitions", "patternProperties")
        for map_name in schema_maps:
            schema_map = value.get(map_name)
            if isinstance(schema_map, dict):
                for child_schema in schema_map.values():
                    _remove_unsupported_snowflake_constraints(child_schema)
        for keyword in _UNSUPPORTED_SNOWFLAKE_SCHEMA_KEYWORDS:
            value.pop(keyword, None)
        for key, child in value.items():
            if key in schema_maps:
                continue
            _remove_unsupported_snowflake_constraints(child)
    elif isinstance(value, list):
        for child in value:
            _remove_unsupported_snowflake_constraints(child)


def _metadata_usage(metadata: Mapping[str, Any]) -> TokenUsage:
    """Read the usage accumulated across an enrichment call's retries."""

    usage = metadata.get("token_usage")
    return usage if isinstance(usage, TokenUsage) else TokenUsage()
