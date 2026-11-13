from __future__ import annotations

import httpx
import pytest
from snowflake_fakes import CONFIG, FakeCortexServer, completion_reply, sse_reply

from kg_processor.adapters.llm.openai_common import ChatResponseError
from kg_processor.adapters.llm.snowflake_cortex import (
    SnowflakeCortexLlmProvider,
    _completion_from_events,
    _snowflake_json_schema,
)
from kg_processor.adapters.snowflake_cortex_rest import CortexRestError
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    DescriptionMergeRequest,
    StructuredCompletionRequest,
    TruncatedChatResponseError,
)


def _extraction_request(**overrides: object) -> StructuredCompletionRequest:
    fields: dict[str, object] = {
        "task_name": "entity_extraction",
        "model": "llama3.3-70b",
        "system": "Extract grounded entities.",
        "user": "Document text",
        "json_schema": {
            "type": "object",
            "properties": {
                "records": {"type": "array", "maxItems": 10, "items": {"type": "string"}}
            },
            "required": ["records"],
        },
        "prompt_metadata": {"prompt_revision": "entity@1"},
    }
    fields.update(overrides)
    return StructuredCompletionRequest.model_validate(fields)


def test_snowflake_cortex_sends_a_structured_completion_to_the_rest_endpoint() -> None:
    """Cortex should receive the joined prompt, sampling settings, and a strict schema."""

    server = FakeCortexServer(
        completion_reply(
            {"records": ["Alice"]},
            usage={"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        )
    )
    provider = _provider(server)

    result = provider.complete_structured(_extraction_request())

    request = server.requests[0]
    assert request.method == "POST"
    assert str(request.url) == (
        "https://example-account.snowflakecomputing.com/api/v2/cortex/inference:complete"
    )
    assert request.headers["Authorization"] == "Bearer token-0"
    assert "text/event-stream" in request.headers["Accept"]
    body = server.bodies[0]
    assert body["model"] == "llama3.3-70b"
    assert body["messages"] == [
        {"role": "user", "content": "Extract grounded entities.\n\nDocument text"}
    ]
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 4096
    assert body["stream"] is True
    assert body["response_format"]["type"] == "json"
    schema = body["response_format"]["schema"]
    assert schema["required"] == ["records"]
    assert schema["additionalProperties"] is False
    assert "maxItems" not in schema["properties"]["records"]
    assert result.payload == {"records": ["Alice"]}
    assert result.usage.prompt_tokens == 9
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 12
    assert result.provider_metadata["usage"] == {
        "prompt_tokens": 9,
        "completion_tokens": 3,
        "total_tokens": 12,
    }
    assert result.provider_metadata["model"] == "llama3.3-70b"
    assert result.provider_metadata["prompt_revision"] == "entity@1"
    provider.close()
    assert server.credentials.closed


def test_snowflake_cortex_applies_the_requested_completion_timeout() -> None:
    """The port contract's per-request timeout must bound the HTTP call.

    A stalled completion otherwise holds an extraction thread for the whole
    retry budget instead of failing the one attempt.
    """

    server = FakeCortexServer(completion_reply({"records": []}))

    _provider(server).complete_structured(_extraction_request(timeout_seconds=42))

    assert server.requests[0].extensions["timeout"]["read"] == 42


def test_snowflake_cortex_bounds_enrichment_calls_without_a_request_timeout() -> None:
    """Enrichment carries no per-request timeout, so the adapter's own applies."""

    server = FakeCortexServer(completion_reply({"description": "Grounded."}))
    provider = SnowflakeCortexLlmProvider(
        CONFIG, "llama3.3-70b", timeout_seconds=77, client=server.client()
    )

    provider.merge_entity_description(
        DescriptionMergeRequest(
            entity_name="Alice",
            entity_type="PERSON",
            descriptions=["Alice is present."],
            evidence=[],
        )
    )

    assert server.requests[0].extensions["timeout"]["read"] == 77


def test_snowflake_cortex_regenerates_after_output_validation_failure() -> None:
    """Cortex's 422 for schema-invalid model output enters the shared repair loop."""

    server = FakeCortexServer(
        httpx.Response(
            422,
            json={"message": "json mode output validation error: invalid JSON"},
        ),
        completion_reply({"records": ["Alice"]}, usage={"total_tokens": 5}),
    )

    result = _provider(server).complete_structured(_extraction_request())

    assert len(server.requests) == 2
    assert result.payload == {"records": ["Alice"]}
    assert result.provider_metadata["format_repair_attempts"] == 1


def test_snowflake_cortex_reports_truncation_instead_of_regenerating() -> None:
    """A reply cut at the output limit truncates identically when repeated."""

    server = FakeCortexServer(
        sse_reply(
            {
                "model": "llama3.3-70b",
                "choices": [{"delta": {"content": '{"records": ["Al'}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4096},
            }
        )
    )

    with pytest.raises(TruncatedChatResponseError) as caught:
        _provider(server).complete_structured(_extraction_request())

    assert len(server.requests) == 1
    assert caught.value.usage.completion_tokens == 4096  # type: ignore[attr-defined]


def test_snowflake_cortex_fails_permanent_errors_without_retrying() -> None:
    """A missing grant cannot be fixed by asking again, and the error says what to fix."""

    server = FakeCortexServer(
        httpx.Response(403, json={"message": "Not Authorized", "code": "399504"})
    )

    with pytest.raises(CortexRestError, match="CORTEX_USER") as caught:
        _provider(server).complete_structured(_extraction_request())

    assert len(server.requests) == 1
    assert caught.value.status_code == 403
    assert "completion with model llama3.3-70b" in str(caught.value)


def test_snowflake_cortex_summarizes_community() -> None:
    server = FakeCortexServer(
        completion_reply(
            {
                "title": "Acme",
                "summary": "Acme community",
                "rating": 8,
                "rating_explanation": "Important because Alice anchors Acme.",
                "findings": [{"summary": "Finding", "explanation": "Because"}],
                "suggested_questions": ["Who anchors Acme?"],
            },
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )
    )

    summary = _provider(server).summarize_community(
        CommunitySummaryRequest(title_seed="Acme", members=["Alice", "Acme"], relations=[])
    )

    body = server.bodies[0]
    assert body["max_tokens"] == 2048
    assert body["temperature"] == 0
    assert body["response_format"]["schema"]["required"] == [
        "title",
        "summary",
        "rating",
        "rating_explanation",
        "findings",
        "suggested_questions",
    ]
    assert summary.title == "Acme"
    assert summary.rating == 8
    assert summary.findings == [("Finding", "Because")]
    assert summary.suggested_questions == ["Who anchors Acme?"]
    assert summary.usage.total_tokens == 120
    assert summary.provider_metadata["prompt_name"] == "community_report"


def test_snowflake_cortex_clamps_out_of_range_community_rating() -> None:
    """Do not let one provider rating violate the Community domain contract."""

    server = FakeCortexServer(
        completion_reply(
            {
                "title": "Acme",
                "summary": "Acme community",
                "rating": 99,
                "rating_explanation": "Provider exceeded the scale.",
                "findings": [],
                "suggested_questions": [],
            }
        )
    )

    summary = _provider(server).summarize_community(
        CommunitySummaryRequest(title_seed="Acme", members=["Acme"], relations=[])
    )

    assert summary.rating == 10.0


def test_snowflake_cortex_merges_entity_description() -> None:
    server = FakeCortexServer(
        completion_reply(
            {"description": "Alice Smith works at Acme Corp."},
            usage={"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
        )
    )

    result = _provider(server).merge_entity_description(
        DescriptionMergeRequest(
            entity_name="Alice Smith",
            entity_type="PERSON",
            descriptions=["Alice is present.", "Alice works at Acme Corp."],
            evidence=["Alice Smith works at Acme Corp."],
        )
    )

    body = server.bodies[0]
    assert "merge observed descriptions" in body["messages"][0]["content"]
    assert body["response_format"]["schema"]["required"] == ["description"]
    assert result.description == "Alice Smith works at Acme Corp."
    assert result.usage.total_tokens == 48
    assert result.provider_metadata["usage"]["total_tokens"] == 48
    assert result.provider_metadata["prompt_name"] == "entity_description_merge"


def test_snowflake_cortex_drops_null_enrichment_strings() -> None:
    description_server = FakeCortexServer(completion_reply({"description": None}))
    description = _provider(description_server).merge_entity_description(
        DescriptionMergeRequest(
            entity_name="Alice",
            entity_type="PERSON",
            descriptions=["Alice is present."],
            evidence=[],
        )
    )
    summary_server = FakeCortexServer(
        completion_reply(
            {
                "title": None,
                "summary": None,
                "rating": None,
                "rating_explanation": None,
                "findings": [{"summary": None, "explanation": None}],
                "suggested_questions": [None, "Valid question?"],
            }
        )
    )
    summary = _provider(summary_server).summarize_community(
        CommunitySummaryRequest(title_seed="Fallback", members=[], relations=[])
    )

    assert description.description == ""
    assert summary.title == "Fallback"
    assert summary.summary == ""
    assert summary.rating_explanation == ""
    assert summary.findings == [("", "")]
    assert summary.suggested_questions == ["Valid question?"]


def test_snowflake_cortex_reads_a_non_streamed_reply() -> None:
    """A single JSON body is read the same way as a stream of deltas."""

    completion, metadata = _completion_from_events(
        [
            {
                "model": "mistral-large2",
                "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }
        ]
    )

    assert completion.content == '{"ok": true}'
    assert completion.usage.total_tokens == 6
    assert metadata == {
        "model": "mistral-large2",
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }


def test_snowflake_cortex_reads_content_list_deltas() -> None:
    completion, _metadata = _completion_from_events(
        [
            {"choices": [{"delta": {"content_list": [{"type": "text", "text": '{"ok":'}]}}]},
            {"choices": [{"delta": {"content_list": [{"type": "text", "text": " true}"}]}}]},
        ]
    )

    assert completion.content == '{"ok": true}'


def test_snowflake_cortex_reports_billed_usage_for_an_empty_reply() -> None:
    """An empty reply was still billed, so its usage travels with the failure."""

    with pytest.raises(ChatResponseError, match="no content") as caught:
        _completion_from_events(
            [{"choices": [{"delta": {"content": ""}}], "usage": {"prompt_tokens": 30}}]
        )

    assert caught.value.usage.prompt_tokens == 30


def test_snowflake_cortex_removes_only_unsupported_schema_constraints() -> None:
    """Project full contracts onto Cortex's documented JSON Schema subset."""

    schema = _snowflake_json_schema(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "maxLength": 160},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "items": {
                    "type": "array",
                    "maxItems": 10,
                    "items": {"type": "string"},
                },
            },
            "required": ["name", "score", "items"],
        }
    )

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["name", "score", "items"]
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["score"] == {"type": "number"}
    assert schema["properties"]["items"] == {
        "type": "array",
        "items": {"type": "string"},
    }


def _provider(server: FakeCortexServer) -> SnowflakeCortexLlmProvider:
    """Build an adapter bound to a deterministic fake Cortex endpoint."""

    return SnowflakeCortexLlmProvider(CONFIG, "llama3.3-70b", client=server.client())
