from __future__ import annotations

import httpx
import pytest

from kg_processor.adapters.llm.openai_common import (
    ChatCompletion,
    ChatResponseError,
    chat_completion_from_payload,
    coerce_rating,
    complete_json_object_with_retry,
    complete_structured_with_retry,
    parse_chat_choice_content,
    parse_json_object,
    send_with_http_retry,
)
from kg_processor.domain.consumption import TokenUsage
from kg_processor.ports.llm import (
    RefusedChatResponseError,
    StructuredCompletionRequest,
    TruncatedChatResponseError,
)


def _no_spread(delay: float) -> float:
    """Return the full retry window so delay expectations stay exact."""

    return delay


def test_coerce_rating_clamps_to_domain_bounds() -> None:
    assert coerce_rating("12.5") == 10.0
    assert coerce_rating(-4) == 0.0
    assert coerce_rating("not-a-number") == 0.0


def test_parse_chat_choice_content_rejects_empty_or_null_content() -> None:
    with pytest.raises(ValueError, match="at least one choice"):
        parse_chat_choice_content({"choices": []})
    with pytest.raises(ValueError, match="non-empty string"):
        parse_chat_choice_content({"choices": [{"message": {"content": None}}]})


def test_parse_chat_choice_content_reports_safe_empty_response_diagnostics() -> None:
    """Expose finish state without copying provider refusal or prompt content."""

    with pytest.raises(
        ValueError,
        match=r"finish_reason=length, refusal_present=True",
    ) as error:
        parse_chat_choice_content(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "", "refusal": "sensitive provider text"},
                    }
                ]
            }
        )
    assert "sensitive provider text" not in str(error.value)


def test_parse_json_object_rejects_non_object_payloads() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_json_object("[1, 2, 3]")


def test_http_retry_recovers_server_and_rate_limit_responses() -> None:
    """Retry transient statuses and honor numeric provider delay guidance."""

    request = httpx.Request("POST", "https://example.test/chat")
    responses = iter(
        [
            httpx.Response(500, request=request),
            httpx.Response(429, headers={"Retry-After": "3"}, request=request),
            httpx.Response(200, request=request),
        ]
    )
    delays: list[float] = []

    response = send_with_http_retry(
        lambda: next(responses),
        sleep=delays.append,
        jitter=_no_spread,
    )

    assert response.status_code == 200
    # Provider guidance is a floor: the spread window is added on top of it.
    assert delays == [1.0, 4.0]


def test_http_retry_returns_permanent_client_errors_without_delay() -> None:
    """Do not replay authentication or malformed-request failures."""

    request = httpx.Request("POST", "https://example.test/chat")
    delays: list[float] = []

    response = send_with_http_retry(
        lambda: httpx.Response(401, request=request),
        sleep=delays.append,
    )

    assert response.status_code == 401
    assert delays == []


def test_http_retry_bounds_expensive_read_timeouts() -> None:
    """Avoid multiplying one long inference timeout by every transport attempt."""

    request = httpx.Request("POST", "https://example.test/chat")
    attempts = 0
    delays: list[float] = []

    def timeout() -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("slow inference", request=request)

    with pytest.raises(httpx.ReadTimeout):
        send_with_http_retry(timeout, sleep=delays.append, jitter=_no_spread)

    assert attempts == 2
    assert delays == [1.0]


def test_http_retry_spreads_concurrent_callers_across_the_backoff_window() -> None:
    """A provider-wide rate limit must not send every caller back in one burst."""

    request = httpx.Request("POST", "https://example.test/chat")
    first_delays: list[float] = []

    for _caller in range(24):
        delays: list[float] = []
        responses = iter(
            [httpx.Response(429, request=request), httpx.Response(200, request=request)]
        )
        send_with_http_retry(
            lambda pending=responses: next(pending),  # type: ignore[misc]
            sleep=delays.append,
        )
        first_delays.append(delays[0])

    assert all(0.0 <= delay <= 1.0 for delay in first_delays)
    assert len(set(first_delays)) > 1


def test_structured_retry_recovers_after_multiple_empty_responses() -> None:
    """Treat empty HTTP-success content as a bounded transient provider failure."""

    responses = iter(
        [
            ChatCompletion(content="", usage=TokenUsage(total_tokens=4)),
            ChatCompletion(content="not-json", usage=TokenUsage(total_tokens=4)),
            ChatCompletion(content='{"records": []}', usage=TokenUsage(total_tokens=6)),
        ]
    )
    delays: list[float] = []

    result = complete_structured_with_retry(
        _records_request(),
        provider_name="test",
        max_output_tokens=100,
        supports_seed=False,
        send=lambda *_args: next(responses),
        sleep=delays.append,
        jitter=_no_spread,
    )

    assert result.payload == {"records": []}
    assert result.provider_metadata["format_repair_attempts"] == 2
    assert delays == [1.0, 2.0]


def test_structured_retry_is_bounded_after_persistent_empty_responses() -> None:
    """Stop provider regeneration before an individual queue task can loop forever."""

    attempts = 0

    def empty(*_args: object) -> ChatCompletion:
        """Count every bounded regeneration while returning invalid content."""

        nonlocal attempts
        attempts += 1
        return ChatCompletion(content="")

    with pytest.raises(ValueError):
        complete_structured_with_retry(
            _records_request(),
            provider_name="test",
            max_output_tokens=100,
            supports_seed=False,
            send=empty,
            sleep=lambda _delay: None,
        )

    assert attempts == 6


def test_json_object_retry_recovers_enrichment_response() -> None:
    """Retry ordinary JSON prompts without replaying graph finalization."""

    responses = iter(
        [
            ChatCompletion(content="", usage=TokenUsage(prompt_tokens=5, total_tokens=5)),
            ChatCompletion(content="{broken", usage=TokenUsage(prompt_tokens=5, total_tokens=5)),
            ChatCompletion(
                content='{"description":"grounded"}',
                usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            ),
        ]
    )
    delays: list[float] = []

    payload, usage = complete_json_object_with_retry(
        lambda: next(responses), sleep=delays.append, jitter=_no_spread
    )

    assert payload == {"description": "grounded"}
    assert delays == [1.0, 2.0]
    # Every attempt was billed, so a repaired enrichment reports all three.
    assert usage.total_tokens == 18


def test_chat_completion_records_usage_for_an_unusable_reply() -> None:
    """A billed call that returned nothing usable must still report its tokens."""

    with pytest.raises(ChatResponseError) as error:
        chat_completion_from_payload(
            {
                "choices": [{"finish_reason": "content_filter", "message": {"content": ""}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 0, "total_tokens": 900},
            }
        )

    assert error.value.usage.prompt_tokens == 900
    assert error.value.finish_reason == "content_filter"


def test_structured_retry_reports_tokens_billed_by_a_rejected_reply() -> None:
    """A window that recovers must report every attempt the provider charged for."""

    replies = iter(
        [
            {
                "choices": [{"finish_reason": "content_filter", "message": {"content": ""}}],
                "usage": {"prompt_tokens": 700, "total_tokens": 700},
            },
            {
                "choices": [{"finish_reason": "stop", "message": {"content": '{"records": []}'}}],
                "usage": {"prompt_tokens": 700, "completion_tokens": 20, "total_tokens": 720},
            },
        ]
    )

    result = complete_structured_with_retry(
        _records_request(),
        provider_name="test",
        max_output_tokens=100,
        supports_seed=False,
        send=lambda *_args: chat_completion_from_payload(next(replies)),
        sleep=lambda _delay: None,
    )

    assert result.payload == {"records": []}
    assert result.usage.prompt_tokens == 1400
    assert result.usage.total_tokens == 1420


def test_structured_retry_reports_truncation_instead_of_regenerating() -> None:
    """A reply cut short by the token budget truncates identically on every retry."""

    attempts = 0

    def truncated(*_args: object) -> ChatCompletion:
        nonlocal attempts
        attempts += 1
        return ChatCompletion(
            content='{"records": [{"name": "Ali',
            usage=TokenUsage(completion_tokens=100, total_tokens=100),
            finish_reason="length",
        )

    # The message names the task and the limit it hit: only that budget can grow.
    with pytest.raises(TruncatedChatResponseError, match="for test .*limit of 100 tokens"):
        complete_structured_with_retry(
            _records_request(),
            provider_name="test",
            max_output_tokens=100,
            supports_seed=False,
            send=truncated,
            sleep=lambda _delay: None,
        )

    assert attempts == 1


def test_json_object_retry_reports_truncation_instead_of_regenerating() -> None:
    """Enrichment prompts diagnose the same condition as strict extraction."""

    attempts = 0

    def truncated() -> ChatCompletion:
        nonlocal attempts
        attempts += 1
        return ChatCompletion(content='{"description": "gro', finish_reason="length")

    with pytest.raises(TruncatedChatResponseError, match="output-token limit"):
        complete_json_object_with_retry(truncated, sleep=lambda _delay: None)

    assert attempts == 1


def test_a_reply_declined_on_every_attempt_is_named_a_refusal() -> None:
    """A model that stops its own reply leaves partial JSON, which is not truncation."""

    def declined(*_args: object) -> ChatCompletion:
        return ChatCompletion(content='{"records": [{"name": "Ali', finish_reason="content_filter")

    with pytest.raises(RefusedChatResponseError, match="for test .*declined"):
        complete_structured_with_retry(
            _records_request(),
            provider_name="test",
            max_output_tokens=100,
            supports_seed=False,
            send=declined,
            sleep=lambda _delay: None,
        )
    with pytest.raises(RefusedChatResponseError):
        complete_json_object_with_retry(
            lambda: ChatCompletion(content="", finish_reason="content_filter"),
            sleep=lambda _delay: None,
        )


def test_a_malformed_reply_stays_a_response_error() -> None:
    def malformed(*_args: object) -> ChatCompletion:
        return ChatCompletion(content='{"records": [', finish_reason="stop")

    with pytest.raises(ChatResponseError):
        complete_structured_with_retry(
            _records_request(),
            provider_name="test",
            max_output_tokens=100,
            supports_seed=False,
            send=malformed,
            sleep=lambda _delay: None,
        )


def _records_request() -> StructuredCompletionRequest:
    """Return the strict request shared by the structured retry tests."""

    return StructuredCompletionRequest(
        task_name="test",
        model="model",
        system="system",
        user="user",
        json_schema={
            "type": "object",
            "properties": {"records": {"type": "array", "items": {"type": "string"}}},
            "required": ["records"],
            "additionalProperties": False,
        },
        max_tokens=100,
        timeout_seconds=30,
    )
