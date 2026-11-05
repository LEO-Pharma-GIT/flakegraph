# SPDX-License-Identifier: Apache-2.0
"""Shared response parsing helpers for OpenAI-style chat adapters."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from kg_processor.application.prompt_registry import structured_output_repair_system
from kg_processor.application.structured_output import response_schema_name, strict_json_schema
from kg_processor.domain.consumption import TokenUsage
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    CommunitySummaryResult,
    RefusedChatResponseError,
    StructuredCompletionRequest,
    StructuredCompletionResult,
    TruncatedChatResponseError,
)


def coerce_rating(value: object) -> float:
    """Convert provider rating output into the domain's inclusive 0-10 range."""

    rating = 0.0
    if isinstance(value, int | float):
        rating = float(value)
    elif isinstance(value, str):
        try:
            rating = float(value)
        except ValueError:
            rating = 0.0
    return max(0.0, min(10.0, rating))


def coerce_string_list(value: object) -> list[str]:
    """Return non-blank strings from a provider list field."""

    if not isinstance(value, list):
        return []
    return [text for item in value if isinstance(item, str) and (text := item.strip())]


def optional_string(value: object) -> str:
    """Normalize nullable provider fields without materializing the word ``None``."""

    return value if isinstance(value, str) else ""


def community_summary_result(
    payload: dict[str, object],
    request: CommunitySummaryRequest,
    *,
    usage: TokenUsage,
    provider_metadata: dict[str, Any],
) -> CommunitySummaryResult:
    """Build the community summary from a provider's JSON object.

    Providers sometimes return optional community fields as the wrong scalar
    type. Coercion is intentionally narrow: keep grounded strings and drop
    malformed lists rather than guessing new content.
    """

    raw_findings = payload.get("findings", [])
    findings = raw_findings if isinstance(raw_findings, list) else []
    return CommunitySummaryResult(
        title=optional_string(payload.get("title")) or request.title_seed,
        summary=optional_string(payload.get("summary")),
        rating=coerce_rating(payload.get("rating", 0.0)),
        rating_explanation=optional_string(payload.get("rating_explanation")),
        findings=[
            (optional_string(item.get("summary")), optional_string(item.get("explanation")))
            for item in findings
            if isinstance(item, dict)
        ],
        suggested_questions=coerce_string_list(payload.get("suggested_questions", [])),
        usage=usage,
        provider_metadata=provider_metadata,
    )


def parse_chat_choice_content(payload: Any) -> str:
    """Extract the first message content from an OpenAI-compatible response."""

    if not isinstance(payload, dict):
        raise ValueError("LLM chat response must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM chat response must include at least one choice")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("LLM chat response choice must be an object")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM chat response choice must include a message object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        finish_reason = str(first_choice.get("finish_reason") or "unknown")
        refusal_present = bool(message.get("refusal"))
        raise ValueError(
            "LLM chat response message content must be a non-empty string "
            f"(finish_reason={finish_reason}, refusal_present={refusal_present})"
        )
    return content


def chat_choice_finish_reason(payload: Any) -> str:
    """Return the first choice's finish reason, or an empty string when absent."""

    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    finish_reason = choices[0].get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else ""


def parse_json_object(content: str) -> dict[str, object]:
    """Parse a JSON object response and reject arrays/scalars early."""

    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


@dataclass(frozen=True)
class ChatCompletion:
    """One provider reply, what it consumed, and the state it finished in.

    Usage is returned alongside the content rather than read from the adapter
    afterwards: extraction runs several calls concurrently, so per-adapter state
    would attribute one window's tokens to another.
    """

    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = ""


class ChatResponseError(ValueError):
    """A provider reply that was billed but cannot be used.

    The reply's usage travels with the failure so a window that recovers on a
    later attempt still reports every call it paid for. Its finish reason
    travels with it so an output-limit truncation can be named rather than
    rediscovered as a JSON syntax error on each of the remaining attempts.
    """

    def __init__(
        self,
        message: str,
        *,
        usage: TokenUsage | None = None,
        finish_reason: str = "",
    ) -> None:
        """Retain the failure text alongside the billed usage and finish reason."""

        super().__init__(message)
        self.usage = usage if usage is not None else TokenUsage()
        self.finish_reason = finish_reason


def chat_completion_from_payload(payload: Any) -> ChatCompletion:
    """Build one completion from an OpenAI-compatible chat response body.

    Usage is read before the content is validated. A response whose content is
    null, empty, refused, or filtered was still billed, and reporting zero for
    it makes an incomplete spend total look authoritative.
    """

    usage = TokenUsage.from_provider_payload(
        payload.get("usage") if isinstance(payload, dict) else None
    )
    finish_reason = chat_choice_finish_reason(payload)
    try:
        content = parse_chat_choice_content(payload)
    except ValueError as exc:
        raise ChatResponseError(str(exc), usage=usage, finish_reason=finish_reason) from exc
    return ChatCompletion(content=content, usage=usage, finish_reason=finish_reason)


StructuredChatSender = Callable[
    [list[dict[str, str]], dict[str, object], int, int | None], ChatCompletion
]
JsonChatSender = Callable[[], ChatCompletion]
HttpSender = Callable[[], httpx.Response]
Jitter = Callable[[float], float]
_HTTP_MAX_ATTEMPTS = 4
_HTTP_READ_TIMEOUT_MAX_ATTEMPTS = 2
# A structured window performs several provider calls. A few additional
# schema-preserving retries are cheaper than replaying the complete queue task
# after a burst of HTTP-200 responses with empty or truncated content.
_STRUCTURED_MAX_ATTEMPTS = 6
_HTTP_MAX_RETRY_DELAY_SECONDS = 30.0
_RETRY_AFTER_SPREAD_SECONDS = 1.0
_RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429}
_SERVER_ERROR_STATUS = 500
# Names providers use for a reply cut short by the output-token budget.
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})
# What a reply ends with when the model or a filter in front of it stops it.
_REFUSED_FINISH_REASONS = frozenset({"content_filter"})


def spread_retry_delay(delay: float) -> float:
    """Pick a point in the retry window instead of its exact edge.

    A provider-wide rate limit rejects every concurrent request at nearly the
    same instant. Waiting an identical interval sends the whole fleet back in
    one synchronized burst, which re-triggers the limit; spreading attempts
    across the window lets the earliest ones succeed.
    """

    return random.uniform(0.0, delay)


def send_with_http_retry(
    send: HttpSender,
    *,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Jitter = spread_retry_delay,
) -> httpx.Response:
    """Execute an HTTP request with bounded retries for transient provider failures.

    Timeouts, connection failures, rate limits, request conflicts, and server errors
    can occur after expensive inference has begun. Retrying the individual idempotent
    chat request is safer than failing and replaying an entire document or corpus.
    Authentication, validation, and other permanent client errors return immediately.
    Numeric ``Retry-After`` guidance takes precedence over exponential backoff, and
    every delay is spread so concurrent callers do not retry in one burst.
    """

    last_transport_error: httpx.TransportError | None = None
    for attempt in range(_HTTP_MAX_ATTEMPTS):
        try:
            response = send()
        except httpx.TransportError as exc:
            last_transport_error = exc
            max_attempts = (
                _HTTP_READ_TIMEOUT_MAX_ATTEMPTS
                if isinstance(exc, httpx.ReadTimeout)
                else _HTTP_MAX_ATTEMPTS
            )
            if attempt == max_attempts - 1:
                raise
            sleep(_retry_delay(None, attempt, jitter))
            continue
        if not _is_retryable_status(response.status_code):
            return response
        if attempt == _HTTP_MAX_ATTEMPTS - 1:
            return response
        sleep(_retry_delay(response, attempt, jitter))
    if last_transport_error is not None:  # pragma: no cover - loop re-raises final error
        raise last_transport_error
    raise RuntimeError("HTTP retry loop completed without a response")  # pragma: no cover


def _is_retryable_status(status_code: int) -> bool:
    """Return whether a response status represents a transient provider condition."""

    return status_code in _RETRYABLE_HTTP_STATUSES or status_code >= _SERVER_ERROR_STATUS


def _retry_delay(
    response: httpx.Response | None,
    attempt: int,
    jitter: Jitter = spread_retry_delay,
) -> float:
    """Choose a bounded provider-directed or exponential retry delay."""

    retry_after = _retry_after_seconds(response)
    if retry_after is not None:
        # Provider guidance is a floor rather than a schedule: every rejected
        # caller receives the same value, so the spread is added on top of it.
        return min(
            retry_after + jitter(_RETRY_AFTER_SPREAD_SECONDS),
            _HTTP_MAX_RETRY_DELAY_SECONDS,
        )
    return jitter(min(2.0**attempt, _HTTP_MAX_RETRY_DELAY_SECONDS))


def _retry_after_seconds(response: httpx.Response | None) -> float | None:
    """Read numeric ``Retry-After`` guidance from a rejected response."""

    if response is None:
        return None
    raw_retry_after = response.headers.get("Retry-After")
    if raw_retry_after is None:
        return None
    try:
        return min(max(float(raw_retry_after), 0.0), _HTTP_MAX_RETRY_DELAY_SECONDS)
    except ValueError:
        return None


def _raise_if_truncated(
    error: BaseException,
    model: str,
    usage: TokenUsage,
    *,
    task_name: str = "",
    max_tokens: int | None = None,
) -> None:
    """Report an output-limit truncation instead of retrying an identical request.

    The usage billed so far travels with it, because a truncated reply was paid
    for even though none of it could be used. The message names the task and
    the limit it hit, since only that task's budget can be raised.
    """

    finish_reason = getattr(error, "finish_reason", "")
    if not isinstance(finish_reason, str) or finish_reason.lower() not in (
        _TRUNCATED_FINISH_REASONS
    ):
        return
    task = f" for {task_name}" if task_name else ""
    limit = f" of {max_tokens} tokens" if max_tokens is not None else ""
    truncated = TruncatedChatResponseError(
        f"LLM response{task} from {model} stopped at the output-token limit{limit} "
        f"(finish_reason={finish_reason}); raise the configured max tokens for this task"
    )
    raise _with_billed_usage(truncated, usage) from error


def _final_failure(
    error: BaseException,
    model: str,
    usage: TokenUsage,
    *,
    task_name: str = "",
) -> BaseException:
    """Name the last attempt's failure as a refusal when the reply was declined.

    Only the finish reason decides it, so a reply that was merely malformed
    stays the error it was.
    """

    finish_reason = getattr(error, "finish_reason", "")
    if not isinstance(finish_reason, str) or finish_reason.lower() not in (_REFUSED_FINISH_REASONS):
        return _with_billed_usage(error, usage)
    task = f" for {task_name}" if task_name else ""
    refused = RefusedChatResponseError(
        f"LLM response{task} from {model} was declined (finish_reason={finish_reason})"
    )
    return _with_billed_usage(refused, usage)


def _billed_usage(error: BaseException) -> TokenUsage:
    """Return the usage a failed reply already consumed."""

    usage = getattr(error, "usage", None)
    return usage if isinstance(usage, TokenUsage) else TokenUsage()


def _with_billed_usage(error: BaseException, usage: TokenUsage) -> BaseException:
    """Attach every attempt's billed usage to the failure that ends the call.

    A window that exhausts its repair budget was billed for each attempt. The
    caller records spend from what it is handed, so a failure that carries no
    usage reports an expensive window as free.
    """

    error.usage = usage  # type: ignore[attr-defined]
    return error


def _parse_completion_payload(completion: ChatCompletion) -> dict[str, object]:
    """Parse one reply's JSON object while retaining the state it finished in."""

    try:
        return parse_json_object(completion.content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ChatResponseError(str(exc), finish_reason=completion.finish_reason) from exc


def complete_structured_with_retry(
    request: StructuredCompletionRequest,
    *,
    provider_name: str,
    max_output_tokens: int,
    supports_seed: bool,
    send: StructuredChatSender,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Jitter = spread_retry_delay,
) -> StructuredCompletionResult:
    """Execute one strict request with bounded schema-preserving regeneration.

    Both Azure and OpenAI-compatible adapters share identical structured-output
    semantics. The injected sender owns only endpoint-specific transport. Empty
    message content, malformed choice envelopes, and invalid JSON are retried with
    bounded backoff because providers can return them transiently with HTTP 200.
    A reply cut short by the output-token limit is reported instead, and every
    attempt's usage is accumulated whether or not its content could be used.
    """

    response_format: dict[str, object] = {
        "type": "json_schema",
        "json_schema": {
            "name": response_schema_name(request.task_name),
            "strict": True,
            "schema": strict_json_schema(request.json_schema),
        },
    }
    base_messages = [
        {"role": "system", "content": request.system},
        {"role": "user", "content": request.user},
    ]
    repair_messages = [
        {
            "role": "system",
            "content": structured_output_repair_system(request.system),
        },
        {"role": "user", "content": request.user},
    ]
    payload: dict[str, object] | None = None
    first_error: ValueError | json.JSONDecodeError | None = None
    repair_attempts = 0
    usage = TokenUsage()
    started_at = time.perf_counter()
    for attempt in range(_STRUCTURED_MAX_ATTEMPTS):
        messages = base_messages if attempt == 0 else repair_messages
        try:
            completion = send(
                messages,
                response_format,
                min(request.max_tokens, max_output_tokens),
                request.seed if supports_seed else None,
            )
            # Every attempt was billed, including the ones whose output was
            # unusable, so a repaired window reports what it actually cost.
            usage = usage + completion.usage
            payload = _parse_completion_payload(completion)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            usage = usage + _billed_usage(exc)
            _raise_if_truncated(
                exc,
                request.model,
                usage,
                task_name=request.task_name,
                max_tokens=min(request.max_tokens, max_output_tokens),
            )
            if attempt == _STRUCTURED_MAX_ATTEMPTS - 1:
                raise _final_failure(
                    exc, request.model, usage, task_name=request.task_name
                ) from first_error
            if first_error is None:
                first_error = exc
            repair_attempts += 1
            sleep(_retry_delay(None, attempt, jitter))
    if payload is None:  # pragma: no cover - loop either returns a payload or raises
        raise RuntimeError("structured completion did not produce a payload")
    return StructuredCompletionResult(
        payload=payload,
        usage=usage,
        provider_metadata={
            "provider": provider_name,
            "model": request.model,
            "task_name": request.task_name,
            "format_repair_attempts": repair_attempts,
            "elapsed_seconds": time.perf_counter() - started_at,
            **request.prompt_metadata,
        },
    )


def complete_json_object_with_retry(
    send: JsonChatSender,
    *,
    model: str = "the configured model",
    sleep: Callable[[float], None] = time.sleep,
    jitter: Jitter = spread_retry_delay,
) -> tuple[dict[str, object], TokenUsage]:
    """Regenerate an ordinary JSON chat response after transient content failures.

    Entity-description and community prompts request JSON objects without a
    task-specific schema. Providers can still return an empty message or a
    truncated object with HTTP 200. Retrying that one idempotent prompt prevents
    a single malformed response from replaying an entire graph-finalization task.
    Permanent failure remains bounded by the same attempt budget used for strict
    structured extraction, and a reply cut short by the output-token limit is
    reported rather than regenerated identically.
    """

    first_error: ValueError | json.JSONDecodeError | None = None
    usage = TokenUsage()
    for attempt in range(_STRUCTURED_MAX_ATTEMPTS):
        try:
            completion = send()
            usage = usage + completion.usage
            return _parse_completion_payload(completion), usage
        except (json.JSONDecodeError, ValueError) as exc:
            usage = usage + _billed_usage(exc)
            _raise_if_truncated(exc, model, usage)
            if attempt == _STRUCTURED_MAX_ATTEMPTS - 1:
                raise _final_failure(exc, model, usage) from first_error
            if first_error is None:
                first_error = exc
            sleep(_retry_delay(None, attempt, jitter))
    raise RuntimeError("JSON completion did not produce a payload")  # pragma: no cover
