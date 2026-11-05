# SPDX-License-Identifier: Apache-2.0
"""Shared transport for the Snowflake Cortex REST API.

Cortex model calls go over HTTPS rather than through SQL functions. A SQL
``AI_COMPLETE`` is a warehouse query that holds a concurrency slot for as long
as the model takes, while the warehouse itself only waits; concurrent extraction
then queues every other statement on that warehouse behind model calls. The REST
endpoints run without a warehouse, so model concurrency and SQL concurrency stop
competing for the same slots.

Two credential sources reach the same endpoints:

* inside Snowpark Container Services the container holds an OAuth token that
  Snowflake rotates on disk, which is sent as a bearer token to the host the
  container was given;
* everywhere else the Python connector signs in as configured (password,
  key pair, browser SSO, programmatic access token) and its session token is
  reused, the same way ``snowflake.cortex`` does with an existing session.

:func:`cortex_rest_credentials` is the one place that chooses between them.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from kg_processor.adapters.llm.openai_common import send_with_http_retry
from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnection,
    SnowflakeConnectionConfig,
    connect_snowflake,
)

COMPLETE_PATH = "/api/v2/cortex/inference:complete"
EMBED_PATH = "/api/v2/cortex/inference:embed"

# Snowflake answers a request made with an expired session token with this code,
# sometimes under HTTP 200, and does not perform the operation.
_SESSION_EXPIRED_CODE = "390112"
_UNAUTHORIZED_STATUS = 401
_FORBIDDEN_STATUS = 403
_BAD_REQUEST_STATUS = 400
_UNPROCESSABLE_STATUS = 422
_TOO_MANY_REQUESTS_STATUS = 429
_SUCCESS_STATUSES = range(200, 300)
_OUTPUT_VALIDATION_PREFIX = "json mode output validation error"
# Long enough to identify the failure, short enough to keep a log line readable.
_ERROR_EXCERPT_CHARS = 500
# A connector session token is renewed by reconnecting on this schedule, as the
# SQL adapters do, so a long run does not discover its expiry one call at a time.
_MAX_SESSION_AGE_SECONDS = 3600.0
# Headers that describe the wire encoding of a body already decoded while it was
# read. Carrying them onto the rebuilt response would decode it a second time.
_WIRE_ENCODING_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


class CortexRestError(RuntimeError):
    """A Cortex REST request that failed for a reason retrying will not change."""

    def __init__(self, message: str, *, status_code: int, code: str | None = None) -> None:
        """Keep the HTTP status and Snowflake code beside the readable message."""

        super().__init__(message)
        self.status_code = status_code
        self.code = code


class CortexOutputValidationError(ValueError):
    """The model's output did not satisfy the requested JSON schema.

    This is a ``ValueError`` so the shared structured-output loop regenerates
    the request, as it does for malformed JSON from any other provider. A
    constrained model occasionally emits invalid JSON for a valid schema.
    """


@dataclass(frozen=True)
class RestCredential:
    """Where to send a request and the headers that authenticate it."""

    base_url: str
    headers: dict[str, str]


class CortexRestCredentials(Protocol):
    """Produces credentials for each request and recovers from their expiry."""

    def credential(self) -> RestCredential:
        """Return credentials valid for the next request."""
        ...

    def refresh(self, rejected_authorization: str) -> None:
        """Discard credentials Snowflake rejected, named by their Authorization header."""
        ...

    def close(self) -> None:
        """Release anything held to produce credentials."""
        ...


class OAuthTokenCredentials:
    """Bearer credentials from an OAuth token, re-read before every request.

    Inside SPCS the token file is rewritten every few minutes and each token is
    valid for at most an hour, so a token read once at start-up would fail a
    long run partway through. The token is only valid against the host SPCS
    provides, which is why the host is required rather than derived.
    """

    def __init__(self, config: SnowflakeConnectionConfig) -> None:
        """Bind the token source and host from the shared connection settings."""

        self.config = config

    def credential(self) -> RestCredential:
        """Read the current token and build the bearer headers Snowflake documents."""

        token = self._token()
        return RestCredential(
            base_url=_base_url(self.config.host, self.config.account),
            headers={
                "Authorization": f"Bearer {token}",
                "X-Snowflake-Authorization-Token-Type": "OAUTH",
            },
        )

    def refresh(self, rejected_authorization: str) -> None:
        """Nothing to discard: the next request reads the token file again."""

    def close(self) -> None:
        """Nothing is held between requests."""

    def _token(self) -> str:
        if self.config.oauth_token:
            return self.config.oauth_token
        path = self.config.oauth_token_path
        if path is not None and path.exists():
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token
        raise RuntimeError(
            "Snowflake Cortex REST requires an OAuth token: set snowflake.oauth_token "
            f"or provide a readable token file at {path}"
        )


class ConnectorSessionCredentials:
    """Session-token credentials from one connector sign-in shared by every thread.

    Signing in through the connector supports every authentication mode the
    rest of FlakeGraph accepts, so local runs need no second credential. One
    session serves all threads: the token is a bearer value, and a session per
    thread would mean a browser prompt per thread under SSO.

    An open connection may be lent in, as the access check does with the one it
    already holds; a lent connection is never closed or replaced here.
    """

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        connector_factory: ConnectorFactory | None = None,
        *,
        connection: SnowflakeConnection | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Defer sign-in until the first request unless a connection is lent."""

        self.config = config
        self.connector_factory = connector_factory
        self._lent = connection is not None
        self._connection = connection
        self._clock = clock
        self._opened_at = clock()
        self._lock = threading.Lock()

    def credential(self) -> RestCredential:
        """Return the session's host and token, signing in again when it is old."""

        with self._lock:
            connection = self._current_connection()
            host = getattr(connection, "host", None)
            rest = getattr(connection, "rest", None)
            token = getattr(rest, "token", None)
        if not isinstance(host, str) or not host:
            raise RuntimeError("The Snowflake connection exposes no host for Cortex REST calls")
        if not isinstance(token, str) or not token:
            raise RuntimeError(
                "The Snowflake connection exposes no session token for Cortex REST calls"
            )
        return RestCredential(
            base_url=_base_url(host, None),
            headers={"Authorization": _session_authorization(token)},
        )

    def refresh(self, rejected_authorization: str) -> None:
        """Sign in again on the next request after Snowflake rejects the token.

        Every thread holding the expired token is rejected at about the same
        time. Only the first renewal is acted on: a later rejection naming a
        token that has already been replaced must not discard the fresh session,
        or each thread would sign in again in turn.
        """

        with self._lock:
            if self._lent or self._connection is None:
                return
            current = getattr(getattr(self._connection, "rest", None), "token", None)
            if rejected_authorization == _session_authorization(current):
                self._close_connection()

    def close(self) -> None:
        """Close the session this object opened; a lent one stays with its owner."""

        with self._lock:
            if not self._lent:
                self._close_connection()

    def _current_connection(self) -> SnowflakeConnection:
        if (
            self._connection is not None
            and not self._lent
            and self._clock() - self._opened_at >= _MAX_SESSION_AGE_SECONDS
        ):
            self._close_connection()
        if self._connection is None:
            self._connection = connect_snowflake(self.config, self.connector_factory)
            self._opened_at = self._clock()
        return self._connection

    def _close_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            with suppress(Exception):
                connection.close()


def _session_authorization(token: object) -> str:
    return f'Snowflake Token="{token}"'


def cortex_rest_credentials(
    config: SnowflakeConnectionConfig,
    connector_factory: ConnectorFactory | None = None,
    *,
    connection: SnowflakeConnection | None = None,
) -> CortexRestCredentials:
    """Choose how Cortex REST requests authenticate for this configuration.

    OAuth configuration is what an SPCS job service runs with, and its token is
    sent directly. Every other mode signs in through the connector and reuses
    its session token, so a local run authenticates exactly as its SQL does.
    """

    if config.authenticator is not None and config.authenticator.lower() == "oauth":
        return OAuthTokenCredentials(config)
    return ConnectorSessionCredentials(config, connector_factory, connection=connection)


class CortexRestClient:
    """Send Cortex REST requests with bounded retries and readable failures.

    One client is shared by every thread of an adapter: ``httpx`` clients are
    thread-safe and keep connections alive across the thousands of calls an
    extraction makes.
    """

    def __init__(
        self,
        credentials: CortexRestCredentials,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind credentials, and optionally a transport, sleep, and clock for tests."""

        self.credentials = credentials
        self._client = httpx.Client(transport=transport)
        self._sleep = sleep
        self._clock = clock

    @classmethod
    def from_config(
        cls,
        config: SnowflakeConnectionConfig,
        connector_factory: ConnectorFactory | None = None,
        *,
        connection: SnowflakeConnection | None = None,
    ) -> CortexRestClient:
        """Build a client authenticated the way the configuration selects."""

        return cls(cortex_rest_credentials(config, connector_factory, connection=connection))

    def close(self) -> None:
        """Release keep-alive connections and any session held for credentials."""

        try:
            self._client.close()
        finally:
            self.credentials.close()

    def post(
        self,
        path: str,
        body: Mapping[str, object],
        *,
        timeout_seconds: float,
        operation: str,
    ) -> list[dict[str, Any]]:
        """Send one request and return its JSON body or its server-sent events.

        Rate limits, server errors, and dropped connections are retried with
        bounded backoff that honours ``Retry-After``. An expired credential is
        renewed and the request sent once more. Anything else that fails is
        raised with Snowflake's own message, naming ``operation``.

        ``timeout_seconds`` bounds each attempt from first byte sent to last byte
        read. A streamed reply that keeps trickling would otherwise hold the
        calling thread for as long as the model keeps generating.
        """

        response = self._send_with_retry(path, body, timeout_seconds)
        if _is_session_expired(response):
            self.credentials.refresh(response.request.headers.get("Authorization", ""))
            response = self._send_with_retry(path, body, timeout_seconds)
        _raise_for_failure(response, operation)
        return _response_objects(response, operation)

    def _send_with_retry(
        self,
        path: str,
        body: Mapping[str, object],
        timeout_seconds: float,
    ) -> httpx.Response:
        return send_with_http_retry(
            lambda: self._send_once(path, body, timeout_seconds),
            sleep=self._sleep,
        )

    def _send_once(
        self,
        path: str,
        body: Mapping[str, object],
        timeout_seconds: float,
    ) -> httpx.Response:
        """Send one attempt and read its whole body within the attempt's deadline."""

        credential = self.credentials.credential()
        headers = {
            **credential.headers,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        deadline = self._clock() + timeout_seconds
        with self._client.stream(
            "POST",
            f"{credential.base_url}{path}",
            headers=headers,
            json=dict(body),
            timeout=httpx.Timeout(timeout_seconds),
        ) as response:
            chunks: list[bytes] = []
            for chunk in response.iter_bytes():
                chunks.append(chunk)
                if self._clock() > deadline:
                    raise httpx.ReadTimeout(
                        f"Cortex REST response not complete after {timeout_seconds:g}s",
                        request=response.request,
                    )
        content = b"".join(chunks)
        return httpx.Response(
            _effective_status(response.status_code, content),
            headers={
                name: value
                for name, value in response.headers.items()
                if name.lower() not in _WIRE_ENCODING_HEADERS
            },
            content=content,
            request=response.request,
        )


def _base_url(host: str | None, account: str | None) -> str:
    """Return the HTTPS origin for Cortex requests from a host or account name."""

    if host:
        stripped = host.strip().rstrip("/")
        return stripped if stripped.startswith(("https://", "http://")) else f"https://{stripped}"
    if account:
        return f"https://{account.strip()}.snowflakecomputing.com"
    raise RuntimeError("Snowflake Cortex REST requires snowflake.host or snowflake.account")


def _effective_status(status_code: int, content: bytes) -> int:
    """Report Snowflake's throttling rejection as the rate limit it is.

    Cortex can answer HTTP 400 "all requests were throttled by remote service"
    when the model's capacity is exhausted. It means "try again later", so it is
    surfaced as 429 and takes the same bounded backoff as any other rate limit.
    """

    if status_code == _BAD_REQUEST_STATUS and b"throttled" in content.lower():
        return _TOO_MANY_REQUESTS_STATUS
    return status_code


def _is_session_expired(response: httpx.Response) -> bool:
    if response.status_code == _UNAUTHORIZED_STATUS:
        return True
    payload = _json_or_none(response)
    return isinstance(payload, dict) and str(payload.get("code")) == _SESSION_EXPIRED_CODE


def _raise_for_failure(response: httpx.Response, operation: str) -> None:
    """Raise a failure that says what was attempted and what Snowflake answered."""

    payload = _json_or_none(response)
    code = _error_code(payload)
    if response.status_code in _SUCCESS_STATUSES and code != _SESSION_EXPIRED_CODE:
        return
    message = _error_message(payload, response)
    if response.status_code == _UNPROCESSABLE_STATUS or message.lower().startswith(
        _OUTPUT_VALIDATION_PREFIX
    ):
        raise CortexOutputValidationError(f"Snowflake Cortex {operation} failed: {message}")
    detail = f"HTTP {response.status_code}" + (f", code {code}" if code else "")
    hint = ""
    if response.status_code == _FORBIDDEN_STATUS:
        hint = (
            " (the account must allow the Cortex REST API, and the calling user's "
            "default role needs the SNOWFLAKE.CORTEX_USER database role)"
        )
    raise CortexRestError(
        f"Snowflake Cortex {operation} failed ({detail}): {message}{hint}",
        status_code=response.status_code,
        code=code,
    )


def _response_objects(response: httpx.Response, operation: str) -> list[dict[str, Any]]:
    """Parse a JSON body or a server-sent event stream into JSON objects.

    The completion endpoint streams events; the embedding endpoint answers with
    one JSON document. Both are accepted from either endpoint so a change in how
    Snowflake frames a reply does not break the adapters that read it.
    """

    text = response.text
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type or text.lstrip().startswith(("data:", "event:")):
        events = _server_sent_events(text)
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Snowflake Cortex {operation} returned a body that is not JSON: "
                f"{text[:_ERROR_EXCERPT_CHARS]}"
            ) from exc
        events = [parsed]
    objects: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError(f"Snowflake Cortex {operation} returned a non-object event")
        error = event.get("error")
        if error:
            # A failure after the stream started arrives as an event, under the
            # HTTP 200 already sent, so it must be checked here as well.
            message = _error_text(error)
            if message.lower().startswith(_OUTPUT_VALIDATION_PREFIX):
                raise CortexOutputValidationError(f"Snowflake Cortex {operation} failed: {message}")
            raise CortexRestError(
                f"Snowflake Cortex {operation} failed mid-stream: {message}",
                status_code=response.status_code,
            )
        objects.append(event)
    if not objects:
        raise ValueError(f"Snowflake Cortex {operation} returned an empty response")
    return objects


def _server_sent_events(text: str) -> list[object]:
    """Decode the ``data:`` payloads of an event stream, one JSON value each."""

    events: list[object] = []
    data_lines: list[str] = []

    def flush() -> None:
        if not data_lines:
            return
        data = "\n".join(data_lines)
        data_lines.clear()
        if data.strip() and data.strip() != "[DONE]":
            events.append(json.loads(data))

    for line in text.splitlines():
        if not line.strip():
            flush()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    flush()
    return events


def _json_or_none(response: httpx.Response) -> object:
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _error_code(payload: object) -> str | None:
    if isinstance(payload, dict) and payload.get("code") is not None:
        return str(payload["code"])
    return None


def _error_message(payload: object, response: httpx.Response) -> str:
    if isinstance(payload, dict):
        for key in ("message", "error"):
            value = payload.get(key)
            if value:
                return _error_text(value)
    text = response.text.strip()
    return text[:_ERROR_EXCERPT_CHARS] if text else response.reason_phrase


def _error_text(value: object) -> str:
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, str) and message:
            return message
        return json.dumps(value, sort_keys=True)
    return str(value).strip()
