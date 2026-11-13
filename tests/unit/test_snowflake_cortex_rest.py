from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from snowflake_fakes import CONFIG, FakeConnection, FakeCortexServer, sse_reply

from kg_processor.adapters.snowflake_cortex_rest import (
    COMPLETE_PATH,
    ConnectorSessionCredentials,
    CortexOutputValidationError,
    CortexRestClient,
    CortexRestError,
    OAuthTokenCredentials,
    cortex_rest_credentials,
)

_OK = {"choices": [{"delta": {"content": "{}"}}]}


def _post(client: CortexRestClient) -> list[dict[str, object]]:
    return client.post(COMPLETE_PATH, {"model": "m"}, timeout_seconds=30, operation="completion")


def test_spcs_oauth_token_is_sent_as_a_bearer_token_and_reread_per_request(
    tmp_path: Path,
) -> None:
    """SPCS rotates the token file, so every request must carry the current token."""

    token_file = tmp_path / "token"
    token_file.write_text("first-token\n", encoding="utf-8")
    credentials = OAuthTokenCredentials(
        dataclasses.replace(
            CONFIG,
            authenticator="oauth",
            host="spcs-internal.example.snowflakecomputing.com",
            oauth_token_path=token_file,
        )
    )

    first = credentials.credential()
    token_file.write_text("rotated-token", encoding="utf-8")
    second = credentials.credential()

    assert first.base_url == "https://spcs-internal.example.snowflakecomputing.com"
    assert first.headers == {
        "Authorization": "Bearer first-token",
        "X-Snowflake-Authorization-Token-Type": "OAUTH",
    }
    assert second.headers["Authorization"] == "Bearer rotated-token"


def test_oauth_credentials_explain_a_missing_token(tmp_path: Path) -> None:
    credentials = OAuthTokenCredentials(
        dataclasses.replace(CONFIG, authenticator="oauth", oauth_token_path=tmp_path / "absent")
    )

    with pytest.raises(RuntimeError, match="requires an OAuth token"):
        credentials.credential()


def test_oauth_credentials_fall_back_to_the_account_url() -> None:
    credentials = OAuthTokenCredentials(
        dataclasses.replace(CONFIG, authenticator="oauth", oauth_token="token")
    )

    assert credentials.credential().base_url == "https://account.snowflakecomputing.com"


def test_credentials_follow_the_configured_authenticator() -> None:
    oauth = dataclasses.replace(CONFIG, authenticator="OAUTH", oauth_token="token")

    assert isinstance(cortex_rest_credentials(oauth), OAuthTokenCredentials)
    assert isinstance(cortex_rest_credentials(CONFIG), ConnectorSessionCredentials)


def test_connector_session_token_is_shared_by_every_request() -> None:
    """One sign-in serves all threads; a session per thread would prompt per thread under SSO."""

    opened: list[FakeConnection] = []

    def connect(**_: object) -> FakeConnection:
        opened.append(FakeConnection())
        return opened[-1]

    credentials = ConnectorSessionCredentials(CONFIG, connect)

    first = credentials.credential()
    second = credentials.credential()

    assert len(opened) == 1
    assert first == second
    assert first.base_url == "https://example-account.snowflakecomputing.com"
    assert first.headers == {"Authorization": 'Snowflake Token="session-token"'}
    credentials.close()
    assert opened[0].closed


def test_connector_session_is_renewed_on_schedule_and_on_refresh() -> None:
    opened: list[FakeConnection] = []
    now = [0.0]

    def connect(**_: object) -> FakeConnection:
        opened.append(FakeConnection())
        return opened[-1]

    credentials = ConnectorSessionCredentials(CONFIG, connect, clock=lambda: now[0])

    credentials.credential()
    now[0] = 3601.0
    rejected = credentials.credential().headers["Authorization"]
    credentials.refresh(rejected)
    credentials.credential()

    assert len(opened) == 3
    assert opened[0].closed and opened[1].closed and not opened[2].closed


def test_only_the_first_rejection_of_a_session_token_signs_in_again() -> None:
    """Threads rejected for the same expired token must not each replace the session."""

    opened: list[FakeConnection] = []

    def connect(**_: object) -> FakeConnection:
        opened.append(FakeConnection())
        opened[-1].rest.token = f"session-{len(opened)}"
        return opened[-1]

    credentials = ConnectorSessionCredentials(CONFIG, connect)
    expired = credentials.credential().headers["Authorization"]

    credentials.refresh(expired)
    renewed = credentials.credential()
    credentials.refresh(expired)

    assert renewed.headers["Authorization"] == 'Snowflake Token="session-2"'
    assert credentials.credential() == renewed
    assert len(opened) == 2


def test_a_lent_connection_is_used_but_never_closed() -> None:
    connection = FakeConnection()
    credentials = ConnectorSessionCredentials(CONFIG, connection=connection)

    rejected = credentials.credential().headers["Authorization"]
    credentials.refresh(rejected)
    credentials.close()

    assert not connection.closed


def test_rate_limits_back_off_for_at_least_the_retry_after_interval() -> None:
    server = FakeCortexServer(
        httpx.Response(429, headers={"Retry-After": "2"}, json={"message": "too many requests"}),
        sse_reply(_OK),
    )

    events = _post(server.client())

    assert events == [_OK]
    assert len(server.requests) == 2
    assert len(server.sleeps) == 1
    assert 2.0 <= server.sleeps[0] <= 3.0


def test_rate_limits_give_up_after_a_bounded_number_of_attempts() -> None:
    server = FakeCortexServer(httpx.Response(429, json={"message": "too many requests"}))

    with pytest.raises(CortexRestError, match="HTTP 429") as caught:
        _post(server.client())

    assert caught.value.status_code == 429
    assert len(server.requests) == 4
    assert all(delay <= 30.0 for delay in server.sleeps)


def test_throttled_bad_request_is_retried_as_a_rate_limit() -> None:
    server = FakeCortexServer(
        httpx.Response(400, json={"message": "all requests were throttled by remote service"}),
        sse_reply(_OK),
    )

    assert _post(server.client()) == [_OK]
    assert len(server.requests) == 2


def test_server_timeouts_are_retried() -> None:
    server = FakeCortexServer(
        httpx.Response(503, json={"message": "inference timed out"}),
        sse_reply(_OK),
    )

    assert _post(server.client()) == [_OK]


def test_expired_session_token_is_refreshed_and_the_request_sent_again() -> None:
    """Snowflake reports an expired session token under HTTP 200 with code 390112."""

    server = FakeCortexServer(
        httpx.Response(200, json={"code": "390112", "message": "Your session has expired."}),
        sse_reply(_OK),
    )

    assert _post(server.client()) == [_OK]
    assert server.credentials.rejected == ["Bearer token-0"]
    assert [request.headers["Authorization"] for request in server.requests] == [
        "Bearer token-0",
        "Bearer token-1",
    ]


def test_a_second_rejection_after_refresh_is_reported() -> None:
    server = FakeCortexServer(httpx.Response(401, json={"message": "Invalid token"}))

    with pytest.raises(CortexRestError, match="HTTP 401"):
        _post(server.client())

    assert server.credentials.refreshes == 1
    assert len(server.requests) == 2


def test_unknown_model_is_reported_with_snowflakes_message_and_no_retry() -> None:
    server = FakeCortexServer(httpx.Response(400, json={"message": "unknown model 'nope'"}))

    with pytest.raises(CortexRestError, match="completion failed \\(HTTP 400\\): unknown model"):
        _post(server.client())

    assert len(server.requests) == 1


def test_schema_invalid_model_output_is_a_value_error_for_the_repair_loop() -> None:
    server = FakeCortexServer(
        httpx.Response(422, json={"message": "json mode output validation error: bad JSON"})
    )

    with pytest.raises(CortexOutputValidationError, match="json mode output validation"):
        _post(server.client())

    assert issubclass(CortexOutputValidationError, ValueError)


def test_an_error_event_inside_a_stream_is_raised() -> None:
    server = FakeCortexServer(sse_reply(_OK, {"error": {"message": "model overloaded"}}))

    with pytest.raises(CortexRestError, match="mid-stream: model overloaded"):
        _post(server.client())


def test_event_stream_parsing_skips_the_done_marker() -> None:
    server = FakeCortexServer(
        httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices": []}\n\nevent: done\ndata: [DONE]\n\n',
        )
    )

    assert _post(server.client()) == [{"choices": []}]


def test_a_reply_that_outlives_the_attempt_deadline_is_abandoned() -> None:
    """A trickling stream must not hold the calling thread past the request timeout."""

    now = [0.0]

    def slow_reply(_request: httpx.Request) -> httpx.Response:
        def chunks() -> Iterator[bytes]:
            for _ in range(3):
                now[0] += 20.0
                yield b'data: {"choices": []}\n\n'

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=chunks())

    server = FakeCortexServer(slow_reply)
    client = CortexRestClient(
        server.credentials,
        transport=httpx.MockTransport(server._handle),
        sleep=server.sleeps.append,
        clock=lambda: now[0],
    )

    with pytest.raises(httpx.ReadTimeout):
        client.post(COMPLETE_PATH, {"model": "m"}, timeout_seconds=30, operation="completion")

    # A read timeout is retried once, as for the other HTTP model adapters.
    assert len(server.requests) == 2
