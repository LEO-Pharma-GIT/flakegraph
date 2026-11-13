from __future__ import annotations

from typing import Any

from kg_processor.application.redaction import (
    is_sensitive_key,
    redact_sensitive_data,
    redact_sensitive_text,
)


def test_redact_sensitive_data_recurses_without_mutating_original_payload() -> None:
    payload: dict[str, Any] = {
        "api_key": "llm-secret",
        "database_url": "postgresql://user:password@database/flakegraph",
        "provider": "openai_compatible",
        "nested": [
            {
                "password": "snowflake-secret",
                "connection_string": "AccountKey=secret",
                "api_key_header": "X-API-Key",
                "api_key_prefix": "",
            }
        ],
    }

    redacted = redact_sensitive_data(payload)

    assert redacted == {
        "api_key": "***",
        "database_url": "***",
        "provider": "openai_compatible",
        "nested": [
            {
                "password": "***",
                "connection_string": "***",
                "api_key_header": "X-API-Key",
                "api_key_prefix": "",
            }
        ],
    }
    assert payload["api_key"] == "llm-secret"
    assert payload["nested"][0]["password"] == "snowflake-secret"


def test_is_sensitive_key_covers_provider_credential_names() -> None:
    assert is_sensitive_key("api_key")
    assert is_sensitive_key("mineru_api_key")
    assert is_sensitive_key("oauth_token")
    assert is_sensitive_key("database_url")
    assert is_sensitive_key("authorization")
    assert is_sensitive_key("private_key_path")
    assert is_sensitive_key("client_secret")
    assert is_sensitive_key("artifact_access_key_id")
    assert not is_sensitive_key("api_key_header")
    assert not is_sensitive_key("api_key_prefix")


def test_redact_sensitive_text_covers_dsns_headers_queries_and_jwts() -> None:
    """Keep common connector and HTTP exception formats from leaking credentials."""

    text = (
        "dsn=postgresql://user:db-secret@db/graph password=plain-secret "
        "Authorization: Bearer bearer-secret "
        "url=https://api.example/path?api_key=query-secret "
        "jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature"
    )

    redacted = redact_sensitive_text(text)

    assert "db-secret" not in redacted
    assert "plain-secret" not in redacted
    assert "bearer-secret" not in redacted
    assert "query-secret" not in redacted
    assert "eyJ" not in redacted
    assert "postgresql://user:***@db/graph" in redacted


def test_redaction_covers_quoted_assignments_and_scalar_urls() -> None:
    quoted = redact_sensitive_text('{"api_key":"super-secret","provider":"test"}')
    scalar = redact_sensitive_data("https://storage.example/blob?sv=1&sig=signature-secret&sp=r")

    assert "super-secret" not in quoted
    assert "signature-secret" not in scalar


def test_structured_redaction_preserves_arbitrary_free_text() -> None:
    payload = [
        {"entity": "Token: A Novel"},
        {"description": "Users set a password: rules apply"},
        {"note": "The board secret: to longevity is focus"},
    ]

    assert redact_sensitive_data(payload) == payload
