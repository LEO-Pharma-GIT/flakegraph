from __future__ import annotations

from pathlib import Path

from kg_processor.adapters.auth.snowflake import SnowflakeAuthConfig, SnowflakeAuthProvider
from kg_processor.adapters.snowflake import SnowflakeConnectionConfig


def test_snowflake_connection_config_reads_oauth_token_file(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("spcs-token\n", encoding="utf-8")
    config = SnowflakeConnectionConfig(
        account="EXAMPLE_ACCOUNT",
        host=None,
        user=None,
        password=None,
        authenticator="oauth",
        private_key_path=None,
        oauth_token_path=token_path,
        database="KG_DB",
        schema_name="GRAPH",
        role="KG_PROCESSOR_ROLE",
        warehouse="KG_PROCESSOR_WH",
    )

    kwargs = config.connect_kwargs()

    assert kwargs["authenticator"] == "oauth"
    assert kwargs["paramstyle"] == "qmark"
    assert kwargs["token"] == "spcs-token"


def test_snowflake_connection_config_prefers_explicit_oauth_token(
    tmp_path: Path,
) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("stale-file-token\n", encoding="utf-8")
    config = SnowflakeConnectionConfig(
        account="EXAMPLE_ACCOUNT",
        host=None,
        user=None,
        password=None,
        authenticator="oauth",
        private_key_path=None,
        oauth_token="azure-oauth-token",
        oauth_token_path=token_path,
        database="KG_DB",
        schema_name="GRAPH",
        role="KG_PROCESSOR_ROLE",
        warehouse="KG_PROCESSOR_WH",
    )

    kwargs = config.connect_kwargs()

    assert kwargs["token"] == "azure-oauth-token"


def test_snowflake_auth_provider_reads_oauth_token_file_for_each_connection(
    tmp_path: Path,
) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("first-token\n", encoding="utf-8")
    provider = SnowflakeAuthProvider(
        SnowflakeAuthConfig(
            account="EXAMPLE_ACCOUNT",
            host=None,
            user=None,
            password=None,
            authenticator="oauth",
            private_key_path=None,
            oauth_token_path=token_path,
        )
    )

    first_kwargs = provider.connection_kwargs()
    token_path.write_text("second-token\n", encoding="utf-8")
    second_kwargs = provider.connection_kwargs()

    assert first_kwargs["token"] == "first-token"
    assert second_kwargs["token"] == "second-token"


def test_snowflake_auth_provider_maps_keypair_private_key_file(tmp_path: Path) -> None:
    private_key = tmp_path / "rsa_key.p8"
    provider = SnowflakeAuthProvider(
        SnowflakeAuthConfig(
            account="EXAMPLE_ACCOUNT",
            host=None,
            user="KG_SERVICE_USER",
            password=None,
            authenticator="SNOWFLAKE_JWT",
            private_key_path=private_key,
        )
    )

    kwargs = provider.connection_kwargs()

    assert kwargs["user"] == "KG_SERVICE_USER"
    assert kwargs["authenticator"] == "SNOWFLAKE_JWT"
    assert kwargs["private_key_file"] == str(private_key)


def test_snowflake_auth_provider_maps_pat_secret_to_connector_token() -> None:
    provider = SnowflakeAuthProvider(
        SnowflakeAuthConfig(
            account="EXAMPLE_ACCOUNT",
            host=None,
            user="KG_SERVICE_USER",
            password="programmatic-access-token",
            authenticator="PROGRAMMATIC_ACCESS_TOKEN",
            private_key_path=None,
        )
    )

    kwargs = provider.connection_kwargs()

    assert kwargs["authenticator"] == "PROGRAMMATIC_ACCESS_TOKEN"
    assert kwargs["token"] == "programmatic-access-token"
    assert "password" not in kwargs


def test_snowflake_auth_provider_matches_pat_authenticator_case_insensitively() -> None:
    provider = SnowflakeAuthProvider(
        SnowflakeAuthConfig(
            account="EXAMPLE_ACCOUNT",
            host=None,
            user="KG_SERVICE_USER",
            password="programmatic-access-token",
            authenticator="programmatic_access_token",
            private_key_path=None,
        )
    )

    kwargs = provider.connection_kwargs()

    assert kwargs["token"] == "programmatic-access-token"
    assert "password" not in kwargs


def test_snowflake_connection_config_accepts_injected_auth_provider() -> None:
    config = SnowflakeConnectionConfig(
        account=None,
        host=None,
        user=None,
        password=None,
        authenticator=None,
        private_key_path=None,
        auth_provider=StaticAuthProvider(),
        database="KG_DB",
        schema_name="GRAPH",
        role="KG_ROLE",
        warehouse="KG_WH",
    )

    kwargs = config.connect_kwargs()

    assert kwargs == {
        "database": "KG_DB",
        "paramstyle": "qmark",
        "schema": "GRAPH",
        "authenticator": "externalbrowser",
        "user": "reviewer@example.com",
        "role": "KG_ROLE",
        "warehouse": "KG_WH",
    }


class StaticAuthProvider:
    def connection_kwargs(self) -> dict[str, object]:
        return {
            "authenticator": "externalbrowser",
            "user": "reviewer@example.com",
        }


def test_snowflake_auth_provider_omits_credential_cache_by_default() -> None:
    provider = SnowflakeAuthProvider(
        SnowflakeAuthConfig(
            account="EXAMPLE_ACCOUNT",
            host=None,
            user="someone@example.com",
            password=None,
            authenticator="externalbrowser",
            private_key_path=None,
        )
    )

    assert "client_store_temporary_credential" not in provider.connection_kwargs()


def test_snowflake_auth_provider_enables_credential_cache_when_requested() -> None:
    provider = SnowflakeAuthProvider(
        SnowflakeAuthConfig(
            account="EXAMPLE_ACCOUNT",
            host=None,
            user="someone@example.com",
            password=None,
            authenticator="externalbrowser",
            private_key_path=None,
            store_temporary_credential=True,
        )
    )

    assert provider.connection_kwargs()["client_store_temporary_credential"] is True
