# SPDX-License-Identifier: Apache-2.0
"""Snowflake authentication provider implementations.

Snowflake connection targets can be the same while credentials differ by
runtime: SPCS receives a short-lived OAuth token file, on-prem jobs may use
key-pair JWT, and interactive use can select browser/password-style connector
auth. Keeping those branches here makes the rest of the Snowflake adapters care
only about query contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SnowflakeAuthConfig:
    """Credential inputs supported by Snowflake connector authentication modes."""

    account: str | None
    host: str | None
    user: str | None
    password: str | None
    authenticator: str | None
    private_key_path: Path | None
    oauth_token: str | None = None
    oauth_token_path: Path | None = Path("/snowflake/session/token")
    store_temporary_credential: bool = False


@dataclass(frozen=True)
class SnowflakeAuthProvider:
    """Builds Snowflake connector kwargs from local, on-prem, or SPCS credentials."""

    config: SnowflakeAuthConfig

    def connection_kwargs(self) -> dict[str, object]:
        """Return connector arguments for the configured authentication mode.

        Snowflake documents programmatic access tokens as password replacements,
        so FlakeGraph accepts the secret through the existing ``password`` setting.
        The Python connector nevertheless expects that value in its ``token``
        argument when ``PROGRAMMATIC_ACCESS_TOKEN`` is selected explicitly. Mapping
        it here keeps credential handling centralized and prevents the connector
        from receiving the same secret as both a password and a token.
        """

        kwargs: dict[str, object] = {}
        authenticator = self.config.authenticator
        uses_programmatic_access_token = (
            authenticator is not None and authenticator.upper() == "PROGRAMMATIC_ACCESS_TOKEN"
        )
        optional = {
            "account": self.config.account,
            "host": self.config.host,
            "user": self.config.user,
            "password": None if uses_programmatic_access_token else self.config.password,
            "authenticator": authenticator,
        }
        kwargs.update({key: value for key, value in optional.items() if value})
        if uses_programmatic_access_token and self.config.password:
            kwargs["token"] = self.config.password
        if authenticator == "oauth":
            token = self._oauth_token()
            if token:
                kwargs["token"] = token
        if self.config.private_key_path:
            kwargs["private_key_file"] = str(self.config.private_key_path)
        if self.config.store_temporary_credential:
            # Browser-based SSO otherwise re-authenticates on every connection,
            # which stalls multi-step CLI work behind a browser tab. The
            # connector caches the short-lived token in the OS credential store
            # when this is enabled, so it stays opt-in rather than silently
            # writing tokens to disk for every deployment.
            kwargs["client_store_temporary_credential"] = True
        return kwargs

    def _oauth_token(self) -> str | None:
        if self.config.oauth_token:
            return self.config.oauth_token
        if self.config.oauth_token_path and self.config.oauth_token_path.exists():
            # SPCS refreshes this file while the container is running. Reading
            # it at connection time avoids carrying an expired token between
            # queue-draining batches.
            return self.config.oauth_token_path.read_text(encoding="utf-8").strip()
        return None
