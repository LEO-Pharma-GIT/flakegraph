# SPDX-License-Identifier: Apache-2.0
"""Shared Snowflake connector helpers.

The helpers isolate connection kwargs, stage path normalization, and JSON result
parsing so individual adapters can focus on their table/function contracts.
"""

from __future__ import annotations

import importlib
import json
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast

from kg_processor.adapters.auth.snowflake import SnowflakeAuthConfig, SnowflakeAuthProvider
from kg_processor.ports.auth import AuthProvider

_UNQUOTED_STAGE_RE = re.compile(
    r"^@[A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*){0,2}(?:/[A-Za-z0-9_./=$-]*)?$"
)
_LIKE_META_CHARS_RE = re.compile(r"([\\_%])")
_SNOWFLAKE_IDENTIFIER_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class SnowflakeCursor(Protocol):
    """Minimal DB-API cursor surface used by Snowflake adapters."""

    def execute(self, sql: str, params: Sequence[object] | None = None) -> object:
        """Execute SQL with qmark parameters and return the connector result."""
        ...

    def fetchone(self) -> Sequence[object] | None:
        """Fetch a single Snowflake row or None."""
        ...

    def fetchall(self) -> list[object]:
        """Fetch all remaining Snowflake rows."""
        ...

    def close(self) -> object:
        """Release cursor resources held by the connector."""
        ...


class SnowflakeConnection(Protocol):
    """Minimal DB-API connection surface used by Snowflake adapters."""

    def cursor(self) -> SnowflakeCursor:
        """Create a cursor for executing Snowflake statements."""
        ...

    def commit(self) -> object:
        """Commit the current Snowflake transaction."""
        ...

    def rollback(self) -> object:
        """Roll back the current Snowflake transaction."""
        ...

    def close(self) -> object:
        """Close the Snowflake connection."""
        ...


ConnectorFactory = Callable[..., SnowflakeConnection]


@dataclass(frozen=True)
class SnowflakeConnectionConfig:
    """Connection and session context needed by Snowflake adapters."""

    account: str | None
    host: str | None
    user: str | None
    password: str | None
    authenticator: str | None
    private_key_path: Path | None
    database: str
    schema_name: str
    role: str | None
    warehouse: str | None
    oauth_token: str | None = None
    oauth_token_path: Path | None = Path("/snowflake/session/token")
    store_temporary_credential: bool = False
    auth_provider: AuthProvider | None = None

    @classmethod
    def from_settings(cls, snowflake: Any) -> SnowflakeConnectionConfig:
        """Build the shared connection contract from validated Snowflake settings."""

        if not snowflake.database or not snowflake.schema_name:
            raise ValueError("Snowflake providers require database and schema")
        return cls(
            account=snowflake.account,
            host=snowflake.host,
            user=snowflake.user,
            password=snowflake.password,
            authenticator=snowflake.authenticator,
            private_key_path=snowflake.private_key_path,
            oauth_token=snowflake.oauth_token,
            oauth_token_path=snowflake.oauth_token_path,
            store_temporary_credential=snowflake.store_temporary_credential,
            database=snowflake.database,
            schema_name=snowflake.schema_name,
            role=snowflake.role,
            warehouse=snowflake.warehouse,
        )

    def connect_kwargs(self) -> dict[str, object]:
        """Build connector kwargs from auth plus role/warehouse context."""

        kwargs: dict[str, object] = {
            "database": self.database,
            # Every Snowflake adapter in this package uses DB-API qmark
            # placeholders. The Python connector defaults to pyformat, which
            # tries to apply Python percent-formatting to queries containing
            # `?` and fails before the statement reaches Snowflake.
            "paramstyle": "qmark",
            "schema": self.schema_name,
        }
        kwargs.update(self._auth_provider().connection_kwargs())
        session_context = {
            "role": self.role,
            "warehouse": self.warehouse,
        }
        kwargs.update({key: value for key, value in session_context.items() if value})
        return kwargs

    def _auth_provider(self) -> AuthProvider:
        if self.auth_provider is not None:
            return self.auth_provider
        return SnowflakeAuthProvider(
            SnowflakeAuthConfig(
                account=self.account,
                host=self.host,
                user=self.user,
                password=self.password,
                authenticator=self.authenticator,
                private_key_path=self.private_key_path,
                oauth_token=self.oauth_token,
                oauth_token_path=self.oauth_token_path,
                store_temporary_credential=self.store_temporary_credential,
            )
        )


def load_snowflake_connector() -> ConnectorFactory:
    """Import the optional Snowflake connector only when a Snowflake path is used."""

    module = importlib.import_module("snowflake.connector")
    connect = module.connect
    if not callable(connect):
        raise RuntimeError("snowflake.connector.connect is not callable")
    return cast(ConnectorFactory, connect)


def connect_snowflake(
    config: SnowflakeConnectionConfig,
    connector_factory: ConnectorFactory | None = None,
) -> SnowflakeConnection:
    """Open a Snowflake connection using the provided or default connector factory."""

    factory = connector_factory or load_snowflake_connector()
    return factory(**config.connect_kwargs())


# Comfortably inside any Snowflake credential lifetime, including the OAuth token
# SPCS mounts for a job service. The cost is one handshake per hour per thread.
_MAX_CONNECTION_AGE_SECONDS = 3600.0


class ReusableSnowflakeConnections:
    """Retain one Snowflake connection per calling thread for an adapter."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        connector_factory: ConnectorFactory | None = None,
    ) -> None:
        self.config = config
        self.connector_factory = connector_factory
        self._local = threading.local()
        self._connections: dict[int, SnowflakeConnection] = {}
        self._lock = threading.Lock()

    def get(self) -> SnowflakeConnection:
        """Return the current thread's live connection, opening it lazily.

        A connection is replaced once it reaches :data:`_MAX_CONNECTION_AGE_SECONDS`
        even while it is working. A session cannot outlive the credential that
        opened it, and inside SPCS that credential is an OAuth token with a fixed
        expiry: a connection held for the length of a long run eventually dies
        mid-flight, failing every document in progress at once. Reconnecting on a
        schedule costs one handshake an hour and reads the token file again, which
        SPCS keeps refreshed.
        """

        connection = getattr(self._local, "connection", None)
        opened_at = getattr(self._local, "opened_at", 0.0)
        if connection is not None and monotonic() - opened_at >= _MAX_CONNECTION_AGE_SECONDS:
            self.invalidate()
            connection = None
        if connection is None:
            connection = connect_snowflake(self.config, self.connector_factory)
            self._local.connection = connection
            self._local.opened_at = monotonic()
            with self._lock:
                self._connections[id(connection)] = connection
        return cast(SnowflakeConnection, connection)

    @contextmanager
    def cursor(self, *, commit: bool = False) -> Iterator[SnowflakeCursor]:
        """Run one unit of work on this thread's connection.

        A failure inside the block discards the connection: after a connector
        error its session state cannot be trusted, and the next call opens a
        fresh one. With ``commit`` the block's statements are committed on the
        way out and rolled back on failure, which matters once a caller has
        turned autocommit off for the session.
        """

        connection = self.get()
        cursor = connection.cursor()
        try:
            yield cursor
            if commit:
                connection.commit()
        except Exception:
            if commit:
                with suppress(Exception):
                    connection.rollback()
            with suppress(Exception):
                self.invalidate()
            raise
        finally:
            cursor.close()

    @contextmanager
    def transaction(self) -> Iterator[SnowflakeCursor]:
        """Run several statements as one transaction on this thread's connection.

        The connection is discarded on failure for the same reason as in
        :meth:`cursor`.
        """

        connection = self.get()
        try:
            with snowflake_transaction(connection) as cursor:
                yield cursor
        except Exception:
            with suppress(Exception):
                self.invalidate()
            raise

    def invalidate(self) -> None:
        """Discard the current thread's connection after a connector failure."""

        connection = getattr(self._local, "connection", None)
        if connection is None:
            return
        try:
            connection.close()
        finally:
            self._local.connection = None
            with self._lock:
                self._connections.pop(id(connection), None)

    def close(self) -> None:
        """Close every retained thread connection owned by this adapter."""

        with self._lock:
            connections = list(self._connections.values())
            self._connections.clear()
        self._local.connection = None
        failures: list[Exception] = []
        for connection in connections:
            try:
                connection.close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ExceptionGroup("Failed to close Snowflake connections", failures)


def set_snowflake_autocommit(connection: object, enabled: bool) -> bool:
    """Toggle connector autocommit and report whether the hook was available.

    The Snowflake connector enables autocommit by default, so a multi-statement
    unit of work must disable it before its first statement or its ``commit`` and
    ``rollback`` calls have no effect. The returned flag tells the caller whether
    it owns restoring the connector's default.
    """

    autocommit = getattr(connection, "autocommit", None)
    if not callable(autocommit):
        return False
    autocommit(enabled)
    return True


@contextmanager
def snowflake_transaction(connection: SnowflakeConnection) -> Iterator[SnowflakeCursor]:
    """Commit a block's statements together, or roll every one of them back.

    The connector's default is autocommit, so it is switched off before the
    first statement and restored afterwards whatever happened; the caller's
    error is what propagates, never a failed rollback or restore.
    """

    cursor = connection.cursor()
    autocommit_disabled = False
    try:
        autocommit_disabled = set_snowflake_autocommit(connection, False)
        yield cursor
        connection.commit()
    except Exception:
        with suppress(Exception):
            connection.rollback()
        if autocommit_disabled:
            autocommit_disabled = False
            with suppress(Exception):
                set_snowflake_autocommit(connection, True)
        raise
    finally:
        cursor.close()
        if autocommit_disabled:
            set_snowflake_autocommit(connection, True)


def validate_stage_location(value: str) -> str:
    """Validate an unquoted Snowflake stage URI before interpolating it into SQL."""

    if not _UNQUOTED_STAGE_RE.fullmatch(value):
        raise ValueError(
            "Snowflake stage locations must use unquoted identifiers and safe path "
            f"characters, got: {value}"
        )
    return value


def split_stage_uri(source_uri: str) -> tuple[str, str]:
    """Split an @stage/path URI into the stage name and relative path."""

    value = validate_stage_location(source_uri)
    without_at = value[1:]
    if "/" not in without_at:
        raise ValueError(f"Stage file URI must include a relative path: {source_uri}")
    stage_name, relative_path = without_at.split("/", 1)
    if not relative_path:
        raise ValueError(f"Stage file URI must include a relative path: {source_uri}")
    return f"@{stage_name}", relative_path


def stage_path(stage: str, prefix: str | None = None) -> str:
    """Join and validate a Snowflake stage location and optional prefix."""

    base = validate_stage_location(stage.rstrip("/"))
    if not prefix:
        return base
    normalized_prefix = prefix.strip("/")
    return validate_stage_location(f"{base}/{normalized_prefix}") if normalized_prefix else base


def quote_sql_string(value: str) -> str:
    """Return a single-quoted SQL literal for Snowflake commands without parameters."""

    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def validate_snowflake_identifier(value: str) -> str:
    """Return a safe uppercase Snowflake identifier or reject interpolation."""

    if not _SNOWFLAKE_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Unsafe Snowflake identifier: {value}")
    return value


def compact_json(value: object) -> str:
    """Serialize deterministic compact JSON for bound Snowflake parameters."""

    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def quote_like_pattern(value: str) -> str:
    """Quote a Snowflake SHOW LIKE pattern that should match a literal name."""

    # SHOW ... LIKE accepts pattern wildcards. Escaping here prevents object
    # names such as WH_1 from matching WHA1 during readiness checks.
    return quote_sql_string(_LIKE_META_CHARS_RE.sub(r"\\\1", value))


def scalar_from_first_row(cursor: SnowflakeCursor) -> object:
    """Return the first column of the first row or raise on an empty result."""

    row = cursor.fetchone()
    if row is None or len(row) == 0:
        raise RuntimeError("Snowflake query returned no rows")
    return row[0]


def as_json_object(value: object) -> dict[str, Any]:
    """Normalize Snowflake VARIANT/OBJECT results into a Python mapping."""

    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Expected Snowflake result to be a JSON object")
