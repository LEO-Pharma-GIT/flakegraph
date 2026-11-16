# SPDX-License-Identifier: Apache-2.0
"""Non-mutating Snowflake access checks for account readiness."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnectionConfig,
    SnowflakeCursor,
    connect_snowflake,
    quote_like_pattern,
    stage_path,
)
from kg_processor.adapters.snowflake_cortex_rest import (
    COMPLETE_PATH,
    EMBED_PATH,
    CortexRestClient,
)
from kg_processor.application.redaction import redact_sensitive_text
from kg_processor.application.snowflake_schema import snowflake_schema_columns
from kg_processor.config.settings import Settings

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
# The canaries are tiny model calls, so a short bound keeps a hung endpoint from
# stalling an otherwise quick readiness report.
_CORTEX_CANARY_TIMEOUT_SECONDS = 60
_CORTEX_PARSE_SQL = "SELECT AI_PARSE_DOCUMENT(TO_FILE(?, ?), PARSE_JSON(?)::OBJECT, TRUE)"
_IMAGE_REPOSITORY_PARTS = 3
_COLUMN_RESULT_WIDTH = 2
_SYSTEM_COMPUTE_POOL_PREFIX = "SYSTEM_COMPUTE_POOL"

REQUIRED_SNOWFLAKE_COLUMNS = snowflake_schema_columns()
REQUIRED_SNOWFLAKE_TABLES = tuple(REQUIRED_SNOWFLAKE_COLUMNS)


class SnowflakeAccessCheck(BaseModel):
    """One non-mutating Snowflake readiness check."""

    name: str
    ok: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class SnowflakeAccessReport(BaseModel):
    """Aggregate Snowflake access report consumed by the CLI."""

    ok: bool
    checks: list[SnowflakeAccessCheck]


def run_snowflake_access_check(
    settings: Settings,
    connector_factory: ConnectorFactory | None = None,
    check_cortex: bool = True,
    cortex_client: CortexRestClient | None = None,
) -> SnowflakeAccessReport:
    """Check connection context, objects, stages, compute, images, and Cortex.

    Model and embedding canaries go through the Cortex REST API, as the worker's
    calls do, so an account that has not enabled REST access or a default role
    without the Cortex grant fails here rather than in the first job. Without
    an explicit ``cortex_client`` they reuse this check's own session.
    """

    try:
        config = _connection_config(settings)
        connection = connect_snowflake(config, connector_factory)
    except Exception as exc:
        return SnowflakeAccessReport(
            ok=False,
            checks=[
                SnowflakeAccessCheck(
                    name="connect",
                    ok=False,
                    message="Snowflake connection could not be opened",
                    details=_error_details(exc),
                )
            ],
        )

    cursor = connection.cursor()
    checks: list[SnowflakeAccessCheck] = []
    owned_client: CortexRestClient | None = None
    try:
        checks.append(_guarded_check("connection_context", lambda: _check_context(cursor)))
        if settings.snowflake.warehouse:
            checks.append(
                _guarded_check(
                    "warehouse",
                    lambda: _check_warehouse(cursor, settings.snowflake.warehouse or ""),
                )
            )
        checks.append(
            _guarded_check("target_tables", lambda: _check_required_tables(cursor, settings))
        )
        checks.append(
            _guarded_check(
                "target_table_columns",
                lambda: _check_required_table_columns(cursor, settings),
            )
        )
        if settings.writer.provider in {"snowflake_direct", "snowflake_bulk"}:
            checks.append(
                _guarded_check(
                    "embedding_dimension",
                    lambda: _check_embedding_dimension(cursor, settings),
                )
            )
        checks.extend(_configured_stage_checks(cursor, settings))
        if settings.runtime.runtime == "spcs" or settings.snowflake.compute_pool:
            checks.append(
                _guarded_check(
                    "compute_pool",
                    lambda: _check_compute_pool(cursor, settings.snowflake.compute_pool),
                )
            )
        if settings.runtime.runtime == "spcs" or settings.snowflake.image_repository:
            checks.append(
                _guarded_check(
                    "image_repository",
                    lambda: _check_image_repository(cursor, settings.snowflake.image_repository),
                )
            )
        uses_cortex_rest = "snowflake_cortex" in {
            settings.llm.provider,
            settings.embedding.provider,
        }
        if check_cortex and uses_cortex_rest and cortex_client is None:
            owned_client = CortexRestClient.from_config(
                config, connector_factory, connection=connection
            )
        rest_client = cortex_client or owned_client
        if check_cortex and rest_client is not None and settings.llm.provider == "snowflake_cortex":
            checks.append(
                _guarded_check(
                    "cortex_llm",
                    lambda: _check_cortex_llm(rest_client, settings.llm.model),
                )
            )
        if check_cortex and settings.ocr.provider == "snowflake_cortex":
            checks.append(
                _guarded_check(
                    "cortex_ocr",
                    lambda: _check_cortex_ocr(cursor, settings),
                )
            )
        if (
            check_cortex
            and rest_client is not None
            and settings.embedding.provider == "snowflake_cortex"
        ):
            checks.append(
                _guarded_check(
                    "cortex_embedding",
                    lambda: _check_cortex_embedding(rest_client, settings.embedding.model),
                )
            )
    finally:
        if owned_client is not None:
            owned_client.close()
        cursor.close()
        connection.close()

    return SnowflakeAccessReport(ok=all(check.ok for check in checks), checks=checks)


def _guarded_check(name: str, fn: Callable[[], SnowflakeAccessCheck]) -> SnowflakeAccessCheck:
    try:
        return fn()
    except Exception as exc:
        return SnowflakeAccessCheck(
            name=name,
            ok=False,
            message=f"{name} check failed",
            details=_error_details(exc),
        )


def _check_context(cursor: SnowflakeCursor) -> SnowflakeAccessCheck:
    cursor.execute(
        "SELECT CURRENT_ACCOUNT(), CURRENT_REGION(), CURRENT_USER(), CURRENT_ROLE(), "
        "CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_WAREHOUSE()"
    )
    row = _first_row(cursor)
    details = {
        "account": _row_value(row, 0),
        "region": _row_value(row, 1),
        "user": _row_value(row, 2),
        "role": _row_value(row, 3),
        "database": _row_value(row, 4),
        "schema": _row_value(row, 5),
        "warehouse": _row_value(row, 6),
    }
    return SnowflakeAccessCheck(
        name="connection_context",
        ok=True,
        message="Snowflake connection context is available",
        details=details,
    )


def _check_warehouse(cursor: SnowflakeCursor, warehouse: str) -> SnowflakeAccessCheck:
    safe_warehouse = _identifier(warehouse, "warehouse")
    cursor.execute(f"SHOW WAREHOUSES LIKE {quote_like_pattern(safe_warehouse)}")
    rows = cursor.fetchall()
    return SnowflakeAccessCheck(
        name="warehouse",
        ok=bool(rows),
        message=(
            f"Warehouse is visible: {safe_warehouse}"
            if rows
            else f"Warehouse is not visible: {safe_warehouse}"
        ),
        details={"warehouse": safe_warehouse, "rows": len(rows)},
    )


def _check_required_tables(cursor: SnowflakeCursor, settings: Settings) -> SnowflakeAccessCheck:
    database = _identifier(_required(settings.snowflake.database, "database"), "database")
    schema = _identifier(_required(settings.snowflake.schema_name, "schema"), "schema")
    placeholders = ", ".join("?" for _ in REQUIRED_SNOWFLAKE_TABLES)
    cursor.execute(
        (
            f"SELECT TABLE_NAME FROM {database}.INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = ? AND TABLE_NAME IN ({placeholders})"
        ),
        [schema.upper(), *REQUIRED_SNOWFLAKE_TABLES],
    )
    found = {str(row[0]).upper() for row in cursor.fetchall() if isinstance(row, Sequence) and row}
    missing = sorted(set(REQUIRED_SNOWFLAKE_TABLES) - found)
    return SnowflakeAccessCheck(
        name="target_tables",
        ok=not missing,
        message=(
            "All FlakeGraph tables are visible"
            if not missing
            else "Some FlakeGraph tables are missing or not visible"
        ),
        details={"database": database, "schema": schema, "missing": missing},
    )


def _check_required_table_columns(
    cursor: SnowflakeCursor,
    settings: Settings,
) -> SnowflakeAccessCheck:
    database = _identifier(_required(settings.snowflake.database, "database"), "database")
    schema = _identifier(_required(settings.snowflake.schema_name, "schema"), "schema")
    placeholders = ", ".join("?" for _ in REQUIRED_SNOWFLAKE_TABLES)
    cursor.execute(
        (
            f"SELECT TABLE_NAME, COLUMN_NAME FROM {database}.INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = ? AND TABLE_NAME IN ({placeholders})"
        ),
        [schema.upper(), *REQUIRED_SNOWFLAKE_TABLES],
    )
    found: dict[str, set[str]] = {table: set() for table in REQUIRED_SNOWFLAKE_TABLES}
    for row in cursor.fetchall():
        if isinstance(row, Sequence) and len(row) >= _COLUMN_RESULT_WIDTH:
            table_name = str(row[0]).upper()
            column_name = str(row[1]).upper()
            if table_name in found:
                found[table_name].add(column_name)
    missing = {
        table: sorted(set(columns) - found[table])
        for table, columns in REQUIRED_SNOWFLAKE_COLUMNS.items()
        if set(columns) - found[table]
    }
    return SnowflakeAccessCheck(
        name="target_table_columns",
        ok=not missing,
        message=(
            "All required FlakeGraph table columns are visible"
            if not missing
            else "Some FlakeGraph table columns are missing or not visible"
        ),
        details={"database": database, "schema": schema, "missing": missing},
    )


def _check_embedding_dimension(
    cursor: SnowflakeCursor,
    settings: Settings,
) -> SnowflakeAccessCheck:
    """Compare the configured embedding width with the columns already created.

    A Snowflake VECTOR column fixes its width when the table is created, and the
    writer is the last stage of a run. A mismatch therefore surfaces only after
    every document has been parsed, extracted and summarised — the whole Cortex
    spend of the run — and then rejects the write outright. Read here, it costs
    one query before anything is billed.

    ``SHOW COLUMNS`` is used rather than ``INFORMATION_SCHEMA``, whose
    ``DATA_TYPE`` reports a vector column only as ``VECTOR`` and omits the width
    that has to be compared.
    """

    database = _identifier(_required(settings.snowflake.database, "database"), "database")
    schema = _identifier(_required(settings.snowflake.schema_name, "schema"), "schema")
    cursor.execute(f"SHOW COLUMNS LIKE 'EMBEDDING' IN SCHEMA {database}.{schema}")
    configured = settings.embedding.dimension
    stored: dict[str, int] = {}
    for row in cursor.fetchall():
        table, dimension = _embedding_column_dimension(row)
        if table and dimension is not None and dimension != configured:
            stored[table] = dimension
    return SnowflakeAccessCheck(
        name="embedding_dimension",
        ok=not stored,
        message=(
            "Embedding dimension matches the target tables"
            if not stored
            else (
                f"Target tables store a different embedding width than the configured "
                f"{configured}; choose a model of the stored width or recreate the tables"
            )
        ),
        details={"configured": configured, "stored": stored},
    )


def _embedding_column_dimension(row: object) -> tuple[str, int | None]:
    """Read one ``SHOW COLUMNS`` row as a table name and its vector width.

    The row describes its type as JSON, and a column that is not a vector simply
    carries no dimension.
    """

    if not isinstance(row, Sequence) or isinstance(row, str | bytes):
        return "", None
    values = [item for item in row if isinstance(item, str)]
    table = next(
        (value.upper() for value in values if value.upper() in REQUIRED_SNOWFLAKE_TABLES),
        "",
    )
    for value in values:
        if "dimension" not in value:
            continue
        try:
            declared = json.loads(value)
        except ValueError:
            continue
        if isinstance(declared, dict) and isinstance(declared.get("dimension"), int):
            return table, int(declared["dimension"])
    return table, None


def _configured_stage_checks(
    cursor: SnowflakeCursor,
    settings: Settings,
) -> list[SnowflakeAccessCheck]:
    checks: list[SnowflakeAccessCheck] = []
    if settings.files.source == "snowflake_stage" or settings.ocr.provider == ("snowflake_cortex"):
        checks.append(
            _guarded_check(
                "document_stage",
                lambda: _check_stage_list(
                    cursor,
                    stage_path(
                        _required(settings.snowflake.stage, "document stage"),
                        settings.files.stage_prefix,
                    ),
                    "document_stage",
                ),
            )
        )
    if settings.writer.provider == "snowflake_bulk":
        checks.append(
            _guarded_check(
                "bulk_stage",
                lambda: _check_stage_list(
                    cursor,
                    stage_path(_required(settings.snowflake.bulk_stage, "bulk stage")),
                    "bulk_stage",
                ),
            )
        )
    if settings.runtime.runtime == "spcs":
        checks.append(
            _guarded_check(
                "service_spec_stage",
                lambda: _check_stage_list(
                    cursor,
                    stage_path(
                        _required(settings.snowflake.service_spec_stage, "service spec stage")
                    ),
                    "service_spec_stage",
                ),
            )
        )
    return checks


def _check_stage_list(
    cursor: SnowflakeCursor,
    stage_location: str,
    name: str,
) -> SnowflakeAccessCheck:
    cursor.execute(f"LIST {stage_location}")
    rows = cursor.fetchall()
    return SnowflakeAccessCheck(
        name=name,
        ok=True,
        message=f"Stage can be listed: {stage_location}",
        details={"stage": stage_location, "rows": len(rows)},
    )


def _check_compute_pool(
    cursor: SnowflakeCursor,
    compute_pool: str | None,
) -> SnowflakeAccessCheck:
    pool = _identifier(_required(compute_pool, "compute pool"), "compute pool")
    cursor.execute(f"SHOW COMPUTE POOLS LIKE {quote_like_pattern(pool)}")
    rows = cursor.fetchall()
    is_system_pool = pool.startswith(_SYSTEM_COMPUTE_POOL_PREFIX)
    message = (
        f"Compute pool is visible: {pool}"
        if rows and not is_system_pool
        else f"System compute pool is not compatible with FlakeGraph job services: {pool}"
        if rows
        else f"Compute pool is not visible: {pool}"
    )
    return SnowflakeAccessCheck(
        name="compute_pool",
        ok=bool(rows) and not is_system_pool,
        message=message,
        details={
            "compute_pool": pool,
            "rows": len(rows),
            "requires_dedicated_pool": is_system_pool,
        },
    )


def _check_image_repository(
    cursor: SnowflakeCursor,
    image_repository: str | None,
) -> SnowflakeAccessCheck:
    database, schema, repository = _image_repository_parts(
        _required(image_repository, "image repository")
    )
    cursor.execute(
        "SHOW IMAGE REPOSITORIES LIKE "
        f"{quote_like_pattern(repository)} IN SCHEMA {database}.{schema}"
    )
    rows = cursor.fetchall()
    return SnowflakeAccessCheck(
        name="image_repository",
        ok=bool(rows),
        message=(
            f"Image repository is visible: {database}.{schema}.{repository}"
            if rows
            else f"Image repository is not visible: {database}.{schema}.{repository}"
        ),
        details={"image_repository": f"{database}.{schema}.{repository}", "rows": len(rows)},
    )


def _check_cortex_llm(client: CortexRestClient, model: str) -> SnowflakeAccessCheck:
    """Make an eight-token structured completion through the REST endpoint."""

    client.post(
        COMPLETE_PATH,
        {
            "model": model,
            "messages": [{"role": "user", "content": 'Return {"ok": true}.'}],
            "temperature": 0,
            "max_tokens": 8,
            "response_format": {
                "type": "json",
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                },
            },
            "stream": True,
        },
        timeout_seconds=_CORTEX_CANARY_TIMEOUT_SECONDS,
        operation=f"access-check completion with model {model}",
    )
    return SnowflakeAccessCheck(
        name="cortex_llm",
        ok=True,
        message=f"Cortex REST completion succeeded for model: {model}",
        details={"model": model},
    )


def _check_cortex_ocr(cursor: SnowflakeCursor, settings: Settings) -> SnowflakeAccessCheck:
    """Parse only the first page of one staged PDF to verify Cortex OCR access."""

    stage = stage_path(_required(settings.snowflake.stage, "document stage"))
    location = stage_path(stage, settings.files.stage_prefix)
    cursor.execute(f"LIST {location}")
    relative_path = _first_staged_pdf_path(stage, cursor.fetchall())
    if relative_path is None:
        return SnowflakeAccessCheck(
            name="cortex_ocr",
            ok=False,
            message="Cortex OCR access check requires at least one PDF in the document stage",
            details={"stage": location},
        )
    options = {
        "mode": settings.ocr.snowflake_parse_mode,
        "page_split": True,
        # A single page proves account/model access while bounding the canary cost.
        "page_filter": [{"start": 0, "end": 1}],
    }
    cursor.execute(
        _CORTEX_PARSE_SQL,
        [stage, relative_path, json.dumps(options, sort_keys=True)],
    )
    _first_row(cursor)
    return SnowflakeAccessCheck(
        name="cortex_ocr",
        ok=True,
        message=f"Cortex AI_PARSE_DOCUMENT succeeded for: {relative_path}",
        details={"stage": stage, "relative_path": relative_path, "pages": 1},
    )


def _check_cortex_embedding(client: CortexRestClient, model: str) -> SnowflakeAccessCheck:
    """Embed one short text through the REST endpoint."""

    client.post(
        EMBED_PATH,
        {"model": model, "text": ["FlakeGraph access check"]},
        timeout_seconds=_CORTEX_CANARY_TIMEOUT_SECONDS,
        operation=f"access-check embedding with model {model}",
    )
    return SnowflakeAccessCheck(
        name="cortex_embedding",
        ok=True,
        message=f"Cortex REST embedding succeeded for model: {model}",
        details={"model": model},
    )


def _first_staged_pdf_path(stage: str, rows: list[object]) -> str | None:
    """Return a stage-relative PDF path from Snowflake LIST result shapes."""

    stage_name = stage.lstrip("@").split("/", 1)[0].split(".")[-1]
    marker = f"{stage_name}/"
    for row in rows:
        raw_name: object | None = None
        if isinstance(row, dict):
            raw_name = row.get("name") or row.get("NAME")
        elif isinstance(row, Sequence) and row:
            raw_name = row[0]
        if raw_name is None:
            continue
        name = str(raw_name).lstrip("@")
        if not name.lower().endswith(".pdf"):
            continue
        marker_index = name.lower().find(marker.lower())
        if marker_index >= 0:
            return name[marker_index + len(marker) :]
        return name.split("/", 1)[1] if "/" in name else name
    return None


def _connection_config(settings: Settings) -> SnowflakeConnectionConfig:
    return SnowflakeConnectionConfig.from_settings(settings.snowflake)


def _first_row(cursor: SnowflakeCursor) -> Sequence[object]:
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("Snowflake query returned no rows")
    return row


def _row_value(row: Sequence[object], index: int) -> object | None:
    return row[index] if len(row) > index else None


def _image_repository_parts(value: str) -> tuple[str, str, str]:
    normalized = value[1:] if value.startswith("/") else value
    parts = normalized.replace("/", ".").split(".")
    if len(parts) != _IMAGE_REPOSITORY_PARTS:
        raise ValueError("image repository must be a 3-part identifier")
    return (
        _identifier(parts[0], "image repository database"),
        _identifier(parts[1], "image repository schema"),
        _identifier(parts[2], "image repository name"),
    )


def _identifier(value: str, label: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{label} must be an unquoted Snowflake identifier, got: {value}")
    return value.upper()


def _required(value: str | None, label: str) -> str:
    if not value:
        raise ValueError(f"Snowflake {label} is required")
    return value


def _error_details(exc: Exception) -> dict[str, str]:
    # A connector error can echo the DSN or token it failed with, and this
    # record is what the access report persists and displays.
    return {"type": type(exc).__name__, "message": redact_sensitive_text(str(exc))}
