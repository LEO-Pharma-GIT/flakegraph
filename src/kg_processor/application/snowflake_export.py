# SPDX-License-Identifier: Apache-2.0
"""Materialize a Snowflake-resident graph as local inspection artifacts.

Snowflake runs write the graph straight into ``KG_*`` tables, while every
inspection and evaluation tool reads the local parquet snapshot. Without a
bridge, a graph built on Snowflake cannot be compared against a gold fixture or
against a local run, which makes benchmark parity unverifiable on the one
runtime where the pipeline is most likely to be operated.

The export deliberately reads tables rather than reconstructing domain objects.
The Snowflake schema is generated from the same field definitions the artifact
writer uses, so column names already line up, and a row-level copy keeps the
export correct as the graph model evolves.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnectionConfig,
    SnowflakeCursor,
    connect_snowflake,
    validate_snowflake_identifier,
)
from kg_processor.application.output_ownership import (
    assert_replaceable_output,
    write_ownership_manifest,
)
from kg_processor.application.snowflake_schema import (
    snowflake_schema_array_columns,
    snowflake_schema_columns,
)
from kg_processor.config.settings import Settings

# Artifact table name -> Snowflake table name. The artifact side is the parquet
# file stem that inspection reads; the Snowflake side is where a run wrote it.
_ARTIFACT_TABLES: dict[str, str] = {
    "documents": "KG_DOCUMENT",
    "pages": "KG_PAGE",
    "blocks": "KG_BLOCK",
    "assets": "KG_ASSET",
    "chunks": "KG_CHUNK",
    "nodes": "KG_NODE",
    "edges": "KG_EDGE",
    "edge_observations": "KG_EDGE_OBSERVATION",
    "evidence": "KG_EVIDENCE",
    "entity_sources": "KG_ENTITY_SOURCE",
    "communities": "KG_COMMUNITY",
    "community_findings": "KG_COMMUNITY_FINDING",
    "discarded_windows": "KG_DISCARDED_WINDOW",
    "rejected_records": "KG_REJECTED_RECORD",
    "failed_documents": "KG_FAILED_DOCUMENT",
}

# Bookkeeping columns exist in Snowflake but are not part of the graph contract.
_EXCLUDED_COLUMNS = frozenset({"UPDATED_AT", "CREATED_AT"})
_DIAGNOSTIC_TABLES = {
    "run_report": ("KG_RUN_REPORT", "REPORT"),
    "graph_metrics": ("KG_GRAPH_METRICS", "METRICS"),
}
# Unlike arbitrary VARIANT objects, relation quantities have a list contract.
_LIST_VARIANT_COLUMNS = {"KG_EDGE_OBSERVATION": frozenset({"QUANTITIES"})}
_DIAGNOSTIC_ROW_WIDTH = 3


def export_snowflake_graph(
    settings: Settings,
    graph_id: str,
    output_path: Path,
    connector_factory: ConnectorFactory | None = None,
    *,
    job_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Write one graph's Snowflake tables to a local artifact directory.

    Returns per-table row counts and the selected diagnostic identity. Reports
    and metrics come from one job/run pair: the graph's only one, or the one the
    caller selects. A graph holding several pairs exports its tables without
    diagnostics unless one is selected, rather than combining unrelated
    consumption. Graph tables represent the graph's current Snowflake snapshot.
    """

    import pandas as pd  # noqa: PLC0415 — optional at import time, required here

    if not graph_id:
        raise ValueError("graph export requires a graph id")

    config = SnowflakeConnectionConfig.from_settings(settings.snowflake)
    schema_columns = snowflake_schema_columns()
    counts: dict[str, int] = {}
    frames: dict[str, Any] = {}

    connection = connect_snowflake(config, connector_factory)
    try:
        cursor = connection.cursor()
        try:
            for artifact_name, table in _ARTIFACT_TABLES.items():
                columns = [
                    name for name in schema_columns[table] if name.upper() not in _EXCLUDED_COLUMNS
                ]
                frame = _read_table(cursor, pd, config, table, columns, graph_id)
                frames[artifact_name] = frame
                counts[artifact_name] = len(frame.index)
            diagnostics, identity = _read_diagnostics(
                cursor, config, graph_id, job_id=job_id, run_id=run_id
            )
        finally:
            cursor.close()
    finally:
        connection.close()

    _write_artifacts(output_path, frames, diagnostics)
    return {
        "graph_id": graph_id,
        "output": str(output_path),
        "job_id": identity[0] if identity else None,
        "run_id": identity[1] if identity else None,
        "tables": counts,
        "rows": sum(counts.values()),
    }


def _read_table(
    cursor: SnowflakeCursor,
    pd: Any,
    config: SnowflakeConnectionConfig,
    table: str,
    columns: list[str],
    graph_id: str,
) -> Any:
    """Read one graph-scoped table into a frame using artifact column names.

    Columns are taken from the schema generator rather than the result cursor so
    the projection stays pinned to the declared contract, and a column added to
    Snowflake out of band cannot silently change the artifact shape.

    The database and schema come from configuration and cannot be bound as
    parameters, so they are validated as unquoted identifiers before they reach
    the statement text.
    """

    qualified = ".".join(
        (
            _qualifier(config.database, "database"),
            _qualifier(config.schema_name, "schema"),
            validate_snowflake_identifier(table),
        )
    )
    projection = ", ".join(validate_snowflake_identifier(column) for column in columns)
    cursor.execute(f"SELECT {projection} FROM {qualified} WHERE GRAPH_ID = ?", [graph_id])
    rows = cursor.fetchall()
    list_columns = snowflake_schema_array_columns().get(table, frozenset()) | (
        _LIST_VARIANT_COLUMNS.get(table, frozenset())
    )
    normalized: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(f"Snowflake {table} returned an invalid row")
        normalized.append(
            [
                _list_value(value, table, column) if column in list_columns else value
                for column, value in zip(columns, row, strict=True)
            ]
        )
    frame = pd.DataFrame(normalized, columns=columns)
    # Inspection reads lower-case artifact columns; Snowflake returns them upper.
    frame.columns = [name.lower() for name in frame.columns]
    return frame


def _list_value(value: Any, table: str, column: str) -> Any:
    """Decode connector JSON only for fields declared to hold graph lists."""

    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        raise ValueError(f"Snowflake {table}.{column} must contain a JSON list") from None
    if decoded is not None and not isinstance(decoded, list):
        raise ValueError(f"Snowflake {table}.{column} must contain a JSON list")
    return decoded


def _read_diagnostics(
    cursor: SnowflakeCursor,
    config: SnowflakeConnectionConfig,
    graph_id: str,
    *,
    job_id: str | None,
    run_id: str | None,
) -> tuple[dict[str, dict[str, Any]], tuple[str, str] | None]:
    """Select one shared job/run identity for persisted report and metrics.

    A file-queue graph can have diagnostics from several batches, and a revised
    graph from several runs. There is no meaningful single total to infer from
    their ordering, so without a selector such a graph exports no diagnostics,
    and a selector that matches several pairs is refused.
    """

    selected: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    identities: set[tuple[str, str]] = set()
    for name, (table, payload_column) in _DIAGNOSTIC_TABLES.items():
        records: dict[tuple[str, str], dict[str, Any]] = {}
        selected[name] = records
        qualified = ".".join(
            (
                _qualifier(config.database, "database"),
                _qualifier(config.schema_name, "schema"),
                table,
            )
        )
        predicates = ["GRAPH_ID = ?"]
        params = [graph_id]
        if job_id is not None:
            predicates.append("JOB_ID = ?")
            params.append(job_id)
        if run_id is not None:
            predicates.append("RUN_ID = ?")
            params.append(run_id)
        cursor.execute(
            f"SELECT JOB_ID, RUN_ID, {payload_column} FROM {qualified} WHERE "
            + " AND ".join(predicates),
            params,
        )
        for row in cursor.fetchall():
            identity, payload = _diagnostic_record(row, table)
            if identity in records:
                raise ValueError(f"Snowflake {table} contains duplicate job/run diagnostics")
            records[identity] = payload
            identities.add(identity)
    if len(identities) != 1:
        if job_id is None and run_id is None:
            return {name: {} for name in _DIAGNOSTIC_TABLES}, None
        raise ValueError(
            "Snowflake graph has diagnostics for multiple jobs/runs; "
            "select one with --job-id and, when needed, --run-id"
            if identities
            else "No Snowflake diagnostics match the requested graph/job/run"
        )
    identity = next(iter(identities))
    diagnostics = {name: records.get(identity, {}) for name, records in selected.items()}
    report = diagnostics["run_report"]
    if identity in selected["run_report"]:
        for field, value in zip(
            ("graph_id", "job_id", "run_id"), (graph_id, *identity), strict=True
        ):
            if field in report and report[field] != value:
                raise ValueError(f"Snowflake run report {field} conflicts with its stored identity")
            report[field] = value
    return diagnostics, identity


def _diagnostic_record(row: object, table: str) -> tuple[tuple[str, str], dict[str, Any]]:
    """Validate one diagnostic row before indexing its identity and object payload."""

    if (
        not isinstance(row, Sequence)
        or isinstance(row, (str, bytes))
        or len(row) != _DIAGNOSTIC_ROW_WIDTH
    ):
        raise ValueError(f"Snowflake {table} returned an invalid diagnostic row")
    return (str(row[0]), str(row[1])), _diagnostic_payload(row[2], table)


def _diagnostic_payload(value: Any, table: str) -> dict[str, Any]:
    """Decode a Snowflake VARIANT diagnostic object without accepting other shapes."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(f"Snowflake {table} diagnostics must contain a JSON object") from None
    if not isinstance(value, dict):
        raise ValueError(f"Snowflake {table} diagnostics must contain a JSON object")
    return dict(value)


def _qualifier(value: str, label: str) -> str:
    """Normalize and validate one configured name-resolution component.

    Snowflake resolves unquoted identifiers case-insensitively, so a configured
    lower-case name is upper-cased to match the identifier contract rather than
    being rejected.
    """

    try:
        return validate_snowflake_identifier(value.upper())
    except ValueError:
        raise ValueError(
            f"Snowflake {label} must be an unquoted identifier, got: {value}"
        ) from None


def _write_artifacts(
    output_path: Path, frames: dict[str, Any], diagnostics: dict[str, dict[str, Any]]
) -> None:
    """Publish the snapshot only after every table has been written.

    A partially written directory would be read by inspection as a complete but
    truncated graph, so the whole set is staged first and moved into place. The
    destination is checked with the same ownership rule as the local writer:
    an export path that names a directory FlakeGraph did not write is refused
    rather than deleted.
    """

    import shutil  # noqa: PLC0415
    import uuid  # noqa: PLC0415

    output_path.parent.mkdir(parents=True, exist_ok=True)
    assert_replaceable_output(output_path)
    temp_path = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    temp_path.mkdir(parents=False, exist_ok=False)
    try:
        for name, frame in frames.items():
            frame.to_parquet(temp_path / f"{name}.parquet", index=False)
        for name, payload in diagnostics.items():
            (temp_path / f"{name}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        write_ownership_manifest(temp_path)
        if output_path.exists():
            shutil.rmtree(output_path)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
