# SPDX-License-Identifier: Apache-2.0
"""Bulk Snowflake writer using Parquet staging, COPY, and MERGE.

Rows are first serialized with the direct-writer mapping, then staged as strings
so Snowflake can cast arrays, variants, and vectors explicitly during MERGE.
"""

from __future__ import annotations

import re
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence, Set
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, cast

import pandas as pd

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnectionConfig,
    load_snowflake_connector,
    quote_sql_string,
    snowflake_transaction,
    stage_path,
    validate_stage_location,
)
from kg_processor.adapters.snowflake import (
    compact_json as _json,
)
from kg_processor.adapters.snowflake import (
    validate_snowflake_identifier as _validate_identifier,
)
from kg_processor.adapters.writers.snowflake_direct import (
    TABLE_COLUMNS,
    ColumnSpec,
    build_edge_reconciliation_statements,
    build_node_reconciliation_statements,
    build_reindex_delete_statements,
    build_snowflake_rows,
    preserved_columns,
)
from kg_processor.application.snowflake_schema import (
    render_snowflake_schema_sql,
    split_sql_statements,
)
from kg_processor.domain.graph import GraphWriteBatch

_UNSAFE_STAGE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.$=-]+")
_NUMBER_COLUMNS = {
    "SIZE_BYTES",
    "PAGE_NUMBER",
    "CHUNK_INDEX",
    "START_OFFSET",
    "END_OFFSET",
    "TOKEN_COUNT",
    "DEGREE",
    "MENTION_COUNT",
    "EVIDENCE_COUNT",
    "LEVEL",
    "STEP_INDEX",
}
_FLOAT_COLUMNS = {"RANK", "WEIGHT", "RATING", "CONFIDENCE"}
DEFAULT_TARGET_FILE_SIZE_BYTES = 128 * 1024 * 1024
_PUT_STATUS_INDEX = 6


@dataclass(frozen=True)
class BulkLoadFile:
    """Metadata for one local Parquet file staged into a Snowflake temp table."""

    table_name: str
    load_table_name: str
    local_path: Path
    stage_file_location: str
    row_count: int


class SnowflakeBulkWriter:
    """Persist production-sized graph snapshots through staged Parquet bulk loads.

    Rows are partitioned into bounded files, uploaded to a Snowflake stage, and
    merged transactionally through temporary load tables. This avoids the
    per-record network overhead of the direct writer for large corpora.
    """

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        embedding_dimension: int,
        bulk_stage: str,
        connector_factory: ConnectorFactory | None = None,
        local_temp_dir: Path | None = None,
        load_id: str | None = None,
        target_file_size_bytes: int = DEFAULT_TARGET_FILE_SIZE_BYTES,
    ) -> None:
        """Configure staging, temp-file, and connector dependencies for bulk loads."""

        self.config = config
        self.embedding_dimension = embedding_dimension
        self.bulk_stage = validate_stage_location(bulk_stage.rstrip("/"))
        self.connector_factory = connector_factory or load_snowflake_connector()
        self.local_temp_dir = local_temp_dir
        self.load_id = _safe_path_segment(load_id or uuid.uuid4().hex[:12]).upper()
        self.target_file_size_bytes = target_file_size_bytes

    def write(self, batch: GraphWriteBatch) -> None:
        """Stage graph rows as Parquet, COPY them, and MERGE into final tables."""

        with self.stage_publication(batch) as commit:
            commit()

    @contextmanager
    def stage_publication(
        self,
        batch: GraphWriteBatch,
        *,
        publication_id: str | None = None,
        generation: int | None = None,
    ) -> Iterator[Callable[[], None]]:
        """Stage the batch before yielding the final replacement transaction."""

        load_id = f"PUB{generation}" if generation is not None else self.load_id
        prefix = build_bulk_load_prefix(batch, load_id)
        connection = self.connector_factory(**self.config.connect_kwargs())
        cursor = connection.cursor()
        try:
            for statement in split_sql_statements(
                render_snowflake_schema_sql(self.embedding_dimension)
            ):
                cursor.execute(statement)
            with tempfile.TemporaryDirectory(
                prefix="kg-snowflake-bulk-",
                dir=str(self.local_temp_dir) if self.local_temp_dir else None,
            ) as tmp:
                load_files = write_bulk_load_files(
                    build_snowflake_rows(batch),
                    Path(tmp),
                    self.bulk_stage,
                    prefix,
                    load_id,
                    self.target_file_size_bytes,
                )
                load_files_by_table = _load_files_by_table(load_files)
                for table_name, table_load_files in load_files_by_table.items():
                    columns = TABLE_COLUMNS[table_name]
                    load_table_name = table_load_files[0].load_table_name
                    cursor.execute(
                        build_create_load_table_statement(
                            load_table_name,
                            columns,
                        )
                    )
                    for load_file in table_load_files:
                        cursor.execute(
                            build_put_statement(
                                load_file.local_path,
                                stage_path(self.bulk_stage, prefix),
                            )
                        )
                        validate_put_result(cursor)
                        cursor.execute(
                            build_copy_into_load_table_statement(
                                load_table_name,
                                load_file.stage_file_location,
                            )
                        )
                # Staging above is per statement; from here the replacement
                # of the graph is one transaction.
                if generation is not None:
                    if generation < 1 or not publication_id:
                        raise ValueError("publication requires an identity and positive generation")
                    ensure_publication_fence(cursor)

                def commit() -> None:
                    """Publish only after the caller validates the durable lease."""

                    with snowflake_transaction(connection) as transaction:
                        if (
                            generation is not None
                            and publication_id
                            and not acquire_publication_fence(
                                transaction, batch.graph_id, generation, publication_id
                            )
                        ):
                            raise RuntimeError("a newer publication owns the destination")
                        for sql, params in build_reindex_delete_statements(batch):
                            transaction.execute(sql, params)
                        for table_name, table_load_files in load_files_by_table.items():
                            columns = TABLE_COLUMNS[table_name]
                            load_table_name = table_load_files[0].load_table_name
                            transaction.execute(
                                build_bulk_merge_statement(
                                    table_name,
                                    load_table_name,
                                    columns,
                                    self.embedding_dimension,
                                    preserve_on_match=preserved_columns(batch, table_name),
                                )
                            )
                        for sql, params in build_edge_reconciliation_statements(batch):
                            transaction.execute(sql, params)
                        for sql, params in build_node_reconciliation_statements(batch):
                            transaction.execute(sql, params)

                yield commit
        finally:
            best_effort_remove_stage_prefix(cursor, connection, self.bulk_stage, prefix)
            cursor.close()
            connection.close()


def build_bulk_load_prefix(batch: GraphWriteBatch, load_id: str) -> str:
    """Build a run-scoped stage prefix from graph id, job id, and load id."""

    job_id = str(batch.run_report.get("job_id", "job"))
    return "/".join(
        [
            "kg_processor",
            _safe_path_segment(batch.graph_id),
            _safe_path_segment(job_id),
            _safe_path_segment(load_id),
        ]
    )


def write_bulk_load_files(
    rows_by_table: Mapping[str, list[dict[str, object]]],
    output_dir: Path,
    bulk_stage: str,
    prefix: str,
    load_id: str,
    target_file_size_bytes: int = DEFAULT_TARGET_FILE_SIZE_BYTES,
) -> list[BulkLoadFile]:
    """Write staged Parquet files and return their Snowflake load metadata."""

    if target_file_size_bytes <= 0:
        raise ValueError("target_file_size_bytes must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    load_files: list[BulkLoadFile] = []
    for table_name, rows in rows_by_table.items():
        if not rows:
            continue
        columns = TABLE_COLUMNS[table_name]
        deduped_rows = _dedupe_rows_by_id(rows)
        # Values are staged as strings so COPY can load a simple temporary table;
        # the generated MERGE performs the authoritative ARRAY/VARIANT/VECTOR
        # casts with the configured embedding dimension.
        partitions = iter(_partition_stage_rows(deduped_rows, columns, target_file_size_bytes))
        first = next(partitions, None)
        if first is None:
            continue
        second = next(partitions, None)
        staged_partitions = (
            [(1, first, second is None)]
            if second is None
            else [(1, first, False), (2, second, False)]
        )
        for index, staged_rows, only_partition in chain(
            staged_partitions,
            ((index, rows, False) for index, rows in enumerate(partitions, start=3)),
        ):
            local_path = output_dir / _bulk_file_name(table_name, index, only_partition)
            frame = pd.DataFrame(
                staged_rows,
                columns=[column.name for column in columns],
            )
            frame.to_parquet(local_path, index=False)
            load_files.append(
                BulkLoadFile(
                    table_name=table_name,
                    load_table_name=build_load_table_name(table_name, load_id),
                    local_path=local_path,
                    stage_file_location=stage_path(bulk_stage, f"{prefix}/{local_path.name}"),
                    row_count=len(staged_rows),
                )
            )
    return load_files


def build_load_table_name(table_name: str, load_id: str) -> str:
    """Build a valid temporary load-table name for one target table."""

    table = _validate_identifier(table_name)
    safe_load_id = re.sub(r"[^A-Z0-9_]", "_", load_id.upper())
    return _validate_identifier(f"KG_LOAD_{safe_load_id}_{table}")


def _dedupe_rows_by_id(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Collapse duplicate staged IDs before Snowflake MERGE sees the source rows."""

    rows_by_id: dict[str, dict[str, object]] = {}
    ordered_ids: list[str] = []
    passthrough: list[dict[str, object]] = []
    for row in rows:
        row_id = str(row.get("ID", "")).strip()
        if not row_id:
            passthrough.append(row)
            continue
        if row_id not in rows_by_id:
            ordered_ids.append(row_id)
        rows_by_id[row_id] = row
    return [rows_by_id[row_id] for row_id in ordered_ids] + passthrough


def build_create_load_table_statement(
    load_table_name: str,
    columns: Sequence[ColumnSpec],
) -> str:
    """Return SQL for a string-typed temporary bulk load table."""

    table = _validate_identifier(load_table_name)
    column_sql = ", ".join(f"{column.name} STRING" for column in columns)
    return f"CREATE TEMP TABLE {table} ({column_sql})"


def build_put_statement(local_path: Path, stage_location: str) -> str:
    """Return a Snowflake PUT statement for one generated Parquet file."""

    uri = f"file://{local_path.resolve().as_posix()}"
    return (
        f"PUT {quote_sql_string(uri)} {validate_stage_location(stage_location)} "
        "AUTO_COMPRESS = FALSE OVERWRITE = TRUE"
    )


def validate_put_result(cursor: object) -> None:
    """Reject partial, skipped, or failed PUT results before COPY can see stale bytes."""

    fetchall = getattr(cursor, "fetchall", None)
    if not callable(fetchall):
        raise RuntimeError("Snowflake PUT did not expose an upload result")
    rows = list(fetchall())
    if not rows:
        raise RuntimeError("Snowflake PUT returned no upload status")
    failures: list[str] = []
    for row in rows:
        status = _put_status(row)
        if status != "UPLOADED":
            failures.append(status or "UNKNOWN")
    if failures:
        raise RuntimeError(
            "Snowflake PUT did not overwrite every staged object: " + ", ".join(failures)
        )


def _put_status(row: object) -> str:
    """Extract Snowflake's PUT status from tuple, mapping, or Row-like results."""

    if hasattr(row, "as_dict"):
        row = row.as_dict()
    if isinstance(row, Mapping):
        normalized = {str(key).lower(): value for key, value in row.items()}
        return str(normalized.get("status") or "").upper()
    if isinstance(row, Sequence) and not isinstance(row, str | bytes):
        # Snowflake PUT result columns place STATUS at index six.
        return str(
            row[_PUT_STATUS_INDEX] if len(row) > _PUT_STATUS_INDEX else row[-1] if row else ""
        ).upper()
    return str(getattr(row, "status", "")).upper()


def ensure_publication_fence(cursor: Any) -> None:
    """Create the destination generation ledger before opening the data transaction."""

    cursor.execute(
        "CREATE TABLE IF NOT EXISTS KG_GRAPH_PUBLICATION_FENCE ("
        "GRAPH_ID STRING PRIMARY KEY, GENERATION NUMBER NOT NULL, "
        "PUBLICATION_ID STRING NOT NULL, UPDATED_AT TIMESTAMP_TZ DEFAULT CURRENT_TIMESTAMP())"
    )


def acquire_publication_fence(
    cursor: Any, graph_id: str, generation: int, publication_id: str
) -> bool:
    """Make newer durable generations permanently supersede older deliveries."""

    cursor.execute(
        "MERGE INTO KG_GRAPH_PUBLICATION_FENCE target USING ("
        "SELECT ? AS GRAPH_ID, ? AS GENERATION, ? AS PUBLICATION_ID"
        ") source ON target.GRAPH_ID = source.GRAPH_ID "
        "WHEN MATCHED AND target.GENERATION < source.GENERATION THEN UPDATE SET "
        "GENERATION = source.GENERATION, PUBLICATION_ID = source.PUBLICATION_ID, "
        "UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHEN NOT MATCHED THEN INSERT (GRAPH_ID, GENERATION, PUBLICATION_ID) "
        "VALUES (source.GRAPH_ID, source.GENERATION, source.PUBLICATION_ID)",
        (graph_id, generation, publication_id),
    )
    cursor.execute(
        "SELECT PUBLICATION_ID FROM KG_GRAPH_PUBLICATION_FENCE "
        "WHERE GRAPH_ID = ? AND GENERATION = ?",
        (graph_id, generation),
    )
    row = cursor.fetchone()
    return row is not None and str(row[0]) == publication_id


def best_effort_remove_stage_prefix(
    cursor: object,
    connection: object,
    bulk_stage: str,
    prefix: str,
) -> None:
    """Attempt stage cleanup on every exit without masking publication outcome."""

    try:
        cast(Any, cursor).execute(f"REMOVE {stage_path(bulk_stage, prefix)}")
        cast(Any, connection).commit()
    except Exception:
        with suppress(Exception):
            cast(Any, connection).rollback()


def build_copy_into_load_table_statement(load_table_name: str, stage_file_location: str) -> str:
    """Return SQL that loads one staged Parquet file into its temp table."""

    return (
        f"COPY INTO {_validate_identifier(load_table_name)} "
        f"FROM {validate_stage_location(stage_file_location)} "
        "FILE_FORMAT = (TYPE = PARQUET USE_LOGICAL_TYPE = TRUE) "
        "MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE"
    )


def build_bulk_merge_statement(
    table_name: str,
    load_table_name: str,
    columns: Sequence[ColumnSpec],
    embedding_dimension: int,
    preserve_on_match: Set[str] = frozenset(),
) -> str:
    """Return SQL that casts staged strings and merges rows into a target table."""

    table = _validate_identifier(table_name)
    load_table = _validate_identifier(load_table_name)
    # The source SELECT is where staged strings become typed Snowflake values.
    # Keeping casts here makes staging files easy to inspect and avoids relying
    # on Snowflake's implicit JSON/vector coercion.
    select_parts = [
        f"{_stage_select_expression(column, embedding_dimension)} AS {column.name}"
        for column in columns
    ]
    source_sql = f"SELECT {', '.join(select_parts)} FROM {load_table}"
    update_columns = [
        column.name
        for column in columns
        if column.name != "ID" and column.name not in preserve_on_match
    ]
    update_sql = ", ".join(f"{name} = source.{name}" for name in update_columns)
    column_names = ", ".join(column.name for column in columns)
    value_names = ", ".join(f"source.{column.name}" for column in columns)
    return (
        f"MERGE INTO {table} target USING ({source_sql}) source ON target.ID = source.ID "
        f"WHEN MATCHED THEN UPDATE SET {update_sql}, UPDATED_AT = CURRENT_TIMESTAMP() "
        f"WHEN NOT MATCHED THEN INSERT ({column_names}) VALUES ({value_names})"
    )


def _stage_row(row: Mapping[str, object], columns: Sequence[ColumnSpec]) -> dict[str, str | None]:
    staged: dict[str, str | None] = {}
    for column in columns:
        value = row.get(column.name)
        if value is None:
            staged[column.name] = None
        elif column.kind in {"array", "variant", "vector"}:
            staged[column.name] = _json(value)
        else:
            staged[column.name] = str(value)
    return staged


def _partition_stage_rows(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[ColumnSpec],
    target_file_size_bytes: int,
) -> Iterator[list[dict[str, str | None]]]:
    current: list[dict[str, str | None]] = []
    current_size = 0
    for row in rows:
        staged = _stage_row(row, columns)
        row_size = _estimated_stage_row_size(staged)
        if current and current_size + row_size > target_file_size_bytes:
            yield current
            current = []
            current_size = 0
        current.append(staged)
        current_size += row_size
    if current:
        yield current


def _estimated_stage_row_size(row: Mapping[str, str | None]) -> int:
    # The exact compressed Parquet size is only known after writing. This
    # estimate uses staged string payload bytes plus a small per-column overhead
    # to keep bulk files near the configured COPY-friendly target without a
    # second serialization pass.
    return sum(len(value.encode("utf-8")) if value is not None else 0 for value in row.values()) + (
        len(row) * 8
    )


def _bulk_file_name(table_name: str, part_index: int, only_partition: bool) -> str:
    if only_partition:
        return f"{table_name}.parquet"
    return f"{table_name}_{part_index:05d}.parquet"


def _load_files_by_table(load_files: Sequence[BulkLoadFile]) -> dict[str, list[BulkLoadFile]]:
    by_table: dict[str, list[BulkLoadFile]] = {}
    for load_file in load_files:
        by_table.setdefault(load_file.table_name, []).append(load_file)
    return by_table


def _stage_select_expression(column: ColumnSpec, embedding_dimension: int) -> str:
    name = column.name
    if column.kind == "vector":
        return f"PARSE_JSON({name})::VECTOR(FLOAT, {embedding_dimension})"
    if column.kind == "array":
        return f"PARSE_JSON({name})::ARRAY"
    if column.kind == "variant":
        return f"PARSE_JSON({name})"
    if name in _NUMBER_COLUMNS:
        return f"{name}::NUMBER"
    if name in _FLOAT_COLUMNS:
        return f"{name}::FLOAT"
    return f"{name}::STRING"


def _safe_path_segment(value: str) -> str:
    stripped = value.strip().strip("/")
    safe = _UNSAFE_STAGE_SEGMENT_RE.sub("_", stripped)
    return safe or "unknown"
