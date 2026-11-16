# SPDX-License-Identifier: Apache-2.0
"""Direct Snowflake writer using parameterized row-level MERGE statements.

This writer favors clarity and smoke-testability. The bulk writer reuses the
same row mapping and table column specs for production-sized loads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Set
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, cast

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnectionConfig,
    load_snowflake_connector,
    snowflake_transaction,
)
from kg_processor.adapters.snowflake import (
    compact_json as _json,
)
from kg_processor.adapters.snowflake import (
    validate_snowflake_identifier as _validate_identifier,
)
from kg_processor.application.redaction import redact_sensitive_data
from kg_processor.application.snowflake_schema import (
    render_snowflake_schema_sql,
    split_sql_statements,
)
from kg_processor.domain.graph import Chunk, GraphWriteBatch
from kg_processor.domain.ids import stable_id

__all__ = [
    "ColumnSpec",
    "SnowflakeConnectionConfig",
    "SnowflakeDirectWriter",
    "build_merge_statement",
    "preserved_columns",
    "build_node_reconciliation_statements",
    "build_reindex_delete_statements",
    "build_snowflake_rows",
]


@dataclass(frozen=True)
class ColumnSpec:
    """Snowflake column name plus the casting behavior required for its values."""

    name: str
    kind: str = "scalar"


TABLE_COLUMNS: dict[str, tuple[ColumnSpec, ...]] = {
    # This table map is the source of truth for both direct MERGE statements and
    # bulk staged loads. Adding a persisted field here gives both writer paths
    # the same Snowflake casting behavior.
    "KG_DOCUMENT": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("CHECKSUM"),
        ColumnSpec("SOURCE_URI"),
        ColumnSpec("MIME_TYPE"),
        ColumnSpec("SIZE_BYTES"),
        ColumnSpec("OCR_PROVIDER"),
        ColumnSpec("KIND"),
        ColumnSpec("SUMMARY"),
        ColumnSpec("RELATED_FILE_IDS", "array"),
        ColumnSpec("FOLDER"),
        ColumnSpec("PROVENANCE"),
    ),
    "KG_PAGE": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("PAGE_NUMBER"),
        ColumnSpec("MARKDOWN"),
        ColumnSpec("RAW_TEXT"),
        ColumnSpec("DETECTED_LANGUAGE"),
    ),
    "KG_BLOCK": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("PAGE_NUMBER"),
        ColumnSpec("KIND"),
        ColumnSpec("TEXT"),
        ColumnSpec("BBOX", "array"),
        ColumnSpec("METADATA", "variant"),
    ),
    "KG_ASSET": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("KIND"),
        ColumnSpec("PAGE_NUMBER"),
        ColumnSpec("URI"),
        ColumnSpec("METADATA", "variant"),
    ),
    "KG_CHUNK": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("DOCUMENT_ID"),
        ColumnSpec("PAGE_NUMBER"),
        ColumnSpec("CHUNK_INDEX"),
        ColumnSpec("CONTENT"),
        ColumnSpec("START_OFFSET"),
        ColumnSpec("END_OFFSET"),
        ColumnSpec("TOKEN_COUNT"),
        ColumnSpec("CONTENT_HASH"),
        ColumnSpec("SECTION_PATH", "array"),
        ColumnSpec("BLOCK_IDS", "array"),
        ColumnSpec("ASSET_IDS", "array"),
        ColumnSpec("OCR_GENERATION_ID"),
        ColumnSpec("EMBEDDING", "vector"),
    ),
    "KG_NODE": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("NORMALIZED_NAME"),
        ColumnSpec("NAME"),
        ColumnSpec("PRIMARY_TYPE"),
        ColumnSpec("TYPES", "array"),
        ColumnSpec("ALIASES", "array"),
        ColumnSpec("DESCRIPTION"),
        ColumnSpec("EMBEDDING", "vector"),
        ColumnSpec("SOURCE_CHUNK_IDS", "array"),
        ColumnSpec("DEGREE"),
        ColumnSpec("RANK"),
    ),
    "KG_EDGE": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("SOURCE_NODE_ID"),
        ColumnSpec("TARGET_NODE_ID"),
        ColumnSpec("RELATION_TYPE"),
        ColumnSpec("DESCRIPTION"),
        ColumnSpec("WEIGHT"),
        ColumnSpec("CONFIDENCE"),
        ColumnSpec("SOURCE_FILE_ID"),
        ColumnSpec("SOURCE_FILE_IDS", "array"),
        ColumnSpec("SOURCE_CHUNK_IDS", "array"),
        ColumnSpec("EVIDENCE_COUNT"),
        ColumnSpec("ASSERTION"),
        ColumnSpec("EMBEDDING", "vector"),
    ),
    "KG_EDGE_OBSERVATION": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("EDGE_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("CHUNK_ID"),
        ColumnSpec("DESCRIPTION"),
        ColumnSpec("WEIGHT"),
        ColumnSpec("CONFIDENCE"),
        ColumnSpec("EVIDENCE_ID"),
        ColumnSpec("QUANTITIES", "variant"),
        ColumnSpec("PROPOSED_RELATION_TYPE"),
        ColumnSpec("ASSERTION"),
    ),
    "KG_EVIDENCE": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("SUBJECT_ID"),
        ColumnSpec("SUBJECT_KIND"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("CHUNK_ID"),
        ColumnSpec("PAGE_NUMBER"),
        ColumnSpec("START_OFFSET"),
        ColumnSpec("END_OFFSET"),
        ColumnSpec("QUOTE"),
        ColumnSpec("METHOD"),
    ),
    "KG_ENTITY_SOURCE": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("NODE_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("PER_FILE_DESCRIPTION"),
        ColumnSpec("MENTION_COUNT"),
        ColumnSpec("ALIASES", "array"),
    ),
    "KG_COMMUNITY": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("STABLE_KEY"),
        ColumnSpec("LEVEL"),
        ColumnSpec("TITLE"),
        ColumnSpec("SUMMARY"),
        ColumnSpec("RATING"),
        ColumnSpec("RATING_EXPLANATION"),
        ColumnSpec("MEMBER_NODE_IDS", "array"),
        ColumnSpec("PARENT_COMMUNITY_ID"),
        ColumnSpec("CHILD_COMMUNITY_IDS", "array"),
        ColumnSpec("SUGGESTED_QUESTIONS", "array"),
        ColumnSpec("EMBEDDING", "vector"),
    ),
    "KG_COMMUNITY_FINDING": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("COMMUNITY_ID"),
        ColumnSpec("SUMMARY"),
        ColumnSpec("EXPLANATION"),
    ),
    "KG_DISCARDED_WINDOW": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("DOCUMENT_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("STAGE"),
        ColumnSpec("WINDOW_ID"),
        ColumnSpec("CHUNK_IDS", "array"),
        ColumnSpec("PAGE_START"),
        ColumnSpec("PAGE_END"),
        ColumnSpec("EXTRACTED_RECORDS"),
        ColumnSpec("RECORD_ACTIONS", "variant"),
        ColumnSpec("PREVIEW"),
    ),
    "KG_REJECTED_RECORD": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("DOCUMENT_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("WINDOW_ID"),
        ColumnSpec("CHUNK_ID"),
        ColumnSpec("PAGE_NUMBER"),
        ColumnSpec("KIND"),
        ColumnSpec("REASON"),
        ColumnSpec("NAME"),
        ColumnSpec("TYPE"),
        ColumnSpec("SOURCE"),
        ColumnSpec("SOURCE_TYPE"),
        ColumnSpec("TARGET"),
        ColumnSpec("TARGET_TYPE"),
        ColumnSpec("QUOTE"),
    ),
    "KG_FAILED_DOCUMENT": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("DOCUMENT_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("SOURCE_URI"),
        ColumnSpec("MIME_TYPE"),
        ColumnSpec("SIZE_BYTES"),
        ColumnSpec("PROVIDER"),
        ColumnSpec("ERROR_TYPE"),
        ColumnSpec("ERROR_MESSAGE"),
    ),
    "KG_RUN_REPORT": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("JOB_ID"),
        ColumnSpec("RUN_ID"),
        ColumnSpec("REPORT", "variant"),
    ),
    "KG_GRAPH_METRICS": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("JOB_ID"),
        ColumnSpec("RUN_ID"),
        ColumnSpec("METRICS", "variant"),
    ),
    "KG_EXTRACTION_TRACE": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("JOB_ID"),
        ColumnSpec("RUN_ID"),
        ColumnSpec("STEP_INDEX"),
        ColumnSpec("STAGE"),
        ColumnSpec("PAYLOAD", "variant"),
    ),
}

_GRAPH_SNAPSHOT_DELETE_ORDER = (
    "KG_EXTRACTION_TRACE",
    "KG_RUN_REPORT",
    "KG_GRAPH_METRICS",
    "KG_COMMUNITY_FINDING",
    "KG_COMMUNITY",
    "KG_DISCARDED_WINDOW",
    "KG_REJECTED_RECORD",
    "KG_FAILED_DOCUMENT",
    "KG_ENTITY_SOURCE",
    "KG_EVIDENCE",
    "KG_EDGE_OBSERVATION",
    "KG_EDGE",
    "KG_NODE",
    "KG_CHUNK",
    "KG_BLOCK",
    "KG_ASSET",
    "KG_PAGE",
    "KG_DOCUMENT",
)

_JOB_SCOPED_TABLES = {
    "KG_EXTRACTION_TRACE",
    "KG_RUN_REPORT",
    "KG_GRAPH_METRICS",
}

_FILE_BATCH_DELETE_ORDER = (
    ("KG_DISCARDED_WINDOW", "FILE_ID"),
    ("KG_REJECTED_RECORD", "FILE_ID"),
    ("KG_FAILED_DOCUMENT", "FILE_ID"),
    ("KG_ENTITY_SOURCE", "FILE_ID"),
    ("KG_EVIDENCE", "FILE_ID"),
    ("KG_EDGE_OBSERVATION", "FILE_ID"),
    ("KG_CHUNK", "FILE_ID"),
    ("KG_BLOCK", "FILE_ID"),
    ("KG_ASSET", "FILE_ID"),
    ("KG_PAGE", "FILE_ID"),
    ("KG_DOCUMENT", "FILE_ID"),
)


class SnowflakeDirectWriter:
    """Persist graph snapshots with parameterized row-level Snowflake statements.

    This implementation favors transparent SQL and small-run simplicity. It
    creates the required schema, removes stale snapshot rows, merges current
    artifacts, and reconciles edges within one transaction.
    """

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        embedding_dimension: int,
        connector_factory: ConnectorFactory | None = None,
    ) -> None:
        """Store connector dependencies and the vector dimension for casts."""

        self.config = config
        self.embedding_dimension = embedding_dimension
        self.connector_factory = connector_factory or load_snowflake_connector()

    def write(self, batch: GraphWriteBatch) -> None:
        """Replace one graph snapshot atomically through direct SQL operations.

        Autocommit is disabled when supported so schema preparation, stale-row
        deletion, row merges, and edge reconciliation either commit together or
        roll back as a unit. The connection is released by the outermost block so
        a cursor that cannot be opened does not strand an open session.
        """

        connection = self.connector_factory(**self.config.connect_kwargs())
        try:
            with snowflake_transaction(connection) as cursor:
                for statement in split_sql_statements(
                    render_snowflake_schema_sql(self.embedding_dimension)
                ):
                    cursor.execute(statement)
                for sql, params in build_reindex_delete_statements(batch):
                    cursor.execute(sql, params)
                for table_name, rows in build_snowflake_rows(batch).items():
                    for row in rows:
                        sql, params = build_merge_statement(
                            table_name,
                            row,
                            TABLE_COLUMNS[table_name],
                            self.embedding_dimension,
                            preserve_on_match=preserved_columns(batch, table_name),
                        )
                        cursor.execute(sql, params)
                for sql, params in build_edge_reconciliation_statements(batch):
                    cursor.execute(sql, params)
                for sql, params in build_node_reconciliation_statements(batch):
                    cursor.execute(sql, params)
        finally:
            connection.close()

    def stage_publication(
        self, batch: GraphWriteBatch, *, publication_id: str, generation: int
    ) -> AbstractContextManager[Callable[[], None]]:
        """Reject direct SQL publication, which distributed submission does not support."""

        raise ValueError("distributed publication requires local_artifacts or snowflake_bulk")


def build_snowflake_rows(batch: GraphWriteBatch) -> dict[str, list[dict[str, object]]]:
    """Map domain artifacts into the uppercase rows expected by generated DDL.

    Sensitive diagnostic values are redacted before serialization, and shared
    run identity is attached consistently to graph, trace, and metrics records.
    """

    job_id = str(batch.run_report.get("job_id", ""))
    run_id = _run_id(batch)
    run_report = cast(dict[str, Any], redact_sensitive_data(batch.run_report))
    graph_metrics = cast(dict[str, Any], redact_sensitive_data(batch.graph_metrics))
    extraction_trace = cast(list[dict[str, Any]], redact_sensitive_data(batch.extraction_trace))
    # Domain models use Pythonic lower-case fields. Snowflake rows are upper-case
    # to match generated DDL and avoid quoted identifier surprises.
    return {
        "KG_DOCUMENT": [_document_row(batch.graph_id, row) for row in batch.documents],
        "KG_PAGE": [_page_row(batch.graph_id, row) for row in batch.pages],
        "KG_BLOCK": [_block_row(batch.graph_id, row) for row in batch.blocks],
        "KG_ASSET": [_asset_row(batch.graph_id, row) for row in batch.assets],
        "KG_CHUNK": [_chunk_row(batch.graph_id, row) for row in batch.chunks],
        "KG_NODE": [_upper_model(row.model_dump(mode="json")) for row in batch.nodes],
        "KG_EDGE": [_upper_model(row.model_dump(mode="json")) for row in batch.edges],
        "KG_EDGE_OBSERVATION": [
            _upper_model(row.model_dump(mode="json")) for row in batch.edge_observations
        ],
        "KG_EVIDENCE": [_upper_model(row.model_dump(mode="json")) for row in batch.evidence],
        "KG_ENTITY_SOURCE": [
            _upper_model(row.model_dump(mode="json")) for row in batch.entity_sources
        ],
        # Communities are graph-wide derived artifacts. A leased file batch does
        # not have enough context to produce a valid replacement, so incremental
        # writes invalidate them and a later graph finalization recreates them.
        "KG_COMMUNITY": [
            _upper_model(row.model_dump(mode="json"))
            for row in batch.communities
            if batch.write_scope == "graph_snapshot"
        ],
        "KG_COMMUNITY_FINDING": [
            {**_upper_model(row.model_dump(mode="json")), "GRAPH_ID": batch.graph_id}
            for row in batch.community_findings
            if batch.write_scope == "graph_snapshot"
        ],
        # Gaps are per file, like evidence: a leased file batch replaces its own
        # files' rows and a snapshot replaces the graph's.
        "KG_DISCARDED_WINDOW": [
            {**_upper_model(row.model_dump(mode="json")), "GRAPH_ID": batch.graph_id}
            for row in batch.discarded_windows
        ],
        # Per file, like the gaps: what validation turned away in each file.
        "KG_REJECTED_RECORD": [
            {**_upper_model(row.model_dump(mode="json")), "GRAPH_ID": batch.graph_id}
            for row in batch.rejected_records
        ],
        # Per file too: a batch that retries a file replaces its failure row.
        "KG_FAILED_DOCUMENT": [
            {**_upper_model(row.model_dump(mode="json")), "GRAPH_ID": batch.graph_id}
            for row in batch.failed_documents
        ],
        "KG_RUN_REPORT": [
            {
                "ID": stable_id("run_report", batch.graph_id, job_id, run_id),
                "GRAPH_ID": batch.graph_id,
                "JOB_ID": job_id,
                "RUN_ID": run_id,
                "REPORT": run_report,
            }
        ],
        "KG_GRAPH_METRICS": [
            {
                "ID": stable_id("graph_metrics", batch.graph_id, job_id, run_id),
                "GRAPH_ID": batch.graph_id,
                "JOB_ID": job_id,
                "RUN_ID": run_id,
                "METRICS": graph_metrics,
            }
        ],
        "KG_EXTRACTION_TRACE": [
            {
                "ID": stable_id(
                    "trace",
                    batch.graph_id,
                    job_id,
                    run_id,
                    index,
                    row.get("stage", ""),
                ),
                "GRAPH_ID": batch.graph_id,
                "JOB_ID": job_id,
                "RUN_ID": run_id,
                "STEP_INDEX": index,
                "STAGE": str(row.get("stage", "")),
                "PAYLOAD": row,
            }
            for index, row in enumerate(extraction_trace)
        ],
    }


def build_edge_reconciliation_statements(
    batch: GraphWriteBatch,
) -> list[tuple[str, list[object]]]:
    """Rebuild canonical edge aggregates from durable observation rows.

    This post-MERGE step makes file-batch reindexing lossless: deleting one
    file's observations cannot erase support contributed by another file.
    """

    aggregate_sql = (
        "MERGE INTO KG_EDGE target USING ("
        "SELECT EDGE_ID, LEAST(SUM(WEIGHT), ?) AS WEIGHT, "
        "MAX(CONFIDENCE) AS CONFIDENCE, MIN(FILE_ID) AS SOURCE_FILE_ID, "
        "ARRAY_AGG(DISTINCT FILE_ID) WITHIN GROUP (ORDER BY FILE_ID) AS SOURCE_FILE_IDS, "
        "ARRAY_AGG(DISTINCT CHUNK_ID) WITHIN GROUP (ORDER BY CHUNK_ID) AS SOURCE_CHUNK_IDS, "
        "COUNT(*) AS EVIDENCE_COUNT, MAX_BY(DESCRIPTION, LENGTH(DESCRIPTION)) AS DESCRIPTION, "
        # Claimed only when every observation is, as finalization aggregates it.
        "IFF(BOOLAND_AGG(ASSERTION = 'claimed'), 'claimed', 'reported') AS ASSERTION "
        "FROM KG_EDGE_OBSERVATION WHERE GRAPH_ID = ? GROUP BY EDGE_ID"
        ") source ON target.ID = source.EDGE_ID "
        "WHEN MATCHED THEN UPDATE SET WEIGHT = source.WEIGHT, "
        "CONFIDENCE = source.CONFIDENCE, SOURCE_FILE_ID = source.SOURCE_FILE_ID, "
        "SOURCE_FILE_IDS = source.SOURCE_FILE_IDS, SOURCE_CHUNK_IDS = source.SOURCE_CHUNK_IDS, "
        "EVIDENCE_COUNT = source.EVIDENCE_COUNT, DESCRIPTION = source.DESCRIPTION, "
        "ASSERTION = source.ASSERTION, UPDATED_AT = CURRENT_TIMESTAMP()"
    )
    orphan_delete_sql = (
        "DELETE FROM KG_EDGE target WHERE GRAPH_ID = ? AND NOT EXISTS ("
        "SELECT 1 FROM KG_EDGE_OBSERVATION observation "
        "WHERE observation.EDGE_ID = target.ID)"
    )
    return [
        (aggregate_sql, [batch.relation_weight_max, batch.graph_id]),
        (orphan_delete_sql, [batch.graph_id]),
    ]


def build_node_reconciliation_statements(
    batch: GraphWriteBatch,
) -> list[tuple[str, list[object]]]:
    """Rebuild file-sensitive node aggregates after incremental reindexing.

    Durable ``KG_ENTITY_SOURCE`` and node evidence rows are the authoritative
    per-file support. Reconciliation prevents one file batch from overwriting a
    shared node, removes nodes whose final source disappeared, and recalculates
    degree/rank from the surviving canonical edges. Full snapshots already carry
    complete node aggregates and need no additional statements.
    """

    if batch.write_scope != "file_batch":
        return []
    support_sql = (
        "MERGE INTO KG_NODE target USING ("
        "SELECT source.NODE_ID, MAX_BY(source.PER_FILE_DESCRIPTION, "
        "LENGTH(source.PER_FILE_DESCRIPTION)) AS DESCRIPTION, "
        "ARRAY_AGG(DISTINCT evidence.CHUNK_ID) "
        "WITHIN GROUP (ORDER BY evidence.CHUNK_ID) AS SOURCE_CHUNK_IDS "
        "FROM KG_ENTITY_SOURCE source JOIN KG_EVIDENCE evidence "
        "ON evidence.GRAPH_ID = source.GRAPH_ID AND evidence.SUBJECT_ID = source.NODE_ID "
        "AND evidence.SUBJECT_KIND = 'node' AND evidence.FILE_ID = source.FILE_ID "
        "WHERE source.GRAPH_ID = ? GROUP BY source.NODE_ID"
        ") source ON target.ID = source.NODE_ID "
        "WHEN MATCHED THEN UPDATE SET DESCRIPTION = source.DESCRIPTION, "
        "SOURCE_CHUNK_IDS = source.SOURCE_CHUNK_IDS, UPDATED_AT = CURRENT_TIMESTAMP()"
    )
    aliases_sql = (
        "MERGE INTO KG_NODE target USING ("
        "SELECT source.NODE_ID, ARRAY_AGG(DISTINCT alias.VALUE::STRING) "
        "WITHIN GROUP (ORDER BY alias.VALUE::STRING) AS ALIASES "
        "FROM KG_ENTITY_SOURCE source, "
        "LATERAL FLATTEN(INPUT => source.ALIASES) alias "
        "WHERE source.GRAPH_ID = ? GROUP BY source.NODE_ID"
        ") source ON target.ID = source.NODE_ID "
        "WHEN MATCHED THEN UPDATE SET ALIASES = "
        "ARRAY_EXCEPT(source.ALIASES, ARRAY_CONSTRUCT(target.NAME)), "
        "UPDATED_AT = CURRENT_TIMESTAMP()"
    )
    reset_degrees_sql = (
        "UPDATE KG_NODE SET DEGREE = 0, RANK = 0, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE GRAPH_ID = ?"
    )
    degree_sql = (
        "MERGE INTO KG_NODE target USING ("
        "SELECT NODE_ID, COUNT(DISTINCT EDGE_ID) AS DEGREE FROM ("
        "SELECT ID AS EDGE_ID, SOURCE_NODE_ID AS NODE_ID FROM KG_EDGE WHERE GRAPH_ID = ? "
        "UNION ALL SELECT ID AS EDGE_ID, TARGET_NODE_ID AS NODE_ID FROM KG_EDGE WHERE GRAPH_ID = ?"
        ") GROUP BY NODE_ID"
        ") source ON target.ID = source.NODE_ID "
        "WHEN MATCHED THEN UPDATE SET DEGREE = source.DEGREE, RANK = source.DEGREE, "
        "UPDATED_AT = CURRENT_TIMESTAMP()"
    )
    orphan_delete_sql = (
        "DELETE FROM KG_NODE target WHERE GRAPH_ID = ? AND NOT EXISTS ("
        "SELECT 1 FROM KG_ENTITY_SOURCE source WHERE source.GRAPH_ID = target.GRAPH_ID "
        "AND source.NODE_ID = target.ID)"
    )
    return [
        (support_sql, [batch.graph_id]),
        ("UPDATE KG_NODE SET ALIASES = ARRAY_CONSTRUCT() WHERE GRAPH_ID = ?", [batch.graph_id]),
        (aliases_sql, [batch.graph_id]),
        (reset_degrees_sql, [batch.graph_id]),
        (degree_sql, [batch.graph_id, batch.graph_id]),
        (orphan_delete_sql, [batch.graph_id]),
    ]


def build_reindex_delete_statements(batch: GraphWriteBatch) -> list[tuple[str, list[object]]]:
    """Build delete statements for snapshot rewrites or per-file reindexing."""

    if batch.write_scope == "file_batch":
        return _file_batch_delete_statements(batch)
    job_id = str(batch.run_report.get("job_id", ""))
    statements: list[tuple[str, list[object]]] = []
    for table_name in _GRAPH_SNAPSHOT_DELETE_ORDER:
        table = _validate_identifier(table_name)
        if table_name in _JOB_SCOPED_TABLES:
            statements.append(
                (
                    f"DELETE FROM {table} WHERE GRAPH_ID = ? AND JOB_ID = ?",
                    [batch.graph_id, job_id],
                )
            )
        else:
            statements.append((f"DELETE FROM {table} WHERE GRAPH_ID = ?", [batch.graph_id]))
    return statements


def _file_batch_delete_statements(batch: GraphWriteBatch) -> list[tuple[str, list[object]]]:
    job_id = str(batch.run_report.get("job_id", ""))
    run_id = _run_id(batch)
    file_ids = [file_id for file_id in batch.reindex_file_ids if file_id.strip()]
    statements: list[tuple[str, list[object]]] = [
        (
            "DELETE FROM KG_EXTRACTION_TRACE WHERE GRAPH_ID = ? AND JOB_ID = ? AND RUN_ID = ?",
            [batch.graph_id, job_id, run_id],
        ),
        (
            "DELETE FROM KG_RUN_REPORT WHERE GRAPH_ID = ? AND JOB_ID = ? AND RUN_ID = ?",
            [batch.graph_id, job_id, run_id],
        ),
        (
            "DELETE FROM KG_GRAPH_METRICS WHERE GRAPH_ID = ? AND JOB_ID = ? AND RUN_ID = ?",
            [batch.graph_id, job_id, run_id],
        ),
        (
            "DELETE FROM KG_COMMUNITY_FINDING WHERE GRAPH_ID = ?",
            [batch.graph_id],
        ),
        ("DELETE FROM KG_COMMUNITY WHERE GRAPH_ID = ?", [batch.graph_id]),
    ]
    if not file_ids:
        return statements
    for table_name, file_column in _FILE_BATCH_DELETE_ORDER:
        table = _validate_identifier(table_name)
        column = _validate_identifier(file_column)
        placeholders = ", ".join("?" for _ in file_ids)
        statements.append(
            (
                f"DELETE FROM {table} WHERE GRAPH_ID = ? AND {column} IN ({placeholders})",
                [batch.graph_id, *file_ids],
            )
        )
    return statements


# Node aggregates a file batch cannot know: they are rebuilt once every
# per-file observation has been merged, so a merge must not overwrite them.
_FILE_BATCH_PRESERVED_NODE_COLUMNS = frozenset(
    {
        "ALIASES",
        "DESCRIPTION",
        "EMBEDDING",
        "NAME",
        "PRIMARY_TYPE",
        "SOURCE_CHUNK_IDS",
        "TYPES",
        "DEGREE",
        "RANK",
    }
)


def preserved_columns(batch: GraphWriteBatch, table_name: str) -> frozenset[str]:
    """Return the columns a merge of this batch into ``table_name`` must leave alone."""

    if batch.write_scope == "file_batch" and table_name == "KG_NODE":
        return _FILE_BATCH_PRESERVED_NODE_COLUMNS
    return frozenset()


def build_merge_statement(
    table_name: str,
    row: Mapping[str, object],
    columns: Sequence[ColumnSpec],
    embedding_dimension: int,
    preserve_on_match: Set[str] = frozenset(),
) -> tuple[str, list[object]]:
    """Build a parameterized MERGE statement for a single Snowflake row.

    ``preserve_on_match`` supports incremental aggregates whose authoritative
    value is rebuilt after all per-file observations have been merged.
    """

    table = _validate_identifier(table_name)
    select_parts: list[str] = []
    params: list[object] = []
    for column in columns:
        value = row.get(column.name)
        expression, expression_params = _select_expression(column, value, embedding_dimension)
        select_parts.append(f"{expression} AS {column.name}")
        params.extend(expression_params)

    source_sql = "SELECT " + ", ".join(select_parts)
    update_columns = [
        column.name
        for column in columns
        if column.name != "ID" and column.name not in preserve_on_match
    ]
    update_sql = ", ".join(f"{name} = source.{name}" for name in update_columns)
    column_names = ", ".join(column.name for column in columns)
    value_names = ", ".join(f"source.{column.name}" for column in columns)
    sql = (
        f"MERGE INTO {table} target USING ({source_sql}) source ON target.ID = source.ID "
        f"WHEN MATCHED THEN UPDATE SET {update_sql}, UPDATED_AT = CURRENT_TIMESTAMP() "
        f"WHEN NOT MATCHED THEN INSERT ({column_names}) VALUES ({value_names})"
    )
    return sql, params


def _select_expression(
    column: ColumnSpec,
    value: object,
    embedding_dimension: int,
) -> tuple[str, list[object]]:
    if value is None:
        return "NULL", []
    if column.kind == "vector":
        return f"PARSE_JSON(?)::VECTOR(FLOAT, {embedding_dimension})", [_json(value)]
    if column.kind == "array":
        return "PARSE_JSON(?)::ARRAY", [_json(value)]
    if column.kind == "variant":
        return "PARSE_JSON(?)", [_json(value)]
    return "?", [value]


def _document_row(graph_id: str, row: Mapping[str, object]) -> dict[str, object]:
    file_id = str(row["file_id"])
    return {
        "ID": stable_id("document", graph_id, file_id),
        "GRAPH_ID": graph_id,
        "FILE_ID": file_id,
        "CHECKSUM": str(row["checksum"]),
        "SOURCE_URI": str(row["source_uri"]),
        "MIME_TYPE": str(row["mime_type"]),
        "SIZE_BYTES": row.get("size_bytes"),
        "OCR_PROVIDER": row.get("ocr_provider"),
        "KIND": row.get("kind"),
        "SUMMARY": row.get("summary"),
        "RELATED_FILE_IDS": row.get("related_file_ids"),
        "FOLDER": row.get("folder"),
        "PROVENANCE": row.get("provenance"),
    }


def _chunk_row(graph_id: str, row: Chunk) -> dict[str, object]:
    """Map a chunk and point its document reference at the persisted document key."""

    converted = _upper_model(row.model_dump(mode="json"), graph_id)
    converted["DOCUMENT_ID"] = stable_id("document", graph_id, str(converted["FILE_ID"]))
    converted["BLOCK_IDS"] = [stable_id("block", graph_id, value) for value in row.block_ids]
    converted["ASSET_IDS"] = [stable_id("asset", graph_id, value) for value in row.asset_ids]
    return converted


def _page_row(graph_id: str, row: Mapping[str, object]) -> dict[str, object]:
    file_id = str(row["file_id"])
    raw_page_number = row["page_number"]
    if not isinstance(raw_page_number, int | str):
        raise ValueError("page_number must be an integer-compatible value")
    page_number = int(raw_page_number)
    return {
        "ID": stable_id("page", graph_id, file_id, page_number),
        "GRAPH_ID": graph_id,
        "FILE_ID": file_id,
        "PAGE_NUMBER": page_number,
        "MARKDOWN": row.get("markdown"),
        "RAW_TEXT": row.get("raw_text"),
        "DETECTED_LANGUAGE": row.get("detected_language"),
    }


def _asset_row(graph_id: str, row: Mapping[str, object]) -> dict[str, object]:
    return {
        "ID": stable_id("asset", graph_id, str(row["id"])),
        "GRAPH_ID": graph_id,
        "FILE_ID": str(row["file_id"]),
        "KIND": str(row["kind"]),
        "PAGE_NUMBER": row.get("page_number"),
        "URI": row.get("uri"),
        "METADATA": row.get("metadata") or {},
    }


def _block_row(graph_id: str, row: Mapping[str, object]) -> dict[str, object]:
    return {
        "ID": stable_id("block", graph_id, str(row["id"])),
        "GRAPH_ID": graph_id,
        "FILE_ID": str(row["file_id"]),
        "PAGE_NUMBER": row.get("page_number"),
        "KIND": str(row["kind"]),
        "TEXT": row.get("text"),
        "BBOX": row.get("bbox"),
        "METADATA": row.get("metadata") or {},
    }


def _upper_model(row: Mapping[str, object], graph_id: str | None = None) -> dict[str, object]:
    # Domain fields are snake_case already, so the column name is the upper case.
    converted = {key.upper(): value for key, value in row.items()}
    if graph_id is not None:
        converted["GRAPH_ID"] = graph_id
    return converted


def _run_id(batch: GraphWriteBatch) -> str:
    value = batch.run_report.get("run_id")
    if isinstance(value, str) and value.strip():
        return value
    return stable_id(
        "run",
        batch.graph_id,
        batch.run_report.get("job_id", ""),
        batch.write_scope,
        batch.reindex_file_ids,
    )
