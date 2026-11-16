from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from snowflake_fakes import CONFIG, FakeConnection, FakeCursor, sample_batch

from kg_processor.adapters.writers.snowflake_direct import (
    ColumnSpec,
    SnowflakeDirectWriter,
    build_edge_reconciliation_statements,
    build_merge_statement,
    build_node_reconciliation_statements,
    build_reindex_delete_statements,
    build_snowflake_rows,
)
from kg_processor.application.snowflake_schema import (
    render_snowflake_schema_sql,
    split_sql_statements,
)
from kg_processor.config.preflight import run_preflight
from kg_processor.config.settings import Settings
from kg_processor.domain.ids import stable_id


def test_snowflake_schema_uses_configured_vector_dimension() -> None:
    sql = render_snowflake_schema_sql(1536)

    assert "VECTOR(FLOAT, 1536)" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_JOB" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_JOB_FILE" in sql
    assert "WORKER_ID STRING" in sql
    assert "LEASE_UNTIL TIMESTAMP_NTZ" in sql
    assert "ROWS_WRITTEN NUMBER" in sql
    assert "ROW_COUNTS VARIANT" in sql
    assert "AUDIT VARIANT" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_NODE" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_BLOCK" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_ASSET" in sql
    assert "BBOX ARRAY" in sql
    assert "METADATA VARIANT" in sql
    assert "DOCUMENT_ID STRING NOT NULL" in sql
    assert "SECTION_PATH ARRAY" in sql
    assert "BLOCK_IDS ARRAY" in sql
    assert "ASSET_IDS ARRAY" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_GRAPH_METRICS" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_EXTRACTION_TRACE" in sql
    assert "RATING_EXPLANATION STRING" in sql
    assert "SUGGESTED_QUESTIONS ARRAY" in sql
    assert "RUN_ID STRING NOT NULL" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_OCR_CACHE" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_EXTRACTION_CACHE" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_BULK_LOAD" in sql
    assert len(split_sql_statements(sql)) >= 19


def test_snowflake_schema_rejects_invalid_vector_dimension() -> None:
    with pytest.raises(ValueError, match="VECTOR dimension"):
        render_snowflake_schema_sql(5000)


def test_merge_statement_casts_arrays_variants_and_vectors() -> None:
    sql, params = build_merge_statement(
        "KG_NODE",
        {
            "ID": "node_1",
            "GRAPH_ID": "graph",
            "TYPES": ["PERSON"],
            "EMBEDDING": [0.1, 0.2],
            "SOURCE_CHUNK_IDS": ["chunk_1"],
        },
        (
            ColumnSpec("ID"),
            ColumnSpec("GRAPH_ID"),
            ColumnSpec("TYPES", "array"),
            ColumnSpec("EMBEDDING", "vector"),
            ColumnSpec("SOURCE_CHUNK_IDS", "array"),
        ),
        2,
    )

    assert "PARSE_JSON(?)::VECTOR(FLOAT, 2) AS EMBEDDING" in sql
    assert "PARSE_JSON(?)::ARRAY AS TYPES" in sql
    assert params == ["node_1", "graph", '["PERSON"]', "[0.1,0.2]", '["chunk_1"]']


def test_build_snowflake_rows_maps_graph_batch() -> None:
    batch = sample_batch()

    rows = build_snowflake_rows(batch)

    assert rows["KG_DOCUMENT"][0]["ID"] == stable_id("document", "graph", "file_1")
    other_graph_rows = build_snowflake_rows(batch.model_copy(update={"graph_id": "other_graph"}))
    assert other_graph_rows["KG_DOCUMENT"][0]["ID"] != rows["KG_DOCUMENT"][0]["ID"]
    assert rows["KG_PAGE"][0]["GRAPH_ID"] == "graph"
    assert rows["KG_BLOCK"][0]["ID"] == stable_id("block", "graph", "block_1")
    assert rows["KG_BLOCK"][0]["BBOX"] == [0.0, 1.0, 2.0, 3.0]
    assert rows["KG_BLOCK"][0]["METADATA"] == {"layout": "body"}
    assert rows["KG_ASSET"][0]["ID"] == stable_id("asset", "graph", "asset_1")
    assert rows["KG_ASSET"][0]["GRAPH_ID"] == "graph"
    assert rows["KG_ASSET"][0]["METADATA"] == {"layout": "figure"}
    assert rows["KG_CHUNK"][0]["DOCUMENT_ID"] == rows["KG_DOCUMENT"][0]["ID"]
    assert rows["KG_CHUNK"][0]["SECTION_PATH"] == ["Intro"]
    assert rows["KG_CHUNK"][0]["BLOCK_IDS"] == [stable_id("block", "graph", "block_1")]
    assert rows["KG_CHUNK"][0]["ASSET_IDS"] == [stable_id("asset", "graph", "asset_1")]
    assert rows["KG_CHUNK"][0]["OCR_GENERATION_ID"] == "ocr-run-1"
    assert rows["KG_CHUNK"][0]["EMBEDDING"] == [0.1, 0.2]
    assert rows["KG_NODE"][0]["SOURCE_CHUNK_IDS"] == ["chunk_1"]
    assert rows["KG_COMMUNITY"][0]["RATING_EXPLANATION"] == "Important Alice cluster."
    assert rows["KG_COMMUNITY"][0]["SUGGESTED_QUESTIONS"] == ["Who is Alice linked to?"]
    assert rows["KG_COMMUNITY_FINDING"][0]["GRAPH_ID"] == "graph"
    report = rows["KG_RUN_REPORT"][0]["REPORT"]
    assert isinstance(report, dict)
    assert report["job_id"] == "job"
    assert rows["KG_RUN_REPORT"][0]["RUN_ID"] == "run_1"
    assert rows["KG_GRAPH_METRICS"][0]["RUN_ID"] == "run_1"
    metrics = rows["KG_GRAPH_METRICS"][0]["METRICS"]
    assert isinstance(metrics, dict)
    counts = metrics["counts"]
    assert isinstance(counts, dict)
    assert counts["nodes"] == 1
    assert rows["KG_EXTRACTION_TRACE"][0]["STAGE"] == "ocr"
    assert rows["KG_EXTRACTION_TRACE"][0]["RUN_ID"] == "run_1"


def test_snowflake_row_mapping_redacts_sensitive_variant_artifacts() -> None:
    batch = sample_batch().model_copy(
        update={
            "run_report": {
                "job_id": "job",
                "graph_id": "graph",
                "run_id": "run_1",
                "api_key": "report-secret",
            },
            "graph_metrics": {
                "counts": {"nodes": 1},
                "connection_string": "AccountKey=metric-secret",
            },
            "extraction_trace": [
                {
                    "stage": "llm",
                    "provider_metadata": {
                        "provider": "azure_openai",
                        "oauth_token": "trace-token",
                    },
                }
            ],
        }
    )

    rows = build_snowflake_rows(batch)
    report = cast(dict[str, Any], rows["KG_RUN_REPORT"][0]["REPORT"])
    metrics = cast(dict[str, Any], rows["KG_GRAPH_METRICS"][0]["METRICS"])
    trace_payload = cast(dict[str, Any], rows["KG_EXTRACTION_TRACE"][0]["PAYLOAD"])
    provider_metadata = cast(dict[str, Any], trace_payload["provider_metadata"])

    assert report["api_key"] == "***"
    assert metrics["connection_string"] == "***"
    assert provider_metadata["oauth_token"] == "***"
    assert provider_metadata["provider"] == "azure_openai"


def test_reindex_delete_statements_remove_graph_snapshot_and_current_job_rows() -> None:
    batch = sample_batch()

    statements = build_reindex_delete_statements(batch)

    assert statements[:3] == [
        ("DELETE FROM KG_EXTRACTION_TRACE WHERE GRAPH_ID = ? AND JOB_ID = ?", ["graph", "job"]),
        ("DELETE FROM KG_RUN_REPORT WHERE GRAPH_ID = ? AND JOB_ID = ?", ["graph", "job"]),
        ("DELETE FROM KG_GRAPH_METRICS WHERE GRAPH_ID = ? AND JOB_ID = ?", ["graph", "job"]),
    ]
    assert ("DELETE FROM KG_NODE WHERE GRAPH_ID = ?", ["graph"]) in statements
    assert ("DELETE FROM KG_BLOCK WHERE GRAPH_ID = ?", ["graph"]) in statements
    assert ("DELETE FROM KG_ASSET WHERE GRAPH_ID = ?", ["graph"]) in statements
    assert ("DELETE FROM KG_DOCUMENT WHERE GRAPH_ID = ?", ["graph"]) in statements


def test_reindex_delete_statements_for_file_batch_are_file_scoped() -> None:
    """Ensure incremental deletion cannot remove artifacts from unrelated source files.

    Run diagnostics remain scoped by run identity.
    """

    batch = sample_batch().model_copy(
        update={
            "write_scope": "file_batch",
            "reindex_file_ids": ["file_1"],
        }
    )

    statements = build_reindex_delete_statements(batch)

    assert (
        "DELETE FROM KG_EXTRACTION_TRACE WHERE GRAPH_ID = ? AND JOB_ID = ? AND RUN_ID = ?",
        ["graph", "job", "run_1"],
    ) in statements
    assert (
        "DELETE FROM KG_DOCUMENT WHERE GRAPH_ID = ? AND FILE_ID IN (?)",
        ["graph", "file_1"],
    ) in statements
    assert (
        "DELETE FROM KG_BLOCK WHERE GRAPH_ID = ? AND FILE_ID IN (?)",
        ["graph", "file_1"],
    ) in statements
    assert (
        "DELETE FROM KG_ASSET WHERE GRAPH_ID = ? AND FILE_ID IN (?)",
        ["graph", "file_1"],
    ) in statements
    assert (
        "DELETE FROM KG_EDGE_OBSERVATION WHERE GRAPH_ID = ? AND FILE_ID IN (?)",
        ["graph", "file_1"],
    ) in statements
    assert not any(sql == "DELETE FROM KG_NODE WHERE GRAPH_ID = ?" for sql, _ in statements)
    assert ("DELETE FROM KG_COMMUNITY WHERE GRAPH_ID = ?", ["graph"]) in statements
    assert (
        "DELETE FROM KG_COMMUNITY_FINDING WHERE GRAPH_ID = ?",
        ["graph"],
    ) in statements


def test_snowflake_direct_writer_executes_schema_and_merges() -> None:
    batch = sample_batch()
    connection = FakeConnection()
    writer = SnowflakeDirectWriter(
        CONFIG,
        embedding_dimension=2,
        connector_factory=lambda **_: connection,
    )

    writer.write(batch)

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert connection.committed
    assert connection.autocommit_calls == [False, True]
    assert connection.closed
    assert any(sql.startswith("CREATE TABLE IF NOT EXISTS KG_DOCUMENT") for sql in executed_sql)
    assert any(sql.startswith("MERGE INTO KG_NODE") for sql in executed_sql)
    delete_index = next(
        index for index, sql in enumerate(executed_sql) if sql.startswith("DELETE FROM KG_NODE")
    )
    merge_index = next(
        index for index, sql in enumerate(executed_sql) if sql.startswith("MERGE INTO KG_NODE")
    )
    assert delete_index < merge_index


def test_file_batch_writer_does_not_clobber_shared_node_identity_fields() -> None:
    batch = sample_batch().model_copy(
        update={"write_scope": "file_batch", "reindex_file_ids": ["file_1"]}
    )
    connection = FakeConnection()
    SnowflakeDirectWriter(
        CONFIG,
        embedding_dimension=2,
        connector_factory=lambda **_: connection,
    ).write(batch)

    node_merge = next(
        sql
        for sql, _params in connection.cursor_instance.executed
        if sql.startswith("MERGE INTO KG_NODE")
    )
    update_clause = node_merge.split("WHEN MATCHED THEN UPDATE SET ", 1)[1].split(
        "WHEN NOT MATCHED", 1
    )[0]
    for field in (
        "NAME",
        "PRIMARY_TYPE",
        "TYPES",
        "ALIASES",
        "DESCRIPTION",
        "SOURCE_CHUNK_IDS",
    ):
        assert f"{field} = source.{field}" not in update_clause


def test_snowflake_writer_preflight_requires_target(tmp_path: Path) -> None:
    settings = Settings.load(
        overrides={
            "files": {"input_path": tmp_path},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "writer": {"provider": "snowflake_direct"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("snowflake_direct writer requires" in error for error in result.errors)


def test_edge_reconciliation_rebuilds_canonical_rows_from_observations() -> None:
    """Ensure Snowflake edges are rebuilt from remaining per-file assertions.

    Aggregate files, weights, and evidence counts must be recomputed.
    """

    statements = build_edge_reconciliation_statements(sample_batch())

    assert len(statements) == 2
    aggregate_sql, aggregate_params = statements[0]
    assert "FROM KG_EDGE_OBSERVATION" in aggregate_sql
    assert "ARRAY_AGG(DISTINCT FILE_ID)" in aggregate_sql
    assert "COUNT(*) AS EVIDENCE_COUNT" in aggregate_sql
    # Claimed only when every remaining observation is, as finalization decides.
    assert "IFF(BOOLAND_AGG(ASSERTION = 'claimed'), 'claimed', 'reported')" in aggregate_sql
    assert "ASSERTION = source.ASSERTION" in aggregate_sql
    assert aggregate_params == [10.0, "graph"]
    assert "NOT EXISTS" in statements[1][0]


def test_file_batch_node_reconciliation_uses_durable_per_file_support() -> None:
    """Rebuild shared node fields and remove unsupported nodes after a file reindex."""

    batch = sample_batch().model_copy(update={"write_scope": "file_batch"})

    statements = build_node_reconciliation_statements(batch)

    assert len(statements) == 6
    assert "FROM KG_ENTITY_SOURCE source JOIN KG_EVIDENCE evidence" in statements[0][0]
    assert "SET DEGREE = 0, RANK = 0" in statements[3][0]
    assert "COUNT(DISTINCT EDGE_ID)" in statements[4][0]
    assert "NOT EXISTS" in statements[5][0]
    assert "SET ALIASES = ARRAY_CONSTRUCT()" in statements[1][0]
    assert "LATERAL FLATTEN(INPUT => source.ALIASES)" in statements[2][0]
    assert "ARRAY_EXCEPT(source.ALIASES, ARRAY_CONSTRUCT(target.NAME))" in statements[2][0]
    assert all("graph" in params for _sql, params in statements)


def test_file_batch_node_merge_preserves_fields_rebuilt_after_observation_merge() -> None:
    """Do not clobber shared node aggregates with one batch's partial values."""

    sql, _params = build_merge_statement(
        "KG_NODE",
        {"ID": "node", "GRAPH_ID": "graph", "DESCRIPTION": "partial", "DEGREE": 1},
        (
            ColumnSpec("ID"),
            ColumnSpec("GRAPH_ID"),
            ColumnSpec("DESCRIPTION"),
            ColumnSpec("DEGREE"),
        ),
        2,
        preserve_on_match={"DESCRIPTION", "DEGREE"},
    )

    update_clause = sql.split("WHEN MATCHED THEN UPDATE SET ", 1)[1].split("WHEN NOT MATCHED", 1)[0]
    assert "DESCRIPTION =" not in update_clause
    assert "DEGREE =" not in update_clause


def test_direct_writer_releases_the_session_when_no_cursor_can_be_opened() -> None:
    """A connection that outlives a failed write holds a Snowflake session open."""

    class _CursorlessConnection(FakeConnection):
        def cursor(self) -> FakeCursor:
            """Fail the way a connector reports an unusable session."""

            raise RuntimeError("session is no longer usable")

    connection = _CursorlessConnection()
    writer = SnowflakeDirectWriter(
        CONFIG,
        embedding_dimension=2,
        connector_factory=lambda **_: connection,
    )

    with pytest.raises(RuntimeError, match="session is no longer usable"):
        writer.write(sample_batch())

    assert connection.closed
