from __future__ import annotations

import json
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from snowflake_fakes import CONFIG, FakeConnection, FakeCursor, sample_batch

from kg_processor.adapters.distributed.local_blob import LocalBlobStore
from kg_processor.adapters.writers.snowflake_bulk import (
    SnowflakeBulkWriter,
    build_bulk_merge_statement,
    build_copy_into_load_table_statement,
    build_create_load_table_statement,
    build_load_table_name,
    build_put_statement,
    write_bulk_load_files,
)
from kg_processor.adapters.writers.snowflake_direct import (
    TABLE_COLUMNS,
    build_snowflake_rows,
)
from kg_processor.adapters.writers.snowflake_manifest import (
    _MANIFEST_TABLES,
    SnowflakeManifestWriter,
    _empty_batch,
)
from kg_processor.domain.finalization import DatasetFile, DatasetTableManifest, GraphDatasetManifest
from kg_processor.domain.ids import stable_id


def _all_tables(**given: DatasetTableManifest) -> dict[str, DatasetTableManifest]:
    """Every table a manifest names, empty but for the ones a test fills."""

    empty = {name: DatasetTableManifest(name=name, row_count=0) for name in _MANIFEST_TABLES}
    return {**empty, **given}


def test_manifest_batch_preserves_configured_relation_weight_cap() -> None:
    manifest = GraphDatasetManifest(
        run_id="run-1",
        graph_id="graph-1",
        engine="spark",
        tables={},
        relation_weight_max=3.5,
    )

    assert _empty_batch(manifest).relation_weight_max == 3.5


def test_bulk_load_files_write_string_parquet_rows(tmp_path: Path) -> None:
    rows_by_table = build_snowflake_rows(sample_batch())

    load_files = write_bulk_load_files(
        rows_by_table,
        tmp_path,
        "@DB.SCHEMA.KG_LOAD_STAGE",
        "kg_processor/graph/job/LOAD123",
        "LOAD123",
    )

    node_file = next(item for item in load_files if item.table_name == "KG_NODE")
    frame = pd.read_parquet(node_file.local_path)
    assert node_file.load_table_name == "KG_LOAD_LOAD123_KG_NODE"
    assert node_file.stage_file_location == (
        "@DB.SCHEMA.KG_LOAD_STAGE/kg_processor/graph/job/LOAD123/KG_NODE.parquet"
    )
    assert frame.loc[0, "TYPES"] == '["PERSON"]'
    assert frame.loc[0, "EMBEDDING"] == "[0.1,0.2]"
    assert frame.loc[0, "DEGREE"] == "1"
    block_file = next(item for item in load_files if item.table_name == "KG_BLOCK")
    block_frame = pd.read_parquet(block_file.local_path)
    assert block_frame.loc[0, "BBOX"] == "[0.0,1.0,2.0,3.0]"
    assert block_frame.loc[0, "METADATA"] == '{"layout":"body"}'
    asset_file = next(item for item in load_files if item.table_name == "KG_ASSET")
    asset_frame = pd.read_parquet(asset_file.local_path)
    assert asset_frame.loc[0, "METADATA"] == '{"layout":"figure"}'
    chunk_file = next(item for item in load_files if item.table_name == "KG_CHUNK")
    chunk_frame = pd.read_parquet(chunk_file.local_path)
    assert chunk_frame.loc[0, "SECTION_PATH"] == '["Intro"]'
    assert chunk_frame.loc[0, "BLOCK_IDS"] == json.dumps([stable_id("block", "graph", "block_1")])
    assert chunk_frame.loc[0, "ASSET_IDS"] == json.dumps([stable_id("asset", "graph", "asset_1")])
    community_file = next(item for item in load_files if item.table_name == "KG_COMMUNITY")
    community_frame = pd.read_parquet(community_file.local_path)
    assert community_frame.loc[0, "RATING_EXPLANATION"] == "Important Alice cluster."
    assert community_frame.loc[0, "SUGGESTED_QUESTIONS"] == '["Who is Alice linked to?"]'


def test_bulk_load_files_split_large_tables_by_target_size(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {
            "ID": f"node_{index}",
            "GRAPH_ID": "graph",
            "NORMALIZED_NAME": f"node_{index}",
            "NAME": f"Node {index}",
            "PRIMARY_TYPE": "CONCEPT",
            "TYPES": ["CONCEPT"],
            "DESCRIPTION": "large staged payload " * 10,
        }
        for index in range(5)
    ]

    load_files = write_bulk_load_files(
        {"KG_NODE": rows},
        tmp_path,
        "@DB.SCHEMA.KG_LOAD_STAGE",
        "kg_processor/graph/job/LOAD123",
        "LOAD123",
        target_file_size_bytes=350,
    )

    assert len(load_files) > 1
    assert {load_file.load_table_name for load_file in load_files} == {"KG_LOAD_LOAD123_KG_NODE"}
    assert sum(load_file.row_count for load_file in load_files) == len(rows)
    assert load_files[0].local_path.name == "KG_NODE_00001.parquet"
    assert load_files[-1].stage_file_location.endswith(f"/{load_files[-1].local_path.name}")


def test_bulk_load_files_deduplicates_source_rows_by_id(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {
            "ID": "node_1",
            "GRAPH_ID": "graph",
            "NORMALIZED_NAME": "alice",
            "NAME": "Alice Smith",
            "PRIMARY_TYPE": "PERSON",
            "TYPES": ["PERSON"],
            "DESCRIPTION": "older description",
        },
        {
            "ID": "node_1",
            "GRAPH_ID": "graph",
            "NORMALIZED_NAME": "alice",
            "NAME": "Alice Smith",
            "PRIMARY_TYPE": "PERSON",
            "TYPES": ["PERSON"],
            "DESCRIPTION": "newer description",
        },
    ]

    load_files = write_bulk_load_files(
        {"KG_NODE": rows},
        tmp_path,
        "@DB.SCHEMA.KG_LOAD_STAGE",
        "kg_processor/graph/job/LOAD123",
        "LOAD123",
    )

    frame = pd.read_parquet(load_files[0].local_path)
    assert len(frame) == 1
    assert frame.loc[0, "DESCRIPTION"] == "newer description"


def test_bulk_sql_builders_cast_from_staging_columns(tmp_path: Path) -> None:
    put_path = tmp_path / "KG_NODE.parquet"
    put_path.write_text("placeholder", encoding="utf-8")

    assert build_load_table_name("KG_NODE", "load_1") == "KG_LOAD_LOAD_1_KG_NODE"
    assert build_create_load_table_statement(
        "KG_LOAD_LOAD_1_KG_NODE",
        TABLE_COLUMNS["KG_NODE"],
    ).startswith("CREATE TEMP TABLE KG_LOAD_LOAD_1_KG_NODE")
    assert build_put_statement(
        put_path,
        "@DB.SCHEMA.KG_LOAD_STAGE/prefix",
    ).startswith("PUT 'file://")
    assert build_copy_into_load_table_statement(
        "KG_LOAD_LOAD_1_KG_NODE",
        "@DB.SCHEMA.KG_LOAD_STAGE/prefix/KG_NODE.parquet",
    ) == (
        "COPY INTO KG_LOAD_LOAD_1_KG_NODE "
        "FROM @DB.SCHEMA.KG_LOAD_STAGE/prefix/KG_NODE.parquet "
        "FILE_FORMAT = (TYPE = PARQUET USE_LOGICAL_TYPE = TRUE) "
        "MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE"
    )
    merge_sql = build_bulk_merge_statement(
        "KG_NODE",
        "KG_LOAD_LOAD_1_KG_NODE",
        TABLE_COLUMNS["KG_NODE"],
        2,
    )
    assert "PARSE_JSON(EMBEDDING)::VECTOR(FLOAT, 2) AS EMBEDDING" in merge_sql
    assert "PARSE_JSON(TYPES)::ARRAY AS TYPES" in merge_sql
    assert "DEGREE::NUMBER AS DEGREE" in merge_sql
    block_merge_sql = build_bulk_merge_statement(
        "KG_BLOCK",
        "KG_LOAD_LOAD_1_KG_BLOCK",
        TABLE_COLUMNS["KG_BLOCK"],
        2,
    )
    assert "PAGE_NUMBER::NUMBER AS PAGE_NUMBER" in block_merge_sql
    assert "PARSE_JSON(BBOX)::ARRAY AS BBOX" in block_merge_sql
    assert "PARSE_JSON(METADATA) AS METADATA" in block_merge_sql
    asset_merge_sql = build_bulk_merge_statement(
        "KG_ASSET",
        "KG_LOAD_LOAD_1_KG_ASSET",
        TABLE_COLUMNS["KG_ASSET"],
        2,
    )
    assert "PAGE_NUMBER::NUMBER AS PAGE_NUMBER" in asset_merge_sql
    assert "PARSE_JSON(METADATA) AS METADATA" in asset_merge_sql
    chunk_merge_sql = build_bulk_merge_statement(
        "KG_CHUNK",
        "KG_LOAD_LOAD_1_KG_CHUNK",
        TABLE_COLUMNS["KG_CHUNK"],
        2,
    )
    assert "PARSE_JSON(SECTION_PATH)::ARRAY AS SECTION_PATH" in chunk_merge_sql
    assert "PARSE_JSON(BLOCK_IDS)::ARRAY AS BLOCK_IDS" in chunk_merge_sql
    assert "PARSE_JSON(ASSET_IDS)::ARRAY AS ASSET_IDS" in chunk_merge_sql
    community_merge_sql = build_bulk_merge_statement(
        "KG_COMMUNITY",
        "KG_LOAD_LOAD_1_KG_COMMUNITY",
        TABLE_COLUMNS["KG_COMMUNITY"],
        2,
    )
    assert "PARSE_JSON(SUGGESTED_QUESTIONS)::ARRAY AS SUGGESTED_QUESTIONS" in (community_merge_sql)


def test_snowflake_bulk_writer_executes_put_copy_and_merge(tmp_path: Path) -> None:
    batch = sample_batch()
    connection = FakeConnection()
    writer = SnowflakeBulkWriter(
        CONFIG,
        embedding_dimension=2,
        bulk_stage="@DB.SCHEMA.KG_LOAD_STAGE",
        connector_factory=lambda **_: connection,
        local_temp_dir=tmp_path,
        load_id="LOAD123",
    )

    writer.write(batch)

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert connection.committed
    assert connection.autocommit_calls == [False, True]
    assert connection.closed
    assert any(sql.startswith("CREATE TABLE IF NOT EXISTS KG_DOCUMENT") for sql in executed_sql)
    assert any(sql.startswith("CREATE TEMP TABLE KG_LOAD_LOAD123_KG_BLOCK") for sql in executed_sql)
    assert any(sql.startswith("CREATE TEMP TABLE KG_LOAD_LOAD123_KG_ASSET") for sql in executed_sql)
    assert any(sql.startswith("CREATE TEMP TABLE KG_LOAD_LOAD123_KG_NODE") for sql in executed_sql)
    assert any(sql.startswith("PUT 'file://") for sql in executed_sql)
    assert any(sql.startswith("COPY INTO KG_LOAD_LOAD123_KG_NODE") for sql in executed_sql)
    assert any(sql.startswith("MERGE INTO KG_NODE") for sql in executed_sql)
    assert sum(1 for sql in executed_sql if sql.startswith("MERGE INTO KG_NODE")) == 1
    delete_index = next(
        index for index, sql in enumerate(executed_sql) if sql.startswith("DELETE FROM KG_NODE")
    )
    load_table_index = next(
        index
        for index, sql in enumerate(executed_sql)
        if sql.startswith("CREATE TEMP TABLE KG_LOAD_LOAD123_KG_NODE")
    )
    merge_index = next(
        index for index, sql in enumerate(executed_sql) if sql.startswith("MERGE INTO KG_NODE")
    )
    assert load_table_index < delete_index
    assert delete_index < merge_index
    assert executed_sql[-1] == "REMOVE @DB.SCHEMA.KG_LOAD_STAGE/kg_processor/graph/job/LOAD123"


def test_snowflake_bulk_file_batch_preserves_shared_node_identity(tmp_path: Path) -> None:
    batch = sample_batch().model_copy(
        update={"write_scope": "file_batch", "reindex_file_ids": ["file_1"]}
    )
    connection = FakeConnection()

    SnowflakeBulkWriter(
        CONFIG,
        embedding_dimension=2,
        bulk_stage="@DB.SCHEMA.KG_LOAD_STAGE",
        connector_factory=lambda **_: connection,
        local_temp_dir=tmp_path,
        load_id="IDENTITY",
    ).write(batch)

    node_merge = next(
        sql
        for sql, _params in connection.cursor_instance.executed
        if sql.startswith("MERGE INTO KG_NODE")
    )
    update_clause = node_merge.split("WHEN MATCHED THEN UPDATE SET ", 1)[1].split(
        "WHEN NOT MATCHED", 1
    )[0]
    for field in ("NAME", "PRIMARY_TYPE", "TYPES", "ALIASES"):
        assert f"{field} = source.{field}" not in update_clause


def test_manifest_publication_fence_uses_qmark_bindings(tmp_path: Path) -> None:
    class _FenceCursor(FakeCursor):
        def fetchone(self) -> Sequence[object] | None:
            return ["publication-1"]

    writer = SnowflakeManifestWriter(
        CONFIG,
        embedding_dimension=2,
        bulk_stage="@DB.SCHEMA.KG_LOAD_STAGE",
        blob_store=LocalBlobStore(tmp_path.as_uri()),
        publication_id="publication-1",
        publication_generation=3,
    )
    cursor = _FenceCursor()

    assert writer._acquire_publication_fence(cursor, "graph-1")  # noqa: SLF001
    assert all("%s" not in sql for sql, _params in cursor.executed)
    assert cursor.executed[0][1] == ("graph-1", 3, "publication-1")
    assert cursor.executed[1][1] == ("graph-1", 3)


def test_manifest_writer_streams_partitions_before_one_transactional_merge(tmp_path: Path) -> None:
    """Exercise the Spark-manifest path without materializing unrelated graph tables."""

    buffer = BytesIO()
    write_table = cast(Any, pq.write_table)
    write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "document-1",
                    "graph_id": "graph-1",
                    "file_id": "file-1",
                    "checksum": "abc",
                    "source_uri": "file:///document.pdf",
                    "mime_type": "application/pdf",
                    "size_bytes": 123,
                    "ocr_provider": "builtin_text",
                }
            ]
        ),
        buffer,
    )
    store = LocalBlobStore(tmp_path.as_uri())
    uri = store.put("documents.parquet", buffer.getvalue(), "application/octet-stream")
    manifest = GraphDatasetManifest(
        run_id="run-1",
        graph_id="graph-1",
        engine="spark",
        tables=_all_tables(
            documents=DatasetTableManifest(
                name="documents",
                row_count=1,
                files=[DatasetFile(uri=uri, size_bytes=len(buffer.getvalue()))],
            )
        ),
        metrics={"node_count": 0},
    )
    connection = FakeConnection()

    SnowflakeManifestWriter(
        CONFIG,
        embedding_dimension=2,
        bulk_stage="@DB.SCHEMA.KG_LOAD_STAGE",
        blob_store=store,
        connector_factory=lambda **_: connection,
    ).write(manifest)

    statements = [sql for sql, _params in connection.cursor_instance.executed]
    assert connection.autocommit_calls == [False, True]
    assert connection.committed
    assert any(sql.startswith("COPY INTO") and "KG_DOCUMENT" in sql for sql in statements)
    assert sum(sql.startswith("MERGE INTO KG_DOCUMENT") for sql in statements) == 1
    assert sum(sql.startswith("MERGE INTO KG_RUN_REPORT") for sql in statements) == 1


def test_manifest_publication_rebuilds_edges_from_durable_observations(tmp_path: Path) -> None:
    """Canonical edge aggregates must be derived the same way in every writer.

    A manifest carries KG_EDGE_OBSERVATION rows as the authoritative per-file
    support, so publication reconciles KG_EDGE against them and drops edges whose
    last observation disappeared.
    """

    buffer = BytesIO()
    write_table = cast(Any, pq.write_table)
    write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "edge-observation-1",
                    "graph_id": "graph-1",
                    "edge_id": "edge-1",
                    "file_id": "file-1",
                    "chunk_id": "chunk-1",
                    "weight": 1.0,
                    "confidence": 0.9,
                    "description": "Alice knows Bob.",
                    "evidence_id": "evidence-1",
                }
            ]
        ),
        buffer,
    )
    store = LocalBlobStore(tmp_path.as_uri())
    uri = store.put("edge-observations.parquet", buffer.getvalue(), "application/octet-stream")
    manifest = GraphDatasetManifest(
        run_id="run-1",
        graph_id="graph-1",
        engine="spark",
        tables=_all_tables(
            edge_observations=DatasetTableManifest(
                name="edge_observations",
                row_count=1,
                files=[DatasetFile(uri=uri, size_bytes=len(buffer.getvalue()))],
            )
        ),
    )
    connection = FakeConnection()

    SnowflakeManifestWriter(
        CONFIG,
        embedding_dimension=2,
        bulk_stage="@DB.SCHEMA.KG_LOAD_STAGE",
        blob_store=store,
        connector_factory=lambda **_: connection,
    ).write(manifest)

    statements = [sql for sql, _params in connection.cursor_instance.executed]
    assert any("FROM KG_EDGE_OBSERVATION WHERE GRAPH_ID = ?" in sql for sql in statements)
    assert any(sql.startswith("DELETE FROM KG_EDGE target") for sql in statements)
    assert connection.committed
