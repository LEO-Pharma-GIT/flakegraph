# SPDX-License-Identifier: Apache-2.0
"""Offline coverage of distributed and Snowflake publication contracts."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from snowflake_fakes import CONFIG, FakeConnection, sample_batch
from test_distributed_execution import MemoryDistributedStore, _settings

from kg_processor.adapters.distributed.local_blob import LocalBlobStore
from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.adapters.writers import snowflake_manifest
from kg_processor.adapters.writers.snowflake_bulk import write_bulk_load_files
from kg_processor.adapters.writers.snowflake_direct import (
    TABLE_COLUMNS,
    build_merge_statement,
    build_snowflake_rows,
)
from kg_processor.application.distributed_planner import (
    DistributedRunPlanner,
    _graph_output_payload,
)
from kg_processor.application.distributed_worker import DistributedWorker
from kg_processor.application.progress import NullProgressSink
from kg_processor.application.spark_finalization import (
    SparkFinalizationRequest,
    SparkGraphFinalizer,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import (
    ArtifactKind,
    ArtifactRef,
    PublicationLease,
    RunStatus,
    StoredArtifact,
    TaskLease,
    TaskStage,
)
from kg_processor.domain.extraction import ExtractionObservations
from kg_processor.domain.finalization import (
    DatasetFile,
    DatasetTableManifest,
    GraphDatasetManifest,
)
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.domain.stages import ExtractedDocumentShard, PreparedDocumentShard
from kg_processor.factories import build_distributed_pipeline


@pytest.mark.parametrize("changed", ["graph", "owner", "source", "ontology", "output"])
def test_explicit_run_identity_rejects_changed_submission(tmp_path: Path, changed: str) -> None:
    """Worker compatibility must not substitute for immutable submission identity."""

    inputs = tmp_path / "input"
    inputs.mkdir()
    source = inputs / "source.txt"
    source.write_text("Alpha is public text.")
    settings = _settings(tmp_path)
    store = MemoryDistributedStore()
    DistributedRunPlanner(settings, LocalFileSource(inputs), store, store).submit("stable")
    if changed == "graph":
        settings = settings.model_copy(
            update={"job": settings.job.model_copy(update={"graph_id": "b"})}
        )
    elif changed == "owner":
        settings = settings.model_copy(
            update={"job": settings.job.model_copy(update={"owner": "OTHER"})}
        )
    elif changed == "source":
        source.write_text("Changed immutable source content.")
    elif changed == "ontology":
        settings = settings.model_copy(
            update={
                "ontology": settings.ontology.model_copy(
                    update={
                        "profile": {
                            "name": "changed",
                            "description": "An explicitly changed vocabulary.",
                            "mode": "open",
                            "entity_types": [],
                        }
                    }
                )
            }
        )
    else:
        settings = settings.model_copy(
            update={"writer": settings.writer.model_copy(update={"provider": "snowflake_bulk"})}
        )
    with pytest.raises(ValueError, match="different inputs"):
        DistributedRunPlanner(settings, LocalFileSource(inputs), store, store).submit("stable")
    assert store.initial_task_streams == 1
    assert store.cancelled_run_ids == []


@pytest.mark.parametrize("inline_ontology", [False, True])
def test_worker_materializes_requested_graph_with_real_pipeline(
    tmp_path: Path, inline_ontology: bool
) -> None:
    """Preparation and local finalization share the claimed run's authoritative graph."""

    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "source.txt").write_text("Alpha Corp partners with Beta Ltd.")
    base = _settings(tmp_path)
    submitted = base.model_copy(
        update={"job": base.job.model_copy(update={"graph_id": "requested"})}
    )
    if inline_ontology:
        submitted = submitted.model_copy(
            update={
                "ontology": submitted.ontology.model_copy(
                    update={
                        "profile": {
                            "name": "inline",
                            "description": "Synthetic profile",
                            "mode": "open",
                            "entity_types": [
                                {"name": "ORGANIZATION", "description": "An organization."},
                                {"name": "CONCEPT", "description": "A concept."},
                            ],
                        }
                    }
                )
            }
        )
    store = MemoryDistributedStore()
    DistributedRunPlanner(submitted, LocalFileSource(inputs), store, store).submit("requested-run")
    pipeline = build_distributed_pipeline(
        base,
        {TaskStage.PREPARE_DOCUMENT, TaskStage.FINALIZE_GRAPH},
        progress_sink=NullProgressSink(),
    )
    worker = DistributedWorker(
        base,
        "worker",
        {TaskStage.PREPARE_DOCUMENT, TaskStage.FINALIZE_GRAPH},
        pipeline,
        store,
        store,
        pipeline_for=pipeline.for_settings,
    )

    def execute(stage: TaskStage) -> Any:
        task = next(task for task in store.tasks if task.stage is stage)
        store.lease = TaskLease(
            task=task,
            worker_id="worker",
            attempt=1,
            claim_token="claim-1",
            lease_expires_at=datetime.now(UTC) + timedelta(minutes=1),
        )
        result = worker.process_one()
        assert result.succeeded, result.error
        return result

    try:
        result = execute(TaskStage.PREPARE_DOCUMENT)
        prepared = PreparedDocumentShard.model_validate_json(
            store.get(result.output_artifact_ids[0]).payload
        )
        assert {row["graph_id"] for row in prepared.document_rows} == {"requested"}
        assert {chunk.graph_id for chunk in prepared.chunks} == {"requested"}
        run_pipeline = worker._run_context("requested-run").pipeline
        contextual = run_pipeline.extract_document_context(prepared)
        entities = run_pipeline.extract_window_entities(contextual)
        relations = run_pipeline.extract_window_relations(contextual, entities.entities)
        shard = ExtractedDocumentShard(
            prepared=prepared,
            observations=ExtractionObservations(
                entities=entities.entities,
                relations=relations.relations,
                chunk_count=len(prepared.chunks),
                window_count=1,
            ),
        )
        store.put(
            "requested-run",
            ArtifactKind.EXTRACTED_DOCUMENT,
            shard.model_dump_json().encode(),
            "application/json",
            metadata={"file_ids": prepared.file_ids},
        )
        finalized = execute(TaskStage.FINALIZE_GRAPH)
        batch = GraphWriteBatch.model_validate_json(
            store.get(finalized.output_artifact_ids[0]).payload
        )
        assert batch.graph_id == "requested"
        assert {row["graph_id"] for row in batch.documents} == {"requested"}
        assert batch.nodes
        assert {node.graph_id for node in batch.nodes} == {"requested"}
        # The compute lease persists an immutable result; delivery owns the writer.
        assert not (base.writer.output_path / "nodes.parquet").exists()
    finally:
        pipeline.close()


def test_snowflake_artifact_keys_and_chunk_references_are_graph_scoped(tmp_path: Path) -> None:
    """Publishing and replacing another graph cannot steal a source artifact's row."""

    batch = sample_batch()
    first = build_snowflake_rows(batch)
    second = build_snowflake_rows(batch.model_copy(update={"graph_id": "other"}))
    persisted: dict[str, dict[str, dict[str, object]]] = {"KG_BLOCK": {}, "KG_ASSET": {}}
    for rows in (first, second):
        for table, records in persisted.items():
            records.update({str(row["ID"]): row for row in rows[table]})
    for table, records in persisted.items():
        persisted[table] = {key: row for key, row in records.items() if row["GRAPH_ID"] != "other"}
        assert list(persisted[table].values()) == first[table]
    assert first["KG_CHUNK"][0]["BLOCK_IDS"] == [first["KG_BLOCK"][0]["ID"]]
    assert first["KG_CHUNK"][0]["ASSET_IDS"] == [first["KG_ASSET"][0]["ID"]]
    assert second["KG_CHUNK"][0]["BLOCK_IDS"] == [second["KG_BLOCK"][0]["ID"]]
    files = write_bulk_load_files(second, tmp_path, "@STAGE", "prefix", "LOAD")
    for table in ("KG_BLOCK", "KG_ASSET"):
        staged = pd.read_parquet(
            next(file.local_path for file in files if file.table_name == table)
        )
        assert staged.loc[0, "ID"] == second[table][0]["ID"]


def test_snowflake_document_metadata_survives_direct_and_bulk_conversion(tmp_path: Path) -> None:
    """Every declared enriched document field reaches SQL and staged Parquet."""

    batch = sample_batch()
    metadata = {
        "kind": "dataset",
        "summary": "Table summary",
        "related_file_ids": ["related"],
        "folder": "datasets",
        "provenance": "revalidated",
    }
    batch = batch.model_copy(update={"documents": [{**batch.documents[0], **metadata}]})
    rows = build_snowflake_rows(batch)
    row = rows["KG_DOCUMENT"][0]
    for name, value in metadata.items():
        assert row[name.upper()] == value
    sql, params = build_merge_statement("KG_DOCUMENT", row, TABLE_COLUMNS["KG_DOCUMENT"], 2)
    assert "PARSE_JSON(?)::ARRAY AS RELATED_FILE_IDS" in sql
    assert json.dumps(metadata["related_file_ids"], separators=(",", ":")) in params
    files = write_bulk_load_files(rows, tmp_path, "@STAGE", "prefix", "LOAD")
    staged = pd.read_parquet(
        next(file.local_path for file in files if file.table_name == "KG_DOCUMENT")
    )
    for name, value in metadata.items():
        assert staged.loc[0, name.upper()] == (
            json.dumps(value) if isinstance(value, list) else value
        )


@pytest.mark.parametrize("explicit_credentials", [True, False])
def test_spark_standard_s3_uses_explicit_keys_or_default_chain(
    monkeypatch: pytest.MonkeyPatch, explicit_credentials: bool
) -> None:
    """The actual session builder configures standard AWS S3 without endpoint overrides."""

    class Builder:
        def __init__(self) -> None:
            self.options: dict[str, object] = {}

        def appName(self, name: str) -> Builder:
            return self

        def master(self, master: str) -> Builder:
            return self

        def config(self, key: str, value: object) -> Builder:
            self.options[key] = value
            return self

        def getOrCreate(self) -> Builder:
            return self

    builder = Builder()
    module = ModuleType("pyspark.sql")
    module.SparkSession = SimpleNamespace(builder=builder, getActiveSession=lambda: None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyspark.sql", module)
    distributed: dict[str, object] = {"artifact_uri": "s3://synthetic-bucket/artifacts"}
    if explicit_credentials:
        distributed.update(
            artifact_access_key_id="synthetic-access", artifact_secret_access_key="synthetic-secret"
        )
    settings = Settings.load(env={}, overrides={"distributed": distributed})
    SparkGraphFinalizer(settings)._spark_session(
        SparkFinalizationRequest(
            run_id="run", graph_id="graph", attempt=1, artifact_ids=frozenset({"artifact"})
        )
    )
    assert "spark.hadoop.fs.s3a.endpoint" not in builder.options
    assert "spark.hadoop.fs.s3a.path.style.access" not in builder.options
    if explicit_credentials:
        assert builder.options["spark.hadoop.fs.s3a.access.key"] == "synthetic-access"
        assert builder.options["spark.hadoop.fs.s3a.secret.key"] == "synthetic-secret"
    else:
        assert "spark.hadoop.fs.s3a.access.key" not in builder.options
        assert "spark.hadoop.fs.s3a.secret.key" not in builder.options


def test_cancelled_real_local_finalizer_never_invokes_destination_writer(tmp_path: Path) -> None:
    """Cancellation during computation leaves the existing destination untouched."""

    settings = _settings(tmp_path)
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "source.txt").write_text("Alpha Corp partners with Beta Ltd.")

    class CancellableStore(MemoryDistributedStore):
        def complete_task(self, *args: Any, **kwargs: Any) -> None:
            assert self.run is not None
            if self.run.status is RunStatus.CANCELLED:
                raise RuntimeError("lease revoked")
            super().complete_task(*args, **kwargs)

    store = CancellableStore()
    DistributedRunPlanner(settings, LocalFileSource(inputs), store, store).submit("cancelled")
    pipeline = build_distributed_pipeline(
        settings,
        {TaskStage.PREPARE_DOCUMENT, TaskStage.FINALIZE_GRAPH},
        progress_sink=NullProgressSink(),
    )
    prepared = pipeline.prepare_documents(LocalFileSource(inputs).list_files())
    contextual = pipeline.extract_document_context(prepared)
    entities = pipeline.extract_window_entities(contextual)
    relations = pipeline.extract_window_relations(contextual, entities.entities)
    shard = ExtractedDocumentShard(
        prepared=prepared,
        observations=ExtractionObservations(
            entities=entities.entities,
            relations=relations.relations,
            chunk_count=len(prepared.chunks),
            window_count=1,
        ),
    )
    store.put(
        "cancelled",
        ArtifactKind.EXTRACTED_DOCUMENT,
        shard.model_dump_json().encode(),
        "application/json",
    )
    final = next(task for task in store.tasks if task.stage is TaskStage.FINALIZE_GRAPH)
    store.lease = TaskLease(
        task=final,
        worker_id="worker",
        attempt=1,
        claim_token="claim-1",
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )
    base_embedding = pipeline.embeddings

    class CancellingEmbedding:
        def embed(self, texts: list[str], options: Any) -> Any:
            assert store.run is not None
            store.run = store.run.model_copy(update={"status": RunStatus.CANCELLED})
            return base_embedding.embed(texts, options)

    class RecordingWriter:
        calls = 0

        def write(self, batch: GraphWriteBatch) -> None:
            self.calls += 1

        def stage_publication(
            self, batch: GraphWriteBatch, *, publication_id: str, generation: int
        ) -> AbstractContextManager[Callable[[], None]]:
            raise AssertionError("A cancelled compute task cannot stage publication")

    writer = RecordingWriter()
    pipeline.embeddings = CancellingEmbedding()
    pipeline.writer = writer
    try:
        result = DistributedWorker(
            settings, "worker", {TaskStage.FINALIZE_GRAPH}, pipeline, store, store
        ).process_one()
        assert not result.succeeded
        assert writer.calls == 0
        assert not store.completed
    finally:
        pipeline.close()


def test_manifest_writer_retains_enriched_documents_and_alias_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streaming manifest adapter stages the same complete rows as direct/bulk output."""

    batch = sample_batch()
    batch.documents[0].update(
        kind="dataset",
        summary="Summary",
        related_file_ids=["related"],
        folder="datasets",
        provenance="revalidated",
    )
    batch.entity_sources[0].aliases = ["Alice", "A. Example"]
    blob = LocalBlobStore(tmp_path.as_uri())
    tables = {}
    for name in ["documents", "blocks", "assets", "chunks", "entity_sources"]:
        values = getattr(batch, name)
        rows = [
            value.model_dump(mode="json") if hasattr(value, "model_dump") else value
            for value in values
        ]
        path = tmp_path / f"{name}.parquet"
        cast(Any, pq.write_table)(pa.Table.from_pylist(rows), path)
        uri = blob.put(f"staged/{path.name}", path.read_bytes(), "application/octet-stream")
        tables[name] = DatasetTableManifest(
            name=name,
            row_count=len(rows),
            files=[DatasetFile(uri=uri, size_bytes=path.stat().st_size)],
        )
    captured: dict[str, list[dict[str, object]]] = {}
    original = write_bulk_load_files

    def record(rows: Any, *args: Any, **kwargs: Any) -> Any:
        for table, values in rows.items():
            captured.setdefault(table, []).extend(values)
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(snowflake_manifest, "write_bulk_load_files", record)
    connection = FakeConnection()
    snowflake_manifest.SnowflakeManifestWriter(
        CONFIG, 2, "@STAGE", blob, connector_factory=lambda **_: connection
    ).write(
        GraphDatasetManifest(
            run_id="run",
            graph_id="graph",
            engine="spark",
            tables={
                **{
                    name: DatasetTableManifest(name=name, row_count=0)
                    for name in snowflake_manifest._MANIFEST_TABLES
                },
                **tables,
            },
        )
    )
    assert captured["KG_DOCUMENT"][0]["SUMMARY"] == "Summary"
    assert captured["KG_DOCUMENT"][0]["RELATED_FILE_IDS"] == ["related"]
    assert captured["KG_DOCUMENT"][0]["KIND"] == "dataset"
    assert captured["KG_DOCUMENT"][0]["FOLDER"] == "datasets"
    assert captured["KG_DOCUMENT"][0]["PROVENANCE"] == "revalidated"
    assert captured["KG_CHUNK"][0]["BLOCK_IDS"] == [captured["KG_BLOCK"][0]["ID"]]
    assert captured["KG_CHUNK"][0]["ASSET_IDS"] == [captured["KG_ASSET"][0]["ID"]]
    assert captured["KG_ENTITY_SOURCE"][0]["ALIASES"] == ["Alice", "A. Example"]
    assert connection.committed


def test_spark_worker_publishes_local_engine_batch_without_a_pipeline_writer(
    tmp_path: Path,
) -> None:
    """A Spark worker delivers a local-engine batch using only destination dependencies."""

    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "distributed": settings.distributed.model_copy(
                update={"finalization_engine": "spark", "artifact_uri": tmp_path.as_uri()}
            )
        }
    )
    batch = sample_batch()
    payload = batch.model_dump_json().encode()
    artifact = StoredArtifact(
        ref=ArtifactRef(
            id="batch",
            run_id="run",
            kind=ArtifactKind.GRAPH_RESULT,
            media_type="application/json",
            checksum="synthetic",
            size_bytes=len(payload),
            metadata={"schema": GraphWriteBatch.__name__},
        ),
        payload=payload,
    )
    lease = PublicationLease(
        id="publication",
        run_id="run",
        artifact_id="batch",
        task_payload={"output": {"provider": "local_artifacts"}},
        worker_id="spark-worker",
        attempt=1,
        generation=1,
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )

    class Store:
        committed = False

        def claim_publication(self, *args: Any) -> PublicationLease:
            return lease

        def heartbeat_publication(self, *args: Any) -> None:
            pass

        def get(self, artifact_id: str) -> StoredArtifact:
            assert artifact_id == artifact.ref.id
            return artifact

        def commit_publication(self, current: PublicationLease, commit: Callable[[], None]) -> None:
            assert current == lease
            commit()
            self.committed = True

        def fail_publication(self, *args: Any) -> None:
            raise AssertionError(args)

    store = Store()
    pipeline = build_distributed_pipeline(
        settings, {TaskStage.FINALIZE_GRAPH}, progress_sink=NullProgressSink()
    )
    publisher = snowflake_manifest.TaskConfiguredManifestPublisher(
        settings, LocalBlobStore(tmp_path.as_uri())
    )
    try:
        worker = DistributedWorker(
            settings,
            "spark-worker",
            {TaskStage.FINALIZE_GRAPH},
            pipeline,
            cast(Any, store),
            cast(Any, store),
            manifest_publisher=publisher,
        )
        result = worker.process_one()
        assert result.succeeded, result.error
        assert store.committed
        assert (
            pd.read_parquet(settings.writer.output_path / "nodes.parquet").loc[0, "id"]
            == batch.nodes[0].id
        )
    finally:
        pipeline.close()


def test_batch_publisher_uses_saved_target_and_worker_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local-engine batches preserve the destination contract even on another worker profile."""

    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "writer": settings.writer.model_copy(update={"provider": "snowflake_bulk"}),
            "snowflake": settings.snowflake.model_copy(
                update={
                    "account": "worker-account",
                    "database": "OLD_DB",
                    "schema_name": "OLD_SCHEMA",
                    "bulk_stage": "@OLD_STAGE",
                }
            ),
        }
    )
    connection = FakeConnection(rows=[("publication",)])
    kwargs: dict[str, Any] = {}

    def connector(**values: Any) -> FakeConnection:
        kwargs.update(values)
        return connection

    monkeypatch.setattr(snowflake_manifest, "load_snowflake_connector", lambda: connector)
    publisher = snowflake_manifest.TaskConfiguredManifestPublisher(
        settings, LocalBlobStore(tmp_path.as_uri())
    )
    target = settings.model_copy(
        update={
            "snowflake": settings.snowflake.model_copy(
                update={
                    "account": "untrusted-task-account",
                    "database": "TARGET_DB",
                    "schema_name": "TARGET_SCHEMA",
                    "bulk_stage": "@TARGET_DB.TARGET_SCHEMA.LOAD_STAGE",
                }
            )
        }
    )
    payload = {
        **_graph_output_payload(target),
        "publication": {"id": "publication", "generation": 1},
    }
    with publisher.stage_batch_publication(sample_batch(), payload) as commit:
        assert not connection.committed
        commit()
    assert kwargs["account"] == "worker-account"
    assert kwargs["database"] == "TARGET_DB"
    assert kwargs["schema"] == "TARGET_SCHEMA"
    assert connection.committed
