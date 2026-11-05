"""Behavior tests for distributed planning and stage execution boundaries."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from typing import Any

import pytest
import yaml
from documents import chunk

from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.application.content_kinds import annotate_documents
from kg_processor.application.discarded_windows import (
    discarded_window_metrics,
    discarded_windows_from_trace,
    is_discarded_window_event,
)
from kg_processor.application.distributed_planner import (
    DistributedRunPlanner,
    _source_staging_workers,
    distributed_processing_compatibility_config,
    distributed_processing_config_digest,
)
from kg_processor.application.distributed_worker import (
    DistributedWorker,
    _prepared_projection_for_extracted_artifact,
)
from kg_processor.application.extraction_windows import build_extraction_windows
from kg_processor.application.llm_extractors import LlmEntityExtractor, LlmRelationExtractor
from kg_processor.application.pipeline import _note_discarded_windows
from kg_processor.application.revalidation import (
    RevalidationRequest,
    begin_revalidation,
    revalidate_entity_window,
    revalidate_observations,
    revalidate_relation_window,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.consumption import ConsumptionEvent
from kg_processor.domain.distributed import (
    STAGE_PRIORITY,
    ArtifactKind,
    ArtifactRef,
    InheritedDocuments,
    PriorityBand,
    PublicationLease,
    RevisionRequest,
    RunDefinition,
    RunOverview,
    RunSnapshot,
    RunStatus,
    RunSummary,
    StoredArtifact,
    TaskCount,
    TaskDefinition,
    TaskLease,
    TaskProgress,
    TaskSnapshot,
    TaskStage,
    TaskStatus,
    document_provenance,
    document_selected,
    task_priority,
)
from kg_processor.domain.documents import InputFile
from kg_processor.domain.extraction import (
    ColumnRelation,
    EntityMention,
    EntityVerificationDecision,
    ExtractionObservations,
    ExtractionWindow,
    RelationObservation,
    VerificationDecision,
    VerificationOutcome,
)
from kg_processor.domain.graph import Chunk, GraphWriteBatch
from kg_processor.domain.ids import sha256_hex, stable_id
from kg_processor.domain.ontology import (
    EntityTypeDefinition,
    OntologyProfile,
    RelationTypeDefinition,
)
from kg_processor.domain.stages import (
    DocumentContextShard,
    DocumentEntityInventoryShard,
    DocumentRevalidationShard,
    EntityWindowShard,
    ExtractedDocumentShard,
    ExtractionWindowShard,
    PreparedDocumentShard,
    RelationWindowShard,
    RevalidatedRelationWindowShard,
    RevalidationWindow,
    RevalidationWindowShard,
    combine_extracted_shards,
)
from kg_processor.ports.llm import (
    LlmCapabilities,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)
from kg_processor.ports.task_store import TaskStoreUnavailableError


class MemoryDistributedStore:
    """Minimal deterministic port implementation used to test application behavior."""

    def __init__(self) -> None:
        self.run: RunDefinition | None = None
        self.tasks: list[TaskDefinition] = []
        self.artifacts: dict[str, StoredArtifact] = {}
        self.cancelled_run_ids: list[str] = []
        self.lease: TaskLease | None = None
        self.completed: list[tuple[str, str, list[str], list[TaskDefinition], str | None]] = []
        self.failed: list[tuple[str, str, dict[str, Any], timedelta]] = []
        self.progress_updates: list[tuple[str, str, TaskProgress]] = []
        self.served_configurations: dict[TaskStage, str] = {}
        self.initial_task_streams = 0
        # Finished runs a revision may build on, beside the one being planned.
        self.earlier_runs: dict[str, tuple[RunDefinition, list[TaskDefinition]]] = {}
        # What each earlier run's tasks handed to completion, by task id.
        self.task_outputs: dict[str, list[str]] = {}
        self.abandoned_sweeps = 0
        self.consumption_ledger: dict[tuple[str, str], ConsumptionEvent] = {}

    def record_consumption(self, run_id: str, event: ConsumptionEvent) -> None:
        """Keep events across fake task attempts with production idempotence."""

        self.consumption_ledger[(run_id, event.event_id)] = event

    def consumption_events(self, run_id: str) -> list[ConsumptionEvent]:
        """Return only the requested run's accumulated provider work."""

        return [
            event for (identity, _), event in self.consumption_ledger.items() if identity == run_id
        ]

    def initialize(self) -> None:
        """Satisfy the idempotent setup port for this already-initialized fake."""

    def list_runs(self, limit: int = 100) -> list[RunOverview]:
        """Return no history because these tests query the active run directly."""

        _ = limit
        return []

    def retry_run(self, run_id: str, skipped_file_ids: list[str] | None = None) -> None:
        """Record no state because retry transitions are covered by PostgreSQL tests."""

        assert self.run is not None and self.run.id == run_id
        assert skipped_file_ids is None

    def create_run(self, run: RunDefinition) -> None:
        self.run = run

    def add_tasks(self, run_id: str, tasks: list[TaskDefinition]) -> None:
        assert self.run is not None and self.run.id == run_id
        self.tasks = tasks

    def add_initial_tasks(self, run_id: str, tasks: Iterable[TaskDefinition]) -> None:
        """Consume the production planner's constant-memory initial-plan stream."""

        assert self.run is not None and self.run.id == run_id
        self.initial_task_streams += 1
        self.tasks = list(tasks)

    def activate_run(self, run_id: str) -> None:
        assert self.run is not None and self.run.id == run_id
        self.run = self.run.model_copy(update={"status": RunStatus.QUEUED})

    def cancel_run(self, run_id: str) -> None:
        self.cancelled_run_ids.append(run_id)

    def get_run(self, run_id: str) -> RunSnapshot:
        if self.run is None or self.run.id != run_id:
            raise KeyError(run_id)
        now = datetime.now(UTC)
        return RunSnapshot(
            run=self.run,
            tasks=[
                TaskSnapshot(task=task, status=TaskStatus.QUEUED, attempts=0) for task in self.tasks
            ],
            created_at=now,
            updated_at=now,
        )

    def get_run_summary(self, run_id: str) -> RunSummary:
        """Aggregate the fake's task state like the production status query."""

        if run_id in self.earlier_runs:
            run, tasks = self.earlier_runs[run_id]
        elif self.run is not None and self.run.id == run_id:
            run, tasks = self.run, self.tasks
        else:
            raise KeyError(run_id)
        now = datetime.now(UTC)
        counts: dict[tuple[TaskStage, TaskStatus], int] = {}
        for task in tasks:
            key = (task.stage, TaskStatus.QUEUED)
            counts[key] = counts.get(key, 0) + 1
        return RunSummary(
            run=run,
            task_counts=[
                TaskCount(stage=stage, status=status, count=count)
                for (stage, status), count in sorted(
                    counts.items(), key=lambda item: (item[0][0].value, item[0][1].value)
                )
            ],
            total_tasks=len(tasks),
            created_at=now,
            updated_at=now,
        )

    def put(
        self,
        run_id: str,
        kind: ArtifactKind,
        payload: bytes,
        media_type: str,
        metadata: dict[str, Any] | None = None,
        identity_key: str | None = None,
    ) -> ArtifactRef:
        checksum = sha256_hex(payload)
        artifact_id = (
            stable_id("artifact", run_id, kind.value, checksum, identity_key)
            if identity_key is not None
            else stable_id("artifact", run_id, kind.value, checksum)
        )
        ref = ArtifactRef(
            id=artifact_id,
            run_id=run_id,
            kind=kind,
            media_type=media_type,
            checksum=checksum,
            size_bytes=len(payload),
            metadata=metadata or {},
        )
        self.artifacts[artifact_id] = StoredArtifact(ref=ref, payload=payload)
        return ref

    def get(self, artifact_id: str) -> StoredArtifact:
        return self.artifacts[artifact_id]

    def get_many(self, artifact_ids: list[str]) -> list[StoredArtifact]:
        """Mirror ordered batch reads so worker tests exercise the optimized capability."""

        return [self.artifacts[artifact_id] for artifact_id in artifact_ids]

    def get_run_artifacts(
        self,
        run_id: str,
        kinds: set[ArtifactKind],
    ) -> list[StoredArtifact]:
        """Return run outputs in stable identity order for local finalization tests."""

        return [
            self.artifacts[artifact_id] for artifact_id in self.get_run_artifact_ids(run_id, kinds)
        ]

    def get_run_artifact_ids(self, run_id: str, kinds: set[ArtifactKind]) -> list[str]:
        return sorted(
            artifact.ref.id
            for artifact in self.artifacts.values()
            if artifact.ref.run_id == run_id and artifact.ref.kind in kinds
        )

    def list_run_artifacts(
        self,
        run_id: str,
        kinds: set[ArtifactKind],
        *,
        linked: bool = True,
    ) -> list[ArtifactRef]:
        """Return references only; the fake links every artifact it holds."""

        del linked
        return [
            self.artifacts[artifact_id].ref
            for artifact_id in self.get_run_artifact_ids(run_id, kinds)
        ]

    def get_stage_tasks(self, run_id: str, stage: TaskStage) -> list[TaskSnapshot]:
        """Return the fake's tasks of one stage as queued snapshots."""

        if run_id in self.earlier_runs:
            tasks = self.earlier_runs[run_id][1]
        elif self.run is not None and self.run.id == run_id:
            tasks = self.tasks
        else:
            raise KeyError(run_id)
        return [
            TaskSnapshot(
                task=task,
                status=TaskStatus.QUEUED,
                attempts=0,
                output_artifact_ids=self.task_outputs.get(task.id, []),
            )
            for task in tasks
            if task.stage == stage
        ]

    def claim_task(
        self,
        worker_id: str,
        stages: set[TaskStage],
        lease_duration: timedelta,
        expected_config_digest: str | None = None,
        skip_dependency_outputs_for_stages: set[TaskStage] | None = None,
    ) -> TaskLease | None:
        _ = lease_duration
        if (
            expected_config_digest is not None
            and self.run is not None
            and self.run.config_digest != expected_config_digest
        ):
            return None
        if self.lease is None or self.lease.worker_id != worker_id:
            return None
        if self.lease.task.stage not in stages:
            return None
        lease, self.lease = self.lease, None
        if lease.task.stage in (skip_dependency_outputs_for_stages or set()):
            lease = lease.model_copy(
                update={
                    "task": lease.task.model_copy(update={"dependency_ids": []}),
                    "dependency_outputs": {},
                }
            )
        return lease

    def fail_abandoned_tasks(self) -> None:
        """Count sweeps: the fake's leases never lapse, so there is nothing to fail."""

        self.abandoned_sweeps += 1

    def record_served_configuration(self, stages: set[TaskStage], config_digest: str) -> None:
        """Remember the fleet's digest per stage so worker tests can see what it declared."""

        for stage in stages:
            self.served_configurations[stage] = config_digest

    def heartbeat(
        self, task_id: str, worker_id: str, lease_duration: timedelta, *, claim_token: str
    ) -> None:
        """Record nothing: a unit test finishes long before the first renewal is due."""

    def report_task_progress(
        self,
        task_id: str,
        worker_id: str,
        progress: TaskProgress,
        *,
        claim_token: str,
    ) -> None:
        """Record bounded in-task updates for finalization progress assertions."""

        self.progress_updates.append((task_id, worker_id, progress))

    def complete_task(
        self,
        task_id: str,
        worker_id: str,
        output_artifact_ids: list[str],
        follow_up_tasks: list[TaskDefinition] | None = None,
        barrier_task_id: str | None = None,
        *,
        claim_token: str,
    ) -> None:
        self.completed.append(
            (task_id, worker_id, output_artifact_ids, follow_up_tasks or [], barrier_task_id)
        )

    def fail_task(
        self,
        task_id: str,
        worker_id: str,
        error: dict[str, Any],
        retry_delay: timedelta,
        *,
        claim_token: str,
    ) -> None:
        self.failed.append((task_id, worker_id, error, retry_delay))

    def claim_publication(
        self, worker_id: str, lease_duration: timedelta
    ) -> PublicationLease | None:
        # Nothing here publishes to a destination, so there is never a command to lease.
        del worker_id, lease_duration
        return None

    def publish_claimed(
        self,
        publication_id: str,
        worker_id: str,
        publish: Callable[[PublicationLease], None],
    ) -> None:
        del worker_id, publish
        raise KeyError(publication_id)

    def commit_publication(self, lease: PublicationLease, commit: Callable[[], None]) -> None:
        """No publications are enqueued by this minimal store."""

        del commit
        raise KeyError(lease.id)

    def heartbeat_publication(
        self, publication_id: str, worker_id: str, lease_duration: timedelta
    ) -> None:
        del worker_id, lease_duration
        raise KeyError(publication_id)

    def fail_publication(
        self,
        publication_id: str,
        worker_id: str,
        error: dict[str, Any],
        retry_delay: timedelta,
    ) -> None:
        del worker_id, error, retry_delay
        raise KeyError(publication_id)


class RecordingPipeline:
    """Record stage inputs while returning small valid domain artifacts."""

    def __init__(self) -> None:
        self.prepared_files: list[str] = []
        self.record_failures: list[bool] = []
        self.context_file_ids: list[list[str]] = []
        self.extracted_file_ids: list[list[str]] = []
        self.document_openings: list[dict[str, str]] = []
        self.finalized_file_ids: list[list[str]] = []
        self.finalized_document_rows: list[dict[str, Any]] = []
        self.relation_columns: list[list[ColumnRelation] | None] = []
        self.published_batches: list[GraphWriteBatch] = []
        # Trace events the context stage adds, as a dataset profile would.
        self.context_trace: list[dict[str, Any]] = []

    @contextmanager
    def stage_graph_publication(
        self, batch: GraphWriteBatch, publication_id: str, generation: int
    ) -> Iterator[Callable[[], None]]:
        """Record destination delivery only when its staged commit is invoked."""

        assert publication_id and generation > 0
        yield lambda: self.published_batches.append(batch)

    def prepare_documents(
        self, files: list[Any], *, record_failures: bool = True
    ) -> PreparedDocumentShard:
        self.prepared_files.extend(str(file.path) for file in files)
        self.record_failures.append(record_failures)
        chunks = [
            chunk("public source", chunk_id=f"chunk-{file.id}", file_id=file.id) for file in files
        ]
        return PreparedDocumentShard(
            file_ids=[file.id for file in files],
            files_seen=len(files),
            documents_processed=len(files),
            chunks=chunks,
        )

    def extract_document_context(self, prepared: PreparedDocumentShard) -> PreparedDocumentShard:
        """Record context-stage input and preserve the serializable shard."""

        self.context_file_ids.append(prepared.file_ids)
        return prepared.model_copy(update={"trace": [*prepared.trace, *self.context_trace]})

    def extract_window_entities(self, prepared: PreparedDocumentShard) -> EntityWindowShard:
        """Record a first-phase extraction without copying prepared metadata."""

        self.extracted_file_ids.append(prepared.file_ids)
        return EntityWindowShard(
            file_ids=prepared.file_ids,
            chunk_ids=[chunk.id for chunk in prepared.chunks],
        )

    def extract_window_relations(
        self,
        prepared: PreparedDocumentShard,
        document_entities: list[EntityMention],
        document_openings: dict[str, str] | None = None,
        *,
        dataset_columns: list[ColumnRelation] | None = None,
    ) -> RelationWindowShard:
        """Record a second-phase extraction while accepting the shared inventory."""

        _ = document_entities
        self.document_openings.append(document_openings or {})
        self.extracted_file_ids.append(prepared.file_ids)
        self.relation_columns.append(dataset_columns)
        return RelationWindowShard(
            file_ids=prepared.file_ids,
            chunk_ids=[chunk.id for chunk in prepared.chunks],
        )

    def begin_revalidation(
        self, prepared: PreparedDocumentShard, observations: ExtractionObservations
    ) -> tuple[DocumentRevalidationShard, list[RevalidationWindow]]:
        raise AssertionError("Revalidation is not used by this stage-recording fixture")

    def revalidate_entity_window(self, window: RevalidationWindowShard) -> EntityWindowShard:
        raise AssertionError("Revalidation is not used by this stage-recording fixture")

    def revalidate_relation_window(
        self, window: RevalidationWindowShard, inventory: DocumentEntityInventoryShard
    ) -> RevalidatedRelationWindowShard:
        raise AssertionError("Revalidation is not used by this stage-recording fixture")

    def finalize_document_shards(
        self,
        shards: list[ExtractedDocumentShard],
        *,
        write: bool = True,
    ) -> GraphWriteBatch:
        assert write is False
        self.finalized_file_ids.append(
            [file_id for shard in shards for file_id in shard.prepared.file_ids]
        )
        self.finalized_document_rows.extend(
            row for shard in shards for row in shard.prepared.document_rows
        )
        return _empty_batch("test-graph")


def test_planner_persists_sources_and_defers_window_fan_out(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.md"
    first.write_text("Alpha", encoding="utf-8")
    second.write_text("Beta", encoding="utf-8")
    settings = _settings(tmp_path)
    store = MemoryDistributedStore()

    snapshot = DistributedRunPlanner(
        settings,
        LocalFileSource(tmp_path),
        store,
        store,
    ).submit("run-test")

    assert snapshot.run.status == RunStatus.QUEUED
    assert store.initial_task_streams == 1
    assert len(store.artifacts) == 2
    assert len(store.tasks) == 3
    prepare = [task for task in store.tasks if task.stage == TaskStage.PREPARE_DOCUMENT]
    final = [task for task in store.tasks if task.stage == TaskStage.FINALIZE_GRAPH]
    assert len(prepare) == 2
    assert len(final) == 1
    assert final[0].dependency_ids == []
    assert final[0].payload == {"output": {"provider": "local_artifacts"}}
    assert "database_url" not in snapshot.run.config
    llm_config = snapshot.run.config["llm"]
    assert isinstance(llm_config, dict)
    assert llm_config["api_key"] == "***"


def test_planner_places_only_snowflake_secret_references_on_final_task(tmp_path: Path) -> None:
    """Route per-run fleet output without persisting a password in the task queue."""

    document = tmp_path / "document.txt"
    document.write_text("Alpha", encoding="utf-8")
    settings = _settings(tmp_path)
    settings.writer.provider = "snowflake_bulk"
    settings.snowflake.account = "account"
    settings.snowflake.user = "user"
    settings.snowflake.database = "DB"
    settings.snowflake.schema_name = "GRAPH"
    settings.snowflake.warehouse = "WH"
    settings.snowflake.bulk_stage = "@DB.GRAPH.LOAD_STAGE"
    settings.snowflake.password = "must-not-be-persisted"
    settings.snowflake.password_environment_variable = "KG_SNOWFLAKE_PASSWORD"
    store = MemoryDistributedStore()

    DistributedRunPlanner(settings, LocalFileSource(tmp_path), store, store).submit("run-snowflake")

    final = next(task for task in store.tasks if task.stage == TaskStage.FINALIZE_GRAPH)
    serialized = str(final.payload)
    output = final.payload["output"]
    assert isinstance(output, dict)
    target = output["snowflake"]
    assert isinstance(target, dict)
    assert target["database"] == "DB"
    assert target["password_credential_slot"] == "password"
    assert "must-not-be-persisted" not in serialized


def test_planner_refills_source_uploads_behind_a_slow_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep upload slots occupied without retaining unbounded source payloads."""

    settings = _settings(tmp_path)
    store = MemoryDistributedStore()
    planner = DistributedRunPlanner(settings, LocalFileSource(tmp_path), store, store)
    third_started = Event()
    files: list[InputFile] = []
    for index in range(3):
        path = tmp_path / f"{index}.txt"
        path.write_text(str(index), encoding="utf-8")
        files.extend(LocalFileSource(path).list_files())

    def store_source(run_id: str, file: InputFile) -> ArtifactRef:
        """Block the first upload until completed work replenishes the pool."""

        if file.path.stem == "0":
            assert third_started.wait(timeout=2)
        elif file.path.stem == "2":
            third_started.set()
        payload = file.path.read_bytes()
        return ArtifactRef(
            id=f"artifact-{file.path.stem}",
            run_id=run_id,
            kind=ArtifactKind.SOURCE_DOCUMENT,
            media_type=file.mime_type,
            checksum=sha256_hex(payload),
            size_bytes=len(payload),
        )

    monkeypatch.setattr(planner, "_store_source", store_source)
    monkeypatch.setattr(
        "kg_processor.application.distributed_planner._source_staging_workers",
        lambda _files: 2,
    )

    staged = list(planner._iter_staged_sources("run-test", files))

    assert {file.id for file, _ref in staged} == {file.id for file in files}
    assert {ref.id for _file, ref in staged} == {
        "artifact-0",
        "artifact-1",
        "artifact-2",
    }
    assert third_started.is_set()


def test_source_staging_limits_memory_for_large_documents(tmp_path: Path) -> None:
    """Keep bounded source concurrency from multiplying complete-file buffers."""

    files = [
        InputFile(
            id=f"file-{index}",
            path=tmp_path / f"file-{index}.pdf",
            source_uri=f"file-{index}.pdf",
            checksum="checksum",
            mime_type="application/pdf",
            size_bytes=128 * 1024 * 1024,
        )
        for index in range(20)
    ]

    assert _source_staging_workers(files) == 4
    assert (
        _source_staging_workers([file.model_copy(update={"size_bytes": 1024}) for file in files])
        == 16
    )


def test_planner_preserves_provenance_for_byte_identical_sources(tmp_path: Path) -> None:
    """Equal bytes at different paths must produce distinct tasks and artifacts."""

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("identical public text", encoding="utf-8")
    second.write_text("identical public text", encoding="utf-8")
    settings = _settings(tmp_path)
    store = MemoryDistributedStore()

    DistributedRunPlanner(settings, LocalFileSource(tmp_path), store, store).submit("identical-run")

    source_artifacts = [
        item for item in store.artifacts.values() if item.ref.kind == ArtifactKind.SOURCE_DOCUMENT
    ]
    preparation_tasks = [task for task in store.tasks if task.stage == TaskStage.PREPARE_DOCUMENT]
    assert len(source_artifacts) == 2
    assert len(preparation_tasks) == 2
    assert len({task.scope_id for task in preparation_tasks}) == 2
    assert {item.ref.metadata["input_file"]["filename"] for item in source_artifacts} == {
        "first.txt",
        "second.txt",
    }


def test_planner_fixes_each_file_location_under_the_run_source_root(tmp_path: Path) -> None:
    """A worker reads a staged copy under its own settings, so the location travels with it."""

    folder = tmp_path / "Lab A" / "Trial 3"
    folder.mkdir(parents=True)
    (folder / "notes.txt").write_text("public text", encoding="utf-8")
    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={"files": settings.files.model_copy(update={"input_path": tmp_path})}
    )
    store = MemoryDistributedStore()

    DistributedRunPlanner(settings, LocalFileSource(tmp_path), store, store).submit("location-run")

    [source] = [
        item for item in store.artifacts.values() if item.ref.kind == ArtifactKind.SOURCE_DOCUMENT
    ]
    assert source.ref.metadata["input_file"]["location"] == "Lab A/Trial 3/notes.txt"


def test_a_revision_keeps_replaces_skips_and_drops_against_its_base(tmp_path: Path) -> None:
    """Plan a revision from what the base holds and what the source offers.

    Held bytes under any name are skipped, a held identity with new bytes is
    replaced, a dropped document leaves the kept set, and the finalizer names
    the kept documents per run it inherits them from - flattening a base that
    itself inherited.
    """

    settings = _settings(tmp_path)
    store = MemoryDistributedStore()
    base = RunDefinition(
        id="base-run",
        graph_id="test-graph",
        config={},
        config_digest="d",
        status=RunStatus.SUCCEEDED,
    )
    store.earlier_runs["base-run"] = (
        base,
        [
            TaskDefinition(
                id="base-final",
                run_id="base-run",
                stage=TaskStage.FINALIZE_GRAPH,
                scope_id="test-graph",
                payload={
                    "output": {"provider": "local_artifacts"},
                    "inherit": [{"run_id": "older-run", "file_ids": ["older-doc"]}],
                },
            )
        ],
    )
    store.earlier_runs["older-run"] = (base.model_copy(update={"id": "older-run"}), [])
    # The base extracted two documents itself and inherited one from an older run.
    for run_id, file_id, text in [
        ("base-run", "kept-doc", "kept text"),
        ("base-run", "changed-doc", "old text"),
        ("older-run", "older-doc", "older text"),
    ]:
        payload = text.encode()
        store.put(
            run_id,
            ArtifactKind.SOURCE_DOCUMENT,
            payload,
            "text/plain",
            metadata={"input_file": {"id": file_id, "checksum": sha256_hex(payload)}},
            identity_key=file_id,
        )
        store.put(
            run_id,
            ArtifactKind.EXTRACTED_DOCUMENT,
            b"{}",
            "application/json",
            metadata={"file_ids": [file_id]},
            identity_key=file_id,
        )
    offered = [
        _input_file(tmp_path, "again.txt", "kept text", file_id="renamed-copy"),
        _input_file(tmp_path, "changed.txt", "new text", file_id="changed-doc"),
        _input_file(tmp_path, "new.txt", "new document", file_id="new-doc"),
    ]

    DistributedRunPlanner(settings, _StaticFileSource(offered), store, store).submit(
        "revision-run",
        RevisionRequest(base_run_id="base-run", drop_file_ids=["older-doc"]),
    )

    prepare = sorted(t.scope_id for t in store.tasks if t.stage == TaskStage.PREPARE_DOCUMENT)
    assert prepare == ["changed-doc", "new-doc"]
    (final,) = [t for t in store.tasks if t.stage == TaskStage.FINALIZE_GRAPH]
    assert final.payload["inherit"] == [{"run_id": "base-run", "file_ids": ["kept-doc"]}]

    with pytest.raises(ValueError, match="not in the graph: missing-doc"):
        DistributedRunPlanner(settings, _StaticFileSource([]), store, store).submit(
            "bad-revision",
            RevisionRequest(base_run_id="base-run", drop_file_ids=["missing-doc"]),
        )
    store.earlier_runs["base-run"] = (base.model_copy(update={"status": RunStatus.FAILED}), [])
    with pytest.raises(ValueError, match="only a succeeded run can be revised"):
        DistributedRunPlanner(settings, _StaticFileSource([]), store, store).submit(
            "bad-revision", RevisionRequest(base_run_id="base-run")
        )


def test_explicit_run_id_submit_is_idempotent_without_cancelling_live_run(
    tmp_path: Path,
) -> None:
    """Return an existing compatible run when a submit request is retried."""

    source = tmp_path / "source.txt"
    source.write_text("public text", encoding="utf-8")
    settings = _settings(tmp_path)
    store = MemoryDistributedStore()
    planner = DistributedRunPlanner(settings, LocalFileSource(tmp_path), store, store)

    first = planner.submit("stable-run")
    second = planner.submit("stable-run")

    assert second.run.id == first.run.id
    assert store.cancelled_run_ids == []


def test_every_distributed_run_records_who_its_graph_belongs_to(tmp_path: Path) -> None:
    """A new graph needs an owner; a revision keeps the one its base recorded.

    The console shows a graph only to its owner and the people they share it
    with, so a graph started without one would be open to nobody.
    """

    source = tmp_path / "source.txt"
    source.write_text("public text", encoding="utf-8")
    ownerless = _settings(tmp_path).model_copy(
        update={"job": _settings(tmp_path).job.model_copy(update={"owner": None})}
    )
    store = MemoryDistributedStore()
    with pytest.raises(ValueError, match="name the graph's owner"):
        DistributedRunPlanner(ownerless, LocalFileSource(tmp_path), store, store).submit("run-1")
    assert store.run is None

    planner = DistributedRunPlanner(_settings(tmp_path), LocalFileSource(tmp_path), store, store)
    summary = planner.submit("run-1")
    assert summary.run.config["owner"] == "TESTER@EXAMPLE.COM"

    store.earlier_runs["base-run"] = (
        RunDefinition(
            id="base-run",
            graph_id="test-graph",
            config={"owner": "DANA@EXAMPLE.COM"},
            config_digest="d",
            status=RunStatus.SUCCEEDED,
        ),
        [],
    )
    revising = DistributedRunPlanner(ownerless, LocalFileSource(tmp_path), store, store)
    assert revising._run_owner(RevisionRequest(base_run_id="base-run")) == "DANA@EXAMPLE.COM"
    with pytest.raises(ValueError, match="name the graph's owner"):
        revising._run_owner(None)


def test_an_owner_is_a_sign_in_address_kept_in_one_spelling() -> None:
    """``job.owner`` is validated and stored upper case, as the console keys people."""

    settings = Settings.load(env={}, overrides={"job": {"owner": " Dana@Example.com "}})
    assert settings.job.owner == "DANA@EXAMPLE.COM"
    assert Settings.load(env={}, overrides={"job": {"owner": ""}}).job.owner is None
    with pytest.raises(ValueError, match="sign-in address"):
        Settings.load(env={}, overrides={"job": {"owner": "not an address!"}})


def test_worker_prepares_source_from_artifact_without_shared_input_mount(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = _worker_store(settings)
    source = b"public source"
    source_ref = store.put(
        "run-test",
        ArtifactKind.SOURCE_DOCUMENT,
        source,
        "text/plain",
        metadata={
            "input_file": {
                "id": "file-1",
                "source_uri": "memory://file-1",
                "checksum": sha256_hex(source),
                "mime_type": "text/plain",
                "size_bytes": len(source),
                "filename": "source.txt",
            }
        },
    )
    task = _task(TaskStage.PREPARE_DOCUMENT, payload={"source_artifact_id": source_ref.id})
    store.lease = _lease(task)
    pipeline = RecordingPipeline()

    result = _worker(settings, store, pipeline, TaskStage.PREPARE_DOCUMENT).process_one()

    assert result.succeeded is True
    assert store.served_configurations == {
        TaskStage.PREPARE_DOCUMENT: distributed_processing_config_digest(settings)
    }
    assert len(store.completed) == 1
    output = store.get(store.completed[0][2][0])
    assert output.ref.kind == ArtifactKind.PREPARED_DOCUMENT
    prepared = PreparedDocumentShard.model_validate_json(output.payload)
    assert prepared.file_ids == ["file-1"]
    follow_ups = store.completed[0][3]
    assert len(follow_ups) == 1
    assert follow_ups[0].stage == TaskStage.EXTRACT_DOCUMENT_CONTEXT
    assert follow_ups[0].dependency_ids == [task.id]
    assert follow_ups[0].payload == {}
    assert store.completed[0][4] == stable_id("task", "run-test", TaskStage.FINALIZE_GRAPH.value)
    assert pipeline.prepared_files[0].endswith("source.txt")
    assert not Path(pipeline.prepared_files[0]).exists()


@pytest.mark.parametrize(("attempt", "records"), [(1, False), (2, False), (3, True)])
def test_worker_records_an_unreadable_document_only_on_its_last_attempt(
    tmp_path: Path, attempt: int, records: bool
) -> None:
    """A parser that was briefly down costs an attempt; only the last one gives up.

    Earlier attempts raise the failure so the task is retried. The last records
    the document in its trace and lets the run go on without it.
    """

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    source = b"public source"
    source_ref = store.put(
        "run-test",
        ArtifactKind.SOURCE_DOCUMENT,
        source,
        "text/plain",
        metadata={
            "input_file": {
                "id": "file-1",
                "source_uri": "memory://file-1",
                "checksum": sha256_hex(source),
                "mime_type": "text/plain",
                "size_bytes": len(source),
                "filename": "source.txt",
            }
        },
    )
    task = _task(TaskStage.PREPARE_DOCUMENT, payload={"source_artifact_id": source_ref.id})
    assert task.max_attempts == 3
    store.lease = _lease(task).model_copy(update={"attempt": attempt})
    pipeline = RecordingPipeline()

    _worker(settings, store, pipeline, TaskStage.PREPARE_DOCUMENT).process_one()

    assert pipeline.record_failures == [records]


def test_context_stage_reuses_prepared_shard_and_fans_out_windows(tmp_path: Path) -> None:
    """One context task must precede all dynamically queueable body windows."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        chunks=[chunk("public source", chunk_id="chunk-1", file_id="file-1")],
    )
    prepared_ref = store.put(
        "run-test",
        ArtifactKind.PREPARED_DOCUMENT,
        prepared.model_dump_json().encode(),
        "application/json",
    )
    task = _task(
        TaskStage.EXTRACT_DOCUMENT_CONTEXT,
        dependency_ids=["prepare-1"],
    )
    store.lease = _lease(task, {"prepare-1": [prepared_ref.id]})
    pipeline = RecordingPipeline()

    result = _worker(
        settings,
        store,
        pipeline,
        TaskStage.EXTRACT_DOCUMENT_CONTEXT,
    ).process_one()

    assert result.succeeded is True
    assert pipeline.context_file_ids == [["file-1"]]
    output = store.get(store.completed[0][2][0])
    assert output.ref.kind == ArtifactKind.DOCUMENT_CONTEXT
    contextualized = DocumentContextShard.model_validate_json(output.payload)
    assert contextualized.file_ids == ["file-1"]
    # The document's opening is read here, where all of its chunks are, once.
    assert contextualized.document_openings == {"file-1": "public source"}
    follow_ups = store.completed[0][3]
    assert len(follow_ups) == 2
    window_task = next(item for item in follow_ups if item.stage == TaskStage.EXTRACT_ENTITY_WINDOW)
    compact_task = next(
        item for item in follow_ups if item.stage == TaskStage.COMPACT_ENTITY_INVENTORY
    )
    assert window_task.dependency_ids == [task.id]
    assert set(compact_task.dependency_ids) == {task.id, window_task.id}
    assert compact_task.payload["prepared_artifact_id"] == prepared_ref.id
    window_ref = store.get(window_task.payload["window_artifact_id"])
    assert window_ref.ref.kind == ArtifactKind.EXTRACTION_WINDOW
    window = ExtractionWindowShard.model_validate_json(window_ref.payload)
    assert [chunk.id for chunk in window.chunks] == ["chunk-1"]
    assert store.completed[0][4] == stable_id("task", "run-test", TaskStage.FINALIZE_GRAPH.value)


@pytest.mark.parametrize("catalogue_relations", [True, False])
def test_a_catalogue_sends_its_column_relations_to_its_relation_windows(
    tmp_path: Path, catalogue_relations: bool
) -> None:
    """The context stage hands a catalogue's columns to every window it fans out."""

    loaded = _settings(tmp_path)
    settings = loaded.model_copy(
        update={
            "graph": loaded.graph.model_copy(update={"catalogue_relations": catalogue_relations})
        }
    )
    store = _worker_store(settings)
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        chunks=[chunk("Steel bolt\tNorthwind Metals", chunk_id="chunk-1", file_id="file-1")],
    )
    prepared_ref = store.put(
        "run-test",
        ArtifactKind.PREPARED_DOCUMENT,
        prepared.model_dump_json().encode(),
        "application/json",
    )
    task = _task(TaskStage.EXTRACT_DOCUMENT_CONTEXT, dependency_ids=["prepare-1"])
    store.lease = _lease(task, {"prepare-1": [prepared_ref.id]})
    columns = [
        ColumnRelation(column="Component", role="subject", entity_type="COMPONENT"),
        ColumnRelation(
            column="Supplier",
            role="relation",
            entity_type="ORGANIZATION",
            relation_type="SUPPLIES",
            source="column",
        ),
    ]
    pipeline = RecordingPipeline()
    pipeline.context_trace = [
        {
            "stage": "dataset_profile",
            "document_id": "file-1",
            "kind": "catalogue",
            "columns": [column.model_dump(exclude_none=True) for column in columns],
        }
    ]

    result = _worker(settings, store, pipeline, TaskStage.EXTRACT_DOCUMENT_CONTEXT).process_one()

    assert result.succeeded is True
    window_task = next(
        item for item in store.completed[0][3] if item.stage == TaskStage.EXTRACT_ENTITY_WINDOW
    )
    window = ExtractionWindowShard.model_validate_json(
        store.get(window_task.payload["window_artifact_id"]).payload
    )
    assert window.extract_relations is catalogue_relations
    assert window.dataset_columns == (columns if catalogue_relations else [])


def test_a_relation_window_reads_its_rows_by_the_columns_it_carries(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = _worker_store(settings)
    columns = [ColumnRelation(column="Component", role="subject", entity_type="COMPONENT")]
    inventory_ref = store.put(
        "run-test",
        ArtifactKind.DOCUMENT_ENTITY_INVENTORY,
        DocumentEntityInventoryShard(file_ids=["file-1"], document_openings={})
        .model_dump_json()
        .encode(),
        "application/json",
    )
    window = ExtractionWindowShard(
        file_ids=["file-1"],
        chunks=[chunk("Steel bolt", chunk_id="chunk-1", file_id="file-1")],
        dataset_columns=columns,
    )
    window_ref = store.put(
        "run-test",
        ArtifactKind.EXTRACTION_WINDOW,
        window.model_dump_json().encode(),
        "application/json",
    )
    task = _task(
        TaskStage.EXTRACT_RELATION_WINDOW,
        payload={"window_artifact_id": window_ref.id},
        dependency_ids=["inventory"],
    )
    store.lease = _lease(task, {"inventory": [inventory_ref.id]})
    pipeline = RecordingPipeline()

    result = _worker(settings, store, pipeline, TaskStage.EXTRACT_RELATION_WINDOW).process_one()

    assert result.succeeded is True
    assert pipeline.relation_columns == [columns]


class _FailingWindowPipeline(RecordingPipeline):
    """Fail every window extraction with ``error``, as a provider might."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def extract_window_entities(self, prepared: PreparedDocumentShard) -> EntityWindowShard:
        raise self.error


@pytest.mark.parametrize(
    ("attempt", "error", "recorded"),
    [
        (1, RuntimeError("all 1 extraction windows failed"), False),
        (3, RuntimeError("all 1 extraction windows failed"), True),
        (3, RuntimeError("401 Unauthorized: invalid api key"), False),
    ],
)
def test_a_window_that_fails_its_last_attempt_is_recorded_and_the_document_goes_on(
    tmp_path: Path, attempt: int, error: Exception, recorded: bool
) -> None:
    """A failed window is asked again while attempts remain; the last one records it.

    The recorded window carries a ``window_error`` event and no records, so its
    document's inventory still counts every window. A provider that refuses
    every request is no one window's failure and fails the task on any attempt.
    """

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    context_ref = store.put(
        "run-test",
        ArtifactKind.DOCUMENT_CONTEXT,
        DocumentContextShard(file_ids=["file-1"], document_openings={}).model_dump_json().encode(),
        "application/json",
    )
    window_ref = store.put(
        "run-test",
        ArtifactKind.EXTRACTION_WINDOW,
        ExtractionWindowShard(
            file_ids=["file-1"],
            chunks=[chunk("public source", chunk_id="chunk-1", file_id="file-1")],
        )
        .model_dump_json()
        .encode(),
        "application/json",
    )
    task = _task(
        TaskStage.EXTRACT_ENTITY_WINDOW,
        payload={"window_artifact_id": window_ref.id},
        dependency_ids=["context"],
    )
    store.lease = _lease(task, {"context": [context_ref.id]}).model_copy(
        update={"attempt": attempt}
    )

    result = _worker(
        settings, store, _FailingWindowPipeline(error), TaskStage.EXTRACT_ENTITY_WINDOW
    ).process_one()

    assert result.succeeded is recorded
    if not recorded:
        assert [failure[0] for failure in store.failed] == [task.id]
        return
    (output_id,) = store.completed[0][2]
    shard = EntityWindowShard.model_validate_json(store.get(output_id).payload)
    assert shard.chunk_ids == ["chunk-1"]
    assert shard.entities == []
    (event,) = shard.trace
    assert event["stage"] == "window_error"
    assert event["window_id"] == task.scope_id
    assert event["error"]["error_type"] == "RuntimeError"


def test_context_stage_queues_one_task_per_window(tmp_path: Path) -> None:
    """Expose every window to the fleet's slots, so no lease hides a serial wave."""

    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "graph": settings.graph.model_copy(
                update={"extraction_window_tokens": 100, "max_chunks_per_llm_call": 1}
            )
        }
    )
    chunks = [
        chunk(
            "public source",
            chunk_id=f"chunk-{index}",
            file_id="file-1",
            chunk_index=index,
            content_hash=sha256_hex(f"chunk-{index}"),
        )
        for index in range(5)
    ]
    store = _worker_store(settings)
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        chunks=chunks,
    )
    prepared_ref = store.put(
        "run-test",
        ArtifactKind.PREPARED_DOCUMENT,
        prepared.model_dump_json().encode(),
        "application/json",
    )
    task = _task(TaskStage.EXTRACT_DOCUMENT_CONTEXT, dependency_ids=["prepare-1"])
    store.lease = _lease(task, {"prepare-1": [prepared_ref.id]})

    result = _worker(
        settings,
        store,
        RecordingPipeline(),
        TaskStage.EXTRACT_DOCUMENT_CONTEXT,
    ).process_one()

    assert result.succeeded is True
    window_tasks = [
        item for item in store.completed[0][3] if item.stage == TaskStage.EXTRACT_ENTITY_WINDOW
    ]
    shards = [
        ExtractionWindowShard.model_validate_json(
            store.get(item.payload["window_artifact_id"]).payload
        )
        for item in window_tasks
    ]
    assert [[item.id for item in shard.chunks] for shard in shards] == [
        [f"chunk-{index}"] for index in range(5)
    ]
    windows = build_extraction_windows(chunks, 100, 1)
    assert [task.scope_id for task in window_tasks] == [window.id for window in windows]


def test_context_stage_skips_reference_window_without_dropping_later_content(
    tmp_path: Path,
) -> None:
    """Keep an appendix after references while avoiding a bibliography LLM call."""

    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "graph": settings.graph.model_copy(
                update={"extraction_window_tokens": 100, "max_chunks_per_llm_call": 1}
            )
        }
    )
    body = chunk("public source", chunk_id="body", file_id="file-1")
    references = body.model_copy(
        update={
            "id": "references",
            "page_number": 2,
            "chunk_index": 1,
            "content": (
                "References\n[1] A. Author. First paper. 2020.\n[2] B. Author. Second paper. 2021."
            ),
            "content_hash": sha256_hex("references"),
        }
    )
    appendix = body.model_copy(
        update={
            "id": "appendix",
            "page_number": 3,
            "chunk_index": 2,
            "content": "Appendix A reports additional ablation experiments and results.",
            "content_hash": sha256_hex("appendix"),
        }
    )
    store = _worker_store(settings)
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        chunks=[body, references, appendix],
    )
    prepared_ref = store.put(
        "run-test",
        ArtifactKind.PREPARED_DOCUMENT,
        prepared.model_dump_json().encode(),
        "application/json",
    )
    task = _task(TaskStage.EXTRACT_DOCUMENT_CONTEXT, dependency_ids=["prepare-1"])
    store.lease = _lease(task, {"prepare-1": [prepared_ref.id]})

    result = _worker(
        settings,
        store,
        RecordingPipeline(),
        TaskStage.EXTRACT_DOCUMENT_CONTEXT,
    ).process_one()

    assert result.succeeded is True
    follow_ups = [
        item for item in store.completed[0][3] if item.stage == TaskStage.EXTRACT_ENTITY_WINDOW
    ]
    assert [
        [
            chunk.id
            for chunk in ExtractionWindowShard.model_validate_json(
                store.get(follow_up.payload["window_artifact_id"]).payload
            ).chunks
        ]
        for follow_up in follow_ups
    ] == [["body"], ["appendix"]]


def test_planner_cancels_partial_run_when_discovered_source_changes(tmp_path: Path) -> None:
    """A stale source checksum must never become a worker artifact."""

    path = tmp_path / "changed.txt"
    path.write_text("current bytes", encoding="utf-8")
    source = _StaticFileSource(
        InputFile(
            id="changed-file",
            path=path,
            source_uri=path.as_uri(),
            checksum="0" * 64,
            mime_type="text/plain",
            size_bytes=path.stat().st_size,
        )
    )
    settings = _settings(tmp_path)
    store = MemoryDistributedStore()

    with pytest.raises(ValueError, match="changed after discovery"):
        DistributedRunPlanner(settings, source, store, store).submit("changed-run")

    assert store.cancelled_run_ids == ["changed-run"]
    assert store.tasks == []


def test_worker_extracts_prepared_dependency_into_portable_artifact(tmp_path: Path) -> None:
    """Extraction workers must consume and emit only serialized stage contracts."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    context = DocumentContextShard(
        document_openings={},
        file_ids=["file-1"],
    )
    context_ref = store.put(
        "run-test",
        ArtifactKind.DOCUMENT_CONTEXT,
        context.model_dump_json().encode(),
        "application/json",
    )
    window = ExtractionWindowShard(
        file_ids=["file-1"],
        chunks=[chunk("public source", chunk_id="chunk-1", file_id="file-1")],
    )
    window_ref = store.put(
        "run-test",
        ArtifactKind.EXTRACTION_WINDOW,
        window.model_dump_json().encode(),
        "application/json",
    )
    task = _task(
        TaskStage.EXTRACT_ENTITY_WINDOW,
        payload={"window_artifact_id": window_ref.id},
        dependency_ids=["prepare-1"],
    )
    store.lease = _lease(task, {"prepare-1": [context_ref.id]})
    pipeline = RecordingPipeline()

    result = _worker(settings, store, pipeline, TaskStage.EXTRACT_ENTITY_WINDOW).process_one()

    assert result.succeeded is True
    assert pipeline.extracted_file_ids == [["file-1"]]
    output = store.get(store.completed[0][2][0])
    assert output.ref.kind == ArtifactKind.EXTRACTED_ENTITY_WINDOW
    extracted = EntityWindowShard.model_validate_json(output.payload)
    assert extracted.chunk_ids == ["chunk-1"]


def test_entity_inventory_barrier_fans_out_relation_windows(tmp_path: Path) -> None:
    """Share all document entities with a second independently queueable wave."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    context_entity = EntityMention(
        id="paper",
        name="A Source Paper",
        type="PAPER",
        description="The focal paper.",
        source_chunk_id="front-matter",
        quote="A Source Paper",
    )
    body_entity = context_entity.model_copy(
        update={
            "id": "method",
            "name": "A Method",
            "type": "METHOD",
            "description": "The method described by the paper.",
            "source_chunk_id": "chunk-1",
            "quote": "A Method",
        }
    )
    context = DocumentContextShard(
        file_ids=["file-1"],
        document_context_entities=[context_entity],
        document_openings={"file-1": "A Source Paper"},
    )
    context_ref = store.put(
        "run-test",
        ArtifactKind.DOCUMENT_CONTEXT,
        context.model_dump_json().encode(),
        "application/json",
    )
    entity_window = EntityWindowShard(
        file_ids=["file-1"],
        chunk_ids=["chunk-1"],
        entities=[body_entity],
    )
    entity_ref = store.put(
        "run-test",
        ArtifactKind.EXTRACTED_ENTITY_WINDOW,
        entity_window.model_dump_json().encode(),
        "application/json",
    )
    window = ExtractionWindowShard(
        file_ids=["file-1"],
        chunks=[chunk("public source", chunk_id="chunk-1", file_id="file-1")],
    )
    window_ref = store.put(
        "run-test",
        ArtifactKind.EXTRACTION_WINDOW,
        window.model_dump_json().encode(),
        "application/json",
    )
    task = _task(
        TaskStage.COMPACT_ENTITY_INVENTORY,
        payload={
            "prepared_artifact_id": "prepared-ref",
            "windows": [{"window_id": "window-1", "artifact_id": window_ref.id}],
        },
        dependency_ids=["context", "entity"],
    )
    store.lease = _lease(
        task,
        {"context": [context_ref.id], "entity": [entity_ref.id]},
    )

    result = _worker(
        settings,
        store,
        RecordingPipeline(),
        TaskStage.COMPACT_ENTITY_INVENTORY,
    ).process_one()

    assert result.succeeded is True
    inventory_output = store.get(store.completed[0][2][0])
    assert inventory_output.ref.kind == ArtifactKind.DOCUMENT_ENTITY_INVENTORY
    inventory = DocumentEntityInventoryShard.model_validate_json(inventory_output.payload)
    assert {entity.id for entity in inventory.entities} == {"paper", "method"}
    assert inventory.document_openings == {"file-1": "A Source Paper"}
    follow_ups = store.completed[0][3]
    relation = next(item for item in follow_ups if item.stage == TaskStage.EXTRACT_RELATION_WINDOW)
    compact = next(item for item in follow_ups if item.stage == TaskStage.COMPACT_DOCUMENT)
    assert relation.payload["window_artifact_id"] == window_ref.id
    assert relation.dependency_ids == [task.id]
    assert set(compact.dependency_ids) == {task.id, relation.id}


def test_worker_compacts_window_results_into_one_complete_document(tmp_path: Path) -> None:
    """Keep every prepared chunk while storing context and observations only once."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    body = chunk("public source", chunk_id="body", file_id="file-1")
    references = body.model_copy(
        update={"id": "references", "chunk_index": 1, "content": "References"}
    )
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        chunks=[body, references],
        trace=[{"stage": "ocr"}],
    )
    prepared_ref = store.put(
        "run-test",
        ArtifactKind.PREPARED_DOCUMENT,
        prepared.model_dump_json().encode(),
        "application/json",
    )
    context = DocumentContextShard(
        document_openings={},
        file_ids=["file-1"],
        trace=[{"stage": "ocr"}, {"stage": "document_context"}],
    )
    inventory = DocumentEntityInventoryShard(
        document_openings={},
        file_ids=["file-1"],
        trace=context.trace,
        chunk_count=1,
        window_count=1,
    )
    inventory_ref = store.put(
        "run-test",
        ArtifactKind.DOCUMENT_ENTITY_INVENTORY,
        inventory.model_dump_json().encode(),
        "application/json",
    )
    # An endpoint a relation response added: compaction adds it to the inventory.
    added = EntityMention(
        id="added-endpoint",
        name="public",
        type="CONCEPT",
        description="",
        source_chunk_id=body.id,
        quote="public source",
    )
    relation_window = RelationWindowShard(
        file_ids=["file-1"],
        chunk_ids=[body.id],
        entities=[added],
        trace=[{"stage": "graph_extraction", "window": "body"}],
    )
    window_ref = store.put(
        "run-test",
        ArtifactKind.EXTRACTED_RELATION_WINDOW,
        relation_window.model_dump_json().encode(),
        "application/json",
    )
    task = _task(
        TaskStage.COMPACT_DOCUMENT,
        payload={"prepared_artifact_id": prepared_ref.id},
        dependency_ids=["context", "window"],
    )
    store.lease = _lease(
        task,
        {"context": [inventory_ref.id], "window": [window_ref.id]},
    )

    result = _worker(settings, store, RecordingPipeline(), TaskStage.COMPACT_DOCUMENT).process_one()

    assert result.succeeded is True
    output = store.get(store.completed[0][2][0])
    assert output.ref.kind == ArtifactKind.EXTRACTED_DOCUMENT
    compacted = ExtractedDocumentShard.model_validate_json(output.payload)
    assert [chunk.id for chunk in compacted.prepared.chunks] == ["body", "references"]
    assert compacted.prepared.trace == prepared.trace
    assert compacted.observations.chunk_count == 1
    assert compacted.observations.window_count == 1
    assert compacted.trace.count({"stage": "document_context"}) == 1
    assert [entity.id for entity in compacted.observations.entities] == ["added-endpoint"]


def test_a_relation_window_reads_its_document_opening_from_the_inventory(
    tmp_path: Path,
) -> None:
    """A window far down a document still verifies with the header that names its subject."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    inventory = DocumentEntityInventoryShard(
        file_ids=["file-1"],
        chunk_count=1,
        window_count=1,
        document_openings={"file-1": "Data sheet for Acme Paint"},
    )
    inventory_ref = store.put(
        "run-test",
        ArtifactKind.DOCUMENT_ENTITY_INVENTORY,
        inventory.model_dump_json().encode(),
        "application/json",
    )
    window = ExtractionWindowShard(
        file_ids=["file-1"],
        chunks=[chunk("Colour: blue", chunk_id="chunk-9", file_id="file-1", chunk_index=9)],
    )
    window_ref = store.put(
        "run-test",
        ArtifactKind.EXTRACTION_WINDOW,
        window.model_dump_json().encode(),
        "application/json",
    )
    task = _task(
        TaskStage.EXTRACT_RELATION_WINDOW,
        payload={"window_artifact_id": window_ref.id},
        dependency_ids=["inventory"],
    )
    store.lease = _lease(task, {"inventory": [inventory_ref.id]})
    pipeline = RecordingPipeline()

    result = _worker(settings, store, pipeline, TaskStage.EXTRACT_RELATION_WINDOW).process_one()

    assert result.succeeded is True
    assert pipeline.document_openings == [{"file-1": "Data sheet for Acme Paint"}]


def test_compaction_removes_rejected_entities_and_records_the_relations_they_cost(
    tmp_path: Path,
) -> None:
    """A mention one window rejects leaves with every relation any window built on it."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    body = chunk("Contains titanium white and other pigments.", chunk_id="body", file_id="file-1")
    prepared = PreparedDocumentShard(
        file_ids=["file-1"], files_seen=1, documents_processed=1, chunks=[body]
    )
    prepared_ref = store.put(
        "run-test",
        ArtifactKind.PREPARED_DOCUMENT,
        prepared.model_dump_json().encode(),
        "application/json",
    )
    mentions = [
        EntityMention(
            id=name,
            name=name,
            type="CONCEPT",
            description="",
            source_chunk_id=body.id,
            quote=name,
        )
        for name in ("paint", "titanium white", "pigments")
    ]
    inventory = DocumentEntityInventoryShard(
        document_openings={}, file_ids=["file-1"], entities=mentions, chunk_count=1, window_count=1
    )
    inventory_ref = store.put(
        "run-test",
        ArtifactKind.DOCUMENT_ENTITY_INVENTORY,
        inventory.model_dump_json().encode(),
        "application/json",
    )
    relations = [
        RelationObservation(
            id=f"paint-{target}",
            source_entity_id="paint",
            target_entity_id=target,
            relation_type="CONTAINS",
            description="",
            source_chunk_id=body.id,
            quote=body.content,
        )
        for target in ("titanium white", "pigments")
    ]
    relation_window = RelationWindowShard(
        file_ids=["file-1"],
        chunk_ids=[body.id],
        relations=relations,
        trace=[
            {
                "stage": "relation_verification",
                "entity_verdicts": [{"mention_id": "pigments", "verdict": "placeholder"}],
            }
        ],
    )
    window_ref = store.put(
        "run-test",
        ArtifactKind.EXTRACTED_RELATION_WINDOW,
        relation_window.model_dump_json().encode(),
        "application/json",
    )
    task = _task(
        TaskStage.COMPACT_DOCUMENT,
        payload={"prepared_artifact_id": prepared_ref.id},
        dependency_ids=["inventory", "window"],
    )
    store.lease = _lease(task, {"inventory": [inventory_ref.id], "window": [window_ref.id]})

    result = _worker(settings, store, RecordingPipeline(), TaskStage.COMPACT_DOCUMENT).process_one()

    assert result.succeeded is True
    compacted = ExtractedDocumentShard.model_validate_json(
        store.get(store.completed[0][2][0]).payload
    )
    observations = compacted.observations
    assert [entity.id for entity in observations.entities] == ["paint", "titanium white"]
    assert [relation.id for relation in observations.relations] == ["paint-titanium white"]
    [cascade] = [item for item in observations.trace if item["stage"] == "verification_cascade"]
    assert [
        (record["reason"], record.get("name"), record.get("target"))
        for record in cascade["rejected_records"]
    ] == [
        ("verification_placeholder", "pigments", None),
        ("verification_endpoint_rejected", "CONTAINS", "pigments"),
    ]


def test_combined_observations_include_compacted_document_context() -> None:
    """Treat a focal context mention as finalization input without a body duplicate."""

    context = EntityMention(
        id="paper-context",
        name="A Scalable Graph Method",
        type="PAPER",
        description="The focal source paper.",
        source_chunk_id="front-matter",
        quote="A Scalable Graph Method",
        is_document_context=True,
        contextual_surfaces=["we", "this paper"],
    )
    shard = ExtractedDocumentShard(
        prepared=PreparedDocumentShard(
            file_ids=["file-1"],
            files_seen=1,
            documents_processed=1,
            document_context_entities=[context],
        ),
        observations=ExtractionObservations(chunk_count=0, window_count=0),
    )

    combined = combine_extracted_shards([shard])

    assert combined.entities == [context]


def test_spark_extracted_artifact_does_not_duplicate_prepared_corpus_rows() -> None:
    """Keep Spark observations compact while preserving local self-contained shards."""

    passage = chunk(
        "A substantial prepared source passage.",
        chunk_id="chunk-1",
        file_id="file-1",
        document_id="document-1",
    )
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        document_rows=[{"id": "document-1"}],
        page_rows=[{"id": "page-1", "text": passage.content}],
        block_rows=[{"id": "block-1", "text": passage.content}],
        asset_rows=[{"id": "asset-1", "uri": "memory://asset"}],
        chunks=[passage],
        trace=[{"stage": "ocr"}],
    )

    projected = _prepared_projection_for_extracted_artifact(
        prepared,
        spark_finalization=True,
    )

    assert projected.file_ids == prepared.file_ids
    assert projected.files_seen == prepared.files_seen
    assert projected.documents_processed == prepared.documents_processed
    assert projected.document_rows == []
    assert projected.page_rows == []
    assert projected.block_rows == []
    assert projected.asset_rows == []
    assert projected.chunks == []
    assert projected.trace == []
    assert (
        _prepared_projection_for_extracted_artifact(
            prepared,
            spark_finalization=False,
        )
        is prepared
    )


def test_worker_retries_transient_coordination_outage_without_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep an idle worker alive when its durable queue is briefly unavailable."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    claim_calls = 0

    def claim_task(
        worker_id: str,
        stages: set[TaskStage],
        lease_duration: timedelta,
        expected_config_digest: str | None = None,
        skip_dependency_outputs_for_stages: set[TaskStage] | None = None,
    ) -> TaskLease | None:
        """Fail once like a DNS outage, then return an ordinary empty poll."""

        nonlocal claim_calls
        _ = (
            worker_id,
            stages,
            lease_duration,
            expected_config_digest,
            skip_dependency_outputs_for_stages,
        )
        claim_calls += 1
        if claim_calls == 1:
            raise TaskStoreUnavailableError("temporary coordination outage")
        return None

    monkeypatch.setattr(store, "claim_task", claim_task)
    monkeypatch.setattr("kg_processor.application.distributed_worker.sleep", lambda _: None)
    iterations: list[Any] = []

    completed = _worker(
        settings,
        store,
        RecordingPipeline(),
        TaskStage.EXTRACT_ENTITY_WINDOW,
    ).run_until_stopped(
        lambda: claim_calls >= 2,
        iterations.append,
    )

    assert completed == 0
    assert claim_calls == 2
    assert len(iterations) == 2
    assert iterations[0].claimed is False
    assert iterations[0].succeeded is False
    assert iterations[0].error == {
        "error_type": "TaskStoreUnavailableError",
        "error_message": "temporary coordination outage",
    }
    assert iterations[1].claimed is False
    assert iterations[1].error is None


def test_worker_finalizes_all_dependency_shards_in_stable_order(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = _worker_store(settings)
    dependency_outputs: dict[str, list[str]] = {}
    for dependency_id, file_id in [("extract-b", "b"), ("extract-a", "a")]:
        prepared = PreparedDocumentShard(
            file_ids=[file_id],
            files_seen=1,
            documents_processed=1,
        )
        shard = ExtractedDocumentShard(
            prepared=prepared,
            observations=ExtractionObservations(chunk_count=0, window_count=0),
        )
        ref = store.put(
            "run-test",
            ArtifactKind.EXTRACTED_DOCUMENT,
            shard.model_dump_json().encode(),
            "application/json",
        )
        dependency_outputs[dependency_id] = [ref.id]
    task = _task(
        TaskStage.FINALIZE_GRAPH,
        dependency_ids=["extract-a", "extract-b"],
    )
    store.lease = _lease(task, dependency_outputs)
    pipeline = RecordingPipeline()

    result = _worker(settings, store, pipeline, TaskStage.FINALIZE_GRAPH).process_one()

    assert result.succeeded is True
    assert pipeline.finalized_file_ids == [["a", "b"]]
    graph = store.get(store.completed[0][2][0])
    assert graph.ref.kind == ArtifactKind.GRAPH_RESULT
    assert GraphWriteBatch.model_validate_json(graph.payload).graph_id == "test-graph"
    assert [update[2].phase for update in store.progress_updates] == [
        "read_artifacts",
        "read_artifacts",
        "materialize_inputs",
        "resolve_entities",
        "resolve_entities",
        "assemble_graph",
        "enrich_graph",
        "build_communities",
        "quality_checks",
        "write_tables",
        "publish_destination",
        "persist_manifest",
        "persist_manifest",
    ]
    assert store.progress_updates[-1][2].phase_index == 12
    assert store.progress_updates[-1][2].completed == 1


def test_a_revision_finalizes_kept_shards_beside_its_own(tmp_path: Path) -> None:
    """The finalizer reads the documents a revision kept from the runs that made them."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)

    def shard(run_id: str, file_id: str) -> str:
        item = ExtractedDocumentShard(
            prepared=PreparedDocumentShard(file_ids=[file_id], files_seen=1, documents_processed=1),
            observations=ExtractionObservations(chunk_count=0, window_count=0),
        )
        return store.put(
            run_id,
            ArtifactKind.EXTRACTED_DOCUMENT,
            item.model_dump_json().encode(),
            "application/json",
            metadata={"file_ids": [file_id]},
            identity_key=file_id,
        ).id

    kept = shard("base-run", "kept")
    shard("base-run", "dropped")
    own = shard("run-test", "added")
    task = _task(
        TaskStage.FINALIZE_GRAPH,
        payload={
            "output": {"provider": "local_artifacts"},
            "inherit": [{"run_id": "base-run", "file_ids": ["kept"]}],
        },
    )
    store.lease = _lease(task, {})
    pipeline = RecordingPipeline()

    result = _worker(settings, store, pipeline, TaskStage.FINALIZE_GRAPH).process_one()

    assert result.succeeded is True
    assert pipeline.finalized_file_ids == [["added", "kept"]]
    assert {kept, own} <= set(store.artifacts)


def test_task_progress_rejects_inconsistent_phase_and_inner_counters() -> None:
    """Keep malformed worker progress out of durable operator status payloads."""

    with pytest.raises(ValueError, match="phase_index"):
        TaskProgress(phase="write", phase_index=3, phase_total=2)
    with pytest.raises(ValueError, match="completed"):
        TaskProgress(
            phase="write",
            phase_index=1,
            phase_total=2,
            completed=3,
            total=2,
        )


def test_worker_does_not_claim_configuration_mismatch(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = _worker_store(settings)
    assert store.run is not None
    store.run = store.run.model_copy(update={"config_digest": "different"})
    store.lease = _lease(_task(TaskStage.EXTRACT_ENTITY_WINDOW))

    result = _worker(
        settings,
        store,
        RecordingPipeline(),
        TaskStage.EXTRACT_ENTITY_WINDOW,
    ).process_one()

    assert result.claimed is False
    assert result.succeeded is None
    assert store.completed == []
    assert store.failed == []
    assert store.lease is not None


def test_worker_stays_alive_when_lost_lease_blocks_failure_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report a lost completion lease without crashing the long-running worker."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    store.lease = _lease(_task(TaskStage.EXTRACT_ENTITY_WINDOW))
    monkeypatch.setattr(
        store,
        "complete_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("lease lost")),
    )
    monkeypatch.setattr(
        store,
        "fail_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("not owner")),
    )

    result = _worker(
        settings,
        store,
        RecordingPipeline(),
        TaskStage.EXTRACT_ENTITY_WINDOW,
    ).process_one()

    assert result.succeeded is False
    assert result.error is not None
    assert result.error["transition_error_type"] == "RuntimeError"


def test_the_ontology_is_the_runs_not_the_fleets(tmp_path: Path) -> None:
    """Keep what a graph extracts out of what makes a worker eligible.

    A fleet serves many graphs, and which vocabulary each extracts is chosen
    per run. Judged by the ontology, two runs with different types would need
    two fleets; judged without it, one fleet claims both and executes each
    with the ontology its run carries. So the digest ignores the profile in
    every form - inline, mounted, empty, or with a plain YAML date in it.
    """

    settings = _settings(tmp_path)
    baseline = distributed_processing_config_digest(settings)
    profile = {
        "name": "general",
        "mode": "closed",
        "entity_types": [{"name": "PERSON", "description": "A named human being."}],
        "relation_types": [{"name": "related_to", "description": "A stated association."}],
    }
    mounted_path = tmp_path / "ontology.yaml"
    mounted_path.write_text(
        yaml.safe_dump({**profile, "revised": "2026-08-07"}, sort_keys=True), encoding="utf-8"
    )
    for ontology in (
        settings.ontology.model_copy(update={"profile": profile}),
        settings.ontology.model_copy(update={"profile": None, "profile_path": mounted_path}),
        settings.ontology.model_copy(update={"profile": {}}),
    ):
        assert (
            distributed_processing_config_digest(settings.model_copy(update={"ontology": ontology}))
            == baseline
        )
    relabelled = settings.model_copy(
        update={"graph": settings.graph.model_copy(update={"entity_types": ["WIDGET"]})}
    )
    assert distributed_processing_config_digest(relabelled) == baseline
    assert "ontology" not in distributed_processing_compatibility_config(settings)


def test_a_worker_executes_a_run_with_the_ontology_the_run_carries(tmp_path: Path) -> None:
    """Build each graph with its own vocabulary on one fleet.

    The run's stored configuration carries the profile the console composed;
    the worker applies it over its own providers. A run that carries none
    executes with the worker's mounted profile, and two runs never share one
    resolved pipeline.
    """

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    profile = {
        "name": "widgets",
        "description": "Only widgets.",
        "mode": "open",
        "entity_types": [{"name": "WIDGET", "description": "A named widget."}],
    }
    assert store.run is not None
    store.run = store.run.model_copy(
        update={"config": {**store.run.config, "ontology": {"profile": profile}}}
    )
    source = b"public source"
    source_ref = store.put(
        "run-test",
        ArtifactKind.SOURCE_DOCUMENT,
        source,
        "text/plain",
        metadata={
            "input_file": {
                "id": "file-1",
                "source_uri": "memory://file-1",
                "checksum": sha256_hex(source),
                "mime_type": "text/plain",
                "size_bytes": len(source),
                "filename": "source.txt",
            }
        },
    )
    store.lease = _lease(
        _task(TaskStage.PREPARE_DOCUMENT, payload={"source_artifact_id": source_ref.id})
    )
    default_pipeline = RecordingPipeline()
    built_for: list[Settings] = []
    run_pipeline = RecordingPipeline()

    def pipeline_for(run_settings: Settings) -> RecordingPipeline:
        built_for.append(run_settings)
        return run_pipeline

    worker = DistributedWorker(
        settings,
        "worker-1",
        {TaskStage.PREPARE_DOCUMENT},
        default_pipeline,
        store,
        store,
        pipeline_for=pipeline_for,
    )
    assert worker.process_one().succeeded is True

    (run_settings,) = built_for
    assert run_settings.ontology.profile == profile
    assert run_settings.ontology.profile_path is None
    assert len(run_pipeline.prepared_files) == 1
    assert default_pipeline.prepared_files == []

    # An omitted profile keeps the mounted vocabulary in a run-scoped pipeline.
    store.run = store.run.model_copy(update={"config": {**store.run.config, "ontology": {}}})
    store.lease = _lease(
        _task(TaskStage.PREPARE_DOCUMENT, payload={"source_artifact_id": source_ref.id})
    )
    worker = DistributedWorker(
        settings,
        "worker-1",
        {TaskStage.PREPARE_DOCUMENT},
        default_pipeline,
        store,
        store,
        pipeline_for=pipeline_for,
    )
    assert worker.process_one().succeeded is True
    assert len(built_for) == 2
    assert built_for[-1].ontology == settings.ontology
    assert built_for[-1].job.job_id == "run-test"
    assert len(run_pipeline.prepared_files) == 2
    assert default_pipeline.prepared_files == []


def test_processing_digest_ignores_worker_local_transport_settings(tmp_path: Path) -> None:
    """Allow equivalent workers to route to different replicas and local paths."""

    settings = _settings(tmp_path)
    relocated = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={"endpoint": "http://node-local-model/v1", "api_key": "local-key"}
            ),
            "embedding": settings.embedding.model_copy(
                update={"endpoint": "http://node-local-embedding/v1", "batch_size": 64}
            ),
            "writer": settings.writer.model_copy(
                update={"output_path": tmp_path / "different-output"}
            ),
            "cache": settings.cache.model_copy(update={"path": tmp_path / "different-cache"}),
            "graph": settings.graph.model_copy(
                update={
                    "extraction_parallelism": 8,
                    "finalization_provider_concurrency": 64,
                }
            ),
        }
    )

    assert distributed_processing_config_digest(relocated) == (
        distributed_processing_config_digest(settings)
    )

    different_model = relocated.model_copy(
        update={"llm": relocated.llm.model_copy(update={"model": "different-model"})}
    )
    assert distributed_processing_config_digest(different_model) != (
        distributed_processing_config_digest(settings)
    )


def test_workers_reading_different_prompts_do_not_share_one_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate images by what they will actually ask the model, not by a label.

    Prompt text and the response contract are executable extraction policy. Held
    apart only by a revision constant someone has to remember to bump, an old and
    a new worker image compute the same digest during a rolling update, both
    claim tasks from the same run, and the graph is assembled from observations
    produced under two different sets of instructions.
    """

    settings = _settings(tmp_path)
    baseline = distributed_processing_config_digest(settings)
    compatibility = distributed_processing_compatibility_config(settings)
    assert compatibility["extraction_prompts"]
    assert compatibility["extraction_contracts"]

    monkeypatch.setattr(
        "kg_processor.application.distributed_planner.two_pass_prompt_fingerprints",
        lambda: {"entity_extraction": {"revision": "changed", "checksum": "0" * 64}},
    )
    assert distributed_processing_config_digest(settings) != baseline
    monkeypatch.undo()

    monkeypatch.setattr(
        "kg_processor.application.distributed_planner.extraction_contract_fingerprint",
        lambda: "0" * 64,
    )
    assert distributed_processing_config_digest(settings) != baseline


def test_processing_digest_ignores_graph_identity(tmp_path: Path) -> None:
    """Allow one immutable worker profile to process independently named graphs."""

    settings = _settings(tmp_path)
    renamed = settings.model_copy(
        update={"job": settings.job.model_copy(update={"graph_id": "another-graph"})}
    )

    assert distributed_processing_config_digest(renamed) == (
        distributed_processing_config_digest(settings)
    )


def _settings(tmp_path: Path) -> Settings:
    """Return a complete lightweight processing profile for distributed tests."""

    return Settings.load(
        env={},
        overrides={
            "job": {"graph_id": "test-graph", "owner": "tester@example.com"},
            "files": {"input_path": str(tmp_path)},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "api_key": "test-secret"},
            "embedding": {"provider": "hash", "dimension": 8},
            "writer": {"provider": "local_artifacts", "output_path": str(tmp_path / "out")},
            "cache": {"provider": "none"},
            "distributed": {"database_url": "postgresql://user:secret@db/test"},
        },
    )


class _StaticFileSource:
    """Return preconstructed file records for source-integrity and revision tests."""

    def __init__(self, input_files: InputFile | list[InputFile]) -> None:
        self.input_files = input_files if isinstance(input_files, list) else [input_files]

    def list_files(self) -> list[InputFile]:
        """Return the records without recomputing an intentionally stale checksum."""

        return list(self.input_files)


def _input_file(directory: Path, name: str, text: str, *, file_id: str) -> InputFile:
    """Write a small source and describe it the way discovery would."""

    path = directory / name
    payload = text.encode("utf-8")
    path.write_bytes(payload)
    return InputFile(
        id=file_id,
        path=path,
        source_uri=str(path),
        checksum=sha256_hex(payload),
        mime_type="text/plain",
        size_bytes=len(payload),
    )


def _worker_store(settings: Settings) -> MemoryDistributedStore:
    """Create a store containing the run identity expected by a worker."""

    store = MemoryDistributedStore()
    store.run = RunDefinition(
        id="run-test",
        graph_id=settings.job.graph_id,
        config={},
        config_digest=distributed_processing_config_digest(settings),
        status=RunStatus.RUNNING,
    )
    return store


def _task(
    stage: TaskStage,
    *,
    payload: dict[str, Any] | None = None,
    dependency_ids: list[str] | None = None,
) -> TaskDefinition:
    """Build one stable task definition for a worker test."""

    return TaskDefinition(
        id=f"task-{stage.value}",
        run_id="run-test",
        stage=stage,
        scope_id="file-1",
        payload=payload or {},
        dependency_ids=dependency_ids or [],
    )


def _lease(
    task: TaskDefinition,
    dependency_outputs: dict[str, list[str]] | None = None,
) -> TaskLease:
    """Wrap a task in an owned lease that remains valid during a unit test."""

    return TaskLease(
        task=task,
        worker_id="worker-1",
        attempt=1,
        claim_token="claim-1",
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        dependency_outputs=dependency_outputs or {},
    )


def _worker(
    settings: Settings,
    store: MemoryDistributedStore,
    pipeline: RecordingPipeline,
    stage: TaskStage,
) -> DistributedWorker:
    """Construct a worker with one eligible stage and in-memory application ports."""

    return DistributedWorker(
        settings,
        "worker-1",
        {stage},
        pipeline,
        store,
        store,
    )


def _empty_batch(graph_id: str) -> GraphWriteBatch:
    """Return the smallest valid final graph artifact."""

    return GraphWriteBatch(
        graph_id=graph_id,
        documents=[],
        pages=[],
        chunks=[],
        nodes=[],
        edges=[],
        evidence=[],
        entity_sources=[],
        communities=[],
        community_findings=[],
        run_report={},
    )


def test_a_window_with_nothing_to_extract_is_not_a_gap() -> None:
    """A title slide or page of furniture genuinely contains no entities."""

    assert not is_discarded_window_event(
        {"stage": "entity_extraction", "input_records": 0, "accepted_records": 0}
    )


def test_a_window_whose_every_record_was_discarded_is_recorded_not_fatal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silence here is how a corpus loses a whole document; a failure is how it loses the corpus.

    The model returned records and validation rejected all of them. Raising
    would let one brochure title slide stop a five-hundred-document run. It is a
    gap instead: named in the log, and carried by the trace into a
    ``discarded_windows`` row the console reports per document.
    """

    event = {
        "stage": "entity_extraction",
        "window_id": "window_1",
        "document_id": "file_601352166d0d26e0ee2ac50bc2a82bc2",
        "chunk_ids": ["chunk_a", "chunk_b"],
        "input_records": 3,
        "accepted_records": 0,
        "record_actions": {"ungrounded_quote": 3, "untrusted_alias": 11},
    }
    assert is_discarded_window_event(event)
    with caplog.at_level(logging.WARNING, logger="kg_processor.application.pipeline"):
        _note_discarded_windows([event], ["file_601352166d0d26e0ee2ac50bc2a82bc2"])
    assert "recorded as a discarded window" in caplog.text
    assert "ungrounded_quote=3" in caplog.text

    chunks = [
        Chunk(
            id="chunk_a",
            file_id="file_601352166d0d26e0ee2ac50bc2a82bc2",
            document_id="file_601352166d0d26e0ee2ac50bc2a82bc2",
            page_number=2,
            chunk_index=0,
            content="An introduction to our capabilities",
            start_offset=0,
            end_offset=35,
            token_count=6,
            content_hash="h1",
        ),
        Chunk(
            id="chunk_b",
            file_id="file_601352166d0d26e0ee2ac50bc2a82bc2",
            document_id="file_601352166d0d26e0ee2ac50bc2a82bc2",
            page_number=3,
            chunk_index=1,
            content="Supporting the world's innovators",
            start_offset=35,
            end_offset=68,
            token_count=5,
            content_hash="h2",
        ),
    ]
    (row,) = discarded_windows_from_trace([event], chunks, "graph_1")
    assert row.stage == "entities"
    assert row.document_id == "file_601352166d0d26e0ee2ac50bc2a82bc2"
    assert (row.page_start, row.page_end) == (2, 3)
    assert row.extracted_records == 3
    assert row.record_actions == {"ungrounded_quote": 3, "untrusted_alias": 11}
    assert row.preview == "An introduction to our capabilities Supporting the world's innovators"
    assert discarded_window_metrics([row]) == {
        "windows": 1,
        "documents": 1,
        "by_stage": {"entities": 1},
        "record_actions": {"ungrounded_quote": 3, "untrusted_alias": 11},
    }


def test_a_window_that_kept_some_records_is_not_a_gap() -> None:
    """Partial rejection is normal validation, not a discarded window."""

    assert not is_discarded_window_event(
        {
            "stage": "entity_extraction",
            "input_records": 9,
            "accepted_records": 7,
            "record_actions": {"ungrounded_quote": 2},
        }
    )
    # Context and verification calls that keep nothing remove no text from
    # the graph, so they are not gaps either.
    assert not is_discarded_window_event(
        {"stage": "document_context_extraction", "input_records": 2, "accepted_records": 0}
    )


def test_bookkeeping_stages_are_never_starved_behind_model_calls() -> None:
    """A compaction finishes in moments and unlocks the next phase of its document.

    Sharing a pool with the model stages, it is claimed before any of them
    among runs holding as many leases: ranked below, a run whose windows were
    all done waited hours for its compaction behind another run's relation
    windows.
    """

    bookkeeping = {
        TaskStage.COMPACT_ENTITY_INVENTORY,
        TaskStage.COMPACT_DOCUMENT,
        TaskStage.REVALIDATE_DOCUMENT,
    }
    model_stages = {
        TaskStage.EXTRACT_DOCUMENT_CONTEXT,
        TaskStage.EXTRACT_ENTITY_WINDOW,
        TaskStage.EXTRACT_RELATION_WINDOW,
        TaskStage.REVALIDATE_DOCUMENT_CONTEXT,
        TaskStage.REVALIDATE_ENTITY_WINDOW,
        TaskStage.REVALIDATE_RELATION_WINDOW,
    }
    assert min(STAGE_PRIORITY[stage] for stage in bookkeeping) > max(
        STAGE_PRIORITY[stage] for stage in model_stages
    )
    assert set(STAGE_PRIORITY) == set(TaskStage)
    assert all(0 <= value <= 20 for value in STAGE_PRIORITY.values())


def test_a_bands_tasks_never_interleave_with_another_bands() -> None:
    """Every interactive task outranks every bulk task, whatever their stages."""

    assert task_priority(TaskStage.COMPACT_DOCUMENT, PriorityBand.INTERACTIVE) == 1018
    assert task_priority(TaskStage.COMPACT_DOCUMENT, PriorityBand.BULK) == 18
    assert min(task_priority(stage, PriorityBand.INTERACTIVE) for stage in TaskStage) > max(
        task_priority(stage, PriorityBand.BULK) for stage in TaskStage
    )


def _held_base(
    store: MemoryDistributedStore, *, run_id: str = "base-run"
) -> dict[str, dict[str, str]]:
    """A succeeded base run of three documents, with every output a revision reads.

    ``report`` sits in ``reports/2024``, ``archive`` in ``reports-old`` and
    ``sheet`` is a dataset. Returns each document's artifact and task ids.
    """

    definition = RunDefinition(
        id=run_id, graph_id="test-graph", config={}, config_digest="d", status=RunStatus.SUCCEEDED
    )
    documents = {
        "report": ("reports/2024/report.txt", "text/plain"),
        "archive": ("reports-old/archive.txt", "text/plain"),
        "sheet": ("data/sheet.csv", "text/csv"),
    }
    tasks: list[TaskDefinition] = []
    ids: dict[str, dict[str, str]] = {}
    for file_id, (location, mime_type) in documents.items():
        payload = f"{file_id} text".encode()
        store.put(
            run_id,
            ArtifactKind.SOURCE_DOCUMENT,
            payload,
            mime_type,
            metadata={
                "input_file": {
                    "id": file_id,
                    "checksum": sha256_hex(payload),
                    "location": location,
                    "mime_type": mime_type,
                    "filename": location.rpartition("/")[2],
                }
            },
            identity_key=file_id,
        )
        refs = {
            kind: store.put(
                run_id,
                kind,
                f"{kind.value}:{file_id}".encode(),
                "application/json",
                metadata={"file_ids": [file_id], "windows": 1},
                identity_key=file_id,
            ).id
            for kind in (
                ArtifactKind.PREPARED_DOCUMENT,
                ArtifactKind.DOCUMENT_CONTEXT,
                ArtifactKind.DOCUMENT_ENTITY_INVENTORY,
                ArtifactKind.EXTRACTED_DOCUMENT,
            )
        }
        context_task = f"{run_id}-context-{file_id}"
        inventory_task = f"{run_id}-inventory-{file_id}"
        windows = [{"window_id": f"w-{file_id}", "artifact_id": f"window-{file_id}"}]
        tasks.extend(
            [
                TaskDefinition(
                    id=context_task,
                    run_id=run_id,
                    stage=TaskStage.EXTRACT_DOCUMENT_CONTEXT,
                    scope_id=file_id,
                ),
                TaskDefinition(
                    id=inventory_task,
                    run_id=run_id,
                    stage=TaskStage.COMPACT_ENTITY_INVENTORY,
                    scope_id=file_id,
                    payload={
                        "prepared_artifact_id": refs[ArtifactKind.PREPARED_DOCUMENT],
                        "windows": windows,
                    },
                ),
            ]
        )
        store.task_outputs[context_task] = [refs[ArtifactKind.DOCUMENT_CONTEXT]]
        store.task_outputs[inventory_task] = [refs[ArtifactKind.DOCUMENT_ENTITY_INVENTORY]]
        ids[file_id] = {
            "prepared": refs[ArtifactKind.PREPARED_DOCUMENT],
            "context": refs[ArtifactKind.DOCUMENT_CONTEXT],
            "inventory": refs[ArtifactKind.DOCUMENT_ENTITY_INVENTORY],
            "extracted": refs[ArtifactKind.EXTRACTED_DOCUMENT],
        }
    tasks.append(
        TaskDefinition(
            id=f"{run_id}-final",
            run_id=run_id,
            stage=TaskStage.FINALIZE_GRAPH,
            scope_id="test-graph",
            payload={"output": {"provider": "local_artifacts"}},
        )
    )
    store.earlier_runs[run_id] = (definition, tasks)
    return ids


def _revise(
    tmp_path: Path,
    store: MemoryDistributedStore,
    revision: RevisionRequest,
    offered: list[InputFile] | None = None,
) -> tuple[list[TaskDefinition], TaskDefinition]:
    """Plan a revision afresh and return its document tasks and its finalizer."""

    # The fake holds one run being planned; each call plans a new one.
    store.run, store.tasks = None, []
    DistributedRunPlanner(
        _settings(tmp_path), _StaticFileSource(offered or []), store, store
    ).submit("revision-run", revision)
    (final,) = [task for task in store.tasks if task.stage == TaskStage.FINALIZE_GRAPH]
    return [task for task in store.tasks if task.stage != TaskStage.FINALIZE_GRAPH], final


def test_a_revision_reextracts_selected_documents_from_the_stage_it_names(
    tmp_path: Path,
) -> None:
    """Each selector picks kept documents; each stage reuses exactly what precedes it.

    Nothing is prepared again: every seed names the text the base stored, and
    the finalizer reads that text from the base while reading the new
    extraction from the revision.
    """

    store = MemoryDistributedStore()
    ids = _held_base(store)

    # A folder prefix matches whole folder names: reports-old is not in reports.
    tasks, final = _revise(
        tmp_path,
        store,
        RevisionRequest(
            base_run_id="base-run", reextract=["folder:reports"], reextract_from="relations"
        ),
    )
    (seed,) = tasks
    assert (seed.stage, seed.scope_id, seed.dependency_ids) == (
        TaskStage.COMPACT_ENTITY_INVENTORY,
        "report",
        [],
    )
    assert seed.payload == {
        "prepared_artifact_id": ids["report"]["prepared"],
        "inventory_artifact_id": ids["report"]["inventory"],
        "windows": [{"window_id": "w-report", "artifact_id": "window-report"}],
        "provenance": "reextracted:relations",
    }
    assert final.payload["inherit"] == [
        {"run_id": "base-run", "file_ids": ["archive", "sheet"], "prepared_file_ids": ["report"]}
    ]
    assert final.payload["reprocessed"] == {"reextracted:relations": ["report"]}
    assert document_provenance(final.payload) == {
        "archive": "reused",
        "sheet": "reused",
        "report": "reextracted:relations",
    }

    # A kind picks the dataset; revalidation takes every other kept document.
    tasks, final = _revise(
        tmp_path,
        store,
        RevisionRequest(
            base_run_id="base-run",
            reextract=["kind:dataset"],
            reextract_from="entities",
            revalidate=True,
        ),
    )
    by_scope = {task.scope_id: task for task in tasks}
    assert by_scope["sheet"].stage == TaskStage.EXTRACT_DOCUMENT_CONTEXT
    assert by_scope["sheet"].payload == {
        "prepared_artifact_id": ids["sheet"]["prepared"],
        "context_artifact_id": ids["sheet"]["context"],
        "provenance": "reextracted:entities",
    }
    for file_id in ("report", "archive"):
        assert by_scope[file_id].stage == TaskStage.REVALIDATE_DOCUMENT_CONTEXT
        assert by_scope[file_id].payload == {
            "prepared_artifact_id": ids[file_id]["prepared"],
            "extracted_artifact_id": ids[file_id]["extracted"],
            "windows": [{"window_id": f"w-{file_id}", "artifact_id": f"window-{file_id}"}],
            "provenance": "revalidated",
        }
    assert final.payload["inherit"] == [
        {"run_id": "base-run", "prepared_file_ids": ["archive", "report", "sheet"]}
    ]

    # From context, only the stored text is reused.
    tasks, _final = _revise(
        tmp_path, store, RevisionRequest(base_run_id="base-run", reextract=["all"])
    )
    assert {(task.stage, task.scope_id) for task in tasks} == {
        (TaskStage.EXTRACT_DOCUMENT_CONTEXT, file_id) for file_id in ids
    }
    assert all(set(task.payload) == {"prepared_artifact_id", "provenance"} for task in tasks)

    # A file with a held identity and new bytes replaces the document outright.
    replacement = _input_file(tmp_path, "report.txt", "new report text", file_id="report")
    tasks, final = _revise(
        tmp_path,
        store,
        RevisionRequest(base_run_id="base-run", reextract=["report", "archive"]),
        offered=[replacement],
    )
    assert sorted((task.stage.value, task.scope_id) for task in tasks) == [
        (TaskStage.EXTRACT_DOCUMENT_CONTEXT.value, "archive"),
        (TaskStage.PREPARE_DOCUMENT.value, "report"),
    ]
    assert final.payload["reprocessed"] == {"reextracted:context": ["archive"]}


def test_revalidation_is_refused_for_extractors_it_cannot_replay(tmp_path: Path) -> None:
    """Stored records replay through the LLM extractors; another has no call to replay."""

    store = MemoryDistributedStore()
    _held_base(store)
    settings = _settings(tmp_path)
    gliner = settings.model_copy(
        update={"extractors": settings.extractors.model_copy(update={"entity_provider": "gliner"})}
    )

    with pytest.raises(ValueError, match="extracts entities with gliner"):
        DistributedRunPlanner(gliner, _StaticFileSource([]), store, store).submit(
            "revision-run", RevisionRequest(base_run_id="base-run", revalidate=True)
        )
    assert store.run is None


def test_a_revision_refuses_selectors_that_pick_nothing_it_keeps(tmp_path: Path) -> None:
    """A selector that names a missing document or picks none is an error, not a no-op."""

    store = MemoryDistributedStore()
    _held_base(store)

    for selector, message in [
        ("file:missing", "not kept in the graph: missing"),
        ("folder:nowhere", "picks no kept document: folder:nowhere"),
        ("kind:image", "picks no kept document: kind:image"),
    ]:
        with pytest.raises(ValueError, match=message):
            _revise(tmp_path, store, RevisionRequest(base_run_id="base-run", reextract=[selector]))
    with pytest.raises(ValueError, match="document kind must be one of"):
        RevisionRequest(base_run_id="base-run", reextract=["kind:video"])
    # A dropped document is no longer kept, so it cannot be re-extracted.
    with pytest.raises(ValueError, match="not kept in the graph: report"):
        _revise(
            tmp_path,
            store,
            RevisionRequest(base_run_id="base-run", drop_file_ids=["report"], reextract=["report"]),
        )


def test_a_document_without_the_outputs_a_stage_reuses_is_refused_before_the_run(
    tmp_path: Path,
) -> None:
    """Planning checks every stored output a seed needs; a document with no windows stays."""

    store = MemoryDistributedStore()
    _held_base(store)
    run, tasks = store.earlier_runs["base-run"]
    store.earlier_runs["base-run"] = (
        run,
        [task for task in tasks if task.stage != TaskStage.EXTRACT_DOCUMENT_CONTEXT],
    )

    with pytest.raises(ValueError, match=r"report \(no stored document context\)"):
        _revise(
            tmp_path,
            store,
            RevisionRequest(
                base_run_id="base-run", reextract=["report"], reextract_from="entities"
            ),
        )
    assert store.run is None

    # A document whose text yielded no windows has nothing to validate again.
    for ref in store.list_run_artifacts("base-run", {ArtifactKind.EXTRACTED_DOCUMENT}):
        if ref.metadata["file_ids"] == ["sheet"]:
            ref.metadata["windows"] = 0
    tasks, final = _revise(
        tmp_path, store, RevisionRequest(base_run_id="base-run", revalidate=True)
    )
    assert {task.scope_id for task in tasks} == {"report", "archive"}
    assert final.payload["inherit"][0]["file_ids"] == ["sheet"]


def test_a_revision_of_a_revision_finds_each_output_in_the_run_that_holds_it(
    tmp_path: Path,
) -> None:
    """A revalidated document's inventory and windows come from its revalidation.

    The first revision revalidated ``report``: its text stays in the base,
    its extraction and entity inventory are the revision's. Re-extracting its
    relations in a second revision starts from those, and the finalizer reads
    each output from the run holding it.
    """

    store = MemoryDistributedStore()
    ids = _held_base(store)
    revalidated = {
        kind: store.put(
            "first-run",
            kind,
            f"revalidated {kind.value}".encode(),
            "application/json",
            metadata={"file_ids": ["report"], "windows": 1},
            identity_key="report",
        ).id
        for kind in (ArtifactKind.EXTRACTED_DOCUMENT, ArtifactKind.DOCUMENT_ENTITY_INVENTORY)
    }
    windows = [{"window_id": "w-report", "artifact_id": "window-report"}]
    store.earlier_runs["first-run"] = (
        RunDefinition(
            id="first-run",
            graph_id="test-graph",
            config={},
            config_digest="d",
            status=RunStatus.SUCCEEDED,
        ),
        [
            TaskDefinition(
                id="first-revalidate",
                run_id="first-run",
                stage=TaskStage.REVALIDATE_DOCUMENT,
                scope_id="report",
                payload={"windows": windows},
            ),
            TaskDefinition(
                id="first-final",
                run_id="first-run",
                stage=TaskStage.FINALIZE_GRAPH,
                scope_id="test-graph",
                payload={
                    "inherit": [
                        {
                            "run_id": "base-run",
                            "file_ids": ["archive", "sheet"],
                            "prepared_file_ids": ["report"],
                        }
                    ],
                    "reprocessed": {"revalidated": ["report"]},
                },
            ),
        ],
    )
    store.task_outputs["first-revalidate"] = [
        revalidated[ArtifactKind.EXTRACTED_DOCUMENT],
        revalidated[ArtifactKind.DOCUMENT_ENTITY_INVENTORY],
        ids["report"]["context"],
    ]

    tasks, final = _revise(
        tmp_path,
        store,
        RevisionRequest(base_run_id="first-run", reextract=["report"], reextract_from="relations"),
    )
    (seed,) = tasks
    assert seed.payload["prepared_artifact_id"] == ids["report"]["prepared"]
    assert (
        seed.payload["inventory_artifact_id"] == revalidated[ArtifactKind.DOCUMENT_ENTITY_INVENTORY]
    )
    assert seed.payload["windows"] == windows

    # Reused as it is, the revalidated extraction is read from the first
    # revision and its text from the base.
    _tasks, final = _revise(tmp_path, store, RevisionRequest(base_run_id="first-run", rebuild=True))
    assert final.payload["inherit"] == [
        {"run_id": "base-run", "file_ids": ["archive", "sheet"], "prepared_file_ids": ["report"]},
        {"run_id": "first-run", "file_ids": ["report"]},
    ]
    assert document_provenance(final.payload) == {
        "archive": "reused",
        "sheet": "reused",
        "report": "reused",
    }


def test_document_selectors_match_ids_whole_folders_and_kinds() -> None:
    """A bare value is a file id; ``all`` is every document."""

    assert document_selected("doc-1", "doc-1", "a", "document")
    assert not document_selected("doc-1", "doc-2", "a", "document")
    assert document_selected("folder:reports/", "d", "reports/2024", "document")
    assert not document_selected("folder:reports", "d", "reports-old", "document")
    assert document_selected("kind:image", "d", None, "image")
    assert document_selected("all", "d", None, None)
    with pytest.raises(ValueError, match="must name at least one document"):
        InheritedDocuments(run_id="r")


def test_seeded_context_stage_reads_the_stored_text_and_may_reuse_its_context(
    tmp_path: Path,
) -> None:
    """A re-extraction seed has no dependency: its payload names the earlier outputs."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        chunks=[chunk("public source", chunk_id="chunk-1", file_id="file-1")],
    )
    prepared_ref = store.put(
        "base-run", ArtifactKind.PREPARED_DOCUMENT, prepared.model_dump_json().encode(), "json"
    )
    context = DocumentContextShard(
        document_openings={},
        file_ids=["file-1"],
        document_context_entities=[
            EntityMention(
                id="subject",
                name="A Source Paper",
                type="PAPER",
                description="",
                source_chunk_id="chunk-1",
                quote="public source",
            )
        ],
        trace=[{"stage": "document_context_extraction", "window_id": "front"}],
    )
    context_ref = store.put(
        "base-run", ArtifactKind.DOCUMENT_CONTEXT, context.model_dump_json().encode(), "json"
    )

    for payload, calls in [
        ({"prepared_artifact_id": prepared_ref.id}, [["file-1"]]),
        ({"prepared_artifact_id": prepared_ref.id, "context_artifact_id": context_ref.id}, []),
    ]:
        store.completed.clear()
        store.lease = _lease(_task(TaskStage.EXTRACT_DOCUMENT_CONTEXT, payload=payload))
        pipeline = RecordingPipeline()

        result = _worker(
            settings, store, pipeline, TaskStage.EXTRACT_DOCUMENT_CONTEXT
        ).process_one()

        assert result.succeeded is True
        assert pipeline.context_file_ids == calls
        output = store.get(store.completed[0][2][0])
        assert output.ref.run_id == "run-test"
        follow_ups = store.completed[0][3]
        compact = next(t for t in follow_ups if t.stage == TaskStage.COMPACT_ENTITY_INVENTORY)
        assert compact.payload["prepared_artifact_id"] == prepared_ref.id
    reused = DocumentContextShard.model_validate_json(output.payload)
    assert reused.document_context_entities == context.document_context_entities
    assert reused.trace == context.trace


def test_seeded_inventory_stage_carries_the_stored_inventory_into_the_run(tmp_path: Path) -> None:
    """Re-extracting relations starts from the stored inventory and windows."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    inventory = DocumentEntityInventoryShard(
        file_ids=["file-1"],
        window_count=1,
        chunk_count=1,
        document_openings={"file-1": "A Source Paper"},
    )
    inventory_ref = store.put(
        "base-run",
        ArtifactKind.DOCUMENT_ENTITY_INVENTORY,
        inventory.model_dump_json().encode(),
        "json",
    )
    task = _task(
        TaskStage.COMPACT_ENTITY_INVENTORY,
        payload={
            "prepared_artifact_id": "base-prepared",
            "inventory_artifact_id": inventory_ref.id,
            "windows": [{"window_id": "window-1", "artifact_id": "base-window"}],
        },
    )
    store.lease = _lease(task)

    result = _worker(
        settings, store, RecordingPipeline(), TaskStage.COMPACT_ENTITY_INVENTORY
    ).process_one()

    assert result.succeeded is True
    output = store.get(store.completed[0][2][0])
    assert output.ref.run_id == "run-test"
    assert DocumentEntityInventoryShard.model_validate_json(output.payload) == inventory
    follow_ups = store.completed[0][3]
    relation = next(t for t in follow_ups if t.stage == TaskStage.EXTRACT_RELATION_WINDOW)
    compact = next(t for t in follow_ups if t.stage == TaskStage.COMPACT_DOCUMENT)
    assert relation.payload == {"window_artifact_id": "base-window"}
    assert compact.payload == {"prepared_artifact_id": "base-prepared"}


def test_a_revalidation_runs_window_by_window_and_a_retry_repeats_one_window(
    tmp_path: Path,
) -> None:
    """A long document is revalidated in short window tasks and combined as one pass would.

    The seed places the stored records by window and queues an entity task
    per window with the inventory behind them; the inventory queues a relation
    task per window with the document's last task behind them, which stores
    the revalidated document, its inventory and its context. One relation
    window's task is stopped after its verifier call, before it stores its
    result: only that task runs again, so no other window's model call is made
    twice, and the stored document is the one a single-task revalidation
    makes. The seed's payload reaches the last task, so a later revision
    still finds the windows it carried.
    """

    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "graph": settings.graph.model_copy(
                update={"max_chunks_per_llm_call": 1, "extraction_parallelism": 1}
            ),
            "ontology": settings.ontology.model_copy(
                update={"profile": _REVALIDATION_ONTOLOGY.model_dump(mode="json")}
            ),
        }
    )
    store = _worker_store(settings)
    texts = [
        "Alice Smith works at Acme Labs.",
        "Acme Labs trains Bob Jones.",
        "Bob Jones works at Acme Labs.",
    ]
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        chunks=[
            chunk(
                text,
                chunk_id=f"c{index}",
                file_id="file-1",
                document_id="file-1",
                chunk_index=index,
            )
            for index, text in enumerate(texts)
        ],
    )
    alice = _stored_mention("alice", "Alice Smith", "Person", "c0")
    acme = _stored_mention("acme", "Acme Labs", "Organization", "c0")
    bob = _stored_mention("bob", "Bob Jones", "Person", "c1")
    stored = ExtractionObservations(
        entities=[alice, acme, bob],
        relations=[
            _stored_relation("works", alice, acme, "WORKS_AT", texts[0], "c0"),
            _stored_relation("trains", acme, bob, "TRAINS", texts[1], "c1"),
        ],
        trace=[
            {
                "stage": "relation_extraction",
                "window_id": "old-window",
                "rejected_records": [
                    {
                        "kind": "relation",
                        "reason": "domain_or_range_violation",
                        "name": "WORKS_AT",
                        "source": "Bob Jones",
                        "source_type": "Person",
                        "target": "Acme Labs",
                        "target_type": "Organization",
                        "quote": texts[2],
                        "chunk_id": "c2",
                        "candidate": {
                            "description": "",
                            "confidence": 0.5,
                            "source_surface": "",
                            "target_surface": "",
                            "quantities": [],
                            "assertion": "reported",
                        },
                    }
                ],
            }
        ],
        chunk_count=3,
        window_count=1,
    )
    prepared_ref = store.put(
        "base-run", ArtifactKind.PREPARED_DOCUMENT, prepared.model_dump_json().encode(), "json"
    )
    extracted_ref = store.put(
        "base-run",
        ArtifactKind.EXTRACTED_DOCUMENT,
        ExtractedDocumentShard(
            prepared=PreparedDocumentShard(
                file_ids=["file-1"], files_seen=1, documents_processed=1
            ),
            observations=stored,
        )
        .model_dump_json()
        .encode(),
        "json",
        metadata={"file_ids": ["file-1"], "windows": 1},
    )
    windows = [{"window_id": "base-window", "artifact_id": "base-window-artifact"}]
    seed = _task(
        TaskStage.REVALIDATE_DOCUMENT_CONTEXT,
        payload={
            "prepared_artifact_id": prepared_ref.id,
            "extracted_artifact_id": extracted_ref.id,
            "windows": windows,
            "provenance": "revalidated",
        },
    )
    verifier = _CountingVerifier()
    pipeline = _RevalidatingPipeline(settings, verifier, stop_once_in="c2")

    ran, outputs = _run_to_completion(settings, store, pipeline, seed)

    assert [task.stage for task in ran] == [
        TaskStage.REVALIDATE_DOCUMENT_CONTEXT,
        *[TaskStage.REVALIDATE_ENTITY_WINDOW] * 3,
        TaskStage.COMPACT_ENTITY_INVENTORY,
        *[TaskStage.REVALIDATE_RELATION_WINDOW] * 3,
        TaskStage.REVALIDATE_DOCUMENT,
    ]
    assert len(store.failed) == 1
    single = _CountingVerifier()
    whole = revalidate_observations(
        prepared.chunks, stored, _RevalidatingPipeline(settings, single).request
    )
    # The failed window is asked twice; every other window once, as in one pass.
    assert verifier.calls == {**single.calls, "c2": single.calls["c2"] + 1}
    assert set(single.calls) == {"c0", "c1", "c2"}
    last = ran[-1]
    assert last.payload["windows"] == windows
    document_id, inventory_id, context_id = outputs[last.id]
    revalidated = ExtractedDocumentShard.model_validate_json(store.get(document_id).payload)
    assert revalidated.observations == whole
    assert {relation.relation_type for relation in whole.relations} == {"WORKS_AT", "TRAINS"}
    assert len(whole.relations) == 3
    inventory = DocumentEntityInventoryShard.model_validate_json(store.get(inventory_id).payload)
    assert inventory.entities == whole.entities
    assert inventory.document_openings.keys() == {"file-1"}
    context = DocumentContextShard.model_validate_json(store.get(context_id).payload)
    assert context.document_openings == inventory.document_openings


_REVALIDATION_ONTOLOGY = OntologyProfile(
    name="revalidation",
    description="People and the organisations they work at.",
    mode="closed",
    entity_types=[
        EntityTypeDefinition(name="Person", description="A person."),
        EntityTypeDefinition(name="Organization", description="An organisation."),
    ],
    relation_types=[
        RelationTypeDefinition(
            name="WORKS_AT",
            description="A person works at an organisation.",
            source_types=["Person"],
            target_types=["Organization"],
        ),
        RelationTypeDefinition(
            name="TRAINS",
            description="An organisation trains a person.",
            source_types=["Organization"],
            target_types=["Person"],
        ),
    ],
)


def _stored_mention(mention_id: str, name: str, type_: str, chunk_id: str) -> EntityMention:
    """A mention a base run accepted, quoted by its name."""

    return EntityMention(
        id=mention_id,
        name=name,
        type=type_,
        description="",
        source_chunk_id=chunk_id,
        quote=name,
        confidence=0.9,
    )


def _stored_relation(
    relation_id: str,
    source: EntityMention,
    target: EntityMention,
    relation_type: str,
    quote: str,
    chunk_id: str,
) -> RelationObservation:
    """A relation a base run accepted, on its evidence."""

    return RelationObservation(
        id=relation_id,
        source_entity_id=source.id,
        target_entity_id=target.id,
        source_surface=source.name,
        target_surface=target.name,
        relation_type=relation_type,
        description=f"{source.name} {relation_type} {target.name}.",
        source_chunk_id=chunk_id,
        quote=quote,
        confidence=0.9,
    )


class _CountingVerifier:
    """Support every relation and entity, counting calls by the window's first chunk."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def verify(
        self,
        window: ExtractionWindow,
        entities: list[EntityMention],
        relations: list[RelationObservation],
        ontology: OntologyProfile,
        **kwargs: Any,
    ) -> VerificationOutcome:
        del entities, ontology
        entities_to_verify = kwargs.get("entities_to_verify") or []
        key = window.chunks[0].id
        self.calls[key] = self.calls.get(key, 0) + 1
        return VerificationOutcome(
            decisions=[
                VerificationDecision(
                    relation_id=relation.id,
                    verdict="supported",
                    confidence=0.9,
                )
                for relation in relations
            ],
            entity_decisions=[
                EntityVerificationDecision(entity_id=entity.id, verdict="specific")
                for entity in entities_to_verify
            ],
            trace={"stage": "relation_verification", "window_id": window.id},
        )


class _NoModelCalls:
    """A provider no revalidation of this document should need to ask."""

    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities()

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        raise AssertionError(f"revalidation asked the model for {request.task_name}")


class _RevalidatingPipeline(RecordingPipeline):
    """Revalidate with the real phases over a scripted verifier and no model.

    ``stop_once_in`` names a chunk whose relation task is stopped once after
    its model calls and before its result is stored, as a worker losing its
    pod would be.
    """

    def __init__(
        self,
        settings: Settings,
        verifier: _CountingVerifier,
        stop_once_in: str | None = None,
    ) -> None:
        super().__init__()
        self.stop_once_in = stop_once_in
        self.request = RevalidationRequest(
            llm=_NoModelCalls(),
            extractors=lambda provider: (
                LlmEntityExtractor(provider),
                LlmRelationExtractor(provider),
            ),
            verifier=verifier,
            settings=settings.graph,
            ontology=_REVALIDATION_ONTOLOGY,
            model="fake",
            timeout_seconds=30,
        )

    def begin_revalidation(
        self,
        prepared: PreparedDocumentShard,
        observations: ExtractionObservations,
    ) -> tuple[DocumentRevalidationShard, list[RevalidationWindow]]:
        return begin_revalidation(prepared.file_ids, prepared.chunks, observations, self.request)

    def revalidate_entity_window(self, window: RevalidationWindowShard) -> EntityWindowShard:
        return revalidate_entity_window(window, self.request)

    def revalidate_relation_window(
        self,
        window: RevalidationWindowShard,
        inventory: DocumentEntityInventoryShard,
    ) -> RevalidatedRelationWindowShard:
        result = revalidate_relation_window(window, inventory, self.request)
        if self.stop_once_in in result.chunk_ids:
            self.stop_once_in = None
            raise RuntimeError("the worker was stopped")
        return result


def _run_to_completion(
    settings: Settings,
    store: MemoryDistributedStore,
    pipeline: RecordingPipeline,
    seed: TaskDefinition,
) -> tuple[list[TaskDefinition], dict[str, list[str]]]:
    """Run a seed and every task it fans out to, each once its dependencies are done.

    A task that fails is run again at once, as a retry would. Returns the
    tasks in the order they succeeded, and each task's outputs.
    """

    worker = DistributedWorker(settings, "worker-1", set(TaskStage), pipeline, store, store)
    pending = [seed]
    ran: list[TaskDefinition] = []
    outputs: dict[str, list[str]] = {}
    while pending:
        task = next(item for item in pending if set(item.dependency_ids) <= set(outputs))
        store.lease = _lease(task, {item: outputs[item] for item in task.dependency_ids})
        if not worker.process_one().succeeded:
            continue
        _task_id, _worker_id, output_ids, follow_ups, _barrier = store.completed[-1]
        pending.remove(task)
        pending.extend(follow_ups)
        ran.append(task)
        outputs[task.id] = output_ids
    return ran, outputs


def test_a_revision_finalizes_reprocessed_documents_with_their_provenance(
    tmp_path: Path,
) -> None:
    """Text kept for a reprocessed document is not read as its extraction.

    The base still holds ``redone``'s old extraction, but the revision names
    only its text; the revision's own extraction is the one finalized, and
    every document row says how the graph came by it.
    """

    settings = _settings(tmp_path)
    store = _worker_store(settings)

    def shard(run_id: str, file_id: str) -> None:
        item = ExtractedDocumentShard(
            prepared=PreparedDocumentShard(
                file_ids=[file_id],
                files_seen=1,
                documents_processed=1,
                document_rows=[{"id": f"{run_id}:{file_id}", "file_id": file_id}],
            ),
            observations=ExtractionObservations(chunk_count=0, window_count=0),
        )
        store.put(
            run_id,
            ArtifactKind.EXTRACTED_DOCUMENT,
            item.model_dump_json().encode(),
            "application/json",
            metadata={"file_ids": [file_id]},
            identity_key=file_id,
        )

    shard("base-run", "kept")
    shard("base-run", "redone")
    shard("run-test", "redone")
    shard("run-test", "added")
    store.lease = _lease(
        _task(
            TaskStage.FINALIZE_GRAPH,
            payload={
                "output": {"provider": "local_artifacts"},
                "inherit": [
                    {"run_id": "base-run", "file_ids": ["kept"], "prepared_file_ids": ["redone"]}
                ],
                "reprocessed": {"revalidated": ["redone"]},
            },
        )
    )
    pipeline = RecordingPipeline()

    result = _worker(settings, store, pipeline, TaskStage.FINALIZE_GRAPH).process_one()

    assert result.succeeded is True
    assert sorted(pipeline.finalized_file_ids[0]) == ["added", "kept", "redone"]
    assert {row["id"]: row["provenance"] for row in pipeline.finalized_document_rows} == {
        "run-test:added": "extracted",
        "base-run:kept": "reused",
        "run-test:redone": "revalidated",
    }


def test_document_rows_carry_provenance_with_extracted_as_the_default() -> None:
    """A labelled document takes its label; every other keeps its own or is extracted."""

    rows = annotate_documents(
        [
            {"file_id": "a"},
            {"file_id": "b", "provenance": "reused"},
            {"file_id": "c", "provenance": "extracted"},
        ],
        [],
        {"c": "reextracted:entities"},
    )

    assert [row["provenance"] for row in rows] == ["extracted", "reused", "reextracted:entities"]


def test_local_finalization_excludes_durable_quarantined_files(tmp_path: Path) -> None:
    """A successful old shard of a quarantined document never reaches final assembly."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    assert store.run is not None
    store.run = store.run.model_copy(update={"config": {"quarantined_file_ids": ["poison"]}})
    for file_id in ["good", "poison"]:
        item = ExtractedDocumentShard(
            prepared=PreparedDocumentShard(file_ids=[file_id], files_seen=1, documents_processed=1),
            observations=ExtractionObservations(chunk_count=0, window_count=0),
        )
        store.put(
            "run-test",
            ArtifactKind.EXTRACTED_DOCUMENT,
            item.model_dump_json().encode(),
            "application/json",
            metadata={"file_ids": [file_id]},
        )
    store.lease = _lease(_task(TaskStage.FINALIZE_GRAPH))
    pipeline = RecordingPipeline()
    result = _worker(settings, store, pipeline, TaskStage.FINALIZE_GRAPH).process_one()
    assert result.succeeded, result.error
    assert pipeline.finalized_file_ids == [["good"]]


def test_a_revision_keeps_the_folder_entity_its_kept_documents_were_extracted_with(
    tmp_path: Path,
) -> None:
    """A kept document's folder entity is stored with its document context.

    Only which documents the folder entity leaves out may change on a rebuild;
    another type, relation, or level reads the folders unlike the stored
    mentions did, unless every kept document's context is extracted again.
    """

    store = MemoryDistributedStore()
    _held_base(store)
    profile = {
        "name": "studies",
        "description": "Studies.",
        "entity_types": [{"name": "STUDY", "description": "A study."}],
        "folder_entity": {"entity_type": "STUDY", "level": 1},
    }
    base, tasks = store.earlier_runs["base-run"]
    store.earlier_runs["base-run"] = (
        base.model_copy(update={"config": {"ontology": {"profile": profile}}}),
        tasks,
    )

    def revise(folder_entity: dict[str, object] | None, revision: RevisionRequest) -> None:
        store.run, store.tasks = None, []
        settings = _settings(tmp_path)
        ontology = settings.ontology.model_copy(
            update={"profile": {**profile, "folder_entity": folder_entity}}
        )
        DistributedRunPlanner(
            settings.model_copy(update={"ontology": ontology}), _StaticFileSource([]), store, store
        ).submit("revision-run", revision)

    rebuild = RevisionRequest(base_run_id="base-run", rebuild=True)
    revise({"entity_type": "STUDY", "level": 1, "excluded_file_ids": ["report"]}, rebuild)
    with pytest.raises(ValueError, match="--reextract all --from context"):
        revise({"entity_type": "STUDY", "level": 2}, rebuild)
    with pytest.raises(ValueError, match="extracted them as STUDY at level 1"):
        revise(None, rebuild)
    revise(
        {"entity_type": "STUDY", "level": 2},
        RevisionRequest(base_run_id="base-run", reextract=["all"], reextract_from="context"),
    )
