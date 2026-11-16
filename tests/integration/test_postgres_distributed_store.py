"""Live PostgreSQL contract tests for leases, barriers, retries, and artifacts."""

from __future__ import annotations

import os
from collections import Counter, deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row

from kg_processor.adapters.distributed.local_blob import LocalBlobStore
from kg_processor.adapters.distributed.postgres import PostgresDistributedStore
from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.application.distributed_planner import DistributedRunPlanner
from kg_processor.application.distributed_worker import DistributedWorker
from kg_processor.config.settings import Settings
from kg_processor.domain.consumption import ConsumptionEvent, Locality, TokenUsage
from kg_processor.domain.distributed import (
    ArtifactKind,
    PriorityBand,
    PublicationLease,
    RevisionRequest,
    RunDefinition,
    RunSnapshot,
    RunStatus,
    TaskDefinition,
    TaskLease,
    TaskProgress,
    TaskStage,
    TaskStatus,
    task_priority,
)
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.domain.stages import ExtractedDocumentShard
from kg_processor.factories import build_distributed_store, build_pipeline

_POSTGRES_DSN = os.getenv("KG_TEST_POSTGRES_DSN")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        not _POSTGRES_DSN,
        reason="Set KG_TEST_POSTGRES_DSN to run live PostgreSQL coordination checks.",
    ),
]


@pytest.fixture
def isolated_postgres_dsn() -> Iterator[str]:
    """Create one private schema per test so queued work cannot leak between runs."""

    assert _POSTGRES_DSN is not None
    schema_name = f"flakegraph_test_{uuid4().hex}"
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
    try:
        yield make_conninfo(_POSTGRES_DSN, options=f"-c search_path={schema_name}")
    finally:
        with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name))
            )


def test_postgres_schema_initialization_is_safe_during_worker_startup_burst(
    isolated_postgres_dsn: str,
) -> None:
    """Every replica may initialize concurrently without PostgreSQL DDL races."""

    stores = [PostgresDistributedStore(isolated_postgres_dsn) for _ in range(12)]
    with ThreadPoolExecutor(max_workers=len(stores)) as executor:
        list(executor.map(lambda store: store.initialize(), stores))


def test_a_schema_at_another_version_is_refused_rather_than_stamped_current(
    isolated_postgres_dsn: str,
) -> None:
    """The statements describe one shape; a database at another must not be papered over."""

    _store(isolated_postgres_dsn)
    with psycopg.connect(isolated_postgres_dsn, autocommit=True) as connection:
        connection.execute("UPDATE flakegraph_schema_version SET version = 7")

    with pytest.raises(RuntimeError, match="version 7 is not this build's version"):
        PostgresDistributedStore(isolated_postgres_dsn).initialize()


def test_a_database_at_this_version_passes_unchanged(isolated_postgres_dsn: str) -> None:
    """Starting against a database already at this version touches nothing in it."""

    _store(isolated_postgres_dsn)
    with psycopg.connect(isolated_postgres_dsn, autocommit=True) as connection:
        stamped = connection.execute("SELECT * FROM flakegraph_schema_version").fetchone()

    PostgresDistributedStore(isolated_postgres_dsn).initialize()

    with psycopg.connect(isolated_postgres_dsn, autocommit=True) as connection:
        assert connection.execute("SELECT * FROM flakegraph_schema_version").fetchone() == stamped


def test_tables_without_a_version_row_are_refused_too(isolated_postgres_dsn: str) -> None:
    """Tables with no version row are an unversioned database, not an empty one."""

    _store(isolated_postgres_dsn)
    with psycopg.connect(isolated_postgres_dsn, autocommit=True) as connection:
        connection.execute("DROP TABLE flakegraph_schema_version")

    with pytest.raises(RuntimeError, match="version 0 is not this build's version"):
        PostgresDistributedStore(isolated_postgres_dsn).initialize()


def test_postgres_schema_indexes_dependency_completion_lookup(
    isolated_postgres_dsn: str,
) -> None:
    """Resolve a completed prerequisite's dependants without scanning the edge table."""

    _store(isolated_postgres_dsn)
    with psycopg.connect(isolated_postgres_dsn, row_factory=dict_row) as connection:
        row = connection.execute(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND indexname = 'flakegraph_task_dependency_prerequisite_idx'
            """
        ).fetchone()

    assert row is not None
    assert "(depends_on_task_id, task_id)" in row["indexdef"]


def test_postgres_schema_indexes_the_ordered_queued_claim_path(
    isolated_postgres_dsn: str,
) -> None:
    """Avoid sorting a corpus-sized ready queue for every worker claim."""

    _store(isolated_postgres_dsn)
    with psycopg.connect(isolated_postgres_dsn, row_factory=dict_row) as connection:
        row = connection.execute(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND indexname = 'flakegraph_task_queued_claim_idx'
            """
        ).fetchone()

    assert row is not None
    assert (
        "(run_id, stage, remaining_dependencies, priority DESC, created_at, id)" in row["indexdef"]
    )
    assert "INCLUDE (available_at)" in row["indexdef"]
    assert "WHERE (status = 'queued'::text)" in row["indexdef"]


def test_postgres_preserves_distinct_source_identities_with_identical_bytes(
    isolated_postgres_dsn: str,
) -> None:
    """Content equality must not collapse separately identified source documents."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))

    first = store.put(
        run_id,
        ArtifactKind.SOURCE_DOCUMENT,
        b"same public document bytes",
        "text/plain",
        identity_key="source-a",
    )
    second = store.put(
        run_id,
        ArtifactKind.SOURCE_DOCUMENT,
        b"same public document bytes",
        "text/plain",
        identity_key="source-b",
    )

    assert first.id != second.id
    assert store.get(first.id).payload == store.get(second.id).payload


def test_a_runs_artifacts_are_the_ones_a_succeeded_task_linked(
    isolated_postgres_dsn: str,
) -> None:
    """An attempt that wrote its output and then lost its lease leaves an orphan.

    The retry's output hashes differently, so both objects exist under the
    run. Finalization must see the linked one alone.
    """

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    store.add_tasks(run_id, [_task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file-1")])
    store.activate_run(run_id)

    orphan = store.put(run_id, ArtifactKind.PREPARED_DOCUMENT, b"attempt 1", "application/json")
    claim = store.claim_task("worker", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert claim is not None
    linked = store.put(run_id, ArtifactKind.PREPARED_DOCUMENT, b"attempt 2", "application/json")
    store.complete_task(claim.task.id, claim.worker_id, [linked.id], claim_token=claim.claim_token)

    assert orphan.id != linked.id
    assert store.get_run_artifact_ids(run_id, {ArtifactKind.PREPARED_DOCUMENT}) == [linked.id]
    assert [
        item.ref.id for item in store.get_run_artifacts(run_id, {ArtifactKind.PREPARED_DOCUMENT})
    ] == [linked.id]


def test_run_summary_counts_the_documents_a_preparation_recorded_as_unreadable(
    isolated_postgres_dsn: str,
) -> None:
    """Their tasks succeed, so only the tag on the linked output says they failed.

    An orphan from an attempt that lost its lease is not counted: it is not
    what the run went on with.
    """

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    store.add_tasks(
        run_id,
        [
            _task(run_id, "prepare-broken", TaskStage.PREPARE_DOCUMENT, "file-broken"),
            _task(run_id, "prepare-fine", TaskStage.PREPARE_DOCUMENT, "file-fine"),
        ],
    )
    store.activate_run(run_id)
    failure = {
        "failed_documents": [
            {"file_id": "file-broken", "error_type": "RuntimeError", "error_message": "400"}
        ]
    }

    store.put(run_id, ArtifactKind.EXTRACTED_DOCUMENT, b"orphan", "application/json", failure)
    for _ in range(2):
        claim = store.claim_task("worker", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
        assert claim is not None
        broken = claim.task.scope_id == "file-broken"
        extracted = store.put(
            run_id,
            ArtifactKind.EXTRACTED_DOCUMENT,
            claim.task.scope_id.encode(),
            "application/json",
            failure if broken else {"file_ids": [claim.task.scope_id]},
        )
        store.complete_task(
            claim.task.id, claim.worker_id, [extracted.id], claim_token=claim.claim_token
        )

    assert store.get_run_summary(run_id).documents_failed == 1


def test_postgres_rejects_unknown_artifact_owner_before_external_upload(
    isolated_postgres_dsn: str,
    tmp_path: Path,
) -> None:
    """The run-identity cache must not weaken the external-object ownership check."""

    blob_store = LocalBlobStore(tmp_path.as_uri())
    store = PostgresDistributedStore(isolated_postgres_dsn, blob_store=blob_store)
    store.initialize()

    with pytest.raises(KeyError, match="unknown distributed run"):
        store.put(
            "missing-run",
            ArtifactKind.SOURCE_DOCUMENT,
            b"payload",
            "application/octet-stream",
        )

    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_postgres_store_enforces_bounded_session_waits(
    isolated_postgres_dsn: str,
) -> None:
    """Keep abandoned or lock-blocked worker transactions from pinning the queue."""

    store = _store(isolated_postgres_dsn)
    with store._connection() as connection:  # noqa: SLF001 - adapter session contract.
        settings = connection.execute(
            """
            SELECT current_setting('lock_timeout') AS lock_timeout,
                   current_setting('idle_in_transaction_session_timeout') AS idle_timeout,
                   current_setting('client_connection_check_interval') AS client_check
            """
        ).fetchone()

    assert settings == {
        "lock_timeout": "30s",
        "idle_timeout": "1min",
        "client_check": "1s",
    }


def test_postgres_streams_initial_plan_without_materializing_a_general_dag(
    isolated_postgres_dsn: str,
) -> None:
    """Bulk-load preparation tasks and one run-wide barrier from an iterator."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))

    def tasks() -> Iterator[TaskDefinition]:
        """Yield the restricted corpus plan accepted by the streaming API."""

        for index in range(3):
            yield _task(
                run_id,
                f"prepare-{index}",
                TaskStage.PREPARE_DOCUMENT,
                f"file-{index}",
            )
        yield _task(run_id, "final", TaskStage.FINALIZE_GRAPH, "graph")

    store.add_initial_tasks(run_id, tasks())
    store.activate_run(run_id)

    summary = store.get_run_summary(run_id)
    assert summary.total_tasks == 4
    assert {(row.stage, row.status, row.count) for row in summary.task_counts} == {
        (TaskStage.PREPARE_DOCUMENT, TaskStatus.QUEUED, 3),
        (TaskStage.FINALIZE_GRAPH, TaskStatus.QUEUED, 1),
    }


def test_postgres_summary_reports_latest_task_activity_without_hot_run_writes(
    isolated_postgres_dsn: str,
) -> None:
    """Expose worker liveness from task heartbeats while leaving the run row cold."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    task = _task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file")
    store.add_tasks(run_id, [task])
    store.activate_run(run_id)
    run_row_updated_at = store.get_run(run_id).updated_at

    claim = store.claim_task(
        "prepare-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
    )
    assert claim is not None

    summary = store.get_run_summary(run_id)
    recent = next(item for item in store.list_runs() if item.id == run_id)
    assert summary.updated_at > run_row_updated_at
    assert recent.updated_at == summary.updated_at


def test_postgres_claims_are_exclusive_and_dependencies_form_a_barrier(
    isolated_postgres_dsn: str,
) -> None:
    """Competing workers must get unique tasks and finalization must wait for both."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    prepare_tasks = [
        _task(run_id, f"prepare-{index}", TaskStage.PREPARE_DOCUMENT, f"file-{index}")
        for index in range(2)
    ]
    final = _task(
        run_id,
        "final",
        TaskStage.FINALIZE_GRAPH,
        "graph",
        dependency_ids=[task.id for task in prepare_tasks],
    )
    store.add_tasks(run_id, [*prepare_tasks, final])
    store.activate_run(run_id)

    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = list(
            executor.map(
                lambda index: store.claim_task(
                    f"worker-{index}",
                    {TaskStage.PREPARE_DOCUMENT},
                    timedelta(minutes=1),
                ),
                range(8),
            )
        )
    claimed = [claim for claim in claims if claim is not None]
    assert len(claimed) == 2
    assert len({claim.task.id for claim in claimed}) == 2
    assert (
        store.claim_task(
            "final-worker",
            {TaskStage.FINALIZE_GRAPH},
            timedelta(minutes=1),
        )
        is None
    )

    artifact_ids: list[str] = []
    for claim in claimed:
        payload = f"output:{claim.task.id}".encode()
        ref = store.put(
            run_id,
            ArtifactKind.PREPARED_DOCUMENT,
            payload,
            "application/json",
            metadata={"task_id": claim.task.id},
        )
        assert store.get(ref.id).payload == payload
        artifact_ids.append(ref.id)
        store.complete_task(claim.task.id, claim.worker_id, [ref.id], claim_token=claim.claim_token)
    batched = store.get_many(list(reversed(artifact_ids)))
    assert [artifact.ref.id for artifact in batched] == list(reversed(artifact_ids))

    final_claim = store.claim_task(
        "final-worker",
        {TaskStage.FINALIZE_GRAPH},
        timedelta(minutes=1),
    )
    assert final_claim is not None
    assert set(final_claim.dependency_outputs) == {task.id for task in prepare_tasks}
    assert {
        artifact_id
        for outputs in final_claim.dependency_outputs.values()
        for artifact_id in outputs
    } == set(artifact_ids)
    store.complete_task(
        final_claim.task.id, final_claim.worker_id, [], claim_token=final_claim.claim_token
    )

    snapshot = store.get_run(run_id)
    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert all(task.status == TaskStatus.SUCCEEDED for task in snapshot.tasks)
    summary = store.get_run_summary(run_id)
    assert summary.run.status == RunStatus.SUCCEEDED
    assert summary.total_tasks == 3
    assert sum(item.count for item in summary.task_counts) == 3
    assert {item.status for item in summary.task_counts} == {TaskStatus.SUCCEEDED}


def test_running_task_progress_is_lease_owned_and_bounded(
    isolated_postgres_dsn: str,
) -> None:
    """Persist one finalizer phase in detailed, summary, and recent-run status."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    final = _task(run_id, "final", TaskStage.FINALIZE_GRAPH, "graph")
    store.add_tasks(run_id, [final])
    store.activate_run(run_id)
    claim = store.claim_task(
        "finalizer-1",
        {TaskStage.FINALIZE_GRAPH},
        timedelta(minutes=1),
    )
    assert claim is not None
    progress = TaskProgress(
        phase="write_tables",
        phase_index=10,
        phase_total=12,
        completed=4,
        total=12,
        message="Writing graph tables",
    )

    with pytest.raises(RuntimeError, match="no longer owns"):
        store.report_task_progress(
            final.id, "another-worker", progress, claim_token=claim.claim_token
        )
    store.report_task_progress(final.id, claim.worker_id, progress, claim_token=claim.claim_token)

    detailed = store.get_run(run_id)
    summary = store.get_run_summary(run_id)
    recent = next(item for item in store.list_runs() if item.id == run_id)
    assert detailed.tasks[0].progress == progress
    assert summary.task_counts[0].progress == progress
    assert recent.task_counts[0].progress == progress


def test_worker_demand_view_tracks_ready_work_and_active_leases(
    isolated_postgres_dsn: str,
) -> None:
    """Keep autoscaling capacity until leases finish and hide blocked barriers."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    prepare_tasks = [
        _task(run_id, f"prepare-{index}", TaskStage.PREPARE_DOCUMENT, f"file-{index}")
        for index in range(2)
    ]
    final = _task(
        run_id,
        "final",
        TaskStage.FINALIZE_GRAPH,
        "graph",
        dependency_ids=[task.id for task in prepare_tasks],
    )
    store.add_tasks(run_id, [*prepare_tasks, final])
    store.activate_run(run_id)

    assert _worker_demand(isolated_postgres_dsn) == {"prepare_document": 2}

    first = store.claim_task("prepare-1", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert first is not None
    assert _worker_demand(isolated_postgres_dsn) == {"prepare_document": 2}
    store.complete_task(first.task.id, first.worker_id, [], claim_token=first.claim_token)
    assert _worker_demand(isolated_postgres_dsn) == {"prepare_document": 1}

    second = store.claim_task("prepare-2", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert second is not None
    store.complete_task(second.task.id, second.worker_id, [], claim_token=second.claim_token)
    assert _worker_demand(isolated_postgres_dsn) == {"finalize_graph": 1}

    final_claim = store.claim_task(
        "finalize-1",
        {TaskStage.FINALIZE_GRAPH},
        timedelta(minutes=1),
    )
    assert final_claim is not None
    assert _worker_demand(isolated_postgres_dsn) == {"finalize_graph": 1}
    store.complete_task(
        final_claim.task.id, final_claim.worker_id, [], claim_token=final_claim.claim_token
    )
    assert _worker_demand(isolated_postgres_dsn) == {}


def test_publication_keeps_finalize_demand_and_a_retry_reactivates_it(
    isolated_postgres_dsn: str,
) -> None:
    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    final = _task(
        run_id,
        "final",
        TaskStage.FINALIZE_GRAPH,
        "graph",
        max_attempts=1,
    ).model_copy(update={"payload": {"output": {"provider": "snowflake_bulk"}}})
    store.add_tasks(run_id, [final])
    store.activate_run(run_id)
    task = store.claim_task("finalizer", {TaskStage.FINALIZE_GRAPH}, timedelta(minutes=1))
    assert task is not None
    manifest = store.put(
        run_id,
        ArtifactKind.GRAPH_RESULT,
        b'{"format":"flakegraph.graph-dataset-manifest/v1"}',
        "application/vnd.flakegraph.graph-manifest+json",
    )
    store.complete_task(task.task.id, task.worker_id, [manifest.id], claim_token=task.claim_token)
    # A finalizer whose graph still awaits publication keeps the run running
    # and keeps asking for a finalizer, before and after the claim.
    assert store.get_run(run_id).run.status == RunStatus.RUNNING
    assert _worker_demand(isolated_postgres_dsn) == {"finalize_graph": 1}
    publication = store.claim_publication("publisher", timedelta(minutes=1))
    assert publication is not None
    assert _worker_demand(isolated_postgres_dsn) == {"finalize_graph": 1}
    store.fail_publication(
        publication.id,
        publication.worker_id,
        {"message": "expired destination credential"},
        timedelta(0),
    )
    assert store.get_run(run_id).run.status == RunStatus.FAILED

    store.retry_run(run_id)

    assert store.get_run(run_id).run.status == RunStatus.RUNNING
    assert _worker_demand(isolated_postgres_dsn) == {"finalize_graph": 1}
    assert store.claim_publication("publisher-2", timedelta(minutes=1)) is not None


def test_postgres_claims_filter_runs_by_processing_config(
    isolated_postgres_dsn: str,
) -> None:
    """Workers must ignore tasks produced by an incompatible runtime configuration."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    task = _task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file")
    store.add_tasks(run_id, [task])
    store.activate_run(run_id)

    assert (
        store.claim_task(
            "incompatible-worker",
            {TaskStage.PREPARE_DOCUMENT},
            timedelta(minutes=1),
            expected_config_digest="different-config",
        )
        is None
    )
    claim = store.claim_task(
        "compatible-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
        expected_config_digest="integration-config",
    )
    assert claim is not None
    assert claim.task.id == task.id


def test_postgres_retries_then_fails_run_after_attempt_budget(
    isolated_postgres_dsn: str,
) -> None:
    """A task remains claimable until its configured attempt budget is exhausted."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    task = _task(
        run_id,
        "prepare",
        TaskStage.PREPARE_DOCUMENT,
        "file",
        max_attempts=2,
    )
    store.add_tasks(run_id, [task])
    store.activate_run(run_id)

    first = store.claim_task("worker-1", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert first is not None and first.attempt == 1
    store.fail_task(
        first.task.id,
        first.worker_id,
        {"message": "first"},
        timedelta(0),
        claim_token=first.claim_token,
    )

    second = store.claim_task("worker-2", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert second is not None and second.attempt == 2
    store.fail_task(
        second.task.id,
        second.worker_id,
        {"message": "second"},
        timedelta(0),
        claim_token=second.claim_token,
    )

    snapshot = store.get_run(run_id)
    assert snapshot.run.status == RunStatus.FAILED
    assert snapshot.tasks[0].status == TaskStatus.FAILED
    assert snapshot.tasks[0].attempts == 2
    assert snapshot.tasks[0].last_error == {"message": "second"}
    summary = store.get_run_summary(run_id)
    assert summary.error == {"task_id": task.id, "message": "second"}

    store.retry_run(run_id)
    retried = store.get_run(run_id)
    retried_summary = store.get_run_summary(run_id)
    assert retried.run.status == RunStatus.QUEUED
    assert retried.tasks[0].status == TaskStatus.QUEUED
    assert retried.tasks[0].attempts == 0
    assert retried.tasks[0].last_error is None
    assert retried_summary.error is None
    assert retried_summary.task_counts[0].started_at is None
    assert retried_summary.task_counts[0].completed_at is None
    assert retried_summary.task_counts[0].progress is None
    # The failure the retry cleared is kept as the run's history.
    assert retried_summary.resolved_failure_count == 1
    (resolved,) = retried_summary.resolved_failures
    assert (resolved.task_id, resolved.stage, resolved.scope_id) == (
        task.id,
        TaskStage.PREPARE_DOCUMENT.value,
        "file",
    )
    assert resolved.attempts == 2
    assert resolved.error == {"message": "second"}
    assert resolved.failed_at is not None
    assert resolved.retried_at >= resolved.failed_at

    # The retried task succeeds, and the history outlives the success.
    claim = store.claim_task("worker-3", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert claim is not None
    store.complete_task(claim.task.id, claim.worker_id, [], claim_token=claim.claim_token)
    assert store.get_run_summary(run_id).resolved_failure_count == 1


def test_a_cancelled_run_resumes_where_it_stopped(isolated_postgres_dsn: str) -> None:
    """Retrying a cancelled run requeues only what the cancellation stopped."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    prepare = _task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file")
    extract = _task(
        run_id,
        "extract",
        TaskStage.EXTRACT_DOCUMENT_CONTEXT,
        "file",
        dependency_ids=[prepare.id],
    )
    store.add_tasks(run_id, [prepare, extract])
    store.activate_run(run_id)
    prepared = store.claim_task("worker-1", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert prepared is not None
    store.complete_task(prepared.task.id, prepared.worker_id, [], claim_token=prepared.claim_token)
    running = store.claim_task(
        "worker-2", {TaskStage.EXTRACT_DOCUMENT_CONTEXT}, timedelta(minutes=1)
    )
    assert running is not None

    store.cancel_run(run_id)
    cancelled = store.get_run(run_id)
    assert cancelled.run.status == RunStatus.CANCELLED
    assert {task.task.id: task.status for task in cancelled.tasks} == {
        prepare.id: TaskStatus.SUCCEEDED,
        extract.id: TaskStatus.CANCELLED,
    }

    store.retry_run(run_id)

    resumed = store.get_run(run_id)
    assert resumed.run.status == RunStatus.QUEUED
    assert {task.task.id: task.status for task in resumed.tasks} == {
        prepare.id: TaskStatus.SUCCEEDED,
        extract.id: TaskStatus.QUEUED,
    }
    # The prerequisite stays done, so the resumed task is claimable at once.
    reclaimed = store.claim_task(
        "worker-3", {TaskStage.EXTRACT_DOCUMENT_CONTEXT}, timedelta(minutes=1)
    )
    assert reclaimed is not None and reclaimed.task.id == extract.id and reclaimed.attempt == 1

    with pytest.raises(ValueError, match="failed or cancelled"):
        store.retry_run(run_id)
    # A cancellation is not a failure: nothing is kept for resuming one.
    assert store.get_run_summary(run_id).resolved_failures == []


def test_postgres_bounds_abandoned_reclaims_and_fails_after_recovery_window(
    isolated_postgres_dsn: str,
) -> None:
    """Consume retry budget on reclaim and terminally fail a repeatedly lost task.

    Claiming does not sweep: the exhausted task stays running, unclaimable,
    until a worker's periodic sweep fails it and its run.
    """

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    task = _task(
        run_id,
        "prepare",
        TaskStage.PREPARE_DOCUMENT,
        "file",
        max_attempts=2,
    )
    store.add_tasks(run_id, [task])
    store.activate_run(run_id)
    lease = store.claim_task(
        "abandoned-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
    )
    assert lease is not None

    with psycopg.connect(isolated_postgres_dsn) as connection:
        connection.execute(
            "UPDATE flakegraph_task SET lease_expires_at = "
            "CURRENT_TIMESTAMP - INTERVAL '1 minute', "
            "updated_at = CURRENT_TIMESTAMP - INTERVAL '1 minute' "
            "WHERE id = %s",
            (task.id,),
        )

    reclaimed = store.claim_task(
        "recovery-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
    )
    assert reclaimed is not None
    assert reclaimed.attempt == 2
    snapshot = store.get_run(run_id)
    assert snapshot.run.status == RunStatus.RUNNING
    assert snapshot.tasks[0].status == TaskStatus.RUNNING
    assert snapshot.tasks[0].lease_owner == "recovery-worker"
    with pytest.raises(RuntimeError, match="no longer owns"):
        store.complete_task(task.id, "abandoned-worker", [], claim_token="stale-claim")

    with psycopg.connect(isolated_postgres_dsn) as connection:
        connection.execute(
            "UPDATE flakegraph_task SET lease_expires_at = "
            "CURRENT_TIMESTAMP - INTERVAL '10 minutes', "
            "updated_at = CURRENT_TIMESTAMP - INTERVAL '10 minutes' "
            "WHERE id = %s",
            (task.id,),
        )

    assert (
        store.claim_task(
            "third-worker",
            {TaskStage.PREPARE_DOCUMENT},
            timedelta(minutes=1),
        )
        is None
    )
    unswept = store.get_run(run_id)
    assert unswept.run.status == RunStatus.RUNNING
    assert unswept.tasks[0].status == TaskStatus.RUNNING

    store.fail_abandoned_tasks()

    failed = store.get_run(run_id)
    assert failed.run.status == RunStatus.FAILED
    assert failed.tasks[0].status == TaskStatus.FAILED
    assert failed.tasks[0].attempts == 2
    assert failed.tasks[0].last_error == {
        "type": "LeaseExpired",
        "message": "worker lease expired after retry budget was exhausted",
    }


def test_postgres_final_attempt_can_complete_during_expiry_recovery_window(
    isolated_postgres_dsn: str,
) -> None:
    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    task = _task(
        run_id,
        "prepare",
        TaskStage.PREPARE_DOCUMENT,
        "file",
        max_attempts=1,
    )
    store.add_tasks(run_id, [task])
    store.activate_run(run_id)
    lease = store.claim_task("slow-worker", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert lease is not None
    with psycopg.connect(isolated_postgres_dsn) as connection:
        connection.execute(
            "UPDATE flakegraph_task SET lease_expires_at = "
            "CURRENT_TIMESTAMP - INTERVAL '1 minute', "
            "updated_at = CURRENT_TIMESTAMP - INTERVAL '1 minute' WHERE id = %s",
            (task.id,),
        )

    # A sweep inside the recovery window leaves the slow worker its last attempt.
    store.fail_abandoned_tasks()
    assert (
        store.claim_task(
            "recovery-worker",
            {TaskStage.PREPARE_DOCUMENT},
            timedelta(minutes=1),
        )
        is None
    )
    store.complete_task(task.id, "slow-worker", [], claim_token=lease.claim_token)

    assert store.get_run(run_id).tasks[0].status == TaskStatus.SUCCEEDED


def test_postgres_reclaims_an_expired_lease_without_losing_priority(
    isolated_postgres_dsn: str,
) -> None:
    """Compare queued and expired candidates while retaining the stage ladder."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    stages = {TaskStage.COMPACT_DOCUMENT, TaskStage.EXTRACT_RELATION_WINDOW}
    high = _task(run_id, "high", TaskStage.COMPACT_DOCUMENT, "high")
    low = _task(run_id, "low", TaskStage.EXTRACT_RELATION_WINDOW, "low")
    store.add_tasks(run_id, [low, high])
    store.activate_run(run_id)

    first = store.claim_task("first-worker", stages, timedelta(minutes=1))
    assert first is not None and first.task.id == high.id and first.attempt == 1
    with psycopg.connect(isolated_postgres_dsn) as connection:
        connection.execute(
            "UPDATE flakegraph_task SET lease_expires_at = CURRENT_TIMESTAMP - INTERVAL '1 minute' "
            "WHERE id = %s",
            (high.id,),
        )

    reclaimed = store.claim_task("recovery-worker", stages, timedelta(minutes=1))
    assert reclaimed is not None
    assert reclaimed.task.id == high.id
    assert reclaimed.attempt == 2


def test_a_run_shares_a_pool_equally_with_one_ten_times_its_size(
    isolated_postgres_dsn: str,
) -> None:
    """While both have ready work, each run holds half of a pool of four workers.

    The large run was submitted first, so every one of its tasks is older than
    the small run's: claimed by age alone, the small run would wait for all
    fifty of them.
    """

    store = _store(isolated_postgres_dsn)
    large, small = (f"run_{uuid4().hex}" for _ in range(2))
    for run_id, size in ((large, 50), (small, 5)):
        store.create_run(_run(run_id))
        store.add_tasks(
            run_id,
            [
                _task(run_id, f"prepare-{index:02d}", TaskStage.PREPARE_DOCUMENT, f"file-{index}")
                for index in range(size)
            ],
        )
        store.activate_run(run_id)
    worker = iter(range(1_000))

    def claim() -> TaskLease:
        lease = store.claim_task(
            f"worker-{next(worker)}", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1)
        )
        assert lease is not None
        return lease

    pool = deque(claim() for _ in range(4))
    assert [lease.task.run_id for lease in pool] == [large, small, large, small]
    served = Counter(lease.task.run_id for lease in pool)
    while served[small] < 5:
        finished = pool.popleft()
        store.complete_task(
            finished.task.id, finished.worker_id, [], claim_token=finished.claim_token
        )
        pool.append(claim())
        served[pool[-1].task.run_id] += 1
        assert Counter(lease.task.run_id for lease in pool) == {large: 2, small: 2}
    assert served == {large: 5, small: 5}

    # With the small run out of ready work, the large one takes the pool.
    finished = pool.popleft()
    store.complete_task(finished.task.id, finished.worker_id, [], claim_token=finished.claim_token)
    assert claim().task.run_id == large


def test_an_interactive_run_is_claimed_ahead_of_every_bulk_run(
    isolated_postgres_dsn: str,
) -> None:
    """Band comes before share: the interactive run's work goes first, all of it."""

    store = _store(isolated_postgres_dsn)
    routine, urgent = (f"run_{uuid4().hex}" for _ in range(2))
    for run_id, band in ((routine, PriorityBand.BULK), (urgent, PriorityBand.INTERACTIVE)):
        store.create_run(_run(run_id, band))
        store.add_tasks(
            run_id,
            [
                _task(run_id, f"prepare-{index}", TaskStage.PREPARE_DOCUMENT, f"file-{index}")
                for index in range(3)
            ],
        )
        store.activate_run(run_id)

    claims = [
        store.claim_task(f"worker-{index}", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
        for index in range(4)
    ]

    assert [claim.task.run_id for claim in claims if claim is not None] == [
        urgent,
        urgent,
        urgent,
        routine,
    ]
    assert set(_task_priorities(isolated_postgres_dsn, urgent).values()) == {1020}
    assert set(_task_priorities(isolated_postgres_dsn, routine).values()) == {20}


def test_work_discovered_by_a_worker_is_claimed_in_its_runs_band(
    isolated_postgres_dsn: str,
) -> None:
    """A follow-up task takes its run's band, not the band its parent was claimed from."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id, PriorityBand.INTERACTIVE))
    prepare = _task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file")
    final = _task(run_id, "final", TaskStage.FINALIZE_GRAPH, "graph")
    store.add_tasks(run_id, [prepare, final])
    store.activate_run(run_id)
    parent = store.claim_task("worker", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert parent is not None
    context = _task(
        run_id, "context", TaskStage.EXTRACT_DOCUMENT_CONTEXT, "file", dependency_ids=[prepare.id]
    )

    store.complete_task(
        parent.task.id, parent.worker_id, [], [context], final.id, claim_token=parent.claim_token
    )

    assert _task_priorities(isolated_postgres_dsn, run_id) == {
        prepare.id: 1020,
        context.id: 1015,
        final.id: 1000,
    }
    assert _banded_worker_demand(isolated_postgres_dsn) == {
        ("extract_document_context", "interactive"): 1
    }


def test_concurrent_claims_on_one_run_all_find_work(isolated_postgres_dsn: str) -> None:
    """Workers claiming the same run at once each get a task, not a missed poll."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    store.add_tasks(
        run_id,
        [
            _task(run_id, f"prepare-{index}", TaskStage.PREPARE_DOCUMENT, f"file-{index}")
            for index in range(40)
        ],
    )
    store.activate_run(run_id)

    with ThreadPoolExecutor(max_workers=12) as executor:
        claims = list(
            executor.map(
                lambda index: store.claim_task(
                    f"worker-{index}", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1)
                ),
                range(12),
            )
        )

    assert all(claim is not None for claim in claims)
    assert len({claim.task.id for claim in claims if claim is not None}) == 12


def test_a_run_id_cannot_be_reused_at_another_band(isolated_postgres_dsn: str) -> None:
    """The band is part of what a run is; resubmitting at another is a different run."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))

    with pytest.raises(ValueError, match="different inputs"):
        store.create_run(_run(run_id, PriorityBand.INTERACTIVE))


def test_postgres_dynamic_windows_extend_final_barrier_atomically(
    isolated_postgres_dsn: str,
) -> None:
    """Finalization must wait for the document task that joins every window."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    prepare = _task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file")
    final = _task(
        run_id,
        "final",
        TaskStage.FINALIZE_GRAPH,
        "graph",
    )
    store.add_tasks(run_id, [prepare, final])
    store.activate_run(run_id)

    parent = store.claim_task(
        "prepare-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
    )
    assert parent is not None
    context = _task(
        run_id,
        "context",
        TaskStage.EXTRACT_DOCUMENT_CONTEXT,
        "file",
        dependency_ids=[prepare.id],
    )
    store.complete_task(
        parent.task.id, parent.worker_id, [], [context], final.id, claim_token=parent.claim_token
    )
    context_claim = store.claim_task(
        "context-worker",
        {TaskStage.EXTRACT_DOCUMENT_CONTEXT},
        timedelta(minutes=1),
    )
    assert context_claim is not None
    windows = [
        _task(
            run_id,
            f"window-{index}",
            TaskStage.EXTRACT_ENTITY_WINDOW,
            f"window-{index}",
            dependency_ids=[context.id],
        )
        for index in range(4)
    ]
    inventory = _task(
        run_id,
        "inventory",
        TaskStage.COMPACT_ENTITY_INVENTORY,
        "file",
        dependency_ids=[context.id, *(window.id for window in windows)],
    )
    store.complete_task(
        context.id,
        context_claim.worker_id,
        [],
        [*windows, inventory],
        final.id,
        claim_token=context_claim.claim_token,
    )

    assert (
        store.claim_task(
            "final-worker",
            {TaskStage.FINALIZE_GRAPH},
            timedelta(minutes=1),
        )
        is None
    )
    claimed_windows = [
        store.claim_task(
            f"extract-worker-{index}",
            {TaskStage.EXTRACT_ENTITY_WINDOW},
            timedelta(minutes=1),
        )
        for index in range(4)
    ]
    assert all(claim is not None for claim in claimed_windows)
    assert len({claim.task.id for claim in claimed_windows if claim is not None}) == 4
    for claim in claimed_windows:
        assert claim is not None
        store.complete_task(claim.task.id, claim.worker_id, [], claim_token=claim.claim_token)

    assert (
        store.claim_task(
            "final-worker",
            {TaskStage.FINALIZE_GRAPH},
            timedelta(minutes=1),
        )
        is None
    )
    inventory_claim = store.claim_task(
        "inventory-worker",
        {TaskStage.COMPACT_ENTITY_INVENTORY},
        timedelta(minutes=1),
    )
    assert inventory_claim is not None
    assert set(inventory_claim.dependency_outputs) == {
        context.id,
        *(window.id for window in windows),
    }
    relation_windows = [
        _task(
            run_id,
            f"relation-{index}",
            TaskStage.EXTRACT_RELATION_WINDOW,
            f"window-{index}",
            dependency_ids=[inventory.id],
        )
        for index in range(4)
    ]
    compact = _task(
        run_id,
        "compact",
        TaskStage.COMPACT_DOCUMENT,
        "file",
        dependency_ids=[inventory.id, *(window.id for window in relation_windows)],
    )
    store.complete_task(
        inventory_claim.task.id,
        inventory_claim.worker_id,
        [],
        [*relation_windows, compact],
        final.id,
        claim_token=inventory_claim.claim_token,
    )

    claimed_relations = [
        store.claim_task(
            f"relation-worker-{index}",
            {TaskStage.EXTRACT_RELATION_WINDOW},
            timedelta(minutes=1),
        )
        for index in range(4)
    ]
    assert all(claim is not None for claim in claimed_relations)
    for claim in claimed_relations:
        assert claim is not None
        store.complete_task(claim.task.id, claim.worker_id, [], claim_token=claim.claim_token)

    compact_claim = store.claim_task(
        "compact-worker",
        {TaskStage.COMPACT_DOCUMENT},
        timedelta(minutes=1),
    )
    assert compact_claim is not None
    store.complete_task(
        compact_claim.task.id, compact_claim.worker_id, [], claim_token=compact_claim.claim_token
    )

    final_claim = store.claim_task(
        "final-worker",
        {TaskStage.FINALIZE_GRAPH},
        timedelta(minutes=1),
    )
    assert final_claim is not None
    # Run-wide finalization discovers stage artifacts through the artifact store;
    # scheduling therefore stays constant-size regardless of document count.
    assert final_claim.dependency_outputs == {}


def test_spark_finalizer_claim_keeps_barrier_without_loading_outputs(
    isolated_postgres_dsn: str,
) -> None:
    """Avoid materializing corpus-sized dependency maps in a Spark driver lease."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    prepare = _task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file")
    final = _task(
        run_id,
        "final",
        TaskStage.FINALIZE_GRAPH,
        "graph",
        dependency_ids=[prepare.id],
    )
    store.add_tasks(run_id, [prepare, final])
    store.activate_run(run_id)

    assert (
        store.claim_task(
            "spark-driver",
            {TaskStage.FINALIZE_GRAPH},
            timedelta(minutes=1),
            skip_dependency_outputs_for_stages={TaskStage.FINALIZE_GRAPH},
        )
        is None
    )
    parent = store.claim_task(
        "prepare-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
    )
    assert parent is not None
    artifact = store.put(
        run_id,
        ArtifactKind.PREPARED_DOCUMENT,
        b"{}",
        "application/json",
    )
    store.complete_task(
        parent.task.id, parent.worker_id, [artifact.id], claim_token=parent.claim_token
    )

    final_claim = store.claim_task(
        "spark-driver",
        {TaskStage.FINALIZE_GRAPH},
        timedelta(minutes=1),
        skip_dependency_outputs_for_stages={TaskStage.FINALIZE_GRAPH},
    )
    assert final_claim is not None
    assert final_claim.dependency_outputs == {}
    assert final_claim.task.dependency_ids == []


def test_graph_manifest_and_active_version_publish_in_one_transaction(
    isolated_postgres_dsn: str,
    tmp_path: Path,
) -> None:
    """A successful final task must activate only its complete immutable artifact."""

    blob_store = LocalBlobStore((tmp_path / "objects").as_uri())
    store = PostgresDistributedStore(isolated_postgres_dsn, blob_store=blob_store)
    store.initialize()
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    final = _task(run_id, "final", TaskStage.FINALIZE_GRAPH, "integration-graph")
    store.add_tasks(run_id, [final])
    store.activate_run(run_id)
    claim = store.claim_task("final-worker", {TaskStage.FINALIZE_GRAPH}, timedelta(minutes=1))
    assert claim is not None
    payload = b'{"format":"flakegraph.graph-dataset-manifest/v1"}'
    artifact = store.put(
        run_id,
        ArtifactKind.GRAPH_RESULT,
        payload,
        "application/vnd.flakegraph.graph-manifest+json",
    )

    store.complete_task(
        claim.task.id, claim.worker_id, [artifact.id], claim_token=claim.claim_token
    )

    with psycopg.connect(isolated_postgres_dsn, row_factory=dict_row) as connection:
        head = connection.execute(
            """
            SELECT head.graph_id, version.run_id, version.artifact_id
            FROM flakegraph_graph_head AS head
            JOIN flakegraph_graph_version AS version ON version.id = head.version_id
            """
        ).fetchone()
    assert head == {
        "graph_id": "integration-graph",
        "run_id": run_id,
        "artifact_id": artifact.id,
    }
    assert store.get(artifact.id).payload == payload


def _corpus_settings(dsn: str, tmp_path: Path, input_path: Path) -> Settings:
    """Settings for a small end-to-end corpus run against the live store."""

    return Settings.load(
        env={},
        overrides={
            "runtime": {"runtime": "kubernetes"},
            "job": {
                "job_id": "distributed-test",
                "graph_id": "distributed-test-graph",
                "owner": "tester@example.com",
            },
            "files": {"source": "local", "input_path": input_path},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "model": "hash", "dimension": 8},
            "graph": {
                "chunk_token_size": 30,
                "chunk_token_overlap": 5,
                "max_chunks_per_llm_call": 1,
                "extraction_parallelism": 1,
                "gleaning_max_passes": 0,
                "fail_on_quality_error": False,
            },
            "writer": {
                "provider": "local_artifacts",
                "output_path": tmp_path / "output",
            },
            "cache": {"provider": "none"},
            "distributed": {
                "database_url": dsn,
                "poll_interval_seconds": 0.01,
            },
        },
    )


def _write_source(input_path: Path, index: int) -> Path:
    """Write one small document whose sentences name index-specific entities."""

    path = input_path / f"source-{index}.txt"
    path.write_text(
        " ".join(
            [
                f"Teacher {index} founded School {index} in City {index}.",
                f"School {index} teaches Method {index} to Student {index}.",
            ]
            * 8
        ),
        encoding="utf-8",
    )
    return path


def _drain(store: PostgresDistributedStore, settings: Settings, run_id: str) -> RunSnapshot:
    """Run six workers over every stage until the run reaches a terminal state."""

    workers = [
        DistributedWorker(
            settings,
            f"worker-{index}",
            set(TaskStage),
            build_pipeline(settings),
            store,
            store,
        )
        for index in range(6)
    ]
    for _iteration in range(100):
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            list(executor.map(lambda worker: worker.process_one(), workers))
        snapshot = store.get_run(run_id)
        if snapshot.run.status == RunStatus.SUCCEEDED:
            return snapshot
        assert snapshot.run.status != RunStatus.FAILED, snapshot.run
    pytest.fail("distributed workers did not drain the queue within 100 iterations")


def _graph_of(store: PostgresDistributedStore, snapshot: RunSnapshot) -> GraphWriteBatch:
    """The graph a finished run published."""

    (final,) = [task for task in snapshot.tasks if task.task.stage == TaskStage.FINALIZE_GRAPH]
    (artifact_id,) = final.output_artifact_ids
    return GraphWriteBatch.model_validate_json(store.get(artifact_id).payload)


def _document_names(graph: GraphWriteBatch) -> set[str]:
    """The source files a published graph says it was built from."""

    return {
        Path(str(document.get("source_uri") or document.get("id"))).name
        for document in graph.documents
    }


def test_multiple_workers_drain_dynamic_pipeline_end_to_end(
    isolated_postgres_dsn: str,
    tmp_path: Path,
) -> None:
    """Workers should pull all discovered windows and publish one complete graph."""

    input_path = tmp_path / "input"
    input_path.mkdir()
    for index in range(4):
        _write_source(input_path, index)
    settings = _corpus_settings(isolated_postgres_dsn, tmp_path, input_path)
    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    DistributedRunPlanner(
        settings,
        LocalFileSource(input_path),
        store,
        store,
    ).submit(run_id)

    snapshot = _drain(store, settings, run_id)
    preparation_tasks = [
        task for task in snapshot.tasks if task.task.stage == TaskStage.PREPARE_DOCUMENT
    ]
    context_tasks = [
        task for task in snapshot.tasks if task.task.stage == TaskStage.EXTRACT_DOCUMENT_CONTEXT
    ]
    entity_tasks = [
        task for task in snapshot.tasks if task.task.stage == TaskStage.EXTRACT_ENTITY_WINDOW
    ]
    inventory_tasks = [
        task for task in snapshot.tasks if task.task.stage == TaskStage.COMPACT_ENTITY_INVENTORY
    ]
    relation_tasks = [
        task for task in snapshot.tasks if task.task.stage == TaskStage.EXTRACT_RELATION_WINDOW
    ]
    compact_tasks = [
        task for task in snapshot.tasks if task.task.stage == TaskStage.COMPACT_DOCUMENT
    ]
    final_tasks = [task for task in snapshot.tasks if task.task.stage == TaskStage.FINALIZE_GRAPH]
    assert len(preparation_tasks) == 4
    assert len(context_tasks) == len(preparation_tasks)
    assert len(entity_tasks) > len(preparation_tasks)
    assert len(relation_tasks) == len(entity_tasks)
    assert len(inventory_tasks) == len(preparation_tasks)
    assert len(compact_tasks) == len(preparation_tasks)
    assert len(final_tasks) == 1
    assert all(task.status == TaskStatus.SUCCEEDED for task in snapshot.tasks)
    assert len(final_tasks[0].output_artifact_ids) == 1
    graph = store.get(final_tasks[0].output_artifact_ids[0])
    assert graph.ref.kind == ArtifactKind.GRAPH_RESULT
    assert (tmp_path / "output" / "run_report.json").is_file()


def test_every_task_of_an_interactive_run_is_claimed_in_its_band(
    isolated_postgres_dsn: str,
    tmp_path: Path,
) -> None:
    """Workers serving every band stamp none of their own on the work they discover."""

    input_path = tmp_path / "input"
    input_path.mkdir()
    _write_source(input_path, 0)
    settings = _corpus_settings(isolated_postgres_dsn, tmp_path, input_path)
    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    DistributedRunPlanner(settings, LocalFileSource(input_path), store, store).submit(
        run_id, priority_band=PriorityBand.INTERACTIVE
    )

    snapshot = _drain(store, settings, run_id)

    stages = {task.task.id: task.task.stage for task in snapshot.tasks}
    assert set(stages.values()) > {TaskStage.PREPARE_DOCUMENT, TaskStage.FINALIZE_GRAPH}
    assert _task_priorities(isolated_postgres_dsn, run_id) == {
        task_id: task_priority(stage, PriorityBand.INTERACTIVE) for task_id, stage in stages.items()
    }


def test_one_multi_slot_worker_builds_the_graph_many_polling_workers_build(
    isolated_postgres_dsn: str,
    tmp_path: Path,
) -> None:
    """Slots share one process's pipeline and store and still build the same graph.

    One worker with four slots drains a corpus through every stage over a
    store whose pool is sized from those slots; the graph it publishes has
    the entities and relations six single-task workers publish for the same
    documents, and no task needed a second attempt.
    """

    input_path = tmp_path / "input"
    input_path.mkdir()
    for index in range(4):
        _write_source(input_path, index)
    settings = _corpus_settings(isolated_postgres_dsn, tmp_path, input_path)
    store = _store(isolated_postgres_dsn)
    polled_run = f"run_{uuid4().hex}"
    DistributedRunPlanner(settings, LocalFileSource(input_path), store, store).submit(polled_run)
    polled = _graph_of(store, _drain(store, settings, polled_run))

    slotted_settings = settings.model_copy(
        update={
            "job": settings.job.model_copy(update={"graph_id": "slotted-graph"}),
            "distributed": settings.distributed.model_copy(update={"worker_slots": 4}),
        }
    )
    slotted_store = build_distributed_store(slotted_settings)
    slotted_run = f"run_{uuid4().hex}"
    DistributedRunPlanner(
        slotted_settings, LocalFileSource(input_path), slotted_store, slotted_store
    ).submit(slotted_run)
    worker = DistributedWorker(
        slotted_settings,
        "slotted-worker",
        set(TaskStage),
        build_pipeline(slotted_settings),
        slotted_store,
        slotted_store,
    )
    deadline = monotonic() + 120

    def finished() -> bool:
        status = slotted_store.get_run_summary(slotted_run).run.status
        assert monotonic() < deadline, "the slotted worker did not drain the run in time"
        return status in {RunStatus.SUCCEEDED, RunStatus.FAILED}

    worker.run_until_stopped(finished)

    snapshot = slotted_store.get_run(slotted_run)
    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert all(task.attempts == 1 for task in snapshot.tasks)
    slotted = _graph_of(slotted_store, snapshot)
    assert _facts(slotted) == _facts(polled)
    assert _facts(polled)[1]


def _facts(graph: GraphWriteBatch) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    """A graph's entities and relations by name, apart from its graph identity."""

    names = {node.id: node.name for node in graph.nodes}
    return (
        {(node.name, node.primary_type) for node in graph.nodes},
        {
            (names[edge.source_node_id], edge.relation_type, names[edge.target_node_id])
            for edge in graph.edges
        },
    )


def test_a_revision_keeps_drops_and_adds_documents_without_re_extracting_kept_ones(
    isolated_postgres_dsn: str,
    tmp_path: Path,
) -> None:
    """A second run of a graph is built from kept outputs plus what it adds.

    The base graph has documents 0 and 1. The revision drops 1, is offered 0
    again (already held: skipped) and 2 (new), so its workers prepare and
    extract only document 2, and the graph it publishes names the entities of
    0 and 2 but none of 1. The head moves to the revision; the base version
    stays readable.
    """

    base_input = tmp_path / "base"
    base_input.mkdir()
    for index in (0, 1):
        _write_source(base_input, index)
    settings = _corpus_settings(isolated_postgres_dsn, tmp_path, base_input)
    store = _store(isolated_postgres_dsn)
    base_run_id = f"run_{uuid4().hex}"
    DistributedRunPlanner(settings, LocalFileSource(base_input), store, store).submit(base_run_id)
    base = _drain(store, settings, base_run_id)
    assert _document_names(_graph_of(store, base)) == {"source-0.txt", "source-1.txt"}
    dropped_file_id = next(
        task.task.scope_id
        for task in base.tasks
        if task.task.stage == TaskStage.PREPARE_DOCUMENT
        and "source-1" in str(store.get(task.task.payload["source_artifact_id"]).ref.metadata)
    )

    revision_input = tmp_path / "revision"
    revision_input.mkdir()
    _write_source(revision_input, 0)
    _write_source(revision_input, 2)
    revision_run_id = f"run_{uuid4().hex}"
    summary = DistributedRunPlanner(
        settings,
        LocalFileSource(revision_input),
        store,
        store,
    ).submit(
        revision_run_id, RevisionRequest(base_run_id=base_run_id, drop_file_ids=[dropped_file_id])
    )
    prepared = [item for item in summary.task_counts if item.stage == TaskStage.PREPARE_DOCUMENT]
    assert sum(item.count for item in prepared) == 1
    (finalizer,) = store.get_stage_tasks(revision_run_id, TaskStage.FINALIZE_GRAPH)
    assert finalizer.task.payload["inherit"] == [
        {
            "run_id": base_run_id,
            "file_ids": sorted(
                task.task.scope_id
                for task in base.tasks
                if task.task.stage == TaskStage.PREPARE_DOCUMENT
                and task.task.scope_id != dropped_file_id
            ),
        }
    ]

    revised = _drain(store, settings, revision_run_id)
    revised_graph = _graph_of(store, revised)
    assert _document_names(revised_graph) == {"source-0.txt", "source-2.txt"}
    assert revised_graph.nodes and revised_graph.evidence
    assert len([t for t in revised.tasks if t.task.stage == TaskStage.PREPARE_DOCUMENT]) == 1
    with psycopg.connect(isolated_postgres_dsn, row_factory=dict_row) as connection:
        head = connection.execute(
            """
            SELECT version.run_id
            FROM flakegraph_graph_head AS head
            JOIN flakegraph_graph_version AS version ON version.id = head.version_id
            WHERE head.graph_id = %s
            """,
            (settings.job.graph_id,),
        ).fetchone()
        versions = connection.execute(
            "SELECT run_id FROM flakegraph_graph_version WHERE graph_id = %s ORDER BY created_at",
            (settings.job.graph_id,),
        ).fetchall()
    assert head == {"run_id": revision_run_id}
    assert [row["run_id"] for row in versions] == [base_run_id, revision_run_id]

    # Offering only what the revised graph already holds changes nothing.
    with pytest.raises(ValueError, match="would not change; the 2 offered are already in it"):
        DistributedRunPlanner(settings, LocalFileSource(revision_input), store, store).submit(
            f"run_{uuid4().hex}", RevisionRequest(base_run_id=revision_run_id)
        )

    # Dropping a document and adding none prepares nothing: the finalizer
    # alone builds the smaller graph from what it inherits, and heads move.
    dropped_again = next(
        task.task.scope_id
        for task in revised.tasks
        if task.task.stage == TaskStage.PREPARE_DOCUMENT
    )
    drop_only_run_id = f"run_{uuid4().hex}"
    drop_only = DistributedRunPlanner(settings, None, store, store).submit(
        drop_only_run_id,
        RevisionRequest(base_run_id=revision_run_id, drop_file_ids=[dropped_again]),
    )
    assert not [t for t in drop_only.task_counts if t.stage == TaskStage.PREPARE_DOCUMENT]
    smaller = _graph_of(store, _drain(store, settings, drop_only_run_id))
    assert _document_names(smaller) == {"source-0.txt"}
    assert _graph_head(isolated_postgres_dsn, settings.job.graph_id) == drop_only_run_id

    # A rebuild changes no document and is still a new version: the documents
    # are the same, but how a graph is built from them has changed - a
    # finalizer fix - and finalizing again is the only way to apply it.
    # Nothing is prepared or extracted; the finalizer alone rebuilds it.
    with pytest.raises(ValueError, match="would not change"):
        DistributedRunPlanner(settings, None, store, store).submit(
            f"run_{uuid4().hex}", RevisionRequest(base_run_id=drop_only_run_id)
        )
    rebuild_run_id = f"run_{uuid4().hex}"
    rebuilt_plan = DistributedRunPlanner(settings, None, store, store).submit(
        rebuild_run_id, RevisionRequest(base_run_id=drop_only_run_id, rebuild=True)
    )
    assert [count.stage for count in rebuilt_plan.task_counts] == [TaskStage.FINALIZE_GRAPH]
    rebuilt = _graph_of(store, _drain(store, settings, rebuild_run_id))
    assert _document_names(rebuilt) == {"source-0.txt"}
    assert _graph_head(isolated_postgres_dsn, settings.job.graph_id) == rebuild_run_id


def test_a_revision_reextracts_and_revalidates_kept_documents_without_reading_them_again(
    isolated_postgres_dsn: str,
    tmp_path: Path,
) -> None:
    """Kept documents can be extracted again from a stage, or revalidated, from their outputs.

    The base holds documents 0, 1 and 2. The first revision extracts
    document 0's relations again and revalidates 1 and 2: nothing is
    prepared, no entity window is extracted, and only document 0's relation
    windows call for extraction; 1 and 2 are replayed window by window. A
    second revision, of the first, extracts document
    1's entities again - from the context the first revision carried forward
    - and reuses the rest, including document 0, whose text still lives in the
    base and whose extraction lives in the first revision. Every published
    graph says how it came by each document.
    """

    base_input = tmp_path / "base"
    base_input.mkdir()
    for index in (0, 1, 2):
        _write_source(base_input, index)
    settings = _corpus_settings(isolated_postgres_dsn, tmp_path, base_input)
    store = _store(isolated_postgres_dsn)
    base_run_id = f"run_{uuid4().hex}"
    DistributedRunPlanner(settings, LocalFileSource(base_input), store, store).submit(base_run_id)
    base = _drain(store, settings, base_run_id)
    file_ids = {
        Path(str(row["source_uri"])).name: str(row["file_id"])
        for row in _graph_of(store, base).documents
    }
    assert {row["provenance"] for row in _graph_of(store, base).documents} == {"extracted"}

    first_run_id = f"run_{uuid4().hex}"
    DistributedRunPlanner(settings, None, store, store).submit(
        first_run_id,
        RevisionRequest(
            base_run_id=base_run_id,
            reextract=[file_ids["source-0.txt"]],
            reextract_from="relations",
            revalidate=True,
        ),
    )
    first = _drain(store, settings, first_run_id)
    stages = [task.task.stage for task in first.tasks]
    assert TaskStage.PREPARE_DOCUMENT not in stages
    assert TaskStage.EXTRACT_ENTITY_WINDOW not in stages
    # Each revalidated document is started, replayed window by window, and
    # combined by its revalidate_document task.
    assert stages.count(TaskStage.REVALIDATE_DOCUMENT_CONTEXT) == 2
    assert TaskStage.REVALIDATE_ENTITY_WINDOW in stages
    assert TaskStage.REVALIDATE_RELATION_WINDOW in stages
    assert stages.count(TaskStage.REVALIDATE_DOCUMENT) == 2
    assert {
        task.task.scope_id for task in first.tasks if task.task.stage == TaskStage.COMPACT_DOCUMENT
    } == {file_ids["source-0.txt"]}
    first_graph = _graph_of(store, first)
    assert {
        Path(str(row["source_uri"])).name: row["provenance"] for row in first_graph.documents
    } == {
        "source-0.txt": "reextracted:relations",
        "source-1.txt": "revalidated",
        "source-2.txt": "revalidated",
    }
    assert first_graph.nodes and first_graph.edges
    for task in first.tasks:
        if task.task.stage != TaskStage.REVALIDATE_DOCUMENT:
            continue
        revalidated = ExtractedDocumentShard.model_validate_json(
            store.get(task.output_artifact_ids[0]).payload
        )
        (summary,) = [e for e in revalidated.trace if e["stage"] == "revalidation"]
        # The rules did not change, so what the base accepted stands, and
        # nothing is admitted that it turned away.
        assert summary["entities"]["kept"] == summary["entities"]["before"] > 0
        assert summary["relations"]["kept"] == summary["relations"]["before"] > 0
        assert summary["entities"]["admitted"] == summary["relations"]["admitted"] == 0
    (first_finalizer,) = store.get_stage_tasks(first_run_id, TaskStage.FINALIZE_GRAPH)
    assert first_finalizer.task.payload["inherit"] == [
        {"run_id": base_run_id, "prepared_file_ids": sorted(file_ids.values())}
    ]

    second_run_id = f"run_{uuid4().hex}"
    DistributedRunPlanner(settings, None, store, store).submit(
        second_run_id,
        RevisionRequest(
            base_run_id=first_run_id,
            reextract=[f"file:{file_ids['source-1.txt']}"],
            reextract_from="entities",
        ),
    )
    second = _drain(store, settings, second_run_id)
    (context,) = [t for t in second.tasks if t.task.stage == TaskStage.EXTRACT_DOCUMENT_CONTEXT]
    assert context.task.scope_id == file_ids["source-1.txt"]
    assert "context_artifact_id" in context.task.payload
    second_graph = _graph_of(store, second)
    assert {
        Path(str(row["source_uri"])).name: row["provenance"] for row in second_graph.documents
    } == {
        "source-0.txt": "reused",
        "source-1.txt": "reextracted:entities",
        "source-2.txt": "reused",
    }
    (second_finalizer,) = store.get_stage_tasks(second_run_id, TaskStage.FINALIZE_GRAPH)
    # Entries come in run id order, which is random here.
    assert sorted(
        second_finalizer.task.payload["inherit"], key=lambda entry: entry["run_id"] != base_run_id
    ) == [
        {"run_id": base_run_id, "prepared_file_ids": sorted(file_ids.values())},
        {
            "run_id": first_run_id,
            "file_ids": sorted([file_ids["source-0.txt"], file_ids["source-2.txt"]]),
        },
    ]
    assert _graph_head(isolated_postgres_dsn, settings.job.graph_id) == second_run_id


def test_an_invalid_initial_plan_leaves_no_tasks_behind(isolated_postgres_dsn: str) -> None:
    """A refused plan must not leave a half-plan a worker could claim.

    Batches are committed as they stream, so a plan whose final task
    inherits nothing and prepares nothing has already copied its finalizer
    when the check refuses it. Nothing of it may remain: the planner then
    sees an empty run and cancels it, and the console lists no phantom.
    """

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    with pytest.raises(ValueError, match="document tasks unless the final task inherits"):
        store.add_initial_tasks(
            run_id, iter([_task(run_id, "final", TaskStage.FINALIZE_GRAPH, "graph")])
        )
    assert store.get_run_summary(run_id).total_tasks == 0

    # The same plan with an inheriting finalizer is a drop-only revision.
    inheriting = _task(run_id, "final", TaskStage.FINALIZE_GRAPH, "graph").model_copy(
        update={"payload": {"inherit": [{"run_id": "earlier", "file_ids": ["doc"]}]}}
    )
    store.add_initial_tasks(run_id, iter([inheriting]))
    assert store.get_run_summary(run_id).total_tasks == 1


def _graph_head(dsn: str, graph_id: str) -> str | None:
    """Return the run whose version is the graph's head."""

    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        row = connection.execute(
            """
            SELECT version.run_id
            FROM flakegraph_graph_head AS head
            JOIN flakegraph_graph_version AS version ON version.id = head.version_id
            WHERE head.graph_id = %s
            """,
            (graph_id,),
        ).fetchone()
    return None if row is None else str(row["run_id"])


def _worker_demand(dsn: str) -> dict[str, int]:
    """Read the database contract consumed by KEDA's PostgreSQL scaler."""

    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        rows = connection.execute(
            "SELECT stage, desired_workers FROM flakegraph_worker_demand ORDER BY stage"
        ).fetchall()
    return {str(row["stage"]): int(row["desired_workers"]) for row in rows}


def _banded_worker_demand(dsn: str) -> dict[tuple[str, str], int]:
    """The same contract by stage and band, as KEDA's per-band triggers read it."""

    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        rows = connection.execute(
            "SELECT stage, priority_band, desired_workers FROM flakegraph_worker_demand"
        ).fetchall()
    return {
        (str(row["stage"]), str(row["priority_band"])): int(row["desired_workers"]) for row in rows
    }


def _task_priorities(dsn: str, run_id: str) -> dict[str, int]:
    """Each task's stored claim priority, by task id."""

    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        rows = connection.execute(
            "SELECT id, priority FROM flakegraph_task WHERE run_id = %s", (run_id,)
        ).fetchall()
    return {str(row["id"]): int(row["priority"]) for row in rows}


def test_deleting_a_graph_removes_its_runs_rows_and_objects_and_nothing_else(
    isolated_postgres_dsn: str,
    tmp_path: Path,
) -> None:
    """A graph with a revision goes whole; another graph beside it is untouched.

    The revision is built from its base run's stored outputs, which is why a
    graph is deleted as a unit. Every row naming its runs and every object
    under their prefixes goes; the other graph still opens. Deleting again
    finds nothing, and a graph with work under way is refused.
    """

    blobs = tmp_path / "blobs"
    store = PostgresDistributedStore(
        isolated_postgres_dsn, blob_store=LocalBlobStore(blobs.as_uri())
    )
    store.initialize()
    base_input = tmp_path / "base"
    base_input.mkdir()
    _write_source(base_input, 0)
    settings = _corpus_settings(isolated_postgres_dsn, tmp_path, base_input)
    base_run = f"run_{uuid4().hex}"
    DistributedRunPlanner(settings, LocalFileSource(base_input), store, store).submit(base_run)
    _drain(store, settings, base_run)
    added = tmp_path / "added"
    added.mkdir()
    _write_source(added, 1)
    revision_run = f"run_{uuid4().hex}"
    DistributedRunPlanner(settings, LocalFileSource(added), store, store).submit(
        revision_run, RevisionRequest(base_run_id=base_run)
    )
    _drain(store, settings, revision_run)

    other_settings = settings.model_copy(
        update={"job": settings.job.model_copy(update={"graph_id": "other-graph"})}
    )
    other_run = f"run_{uuid4().hex}"
    DistributedRunPlanner(other_settings, LocalFileSource(base_input), store, store).submit(
        other_run
    )
    other = _drain(store, other_settings, other_run)
    assert (blobs / base_run).is_dir() and (blobs / revision_run).is_dir()

    deletion = store.delete_graph(settings.job.graph_id)

    assert deletion.run_ids == [base_run, revision_run]
    assert deletion.tasks > 0 and deletion.artifacts > 0 and deletion.objects > 0
    assert not (blobs / base_run).exists() and not (blobs / revision_run).exists()
    with psycopg.connect(isolated_postgres_dsn, row_factory=dict_row) as connection:
        for table in ("flakegraph_task", "flakegraph_artifact", "flakegraph_publication"):
            row = connection.execute(
                sql.SQL("SELECT COUNT(*) AS n FROM {} WHERE run_id = ANY(%s)").format(
                    sql.Identifier(table)
                ),
                ([base_run, revision_run],),
            ).fetchone()
            assert row == {"n": 0}, table
        for table in ("flakegraph_run", "flakegraph_graph_version", "flakegraph_graph_head"):
            row = connection.execute(
                sql.SQL("SELECT COUNT(*) AS n FROM {} WHERE graph_id = %s").format(
                    sql.Identifier(table)
                ),
                (settings.job.graph_id,),
            ).fetchone()
            assert row == {"n": 0}, table
    # The graph beside it keeps its rows, objects and published graph.
    assert (blobs / other_run).is_dir()
    assert _graph_of(store, store.get_run(other_run)).nodes
    assert other.run.status == RunStatus.SUCCEEDED

    again = store.delete_graph(settings.job.graph_id)
    assert (again.run_ids, again.tasks, again.artifacts, again.objects) == ([], 0, 0, 0)

    busy_run = f"run_{uuid4().hex}"
    DistributedRunPlanner(settings, LocalFileSource(base_input), store, store).submit(busy_run)
    with pytest.raises(ValueError, match="still has work under way"):
        store.delete_graph(settings.job.graph_id)
    assert store.get_run(busy_run).run.status != RunStatus.SUCCEEDED


def _store(dsn: str) -> PostgresDistributedStore:
    """Return an initialized adapter using the opt-in integration DSN."""

    store = PostgresDistributedStore(dsn)
    store.initialize()
    return store


def _run(run_id: str, band: PriorityBand = PriorityBand.BULK) -> RunDefinition:
    """Build one unique live-test run definition."""

    return RunDefinition(
        id=run_id,
        graph_id="integration-graph",
        config={"test": True},
        config_digest="integration-config",
        priority_band=band,
    )


def _task(
    run_id: str,
    suffix: str,
    stage: TaskStage,
    scope_id: str,
    *,
    dependency_ids: list[str] | None = None,
    max_attempts: int = 3,
) -> TaskDefinition:
    """Build a task whose identity is isolated to the current live-test run."""

    return TaskDefinition(
        id=f"{run_id}-{suffix}",
        run_id=run_id,
        stage=stage,
        scope_id=scope_id,
        dependency_ids=dependency_ids or [],
        max_attempts=max_attempts,
    )


def test_retry_quarantines_failed_document_descendants_without_requeueing_them(
    isolated_postgres_dsn: str,
) -> None:
    """A failed window can be quarantined by file ID while other work remains retryable."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    prepare = _task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "poison")
    window = _task(run_id, "window", TaskStage.EXTRACT_ENTITY_WINDOW, "window-id", max_attempts=1)
    window = window.model_copy(update={"dependency_ids": [prepare.id]})
    compact = _task(run_id, "compact", TaskStage.COMPACT_DOCUMENT, "poison")
    compact = compact.model_copy(update={"dependency_ids": [window.id]})
    good = _task(run_id, "good", TaskStage.PREPARE_DOCUMENT, "good")
    final = _task(run_id, "final", TaskStage.FINALIZE_GRAPH, "graph")
    store.add_tasks(run_id, [prepare, window, compact, good, final])
    store.activate_run(run_id)
    claim = store.claim_task("worker", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert claim is not None
    if claim.task.id == good.id:
        store.complete_task(good.id, "worker", [], claim_token=claim.claim_token)
        claim = store.claim_task("worker", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
        assert claim is not None
    assert claim.task.id == prepare.id
    store.complete_task(prepare.id, "worker", [], claim_token=claim.claim_token)
    claim = store.claim_task("worker", {TaskStage.EXTRACT_ENTITY_WINDOW}, timedelta(minutes=1))
    assert claim is not None
    store.fail_task(
        claim.task.id,
        "worker",
        {"message": "poison window"},
        timedelta(0),
        claim_token=claim.claim_token,
    )

    with pytest.raises(ValueError, match="must name documents of this run"):
        store.retry_run(run_id, ["unknown"])
    assert store.get_run(run_id).run.status is RunStatus.FAILED
    # A healthy document has nothing to retry, so naming it excludes nothing.
    store.retry_run(run_id, ["good", "poison"])
    snapshot = store.get_run(run_id)
    assert snapshot.run.config["quarantined_file_ids"] == ["poison"]
    by_id = {item.task.id: item for item in snapshot.tasks}
    assert by_id[window.id].status is TaskStatus.SUCCEEDED
    assert by_id[compact.id].status is TaskStatus.SUCCEEDED
    assert by_id[window.id].attempts == 1
    assert (
        store.claim_task("worker", {TaskStage.EXTRACT_ENTITY_WINDOW}, timedelta(minutes=1)) is None
    )
    # Successful prerequisites retain their original attempt and immutable outputs.
    assert by_id[prepare.id].attempts == 1
    assert store.get_run_summary(run_id).resolved_failures


def test_staged_publication_commit_rejects_cancelled_or_superseded_lease(
    isolated_postgres_dsn: str,
) -> None:
    """The destination callback is never entered once ownership or graph head changes."""

    store = _store(isolated_postgres_dsn)

    def finalized_run(run_id: str) -> PublicationLease:
        store.create_run(_run(run_id))
        task = _task(run_id, "final", TaskStage.FINALIZE_GRAPH, "graph")
        task = task.model_copy(update={"payload": {"output": {"provider": "local_artifacts"}}})
        store.add_tasks(run_id, [task])
        store.activate_run(run_id)
        claim = store.claim_task("worker", {TaskStage.FINALIZE_GRAPH}, timedelta(minutes=1))
        assert claim is not None
        artifact = store.put(run_id, ArtifactKind.GRAPH_RESULT, b"{}", "application/json")
        store.complete_task(task.id, "worker", [artifact.id], claim_token=claim.claim_token)
        publication = store.claim_publication("publisher", timedelta(minutes=1))
        assert publication is not None
        return publication

    stale = finalized_run(f"run_{uuid4().hex}")
    store.cancel_run(stale.run_id)
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="no longer owns"):
        store.commit_publication(stale, lambda: calls.append("cancelled"))
    first = finalized_run(f"run_{uuid4().hex}")
    second = finalized_run(f"run_{uuid4().hex}")
    with pytest.raises(RuntimeError, match="no longer owns"):
        store.commit_publication(first, lambda: calls.append("superseded"))
    store.commit_publication(second, lambda: calls.append("latest"))
    assert calls == ["latest"]
    assert store.get_run_summary(second.run_id).run.status is RunStatus.SUCCEEDED


def test_staged_publication_preserves_fence_during_slow_destination_commit(
    isolated_postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transaction-local idle exemption keeps locks while external replacement runs."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    task = _task(run_id, "final", TaskStage.FINALIZE_GRAPH, "graph").model_copy(
        update={"payload": {"output": {"provider": "local_artifacts"}}}
    )
    store.add_tasks(run_id, [task])
    store.activate_run(run_id)
    claim = store.claim_task("worker", {TaskStage.FINALIZE_GRAPH}, timedelta(minutes=1))
    assert claim is not None
    artifact = store.put(run_id, ArtifactKind.GRAPH_RESULT, b"{}", "application/json")
    store.complete_task(task.id, "worker", [artifact.id], claim_token=claim.claim_token)
    lease = store.claim_publication("publisher", timedelta(minutes=1))
    assert lease is not None
    original = store._connection
    settings: dict[str, str] = {}
    active: list[psycopg.Connection[dict[str, Any]]] = []

    @contextmanager
    def short_idle() -> Iterator[psycopg.Connection[dict[str, Any]]]:
        """Make loss of an unprotected fence observable without a minute-long test."""

        with original() as connection:
            connection.execute("SET LOCAL idle_in_transaction_session_timeout = '100ms'")
            connection.execute("SET LOCAL statement_timeout = '5s'")
            active.append(connection)
            yield connection

    def commit() -> None:
        """Outlast the tiny idle budget and prove the run fence still excludes a rival."""

        sleep(0.25)
        row = (
            active[-1]
            .execute("""
            SELECT current_setting('idle_in_transaction_session_timeout') AS idle,
                   current_setting('statement_timeout') AS statement
        """)
            .fetchone()
        )
        assert row is not None
        settings.update(row)
        with (
            psycopg.connect(isolated_postgres_dsn, autocommit=True) as rival,
            pytest.raises(psycopg.errors.LockNotAvailable),
        ):
            rival.execute(
                "SELECT id FROM flakegraph_run WHERE id = %s FOR UPDATE NOWAIT", (run_id,)
            )

    with monkeypatch.context() as patch:
        patch.setattr(store, "_connection", short_idle)
        store.commit_publication(lease, commit)
    assert settings == {"idle": "0", "statement": "5s"}
    assert store.get_run_summary(run_id).run.status is RunStatus.SUCCEEDED
    with original() as connection:
        restored = connection.execute(
            "SELECT current_setting('idle_in_transaction_session_timeout') AS idle"
        ).fetchone()
    assert restored == {"idle": "1min"}


def test_claim_token_fences_every_task_transition(isolated_postgres_dsn: str) -> None:
    """Only the opaque token returned by a claim can mutate that task."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    task = _task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file")
    store.add_tasks(run_id, [task])
    store.activate_run(run_id)
    claim = store.claim_task("worker", {task.stage}, timedelta(minutes=1))
    assert claim is not None and claim.claim_token
    progress = TaskProgress(phase="reading", phase_index=1, phase_total=1)
    transitions: tuple[Callable[[], None], ...] = (
        lambda: store.heartbeat(task.id, "worker", timedelta(minutes=1), claim_token="wrong"),
        lambda: store.report_task_progress(task.id, "worker", progress, claim_token="wrong"),
        lambda: store.complete_task(task.id, "worker", [], claim_token="wrong"),
        lambda: store.fail_task(task.id, "worker", {}, timedelta(0), claim_token="wrong"),
    )
    for transition in transitions:
        with pytest.raises(RuntimeError, match="no longer owns"):
            transition()
    store.heartbeat(task.id, "worker", timedelta(minutes=1), claim_token=claim.claim_token)
    store.complete_task(task.id, "worker", [], claim_token=claim.claim_token)
    assert store.get_run(run_id).tasks[0].status == TaskStatus.SUCCEEDED


def test_deletion_reservation_blocks_retry_and_submission_during_blob_removal(
    isolated_postgres_dsn: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reservation remains committed throughout external object deletion."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    run = _run(run_id)
    store.create_run(run)
    store.cancel_run(run_id)
    blobs = LocalBlobStore(tmp_path.as_uri())
    store.blob_store = blobs
    original = blobs.delete_prefix
    checks = []

    def remove(prefix: str) -> int:
        with pytest.raises(ValueError, match="being deleted"):
            store.retry_run(run_id)
        with pytest.raises(ValueError, match="being deleted"):
            store.create_run(
                _run(f"run_{uuid4().hex}").model_copy(update={"graph_id": run.graph_id})
            )
        with pytest.raises(ValueError, match="already executing"):
            store.delete_graph(run.graph_id)
        checks.append(prefix)
        return original(prefix)

    monkeypatch.setattr(blobs, "delete_prefix", remove)
    result = store.delete_graph(run.graph_id)
    assert checks == [run_id]
    assert result.run_ids == [run_id]
    assert store.list_runs() == []
    # A completed deletion releases the graph identifier for a fresh submission.
    store.create_run(_run(f"run_{uuid4().hex}").model_copy(update={"graph_id": run.graph_id}))


def test_retry_uses_a_new_claim_token_when_the_attempt_budget_resets(
    isolated_postgres_dsn: str,
) -> None:
    """A new retry budget cannot restore the authority of a completed claim."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    task = _task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file", max_attempts=1)
    store.add_tasks(run_id, [task])
    store.activate_run(run_id)
    first = store.claim_task("worker", {task.stage}, timedelta(minutes=1))
    assert first is not None
    store.fail_task(task.id, "worker", {}, timedelta(0), claim_token=first.claim_token)
    store.retry_run(run_id)
    second = store.claim_task("worker", {task.stage}, timedelta(minutes=1))
    assert second is not None
    assert second.attempt == first.attempt == 1
    assert second.claim_token != first.claim_token
    with pytest.raises(RuntimeError, match="no longer owns"):
        store.complete_task(task.id, "worker", [], claim_token=first.claim_token)
    store.complete_task(task.id, "worker", [], claim_token=second.claim_token)


def test_durable_consumption_is_idempotent_and_scoped_to_the_run(
    isolated_postgres_dsn: str,
) -> None:
    """Repeated delivery preserves one event, while separate calls remain distinct."""

    store = _store(isolated_postgres_dsn)
    first, second = (f"run_{uuid4().hex}" for _ in range(2))
    store.create_run(_run(first))
    store.create_run(_run(second))
    event = ConsumptionEvent(
        graph_id="g",
        job_id=first,
        run_id=first,
        stage="extraction",
        operation="completion",
        provider="fake",
        model="fixture",
        locality=Locality.LOCAL,
        usage=TokenUsage(prompt_tokens=3, completion_tokens=2),
    )
    store.record_consumption(first, event)
    store.record_consumption(first, event)
    assert store.consumption_events(first) == [event]
    assert store.consumption_events(second) == []
    with pytest.raises(ValueError, match="belong"):
        store.record_consumption(second, event)
    store.record_consumption(first, event.model_copy(update={"event_id": uuid4().hex}))
    assert len(store.consumption_events(first)) == 2


def test_graph_deletion_waits_for_live_siblings_of_a_failed_run(
    isolated_postgres_dsn: str,
) -> None:
    """Failure of one task cannot authorize deletion under another live lease."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    run = _run(run_id)
    store.create_run(run)
    tasks = [
        _task(
            run_id, f"prepare-{index}", TaskStage.PREPARE_DOCUMENT, f"file-{index}", max_attempts=1
        )
        for index in range(2)
    ]
    store.add_tasks(run_id, tasks)
    store.activate_run(run_id)
    first = store.claim_task("worker-1", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    second = store.claim_task("worker-2", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert first is not None and second is not None
    store.fail_task(first.task.id, first.worker_id, {}, timedelta(0), claim_token=first.claim_token)
    with pytest.raises(ValueError, match="work under way"):
        store.delete_graph(run.graph_id)
    store.complete_task(second.task.id, second.worker_id, [], claim_token=second.claim_token)
    assert store.delete_graph(run.graph_id).run_ids == [run_id]
