"""Prove a multi-slot worker keeps several leased tasks in flight without mixing them up."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic, sleep
from typing import Any, cast

import pytest

from kg_processor.application import distributed_worker
from kg_processor.application.distributed_worker import (
    DistributedWorker,
    TaskExecution,
    WorkerIteration,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import PublicationLease, TaskDefinition, TaskLease, TaskStage
from kg_processor.factories import build_distributed_store, worker_pool_connections
from kg_processor.ports.artifact_store import ArtifactStore
from kg_processor.ports.distributed_pipeline import DistributedPipeline
from kg_processor.ports.task_store import TaskStore, TaskStoreUnavailableError

# Long enough for a lock-step test on a loaded machine, short enough to fail fast.
_WAIT_SECONDS = 10.0


class _SlotStore:
    """A thread-safe queue of ready tasks that records what each claim and lease did."""

    def __init__(self, tasks: list[TaskDefinition]) -> None:
        self._lock = threading.Lock()
        self.ready = list(tasks)
        self.claimed: list[str] = []
        self.claimed_stages: list[set[TaskStage]] = []
        self.heartbeats: dict[str, list[str]] = {}
        self.completed: list[str] = []
        self.failed: dict[str, dict[str, Any]] = {}
        self.sweeps = 0
        self.publication_claims = 0

    def claim_publication(
        self, worker_id: str, lease_duration: timedelta
    ) -> PublicationLease | None:
        """Expose the empty outbox that a finalizer-only slot worker still polls."""

        del worker_id, lease_duration
        with self._lock:
            self.publication_claims += 1
        return None

    def record_served_configuration(self, stages: set[TaskStage], config_digest: str) -> None:
        del stages, config_digest

    def fail_abandoned_tasks(self) -> None:
        with self._lock:
            self.sweeps += 1

    def claim_task(
        self,
        worker_id: str,
        stages: set[TaskStage],
        lease_duration: timedelta,
        expected_config_digest: str | None = None,
        skip_dependency_outputs_for_stages: set[TaskStage] | None = None,
    ) -> TaskLease | None:
        del expected_config_digest, skip_dependency_outputs_for_stages
        with self._lock:
            self.claimed_stages.append(set(stages))
            task = next((task for task in self.ready if task.stage in stages), None)
            if task is None:
                return None
            self.ready.remove(task)
            self.claimed.append(task.id)
            return TaskLease(
                task=task,
                worker_id=worker_id,
                attempt=1,
                claim_token="claim-1",
                lease_expires_at=datetime.now(UTC) + lease_duration,
            )

    def heartbeat(
        self, task_id: str, worker_id: str, lease_duration: timedelta, *, claim_token: str
    ) -> None:
        del lease_duration
        with self._lock:
            self.heartbeats.setdefault(task_id, []).append(worker_id)

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
        del worker_id, output_artifact_ids, follow_up_tasks, barrier_task_id
        with self._lock:
            self.completed.append(task_id)

    def fail_task(
        self,
        task_id: str,
        worker_id: str,
        error: dict[str, Any],
        retry_delay: timedelta,
        *,
        claim_token: str,
    ) -> None:
        del worker_id, retry_delay
        with self._lock:
            self.failed[task_id] = error


def _settings(tmp_path: Path, *, slots: int, lease_seconds: int = 300) -> Settings:
    """A worker profile with ``slots`` slots that polls and retries without delay."""

    return Settings.load(
        env={},
        overrides={
            "job": {"graph_id": "test-graph", "owner": "tester@example.com"},
            "files": {"input_path": str(tmp_path)},
            "llm": {"provider": "fake", "api_key": "test-secret"},
            "embedding": {"provider": "hash", "dimension": 8},
            "writer": {"provider": "local_artifacts", "output_path": str(tmp_path / "out")},
            "cache": {"provider": "none"},
            "distributed": {
                "database_url": "postgresql://user:secret@db/test",
                "worker_slots": slots,
                "lease_seconds": lease_seconds,
                "poll_interval_seconds": 0.01,
                "retry_delay_seconds": 0.01,
            },
        },
    )


def _tasks(stage: TaskStage, count: int, prefix: str = "task") -> list[TaskDefinition]:
    return [
        TaskDefinition(id=f"{prefix}-{index}", run_id="run-1", stage=stage, scope_id=f"w-{index}")
        for index in range(count)
    ]


def _worker(
    settings: Settings,
    store: _SlotStore,
    stages: set[TaskStage],
    execute: Callable[[TaskLease], TaskExecution],
) -> DistributedWorker:
    """A worker whose task execution is ``execute``, so only the slot machinery is real."""

    worker = DistributedWorker(
        settings,
        "worker-1",
        stages,
        cast(DistributedPipeline, object()),
        cast(TaskStore, store),
        cast(ArtifactStore, object()),
    )
    worker._execute = execute  # type: ignore[method-assign, assignment]
    return worker


def _until(condition: Callable[[], bool]) -> None:
    deadline = monotonic() + _WAIT_SECONDS
    while not condition():
        if monotonic() > deadline:
            raise AssertionError("condition was not reached in time")
        sleep(0.01)


def test_slots_run_distinct_tasks_at_once_each_under_its_own_lease(tmp_path: Path) -> None:
    """Every slot claims its own task, renews that task's lease, and completes it.

    The tasks meet at a barrier only all four slots together can pass, so the
    run proves they were in flight at the same time; each then waits for a
    renewal of its own lease before finishing.
    """

    settings = _settings(tmp_path, slots=4, lease_seconds=3)
    store = _SlotStore(_tasks(TaskStage.EXTRACT_ENTITY_WINDOW, 4))
    together = threading.Barrier(4, timeout=_WAIT_SECONDS)
    threads: dict[str, str] = {}

    def execute(lease: TaskLease) -> TaskExecution:
        threads[lease.task.id] = threading.current_thread().name
        together.wait()
        _until(lambda: bool(store.heartbeats.get(lease.task.id)))
        return TaskExecution((f"out-{lease.task.id}",))

    worker = _worker(settings, store, {TaskStage.EXTRACT_ENTITY_WINDOW}, execute)
    iterations: list[WorkerIteration] = []

    completed = worker.run_until_stopped(lambda: len(store.completed) == 4, iterations.append)

    assert completed == 4
    assert sorted(store.completed) == sorted(store.claimed) == [f"task-{i}" for i in range(4)]
    assert len(set(threads.values())) == 4
    for task_id in store.claimed:
        assert set(store.heartbeats[task_id]) == {"worker-1"}
    assert {item.task_id for item in iterations if item.claimed} == set(store.claimed)
    assert all(item.succeeded for item in iterations if item.claimed)


def test_a_failing_slot_leaves_the_other_slots_to_finish(tmp_path: Path) -> None:
    """One task's error fails that task alone; its neighbours complete as usual."""

    settings = _settings(tmp_path, slots=3)
    store = _SlotStore(_tasks(TaskStage.EXTRACT_RELATION_WINDOW, 3))
    together = threading.Barrier(3, timeout=_WAIT_SECONDS)

    def execute(lease: TaskLease) -> TaskExecution:
        together.wait()
        if lease.task.id == "task-1":
            raise ValueError("the model answered with nothing usable")
        return TaskExecution((f"out-{lease.task.id}",))

    worker = _worker(settings, store, {TaskStage.EXTRACT_RELATION_WINDOW}, execute)
    iterations: list[WorkerIteration] = []

    worker.run_until_stopped(
        lambda: len(store.completed) + len(store.failed) == 3, iterations.append
    )

    assert sorted(store.completed) == ["task-0", "task-2"]
    assert store.failed == {
        "task-1": {
            "error_type": "ValueError",
            "error_message": "the model answered with nothing usable",
        }
    }
    (failure,) = [item for item in iterations if item.succeeded is False]
    assert failure.task_id == "task-1"


def test_a_slot_that_cannot_record_its_outcome_leaves_the_others_running(
    tmp_path: Path,
) -> None:
    """An outage while one slot completes its task is reported, not raised.

    The task's lease is left to lapse, so another worker claims it again;
    the claiming loop and the other slot carry on.
    """

    settings = _settings(tmp_path, slots=2)
    store = _SlotStore(_tasks(TaskStage.EXTRACT_ENTITY_WINDOW, 2))
    complete = store.complete_task

    def complete_task(task_id: str, *args: Any, **kwargs: Any) -> None:
        if task_id == "task-0":
            raise TaskStoreUnavailableError("PostgreSQL coordination store is unavailable")
        complete(task_id, *args, **kwargs)

    store.complete_task = complete_task  # type: ignore[method-assign]
    fail = store.fail_task

    def fail_task(task_id: str, *args: Any, **kwargs: Any) -> None:
        if task_id == "task-0":
            raise TaskStoreUnavailableError("PostgreSQL coordination store is unavailable")
        fail(task_id, *args, **kwargs)

    store.fail_task = fail_task  # type: ignore[method-assign]
    iterations: list[WorkerIteration] = []
    worker = _worker(
        settings,
        store,
        {TaskStage.EXTRACT_ENTITY_WINDOW},
        lambda lease: TaskExecution((lease.task.id,)),
    )

    worker.run_until_stopped(
        lambda: len([item for item in iterations if item.claimed]) == 2, iterations.append
    )

    assert store.completed == ["task-1"]
    (outage,) = [item for item in iterations if item.task_id == "task-0"]
    assert outage.succeeded is False
    assert outage.error is not None
    assert outage.error["error_type"] == "TaskStoreUnavailableError"


def test_a_stop_request_claims_nothing_more_and_lets_every_slot_finish(
    tmp_path: Path,
) -> None:
    """A graceful shutdown drains: in-flight tasks complete, queued ones stay queued."""

    settings = _settings(tmp_path, slots=3)
    store = _SlotStore(_tasks(TaskStage.EXTRACT_ENTITY_WINDOW, 5))
    together = threading.Barrier(4, timeout=_WAIT_SECONDS)
    release = threading.Event()
    stopping = threading.Event()

    def execute(lease: TaskLease) -> TaskExecution:
        together.wait()
        assert release.wait(_WAIT_SECONDS)
        return TaskExecution((lease.task.id,))

    worker = _worker(settings, store, {TaskStage.EXTRACT_ENTITY_WINDOW}, execute)
    result: list[int] = []
    runner = threading.Thread(
        target=lambda: result.append(worker.run_until_stopped(stopping.is_set)),
        daemon=True,
    )
    runner.start()
    together.wait()
    stopping.set()
    sleep(0.2)

    assert runner.is_alive()
    assert store.completed == []
    release.set()
    runner.join(_WAIT_SECONDS)

    assert not runner.is_alive()
    assert result == [3]
    assert sorted(store.completed) == sorted(store.claimed)
    assert len(store.claimed) == 3
    assert [task.id for task in store.ready] == [
        task_id
        for task_id in ("task-0", "task-1", "task-2", "task-3", "task-4")
        if task_id not in store.claimed
    ]


def test_a_process_runs_one_finalization_at_a_time(tmp_path: Path) -> None:
    """A second finalization waits for the first while other slots keep extracting."""

    settings = _settings(tmp_path, slots=4)
    store = _SlotStore(
        [
            *_tasks(TaskStage.FINALIZE_GRAPH, 2, prefix="finalize"),
            *_tasks(TaskStage.EXTRACT_ENTITY_WINDOW, 2, prefix="window"),
        ]
    )
    release = threading.Event()
    running_finalizations = 0
    most_at_once = 0
    lock = threading.Lock()

    def execute(lease: TaskLease) -> TaskExecution:
        nonlocal running_finalizations, most_at_once
        if lease.task.stage != TaskStage.FINALIZE_GRAPH:
            return TaskExecution((lease.task.id,))
        with lock:
            running_finalizations += 1
            most_at_once = max(most_at_once, running_finalizations)
        assert release.wait(_WAIT_SECONDS)
        with lock:
            running_finalizations -= 1
        return TaskExecution((lease.task.id,))

    worker = _worker(
        settings,
        store,
        {TaskStage.FINALIZE_GRAPH, TaskStage.EXTRACT_ENTITY_WINDOW},
        execute,
    )
    runner = threading.Thread(
        target=lambda: worker.run_until_stopped(lambda: len(store.completed) == 4),
        daemon=True,
    )
    runner.start()
    _until(
        lambda: {"window-0", "window-1"} <= set(store.completed) and store.publication_claims > 0
    )

    assert store.claimed.count("finalize-1") == 0
    assert TaskStage.FINALIZE_GRAPH not in store.claimed_stages[-1]
    release.set()
    runner.join(_WAIT_SECONDS)

    assert not runner.is_alive()
    assert sorted(store.completed) == ["finalize-0", "finalize-1", "window-0", "window-1"]
    assert most_at_once == 1


def test_publications_drain_beside_the_task_slots(tmp_path: Path) -> None:
    """A slow graph publication never takes a slot: tasks complete while it runs."""

    settings = _settings(tmp_path, slots=2)
    store = _SlotStore(_tasks(TaskStage.EXTRACT_ENTITY_WINDOW, 2))
    worker = _worker(
        settings,
        store,
        {TaskStage.EXTRACT_ENTITY_WINDOW},
        lambda lease: TaskExecution((lease.task.id,)),
    )
    worker.manifest_publisher = cast(Any, object())
    published = threading.Event()
    completed_during_publication: list[str] = []

    def publish_one() -> WorkerIteration | None:
        if published.is_set():
            return None
        _until(lambda: len(store.completed) == 2)
        completed_during_publication.extend(store.completed)
        published.set()
        return WorkerIteration(claimed=True, task_id="publication-1", succeeded=True)

    worker._process_one_publication = publish_one  # type: ignore[method-assign]

    deadline = monotonic() + _WAIT_SECONDS
    completed = worker.run_until_stopped(lambda: published.is_set() or monotonic() > deadline)

    assert published.is_set(), "the configured publisher never drained its command"
    assert sorted(completed_during_publication) == ["task-0", "task-1"]
    assert completed == 3


def test_the_abandoned_task_sweep_runs_once_per_lease_period(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claims do not each scan for abandoned tasks; a worker sweeps on a schedule.

    A sweep the store could not run is owed, and runs at the next claim.
    """

    settings = _settings(tmp_path, slots=2, lease_seconds=60)
    store = _SlotStore([])
    worker = _worker(
        settings,
        store,
        {TaskStage.EXTRACT_ENTITY_WINDOW},
        lambda lease: TaskExecution(()),
    )
    now = [1000.0]
    monkeypatch.setattr(distributed_worker, "monotonic", lambda: now[0])

    for _ in range(5):
        assert worker.process_one().claimed is False
    assert store.sweeps == 1
    assert store.publication_claims == 0

    now[0] += 59
    worker.process_one()
    assert store.sweeps == 1
    now[0] += 1
    worker.process_one()
    assert store.sweeps == 2

    def unavailable() -> None:
        raise TaskStoreUnavailableError("PostgreSQL coordination store is unavailable")

    store.fail_abandoned_tasks = unavailable  # type: ignore[method-assign]
    now[0] += 60
    with pytest.raises(TaskStoreUnavailableError):
        worker.process_one()
    del store.fail_abandoned_tasks
    worker.process_one()
    assert store.sweeps == 3


def test_the_coordination_pool_grows_with_the_slots(tmp_path: Path) -> None:
    """Two connections serve a single slot; every four more slots add one."""

    assert [worker_pool_connections(slots) for slots in (1, 2, 4, 8, 16)] == [2, 2, 3, 4, 6]
    store = build_distributed_store(_settings(tmp_path, slots=8))
    try:
        assert store._pool.max_size == 4
    finally:
        store.close()
