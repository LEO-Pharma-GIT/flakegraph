# SPDX-License-Identifier: Apache-2.0
"""Durable task-store port used by distributed coordinators and workers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import Any, Protocol

from kg_processor.domain.consumption import ConsumptionEvent
from kg_processor.domain.distributed import (
    PublicationLease,
    RunDefinition,
    RunOverview,
    RunSnapshot,
    RunSummary,
    TaskDefinition,
    TaskLease,
    TaskProgress,
    TaskSnapshot,
    TaskStage,
)


class TaskStoreUnavailableError(RuntimeError):
    """Signal a transient coordination-store connectivity failure.

    Application workers use this provider-neutral exception to distinguish an
    unavailable queue from a malformed task or deterministic pipeline error.
    Long-running workers may therefore back off and reconnect without importing
    PostgreSQL-specific exception classes or restarting their containers.
    """


class TaskStore(Protocol):
    """Coordinate dependency-aware tasks through leases and atomic transitions.

    Publication commands are an outbox the store owns. ``publish_claimed`` must
    not hold a store lock across the callback: external delivery can outlast any
    lease, and blocking cancellation and lease reclamation behind it would make
    an unresponsive destination stall the run. A stale finalizer is fenced
    optimistically instead: the store validates ownership before the callback
    and publishes the lease's generation and attempt with it, then acknowledges
    completion only when the row still carries that same generation, attempt,
    owner, and live lease under a still-running run. A superseded or reclaimed
    finalizer therefore fails to acknowledge, and the destination uses the
    published generation to reject its late write.
    """

    def initialize(self) -> None:
        """Create the durable coordination structures, or check the ones already there."""
        ...

    def record_consumption(self, run_id: str, event: ConsumptionEvent) -> None:
        """Persist one billed or local provider event idempotently by event ID."""
        ...

    def consumption_events(self, run_id: str) -> list[ConsumptionEvent]:
        """Read consumption across all attempts and workers serving one run."""
        ...

    def create_run(self, run: RunDefinition) -> None:
        """Persist a planning run while rejecting conflicting run identities."""
        ...

    def add_tasks(self, run_id: str, tasks: list[TaskDefinition]) -> None:
        """Persist a complete validated task graph for a planning run."""
        ...

    def activate_run(self, run_id: str) -> None:
        """Make a fully planned run available to workers."""
        ...

    def claim_task(
        self,
        worker_id: str,
        stages: set[TaskStage],
        lease_duration: timedelta,
        expected_config_digest: str | None = None,
        skip_dependency_outputs_for_stages: set[TaskStage] | None = None,
    ) -> TaskLease | None:
        """Atomically claim a compatible highest-priority task for this worker.

        Stores must apply the optional configuration digest before incrementing a
        task attempt so a stale worker cannot exhaust healthy work's retry budget.
        Stages in ``skip_dependency_outputs_for_stages`` still wait for every
        dependency but omit their output IDs from the lease. This supports engines
        such as Spark that scan immutable stage prefixes directly.
        """
        ...

    def fail_abandoned_tasks(self) -> None:
        """Fail running tasks whose lease lapsed after their last attempt.

        Such a task can never be claimed again, so without this its run waits
        forever. The sweep scans every running task, which is why workers run
        it on a schedule of their own rather than with every claim.
        """
        ...

    def record_served_configuration(self, stages: set[TaskStage], config_digest: str) -> None:
        """Record which configuration this fleet serves, for each stage it claims.

        A worker only claims a run carrying its own digest, so demand for a run
        nobody serves can never be met. Recording this lets the autoscaling signal
        ask for workers only where a worker could actually take the work.
        """
        ...

    def heartbeat(
        self, task_id: str, worker_id: str, lease_duration: timedelta, *, claim_token: str
    ) -> None:
        """Extend a running task lease only when the caller still owns it."""
        ...

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
        """Commit outputs and optional dynamically fanned-out work atomically.

        Follow-up tasks become dependencies of ``barrier_task_id`` in the same
        transaction that completes their parent. This prevents the barrier from
        racing ahead between preparation and extraction-window publication.
        """
        ...

    def fail_task(
        self,
        task_id: str,
        worker_id: str,
        error: dict[str, Any],
        retry_delay: timedelta,
        *,
        claim_token: str,
    ) -> None:
        """Record an error and either retry or terminally fail the owning run."""
        ...

    def cancel_run(self, run_id: str) -> None:
        """Cancel unfinished work while preserving completed artifacts and audit data."""
        ...

    def retry_run(self, run_id: str, skipped_file_ids: list[str] | None = None) -> None:
        """Retry stopped work, atomically excluding explicitly quarantined failed files."""
        ...

    def get_run(self, run_id: str) -> RunSnapshot:
        """Return one consistent run and task snapshot or raise when unknown."""
        ...

    def get_stage_tasks(self, run_id: str, stage: TaskStage) -> list[TaskSnapshot]:
        """Return one stage's tasks of a run, or raise when the run is unknown.

        A planner reading a run's finalizer must not load the run's every task
        to find it.
        """
        ...

    def get_run_summary(self, run_id: str) -> RunSummary:
        """Return bounded stage/status counts without loading every task payload."""
        ...

    def list_runs(self, limit: int = 100) -> list[RunOverview]:
        """Return recent bounded run overviews without configuration or task payloads."""
        ...

    def add_initial_tasks(self, run_id: str, tasks: Iterable[TaskDefinition]) -> None:
        """Validate and persist a streamed initial document plan atomically."""
        ...

    def report_task_progress(
        self,
        task_id: str,
        worker_id: str,
        progress: TaskProgress,
        *,
        claim_token: str,
    ) -> None:
        """Replace progress only while the caller owns the running task lease."""
        ...

    def claim_publication(
        self,
        worker_id: str,
        lease_duration: timedelta,
    ) -> PublicationLease | None:
        """Lease one queued or expired publication command."""
        ...

    def publish_claimed(
        self,
        publication_id: str,
        worker_id: str,
        publish: Callable[[PublicationLease], None],
    ) -> None:
        """Run publication outside any store lock and acknowledge it under its fence."""
        ...

    def heartbeat_publication(
        self,
        publication_id: str,
        worker_id: str,
        lease_duration: timedelta,
    ) -> None:
        """Extend an active publication lease while external delivery is in progress."""
        ...

    def commit_publication(self, lease: PublicationLease, commit: Callable[[], None]) -> None:
        """Commit an already staged batch under ownership and graph-head locks.

        Expensive parsing, serialization, uploads and COPY happen before this
        call. Only the final destination replacement runs while locked, making
        cancellation and a superseding graph version serialize with that swap.
        """
        ...

    def fail_publication(
        self,
        publication_id: str,
        worker_id: str,
        error: dict[str, Any],
        retry_delay: timedelta,
    ) -> None:
        """Record a failed delivery attempt and requeue it within its budget."""
        ...
