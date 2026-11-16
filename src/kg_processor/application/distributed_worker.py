# SPDX-License-Identifier: Apache-2.0
"""Lease-driven worker service for provider-neutral distributed graph stages."""

from __future__ import annotations

import logging
import tempfile
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from kg_processor.application.consumption import ConsumptionCollector
from kg_processor.application.content_kinds import (
    dataset_profiles,
    dataset_windows,
    profile_columns,
)
from kg_processor.application.distributed_planner import distributed_processing_config_digest
from kg_processor.application.extraction_windows import (
    build_document_openings,
    build_extraction_windows,
    window_skip_reason,
)
from kg_processor.application.failed_documents import is_failed_document_event
from kg_processor.application.lease_heartbeat import LeaseHeartbeat, heartbeat_interval_seconds
from kg_processor.application.ontology import load_ontology
from kg_processor.application.progress import error_metadata
from kg_processor.application.revalidation import (
    ENTITY_WINDOW_STAGES,
    RELATION_PHASE_STAGES,
    REVALIDATION_STAGE,
    conclude_revalidation,
    revalidated_inventory,
)
from kg_processor.application.spark_finalization import (
    FINALIZATION_PHASES,
    SparkFinalizationRequest,
    SparkGraphFinalizer,
    finalization_progress,
)
from kg_processor.application.two_pass_extraction import apply_entity_verdicts
from kg_processor.application.window_errors import is_systemic_provider_error
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import (
    PROVENANCE_EXTRACTED,
    ArtifactKind,
    ArtifactRef,
    InheritedDocuments,
    PublicationLease,
    StoredArtifact,
    TaskDefinition,
    TaskLease,
    TaskProgress,
    TaskStage,
    document_provenance,
)
from kg_processor.domain.documents import InputFile
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionObservations,
    RelationObservation,
)
from kg_processor.domain.finalization import GraphDatasetManifest
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.domain.ids import stable_id
from kg_processor.domain.ontology import OntologyProfile
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
    RevalidationWindowShard,
)
from kg_processor.ports.artifact_store import ArtifactStore
from kg_processor.ports.distributed_pipeline import DistributedPipeline
from kg_processor.ports.graph_manifest_publisher import GraphManifestPublisher
from kg_processor.ports.task_store import TaskStore, TaskStoreUnavailableError

_JSON_MEDIA_TYPE = "application/json"
_GRAPH_MANIFEST_MEDIA_TYPE = "application/vnd.flakegraph.graph-manifest+json"
# Stages a process runs one at a time, whatever its slot count.
_EXCLUSIVE_STAGES = frozenset({TaskStage.FINALIZE_GRAPH})
# The payload keys a document's revalidation carries from its first task to
# its last: what the first task settled for the whole document, and its windows.
_REVALIDATION_ARTIFACT = "revalidation_artifact_id"
_REVALIDATION_WINDOWS = "revalidation_windows"


@dataclass(frozen=True)
class WorkerIteration:
    """Summarize one polling iteration for CLI logging and focused tests."""

    claimed: bool
    task_id: str | None = None
    stage: TaskStage | None = None
    succeeded: bool | None = None
    output_artifact_ids: tuple[str, ...] = ()
    error: dict[str, Any] | None = None


@dataclass(frozen=True)
class TaskExecution:
    """Carry stage outputs and transactional queue fan-out to task completion.

    Preparation, context extraction, the start of a revalidation, and entity
    compaction may create follow-up work. Keeping the completion command explicit
    lets the task store publish each complete dynamic subgraph atomically without
    stage implementations mutating coordination tables.
    """

    output_artifact_ids: tuple[str, ...]
    follow_up_tasks: tuple[TaskDefinition, ...] = ()
    barrier_task_id: str | None = None


logger = logging.getLogger(__name__)


_RUN_CONTEXT_CACHE_SIZE = 64


@dataclass(frozen=True)
class _RunContext:
    """What one run executes with: its settings and the pipeline running them."""

    settings: Settings
    pipeline: DistributedPipeline


def _run_ontology(settings: Settings) -> OntologyProfile:
    """The ontology profile a run's settings name, as its pipeline loads it."""

    return load_ontology(
        settings.ontology.profile_path,
        settings.graph.entity_types,
        settings.graph.relation_types,
        inline=settings.ontology.profile,
    ).profile


def settings_for_run(
    base: Settings,
    run_config: dict[str, Any],
    *,
    graph_id: str | None = None,
    run_id: str | None = None,
) -> Settings:
    """Apply a run's authoritative graph identity and stored ontology.

    Explicit vocabularies are resolved by the planner and carried inline. A
    submission that selects none uses the worker's mounted profile. The durable
    run ID is also the job identity used by reports and accounting.
    """

    identity = graph_id if graph_id is not None else run_config.get("graph_id")
    if isinstance(identity, str) and identity != base.job.graph_id:
        base = base.model_copy(update={"job": base.job.model_copy(update={"graph_id": identity})})
    if run_id is not None and run_id != base.job.job_id:
        base = base.model_copy(update={"job": base.job.model_copy(update={"job_id": run_id})})
    ontology = run_config.get("ontology")
    profile = ontology.get("profile") if isinstance(ontology, dict) else None
    if not isinstance(profile, dict):
        return base
    if base.ontology.profile == profile and base.ontology.profile_path is None:
        return base
    return base.model_copy(
        update={
            "ontology": base.ontology.model_copy(update={"profile": profile, "profile_path": None})
        }
    )


class DistributedWorker:
    """Execute eligible durable tasks in a bounded number of slots in one process.

    A pipeline task spends most of its life waiting on a model, a parser or
    object storage, so one process keeps ``distributed.worker_slots`` tasks in
    flight. Each slot owns one task: it holds that task's lease, renews it on a
    heartbeat of its own, and completes or fails it independently of the other
    slots. Horizontal scale beyond one process comes from more processes or
    Kubernetes pods. PostgreSQL row leases provide at-least-once recovery after
    process or node failure.
    """

    def __init__(
        self,
        settings: Settings,
        worker_id: str,
        stages: set[TaskStage],
        pipeline: DistributedPipeline,
        task_store: TaskStore,
        artifact_store: ArtifactStore,
        manifest_publisher: GraphManifestPublisher | None = None,
        pipeline_for: Callable[[Settings], DistributedPipeline] | None = None,
    ) -> None:
        """Validate worker identity and retain its injected application dependencies.

        ``pipeline_for`` builds the pipeline that executes one run's settings -
        the worker's own with that run's ontology applied - over the worker's
        adapters. Without it every run executes on ``pipeline`` as configured,
        which is what a test double wants. Slots share the pipeline, the stores
        and the publisher, so each of those must be safe to call from several
        threads at once.
        """

        if not worker_id.strip():
            raise ValueError("distributed worker id must not be blank")
        if not stages:
            raise ValueError("distributed worker requires at least one eligible stage")
        self.settings = settings
        self.worker_id = worker_id
        self.stages = stages
        self.slots = settings.distributed.worker_slots
        self.pipeline = pipeline
        self.task_store = task_store
        self.artifact_store = artifact_store
        self.manifest_publisher = manifest_publisher
        self.lease_duration = timedelta(seconds=settings.distributed.lease_seconds)
        self.retry_delay = timedelta(seconds=settings.distributed.retry_delay_seconds)
        self._config_digest = distributed_processing_config_digest(settings)
        self._declared_configuration = False
        self._pipeline_for = pipeline_for
        self._run_contexts: dict[str, _RunContext] = {}
        # The declaration, the sweep schedule and the run-context cache are
        # reached from every slot and from the claiming loop.
        self._lock = threading.Lock()
        self._next_sweep_at: float | None = None

    def _declare_served_configuration(self) -> None:
        """Record what this fleet serves, once per process.

        Only a run carrying this digest can be claimed here, so the autoscaling
        signal needs to know the digest in order to stop asking for workers to do
        work no worker is able to take. Declared once rather than per poll: it
        cannot change without restarting the process.
        """

        with self._lock:
            if self._declared_configuration:
                return
            self._declared_configuration = True
        try:
            self.task_store.record_served_configuration(self.stages, self._config_digest)
        except Exception:
            # Losing this costs an accurate demand signal, not the ability to
            # work, so a worker that cannot record it still claims tasks.
            logger.warning("Could not record the served configuration", exc_info=True)

    def _sweep_abandoned_tasks(self) -> None:
        """Fail tasks whose last attempt was abandoned, at most once per lease period.

        The sweep scans every running task in the store. Run with each claim,
        it would cost a fleet of multi-slot workers one such scan per task
        started; a lapsed last attempt can wait a lease period to be failed,
        because the store only fails it once its lease has been gone that long.
        """

        now = monotonic()
        with self._lock:
            if self._next_sweep_at is not None and now < self._next_sweep_at:
                return
            self._next_sweep_at = now + self.settings.distributed.lease_seconds
        try:
            self.task_store.fail_abandoned_tasks()
        except BaseException:
            # A sweep the store could not run is owed at the next claim.
            with self._lock:
                self._next_sweep_at = None
            raise

    def _claim(self, stages: set[TaskStage]) -> TaskLease | None:
        """Claim one ready task of ``stages`` for this worker, or None when none is ready."""

        self._declare_served_configuration()
        self._sweep_abandoned_tasks()
        return self.task_store.claim_task(
            self.worker_id,
            stages,
            self.lease_duration,
            self._config_digest,
            ({TaskStage.FINALIZE_GRAPH} if _use_spark_finalization(self.settings) else None),
        )

    def process_one(self) -> WorkerIteration:
        """Claim and execute at most one task, returning immediately when none is ready.

        The task runs in the calling thread; ``run_until_stopped`` is what
        keeps several in flight.
        """

        publication = self._process_one_publication()
        if publication is not None:
            return publication
        lease = self._claim(self.stages)
        if lease is None:
            return WorkerIteration(claimed=False)
        return self._run_claimed(lease)

    def _run_claimed(self, lease: TaskLease) -> WorkerIteration:
        """Execute one claimed task under its own lease heartbeat, then complete or fail it."""

        try:
            with LeaseHeartbeat(
                lambda: self.task_store.heartbeat(
                    lease.task.id,
                    self.worker_id,
                    self.lease_duration,
                    claim_token=lease.claim_token,
                ),
                heartbeat_interval_seconds(self.settings.distributed.lease_seconds),
            ) as heartbeat:
                execution = self._execute(lease)
            heartbeat.raise_if_unhealthy()
            output_ids = list(execution.output_artifact_ids)
            self.task_store.complete_task(
                lease.task.id,
                self.worker_id,
                output_ids,
                list(execution.follow_up_tasks),
                execution.barrier_task_id,
                claim_token=lease.claim_token,
            )
            return WorkerIteration(
                claimed=True,
                task_id=lease.task.id,
                stage=lease.task.stage,
                succeeded=True,
                output_artifact_ids=tuple(output_ids),
            )
        except Exception as exc:
            error = error_metadata(exc)
            try:
                self.task_store.fail_task(
                    lease.task.id,
                    self.worker_id,
                    error,
                    self.retry_delay,
                    claim_token=lease.claim_token,
                )
            except TaskStoreUnavailableError:
                # The caller owns coordination outages and retries after
                # backoff without pretending a durable failure was recorded.
                raise
            except Exception as transition_error:
                # Lost ownership means another worker or lease expiry already
                # controls the durable task state. Surface it in this iteration,
                # but keep the process alive rather than crashing the worker pod.
                error = {
                    **error,
                    "transition_error_type": type(transition_error).__name__,
                }
            return WorkerIteration(
                claimed=True,
                task_id=lease.task.id,
                stage=lease.task.stage,
                succeeded=False,
                error=error,
            )

    def _run_slot(self, lease: TaskLease) -> WorkerIteration:
        """Run one claimed task in a slot thread, reporting an outage rather than raising it.

        A slot that cannot record its task's outcome leaves the lease to lapse
        and the task to be claimed again once it has; the other slots and the
        claiming loop carry on.
        """

        try:
            return self._run_claimed(lease)
        except TaskStoreUnavailableError as exc:
            return WorkerIteration(
                claimed=True,
                task_id=lease.task.id,
                stage=lease.task.stage,
                succeeded=False,
                error=error_metadata(exc),
            )

    def _process_one_publication(self) -> WorkerIteration | None:
        """Deliver one durable destination command, or return None when none is queued."""

        if not self._publication_enabled():
            return None
        store = self.task_store
        lease = store.claim_publication(self.worker_id, self.lease_duration)
        if lease is None:
            return None
        try:
            with LeaseHeartbeat(
                lambda: store.heartbeat_publication(
                    lease.id,
                    self.worker_id,
                    self.lease_duration,
                ),
                heartbeat_interval_seconds(self.settings.distributed.lease_seconds),
            ) as heartbeat:

                def publish_with_health_check(publication: PublicationLease) -> None:
                    self._publish_manifest_lease(publication)
                    heartbeat.raise_if_unhealthy()

                stored = self.artifact_store.get(lease.artifact_id)
                if stored.ref.metadata.get("schema") == GraphWriteBatch.__name__:
                    batch = GraphWriteBatch.model_validate_json(stored.payload)
                    if self.manifest_publisher is not None:
                        staging = self.manifest_publisher.stage_batch_publication(
                            batch,
                            {
                                **lease.task_payload,
                                "publication": {
                                    "id": lease.id,
                                    "generation": lease.generation,
                                    "attempt": lease.attempt,
                                },
                            },
                        )
                    else:
                        pipeline = self._run_context(lease.run_id).pipeline
                        staging = pipeline.stage_graph_publication(
                            batch, lease.id, lease.generation
                        )
                    with staging as commit:
                        heartbeat.raise_if_unhealthy()
                        store.commit_publication(lease, commit)
                else:
                    store.publish_claimed(lease.id, self.worker_id, publish_with_health_check)
            return WorkerIteration(
                claimed=True,
                task_id=lease.id,
                succeeded=True,
                output_artifact_ids=(lease.artifact_id,),
            )
        except Exception as exc:
            error = error_metadata(exc)
            try:
                store.fail_publication(
                    lease.id,
                    self.worker_id,
                    error,
                    self.retry_delay,
                )
            except TaskStoreUnavailableError:
                raise
            except Exception as transition_error:
                error = {
                    **error,
                    "transition_error_type": type(transition_error).__name__,
                }
            return WorkerIteration(
                claimed=True,
                task_id=lease.id,
                succeeded=False,
                error=error,
            )

    def _publish_manifest_lease(self, lease: PublicationLease) -> None:
        """Load and publish the immutable manifest selected by the fenced outbox row."""

        if self.manifest_publisher is None:
            raise RuntimeError("publication lease requires a graph manifest publisher")
        stored = self.artifact_store.get(lease.artifact_id)
        if stored.ref.kind != ArtifactKind.GRAPH_RESULT:
            raise ValueError("publication artifact is not a finalized graph result")
        manifest = GraphDatasetManifest.model_validate_json(stored.payload)
        payload = {
            **lease.task_payload,
            "publication": {
                "id": lease.id,
                "generation": lease.generation,
                "attempt": lease.attempt,
            },
        }
        self.manifest_publisher.publish(manifest, payload)

    def run_until_stopped(
        self,
        stop_requested: Callable[[], bool],
        on_iteration: Callable[[WorkerIteration], None] | None = None,
    ) -> int:
        """Keep up to ``slots`` tasks in flight until the caller requests a graceful shutdown.

        One loop claims: whenever a slot is free it claims a task and hands it
        to that slot, and when nothing is ready it waits a poll interval. Graph
        publications drain on a thread of their own, so a slow destination
        never holds a task slot. Once a stop is requested nothing more is
        claimed and every task in flight runs to completion before this
        returns. Returning a count makes shutdown observable without coupling
        this service to signals, Typer, Kubernetes, or a particular logging
        implementation. ``on_iteration`` is called one call at a time, from
        whichever thread finished the iteration.
        """

        report_lock = threading.Lock()
        completed = 0

        def report(iteration: WorkerIteration) -> None:
            nonlocal completed
            with report_lock:
                if iteration.claimed:
                    completed += 1
                if on_iteration is not None:
                    on_iteration(iteration)

        publisher = (
            threading.Thread(
                target=self._drain_publications,
                args=(stop_requested, report),
                name="flakegraph-publisher",
                daemon=True,
            )
            if self._publication_enabled()
            else None
        )
        if publisher is not None:
            publisher.start()
        in_flight: dict[Future[WorkerIteration], TaskStage] = {}
        with ThreadPoolExecutor(
            max_workers=self.slots, thread_name_prefix="flakegraph-slot"
        ) as executor:
            try:
                while not stop_requested():
                    for future in [future for future in in_flight if future.done()]:
                        del in_flight[future]
                        report(future.result())
                    stages = self._claimable_stages(in_flight.values())
                    if not stages:
                        wait(
                            in_flight,
                            timeout=self.settings.distributed.poll_interval_seconds,
                            return_when=FIRST_COMPLETED,
                        )
                        continue
                    try:
                        lease = self._claim(stages)
                    except TaskStoreUnavailableError as exc:
                        # No durable transition can be trusted while coordination
                        # is unavailable. Keep the pod alive, surface the outage
                        # through normal progress output, and reconnect after
                        # bounded backoff; slots already running carry on.
                        report(
                            WorkerIteration(
                                claimed=False,
                                succeeded=False,
                                error=error_metadata(exc),
                            )
                        )
                        sleep(self.settings.distributed.retry_delay_seconds)
                        continue
                    if lease is None:
                        report(WorkerIteration(claimed=False))
                        sleep(self.settings.distributed.poll_interval_seconds)
                        continue
                    in_flight[executor.submit(self._run_slot, lease)] = lease.task.stage
            finally:
                # Draining: every claimed task finishes and reports, whether the
                # loop ended on a stop request or on an error.
                for future in as_completed(in_flight):
                    report(future.result())
        if publisher is not None:
            publisher.join()
        return completed

    def _publication_enabled(self) -> bool:
        """Serve configured publication adapters and local-finalizer batch outboxes."""

        return self.manifest_publisher is not None or TaskStage.FINALIZE_GRAPH in self.stages

    def _claimable_stages(self, running: Iterable[TaskStage]) -> set[TaskStage]:
        """The stages a free slot may claim, given the stages already running here.

        Empty when every slot is busy. A finalization drives the process's one
        Spark application, or holds the whole corpus in memory, so a process
        runs at most one at a time whatever its slot count.
        """

        busy = list(running)
        if len(busy) >= self.slots:
            return set()
        return self.stages - (_EXCLUSIVE_STAGES & set(busy))

    def _drain_publications(
        self,
        stop_requested: Callable[[], bool],
        report: Callable[[WorkerIteration], None],
    ) -> None:
        """Deliver queued graph publications one at a time until a stop is requested."""

        while not stop_requested():
            try:
                iteration = self._process_one_publication()
            except TaskStoreUnavailableError as exc:
                report(WorkerIteration(claimed=False, succeeded=False, error=error_metadata(exc)))
                sleep(self.settings.distributed.retry_delay_seconds)
                continue
            if iteration is None:
                sleep(self.settings.distributed.poll_interval_seconds)
                continue
            report(iteration)

    def _execute(self, lease: TaskLease) -> TaskExecution:  # noqa: PLR0911, PLR0912
        """Dispatch one validated lease to its stage-specific implementation."""

        run = self._run_context(lease.task.run_id)
        task_pipeline = getattr(run.pipeline, "for_task", None)
        if task_pipeline is not None:
            run = _RunContext(
                settings=run.settings, pipeline=task_pipeline(lease.task.id, lease.attempt)
            )
        if lease.task.stage == TaskStage.PREPARE_DOCUMENT:
            return self._prepare_document(lease, run)
        if lease.task.stage == TaskStage.EXTRACT_DOCUMENT_CONTEXT:
            return self._extract_document_context(lease, run)
        if lease.task.stage == TaskStage.EXTRACT_ENTITY_WINDOW:
            return TaskExecution((self._extract_entity_window(lease, run),))
        if lease.task.stage == TaskStage.COMPACT_ENTITY_INVENTORY:
            return self._compact_entity_inventory(lease)
        if lease.task.stage == TaskStage.EXTRACT_RELATION_WINDOW:
            return TaskExecution((self._extract_relation_window(lease, run),))
        if lease.task.stage == TaskStage.COMPACT_DOCUMENT:
            return TaskExecution((self._compact_document(lease, run),))
        if lease.task.stage == TaskStage.REVALIDATE_DOCUMENT_CONTEXT:
            return self._revalidate_document_context(lease, run)
        if lease.task.stage == TaskStage.REVALIDATE_ENTITY_WINDOW:
            return TaskExecution((self._revalidate_entity_window(lease, run),))
        if lease.task.stage == TaskStage.REVALIDATE_RELATION_WINDOW:
            return TaskExecution((self._revalidate_relation_window(lease, run),))
        if lease.task.stage == TaskStage.REVALIDATE_DOCUMENT:
            return TaskExecution(self._revalidate_document(lease, run))
        if lease.task.stage == TaskStage.FINALIZE_GRAPH:
            return TaskExecution((self._finalize_graph(lease, run),))
        raise ValueError(f"unsupported distributed task stage: {lease.task.stage}")

    def _run_context(self, run_id: str) -> _RunContext:
        """The settings and pipeline one run executes with, resolved once per run.

        A run carries its ontology in its stored configuration. The worker's
        own profile is the default for a run that names none; a run that does
        is executed with its own, so two graphs built by one fleet can extract
        different vocabularies. Bounded: a worker sees a handful of runs.
        """

        with self._lock:
            cached = self._run_contexts.get(run_id)
        if cached is not None:
            return cached
        definition = self.task_store.get_run_summary(run_id).run
        run_settings = settings_for_run(
            self.settings, definition.config, graph_id=definition.graph_id, run_id=definition.id
        )
        pipeline = self.pipeline
        configure = self._pipeline_for or getattr(self.pipeline, "for_settings", None)
        if configure is not None:
            pipeline = configure(run_settings)
        elif run_settings.job.graph_id != self.settings.job.graph_id:
            raise ValueError("a worker serving another graph requires a configurable pipeline")
        with self._lock:
            # Two slots can resolve one run at once; the first to finish is
            # kept, so every slot of a run executes on the same pipeline.
            cached = self._run_contexts.get(run_id)
            if cached is not None:
                return cached
            # Only finalization reads the other workers' usage, and it merges
            # the run's ledger itself, so a worker never loads it here.
            collector = getattr(pipeline, "consumption", None)
            record = getattr(self.task_store, "record_consumption", None)
            if isinstance(collector, ConsumptionCollector) and record is not None:
                collector.bind_sink(lambda event: record(run_id, event))
            context = _RunContext(settings=run_settings, pipeline=pipeline)
            if len(self._run_contexts) >= _RUN_CONTEXT_CACHE_SIZE:
                self._run_contexts.pop(next(iter(self._run_contexts)))
            self._run_contexts[run_id] = context
            return context

    def _prepare_document(self, lease: TaskLease, run: _RunContext) -> TaskExecution:
        """Run OCR/chunking and fan its discovered windows into the shared queue.

        The returned follow-up definitions are committed with this task's success.
        Consequently, finalization can never observe a completed preparation task
        without also waiting for every extraction window it produced.

        A document that cannot be read fails the task like any other error until
        its last attempt, so a parser that was briefly down costs an attempt, not
        the document. The last attempt records it instead: the task succeeds
        with the failure in its trace and no windows, the run goes on without
        that one file, and the finalizer reports it as a failed document.
        """

        artifact_id = lease.task.payload.get("source_artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("prepare_document task is missing source_artifact_id")
        source = self.artifact_store.get(artifact_id)
        if source.ref.kind != ArtifactKind.SOURCE_DOCUMENT:
            raise ValueError("prepare_document input is not a source document artifact")
        input_metadata = source.ref.metadata.get("input_file")
        if not isinstance(input_metadata, dict):
            raise ValueError("source document artifact is missing input_file metadata")
        filename = _safe_filename(input_metadata.get("filename"))
        with tempfile.TemporaryDirectory(prefix="flakegraph-source-") as directory:
            path = Path(directory) / filename
            path.write_bytes(source.payload)
            input_file = InputFile.model_validate({**input_metadata, "path": path})
            prepared = run.pipeline.prepare_documents(
                [input_file],
                record_failures=lease.attempt >= lease.task.max_attempts,
            )
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.PREPARED_DOCUMENT,
            prepared.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={"file_ids": prepared.file_ids, "schema": "PreparedDocumentShard"},
        )
        windows = build_extraction_windows(
            prepared.chunks,
            self.settings.graph.extraction_window_tokens,
            self.settings.graph.max_chunks_per_llm_call,
        )
        if not windows:
            failed = [
                {
                    "file_id": str(event["file_id"]),
                    "error_type": str(event.get("error_type") or ""),
                    "error_message": str(event.get("error_message") or ""),
                }
                for event in prepared.trace
                if is_failed_document_event(event)
            ]
            # A blank or image-only document can legitimately produce no chunks,
            # and one that could not be read produces none at all.
            # Persist an empty extracted-document boundary so Spark still has a
            # readable stage prefix and publishes the document provenance tables.
            empty = ExtractedDocumentShard(
                prepared=_prepared_projection_for_extracted_artifact(
                    prepared,
                    spark_finalization=_use_spark_finalization(self.settings),
                ),
                observations=ExtractionObservations(chunk_count=0, window_count=0),
                trace=prepared.trace,
            )
            empty_ref = self.artifact_store.put(
                lease.task.run_id,
                ArtifactKind.EXTRACTED_DOCUMENT,
                empty.model_dump_json().encode("utf-8"),
                _JSON_MEDIA_TYPE,
                metadata={
                    "file_ids": empty.prepared.file_ids,
                    "windows": 0,
                    "schema": ExtractedDocumentShard.__name__,
                    # What run status and the document list read while the run
                    # is still going; the graph's failed_documents table is the
                    # record once it is finalized.
                    **({"failed_documents": failed} if failed else {}),
                },
                identity_key=lease.task.scope_id,
            )
            return TaskExecution(output_artifact_ids=(ref.id, empty_ref.id))
        final_task_id = stable_id("task", lease.task.run_id, TaskStage.FINALIZE_GRAPH.value)
        context_task = TaskDefinition(
            id=stable_id(
                "task",
                lease.task.run_id,
                TaskStage.EXTRACT_DOCUMENT_CONTEXT.value,
                lease.task.scope_id,
            ),
            run_id=lease.task.run_id,
            stage=TaskStage.EXTRACT_DOCUMENT_CONTEXT,
            scope_id=lease.task.scope_id,
            dependency_ids=[lease.task.id],
            max_attempts=self.settings.distributed.max_attempts,
        )
        return TaskExecution(
            output_artifact_ids=(ref.id,),
            follow_up_tasks=(context_task,),
            barrier_task_id=final_task_id,
        )

    def _extract_document_context(self, lease: TaskLease, run: _RunContext) -> TaskExecution:
        """Extract reusable focal entities, then fan out independent body windows.

        Keeping this queue stage between OCR and window extraction makes one LLM
        context call reusable across the whole document while retaining dynamic
        work stealing for every ordinary extraction window.

        A revision that extracts a kept document again starts here, with no
        preparation task of its own: the payload names the prepared text an
        earlier run stored, and, when only the windows are to be extracted
        again, the context that run extracted, which is then reused as it is.
        """

        seeded = lease.task.payload.get("prepared_artifact_id")
        inputs = (
            [self._stored_artifact(seeded, ArtifactKind.PREPARED_DOCUMENT)]
            if isinstance(seeded, str) and seeded
            else self._dependency_artifacts(lease, ArtifactKind.PREPARED_DOCUMENT)
        )
        if len(inputs) != 1:
            raise ValueError("extract_document_context requires one prepared artifact")
        prepared = PreparedDocumentShard.model_validate_json(inputs[0].payload)
        reused_context = lease.task.payload.get("context_artifact_id")
        if isinstance(reused_context, str) and reused_context:
            stored = DocumentContextShard.model_validate_json(
                self._stored_artifact(reused_context, ArtifactKind.DOCUMENT_CONTEXT).payload
            )
            if stored.file_ids != prepared.file_ids:
                raise ValueError("reused document context and prepared document differ")
            contextualized = prepared.model_copy(
                update={
                    "document_context_entities": stored.document_context_entities,
                    "trace": stored.trace,
                }
            )
        else:
            contextualized = run.pipeline.extract_document_context(prepared)
        context = DocumentContextShard(
            file_ids=contextualized.file_ids,
            document_context_entities=contextualized.document_context_entities,
            trace=contextualized.trace,
            document_openings=build_document_openings(contextualized.chunks),
        )
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.DOCUMENT_CONTEXT,
            context.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": context.file_ids,
                "schema": DocumentContextShard.__name__,
                "document_context_entities": len(context.document_context_entities),
            },
        )
        # A profiled dataset may need no windows, or entity windows only; a
        # catalogue's relation windows read its rows by its column relations.
        profile = next(iter(dataset_profiles(contextualized.trace).values()), None)
        extract_entities, extract_relations = dataset_windows(
            profile, catalogue_relations=self.settings.graph.catalogue_relations
        )
        columns = profile_columns(profile) if extract_relations else []
        windows = [
            window
            for window in build_extraction_windows(
                contextualized.chunks,
                self.settings.graph.extraction_window_tokens,
                self.settings.graph.max_chunks_per_llm_call,
            )
            if extract_entities and window_skip_reason(window) is None
        ]
        window_payloads = [
            self._stored_window(
                lease,
                window.id,
                ExtractionWindowShard(
                    file_ids=contextualized.file_ids,
                    chunks=window.chunks,
                    extract_relations=extract_relations,
                    dataset_columns=columns,
                ),
                len(window.chunks),
            )
            for window in windows
        ]
        entity_tasks = self._window_tasks(lease, TaskStage.EXTRACT_ENTITY_WINDOW, window_payloads)
        # This lightweight barrier materializes the complete document vocabulary.
        # It then emits a second work-stealing wave for relation extraction, so
        # windows can connect entities found elsewhere without serial model work.
        inventory_task = self._document_task(
            lease,
            TaskStage.COMPACT_ENTITY_INVENTORY,
            {"prepared_artifact_id": inputs[0].ref.id, "windows": window_payloads},
            entity_tasks,
        )
        return TaskExecution(
            output_artifact_ids=(ref.id,),
            follow_up_tasks=(*entity_tasks, inventory_task),
            barrier_task_id=stable_id("task", lease.task.run_id, TaskStage.FINALIZE_GRAPH.value),
        )

    def _stored_window(
        self,
        lease: TaskLease,
        window_id: str,
        shard: ExtractionWindowShard | RevalidationWindowShard,
        chunk_count: int,
    ) -> dict[str, str]:
        """Store one window's artifact and name it for the window's task.

        Every window is a task of its own, so the fleet's slots, not threads
        inside one lease, are what extract a document's windows side by side:
        a slot that finishes early takes the next window of any document, and
        a retry repeats one window's model calls and no others.
        """

        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.EXTRACTION_WINDOW,
            shard.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": shard.file_ids,
                "window_id": window_id,
                "chunks": chunk_count,
                "schema": type(shard).__name__,
            },
            identity_key=window_id,
        )
        return {"window_id": window_id, "artifact_id": ref.id}

    def _window_tasks(
        self,
        lease: TaskLease,
        stage: TaskStage,
        windows: list[dict[str, str]],
    ) -> list[TaskDefinition]:
        """One task of ``stage`` per window, each waiting on this task alone.

        ``windows`` names each window and the artifact holding it, in document
        order; the window is the task's scope, so its id is stable across retries.
        """

        return [
            TaskDefinition(
                id=stable_id("task", lease.task.run_id, stage.value, window["window_id"]),
                run_id=lease.task.run_id,
                stage=stage,
                scope_id=window["window_id"],
                payload={"window_artifact_id": window["artifact_id"]},
                dependency_ids=[lease.task.id],
                max_attempts=self.settings.distributed.max_attempts,
            )
            for window in windows
        ]

    def _document_task(
        self,
        lease: TaskLease,
        stage: TaskStage,
        payload: dict[str, Any],
        window_tasks: list[TaskDefinition],
    ) -> TaskDefinition:
        """The document's task of ``stage`` that combines its windows once all are done."""

        return TaskDefinition(
            id=stable_id("task", lease.task.run_id, stage.value, lease.task.scope_id),
            run_id=lease.task.run_id,
            stage=stage,
            scope_id=lease.task.scope_id,
            payload=payload,
            dependency_ids=[lease.task.id, *(task.id for task in window_tasks)],
            max_attempts=self.settings.distributed.max_attempts,
        )

    def _extract_entity_window(self, lease: TaskLease, run: _RunContext) -> str:
        """Extract entities from one bounded window claimed from the shared queue.

        A compact dependency supplies reusable context and a content-addressed
        window artifact supplies only selected chunks. Prepared page and block rows
        remain in their original stage artifact for graph finalization and are not
        transferred repeatedly to extraction workers.
        """

        context_inputs = self._dependency_artifacts(lease, ArtifactKind.DOCUMENT_CONTEXT)
        if len(context_inputs) != 1:
            raise ValueError("extract_entity_window requires exactly one document-context artifact")
        context = DocumentContextShard.model_validate_json(context_inputs[0].payload)
        window_artifact_id = lease.task.payload.get("window_artifact_id")
        if not isinstance(window_artifact_id, str) or not window_artifact_id:
            raise ValueError("extract_entity_window task requires window_artifact_id")
        window_artifact = self.artifact_store.get(window_artifact_id)
        if window_artifact.ref.kind != ArtifactKind.EXTRACTION_WINDOW:
            raise ValueError("extract_entity_window input is not an extraction-window artifact")
        window = ExtractionWindowShard.model_validate_json(window_artifact.payload)
        if window.file_ids != context.file_ids:
            raise ValueError("extraction window and document context file identities differ")
        window_prepared = PreparedDocumentShard(
            file_ids=window.file_ids,
            files_seen=0,
            documents_processed=0,
            chunks=window.chunks,
            document_context_entities=context.document_context_entities,
            # Context trace is persisted once by document compaction rather than
            # copied into every window result.
            trace=[],
        )
        try:
            extracted = run.pipeline.extract_window_entities(window_prepared)
        except Exception as exc:
            error = _window_error_on_last_attempt(lease, window, exc)
            extracted = EntityWindowShard(
                file_ids=window.file_ids,
                chunk_ids=[chunk.id for chunk in window.chunks],
                trace=[error],
            )
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.EXTRACTED_ENTITY_WINDOW,
            extracted.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": extracted.file_ids,
                "window_id": lease.task.scope_id,
                "schema": EntityWindowShard.__name__,
            },
        )
        return ref.id

    def _compact_entity_inventory(self, lease: TaskLease) -> TaskExecution:
        """Compact first-phase mentions and fan out relation-window tasks.

        This stage performs no provider work. It stores one document-wide entity
        inventory, then atomically extends the durable DAG with relation tasks and
        their final document compaction barrier. At-least-once retries are safe
        because task IDs and artifact identities are deterministic.

        A revision that extracts only a kept document's relations again starts
        here: the payload names the inventory and windows an earlier run
        stored, and the inventory is carried into this run unchanged. A
        document being revalidated has its revalidated entities compacted here.
        """

        if _REVALIDATION_WINDOWS in lease.task.payload:
            return self._compact_revalidated_inventory(lease)
        stored = lease.task.payload.get("inventory_artifact_id")
        inventory = (
            DocumentEntityInventoryShard.model_validate_json(
                self._stored_artifact(stored, ArtifactKind.DOCUMENT_ENTITY_INVENTORY).payload
            )
            if isinstance(stored, str) and stored
            else self._compacted_inventory(lease)
        )
        inventory_ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.DOCUMENT_ENTITY_INVENTORY,
            inventory.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": inventory.file_ids,
                "entities": len(inventory.entities),
                "windows": inventory.window_count,
                "schema": DocumentEntityInventoryShard.__name__,
            },
            identity_key=lease.task.scope_id,
        )
        return self._relation_fan_out(
            lease,
            inventory_ref,
            "windows",
            (TaskStage.EXTRACT_RELATION_WINDOW, TaskStage.COMPACT_DOCUMENT),
            {"prepared_artifact_id": lease.task.payload.get("prepared_artifact_id")},
        )

    def _compacted_inventory(self, lease: TaskLease) -> DocumentEntityInventoryShard:
        """Deduplicate a document's context and entity-window mentions into one inventory."""

        dependencies = self._all_dependency_artifacts(lease)
        contexts = [item for item in dependencies if item.ref.kind == ArtifactKind.DOCUMENT_CONTEXT]
        entity_windows = [
            item for item in dependencies if item.ref.kind == ArtifactKind.EXTRACTED_ENTITY_WINDOW
        ]
        unexpected = [
            item.ref.id
            for item in dependencies
            if item.ref.kind
            not in {ArtifactKind.DOCUMENT_CONTEXT, ArtifactKind.EXTRACTED_ENTITY_WINDOW}
        ]
        if unexpected:
            raise ValueError(
                f"compact_entity_inventory received unsupported artifacts: {unexpected}"
            )
        if len(contexts) != 1:
            raise ValueError(
                "compact_entity_inventory requires exactly one document-context artifact"
            )
        context = DocumentContextShard.model_validate_json(contexts[0].payload)
        shards = [EntityWindowShard.model_validate_json(item.payload) for item in entity_windows]
        if any(shard.file_ids != context.file_ids for shard in shards):
            raise ValueError("entity windows and document context file identities differ")
        entities = _unique_entities(
            [
                *context.document_context_entities,
                *(entity for shard in shards for entity in shard.entities),
            ]
        )
        return DocumentEntityInventoryShard(
            file_ids=context.file_ids,
            entities=entities,
            trace=[*context.trace, *(event for shard in shards for event in shard.trace)],
            chunk_count=sum(len(shard.chunk_ids) for shard in shards),
            window_count=len(shards),
            document_openings=context.document_openings,
        )

    def _relation_fan_out(
        self,
        lease: TaskLease,
        inventory_ref: ArtifactRef,
        windows_key: str,
        stages: tuple[TaskStage, TaskStage],
        document_payload: dict[str, Any],
    ) -> TaskExecution:
        """Queue a relation task per window and the document's task behind them.

        ``windows_key`` names the payload's window list; ``stages`` are the
        stage of the window tasks and of the document task that waits on them.
        """

        window_stage, document_stage = stages
        relation_tasks = self._window_tasks(
            lease, window_stage, self._window_payloads(lease, windows_key)
        )
        return TaskExecution(
            output_artifact_ids=(inventory_ref.id,),
            follow_up_tasks=(
                *relation_tasks,
                self._document_task(lease, document_stage, document_payload, relation_tasks),
            ),
            barrier_task_id=stable_id(
                "task",
                lease.task.run_id,
                TaskStage.FINALIZE_GRAPH.value,
            ),
        )

    def _window_payloads(self, lease: TaskLease, key: str) -> list[dict[str, str]]:
        """The windows a payload lists under ``key``, each with its artifact."""

        stage = lease.task.stage.value
        items = lease.task.payload.get(key)
        if not isinstance(items, list):
            raise ValueError(f"{stage} requires a {key} payload")
        windows: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"{stage} windows must be objects")
            window_id = item.get("window_id")
            artifact_id = item.get("artifact_id")
            if not isinstance(window_id, str) or not isinstance(artifact_id, str):
                raise ValueError(f"{stage} window identities must be strings")
            windows.append({"window_id": window_id, "artifact_id": artifact_id})
        return windows

    def _extract_relation_window(self, lease: TaskLease, run: _RunContext) -> str:
        """Extract relations in one window against its complete document inventory."""

        inventory_inputs = self._dependency_artifacts(lease, ArtifactKind.DOCUMENT_ENTITY_INVENTORY)
        if len(inventory_inputs) != 1:
            raise ValueError(
                "extract_relation_window requires exactly one document entity inventory"
            )
        inventory = DocumentEntityInventoryShard.model_validate_json(inventory_inputs[0].payload)
        window_artifact_id = lease.task.payload.get("window_artifact_id")
        if not isinstance(window_artifact_id, str) or not window_artifact_id:
            raise ValueError("extract_relation_window task requires window_artifact_id")
        window_artifact = self.artifact_store.get(window_artifact_id)
        if window_artifact.ref.kind != ArtifactKind.EXTRACTION_WINDOW:
            raise ValueError("extract_relation_window input is not an extraction-window artifact")
        window = ExtractionWindowShard.model_validate_json(window_artifact.payload)
        if window.file_ids != inventory.file_ids:
            raise ValueError("relation window and entity inventory file identities differ")
        if not window.extract_relations:
            # The window still reports, so compaction sees every window covered.
            skipped = RelationWindowShard(
                file_ids=window.file_ids,
                chunk_ids=[chunk.id for chunk in window.chunks],
                trace=[
                    {
                        "stage": "relation_extraction",
                        "window_id": lease.task.scope_id,
                        "skipped": "dataset_without_relations",
                    }
                ],
            )
            return self.artifact_store.put(
                lease.task.run_id,
                ArtifactKind.EXTRACTED_RELATION_WINDOW,
                skipped.model_dump_json().encode("utf-8"),
                _JSON_MEDIA_TYPE,
                metadata={
                    "file_ids": skipped.file_ids,
                    "window_id": lease.task.scope_id,
                    "schema": RelationWindowShard.__name__,
                },
            ).id
        window_prepared = PreparedDocumentShard(
            file_ids=window.file_ids,
            files_seen=0,
            documents_processed=0,
            chunks=window.chunks,
        )
        try:
            extracted = run.pipeline.extract_window_relations(
                window_prepared,
                inventory.entities,
                inventory.document_openings,
                dataset_columns=window.dataset_columns,
            )
        except Exception as exc:
            error = _window_error_on_last_attempt(lease, window, exc)
            extracted = RelationWindowShard(
                file_ids=window.file_ids,
                chunk_ids=[chunk.id for chunk in window.chunks],
                trace=[error],
            )
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.EXTRACTED_RELATION_WINDOW,
            extracted.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": extracted.file_ids,
                "window_id": lease.task.scope_id,
                "schema": RelationWindowShard.__name__,
            },
        )
        return ref.id

    def _compact_document(self, lease: TaskLease, run: _RunContext) -> str:
        """Assemble independently extracted windows into one document artifact.

        The compaction boundary keeps the object count consumed by Spark
        proportional to documents rather than LLM windows. It performs no provider
        calls and is safe to retry because both its source artifacts and resulting
        content-addressed document shard are immutable.
        """

        prepared_artifact_id = lease.task.payload.get("prepared_artifact_id")
        if not isinstance(prepared_artifact_id, str) or not prepared_artifact_id:
            raise ValueError("compact_document requires prepared_artifact_id")
        prepared_artifact = self.artifact_store.get(prepared_artifact_id)
        if prepared_artifact.ref.kind != ArtifactKind.PREPARED_DOCUMENT:
            raise ValueError("compact_document input is not a prepared document artifact")
        prepared = PreparedDocumentShard.model_validate_json(prepared_artifact.payload)

        dependencies = self._all_dependency_artifacts(lease)
        inventories = [
            item for item in dependencies if item.ref.kind == ArtifactKind.DOCUMENT_ENTITY_INVENTORY
        ]
        windows = [
            item for item in dependencies if item.ref.kind == ArtifactKind.EXTRACTED_RELATION_WINDOW
        ]
        unexpected = [
            item.ref.id
            for item in dependencies
            if item.ref.kind
            not in {
                ArtifactKind.DOCUMENT_ENTITY_INVENTORY,
                ArtifactKind.EXTRACTED_RELATION_WINDOW,
            }
        ]
        if unexpected:
            raise ValueError(f"compact_document received unsupported artifacts: {unexpected}")
        if len(inventories) != 1:
            raise ValueError("compact_document requires exactly one document entity inventory")
        inventory = DocumentEntityInventoryShard.model_validate_json(inventories[0].payload)
        if inventory.file_ids != prepared.file_ids:
            raise ValueError("prepared document and entity inventory file identities differ")
        relation_shards = [
            RelationWindowShard.model_validate_json(item.payload) for item in windows
        ]
        if any(shard.file_ids != prepared.file_ids for shard in relation_shards):
            raise ValueError("relation windows and prepared document file identities differ")
        if len(relation_shards) != inventory.window_count:
            raise ValueError("relation windows do not cover the complete entity inventory")
        relation_trace = [event for shard in relation_shards for event in shard.trace]
        compacted = ExtractedDocumentShard(
            prepared=_prepared_projection_for_extracted_artifact(
                prepared,
                spark_finalization=_use_spark_finalization(self.settings),
            ),
            # Verification rejected or retyped some mentions; that reaches the
            # relations using them only now, when every window of the document is in.
            observations=apply_entity_verdicts(
                ExtractionObservations(
                    entities=_unique_mentions(
                        [
                            *inventory.entities,
                            *(e for shard in relation_shards for e in shard.entities),
                        ]
                    ),
                    relations=_unique_relations(
                        [relation for shard in relation_shards for relation in shard.relations]
                    ),
                    held_relations=_unique_relations(
                        [relation for shard in relation_shards for relation in shard.held_relations]
                    ),
                    trace=[*inventory.trace, *relation_trace],
                    chunk_count=inventory.chunk_count,
                    window_count=inventory.window_count,
                ),
                _run_ontology(run.settings),
            ),
            trace=[*inventory.trace, *relation_trace],
        )
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.EXTRACTED_DOCUMENT,
            compacted.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": compacted.prepared.file_ids,
                "windows": len(relation_shards),
                "schema": ExtractedDocumentShard.__name__,
            },
            identity_key=lease.task.scope_id,
        )
        return ref.id

    def _revalidate_document_context(self, lease: TaskLease, run: _RunContext) -> TaskExecution:
        """Place a kept document's stored records by window and fan out their replay.

        A revision's revalidation of a document starts here, with no dependency:
        the payload names the prepared text and the extracted document an earlier
        run stored. The document's context records are validated here and every
        other record goes, with the window that holds it, into a window artifact.
        Each window is then replayed by an entity task and, once the document's
        entities are compacted, by a relation task, as extraction's windows
        are, so no task holds its lease for a whole long document: the slow
        part - grounding fallbacks and the verifier - runs one window a task.
        """

        prepared, _stored, extracted = self._revalidation_inputs(lease)
        begun, windows = run.pipeline.begin_revalidation(prepared, extracted.observations)
        begun_ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.DOCUMENT_CONTEXT,
            begun.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": begun.file_ids,
                "schema": DocumentRevalidationShard.__name__,
                "document_context_entities": len(begun.document_context_entities),
            },
        )
        window_payloads = [
            self._stored_window(
                lease,
                entry.window.id,
                RevalidationWindowShard(file_ids=begun.file_ids, window=entry),
                len(entry.window.chunks),
            )
            for entry in windows
        ]
        entity_tasks = self._window_tasks(
            lease, TaskStage.REVALIDATE_ENTITY_WINDOW, window_payloads
        )
        # The seed's payload travels on to the document's last task, which
        # stores what it names beside the revalidated document.
        inventory_task = self._document_task(
            lease,
            TaskStage.COMPACT_ENTITY_INVENTORY,
            {
                **lease.task.payload,
                _REVALIDATION_ARTIFACT: begun_ref.id,
                _REVALIDATION_WINDOWS: window_payloads,
            },
            entity_tasks,
        )
        return TaskExecution(
            output_artifact_ids=(begun_ref.id,),
            follow_up_tasks=(*entity_tasks, inventory_task),
            barrier_task_id=stable_id("task", lease.task.run_id, TaskStage.FINALIZE_GRAPH.value),
        )

    def _revalidate_entity_window(self, lease: TaskLease, run: _RunContext) -> str:
        """Replay one window's stored entity records under the current rules."""

        window = self._revalidation_window(lease)
        shard = run.pipeline.revalidate_entity_window(window)
        return self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.EXTRACTED_ENTITY_WINDOW,
            shard.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": shard.file_ids,
                "window_id": lease.task.scope_id,
                "schema": EntityWindowShard.__name__,
            },
        ).id

    def _compact_revalidated_inventory(self, lease: TaskLease) -> TaskExecution:
        """Combine a revalidated document's context and entity windows, then fan out relations.

        Like an extraction's inventory, this makes no model call. The entity
        windows are combined in window order, as a single-task revalidation
        combines them, so the document's entities come out the same as when
        one task replays every window.
        """

        dependencies = self._all_dependency_artifacts(lease)
        begun = [item for item in dependencies if item.ref.kind == ArtifactKind.DOCUMENT_CONTEXT]
        entity_windows = [
            item for item in dependencies if item.ref.kind == ArtifactKind.EXTRACTED_ENTITY_WINDOW
        ]
        if len(begun) + len(entity_windows) != len(dependencies) or len(begun) != 1:
            raise ValueError(
                "compact_entity_inventory of a revalidation requires its document's "
                "revalidation and entity windows"
            )
        inventory = revalidated_inventory(
            DocumentRevalidationShard.model_validate_json(begun[0].payload),
            [
                EntityWindowShard.model_validate_json(item.payload)
                for item in self._in_window_order(lease, entity_windows)
            ],
        )
        inventory_ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.DOCUMENT_ENTITY_INVENTORY,
            inventory.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": inventory.file_ids,
                "entities": len(inventory.entities),
                "windows": inventory.window_count,
                "schema": DocumentEntityInventoryShard.__name__,
            },
            identity_key=lease.task.scope_id,
        )
        return self._relation_fan_out(
            lease,
            inventory_ref,
            _REVALIDATION_WINDOWS,
            (TaskStage.REVALIDATE_RELATION_WINDOW, TaskStage.REVALIDATE_DOCUMENT),
            lease.task.payload,
        )

    def _revalidate_relation_window(self, lease: TaskLease, run: _RunContext) -> str:
        """Replay and verify one window's stored relations against the document."""

        inventory_inputs = self._dependency_artifacts(lease, ArtifactKind.DOCUMENT_ENTITY_INVENTORY)
        if len(inventory_inputs) != 1:
            raise ValueError(
                "revalidate_relation_window requires exactly one document entity inventory"
            )
        inventory = DocumentEntityInventoryShard.model_validate_json(inventory_inputs[0].payload)
        window = self._revalidation_window(lease)
        if window.file_ids != inventory.file_ids:
            raise ValueError("revalidation window and entity inventory file identities differ")
        shard = run.pipeline.revalidate_relation_window(window, inventory)
        return self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.EXTRACTED_RELATION_WINDOW,
            shard.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": shard.file_ids,
                "window_id": lease.task.scope_id,
                "schema": RevalidatedRelationWindowShard.__name__,
            },
        ).id

    def _revalidate_document(self, lease: TaskLease, run: _RunContext) -> tuple[str, str, str]:
        """Combine a kept document's revalidated windows and store what they make.

        The payload names the prepared text and the extracted document an
        earlier run stored. The windows are combined in window order and the
        entity verdicts applied to the whole document. The revalidated
        document replaces the extracted one in this run; the prepared text
        stays where it is and the finalizer reads it from there. An entity
        inventory and a document context are stored beside it, so a later
        revision can extract this document's relations, or its entity windows,
        again from what survived.
        """

        prepared, stored, extracted = self._revalidation_inputs(lease)
        observations = self._concluded_revalidation(lease, run, extracted.observations)
        revalidated = ExtractedDocumentShard(
            prepared=_prepared_projection_for_extracted_artifact(
                prepared,
                spark_finalization=_use_spark_finalization(self.settings),
            ),
            observations=observations,
            trace=observations.trace,
        )
        summary = next(
            (event for event in observations.trace if event.get("stage") == REVALIDATION_STAGE),
            {},
        )
        extracted_ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.EXTRACTED_DOCUMENT,
            revalidated.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": prepared.file_ids,
                "windows": stored.ref.metadata["windows"],
                "schema": ExtractedDocumentShard.__name__,
                "revalidated": {
                    "entities": summary.get("entities", {}),
                    "relations": summary.get("relations", {}),
                },
            },
            identity_key=lease.task.scope_id,
        )
        openings = build_document_openings(prepared.chunks)
        inventory = DocumentEntityInventoryShard(
            file_ids=prepared.file_ids,
            entities=observations.entities,
            trace=[
                event
                for event in observations.trace
                if str(event.get("stage") or "") not in RELATION_PHASE_STAGES
            ],
            chunk_count=observations.chunk_count,
            window_count=observations.window_count,
            document_openings=openings,
        )
        context = DocumentContextShard(
            file_ids=prepared.file_ids,
            document_context_entities=[
                mention for mention in observations.entities if mention.is_document_context
            ],
            trace=[
                event
                for event in inventory.trace
                if str(event.get("stage") or "") not in ENTITY_WINDOW_STAGES
            ],
            document_openings=openings,
        )
        context_ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.DOCUMENT_CONTEXT,
            context.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": context.file_ids,
                "schema": DocumentContextShard.__name__,
                "document_context_entities": len(context.document_context_entities),
            },
        )
        inventory_ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.DOCUMENT_ENTITY_INVENTORY,
            inventory.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": inventory.file_ids,
                "entities": len(inventory.entities),
                "windows": inventory.window_count,
                "schema": DocumentEntityInventoryShard.__name__,
            },
            identity_key=lease.task.scope_id,
        )
        return extracted_ref.id, inventory_ref.id, context_ref.id

    def _finalize_graph(self, lease: TaskLease, run: _RunContext) -> str:
        """Select the local or partitioned engine without a corpus-size cutoff."""

        if _use_spark_finalization(self.settings):
            return self._finalize_graph_with_spark(lease, run)
        return self._finalize_graph_locally(lease, run)

    def _finalize_graph_with_spark(self, lease: TaskLease, run: _RunContext) -> str:
        """Coordinate distributed finalization and persist only its small manifest.

        Spark executors read immutable stage objects directly. The leased worker is
        a driver and heartbeat owner; it never deserializes the complete corpus.
        """

        report = self._task_progress_reporter(lease)
        stage_kinds = {ArtifactKind.PREPARED_DOCUMENT, ArtifactKind.EXTRACTED_DOCUMENT}
        inputs = self.artifact_store.list_run_artifacts(lease.task.run_id, stage_kinds)
        inputs.extend(self._inherited_artifact_refs(lease, stage_kinds))
        quarantined = self._quarantined_file_ids(lease.task.run_id)
        inputs = [ref for ref in inputs if not quarantined.intersection(ref.metadata["file_ids"])]
        manifest = SparkGraphFinalizer(run.settings, report).finalize(
            SparkFinalizationRequest(
                run_id=lease.task.run_id,
                graph_id=lease.task.scope_id,
                attempt=lease.attempt,
                artifact_ids=frozenset(ref.id for ref in inputs),
                prepared_run_ids=frozenset(
                    ref.run_id for ref in inputs if ref.kind is ArtifactKind.PREPARED_DOCUMENT
                ),
                extracted_run_ids=frozenset(
                    ref.run_id for ref in inputs if ref.kind is ArtifactKind.EXTRACTED_DOCUMENT
                ),
                document_provenance=document_provenance(lease.task.payload),
                consumption_events=tuple(
                    getattr(self.task_store, "consumption_events", lambda _run: [])(
                        lease.task.run_id
                    )
                ),
            )
        )
        # Destination publication is queued atomically by ``complete_task`` after
        # this immutable manifest exists. A separate fenced outbox delivery owns
        # the externally visible Snowflake effect.
        report(finalization_progress("publish_destination", completed=0, total=1))
        report(finalization_progress("persist_manifest", completed=0, total=1))
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.GRAPH_RESULT,
            manifest.model_dump_json().encode("utf-8"),
            _GRAPH_MANIFEST_MEDIA_TYPE,
            metadata={
                "graph_id": manifest.graph_id,
                "engine": manifest.engine,
                "schema": manifest.format,
                "tables": {name: table.row_count for name, table in manifest.tables.items()},
            },
        )
        report(finalization_progress("persist_manifest", completed=1, total=1))
        return ref.id

    def _finalize_graph_locally(self, lease: TaskLease, run: _RunContext) -> str:
        """Recombine stage objects for the lightweight in-process execution path."""

        report = self._task_progress_reporter(lease)
        report(finalization_progress("read_artifacts", completed=0, total=1))
        inputs = self.artifact_store.get_run_artifacts(
            lease.task.run_id,
            {ArtifactKind.EXTRACTED_DOCUMENT},
        )
        inherited = self._inherited_artifact_refs(lease, {ArtifactKind.EXTRACTED_DOCUMENT})
        inputs.extend(self.artifact_store.get_many([ref.id for ref in inherited]))
        document_shards = [
            ExtractedDocumentShard.model_validate_json(item.payload) for item in inputs
        ]
        quarantined = self._quarantined_file_ids(lease.task.run_id)
        document_shards = [
            shard
            for shard in document_shards
            if not quarantined.intersection(shard.prepared.file_ids)
        ]
        document_shards.sort(key=lambda shard: tuple(shard.prepared.file_ids))
        if not document_shards and not quarantined:
            raise ValueError("finalize_graph requires at least one extracted document")
        provenance = document_provenance(lease.task.payload)
        for shard in document_shards:
            for row in shard.prepared.document_rows:
                row["provenance"] = provenance.get(str(row.get("file_id")), PROVENANCE_EXTRACTED)
        report(finalization_progress("read_artifacts", completed=1, total=1))
        report(finalization_progress("materialize_inputs", completed=1, total=1))
        report(finalization_progress("resolve_entities", completed=0, total=1))
        self._refresh_consumption(lease.task.run_id, run.pipeline)
        batch = run.pipeline.finalize_document_shards(document_shards, write=False)
        for phase, _label in FINALIZATION_PHASES[4:11]:
            report(finalization_progress(phase, completed=1, total=1))
        report(finalization_progress("persist_manifest", completed=0, total=1))
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.GRAPH_RESULT,
            batch.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "graph_id": batch.graph_id,
                "nodes": len(batch.nodes),
                "edges": len(batch.edges),
                "schema": GraphWriteBatch.__name__,
            },
        )
        report(finalization_progress("persist_manifest", completed=1, total=1))
        return ref.id

    def _refresh_consumption(self, run_id: str, pipeline: DistributedPipeline) -> None:
        """Include usage from other worker processes before building graph metrics."""

        collector = getattr(pipeline, "consumption", None)
        read = getattr(self.task_store, "consumption_events", None)
        if isinstance(collector, ConsumptionCollector) and read is not None:
            collector.merge(read(run_id))

    def _quarantined_file_ids(self, run_id: str) -> set[str]:
        """Read durable retry exclusions afresh rather than from the settings cache."""

        config = self.task_store.get_run_summary(run_id).run.config
        return {str(value) for value in config.get("quarantined_file_ids", [])}

    def _inherited_artifact_refs(
        self,
        lease: TaskLease,
        kinds: set[ArtifactKind],
    ) -> list[ArtifactRef]:
        """The stage outputs a revision keeps from earlier runs of its graph.

        The finalizer's payload names each run and the documents kept from it;
        an artifact is kept when every document it covers is. Nothing is copied:
        the earlier run's objects are read where they are.
        """

        entries = lease.task.payload.get("inherit")
        if not isinstance(entries, list) or not entries:
            return []
        refs: list[ArtifactRef] = []
        for raw in entries:
            entry = InheritedDocuments.model_validate(raw)
            for ref in self.artifact_store.list_run_artifacts(entry.run_id, kinds):
                if entry.reads([str(item) for item in ref.metadata["file_ids"]], ref.kind):
                    refs.append(ref)
        return refs

    def _task_progress_reporter(
        self,
        lease: TaskLease,
    ) -> Callable[[TaskProgress], None]:
        """Return a best-effort reporter bound to the task's current lease owner."""

        def report(progress: TaskProgress) -> None:
            """Persist progress without allowing observability failure to abort work."""

            try:
                self.task_store.report_task_progress(
                    lease.task.id, self.worker_id, progress, claim_token=lease.claim_token
                )
            except Exception:
                # Lease heartbeat remains authoritative for ownership and
                # availability. A progress write is useful but non-critical.
                return

        return report

    def _dependency_artifacts(
        self,
        lease: TaskLease,
        expected_kind: ArtifactKind,
    ) -> list[StoredArtifact]:
        """Load direct dependency outputs in deterministic task/output order."""

        artifact_ids = [
            artifact_id
            for dependency_id in sorted(lease.dependency_outputs)
            for artifact_id in lease.dependency_outputs[dependency_id]
        ]
        artifacts = self.artifact_store.get_many(artifact_ids)
        unexpected = [item.ref.id for item in artifacts if item.ref.kind != expected_kind]
        if unexpected:
            raise ValueError(
                f"task dependencies contain artifacts with the wrong kind: {unexpected}"
            )
        return artifacts

    def _stored_artifact(self, artifact_id: str, expected_kind: ArtifactKind) -> StoredArtifact:
        """Load an artifact a payload names, which an earlier run may have stored."""

        stored = self.artifact_store.get(artifact_id)
        if stored.ref.kind != expected_kind:
            raise ValueError(f"artifact {artifact_id} is not a {expected_kind.value} artifact")
        return stored

    def _revalidation_inputs(
        self, lease: TaskLease
    ) -> tuple[PreparedDocumentShard, StoredArtifact, ExtractedDocumentShard]:
        """The prepared text and stored extraction a revalidation's payload names."""

        prepared_id = lease.task.payload.get("prepared_artifact_id")
        extracted_id = lease.task.payload.get("extracted_artifact_id")
        if not isinstance(prepared_id, str) or not isinstance(extracted_id, str):
            raise ValueError(f"{lease.task.stage.value} requires prepared and extracted artifacts")
        prepared = PreparedDocumentShard.model_validate_json(
            self._stored_artifact(prepared_id, ArtifactKind.PREPARED_DOCUMENT).payload
        )
        stored = self._stored_artifact(extracted_id, ArtifactKind.EXTRACTED_DOCUMENT)
        extracted = ExtractedDocumentShard.model_validate_json(stored.payload)
        if extracted.prepared.file_ids != prepared.file_ids:
            raise ValueError("revalidated document and its prepared text differ")
        return prepared, stored, extracted

    def _revalidation_window(self, lease: TaskLease) -> RevalidationWindowShard:
        """The window, with its stored records, a revalidation task replays."""

        artifact_id = lease.task.payload.get("window_artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError(f"{lease.task.stage.value} task requires window_artifact_id")
        return RevalidationWindowShard.model_validate_json(
            self._stored_artifact(artifact_id, ArtifactKind.EXTRACTION_WINDOW).payload
        )

    def _concluded_revalidation(
        self,
        lease: TaskLease,
        run: _RunContext,
        before: ExtractionObservations,
    ) -> ExtractionObservations:
        """Combine a document's revalidation and its relation windows, in window order."""

        begun_id = lease.task.payload.get(_REVALIDATION_ARTIFACT)
        if not isinstance(begun_id, str) or not begun_id:
            raise ValueError(f"revalidate_document requires {_REVALIDATION_ARTIFACT}")
        begun = DocumentRevalidationShard.model_validate_json(
            self._stored_artifact(begun_id, ArtifactKind.DOCUMENT_CONTEXT).payload
        )
        dependencies = self._all_dependency_artifacts(lease)
        inventories = [
            item for item in dependencies if item.ref.kind == ArtifactKind.DOCUMENT_ENTITY_INVENTORY
        ]
        windows = [
            item for item in dependencies if item.ref.kind == ArtifactKind.EXTRACTED_RELATION_WINDOW
        ]
        if len(inventories) + len(windows) != len(dependencies) or len(inventories) != 1:
            raise ValueError(
                "revalidate_document requires one entity inventory and its relation windows"
            )
        return conclude_revalidation(
            before,
            begun,
            DocumentEntityInventoryShard.model_validate_json(inventories[0].payload),
            [
                RevalidatedRelationWindowShard.model_validate_json(item.payload)
                for item in self._in_window_order(lease, windows)
            ],
            _run_ontology(run.settings),
        )

    def _in_window_order(
        self, lease: TaskLease, artifacts: list[StoredArtifact]
    ) -> list[StoredArtifact]:
        """Window results in the order the payload lists their windows, one per window.

        Dependency outputs arrive in task-id order, which says nothing about
        where a window sits in its document; the payload's list does.
        """

        by_window = {str(item.ref.metadata.get("window_id")): item for item in artifacts}
        windows = self._window_payloads(lease, _REVALIDATION_WINDOWS)
        if len(by_window) != len(artifacts) or set(by_window) != {
            window["window_id"] for window in windows
        }:
            raise ValueError(f"{lease.task.stage.value} results do not cover every window")
        return [by_window[window["window_id"]] for window in windows]

    def _all_dependency_artifacts(self, lease: TaskLease) -> list[StoredArtifact]:
        """Load all direct outputs in deterministic task and output order."""

        artifact_ids = [
            artifact_id
            for dependency_id in sorted(lease.dependency_outputs)
            for artifact_id in lease.dependency_outputs[dependency_id]
        ]
        return self.artifact_store.get_many(artifact_ids)


def _window_error_on_last_attempt(
    lease: TaskLease,
    window: ExtractionWindowShard,
    exc: Exception,
) -> dict[str, Any]:
    """Record a window that failed its last attempt, or raise while attempts remain.

    A failed window costs an attempt, so a provider that was briefly down or a
    reply that did not parse is asked again. The last attempt records the
    failure instead, as a ``window_error`` trace event the document keeps, and
    the document goes on without that window's records rather than failing
    its whole run. A provider that refuses every request fails the task on
    every attempt: that is no one window's failure.
    """

    if lease.attempt < lease.task.max_attempts or is_systemic_provider_error(exc):
        raise exc
    logger.warning(
        "window %s failed its last attempt (%s); recorded as a window error",
        lease.task.scope_id,
        type(exc).__name__,
    )
    return {
        "stage": "window_error",
        "window_id": lease.task.scope_id,
        "document_id": window.chunks[0].document_id if window.chunks else "",
        "error": error_metadata(exc),
    }


def _unique_entities(entities: list[EntityMention]) -> list[EntityMention]:
    """Deduplicate entity observations deterministically at a document barrier.

    Mention IDs include grounded source identity, so retaining the first record for
    each ID removes context/window overlap without collapsing distinct occurrences
    that graph-wide entity resolution still needs to adjudicate.
    """

    return list({entity.id: entity for entity in entities}.values())


def _unique_mentions(mentions: list[EntityMention]) -> list[EntityMention]:
    """Deduplicate the inventory and the endpoints relation windows added to it."""

    return list({mention.id: mention for mention in mentions}.values())


def _unique_relations(relations: list[RelationObservation]) -> list[RelationObservation]:
    """Deduplicate identical grounded relation observations after queue fan-in."""

    return list({relation.id: relation for relation in relations}.values())


def _prepared_projection_for_extracted_artifact(
    prepared: PreparedDocumentShard,
    *,
    spark_finalization: bool,
) -> PreparedDocumentShard:
    """Avoid copying prepared corpus rows into Spark extraction artifacts.

    Spark reads the immutable ``prepared_document`` prefix directly for documents,
    pages, blocks, assets, and chunks. Repeating those rows inside every completed
    document doubles object-store bytes and JSON scanning at corpus scale. The
    extracted artifact therefore retains only stable file-level counters on the
    Spark path; its observation payload already contains every entity and relation
    needed by finalization. Local finalization keeps the complete prepared shard
    because it intentionally operates from self-contained document artifacts.
    """

    if not spark_finalization:
        return prepared
    return PreparedDocumentShard(
        file_ids=prepared.file_ids,
        files_seen=prepared.files_seen,
        documents_processed=prepared.documents_processed,
        ocr_cache_hits=prepared.ocr_cache_hits,
    )


def _safe_filename(value: object) -> str:
    """Return a basename suitable for a temporary worker directory."""

    if not isinstance(value, str) or not value.strip():
        return "source.bin"
    return Path(value).name or "source.bin"


def _use_spark_finalization(settings: Settings) -> bool:
    """Resolve automatic engine selection from runtime capabilities, not corpus size.

    Local installations remain dependency-light. A Kubernetes deployment with a
    shared artifact URI automatically receives the scalable engine for both small
    and large corpora, giving one semantic execution path throughout that fleet.
    """

    selected = settings.distributed.finalization_engine
    if selected == "spark":
        return True
    if selected == "local":
        return False
    return settings.runtime.runtime == "kubernetes" and bool(settings.distributed.artifact_uri)
