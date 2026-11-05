# SPDX-License-Identifier: Apache-2.0
"""PostgreSQL task and artifact store for production distributed execution.

The adapter uses short transactions and row-level leases. Long OCR or LLM work
never holds a database lock: workers claim, commit, process externally, then finish
through an ownership-checked transition. Expired leases are safely reclaimable and
immutable artifact checksums protect stage handoffs from silent corruption.
"""

from __future__ import annotations

import zlib
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
from threading import Lock
from typing import Any, cast
from uuid import uuid4
from weakref import finalize

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

from kg_processor.application.ordered_map import bounded_map
from kg_processor.domain.consumption import ConsumptionEvent
from kg_processor.domain.distributed import (
    ArtifactKind,
    ArtifactRef,
    GraphDeletion,
    PriorityBand,
    PublicationLease,
    PublicationStatus,
    ResolvedFailure,
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
    task_priority,
)
from kg_processor.domain.ids import sha256_hex, stable_id
from kg_processor.ports.blob_store import BlobStore
from kg_processor.ports.task_store import TaskStoreUnavailableError

_INITIAL_TASK_COPY_BATCH_SIZE = 2_000
_ARTIFACT_READ_PARALLELISM = 8
_MAX_RUN_LIST_LIMIT = 500
# One row per stage/status bucket, whatever the task count; the newest
# progress payload in the bucket stands for it.
_TASK_COUNT_SELECT = """
    stage, status, COUNT(*) AS task_count,
    COUNT(*) FILTER (
        WHERE status = 'queued'
          AND remaining_dependencies = 0
          AND available_at <= CURRENT_TIMESTAMP
    ) AS ready_count,
    MIN(started_at) AS started_at,
    MAX(completed_at) AS completed_at,
    MAX(updated_at) AS updated_at,
    (jsonb_agg(progress_json ORDER BY progress_updated_at DESC)
        FILTER (WHERE progress_json IS NOT NULL))->0 AS progress_json
"""
# An artifact a succeeded task handed to ``complete_task``; the status is bound
# by the caller. See ``get_run_artifact_ids`` for why linkage matters.
_LINKED_ARTIFACT_CONDITION = """
    EXISTS (
        SELECT 1
        FROM flakegraph_task_output AS output
        JOIN flakegraph_task AS task ON task.id = output.task_id
        WHERE output.artifact_id = artifact.id
          AND task.run_id = artifact.run_id
          AND task.status = %s
    )
"""
# A task row with its linked outputs and dependency edges, as a snapshot needs.
_TASK_SNAPSHOT_SELECT = """
    SELECT task.*,
           COALESCE(
               (
                   SELECT jsonb_agg(output.artifact_id ORDER BY output.position)
                   FROM flakegraph_task_output AS output
                   WHERE output.task_id = task.id
               ),
               '[]'::jsonb
           ) AS output_artifact_ids,
           COALESCE(
               (
                   SELECT jsonb_agg(
                       dependency.depends_on_task_id
                       ORDER BY dependency.depends_on_task_id
                   )
                   FROM flakegraph_task_dependency AS dependency
                   WHERE dependency.task_id = task.id
               ),
               '[]'::jsonb
           ) AS dependency_ids
    FROM flakegraph_task AS task
"""
_SCHEMA_VERSION = 13
# How many of a run's ready tasks per stage a claim considers. Only the first
# is taken; the rest stand in when a concurrent claim has locked it, so this
# bounds how many workers of one pool can claim from one run and stage in the
# same instant before one of them falls through to the next run.
_CLAIM_CANDIDATES_PER_STAGE = 16
# One claim: the band first, then the run holding the fewest live leases in
# this worker's stages, then the stage ladder, then age.
#
# Ordering the whole ready queue by a per-run load would sort every ready task
# on every claim. The query works from the other side. ``load`` counts live
# leases in the claiming stages once, in one pass over the running rows -
# never more than the fleet has workers - and ``pool_run`` sets it beside each
# claimable run. ``ready`` then probes each run and stage for its first few
# ready tasks through the queued claim index, (run_id, stage,
# remaining_dependencies, priority DESC, created_at, id) with available_at
# included, so each probe reads a handful of index entries in order however
# many tasks the run has queued. The run-wide final task and lapsed leases
# have probes of their own. The candidates - at most runs x stages x
# _CLAIM_CANDIDATES_PER_STAGE rows - are sorted, and the first one no
# concurrent claim holds is locked and leased. Eligibility is checked again
# on the locked row, so a task another worker claimed in the meantime is
# skipped rather than claimed twice.
_CLAIM_TASK = """
    WITH load AS MATERIALIZED (
        SELECT leased.run_id, COUNT(*) AS leased
        FROM flakegraph_task AS leased
        WHERE leased.status = 'running'
          AND leased.stage = ANY(%(stages)s)
          AND leased.lease_expires_at >= CURRENT_TIMESTAMP
        GROUP BY leased.run_id
    ),
    pool_run AS MATERIALIZED (
        SELECT run.id,
               run.priority_band = 'interactive' AS interactive,
               COALESCE(load.leased, 0) AS leased
        FROM flakegraph_run AS run
        LEFT JOIN load ON load.run_id = run.id
        WHERE run.status IN ('queued', 'running')
          AND NOT EXISTS (
              SELECT 1 FROM flakegraph_graph_deletion AS deletion
              WHERE deletion.graph_id = run.graph_id
          )
          AND (
              CAST(%(config_digest)s AS TEXT) IS NULL
              OR run.config_digest = %(config_digest)s
          )
    ),
    ready AS MATERIALIZED (
        SELECT pool_run.interactive, pool_run.leased,
               head.id, head.priority, head.created_at
        FROM pool_run
        CROSS JOIN unnest(CAST(%(stages)s AS TEXT[])) AS pool(stage)
        CROSS JOIN LATERAL (
            SELECT task.id, task.priority, task.created_at
            FROM flakegraph_task AS task
            WHERE task.run_id = pool_run.id
              AND task.stage = pool.stage
              AND task.status = 'queued'
              AND task.remaining_dependencies = 0
              AND task.available_at <= CURRENT_TIMESTAMP
            ORDER BY task.priority DESC, task.created_at, task.id
            LIMIT %(candidates_per_stage)s
        ) AS head
        WHERE pool.stage <> 'finalize_graph'
        UNION ALL
        -- The final task's barrier is every other task of its run, not its
        -- dependency counter.
        SELECT pool_run.interactive, pool_run.leased,
               task.id, task.priority, task.created_at
        FROM pool_run
        JOIN flakegraph_task AS task ON task.run_id = pool_run.id
        WHERE 'finalize_graph' = ANY(%(stages)s)
          AND task.stage = 'finalize_graph'
          AND task.status = 'queued'
          AND task.available_at <= CURRENT_TIMESTAMP
          AND NOT EXISTS (
              SELECT 1
              FROM flakegraph_task AS unfinished
              WHERE unfinished.run_id = task.run_id
                AND unfinished.id <> task.id
                AND unfinished.status <> 'succeeded'
          )
        UNION ALL
        -- A lease whose worker stopped renewing it, with attempts left.
        SELECT pool_run.interactive, pool_run.leased,
               task.id, task.priority, task.created_at
        FROM pool_run
        JOIN flakegraph_task AS task ON task.run_id = pool_run.id
        WHERE task.status = 'running'
          AND task.stage = ANY(%(stages)s)
          AND task.lease_expires_at < CURRENT_TIMESTAMP
          AND task.attempts < task.max_attempts
          AND task.available_at <= CURRENT_TIMESTAMP
    ),
    candidate AS (
        SELECT task.id
        FROM ready
        JOIN flakegraph_task AS task ON task.id = ready.id
        WHERE task.available_at <= CURRENT_TIMESTAMP
          AND (
              (
                  task.status = 'queued'
                  AND (task.stage = 'finalize_graph' OR task.remaining_dependencies = 0)
              )
              OR (
                  task.status = 'running'
                  AND task.lease_expires_at < CURRENT_TIMESTAMP
                  AND task.attempts < task.max_attempts
              )
          )
        ORDER BY ready.interactive DESC, ready.leased,
                 ready.priority DESC, ready.created_at, ready.id
        FOR UPDATE OF task SKIP LOCKED
        LIMIT 1
    )
    UPDATE flakegraph_task AS task
    SET status = 'running',
        attempts = task.attempts + 1,
        lease_owner = %(worker_id)s,
        claim_token = %(claim_token)s,
        lease_expires_at = CURRENT_TIMESTAMP + %(lease_duration)s,
        started_at = COALESCE(task.started_at, CURRENT_TIMESTAMP),
        completed_at = NULL,
        progress_json = NULL,
        progress_updated_at = NULL,
        updated_at = CURRENT_TIMESTAMP
    FROM candidate
    WHERE task.id = candidate.id
    RETURNING task.*
"""
# The stages a document's processing can start at. A new document starts with
# preparation; a revision starts a kept document it reprocesses later on, with
# its earlier outputs named in the payload, and never with a dependency.
_INITIAL_DOCUMENT_STAGES = frozenset(
    {
        TaskStage.PREPARE_DOCUMENT,
        TaskStage.EXTRACT_DOCUMENT_CONTEXT,
        TaskStage.COMPACT_ENTITY_INVENTORY,
        TaskStage.REVALIDATE_DOCUMENT_CONTEXT,
    }
)
# How many resolved failures a run summary carries; the count is of all.
_RESOLVED_FAILURE_LIMIT = 50
# The stage a failed graph publication is kept under in a run's failure history.
_PUBLICATION_FAILURE_STAGE = "publish_graph"
_EXHAUSTED_TASK_RECOVERY_GRACE_SECONDS = 300
_POSTGRES_SESSION_OPTIONS = " ".join(
    (
        "-c lock_timeout=30000",
        "-c idle_in_transaction_session_timeout=60000",
        "-c client_connection_check_interval=1000",
    )
)
# PostgreSQL can race while two sessions concurrently create the same table's
# implicit composite type even with IF NOT EXISTS. A stable transaction-scoped
# advisory lock serializes only schema initialization, never normal task work.
_SCHEMA_INITIALIZATION_LOCK_ID = 2_359_716_441


class PostgresDistributedStore:
    """Implement durable task and artifact ports over one PostgreSQL database.

    One database keeps task completion and artifact references transactionally
    consistent while the interfaces remain separate at application boundaries.
    Payloads are compressed before insertion; large source systems can later use a
    different ``ArtifactStore`` without changing coordinator or worker semantics.
    """

    def __init__(
        self,
        dsn: str,
        *,
        compression_level: int = 6,
        max_artifact_bytes: int = 512 * 1024 * 1024,
        blob_store: BlobStore | None = None,
        max_connections: int = 2,
    ) -> None:
        """Store connection and bounded artifact settings without opening a socket.

        ``max_connections`` bounds the pool. Every method holds a connection
        only for its own transaction, so a worker's pool is sized from how
        many tasks it runs at once; a larger pool in every autoscaled pod would
        spend the shared database's connection budget for nothing.
        """

        if not dsn.strip():
            raise ValueError("PostgreSQL DSN must not be blank")
        if max_connections < 1:
            raise ValueError("PostgreSQL pool needs at least one connection")
        self.dsn = dsn
        self.compression_level = compression_level
        self.max_artifact_bytes = max_artifact_bytes
        self.blob_store = blob_store
        # A planner creates its run before concurrently staging sources, and a
        # worker can write only for a run it successfully leased. Remember those
        # database-proven identities so every immutable artifact does not spend a
        # separate transaction rechecking the same foreign-key precondition.
        self._known_run_ids: set[str] = set()
        self._known_run_ids_lock = Lock()
        try:
            configured_options = str(psycopg.conninfo.conninfo_to_dict(dsn).get("options") or "")
        except psycopg.ProgrammingError:
            raise ValueError("PostgreSQL DSN is malformed") from None
        session_options = " ".join(
            value for value in (configured_options, _POSTGRES_SESSION_OPTIONS) if value
        )
        # A worker performs several short transactions per task (claim, artifact
        # metadata, completion, and heartbeats). Reusing a small connection pool
        # removes a network/TLS handshake from each transition while preserving
        # the adapter's existing transaction boundary around every method call.
        self._pool = cast(
            "ConnectionPool[psycopg.Connection[dict[str, Any]]]",
            ConnectionPool(
                conninfo=dsn,
                min_size=0,
                max_size=max_connections,
                max_idle=60,
                timeout=10,
                open=True,
                kwargs={
                    "row_factory": dict_row,
                    "connect_timeout": 10,
                    "application_name": "flakegraph",
                    # Queue transactions should never wait indefinitely behind a
                    # lost worker. PostgreSQL enforces these limits even when a
                    # process disappears mid-COPY: blocked transitions retry after
                    # 30 seconds, abandoned transactions close after one minute,
                    # and active statements check their client socket every second.
                    # No statement timeout is imposed because streaming an initial
                    # 250,000-document plan can legitimately take several minutes.
                    "options": session_options,
                },
            ),
        )
        # Most stores live for the worker process lifetime. The finalizer still
        # closes retained connections when a short-lived CLI/test store is
        # collected, without requiring connection lifecycle on the port itself.
        self._pool_finalizer = finalize(self, self._pool.close)

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        """Yield a transaction or translate transient driver connectivity errors."""

        try:
            with self._pool.connection() as connection:
                yield connection
        except (psycopg.OperationalError, PoolTimeout):
            # Driver messages can echo a full conninfo string. Suppress exception
            # chaining so passwords cannot reappear in CLI or pod tracebacks.
            raise TaskStoreUnavailableError(
                "PostgreSQL coordination store is unavailable"
            ) from None

    def close(self) -> None:
        """Release pooled database connections held by this adapter instance.

        Worker processes normally rely on process shutdown, while tests and
        embedded callers may close a store deterministically. Closing is
        idempotent and does not alter durable run, task, or artifact state.
        """

        self._pool_finalizer()

    def initialize(self) -> None:
        """Create the schema once and make ordinary replica startup a cheap check.

        An empty database gets the schema; one at this build's version is left
        as it is. Any other version is refused: the statements describe one
        shape, and running them over another would leave it stamped current
        without the columns and constraints this build reads.
        """

        if self.blob_store is not None:
            self.blob_store.initialize()

        with self._connection() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (_SCHEMA_INITIALIZATION_LOCK_ID,),
            )
            installed_version = _installed_schema_version(connection)
            if installed_version == _SCHEMA_VERSION:
                return
            if installed_version is not None:
                raise RuntimeError(
                    f"coordination schema version {installed_version} is not this build's "
                    f"version {_SCHEMA_VERSION}; deploy the build that matches the database "
                    "or start from an empty one"
                )
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO flakegraph_schema_version (version) VALUES (%s)",
                (_SCHEMA_VERSION,),
            )

    def record_consumption(self, run_id: str, event: ConsumptionEvent) -> None:
        """Persist one provider event idempotently, including failed task attempts."""

        if event.job_id != run_id:
            raise ValueError("consumption event must belong to its durable run")
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO flakegraph_consumption_event (run_id, id, event_json)
                SELECT id, %s, %s FROM flakegraph_run WHERE id = %s
                ON CONFLICT (run_id, id) DO NOTHING
                """,
                (event.event_id, Jsonb(event.model_dump(mode="json")), run_id),
            )

    def consumption_events(self, run_id: str) -> list[ConsumptionEvent]:
        """Read the run's complete provider ledger in stable insertion order."""

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT event_json FROM flakegraph_consumption_event
                WHERE run_id = %s ORDER BY created_at, id
                """,
                (run_id,),
            ).fetchall()
        return [ConsumptionEvent.model_validate(row["event_json"]) for row in rows]

    @staticmethod
    def _lock_graph(connection: psycopg.Connection[dict[str, Any]], graph_id: str) -> None:
        """Serialize graph creation, retry and deletion reservation across processes."""

        connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (graph_id,))

    @staticmethod
    def _assert_graph_available(
        connection: psycopg.Connection[dict[str, Any]], graph_id: str
    ) -> None:
        """Reject mutations while a graph's durable deletion reservation is held."""

        if connection.execute(
            "SELECT 1 FROM flakegraph_graph_deletion WHERE graph_id = %s", (graph_id,)
        ).fetchone():
            raise ValueError(f"graph {graph_id} is being deleted")

    def create_run(self, run: RunDefinition) -> None:
        """Insert a planning run idempotently while rejecting identity collisions."""

        with self._connection() as connection:
            self._lock_graph(connection, run.graph_id)
            self._assert_graph_available(connection, run.graph_id)
            connection.execute(
                """
                INSERT INTO flakegraph_run (
                    id, graph_id, config_json, config_digest, status, priority_band
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    run.id,
                    run.graph_id,
                    Jsonb(run.config),
                    run.config_digest,
                    run.status.value,
                    run.priority_band.value,
                ),
            )
            existing = connection.execute(
                """
                SELECT graph_id, config_json, config_digest, status, priority_band
                FROM flakegraph_run
                WHERE id = %s
                """,
                (run.id,),
            ).fetchone()
            if existing is None:
                raise RuntimeError(f"failed to create distributed run {run.id}")
            if (
                str(existing["graph_id"]) != run.graph_id
                or existing["config_json"].get("submission_identity")
                != run.config.get("submission_identity")
                or str(existing["config_digest"]) != run.config_digest
                or str(existing["priority_band"]) != run.priority_band.value
            ):
                raise ValueError(f"run id {run.id} already exists with different inputs")
        self._remember_run_id(run.id)

    def add_tasks(self, run_id: str, tasks: list[TaskDefinition]) -> None:
        """Insert an arbitrary dependency graph after validating run ownership.

        The planner submits through ``add_initial_tasks``, whose streamed plan
        needs no graph validation; this general entry takes any acyclic plan
        and is what the store's contract tests build their shapes with.
        """

        if not tasks:
            raise ValueError("a distributed run requires at least one task")
        task_ids = {task.id for task in tasks}
        if len(task_ids) != len(tasks):
            raise ValueError("distributed task ids must be unique")
        if any(task.run_id != run_id for task in tasks):
            raise ValueError("all tasks must belong to the selected run")
        if any(set(task.dependency_ids) - task_ids for task in tasks):
            raise ValueError("task dependencies must reference tasks in the same plan")
        _validate_acyclic(tasks)
        with self._connection() as connection:
            status_row = connection.execute(
                "SELECT status, priority_band FROM flakegraph_run WHERE id = %s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if status_row is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            if str(status_row["status"]) != RunStatus.PLANNING.value:
                raise ValueError("tasks may only be added while a run is planning")
            band = PriorityBand(str(status_row["priority_band"]))
            count_row = connection.execute(
                "SELECT COUNT(*) AS count FROM flakegraph_task WHERE run_id = %s",
                (run_id,),
            ).fetchone()
            if count_row is None:
                raise RuntimeError(f"failed to inspect task plan for run {run_id}")
            existing_count = cast(int, count_row["count"])
            if existing_count:
                raise ValueError(f"distributed run {run_id} already has a task plan")
            # COPY keeps one round trip and one server-side parse for the complete
            # initial plan. At hundreds of thousands of documents, issuing one
            # INSERT per task and dependency would otherwise dominate submission.
            with connection.cursor().copy(
                """
                COPY flakegraph_task (
                    id, run_id, stage, scope_id, payload_json, status,
                    priority, max_attempts, remaining_dependencies
                ) FROM STDIN
                """
            ) as task_copy:
                for task in tasks:
                    task_copy.write_row(
                        (
                            task.id,
                            task.run_id,
                            task.stage.value,
                            task.scope_id,
                            Jsonb(task.payload),
                            TaskStatus.QUEUED.value,
                            task_priority(task.stage, band),
                            task.max_attempts,
                            len(task.dependency_ids),
                        )
                    )
            dependencies = (
                (task.id, dependency_id) for task in tasks for dependency_id in task.dependency_ids
            )
            first_dependency = next(dependencies, None)
            if first_dependency is not None:
                with connection.cursor().copy(
                    """
                    COPY flakegraph_task_dependency (task_id, depends_on_task_id)
                    FROM STDIN
                    """
                ) as dependency_copy:
                    dependency_copy.write_row(first_dependency)
                    for dependency in dependencies:
                        dependency_copy.write_row(dependency)

    def add_initial_tasks(
        self,
        run_id: str,
        tasks: Iterable[TaskDefinition],
    ) -> None:
        """Stream the restricted initial plan through short PostgreSQL transactions.

        Unlike ``add_tasks``, this entry point intentionally accepts only
        independent document tasks - preparation, or the seed of a document a
        revision reprocesses - and one final barrier. That known
        acyclic shape can be validated row by row, letting PostgreSQL's ``COPY``
        consume a 250,000-document plan without an equally large Python list. The
        run remains in ``planning`` until every batch is committed, so workers
        cannot observe a partial plan. Crucially, source uploads happen while no
        database transaction or pooled connection is retained.
        """

        with self._connection() as connection:
            status_row = connection.execute(
                "SELECT status FROM flakegraph_run WHERE id = %s",
                (run_id,),
            ).fetchone()
            if status_row is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            if str(status_row["status"]) != RunStatus.PLANNING.value:
                raise ValueError("tasks may only be added while a run is planning")
            # Existing rows are a resumable prefix left by an interrupted
            # coordinator. Every replayed row is checked for exact identity in
            # ``_copy_initial_task_batch`` before missing rows are inserted.

        document_count = 0
        final_count = 0
        final_seen = False
        final_inherits = False
        batch: list[TaskDefinition] = []
        for task in tasks:
            document_increment, final_increment = _validate_initial_task(
                task,
                run_id,
                final_seen,
            )
            document_count += document_increment
            final_count += final_increment
            final_seen = bool(final_count)
            if final_increment:
                final_inherits = bool(task.payload.get("inherit"))
            batch.append(task)
            if len(batch) >= _INITIAL_TASK_COPY_BATCH_SIZE:
                self._copy_initial_task_batch(run_id, batch)
                batch = []
        if batch:
            self._copy_initial_task_batch(run_id, batch)
        # A revision that only drops documents prepares nothing: its
        # finalizer builds the graph from what it inherits.
        if final_count != 1 or (document_count == 0 and not final_inherits):
            # Batches were committed as they streamed; an invalid plan leaves
            # none of them behind, so the planner sees a run with no tasks
            # and cancels it rather than a half-plan a worker could claim.
            with self._connection() as connection:
                connection.execute(
                    "DELETE FROM flakegraph_task WHERE run_id = %s",
                    (run_id,),
                )
            raise ValueError(
                "initial plan requires exactly one final task, and document tasks "
                "unless the final task inherits documents"
            )
        with self._connection() as connection:
            counts = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE stage <> %s) AS documents,
                       COUNT(*) FILTER (WHERE stage = %s) AS finalizers
                FROM flakegraph_task
                WHERE run_id = %s
                """,
                (
                    TaskStage.FINALIZE_GRAPH.value,
                    TaskStage.FINALIZE_GRAPH.value,
                    run_id,
                ),
            ).fetchone()
            if (
                counts is None
                or int(counts["total"]) != document_count + final_count
                or int(counts["documents"]) != document_count
                or int(counts["finalizers"]) != 1
            ):
                raise ValueError("streamed initial plan conflicts with existing planning rows")
        # COPY can add hundreds of thousands of rows before autovacuum has a
        # chance to refresh planner statistics. Analyze while the run is still
        # invisible so the first worker immediately selects the ordered claim
        # index instead of sorting the complete ready queue.
        with self._connection() as connection:
            connection.execute("ANALYZE flakegraph_task")

    def _copy_initial_task_batch(
        self,
        run_id: str,
        tasks: list[TaskDefinition],
    ) -> None:
        """Commit one bounded initial-plan batch while the run stays invisible."""

        with self._connection() as connection:
            status = connection.execute(
                "SELECT status, priority_band FROM flakegraph_run WHERE id = %s",
                (run_id,),
            ).fetchone()
            if status is None or str(status["status"]) != RunStatus.PLANNING.value:
                raise ValueError("run left planning state while initial tasks were streamed")
            band = PriorityBand(str(status["priority_band"]))
            connection.execute(
                """
                CREATE TEMPORARY TABLE flakegraph_initial_task_batch (
                    id TEXT, run_id TEXT, stage TEXT, scope_id TEXT,
                    payload_json JSONB, status TEXT, priority INTEGER,
                    max_attempts INTEGER, remaining_dependencies INTEGER
                ) ON COMMIT DROP
                """
            )
            with connection.cursor().copy(
                """
                COPY flakegraph_initial_task_batch (
                    id, run_id, stage, scope_id, payload_json, status,
                    priority, max_attempts, remaining_dependencies
                ) FROM STDIN
                """
            ) as task_copy:
                for task in tasks:
                    task_copy.write_row(
                        (
                            task.id,
                            task.run_id,
                            task.stage.value,
                            task.scope_id,
                            Jsonb(task.payload),
                            TaskStatus.QUEUED.value,
                            task_priority(task.stage, band),
                            task.max_attempts,
                            0,
                        )
                    )
            conflict = connection.execute(
                """
                SELECT incoming.id
                FROM flakegraph_initial_task_batch AS incoming
                JOIN flakegraph_task AS existing
                  ON existing.id = incoming.id
                  OR (
                      existing.run_id = incoming.run_id
                      AND existing.stage = incoming.stage
                      AND existing.scope_id = incoming.scope_id
                  )
                WHERE existing.id <> incoming.id
                   OR existing.run_id <> incoming.run_id
                   OR existing.stage <> incoming.stage
                   OR existing.scope_id <> incoming.scope_id
                   OR existing.payload_json <> incoming.payload_json
                   OR existing.priority <> incoming.priority
                   OR existing.max_attempts <> incoming.max_attempts
                LIMIT 1
                """
            ).fetchone()
            if conflict is not None:
                raise ValueError(
                    f"streamed initial task conflicts with existing row: {conflict['id']}"
                )
            connection.execute(
                """
                INSERT INTO flakegraph_task (
                    id, run_id, stage, scope_id, payload_json, status,
                    priority, max_attempts, remaining_dependencies
                )
                SELECT id, run_id, stage, scope_id, payload_json, status,
                       priority, max_attempts, remaining_dependencies
                FROM flakegraph_initial_task_batch
                ON CONFLICT (id) DO NOTHING
                """
            )

    def activate_run(self, run_id: str) -> None:
        """Transition a fully planned run to queued exactly once."""

        with self._connection() as connection:
            updated = connection.execute(
                """
                UPDATE flakegraph_run
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                  AND status = %s
                  AND EXISTS (
                      SELECT 1 FROM flakegraph_task WHERE run_id = flakegraph_run.id
                  )
                RETURNING id
                """,
                (RunStatus.QUEUED.value, run_id, RunStatus.PLANNING.value),
            ).fetchone()
            if updated is None:
                raise ValueError("run must be planning with a non-empty task graph")

    def record_served_configuration(self, stages: set[TaskStage], config_digest: str) -> None:
        """Declare this fleet's configuration for every stage it claims."""

        if not stages:
            return
        with self._connection() as connection:
            for stage in sorted(stage.value for stage in stages):
                connection.execute(
                    """
                    INSERT INTO flakegraph_worker_fleet (stage, config_digest)
                    VALUES (%s, %s)
                    ON CONFLICT (stage) DO UPDATE
                    SET config_digest = EXCLUDED.config_digest,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (stage, config_digest),
                )

    def claim_task(
        self,
        worker_id: str,
        stages: set[TaskStage],
        lease_duration: timedelta,
        expected_config_digest: str | None = None,
        skip_dependency_outputs_for_stages: set[TaskStage] | None = None,
    ) -> TaskLease | None:
        """Claim one dependency-ready task using row locks that skip competitors.

        Interactive runs are served before bulk ones. Within a band the task
        goes to the run with the fewest live leases in the stages this worker
        serves, so runs share a pool equally however large each is; the stage
        ladder, then age, decide among the rest. See ``_CLAIM_TASK``.

        A skipped stage retains the exact SQL dependency barrier but does not load
        prerequisite IDs or outputs into Python. Spark finalization reads run-scoped
        immutable object prefixes, so transferring millions of redundant artifact
        references to its driver would add latency and an avoidable memory spike.
        """

        if not worker_id.strip():
            raise ValueError("worker id must not be blank")
        if not stages:
            raise ValueError("worker must accept at least one task stage")
        _validate_duration(lease_duration, "lease duration")
        with self._connection() as connection:
            row = connection.execute(
                _CLAIM_TASK,
                {
                    "stages": [stage.value for stage in sorted(stages, key=str)],
                    "config_digest": expected_config_digest,
                    "candidates_per_stage": _CLAIM_CANDIDATES_PER_STAGE,
                    "worker_id": worker_id,
                    "claim_token": str(uuid4()),
                    "lease_duration": lease_duration,
                },
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE flakegraph_run
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = %s
                """,
                (RunStatus.RUNNING.value, row["run_id"], RunStatus.QUEUED.value),
            )
            claimed_stage = TaskStage(str(row["stage"]))
            skip_outputs = claimed_stage in (skip_dependency_outputs_for_stages or set())
            dependency_outputs = (
                {} if skip_outputs else self._dependency_outputs(connection, str(row["id"]))
            )
            self._remember_run_id(str(row["run_id"]))
            return TaskLease(
                task=_task_definition(row, dependency_outputs.keys()),
                worker_id=worker_id,
                attempt=int(row["attempts"]),
                claim_token=str(row["claim_token"]),
                lease_expires_at=row["lease_expires_at"],
                dependency_outputs=dependency_outputs,
            )

    def heartbeat(
        self, task_id: str, worker_id: str, lease_duration: timedelta, *, claim_token: str
    ) -> None:
        """Extend an owned running lease without changing its attempt count."""

        _validate_duration(lease_duration, "lease duration")
        with self._connection() as connection:
            row = connection.execute(
                """
                UPDATE flakegraph_task
                SET lease_expires_at = CURRENT_TIMESTAMP + %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = %s AND lease_owner = %s AND claim_token = %s
                RETURNING id
                """,
                (lease_duration, task_id, TaskStatus.RUNNING.value, worker_id, claim_token),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"worker {worker_id} no longer owns task {task_id}")

    def report_task_progress(
        self,
        task_id: str,
        worker_id: str,
        progress: TaskProgress,
        *,
        claim_token: str,
    ) -> None:
        """Persist bounded progress only while the worker owns the running lease."""

        with self._connection() as connection:
            row = connection.execute(
                """
                UPDATE flakegraph_task
                SET progress_json = %s,
                    progress_updated_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = %s AND lease_owner = %s AND claim_token = %s
                RETURNING run_id
                """,
                (
                    Jsonb(progress.model_dump(mode="json")),
                    task_id,
                    TaskStatus.RUNNING.value,
                    worker_id,
                    claim_token,
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"worker {worker_id} no longer owns task {task_id}")
            connection.execute(
                "UPDATE flakegraph_run SET updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (row["run_id"],),
            )

    def claim_publication(
        self,
        worker_id: str,
        lease_duration: timedelta,
    ) -> PublicationLease | None:
        """Lease one durable external-publication command for a finalizer worker."""

        if not worker_id.strip():
            raise ValueError("worker id must not be blank")
        _validate_duration(lease_duration, "publication lease duration")
        with self._connection() as connection:
            expired = connection.execute(
                """
                UPDATE flakegraph_publication
                SET status = %s, lease_owner = NULL, lease_expires_at = NULL,
                    last_error_json = %s, completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = %s
                  AND lease_expires_at < CURRENT_TIMESTAMP
                  AND attempts >= max_attempts
                RETURNING run_id, id
                """,
                (
                    PublicationStatus.FAILED.value,
                    Jsonb(
                        {
                            "type": "LeaseExpired",
                            "message": "publication lease expired",
                        }
                    ),
                    PublicationStatus.RUNNING.value,
                ),
            ).fetchall()
            for item in expired:
                connection.execute(
                    """
                    UPDATE flakegraph_run
                    SET status = %s, error_json = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND status = %s
                    """,
                    (
                        RunStatus.FAILED.value,
                        Jsonb(
                            {
                                "publication_id": str(item["id"]),
                                "type": "LeaseExpired",
                                "message": "publication lease expired",
                            }
                        ),
                        item["run_id"],
                        RunStatus.RUNNING.value,
                    ),
                )
            row = connection.execute(
                """
                WITH candidate AS (
                    SELECT publication.id
                    FROM flakegraph_publication AS publication
                    JOIN flakegraph_run AS run ON run.id = publication.run_id
                    WHERE run.status = %s
                      AND publication.available_at <= CURRENT_TIMESTAMP
                      AND (
                          publication.status = %s
                          OR (
                              publication.status = %s
                              AND publication.lease_expires_at < CURRENT_TIMESTAMP
                              AND publication.attempts < publication.max_attempts
                          )
                      )
                    ORDER BY publication.generation, publication.id
                    FOR UPDATE OF publication SKIP LOCKED
                    LIMIT 1
                )
                UPDATE flakegraph_publication AS publication
                SET status = %s,
                    attempts = publication.attempts + 1,
                    lease_owner = %s,
                    lease_expires_at = CURRENT_TIMESTAMP + %s,
                    updated_at = CURRENT_TIMESTAMP
                FROM candidate
                WHERE publication.id = candidate.id
                RETURNING publication.*
                """,
                (
                    RunStatus.RUNNING.value,
                    PublicationStatus.QUEUED.value,
                    PublicationStatus.RUNNING.value,
                    PublicationStatus.RUNNING.value,
                    worker_id,
                    lease_duration,
                ),
            ).fetchone()
            return self._publication_lease(row, worker_id) if row is not None else None

    def publish_claimed(
        self,
        publication_id: str,
        worker_id: str,
        publish: Callable[[PublicationLease], None],
    ) -> None:
        """Publish externally without holding coordination locks across the callback."""

        # Read and validate the fencing identity in a short transaction. The
        # generation and attempt are checked again when committing completion, so
        # cancellation and retries remain responsive while publication is slow.
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT publication.*
                FROM flakegraph_publication AS publication
                JOIN flakegraph_run AS run ON run.id = publication.run_id
                WHERE publication.id = %s
                  AND publication.status = %s
                  AND publication.lease_owner = %s
                  AND publication.lease_expires_at >= CURRENT_TIMESTAMP
                  AND run.status = %s
                """,
                (
                    publication_id,
                    PublicationStatus.RUNNING.value,
                    worker_id,
                    RunStatus.RUNNING.value,
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    f"worker {worker_id} no longer owns publication {publication_id}"
                )
            lease = self._publication_lease(row, worker_id)
        publish(lease)
        with self._connection() as connection:
            completed = connection.execute(
                """
                UPDATE flakegraph_publication AS publication
                SET status = %s, lease_owner = NULL, lease_expires_at = NULL,
                    completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                FROM flakegraph_run AS run
                WHERE publication.id = %s
                  AND publication.run_id = run.id
                  AND publication.status = %s
                  AND publication.lease_owner = %s
                  AND publication.attempts = %s
                  AND publication.generation = %s
                  AND run.status = %s
                RETURNING publication.run_id
                """,
                (
                    PublicationStatus.SUCCEEDED.value,
                    publication_id,
                    PublicationStatus.RUNNING.value,
                    worker_id,
                    lease.attempt,
                    lease.generation,
                    RunStatus.RUNNING.value,
                ),
            ).fetchone()
            if completed is None:
                raise RuntimeError(
                    f"worker {worker_id} no longer owns publication {publication_id}"
                )
            self._complete_run_after_final_barrier(connection, lease.run_id)

    def commit_publication(self, lease: PublicationLease, commit: Callable[[], None]) -> None:
        """Serialize an already staged destination swap with cancellation and replacement.

        The immutable batch and destination staging already exist. Holding the run,
        publication and graph-head rows only during the final replacement closes
        the ownership-check/write race of the in-flight local finalizer.
        """

        with self._connection() as connection:
            owned = connection.execute(
                """
                SELECT publication.id
                FROM flakegraph_publication AS publication
                JOIN flakegraph_run AS run ON run.id = publication.run_id
                JOIN flakegraph_graph_head AS head ON head.graph_id = run.graph_id
                WHERE publication.id = %s AND publication.run_id = %s
                  AND publication.status = %s AND publication.lease_owner = %s
                  AND publication.attempts = %s AND publication.generation = %s
                  AND publication.lease_expires_at >= CURRENT_TIMESTAMP
                  AND run.status = %s AND head.version_id = publication.artifact_id
                FOR UPDATE OF publication, run, head
                """,
                (
                    lease.id,
                    lease.run_id,
                    PublicationStatus.RUNNING.value,
                    lease.worker_id,
                    lease.attempt,
                    lease.generation,
                    RunStatus.RUNNING.value,
                ),
            ).fetchone()
            if owned is None:
                raise RuntimeError(
                    f"worker {lease.worker_id} no longer owns publication {lease.id}"
                )
            # The final destination transaction may outlast an ordinary queue
            # transaction's idle budget. Losing this connection would release
            # the fence while an external commit was still in progress. Limit
            # the exemption to this transaction; statement/lock timeouts remain.
            connection.execute("SET LOCAL idle_in_transaction_session_timeout = 0")
            commit()
            connection.execute(
                """
                UPDATE flakegraph_publication
                SET status = %s, lease_owner = NULL, lease_expires_at = NULL,
                    completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (PublicationStatus.SUCCEEDED.value, lease.id),
            )
            self._complete_run_after_final_barrier(connection, lease.run_id)

    def heartbeat_publication(
        self,
        publication_id: str,
        worker_id: str,
        lease_duration: timedelta,
    ) -> None:
        """Extend an owned publication lease while its destination write is active."""

        _validate_duration(lease_duration, "publication lease duration")
        with self._connection() as connection:
            refreshed = connection.execute(
                """
                UPDATE flakegraph_publication
                SET lease_expires_at = CURRENT_TIMESTAMP + %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = %s AND lease_owner = %s
                  AND lease_expires_at >= CURRENT_TIMESTAMP
                RETURNING id
                """,
                (
                    lease_duration,
                    publication_id,
                    PublicationStatus.RUNNING.value,
                    worker_id,
                ),
            ).fetchone()
            if refreshed is None:
                raise RuntimeError(
                    f"worker {worker_id} no longer owns publication {publication_id}"
                )

    def fail_publication(
        self,
        publication_id: str,
        worker_id: str,
        error: dict[str, Any],
        retry_delay: timedelta,
    ) -> None:
        """Retry an owned publication or terminally fail its parent run."""

        if retry_delay < timedelta(0):
            raise ValueError("retry delay must not be negative")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM flakegraph_publication
                WHERE id = %s AND status = %s AND lease_owner = %s
                FOR UPDATE
                """,
                (publication_id, PublicationStatus.RUNNING.value, worker_id),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    f"worker {worker_id} no longer owns publication {publication_id}"
                )
            exhausted = int(row["attempts"]) >= int(row["max_attempts"])
            next_status = PublicationStatus.FAILED if exhausted else PublicationStatus.QUEUED
            connection.execute(
                """
                UPDATE flakegraph_publication
                SET status = %s, available_at = CURRENT_TIMESTAMP + %s,
                    lease_owner = NULL, lease_expires_at = NULL,
                    last_error_json = %s,
                    completed_at = CASE WHEN %s = 'failed'
                                        THEN CURRENT_TIMESTAMP ELSE NULL END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (
                    next_status.value,
                    retry_delay,
                    Jsonb(error),
                    next_status.value,
                    publication_id,
                ),
            )
            if exhausted:
                connection.execute(
                    """
                    UPDATE flakegraph_run
                    SET status = %s, error_json = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND status = %s
                    """,
                    (
                        RunStatus.FAILED.value,
                        Jsonb({"publication_id": publication_id, **error}),
                        row["run_id"],
                        RunStatus.RUNNING.value,
                    ),
                )

    @staticmethod
    def _publication_lease(row: dict[str, Any], worker_id: str) -> PublicationLease:
        """Convert a durable outbox row into the callback-facing lease model."""

        return PublicationLease(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            artifact_id=str(row["artifact_id"]),
            task_payload=cast(dict[str, Any], row["task_payload_json"]),
            worker_id=worker_id,
            attempt=int(row["attempts"]),
            generation=int(row["generation"]),
            lease_expires_at=row["lease_expires_at"],
        )

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
        """Commit outputs and dynamically discovered work in one transaction.

        A running parent remains unfinished while follow-up rows and their local
        dependency edges are inserted. Marking the parent successful only
        afterward closes the race in which the run-wide finalizer could otherwise
        observe no unfinished work between parent completion and dynamic fan-out.
        """

        follow_ups = follow_up_tasks or []
        if len(set(output_artifact_ids)) != len(output_artifact_ids):
            raise ValueError("task output artifact ids must be unique")
        with self._connection() as connection:
            task = self._owned_task(connection, task_id, worker_id, claim_token)
            run_id = str(task["run_id"])
            output_rows: list[dict[str, Any]] = []
            if output_artifact_ids:
                output_rows = connection.execute(
                    """
                    SELECT id, kind, media_type FROM flakegraph_artifact
                    WHERE run_id = %s AND id = ANY(%s)
                    """,
                    (run_id, output_artifact_ids),
                ).fetchall()
                if {str(row["id"]) for row in output_rows} != set(output_artifact_ids):
                    raise ValueError("task outputs must reference artifacts from the same run")
            if follow_ups:
                self._insert_follow_up_tasks(
                    connection,
                    run_id,
                    task_id,
                    follow_ups,
                    barrier_task_id,
                )
            elif barrier_task_id is not None:
                raise ValueError("barrier_task_id requires at least one follow-up task")
            for position, artifact_id in enumerate(output_artifact_ids):
                connection.execute(
                    """
                    INSERT INTO flakegraph_task_output (task_id, position, artifact_id)
                    VALUES (%s, %s, %s)
                    """,
                    (task_id, position, artifact_id),
                )
            connection.execute(
                """
                UPDATE flakegraph_task
                SET status = %s, lease_owner = NULL, lease_expires_at = NULL,
                    completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (TaskStatus.SUCCEEDED.value, task_id),
            )
            # Make dependent work ready in the same transaction that publishes
            # this task's outputs. Worker and autoscaler polls can then use one
            # indexed integer predicate instead of repeatedly joining the entire
            # dependency frontier for a large corpus.
            connection.execute(
                """
                UPDATE flakegraph_task AS dependent
                SET remaining_dependencies = dependent.remaining_dependencies - 1,
                    updated_at = CURRENT_TIMESTAMP
                FROM flakegraph_task_dependency AS dependency
                WHERE dependency.depends_on_task_id = %s
                  AND dependent.id = dependency.task_id
                  AND dependent.status = %s
                  AND dependent.stage <> %s
                  AND dependent.remaining_dependencies > 0
                """,
                (
                    task_id,
                    TaskStatus.QUEUED.value,
                    TaskStage.FINALIZE_GRAPH.value,
                ),
            )
            if str(task["stage"]) == TaskStage.FINALIZE_GRAPH.value:
                self._publish_graph_version(connection, run_id, output_rows)
                self._enqueue_graph_publication(connection, task, output_rows)
                self._complete_run_after_final_barrier(connection, run_id)

    def _enqueue_graph_publication(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        task: dict[str, Any],
        output_rows: list[dict[str, Any]],
    ) -> None:
        """Atomically append an external-destination command with final task success."""

        payload = cast(dict[str, Any], task["payload_json"])
        output = payload.get("output")
        if not isinstance(output, dict):
            return
        batch_output = any(
            row["media_type"] == "application/json"
            and str(row["kind"]) == ArtifactKind.GRAPH_RESULT.value
            for row in output_rows
        )
        if output.get("provider") != "snowflake_bulk" and not batch_output:
            return
        graph_outputs = [
            row for row in output_rows if str(row["kind"]) == ArtifactKind.GRAPH_RESULT.value
        ]
        if len(graph_outputs) != 1:
            raise ValueError("Snowflake publication requires exactly one graph result")
        publication_id = stable_id("publication", str(task["run_id"]), str(task["id"]))
        connection.execute(
            """
            INSERT INTO flakegraph_publication (
                id, run_id, task_id, artifact_id, task_payload_json,
                status, max_attempts
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                publication_id,
                task["run_id"],
                task["id"],
                graph_outputs[0]["id"],
                Jsonb(payload),
                PublicationStatus.QUEUED.value,
                task["max_attempts"],
            ),
        )

    def _complete_run_after_final_barrier(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        run_id: str,
    ) -> None:
        """Mark a run successful only after its leased final barrier succeeds.

        The initial planner creates exactly one final task. Its readiness query
        uses the run-scoped unfinished-task index instead of storing one dependency
        edge per document. Consequently ordinary preparation and extraction
        completions never lock one shared barrier row. Restricting the serialized
        run transition to finalization keeps millions of independent queue
        completions concurrent while retaining a defensive unfinished-task check
        for malformed plans created through the low-level store API.
        """

        connection.execute(
            "SELECT id FROM flakegraph_run WHERE id = %s FOR UPDATE",
            (run_id,),
        ).fetchone()
        unfinished = connection.execute(
            """
            SELECT 1 FROM flakegraph_task
            WHERE run_id = %s AND status <> %s
            LIMIT 1
            """,
            (run_id, TaskStatus.SUCCEEDED.value),
        ).fetchone()
        pending_publication = connection.execute(
            """
            SELECT 1 FROM flakegraph_publication
            WHERE run_id = %s AND status <> %s
            LIMIT 1
            """,
            (run_id, PublicationStatus.SUCCEEDED.value),
        ).fetchone()
        if unfinished is None and pending_publication is None:
            connection.execute(
                """
                UPDATE flakegraph_run
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = %s
                """,
                (RunStatus.SUCCEEDED.value, run_id, RunStatus.RUNNING.value),
            )

    def _publish_graph_version(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        run_id: str,
        output_rows: list[dict[str, Any]],
    ) -> None:
        """Atomically activate the final task's immutable graph artifact.

        Metadata-only final tasks may emit no graph result. A real finalizer emits
        exactly one; conflicting outputs are rejected before task completion can
        commit, so readers retain the previous complete graph version.
        """

        graph_outputs = [
            row for row in output_rows if str(row["kind"]) == ArtifactKind.GRAPH_RESULT.value
        ]
        if not graph_outputs:
            return
        if len(graph_outputs) != 1:
            raise ValueError("finalize_graph must publish exactly one graph result")
        artifact = graph_outputs[0]
        run = connection.execute(
            "SELECT graph_id FROM flakegraph_run WHERE id = %s",
            (run_id,),
        ).fetchone()
        if run is None:
            raise KeyError(f"unknown distributed run: {run_id}")
        artifact_id = str(artifact["id"])
        graph_id = str(run["graph_id"])
        connection.execute(
            """
            INSERT INTO flakegraph_graph_version (
                id, graph_id, run_id, artifact_id, media_type
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE
            SET artifact_id = EXCLUDED.artifact_id,
                media_type = EXCLUDED.media_type
            """,
            (artifact_id, graph_id, run_id, artifact_id, str(artifact["media_type"])),
        )
        connection.execute(
            """
            INSERT INTO flakegraph_graph_head (graph_id, version_id, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (graph_id) DO UPDATE
            SET version_id = EXCLUDED.version_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (graph_id, artifact_id),
        )

    def _insert_follow_up_tasks(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        run_id: str,
        parent_task_id: str,
        tasks: list[TaskDefinition],
        barrier_task_id: str | None,
    ) -> None:
        """Insert a validated dynamic subgraph before its parent can succeed.

        Context extraction publishes a small two-level DAG: independent window
        tasks followed by one document compaction task. The complete subgraph is
        inserted in the parent's completion transaction. Run-wide final readiness
        therefore sees either the running parent or all of its queued children and
        can never observe a transiently complete run.
        """

        if barrier_task_id is None:
            raise ValueError("dynamic follow-up tasks require a barrier_task_id")
        parent_stage_row = connection.execute(
            "SELECT stage FROM flakegraph_task WHERE id = %s",
            (parent_task_id,),
        ).fetchone()
        if parent_stage_row is None:
            raise ValueError("dynamic follow-up parent task does not exist")
        parent_stage = TaskStage(str(parent_stage_row["stage"]))
        _validate_dynamic_follow_ups(
            run_id,
            parent_task_id,
            parent_stage,
            tasks,
        )
        barrier = connection.execute(
            """
            SELECT id, run_id, stage, status
            FROM flakegraph_task
            WHERE id = %s
            """,
            (barrier_task_id,),
        ).fetchone()
        if (
            barrier is None
            or str(barrier["run_id"]) != run_id
            or str(barrier["stage"]) != TaskStage.FINALIZE_GRAPH.value
            or str(barrier["status"]) != TaskStatus.QUEUED.value
        ):
            raise ValueError("dynamic follow-ups require a queued final barrier in the same run")
        # Discovered work is claimed in the band its run was submitted at.
        band_row = connection.execute(
            "SELECT priority_band FROM flakegraph_run WHERE id = %s",
            (run_id,),
        ).fetchone()
        if band_row is None:
            raise KeyError(f"unknown distributed run: {run_id}")
        band = PriorityBand(str(band_row["priority_band"]))
        # A long document may create hundreds of windows. COPY keeps task and edge
        # creation to two protocol exchanges. The finalizer observes all unfinished
        # tasks in the run, so dynamic work needs no explicit edge to a shared final
        # row and document completions never serialize on a global lock.
        with connection.cursor().copy(
            """
            COPY flakegraph_task (
                id, run_id, stage, scope_id, payload_json, status,
                priority, max_attempts, remaining_dependencies
            ) FROM STDIN
            """
        ) as task_copy:
            for follow_up in tasks:
                task_copy.write_row(
                    (
                        follow_up.id,
                        follow_up.run_id,
                        follow_up.stage.value,
                        follow_up.scope_id,
                        Jsonb(follow_up.payload),
                        TaskStatus.QUEUED.value,
                        task_priority(follow_up.stage, band),
                        follow_up.max_attempts,
                        len(follow_up.dependency_ids),
                    )
                )
        with connection.cursor().copy(
            """
            COPY flakegraph_task_dependency (task_id, depends_on_task_id)
            FROM STDIN
            """
        ) as dependency_copy:
            for follow_up in tasks:
                for dependency_id in follow_up.dependency_ids:
                    dependency_copy.write_row((follow_up.id, dependency_id))

    def fail_task(
        self,
        task_id: str,
        worker_id: str,
        error: dict[str, Any],
        retry_delay: timedelta,
        *,
        claim_token: str,
    ) -> None:
        """Retry an owned task within its budget or fail the run terminally."""

        if retry_delay < timedelta(0):
            raise ValueError("retry delay must not be negative")
        with self._connection() as connection:
            task = self._owned_task(connection, task_id, worker_id, claim_token)
            exhausted = int(task["attempts"]) >= int(task["max_attempts"])
            next_status = TaskStatus.FAILED if exhausted else TaskStatus.QUEUED
            connection.execute(
                """
                UPDATE flakegraph_task
                SET status = %s,
                    available_at = CURRENT_TIMESTAMP + %s,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    last_error_json = %s,
                    completed_at = CASE WHEN %s = 'failed' THEN CURRENT_TIMESTAMP ELSE NULL END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (next_status.value, retry_delay, Jsonb(error), next_status.value, task_id),
            )
            if exhausted:
                connection.execute(
                    """
                    UPDATE flakegraph_run
                    SET status = %s, error_json = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND status IN (%s, %s)
                    """,
                    (
                        RunStatus.FAILED.value,
                        Jsonb({"task_id": task_id, **error}),
                        task["run_id"],
                        RunStatus.QUEUED.value,
                        RunStatus.RUNNING.value,
                    ),
                )

    @contextmanager
    def _deletion_executor(self, graph_id: str) -> Iterator[None]:
        """Hold one deletion executor across blob I/O without an open transaction.

        A dedicated session avoids consuming the task pool while external storage
        is busy. Closing it releases the advisory lock even when deletion fails;
        the committed reservation remains available for a subsequent retry.
        """

        try:
            with psycopg.connect(
                self.dsn, autocommit=True, connect_timeout=10, row_factory=dict_row
            ) as connection:
                row = connection.execute(
                    "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS locked",
                    (f"flakegraph:delete:{graph_id}",),
                ).fetchone()
                if row is None or not row["locked"]:
                    raise ValueError(f"graph {graph_id} deletion is already executing")
                yield
        except psycopg.OperationalError:
            raise TaskStoreUnavailableError(
                "PostgreSQL coordination store is unavailable"
            ) from None

    def delete_graph(self, graph_id: str) -> GraphDeletion:
        """Reserve a terminal graph and remove its objects and coordination rows."""

        with self._deletion_executor(graph_id):
            return self._delete_reserved_graph(graph_id)

    def _delete_reserved_graph(self, graph_id: str) -> GraphDeletion:
        """Remove a graph for good: every run of it, and everything they stored.

        A graph is deleted whole, never one run at a time, because a revision
        is built from the stored outputs of the versions before it. Its runs'
        objects go first - payloads, stage outputs, Spark checkpoints and the
        published parquet all live under ``<run id>/`` in the blob store - and
        then every row naming those runs, in one transaction. Interrupted
        between the two, the durable reservation rejects submissions and retries,
        and deleting the graph again finishes the job. A graph with queued or running
        work is refused:
        a worker would write into what is being removed.
        """

        deletion_token = str(uuid4())
        with self._connection() as connection:
            self._lock_graph(connection, graph_id)
            runs = connection.execute(
                "SELECT id, status FROM flakegraph_run WHERE graph_id = %s ORDER BY created_at",
                (graph_id,),
            ).fetchall()
            run_ids = [str(row["id"]) for row in runs]
            active = [
                str(row["id"])
                for row in runs
                if str(row["status"])
                in {RunStatus.PLANNING.value, RunStatus.QUEUED.value, RunStatus.RUNNING.value}
            ]
            if active:
                raise ValueError(
                    f"graph {graph_id} still has work under way ({', '.join(active)}); "
                    "cancel it before deleting the graph"
                )
            # A failed run can still have sibling work finishing. Lock those
            # rows before checking leases; an expired holder cannot renew after
            # its row is cancelled by this reservation.
            for table in ("flakegraph_task", "flakegraph_publication"):
                live = connection.execute(
                    f"SELECT id, lease_expires_at >= CURRENT_TIMESTAMP AS live FROM {table} "
                    "WHERE run_id = ANY(%s) AND status = 'running' FOR UPDATE",
                    (run_ids,),
                ).fetchall()
                if any(row["live"] for row in live):
                    raise ValueError(
                        f"graph {graph_id} still has work under way; wait for its leases"
                    )
                connection.execute(
                    f"UPDATE {table} SET status = 'cancelled', lease_owner = NULL, "
                    "lease_expires_at = NULL, updated_at = CURRENT_TIMESTAMP "
                    "WHERE run_id = ANY(%s) AND status IN ('queued', 'running')",
                    (run_ids,),
                )
            # Task transitions and cancellation lock work rows before the run.
            # Recheck terminal status only after acquiring the same row order.
            current = connection.execute(
                "SELECT status FROM flakegraph_run WHERE graph_id = %s FOR UPDATE",
                (graph_id,),
            ).fetchall()
            if any(
                str(row["status"])
                in {RunStatus.PLANNING.value, RunStatus.QUEUED.value, RunStatus.RUNNING.value}
                for row in current
            ):
                raise ValueError(f"graph {graph_id} still has work under way")
            connection.execute(
                "INSERT INTO flakegraph_graph_deletion (graph_id, token) VALUES (%s, %s) "
                "ON CONFLICT (graph_id) DO UPDATE SET token = EXCLUDED.token",
                (graph_id, deletion_token),
            )
        objects = 0
        if self.blob_store is not None:
            for run_id in run_ids:
                objects += self.blob_store.delete_prefix(run_id)
        with self._connection() as connection:
            self._lock_graph(connection, graph_id)
            reservation = connection.execute(
                "SELECT token FROM flakegraph_graph_deletion WHERE graph_id = %s FOR UPDATE",
                (graph_id,),
            ).fetchone()
            if reservation is None or reservation["token"] != deletion_token:
                raise ValueError(f"graph {graph_id} deletion reservation was superseded")
            # Heads and versions hold their artifacts with ON DELETE RESTRICT,
            # and publications theirs, so they go before the artifacts; tasks
            # take their dependencies and outputs with them.
            connection.execute("DELETE FROM flakegraph_graph_head WHERE graph_id = %s", (graph_id,))
            connection.execute(
                "DELETE FROM flakegraph_graph_version WHERE graph_id = %s OR run_id = ANY(%s)",
                (graph_id, run_ids),
            )
            connection.execute(
                "DELETE FROM flakegraph_publication WHERE run_id = ANY(%s)", (run_ids,)
            )
            tasks = connection.execute(
                "DELETE FROM flakegraph_task WHERE run_id = ANY(%s)", (run_ids,)
            ).rowcount
            artifacts = connection.execute(
                "DELETE FROM flakegraph_artifact WHERE run_id = ANY(%s)", (run_ids,)
            ).rowcount
            connection.execute("DELETE FROM flakegraph_run WHERE id = ANY(%s)", (run_ids,))
            connection.execute(
                "DELETE FROM flakegraph_graph_deletion WHERE graph_id = %s", (graph_id,)
            )
        with self._known_run_ids_lock:
            self._known_run_ids.difference_update(run_ids)
        return GraphDeletion(
            graph_id=graph_id,
            run_ids=run_ids,
            tasks=max(tasks, 0),
            artifacts=max(artifacts, 0),
            objects=objects,
        )

    def cancel_run(self, run_id: str) -> None:
        """Cancel a non-terminal run and all unfinished tasks in one transaction."""

        with self._connection() as connection:
            existing = connection.execute(
                """
                SELECT status FROM flakegraph_run WHERE id = %s
                """,
                (run_id,),
            ).fetchone()
            if existing is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            if str(existing["status"]) in {
                RunStatus.SUCCEEDED.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
            }:
                return
            # Workers lock task/publication rows before the parent run row.
            # Cancellation follows the same hierarchy to avoid a task<->run
            # ABBA deadlock with concurrent completion.
            connection.execute(
                """
                UPDATE flakegraph_task
                SET status = %s, lease_owner = NULL, lease_expires_at = NULL,
                    completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = %s AND status IN (%s, %s)
                """,
                (
                    TaskStatus.CANCELLED.value,
                    run_id,
                    TaskStatus.QUEUED.value,
                    TaskStatus.RUNNING.value,
                ),
            )
            connection.execute(
                """
                UPDATE flakegraph_publication
                SET status = %s, lease_owner = NULL, lease_expires_at = NULL,
                    completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE run_id = %s AND status IN (%s, %s)
                """,
                (
                    PublicationStatus.CANCELLED.value,
                    run_id,
                    PublicationStatus.QUEUED.value,
                    PublicationStatus.RUNNING.value,
                ),
            )
            connection.execute(
                """
                UPDATE flakegraph_run
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status NOT IN (%s, %s, %s)
                """,
                (
                    RunStatus.CANCELLED.value,
                    run_id,
                    RunStatus.SUCCEEDED.value,
                    RunStatus.FAILED.value,
                    RunStatus.CANCELLED.value,
                ),
            )

    def retry_run(self, run_id: str, skipped_file_ids: list[str] | None = None) -> None:
        """Requeue a failed run's failed work, or a cancelled run's cancelled work.

        Successful prerequisites and immutable outputs are preserved either way.
        Attempt counts reset only for the tasks being requeued, creating a fresh
        bounded retry budget after the underlying configuration or provider
        problem has been corrected, or after an operator changes their mind about
        a cancellation.
        """

        with self._connection() as connection:
            identity = connection.execute(
                "SELECT graph_id FROM flakegraph_run WHERE id = %s", (run_id,)
            ).fetchone()
            if identity is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            self._lock_graph(connection, str(identity["graph_id"]))
            self._assert_graph_available(connection, str(identity["graph_id"]))
            run = connection.execute(
                "SELECT status, config_json FROM flakegraph_run WHERE id = %s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            status = str(run["status"])
            if status == RunStatus.FAILED.value:
                stopped_task = TaskStatus.FAILED.value
                stopped_publication = PublicationStatus.FAILED.value
            elif status == RunStatus.CANCELLED.value:
                stopped_task = TaskStatus.CANCELLED.value
                stopped_publication = PublicationStatus.CANCELLED.value
            else:
                raise ValueError("only a failed or cancelled distributed run can be retried")
            # What failed, and why, is cleared from the rows being requeued;
            # the run keeps it as history, so a run that later succeeds still
            # shows what it recovered from. A cancellation has no error to keep.
            connection.execute(
                """
                INSERT INTO flakegraph_task_failure (
                    run_id, task_id, stage, scope_id, attempts, error_json, failed_at
                )
                SELECT run_id, id, stage, scope_id, attempts, last_error_json,
                       COALESCE(completed_at, updated_at)
                FROM flakegraph_task
                WHERE run_id = %s AND status = %s
                UNION ALL
                SELECT run_id, task_id, %s, id, attempts, last_error_json,
                       COALESCE(completed_at, updated_at)
                FROM flakegraph_publication
                WHERE run_id = %s AND status = %s
                """,
                (
                    run_id,
                    TaskStatus.FAILED.value,
                    _PUBLICATION_FAILURE_STAGE,
                    run_id,
                    PublicationStatus.FAILED.value,
                ),
            )
            if skipped_file_ids:
                self._quarantine_failed_files(
                    connection, run_id, run["config_json"], skipped_file_ids
                )
            requeued = connection.execute(
                """
                UPDATE flakegraph_task
                SET status = %s,
                    attempts = 0,
                    available_at = CURRENT_TIMESTAMP,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    last_error_json = NULL,
                    started_at = NULL,
                    completed_at = NULL,
                    progress_json = NULL,
                    progress_updated_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = %s AND status = %s
                """,
                (TaskStatus.QUEUED.value, run_id, stopped_task),
            )
            unfinished = connection.execute(
                "SELECT 1 FROM flakegraph_task WHERE run_id = %s AND status IN (%s, %s) LIMIT 1",
                (run_id, TaskStatus.QUEUED.value, TaskStatus.RUNNING.value),
            ).fetchone()
            publication_only_retry = requeued.rowcount == 0 and unfinished is None
            if publication_only_retry:
                publications = connection.execute(
                    """
                    UPDATE flakegraph_publication
                    SET status = %s, attempts = 0,
                        available_at = CURRENT_TIMESTAMP,
                        lease_owner = NULL, lease_expires_at = NULL,
                        last_error_json = NULL, completed_at = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE run_id = %s AND status = %s
                    """,
                    (
                        PublicationStatus.QUEUED.value,
                        run_id,
                        stopped_publication,
                    ),
                )
                if publications.rowcount == 0:
                    raise ValueError(
                        f"{status} run does not contain a {stopped_task} task or publication"
                    )
            connection.execute(
                """
                UPDATE flakegraph_run
                SET status = %s, error_json = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (
                    RunStatus.RUNNING.value if publication_only_retry else RunStatus.QUEUED.value,
                    run_id,
                ),
            )

    def _quarantine_failed_files(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        run_id: str,
        config: dict[str, Any],
        file_ids: list[str],
    ) -> None:
        """Retire only document trees containing a failed task and persist exclusions.

        A window's scope is its window ID, so matching scope_id alone would miss
        failed descendants. Traverse document dependencies, excluding the run-wide
        final barrier, and keep successful artifacts immutable for audit purposes.
        A requested document of this run with no failed task has nothing to
        retry; it is left in the graph rather than excluded.
        """

        requested = set(file_ids)
        if any(not identity.strip() for identity in requested):
            raise ValueError("quarantined file ids must not be blank")
        previous = set(config.get("quarantined_file_ids", []))
        roots = [
            stage.value
            for stage in (
                TaskStage.PREPARE_DOCUMENT,
                TaskStage.EXTRACT_DOCUMENT_CONTEXT,
                TaskStage.COMPACT_ENTITY_INVENTORY,
                TaskStage.COMPACT_DOCUMENT,
                TaskStage.REVALIDATE_DOCUMENT_CONTEXT,
                TaskStage.REVALIDATE_DOCUMENT,
            )
        ]
        tree = """
            WITH RECURSIVE file_tasks AS (
                SELECT id, scope_id AS file_id FROM flakegraph_task
                WHERE run_id = %s AND scope_id = ANY(%s) AND stage = ANY(%s)
                UNION
                SELECT child.id, parent.file_id FROM file_tasks parent
                JOIN flakegraph_task_dependency dependency
                  ON dependency.depends_on_task_id = parent.id
                JOIN flakegraph_task child ON child.id = dependency.task_id
                WHERE child.run_id = %s AND child.stage <> 'finalize_graph'
            )
        """
        candidates = connection.execute(
            tree
            + """
                SELECT tree.file_id, BOOL_OR(task.status = 'failed') AS has_failed_task
                FROM file_tasks tree JOIN flakegraph_task task ON task.id = tree.id
                GROUP BY tree.file_id
            """,
            (run_id, sorted(requested), roots, run_id),
        ).fetchall()
        if requested - previous - {str(row["file_id"]) for row in candidates}:
            raise ValueError("quarantined file ids must name documents of this run")
        eligible = {str(row["file_id"]) for row in candidates if row["has_failed_task"]}
        if not eligible:
            return
        connection.execute(
            tree
            + """
                UPDATE flakegraph_task SET status = 'succeeded', lease_owner = NULL,
                    lease_expires_at = NULL, completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id IN (SELECT id FROM file_tasks) AND status <> 'succeeded'
            """,
            (run_id, sorted(eligible), roots, run_id),
        )
        connection.execute(
            """
            UPDATE flakegraph_task dependent
            SET remaining_dependencies = (
                SELECT COUNT(*) FROM flakegraph_task_dependency dependency
                JOIN flakegraph_task prerequisite ON prerequisite.id = dependency.depends_on_task_id
                WHERE dependency.task_id = dependent.id AND prerequisite.status <> 'succeeded'
            )
            WHERE dependent.run_id = %s AND dependent.stage <> 'finalize_graph'
            """,
            (run_id,),
        )
        connection.execute(
            "UPDATE flakegraph_run SET config_json = %s WHERE id = %s",
            (Jsonb({**config, "quarantined_file_ids": sorted(previous | eligible)}), run_id),
        )

    def get_run(self, run_id: str) -> RunSnapshot:
        """Read one run and its ordered tasks from a repeatable transaction snapshot."""

        with self._connection() as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            run = connection.execute(
                "SELECT * FROM flakegraph_run WHERE id = %s",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            rows = connection.execute(
                f"{_TASK_SNAPSHOT_SELECT} WHERE task.run_id = %s ORDER BY task.created_at, task.id",
                (run_id,),
            ).fetchall()
            tasks = [_task_snapshot(row) for row in rows]
            effective_updated_at = max(
                [run["updated_at"], *(row["updated_at"] for row in rows)],
            )
            return RunSnapshot(
                run=_run_definition(run),
                tasks=tasks,
                created_at=run["created_at"],
                updated_at=effective_updated_at,
            )

    def get_stage_tasks(self, run_id: str, stage: TaskStage) -> list[TaskSnapshot]:
        """Read one stage's tasks without materializing the rest of the run."""

        with self._connection() as connection:
            run = connection.execute(
                "SELECT id FROM flakegraph_run WHERE id = %s",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            rows = connection.execute(
                f"{_TASK_SNAPSHOT_SELECT} WHERE task.run_id = %s AND task.stage = %s "
                "ORDER BY task.created_at, task.id",
                (run_id, stage.value),
            ).fetchall()
            return [_task_snapshot(row) for row in rows]

    def get_run_summary(self, run_id: str) -> RunSummary:
        """Read constant-size stage progress for operator-facing commands.

        The aggregate executes entirely in PostgreSQL and returns at most one row
        per stage/status pair, regardless of whether the run contains ten tasks or
        several million extraction windows.
        """

        with self._connection() as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            run = connection.execute(
                "SELECT * FROM flakegraph_run WHERE id = %s",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            rows = connection.execute(
                f"""
                SELECT {_TASK_COUNT_SELECT}
                FROM flakegraph_task
                WHERE run_id = %s
                GROUP BY stage, status
                ORDER BY stage, status
                """,
                (run_id,),
            ).fetchall()
            task_counts = [_task_count(row) for row in rows]
            # Heartbeats intentionally update task rows rather than one shared run
            # row, which avoids a write hotspot across large worker fleets. The
            # status aggregate already visits each stage/status bucket, so its
            # maximum supplies truthful run liveness without another task scan.
            latest_update = max(
                [
                    run["updated_at"],
                    *(row["updated_at"] for row in rows if row["updated_at"] is not None),
                ]
            )
            fleet = connection.execute(
                "SELECT stage, config_digest FROM flakegraph_worker_fleet"
            ).fetchall()
            # Only a succeeded task's outputs count: an attempt that wrote its
            # artifact and then lost its lease left an orphan, not a record.
            failed_documents = connection.execute(
                """
                SELECT COALESCE(
                    SUM(jsonb_array_length(artifact.metadata_json -> 'failed_documents')), 0
                ) AS documents
                FROM flakegraph_artifact AS artifact
                JOIN flakegraph_task_output AS output ON output.artifact_id = artifact.id
                JOIN flakegraph_task AS task
                  ON task.id = output.task_id AND task.status = %s
                WHERE artifact.run_id = %s
                  AND artifact.kind = %s
                  AND jsonb_typeof(artifact.metadata_json -> 'failed_documents') = 'array'
                """,
                (TaskStatus.SUCCEEDED.value, run_id, ArtifactKind.EXTRACTED_DOCUMENT.value),
            ).fetchone()
            resolved = connection.execute(
                """
                SELECT task_id, stage, scope_id, attempts, error_json, failed_at, retried_at,
                       COUNT(*) OVER () AS total
                FROM flakegraph_task_failure
                WHERE run_id = %s
                ORDER BY retried_at DESC, id DESC
                LIMIT %s
                """,
                (run_id, _RESOLVED_FAILURE_LIMIT),
            ).fetchall()
            return RunSummary(
                run=_run_definition(run),
                task_counts=task_counts,
                total_tasks=sum(item.count for item in task_counts),
                created_at=run["created_at"],
                updated_at=latest_update,
                error=cast(dict[str, Any] | None, run["error_json"]),
                documents_failed=int(failed_documents["documents"]) if failed_documents else 0,
                fleet_config_digests={
                    str(row["stage"]): str(row["config_digest"]) for row in fleet
                },
                resolved_failures=[
                    ResolvedFailure(
                        task_id=str(row["task_id"]),
                        stage=str(row["stage"]),
                        scope_id=str(row["scope_id"]),
                        attempts=int(row["attempts"]),
                        error=cast(dict[str, Any] | None, row["error_json"]),
                        failed_at=row["failed_at"],
                        retried_at=row["retried_at"],
                    )
                    for row in resolved
                ],
                resolved_failure_count=int(resolved[0]["total"]) if resolved else 0,
            )

    def list_runs(self, limit: int = 100) -> list[RunOverview]:
        """List recent runs with bounded grouped task counts in two SQL queries."""

        if limit <= 0 or limit > _MAX_RUN_LIST_LIMIT:
            raise ValueError("distributed run list limit must be between 1 and 500")
        with self._connection() as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            runs = connection.execute(
                """
                SELECT id, graph_id, status, priority_band, created_at, updated_at
                FROM flakegraph_run
                ORDER BY updated_at DESC, id
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
            if not runs:
                return []
            run_ids = [str(run["id"]) for run in runs]
            count_rows = connection.execute(
                f"""
                SELECT run_id, {_TASK_COUNT_SELECT}
                FROM flakegraph_task
                WHERE run_id = ANY(%s)
                GROUP BY run_id, stage, status
                ORDER BY run_id, stage, status
                """,
                (run_ids,),
            ).fetchall()
            counts_by_run: dict[str, list[TaskCount]] = {run_id: [] for run_id in run_ids}
            updated_by_run = {str(run["id"]): run["updated_at"] for run in runs}
            for row in count_rows:
                row_run_id = str(row["run_id"])
                counts_by_run[row_run_id].append(_task_count(row))
                if row["updated_at"] is not None:
                    updated_by_run[row_run_id] = max(
                        updated_by_run[row_run_id],
                        row["updated_at"],
                    )
            return [
                RunOverview(
                    id=str(run["id"]),
                    graph_id=str(run["graph_id"]),
                    status=RunStatus(str(run["status"])),
                    priority_band=PriorityBand(str(run["priority_band"])),
                    task_counts=counts_by_run[str(run["id"])],
                    total_tasks=sum(item.count for item in counts_by_run[str(run["id"])]),
                    created_at=run["created_at"],
                    updated_at=updated_by_run[str(run["id"])],
                )
                for run in runs
            ]

    def put(
        self,
        run_id: str,
        kind: ArtifactKind,
        payload: bytes,
        media_type: str,
        metadata: dict[str, Any] | None = None,
        identity_key: str | None = None,
    ) -> ArtifactRef:
        """Store immutable bytes inline or in the configured scalable blob plane."""

        if len(payload) > self.max_artifact_bytes:
            raise ValueError(
                f"artifact payload exceeds configured {self.max_artifact_bytes}-byte limit"
            )
        checksum = sha256_hex(payload)
        artifact_id = (
            stable_id("artifact", run_id, kind.value, checksum, identity_key)
            if identity_key is not None
            else stable_id("artifact", run_id, kind.value, checksum)
        )
        # Validate once per store process before transferring bytes. Planner and
        # worker paths normally populate this cache from their successful create
        # or claim transaction, while direct artifact callers still receive the
        # same unknown-run error before an external object can be orphaned.
        self._ensure_known_run_id(run_id)
        storage_uri: str | None = None
        database_payload: bytes | None
        compression: str
        if self.blob_store is None:
            database_payload = zlib.compress(payload, self.compression_level)
            compression = "zlib"
        else:
            suffix = _artifact_suffix(media_type)
            storage_uri = self.blob_store.put(
                f"{run_id}/{kind.value}/{artifact_id}{suffix}",
                payload,
                media_type,
            )
            database_payload = None
            compression = "none"
        artifact_metadata = metadata or {}
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO flakegraph_artifact (
                    id, run_id, kind, media_type, checksum, size_bytes,
                    compression, payload, storage_uri, metadata_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    artifact_id,
                    run_id,
                    kind.value,
                    media_type,
                    checksum,
                    len(payload),
                    compression,
                    database_payload,
                    storage_uri,
                    Jsonb(artifact_metadata),
                ),
            )
            row = connection.execute(
                "SELECT * FROM flakegraph_artifact WHERE id = %s",
                (artifact_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"failed to persist artifact {artifact_id}")
            return _artifact_ref(row)

    def _ensure_known_run_id(self, run_id: str) -> None:
        """Validate an uncached artifact owner once without racing staging threads."""

        with self._known_run_ids_lock:
            if run_id in self._known_run_ids:
                return
        with self._connection() as connection:
            run = connection.execute(
                "SELECT id FROM flakegraph_run WHERE id = %s",
                (run_id,),
            ).fetchone()
        if run is None:
            raise KeyError(f"unknown distributed run: {run_id}")
        with self._known_run_ids_lock:
            self._known_run_ids.add(run_id)

    def _remember_run_id(self, run_id: str) -> None:
        """Cache a run identity already proven by a successful transaction."""

        with self._known_run_ids_lock:
            self._known_run_ids.add(run_id)

    def get(self, artifact_id: str) -> StoredArtifact:
        """Load, decompress, size-check, and checksum-verify immutable artifact bytes."""

        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM flakegraph_artifact WHERE id = %s",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown distributed artifact: {artifact_id}")
        return self._load_artifact_row(row)

    def get_many(self, artifact_ids: list[str]) -> list[StoredArtifact]:
        """Resolve metadata once and overlap independent object-store reads.

        Caller order is preserved because window recombination is deterministic.
        A single ``ANY`` lookup replaces one PostgreSQL transaction per artifact;
        bounded parallel reads then hide object-store latency without allowing a
        large document to create an unbounded number of in-flight payloads.
        """

        if not artifact_ids:
            return []
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("batch artifact ids must be unique")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM flakegraph_artifact WHERE id = ANY(%s)",
                (artifact_ids,),
            ).fetchall()
        rows_by_id = {str(row["id"]): row for row in rows}
        missing = [artifact_id for artifact_id in artifact_ids if artifact_id not in rows_by_id]
        if missing:
            raise KeyError(f"unknown distributed artifact: {missing[0]}")
        ordered_rows = [rows_by_id[artifact_id] for artifact_id in artifact_ids]
        workers = min(_ARTIFACT_READ_PARALLELISM, len(ordered_rows))
        if workers == 1:
            return [self._load_artifact_row(ordered_rows[0])]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                bounded_map(
                    executor,
                    self._load_artifact_row,
                    ordered_rows,
                    buffersize=workers,
                )
            )

    def get_run_artifacts(
        self,
        run_id: str,
        kinds: set[ArtifactKind],
    ) -> list[StoredArtifact]:
        """Discover selected stage outputs without encoding them as scheduling edges."""

        return self.get_many(self.get_run_artifact_ids(run_id, kinds))

    def list_run_artifacts(
        self,
        run_id: str,
        kinds: set[ArtifactKind],
        *,
        linked: bool = True,
    ) -> list[ArtifactRef]:
        """Return the references of a run's artifacts, payloads left behind."""

        if not kinds:
            return []
        condition = f"AND {_LINKED_ARTIFACT_CONDITION}" if linked else ""
        parameters: tuple[object, ...] = (
            run_id,
            [kind.value for kind in sorted(kinds, key=str)],
            *((TaskStatus.SUCCEEDED.value,) if linked else ()),
        )
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT artifact.id, artifact.run_id, artifact.kind, artifact.media_type,
                       artifact.checksum, artifact.size_bytes, artifact.storage_uri,
                       artifact.metadata_json
                FROM flakegraph_artifact AS artifact
                WHERE artifact.run_id = %s AND artifact.kind = ANY(%s)
                  {condition}
                ORDER BY artifact.created_at, artifact.id
                """,
                parameters,
            ).fetchall()
        return [_artifact_ref(row) for row in rows]

    def get_run_artifact_ids(
        self,
        run_id: str,
        kinds: set[ArtifactKind],
    ) -> list[str]:
        """Return the ids of a run's stage outputs, linked from a task that succeeded.

        Artifacts are content-addressed and written before the task that made
        them completes, so an attempt that lost its lease after writing leaves
        an object behind, and the retry writes another. Only the one the
        succeeding attempt handed to ``complete_task`` is the stage's output;
        the rest are orphans that would otherwise be read alongside it.
        """

        if not kinds:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT artifact.id
                FROM flakegraph_artifact AS artifact
                WHERE artifact.run_id = %s AND artifact.kind = ANY(%s)
                  AND {_LINKED_ARTIFACT_CONDITION}
                ORDER BY artifact.created_at, artifact.id
                """,
                (
                    run_id,
                    [kind.value for kind in sorted(kinds, key=str)],
                    TaskStatus.SUCCEEDED.value,
                ),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def _load_artifact_row(self, row: dict[str, Any]) -> StoredArtifact:
        """Decode and verify one metadata row after its database lookup."""

        artifact_id = str(row["id"])
        compression = str(row["compression"])
        storage_uri = cast(str | None, row.get("storage_uri"))
        if compression == "zlib" and row["payload"] is not None:
            try:
                payload = zlib.decompress(bytes(row["payload"]))
            except zlib.error as exc:
                raise ValueError(f"artifact {artifact_id} has corrupt compressed data") from exc
        elif compression == "none" and storage_uri and self.blob_store is not None:
            payload = self.blob_store.get(storage_uri)
        else:
            raise ValueError(
                f"artifact {artifact_id} storage is unavailable or inconsistent "
                f"(compression={compression}, storage_uri={storage_uri!r})"
            )
        if len(payload) != int(row["size_bytes"]):
            raise ValueError(f"artifact {artifact_id} size does not match its metadata")
        if sha256_hex(payload) != str(row["checksum"]):
            raise ValueError(f"artifact {artifact_id} checksum verification failed")
        return StoredArtifact(ref=_artifact_ref(row), payload=payload)

    def _owned_task(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        task_id: str,
        worker_id: str,
        claim_token: str,
    ) -> dict[str, Any]:
        """Lock and return a running task only when the caller owns its lease."""

        row = connection.execute(
            """
            SELECT * FROM flakegraph_task
            WHERE id = %s AND status = %s AND lease_owner = %s AND claim_token = %s
            FOR UPDATE
            """,
            (task_id, TaskStatus.RUNNING.value, worker_id, claim_token),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"worker {worker_id} no longer owns task {task_id}")
        return row

    def fail_abandoned_tasks(self) -> None:
        """Fail exhausted task leases only after a heartbeat-aware recovery window.

        A task on its last attempt whose lease lapsed is failed together with
        its run, in one transaction, once the lease has been gone for longer
        than a heartbeat outage could explain.
        """

        with self._connection() as connection:
            failed_runs = connection.execute(
                """
                WITH exhausted AS (
                    UPDATE flakegraph_task
                    SET status = %s,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        last_error_json = %s,
                        completed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE status = %s
                      AND attempts >= max_attempts
                      AND GREATEST(lease_expires_at, updated_at) < (
                          CURRENT_TIMESTAMP - (%s * INTERVAL '1 second')
                      )
                    RETURNING run_id, id
                )
                SELECT run_id, MIN(id) AS representative_task_id
                FROM exhausted
                GROUP BY run_id
                """,
                (
                    TaskStatus.FAILED.value,
                    Jsonb(
                        {
                            "type": "LeaseExpired",
                            "message": "worker lease expired after retry budget was exhausted",
                        }
                    ),
                    TaskStatus.RUNNING.value,
                    _EXHAUSTED_TASK_RECOVERY_GRACE_SECONDS,
                ),
            ).fetchall()
            for failed in failed_runs:
                connection.execute(
                    """
                    UPDATE flakegraph_run
                    SET status = %s,
                        error_json = jsonb_build_object(
                            'type', 'LeaseExpired',
                            'task_id', %s::text,
                            'message', 'worker lease expired after retry budget was exhausted'
                        ),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND status IN (%s, %s)
                    """,
                    (
                        RunStatus.FAILED.value,
                        failed["representative_task_id"],
                        failed["run_id"],
                        RunStatus.QUEUED.value,
                        RunStatus.RUNNING.value,
                    ),
                )

    def _dependency_outputs(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        task_id: str,
    ) -> dict[str, list[str]]:
        """Return ordered artifact outputs for every direct prerequisite task."""

        rows = connection.execute(
            """
            SELECT dependency.depends_on_task_id, output.artifact_id
            FROM flakegraph_task_dependency AS dependency
            LEFT JOIN flakegraph_task_output AS output
              ON output.task_id = dependency.depends_on_task_id
            WHERE dependency.task_id = %s
            ORDER BY dependency.depends_on_task_id, output.position
            """,
            (task_id,),
        ).fetchall()
        outputs: dict[str, list[str]] = {}
        for row in rows:
            dependency_id = str(row["depends_on_task_id"])
            outputs.setdefault(dependency_id, [])
            if row["artifact_id"] is not None:
                outputs[dependency_id].append(str(row["artifact_id"]))
        return outputs


def _task_snapshot(row: dict[str, Any]) -> TaskSnapshot:
    """Convert one ``_TASK_SNAPSHOT_SELECT`` row into a task and its state."""

    return TaskSnapshot(
        task=_task_definition(row, row["dependency_ids"]),
        status=TaskStatus(str(row["status"])),
        attempts=int(row["attempts"]),
        lease_owner=cast(str | None, row["lease_owner"]),
        lease_expires_at=row["lease_expires_at"],
        output_artifact_ids=list(row["output_artifact_ids"]),
        last_error=cast(dict[str, Any] | None, row["last_error_json"]),
        progress=(
            TaskProgress.model_validate(row["progress_json"]) if row["progress_json"] else None
        ),
    )


def _task_definition(
    row: dict[str, Any],
    dependency_ids: Any,
) -> TaskDefinition:
    """Convert a database task row into its provider-neutral domain contract."""

    return TaskDefinition(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        stage=TaskStage(str(row["stage"])),
        scope_id=str(row["scope_id"]),
        payload=cast(dict[str, Any], row["payload_json"]),
        dependency_ids=sorted(str(value) for value in dependency_ids),
        max_attempts=int(row["max_attempts"]),
    )


def _run_definition(row: dict[str, Any]) -> RunDefinition:
    """Convert a database run row into its provider-neutral domain contract."""

    return RunDefinition(
        id=str(row["id"]),
        graph_id=str(row["graph_id"]),
        config=cast(dict[str, Any], row["config_json"]),
        config_digest=str(row["config_digest"]),
        status=RunStatus(str(row["status"])),
        priority_band=PriorityBand(str(row["priority_band"])),
    )


def _task_count(row: dict[str, Any]) -> TaskCount:
    """Convert one stage/status aggregate row into its bounded status bucket."""

    return TaskCount(
        stage=TaskStage(str(row["stage"])),
        status=TaskStatus(str(row["status"])),
        count=int(row["task_count"]),
        ready=int(row["ready_count"]),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        progress=(
            TaskProgress.model_validate(row["progress_json"]) if row["progress_json"] else None
        ),
    )


def _artifact_ref(row: dict[str, Any]) -> ArtifactRef:
    """Convert one artifact row into immutable public metadata."""

    return ArtifactRef(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        kind=ArtifactKind(str(row["kind"])),
        media_type=str(row["media_type"]),
        checksum=str(row["checksum"]),
        size_bytes=int(row["size_bytes"]),
        storage_uri=cast(str | None, row.get("storage_uri")),
        metadata=cast(dict[str, Any], row["metadata_json"]),
    )


def _artifact_suffix(media_type: str) -> str:
    """Choose an inspectable object suffix without making it part of identity."""

    if media_type == "application/json":
        return ".json"
    if media_type in {"application/vnd.apache.parquet", "application/x-parquet"}:
        return ".parquet"
    return ".bin"


def _validate_initial_task(
    task: TaskDefinition,
    run_id: str,
    final_seen: bool,
) -> tuple[int, int]:
    """Validate one streamed plan row and return preparation/final increments."""

    if task.run_id != run_id:
        raise ValueError("all tasks must belong to the selected run")
    if task.dependency_ids:
        raise ValueError("initial document tasks cannot have dependencies")
    if final_seen:
        raise ValueError("the final task must be the last initial-plan record")
    if task.stage in _INITIAL_DOCUMENT_STAGES:
        return 1, 0
    if task.stage is TaskStage.FINALIZE_GRAPH:
        return 0, 1
    raise ValueError("initial plan may contain only document seed and final tasks")


def _validate_dynamic_follow_ups(
    run_id: str,
    parent_task_id: str,
    parent_stage: TaskStage,
    tasks: list[TaskDefinition],
) -> None:
    """Validate the bounded runtime DAG a finished task may add.

    Preparation adds a context child. Context extraction adds entity windows plus
    their inventory barrier. Inventory compaction adds relation windows plus the
    final document barrier. A revalidation takes the same two waves with its own
    window stages and its own final document task. Restricting runtime fan-out
    to these explicit shapes keeps cycle safety obvious while allowing both
    provider-heavy phases to use the shared work-stealing queue.
    """

    task_ids = {task.id for task in tasks}
    if len(task_ids) != len(tasks):
        raise ValueError("dynamic follow-up task ids must be unique")
    if any(task.run_id != run_id for task in tasks):
        raise ValueError("dynamic follow-up tasks must belong to their parent run")
    allowed_children = {
        TaskStage.PREPARE_DOCUMENT: {TaskStage.EXTRACT_DOCUMENT_CONTEXT},
        TaskStage.EXTRACT_DOCUMENT_CONTEXT: {
            TaskStage.EXTRACT_ENTITY_WINDOW,
            TaskStage.COMPACT_ENTITY_INVENTORY,
        },
        TaskStage.REVALIDATE_DOCUMENT_CONTEXT: {
            TaskStage.REVALIDATE_ENTITY_WINDOW,
            TaskStage.COMPACT_ENTITY_INVENTORY,
        },
        TaskStage.COMPACT_ENTITY_INVENTORY: {
            TaskStage.EXTRACT_RELATION_WINDOW,
            TaskStage.COMPACT_DOCUMENT,
            TaskStage.REVALIDATE_RELATION_WINDOW,
            TaskStage.REVALIDATE_DOCUMENT,
        },
    }
    if any(task.stage not in allowed_children.get(parent_stage, set()) for task in tasks):
        raise ValueError(f"{parent_stage.value} cannot create the requested follow-up stage")
    allowed_dependency_ids = {parent_task_id, *task_ids}
    if any(not task.dependency_ids for task in tasks):
        raise ValueError("each dynamic follow-up requires at least one dependency")
    if any(set(task.dependency_ids) - allowed_dependency_ids for task in tasks):
        raise ValueError("dynamic follow-up dependencies must stay inside their subgraph")
    if any(task.id in task.dependency_ids for task in tasks):
        raise ValueError("dynamic follow-up tasks cannot depend on themselves")

    if parent_stage == TaskStage.EXTRACT_DOCUMENT_CONTEXT:
        _validate_window_fan_out(
            parent_task_id,
            tasks,
            TaskStage.EXTRACT_ENTITY_WINDOW,
            TaskStage.COMPACT_ENTITY_INVENTORY,
        )
    elif parent_stage == TaskStage.REVALIDATE_DOCUMENT_CONTEXT:
        _validate_window_fan_out(
            parent_task_id,
            tasks,
            TaskStage.REVALIDATE_ENTITY_WINDOW,
            TaskStage.COMPACT_ENTITY_INVENTORY,
        )
    elif parent_stage == TaskStage.COMPACT_ENTITY_INVENTORY:
        revalidating = any(task.stage == TaskStage.REVALIDATE_DOCUMENT for task in tasks)
        _validate_window_fan_out(
            parent_task_id,
            tasks,
            TaskStage.REVALIDATE_RELATION_WINDOW
            if revalidating
            else TaskStage.EXTRACT_RELATION_WINDOW,
            TaskStage.REVALIDATE_DOCUMENT if revalidating else TaskStage.COMPACT_DOCUMENT,
        )
    elif parent_stage == TaskStage.PREPARE_DOCUMENT and any(
        task.dependency_ids != [parent_task_id] for task in tasks
    ):
        raise ValueError("each preparation follow-up must depend directly on its parent")


def _validate_window_fan_out(
    parent_task_id: str,
    tasks: list[TaskDefinition],
    window_stage: TaskStage,
    barrier_stage: TaskStage,
) -> None:
    """Require windows of one stage followed by exactly one complete document barrier.

    Every window depends on the parent alone, and the barrier on the parent
    and every window, so the barrier cannot run before a window has finished.
    """

    if any(task.stage not in {window_stage, barrier_stage} for task in tasks):
        raise ValueError(
            f"a {barrier_stage.value} fan-out may create only {window_stage.value} tasks"
        )
    window_ids = {task.id for task in tasks if task.stage == window_stage}
    barriers = [task for task in tasks if task.stage == barrier_stage]
    if len(barriers) != 1:
        raise ValueError(f"a window fan-out must create exactly one {barrier_stage.value} task")
    if any(task.dependency_ids != [parent_task_id] for task in tasks if task.stage == window_stage):
        raise ValueError(f"each {window_stage.value} must depend directly on its parent")
    if set(barriers[0].dependency_ids) != {parent_task_id, *window_ids}:
        raise ValueError(
            f"{barrier_stage.value} must depend on its parent and every {window_stage.value}"
        )


def _validate_duration(value: timedelta, label: str) -> None:
    """Require positive finite durations before generating lease timestamps."""

    if value <= timedelta(0):
        raise ValueError(f"{label} must be positive")


def _validate_acyclic(tasks: list[TaskDefinition]) -> None:
    """Reject cyclic plans in linear time using Kahn's topological algorithm."""

    remaining_dependencies = {task.id: len(task.dependency_ids) for task in tasks}
    dependents: dict[str, list[str]] = {task.id: [] for task in tasks}
    for task in tasks:
        for dependency_id in task.dependency_ids:
            dependents[dependency_id].append(task.id)
    ready = deque(
        task_id
        for task_id, dependency_count in remaining_dependencies.items()
        if not dependency_count
    )
    visited = 0
    while ready:
        task_id = ready.popleft()
        visited += 1
        for dependent_id in dependents[task_id]:
            remaining_dependencies[dependent_id] -= 1
            if remaining_dependencies[dependent_id] == 0:
                ready.append(dependent_id)
    if visited != len(tasks):
        raise ValueError("distributed task graph must be acyclic")


def _installed_schema_version(connection: psycopg.Connection[dict[str, Any]]) -> int | None:
    """Return the recorded schema version, or None when nothing was ever installed.

    Tables without a version row count as version 0, which no build accepts.
    """

    version_table = connection.execute(
        "SELECT to_regclass('flakegraph_schema_version') AS relation"
    ).fetchone()
    if version_table is not None and version_table["relation"] is not None:
        version_row = connection.execute(
            "SELECT version FROM flakegraph_schema_version WHERE singleton = TRUE"
        ).fetchone()
        if version_row is not None:
            return int(version_row["version"])
    run_table = connection.execute("SELECT to_regclass('flakegraph_run') AS relation").fetchone()
    if run_table is not None and run_table["relation"] is not None:
        return 0
    return None


# The schema at ``_SCHEMA_VERSION``; initialize() refuses a database at any other.
_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS flakegraph_schema_version (
        singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
        version INTEGER NOT NULL CHECK (version > 0),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_run (
        id TEXT PRIMARY KEY,
        graph_id TEXT NOT NULL,
        config_json JSONB NOT NULL,
        config_digest TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('planning', 'queued', 'running', 'succeeded', 'failed', 'cancelled')
        ),
        -- How urgently the run's work is claimed. Every task of the run is
        -- stored at its stage's rank plus this band's offset.
        priority_band TEXT NOT NULL CHECK (priority_band IN ('interactive', 'bulk')),
        error_json JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_task (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES flakegraph_run(id) ON DELETE CASCADE,
        stage TEXT NOT NULL CHECK (
            stage IN (
                'prepare_document',
                'extract_document_context',
                'extract_entity_window',
                'compact_entity_inventory',
                'extract_relation_window',
                'compact_document',
                'revalidate_document_context',
                'revalidate_entity_window',
                'revalidate_relation_window',
                'revalidate_document',
                'finalize_graph'
            )
        ),
        scope_id TEXT NOT NULL,
        payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        status TEXT NOT NULL CHECK (
            status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')
        ),
        priority INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        remaining_dependencies INTEGER NOT NULL DEFAULT 0
            CHECK (remaining_dependencies >= 0),
        available_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        lease_owner TEXT,
        claim_token TEXT,
        lease_expires_at TIMESTAMPTZ,
        last_error_json JSONB,
        progress_json JSONB,
        progress_updated_at TIMESTAMPTZ,
        started_at TIMESTAMPTZ,
        completed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (run_id, stage, scope_id),
        CHECK (
            (status = 'running' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
            OR status <> 'running'
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_task_dependency (
        task_id TEXT NOT NULL REFERENCES flakegraph_task(id) ON DELETE CASCADE,
        depends_on_task_id TEXT NOT NULL REFERENCES flakegraph_task(id) ON DELETE CASCADE,
        PRIMARY KEY (task_id, depends_on_task_id),
        CHECK (task_id <> depends_on_task_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_artifact (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES flakegraph_run(id) ON DELETE CASCADE,
        kind TEXT NOT NULL CHECK (
            kind IN (
                'source_document',
                'prepared_document',
                'document_context',
                'extraction_window',
                'extracted_entity_window',
                'document_entity_inventory',
                'extracted_relation_window',
                'extracted_document',
                'graph_result'
            )
        ),
        media_type TEXT NOT NULL,
        checksum TEXT NOT NULL,
        size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
        compression TEXT NOT NULL,
        payload BYTEA,
        storage_uri TEXT,
        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT flakegraph_artifact_payload_location_check CHECK (
            (payload IS NOT NULL) <> (storage_uri IS NOT NULL)
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_task_output (
        task_id TEXT NOT NULL REFERENCES flakegraph_task(id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        artifact_id TEXT NOT NULL REFERENCES flakegraph_artifact(id) ON DELETE CASCADE,
        PRIMARY KEY (task_id, position),
        UNIQUE (task_id, artifact_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_graph_version (
        id TEXT PRIMARY KEY,
        graph_id TEXT NOT NULL,
        run_id TEXT NOT NULL UNIQUE REFERENCES flakegraph_run(id) ON DELETE RESTRICT,
        artifact_id TEXT NOT NULL UNIQUE REFERENCES flakegraph_artifact(id) ON DELETE RESTRICT,
        media_type TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_graph_head (
        graph_id TEXT PRIMARY KEY,
        version_id TEXT NOT NULL REFERENCES flakegraph_graph_version(id) ON DELETE RESTRICT,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_publication (
        generation BIGSERIAL PRIMARY KEY,
        id TEXT NOT NULL UNIQUE,
        run_id TEXT NOT NULL REFERENCES flakegraph_run(id) ON DELETE CASCADE,
        task_id TEXT NOT NULL UNIQUE REFERENCES flakegraph_task(id) ON DELETE CASCADE,
        artifact_id TEXT NOT NULL REFERENCES flakegraph_artifact(id) ON DELETE RESTRICT,
        task_payload_json JSONB NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')
        ),
        max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        available_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        lease_owner TEXT,
        lease_expires_at TIMESTAMPTZ,
        last_error_json JSONB,
        completed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CHECK (
            (status = 'running' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
            OR status <> 'running'
        )
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_publication_ready_idx
    ON flakegraph_publication (status, available_at, generation)
    """,
    """
    -- The OCR shim's admission queue. It lives here because the parsing plane
    -- must order waiting requests across every shim replica, which process
    -- memory cannot do, and because a restart should not drop work a client is
    -- still blocked on. Rows are short-lived: one exists only while a caller
    -- waits or is being parsed.
    --
    -- Priority follows the serving convention, where a LOWER value is served
    -- first. That is the opposite of flakegraph_task above, which orders by
    -- priority DESC. The two are different queues: this one shares its bands
    -- with the inference sidecar, not with the run planner.
    CREATE TABLE IF NOT EXISTS flakegraph_ocr_request (
        id TEXT PRIMARY KEY,
        priority INTEGER NOT NULL,
        consumer_class TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('waiting', 'dispatched')),
        shim_owner TEXT NOT NULL,
        -- The parsing replica a dispatched row was sent to. Admission counts
        -- the fleet's load per replica from this column and places the next
        -- request on the emptiest one, so every shim's view is the same.
        replica TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_ocr_request_waiting_idx
    ON flakegraph_ocr_request (priority, created_at, id)
    WHERE status = 'waiting'
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_ocr_request_stale_idx
    ON flakegraph_ocr_request (heartbeat_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_task_ready_claim_idx
    ON flakegraph_task (
        stage, status, remaining_dependencies, available_at,
        priority DESC, created_at
    )
    """,
    # A run's ready work, one run and stage at a time, in the order a claim takes
    # it: see ``_CLAIM_TASK``.
    """
    CREATE INDEX IF NOT EXISTS flakegraph_task_queued_claim_idx
    ON flakegraph_task (
        run_id, stage, remaining_dependencies, priority DESC, created_at, id
    )
    INCLUDE (available_at)
    WHERE status = 'queued'
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_task_run_summary_idx
    ON flakegraph_task (run_id, status, stage)
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_task_dependency_prerequisite_idx
    ON flakegraph_task_dependency (depends_on_task_id, task_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_task_unfinished_run_idx
    ON flakegraph_task (run_id)
    WHERE status <> 'succeeded'
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_task_expired_lease_idx
    ON flakegraph_task (lease_expires_at)
    WHERE status = 'running'
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_artifact_run_idx
    ON flakegraph_artifact (run_id, kind)
    """,
    """
    -- Failures a retry cleared. Retrying a failed run requeues its failed
    -- tasks with a fresh budget and no error, so the error lives on here: a
    -- run that later succeeds still shows what it recovered from.
    CREATE TABLE IF NOT EXISTS flakegraph_task_failure (
        id BIGSERIAL PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES flakegraph_run(id) ON DELETE CASCADE,
        task_id TEXT NOT NULL,
        stage TEXT NOT NULL,
        scope_id TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        error_json JSONB,
        failed_at TIMESTAMPTZ,
        retried_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_task_failure_run_idx
    ON flakegraph_task_failure (run_id, retried_at DESC)
    """,
    """
    -- What configuration the fleet actually serves, per stage.
    --
    -- A worker only claims tasks whose run carries its own configuration digest,
    -- so a run created under a configuration nobody serves any more can never be
    -- claimed. Without this table the demand view cannot tell the difference, and
    -- goes on asking for workers to do work none of them is able to take - which
    -- holds an autoscaled pool at its ceiling indefinitely.
    --
    -- Written by the deployment at every install and upgrade (``distributed init
    -- --serve-stage``) and by workers at startup, rather than on a heartbeat, and
    -- never expired, because the question is "what does this fleet serve", not
    -- "who is alive". Expiring it would erase the answer exactly when a pool has
    -- scaled to zero, which is when the demand signal has to work; and a pool at
    -- zero has no worker to declare a new digest, which is why the deployment
    -- declares it too.
    CREATE TABLE IF NOT EXISTS flakegraph_worker_fleet (
        stage TEXT PRIMARY KEY,
        config_digest TEXT NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE OR REPLACE VIEW flakegraph_worker_demand AS
    SELECT demand.stage,
           demand.priority_band,
           COUNT(*)::BIGINT AS desired_workers
    FROM (
        SELECT task.stage, run.priority_band
        FROM flakegraph_task AS task
        JOIN flakegraph_run AS run ON run.id = task.run_id
        WHERE run.status IN ('queued', 'running')
          -- Only work some worker could take. A stage with no recorded fleet is
          -- counted, so a first install can still scale up from zero before any
          -- worker has ever run.
          AND NOT EXISTS (
              SELECT 1
              FROM flakegraph_worker_fleet AS fleet
              WHERE fleet.stage = task.stage
                AND fleet.config_digest <> run.config_digest
          )
          AND (
              task.status = 'running'
              OR (
                  task.status = 'queued'
                  AND task.available_at <= CURRENT_TIMESTAMP
              )
          )
          AND (
              (
                  task.stage <> 'finalize_graph'
                  AND task.remaining_dependencies = 0
              )
              OR (
                  task.stage = 'finalize_graph'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM flakegraph_task AS unfinished
                      WHERE unfinished.run_id = task.run_id
                        AND unfinished.id <> task.id
                        AND unfinished.status <> 'succeeded'
                  )
              )
          )
        UNION ALL
        -- A publication is finalizer work in the band of the run it publishes.
        SELECT 'finalize_graph' AS stage, run.priority_band
        FROM flakegraph_publication AS publication
        JOIN flakegraph_run AS run ON run.id = publication.run_id
        WHERE run.status = 'running'
          AND (
              publication.status = 'running'
              OR (
                  publication.status = 'queued'
                  AND publication.available_at <= CURRENT_TIMESTAMP
              )
          )
    ) AS demand
    GROUP BY demand.stage, demand.priority_band
    """,
    """
    COMMENT ON VIEW flakegraph_worker_demand IS
    'Autoscaling demand: ready tasks, active leases, and pending graph publications.'
    """,
    # PostgreSQL's own monitoring role may read the queue's state - what is
    # queued, running, failed, and how long it has waited - so a metrics
    # exporter can turn it into series without holding the owner's credential.
    # The grants name columns rather than tables: a task row also carries its
    # payload and its last error, and a monitor needs neither.
    """
    GRANT SELECT (
        id, run_id, stage, status, priority, attempts, max_attempts,
        remaining_dependencies, available_at, lease_expires_at, lease_owner,
        started_at, completed_at, created_at, updated_at
    ) ON flakegraph_task TO pg_monitor
    """,
    """
    GRANT SELECT (id, graph_id, status, priority_band, created_at, updated_at)
    ON flakegraph_run TO pg_monitor
    """,
    """
    GRANT SELECT (
        generation, id, run_id, status, attempts, max_attempts, available_at,
        lease_owner, lease_expires_at, completed_at, created_at, updated_at
    ) ON flakegraph_publication TO pg_monitor
    """,
    """
    GRANT SELECT ON flakegraph_worker_demand TO pg_monitor
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_graph_deletion (
        graph_id TEXT PRIMARY KEY,
        token TEXT NOT NULL,
        reserved_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_consumption_event (
        run_id TEXT NOT NULL REFERENCES flakegraph_run(id) ON DELETE CASCADE,
        id TEXT NOT NULL,
        event_json JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (run_id, id)
    )
    """,
)
