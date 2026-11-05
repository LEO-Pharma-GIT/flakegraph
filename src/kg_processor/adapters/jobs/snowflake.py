# SPDX-License-Identifier: Apache-2.0
"""Snowflake job and per-file lease coordination adapter."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any, cast

from pydantic import BaseModel

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    ReusableSnowflakeConnections,
    SnowflakeConnectionConfig,
    SnowflakeCursor,
)
from kg_processor.adapters.snowflake import (
    compact_json as _json,
)
from kg_processor.application.progress import ProgressEvent
from kg_processor.application.redaction import redact_sensitive_data
from kg_processor.domain.documents import InputFile
from kg_processor.domain.ids import stable_id
from kg_processor.domain.jobs import JobFileClaim, JobFileResult

_SUBMISSION_BATCH_SIZE = 10_000
_OWNERSHIP_COUNT_COLUMNS = 2


class JobRetryResult(BaseModel):
    """Outcome of returning a job's failed file rows to the queue."""

    job_id: str
    graph_id: str
    files_requeued: int
    files_exhausted: int


class JobClaimResult(BaseModel):
    """Outcome returned after attempting to lease a graph-level KG_JOB row."""

    claimed: bool
    job_id: str
    status: str | None = None
    lease_owner: str | None = None


class SnowflakeJobManager:
    """Coordinates graph-level and file-level leases in Snowflake tables."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        connector_factory: ConnectorFactory | None = None,
    ) -> None:
        """Store Snowflake connection details and an optional test connector."""

        self.config = config
        self.connector_factory = connector_factory
        self._connections = ReusableSnowflakeConnections(config, connector_factory)

    def close(self) -> None:
        """Release retained Snowflake sessions."""

        self._connections.close()

    def claim_job(
        self,
        job_id: str,
        graph_id: str,
        lease_owner: str,
        lease_seconds: int,
        config: dict[str, Any],
    ) -> JobClaimResult:
        """Create a pending job if needed and attempt to acquire its lease.

        Autocommit is disabled so the row insert, the lease claim, and the
        confirming read commit together. A failure part-way through therefore
        cannot leave a lease held by a worker that never learned it owns the job.
        """

        with self._connections.transaction() as cursor:
            cursor.execute(
                build_create_pending_job_statement(),
                [job_id, graph_id, _json(config)],
            )
            cursor.execute(
                build_claim_job_statement(lease_seconds),
                [lease_owner, job_id],
            )
            cursor.execute(build_select_claimed_job_statement(), [job_id, lease_owner])
            row = cursor.fetchone()
        if row is None:
            return JobClaimResult(claimed=False, job_id=job_id)
        return JobClaimResult(
            claimed=True,
            job_id=str(row[0]),
            status=str(row[1]),
            lease_owner=str(row[2]),
        )

    def submit_job_files(
        self,
        job_id: str,
        graph_id: str,
        files: Sequence[InputFile],
        config: dict[str, Any],
    ) -> int:
        """Create an immutable job and idempotently queue its discovered files.

        File metadata is loaded through a temporary table in bounded batches so
        submission remains practical for large corpora. Existing queue rows keep
        their processing status, which makes repeating a submission safe without
        causing completed documents to be processed again.
        """

        if not files:
            raise ValueError("Cannot submit a Snowflake job with no input files")
        with self._connections.cursor(commit=True) as cursor:
            cursor.execute(
                build_create_pending_job_statement(),
                [job_id, graph_id, _json(config)],
            )
            cursor.execute(build_create_job_file_submission_table_statement())
            insert_sql = build_insert_job_file_submission_statement()
            for start in range(0, len(files), _SUBMISSION_BATCH_SIZE):
                # `executemany` is part of the connector's DB-API cursor but is
                # outside the small cursor protocol used by read-oriented adapters.
                cast(Any, cursor).executemany(
                    insert_sql,
                    [
                        _job_file_submission_row(job_id, graph_id, file)
                        for file in files[start : start + _SUBMISSION_BATCH_SIZE]
                    ],
                )
            cursor.execute(build_merge_job_file_submission_statement())
        return len(files)

    def complete_job(self, job_id: str, lease_owner: str, report: dict[str, Any]) -> None:
        """Mark a leased graph-level job as succeeded and attach its run report."""

        # The report carries a configuration snapshot and provider metadata;
        # the job table is readable by every role granted the schema, so it
        # gets the same key-based redaction as a failure record.
        self._update_terminal_job(
            build_complete_job_statement(),
            [json.dumps(redact_sensitive_data(report), sort_keys=True), job_id, lease_owner],
        )

    def fail_job(self, job_id: str, lease_owner: str, error: dict[str, Any]) -> None:
        """Mark a leased graph-level job as failed with a structured error."""

        self._update_terminal_job(
            build_fail_job_statement(),
            [json.dumps(redact_sensitive_data(error), sort_keys=True), job_id, lease_owner],
        )

    def heartbeat_job(self, job_id: str, lease_owner: str, lease_seconds: int) -> None:
        """Extend a graph-level job lease while a worker is still alive."""

        self._update_terminal_job(
            build_heartbeat_job_statement(lease_seconds),
            [job_id, lease_owner],
        )

    def update_job_progress(
        self,
        job_id: str,
        graph_id: str,
        lease_owner: str,
        progress: dict[str, Any],
    ) -> None:
        """Record live stage progress so the app can distinguish work from a hang."""

        self._update_terminal_job(
            build_job_progress_statement(),
            [json.dumps(progress, sort_keys=True), job_id, graph_id, lease_owner],
        )

    def claim_job_files(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        lease_seconds: int,
        batch_size: int,
    ) -> list[JobFileClaim]:
        """Claim the next available KG_JOB_FILE rows for a worker.

        Autocommit is disabled so the claim and the confirming read commit
        together. Otherwise a failed read would leave rows CLAIMED against a
        spent attempt and an opaque claim token, leased to a worker that never
        received them and will never heartbeat.
        """

        claim_token = _claim_token(worker_id)
        with self._connections.transaction() as cursor:
            cursor.execute(build_mark_job_running_statement(), [job_id, graph_id])
            cursor.execute(
                build_claim_job_files_statement(lease_seconds, batch_size),
                [worker_id, claim_token, job_id, graph_id, job_id, graph_id, job_id, graph_id],
            )
            cursor.execute(
                build_select_claimed_job_files_statement(batch_size),
                [job_id, graph_id, worker_id, claim_token],
            )
            rows = cursor.fetchall()
        return [_job_file_claim_from_row(row) for row in rows]

    def complete_job_files(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        results: list[JobFileResult],
    ) -> None:
        """Mark claimed file rows done and persist per-file processing metadata."""

        if not results:
            return
        with self._connections.transaction() as cursor:
            statement = build_complete_job_file_statement()
            params = [
                [
                    result.rows_written,
                    _json(result.row_counts),
                    result.stage,
                    result.ocr_provider,
                    result.llm_provider,
                    result.embedding_model,
                    result.embedding_dimension,
                    _job_file_audit_json(result),
                    job_id,
                    graph_id,
                    result.file_id,
                    worker_id,
                ]
                for result in results
            ]
            affected_rows = _execute_many_updates(cursor, statement, params)
            if affected_rows != len(results):
                raise RuntimeError("one or more claimed files are no longer owned by this worker")

    def worker_owns_entire_file_queue(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        file_ids: list[str],
    ) -> bool:
        """Return whether one worker's claim covers every file in the job.

        A complete claim may safely publish graph-wide derived artifacts such as
        communities. Partial and concurrent claims must retain incremental write
        semantics because they do not contain the corpus-wide graph context.
        """

        if not file_ids:
            return False
        with self._connections.cursor() as cursor:
            cursor.execute(
                build_complete_file_queue_ownership_statement(len(file_ids)),
                [worker_id, *file_ids, graph_id],
            )
            row = cursor.fetchone()
        if row is None or len(row) < _OWNERSHIP_COUNT_COLUMNS:
            raise ValueError("Snowflake file-queue ownership query returned no counts")
        total_count = int(str(row[0] or 0))
        owned_count = int(str(row[1] or 0))
        return total_count == len(file_ids) and owned_count == len(file_ids)

    def fail_job_files(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        file_ids: list[str],
        error: dict[str, Any],
    ) -> None:
        """Mark claimed file rows failed with the same structured error payload."""

        if not file_ids:
            return
        with self._connections.transaction() as cursor:
            statement = build_fail_job_file_statement()
            error_json = json.dumps(redact_sensitive_data(error), sort_keys=True)
            affected_rows = _execute_many_updates(
                cursor,
                statement,
                [[error_json, job_id, graph_id, file_id, worker_id] for file_id in file_ids],
            )
            if affected_rows != len(file_ids):
                raise RuntimeError("one or more claimed files are no longer owned by this worker")

    def count_job_files_by_status(self, job_id: str, graph_id: str) -> dict[str, int]:
        """Return this job's queue counts keyed by status."""

        with self._connections.cursor() as cursor:
            return self._status_counts(cursor, job_id, graph_id)

    def retry_failed_job_files(
        self,
        job_id: str,
        graph_id: str,
        max_attempts: int,
    ) -> JobRetryResult:
        """Return failed file rows to the queue and make the job claimable again.

        A run that dies before completing leaves its rows FAILED, which no worker
        will claim, while the job row may already read SUCCEEDED. Both have to be
        reset together or a relaunched worker drains immediately without work.
        """

        with self._connections.transaction() as cursor:
            before = self._status_counts(cursor, job_id, graph_id)
            cursor.execute(
                build_retry_failed_job_files_statement(max_attempts),
                [job_id, graph_id],
            )
            cursor.execute(build_reset_job_for_retry_statement(), [job_id, graph_id])
        after = self.count_job_files_by_status(job_id, graph_id)
        failed_before = before.get("FAILED", 0)
        failed_after = after.get("FAILED", 0)
        return JobRetryResult(
            job_id=job_id,
            graph_id=graph_id,
            files_requeued=max(0, failed_before - failed_after),
            files_exhausted=failed_after,
        )

    def _status_counts(
        self,
        cursor: SnowflakeCursor,
        job_id: str,
        graph_id: str,
    ) -> dict[str, int]:
        cursor.execute(build_count_job_files_by_status_statement(), [job_id, graph_id])
        rows = cast(list[tuple[object, object]], cursor.fetchall())
        return {str(status): int(str(count or 0)) for status, count in rows}

    def heartbeat_job_files(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        file_ids: list[str],
        lease_seconds: int,
    ) -> None:
        """Extend leases for claimed file rows owned by the current worker."""

        if not file_ids:
            return
        with self._connections.cursor(commit=True) as cursor:
            cursor.execute(
                build_heartbeat_job_files_statement(lease_seconds, len(file_ids)),
                [job_id, graph_id, worker_id, *file_ids],
            )
            _require_owned_update(cursor, "job-file heartbeat")

    def update_job_file_progress(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        file_ids: list[str],
        stage: str,
    ) -> None:
        """Update the non-terminal stage text for claimed file rows."""

        if not file_ids:
            return
        with self._connections.cursor(commit=True) as cursor:
            cursor.execute(
                build_update_job_file_progress_statement(len(file_ids)),
                [stage, job_id, graph_id, worker_id, *file_ids],
            )

    def complete_job_if_file_queue_drained(
        self,
        job_id: str,
        graph_id: str,
        report: dict[str, Any],
    ) -> None:
        """Complete the graph-level job once no queued or claimed file rows remain."""

        self._update_terminal_job(
            build_complete_job_if_file_queue_drained_statement(),
            [
                job_id,
                graph_id,
                job_id,
                graph_id,
                job_id,
                graph_id,
                json.dumps(redact_sensitive_data(report), sort_keys=True),
                job_id,
                graph_id,
                job_id,
                graph_id,
            ],
            require_update=False,
        )

    def _update_terminal_job(
        self,
        sql: str,
        params: list[object],
        *,
        require_update: bool = True,
    ) -> None:
        with self._connections.cursor(commit=True) as cursor:
            cursor.execute(sql, params)
            if require_update:
                _require_owned_update(cursor, "job lease")


def _execute_many_updates(
    cursor: SnowflakeCursor,
    statement: str,
    params: Sequence[Sequence[object]],
) -> int:
    """Execute UPDATE bindings in one multi-statement request and total their row counts."""

    concrete = cast(Any, cursor)
    concrete.executemany(statement, params, num_statements=1)
    affected_rows = int(getattr(cursor, "rowcount", 0) or 0)
    nextset = getattr(cursor, "nextset", None)
    while callable(nextset) and nextset():
        affected_rows += int(getattr(cursor, "rowcount", 0) or 0)
    return affected_rows


def _require_owned_update(cursor: Any, operation: str) -> None:
    """Reject a no-op lease mutation instead of reporting stale work as successful."""

    if getattr(cursor, "rowcount", None) == 0:
        raise RuntimeError(f"Snowflake {operation} is no longer owned by this worker")


# Long enough that progress writes stay negligible against a run's real work,
# short enough that an operator watching a stage sees it advance.
_PROGRESS_WRITE_INTERVAL_SECONDS = 5.0
# Supplied by the composition root so the sink can report spend without knowing
# how consumption is collected or priced.
ConsumptionReader = Callable[[], dict[str, object]]


class SnowflakeJobFileProgressSink:
    """Persist live pipeline progress onto claimed KG_JOB_FILE rows.

    KG_JOB_FILE is the table the orchestration app naturally polls while a
    distributed file queue is running. The sink keeps STAGE current for claimed
    rows without changing terminal status; completion and failure still go
    through the explicit job manager methods.
    """

    def __init__(
        self,
        job_manager: SnowflakeJobManager,
        job_id: str,
        graph_id: str,
        worker_id: str,
        claimed_file_ids: list[str],
        consumption: ConsumptionReader | None = None,
    ) -> None:
        self.job_manager = job_manager
        self.job_id = job_id
        self.graph_id = graph_id
        self.worker_id = worker_id
        self.consumption = consumption
        self.claimed_file_ids = list(dict.fromkeys(claimed_file_ids))
        self.claimed_file_id_set = set(self.claimed_file_ids)
        self._last_stage_status: tuple[str, str] | None = None
        self._last_progress_write = 0.0

    def emit(self, event: ProgressEvent) -> None:
        """Persist the event as a file stage and as graph-level stage progress."""

        self._record_job_progress(event)
        file_ids = self._target_file_ids(event)
        if not file_ids:
            return
        self.job_manager.update_job_file_progress(
            self.job_id,
            self.graph_id,
            self.worker_id,
            file_ids,
            _progress_stage(event),
        )

    def _record_job_progress(self, event: ProgressEvent) -> None:
        """Write the current stage and its counts onto the graph-level job row.

        File rows only move between queued, claimed, and done, so a stage that
        runs for many minutes over an unchanged set of claimed files looks
        identical to a stalled worker. The counts the pipeline already emits are
        the only signal that separates the two.

        Writes are throttled because a stage can emit per batch or per candidate,
        and progress reporting must not become a meaningful share of a run's
        query load. Stage transitions are never dropped: they are what the app
        renders, and losing one would leave a finished stage shown as running.
        """

        transition = (event.stage, event.status) != self._last_stage_status
        now = time.monotonic()
        if not transition and (now - self._last_progress_write) < _PROGRESS_WRITE_INTERVAL_SECONDS:
            return
        payload: dict[str, Any] = {
            "stage": event.stage,
            "status": event.status,
            "timestamp": event.timestamp,
        }
        if event.counts:
            payload["counts"] = dict(event.counts)
        if event.message:
            payload["message"] = event.message
        if self.consumption is not None:
            # Tokens, pages and cost as they accrue. A run that takes hours is
            # exactly the one whose spend an operator wants to watch, and it is
            # the only one for which waiting until the end is a real cost.
            with suppress(Exception):
                payload["consumption"] = self.consumption()
        try:
            self.job_manager.update_job_progress(
                self.job_id,
                self.graph_id,
                self.worker_id,
                payload,
            )
        except Exception:
            # Observability must never end a run. The lease heartbeat remains
            # authoritative for liveness, and the next event retries the write.
            return
        self._last_stage_status = (event.stage, event.status)
        self._last_progress_write = now

    def _target_file_ids(self, event: ProgressEvent) -> list[str]:
        if event.file_id:
            if event.file_id not in self.claimed_file_id_set:
                return []
            return [event.file_id]
        return self.claimed_file_ids


def build_create_pending_job_statement() -> str:
    """Return SQL that idempotently creates a pending KG_JOB row."""

    return (
        "MERGE INTO KG_JOB target USING ("
        "SELECT ? AS ID, ? AS GRAPH_ID, PARSE_JSON(?) AS CONFIG"
        ") source ON target.ID = source.ID "
        "WHEN NOT MATCHED THEN INSERT (ID, GRAPH_ID, STATUS, CONFIG) "
        "VALUES (source.ID, source.GRAPH_ID, 'PENDING', source.CONFIG)"
    )


def build_create_job_file_submission_table_statement() -> str:
    """Return DDL for the session-local queue submission buffer."""

    return (
        "CREATE OR REPLACE TEMPORARY TABLE KG_JOB_FILE_SUBMISSION ("
        "ID STRING, JOB_ID STRING, GRAPH_ID STRING, FILE_ID STRING, "
        "SOURCE_URI STRING, BLOB_PATH STRING, CHECKSUM STRING)"
    )


def build_insert_job_file_submission_statement() -> str:
    """Return the batched insert statement for discovered file metadata."""

    return (
        "INSERT INTO KG_JOB_FILE_SUBMISSION "
        "(ID, JOB_ID, GRAPH_ID, FILE_ID, SOURCE_URI, BLOB_PATH, CHECKSUM) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)"
    )


def build_merge_job_file_submission_statement() -> str:
    """Return the idempotent merge from the submission buffer into the queue."""

    return (
        "MERGE INTO KG_JOB_FILE target USING KG_JOB_FILE_SUBMISSION source "
        "ON target.ID = source.ID "
        "WHEN MATCHED THEN UPDATE SET SOURCE_URI = source.SOURCE_URI, "
        "BLOB_PATH = source.BLOB_PATH, CHECKSUM = source.CHECKSUM, "
        "UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHEN NOT MATCHED THEN INSERT "
        "(ID, JOB_ID, GRAPH_ID, FILE_ID, SOURCE_URI, BLOB_PATH, CHECKSUM, STATUS) "
        "VALUES (source.ID, source.JOB_ID, source.GRAPH_ID, source.FILE_ID, "
        "source.SOURCE_URI, source.BLOB_PATH, source.CHECKSUM, 'QUEUED')"
    )


def build_mark_job_running_statement() -> str:
    """Return SQL that moves or creates a KG_JOB row in RUNNING state."""

    return (
        "MERGE INTO KG_JOB target USING ("
        "SELECT ? AS ID, ? AS GRAPH_ID"
        ") source ON target.ID = source.ID "
        "WHEN MATCHED AND target.STATUS NOT IN ('SUCCEEDED', 'FAILED') "
        "AND target.STATUS <> 'CANCELLED' "
        "THEN UPDATE SET STATUS = 'RUNNING', UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHEN NOT MATCHED THEN INSERT (ID, GRAPH_ID, STATUS) "
        "VALUES (source.ID, source.GRAPH_ID, 'RUNNING')"
    )


def build_claim_job_statement(lease_seconds: int) -> str:
    """Return SQL that atomically claims a graph-level job lease."""

    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    return (
        "UPDATE KG_JOB SET STATUS = 'RUNNING', LEASE_OWNER = ?, "
        f"LEASE_EXPIRES_AT = DATEADD(second, {lease_seconds}, CURRENT_TIMESTAMP()), "
        "ATTEMPT = COALESCE(ATTEMPT, 0) + 1, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE ID = ? AND (STATUS IN ('PENDING', 'FAILED') OR "
        "(STATUS = 'RUNNING' AND LEASE_EXPIRES_AT < CURRENT_TIMESTAMP()))"
    )


def build_select_claimed_job_statement() -> str:
    """Return SQL that confirms the current worker owns a claimed KG_JOB row."""

    return (
        "SELECT ID, STATUS, LEASE_OWNER FROM KG_JOB "
        "WHERE ID = ? AND STATUS = 'RUNNING' AND LEASE_OWNER = ?"
    )


def build_complete_job_statement() -> str:
    """Return SQL that completes a running KG_JOB row for its lease owner."""

    return (
        "UPDATE KG_JOB SET STATUS = 'SUCCEEDED', ERROR = NULL, CONFIG = "
        "OBJECT_INSERT(COALESCE(CONFIG, OBJECT_CONSTRUCT()), 'run_report', PARSE_JSON(?), TRUE), "
        "LEASE_EXPIRES_AT = NULL, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE ID = ? AND LEASE_OWNER = ? AND STATUS = 'RUNNING' "
        "AND LEASE_EXPIRES_AT >= CURRENT_TIMESTAMP()"
    )


def build_fail_job_statement() -> str:
    """Return SQL that records failure on a running KG_JOB row."""

    return (
        "UPDATE KG_JOB SET STATUS = 'FAILED', ERROR = PARSE_JSON(?), "
        "LEASE_EXPIRES_AT = NULL, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE ID = ? AND LEASE_OWNER = ? AND STATUS = 'RUNNING' "
        "AND LEASE_EXPIRES_AT >= CURRENT_TIMESTAMP()"
    )


def build_job_progress_statement() -> str:
    """Return SQL that records the worker's current stage progress.

    Bounded to the lease holder so a worker whose lease has expired cannot
    overwrite the progress of whichever worker replaced it.
    """

    return (
        "UPDATE KG_JOB SET "
        "PROGRESS = PARSE_JSON(?), "
        "UPDATED_AT = CURRENT_TIMESTAMP() "
        # A file-queue run takes no job-level lease, so LEASE_OWNER is null and
        # requiring a match would silently match nothing: the write would be a
        # no-op, not an error, and the app would keep showing file-level zeros.
        # The guard holds where a lease exists.
        "WHERE ID = ? AND GRAPH_ID = ? AND (LEASE_OWNER IS NULL OR LEASE_OWNER = ?)"
    )


def build_heartbeat_job_statement(lease_seconds: int) -> str:
    """Return SQL that refreshes a graph-level job lease."""

    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    return (
        "UPDATE KG_JOB SET "
        f"LEASE_EXPIRES_AT = DATEADD(second, {lease_seconds}, CURRENT_TIMESTAMP()), "
        "UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE ID = ? AND LEASE_OWNER = ? AND STATUS = 'RUNNING' "
        "AND LEASE_EXPIRES_AT >= CURRENT_TIMESTAMP()"
    )


# A file row a worker may take: never claimed, or claimed by one whose lease
# ran out.
_CLAIMABLE_FILE = (
    "(STATUS = 'QUEUED' OR (STATUS = 'CLAIMED' AND LEASE_UNTIL < CURRENT_TIMESTAMP()))"
)


def build_claim_job_files_statement(lease_seconds: int, batch_size: int) -> str:
    """Return SQL that claims the next queued or expired file rows.

    The rows are picked in a subquery and re-checked in the UPDATE itself, so a
    row another worker claimed between the two is left alone. The graph-level
    claim already keeps two workers from reaching this statement at once; the
    predicate is there so the statement is safe on its own.
    """

    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return (
        "UPDATE KG_JOB_FILE SET STATUS = 'CLAIMED', WORKER_ID = ?, STAGE = ?, "
        f"LEASE_UNTIL = DATEADD(second, {lease_seconds}, CURRENT_TIMESTAMP()), "
        "ATTEMPTS = COALESCE(ATTEMPTS, 0) + 1, ERROR = NULL, "
        "UPDATED_AT = CURRENT_TIMESTAMP() "
        f"WHERE JOB_ID = ? AND GRAPH_ID = ? AND {_CLAIMABLE_FILE} AND FILE_ID IN ("
        "SELECT FILE_ID FROM ("
        "SELECT FILE_ID FROM KG_JOB_FILE "
        f"WHERE JOB_ID = ? AND GRAPH_ID = ? AND {_CLAIMABLE_FILE} "
        "AND EXISTS (SELECT 1 FROM KG_JOB "
        "WHERE ID = ? AND GRAPH_ID = ? AND STATUS = 'RUNNING') "
        "ORDER BY UPDATED_AT, FILE_ID "
        f"LIMIT {batch_size}"
        ")"
        ")"
    )


def build_select_claimed_job_files_statement(batch_size: int) -> str:
    """Return SQL that fetches file rows just claimed by a worker."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return (
        "SELECT JOB_ID, GRAPH_ID, FILE_ID, COALESCE(SOURCE_URI, BLOB_PATH), CHECKSUM, "
        "STATUS, WORKER_ID, COALESCE(ATTEMPTS, 0) "
        "FROM KG_JOB_FILE "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND STATUS = 'CLAIMED' "
        "AND WORKER_ID = ? AND STAGE = ? "
        "ORDER BY UPDATED_AT, FILE_ID "
        f"LIMIT {batch_size}"
    )


def build_complete_file_queue_ownership_statement(file_count: int) -> str:
    """Return SQL that proves one claim owns every queued row of a graph.

    Scoped to the graph rather than to the job, because the permission it grants
    is graph-wide: a complete claim publishes graph-wide artifacts and replaces
    the graph's rows outright. Proving only that a claim covers its own job would
    let a second, one-document job for the same graph delete everything an
    earlier job wrote. A graph carrying rows from another job is therefore
    written incrementally, which is the additive reading of a second run.
    """

    if file_count <= 0:
        raise ValueError("file_count must be positive")
    placeholders = ", ".join("?" for _ in range(file_count))
    return (
        "SELECT COUNT(*) AS TOTAL_COUNT, COUNT_IF(STATUS = 'CLAIMED' "
        "AND WORKER_ID = ? "
        f"AND FILE_ID IN ({placeholders})) AS OWNED_COUNT "
        "FROM KG_JOB_FILE WHERE GRAPH_ID = ?"
    )


def build_complete_job_file_statement() -> str:
    """Return SQL that completes one claimed KG_JOB_FILE row."""

    return (
        "UPDATE KG_JOB_FILE SET STATUS = 'DONE', LEASE_UNTIL = NULL, "
        "ROWS_WRITTEN = ?, ROW_COUNTS = PARSE_JSON(?), STAGE = ?, "
        "OCR_PROVIDER = ?, LLM_PROVIDER = ?, EMBED_MODEL = ?, EMBED_DIM = ?, "
        "AUDIT = PARSE_JSON(?), ERROR = NULL, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND FILE_ID = ? AND WORKER_ID = ? "
        "AND STATUS = 'CLAIMED' AND LEASE_UNTIL >= CURRENT_TIMESTAMP()"
    )


def build_fail_job_file_statement() -> str:
    """Return SQL that fails one claimed KG_JOB_FILE row."""

    return (
        "UPDATE KG_JOB_FILE SET STATUS = 'FAILED', LEASE_UNTIL = NULL, "
        "ERROR = PARSE_JSON(?), UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND FILE_ID = ? AND WORKER_ID = ? "
        "AND STATUS = 'CLAIMED' AND LEASE_UNTIL >= CURRENT_TIMESTAMP()"
    )


def build_retry_failed_job_files_statement(max_attempts: int) -> str:
    """Return SQL that returns failed file rows to the claimable queue.

    Only rows still inside the attempt budget are reset, so a genuinely poisoned
    document cannot be retried forever by repeating the command.
    """

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    return (
        "UPDATE KG_JOB_FILE SET STATUS = 'QUEUED', WORKER_ID = NULL, "
        "LEASE_UNTIL = NULL, ERROR = NULL, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND STATUS = 'FAILED' "
        f"AND COALESCE(ATTEMPTS, 0) < {max_attempts}"
    )


def build_reset_job_for_retry_statement() -> str:
    """Return SQL that makes a settled KG_JOB row claimable again.

    A drained run marks the job SUCCEEDED. Claiming only accepts PENDING or
    FAILED, so without this reset a retried queue would never be picked up.
    """

    return (
        "UPDATE KG_JOB SET STATUS = 'PENDING', LEASE_OWNER = NULL, "
        "LEASE_EXPIRES_AT = NULL, ERROR = NULL, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE ID = ? AND GRAPH_ID = ? AND STATUS IN ('SUCCEEDED', 'FAILED')"
    )


def build_count_job_files_by_status_statement() -> str:
    """Return SQL that summarizes one job's queue by status."""

    return (
        "SELECT STATUS, COUNT(*) FROM KG_JOB_FILE WHERE JOB_ID = ? AND GRAPH_ID = ? GROUP BY STATUS"
    )


def build_heartbeat_job_files_statement(lease_seconds: int, file_count: int) -> str:
    """Return SQL that refreshes leases for a fixed set of file rows."""

    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    if file_count <= 0:
        raise ValueError("file_count must be positive")
    placeholders = ", ".join("?" for _ in range(file_count))
    return (
        "UPDATE KG_JOB_FILE SET "
        f"LEASE_UNTIL = DATEADD(second, {lease_seconds}, CURRENT_TIMESTAMP()), "
        "UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND WORKER_ID = ? "
        "AND STATUS = 'CLAIMED' AND LEASE_UNTIL >= CURRENT_TIMESTAMP() "
        f"AND FILE_ID IN ({placeholders})"
    )


def build_update_job_file_progress_statement(file_count: int) -> str:
    """Return SQL that writes progress stage text for claimed file rows."""

    if file_count <= 0:
        raise ValueError("file_count must be positive")
    placeholders = ", ".join("?" for _ in range(file_count))
    return (
        "UPDATE KG_JOB_FILE SET STAGE = ?, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND WORKER_ID = ? "
        "AND STATUS = 'CLAIMED' "
        f"AND FILE_ID IN ({placeholders})"
    )


def build_complete_job_if_file_queue_drained_statement() -> str:
    """Return SQL that terminally summarizes a drained file queue.

    A queue containing failed files must fail its parent job instead of leaving it
    permanently RUNNING. Only queued/claimed rows block finalization; terminal file
    failures are counted into a compact structured parent error.
    """

    return (
        "UPDATE KG_JOB SET STATUS = IFF(EXISTS ("
        "SELECT 1 FROM KG_JOB_FILE WHERE JOB_ID = ? AND GRAPH_ID = ? AND STATUS = 'FAILED'"
        "), 'FAILED', 'SUCCEEDED'), ERROR = IFF(EXISTS ("
        "SELECT 1 FROM KG_JOB_FILE WHERE JOB_ID = ? AND GRAPH_ID = ? AND STATUS = 'FAILED'"
        "), OBJECT_CONSTRUCT('failed_files', (SELECT COUNT(*) FROM KG_JOB_FILE "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND STATUS = 'FAILED')), NULL), CONFIG = "
        "OBJECT_INSERT(COALESCE(CONFIG, OBJECT_CONSTRUCT()), 'run_report', PARSE_JSON(?), TRUE), "
        "LEASE_EXPIRES_AT = NULL, UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHERE ID = ? AND GRAPH_ID = ? AND NOT EXISTS ("
        "SELECT 1 FROM KG_JOB_FILE "
        "WHERE JOB_ID = ? AND GRAPH_ID = ? AND STATUS IN ('QUEUED', 'CLAIMED')"
        ") AND STATUS = 'RUNNING'"
    )


def _job_file_claim_from_row(row: object) -> JobFileClaim:
    if not isinstance(row, tuple | list):
        raise ValueError("KG_JOB_FILE claim rows must be sequences")
    return JobFileClaim(
        job_id=str(row[0]),
        graph_id=str(row[1]),
        file_id=str(row[2]),
        source_uri=str(row[3]) if row[3] is not None else None,
        checksum=str(row[4]) if row[4] is not None else None,
        status=str(row[5]),
        worker_id=str(row[6]) if row[6] is not None else None,
        attempts=int(row[7] or 0),
    )


def _job_file_submission_row(
    job_id: str,
    graph_id: str,
    file: InputFile,
) -> tuple[str, str, str, str, str, str, str]:
    """Normalize one discovered file into the durable queue-row contract."""

    return (
        stable_id("snowflake_job_file", job_id, file.id),
        job_id,
        graph_id,
        file.id,
        file.source_uri,
        file.path.as_posix(),
        file.checksum,
    )


def _claim_token(worker_id: str) -> str:
    """Return a per-claim marker used to select only rows claimed in this call."""

    return f"claim:{worker_id}:{uuid.uuid4().hex}"


def _job_file_audit_json(result: JobFileResult) -> str:
    # Keep KG_JOB_FILE.AUDIT self-contained so the orchestration app can display
    # per-file outcomes without joining back to the graph report tables.
    return _json(
        {
            "file_id": result.file_id,
            "rows_written": result.rows_written,
            "row_counts": result.row_counts,
            "stage": result.stage,
            "providers": {
                "ocr": result.ocr_provider,
                "llm": result.llm_provider,
                "embedding_model": result.embedding_model,
                "embedding_dimension": result.embedding_dimension,
            },
            "metadata": result.audit,
        }
    )


def _progress_stage(event: ProgressEvent) -> str:
    """Return the stage name alone, which is what KG_JOB_FILE.STAGE means.

    The app groups file rows by STAGE to derive each stage's state, and separately
    merges the live counts from KG_JOB.PROGRESS onto the stage of the same name;
    a stored ``graph_extraction:progress`` would match neither, and one stage
    would render as two rows — a stale file-derived one and a duplicate live one
    — under the heading "Graph Extraction:Progress". The status is carried by the
    row's own STATUS and by the job row's progress payload.
    """

    return event.stage
