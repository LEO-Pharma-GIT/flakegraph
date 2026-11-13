from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from snowflake_fakes import CONFIG, FakeConnection, FakeCursor

from kg_processor.adapters.jobs.snowflake import (
    SnowflakeJobFileProgressSink,
    SnowflakeJobManager,
    build_claim_job_files_statement,
    build_claim_job_statement,
    build_complete_file_queue_ownership_statement,
    build_complete_job_file_statement,
    build_complete_job_if_file_queue_drained_statement,
    build_complete_job_statement,
    build_create_job_file_submission_table_statement,
    build_create_pending_job_statement,
    build_fail_job_file_statement,
    build_fail_job_statement,
    build_heartbeat_job_files_statement,
    build_heartbeat_job_statement,
    build_insert_job_file_submission_statement,
    build_job_progress_statement,
    build_mark_job_running_statement,
    build_merge_job_file_submission_statement,
    build_reset_job_for_retry_statement,
    build_retry_failed_job_files_statement,
    build_select_claimed_job_files_statement,
    build_select_claimed_job_statement,
    build_update_job_file_progress_statement,
)
from kg_processor.application.progress import ProgressEvent
from kg_processor.domain.documents import InputFile
from kg_processor.domain.jobs import JobFileResult


def test_snowflake_job_manager_claims_available_job() -> None:
    connection = FakeConnection([("job_1", "RUNNING", "worker_1")])
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    result = manager.claim_job("job_1", "graph", "worker_1", 900, {"safe": True})

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert result.claimed
    assert result.status == "RUNNING"
    assert executed_sql == [
        build_create_pending_job_statement(),
        build_claim_job_statement(900),
        build_select_claimed_job_statement(),
    ]
    assert connection.committed
    assert not connection.closed
    manager.close()
    assert connection.closed


def test_snowflake_job_manager_returns_not_claimed_when_row_is_not_owned() -> None:
    connection = FakeConnection([None])
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    result = manager.claim_job("job_1", "graph", "worker_1", 900, {})

    assert not result.claimed
    assert result.job_id == "job_1"
    assert connection.committed


def test_snowflake_job_manager_submits_files_in_one_idempotent_merge() -> None:
    """Verify queue submission persists provenance without changing existing statuses."""

    connection = FakeConnection()
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)
    files = [
        InputFile(
            id="file_1",
            path=Path("incoming/a.pdf"),
            source_uri="@DB.SCHEMA.DOC_STAGE/incoming/a.pdf",
            checksum="checksum-a",
            mime_type="application/pdf",
            size_bytes=123,
        ),
        InputFile(
            id="file_2",
            path=Path("incoming/b.pdf"),
            source_uri="@DB.SCHEMA.DOC_STAGE/incoming/b.pdf",
            checksum="checksum-b",
            mime_type="application/pdf",
            size_bytes=456,
        ),
    ]

    submitted = manager.submit_job_files("job_1", "graph", files, {"safe": True})

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert submitted == 2
    assert executed_sql == [
        build_create_pending_job_statement(),
        build_create_job_file_submission_table_statement(),
        build_merge_job_file_submission_statement(),
    ]
    assert len(connection.cursor_instance.executed_many) == 1
    insert_sql, rows, kwargs = connection.cursor_instance.executed_many[0]
    assert insert_sql == build_insert_job_file_submission_statement()
    assert [row[3] for row in rows] == ["file_1", "file_2"]
    assert [row[5] for row in rows] == ["incoming/a.pdf", "incoming/b.pdf"]
    assert kwargs == {}
    assert "WHEN MATCHED THEN UPDATE" in build_merge_job_file_submission_statement()
    assert (
        "STATUS ="
        not in build_merge_job_file_submission_statement()
        .split("WHEN MATCHED THEN UPDATE", 1)[1]
        .split("WHEN NOT MATCHED", 1)[0]
    )
    assert connection.committed
    assert not connection.closed
    manager.close()
    assert connection.closed


def test_snowflake_job_manager_rejects_empty_submission_before_connecting() -> None:
    """Avoid creating a parent job that no worker could meaningfully process."""

    def factory(**_kwargs: object) -> FakeConnection:
        raise AssertionError("empty submission must not connect")

    manager = SnowflakeJobManager(CONFIG, connector_factory=factory)

    with pytest.raises(ValueError, match="no input files"):
        manager.submit_job_files("job_1", "graph", [], {})


def test_snowflake_job_manager_completes_and_fails_running_job() -> None:
    complete_connection = FakeConnection(rowcount=1)
    fail_connection = FakeConnection(rowcount=1)
    connections = [complete_connection, fail_connection]
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connections.pop(0))

    manager.complete_job("job_1", "worker_1", {"files_processed": 1})
    manager.close()
    manager.fail_job("job_2", "worker_1", {"message": "bad"})
    manager.close()

    assert complete_connection.cursor_instance.executed[0][0] == build_complete_job_statement()
    assert fail_connection.cursor_instance.executed[0][0] == build_fail_job_statement()
    assert complete_connection.committed
    assert fail_connection.committed


def test_snowflake_job_manager_rejects_completion_after_lease_loss() -> None:
    connection = FakeConnection(rowcount=0)
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    with pytest.raises(RuntimeError, match="no longer owned"):
        manager.complete_job("job_1", "stale_worker", {"files_processed": 1})

    assert connection.rolled_back
    assert not connection.committed


def test_snowflake_job_manager_heartbeats_running_job() -> None:
    connection = FakeConnection(rowcount=1)
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    manager.heartbeat_job("job_1", "worker_1", 900)

    assert connection.cursor_instance.executed[0] == (
        build_heartbeat_job_statement(900),
        ["job_1", "worker_1"],
    )
    assert connection.committed


def test_snowflake_job_manager_claims_file_batch() -> None:
    connection = FakeConnection(
        result_sets=[
            [
                (
                    "job_1",
                    "graph",
                    "file_1",
                    "@DB.SCHEMA.DOC_STAGE/input/a.pdf",
                    "checksum-a",
                    "CLAIMED",
                    "worker_1",
                    2,
                )
            ]
        ],
    )
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    claims = manager.claim_job_files("job_1", "graph", "worker_1", 900, 25)

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert [claim.file_id for claim in claims] == ["file_1"]
    assert claims[0].source_uri == "@DB.SCHEMA.DOC_STAGE/input/a.pdf"
    assert claims[0].attempts == 2
    assert executed_sql == [
        build_mark_job_running_statement(),
        build_claim_job_files_statement(900, 25),
        build_select_claimed_job_files_statement(25),
    ]
    claim_params = connection.cursor_instance.executed[1][1]
    select_params = connection.cursor_instance.executed[2][1]
    assert claim_params is not None
    assert select_params is not None
    assert claim_params[0] == "worker_1"
    assert str(claim_params[1]).startswith("claim:worker_1:")
    assert claim_params[2:] == ["job_1", "graph", "job_1", "graph", "job_1", "graph"]
    assert select_params == ["job_1", "graph", "worker_1", claim_params[1]]
    assert connection.committed


@pytest.mark.parametrize(
    ("counts", "expected"),
    [((2, 2), True), ((3, 2), False)],
)
def test_snowflake_job_manager_detects_complete_file_queue_ownership(
    counts: tuple[int, int],
    expected: bool,
) -> None:
    """Only a worker owning every queue row may publish graph-wide artifacts."""

    connection = FakeConnection([counts])
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    owns_queue = manager.worker_owns_entire_file_queue(
        "job_1",
        "graph",
        "worker_1",
        ["file_1", "file_2"],
    )

    assert owns_queue is expected
    assert connection.cursor_instance.executed == [
        (
            build_complete_file_queue_ownership_statement(2),
            # Bound to the graph, not the job: the permission a complete claim
            # grants replaces the whole graph, so the proof has to cover it.
            ["worker_1", "file_1", "file_2", "graph"],
        )
    ]
    assert not connection.closed
    manager.close()
    assert connection.closed


def test_snowflake_job_manager_completes_and_fails_claimed_files() -> None:
    complete_connection = FakeConnection()
    fail_connection = FakeConnection()
    drain_connection = FakeConnection(rowcount=1)
    connections = [complete_connection, fail_connection, drain_connection]
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connections.pop(0))

    manager.complete_job_files(
        "job_1",
        "graph",
        "worker_1",
        [
            JobFileResult(
                file_id="file_1",
                rows_written=9,
                row_counts={"chunks": 2, "nodes": 7},
                stage="written",
                ocr_provider="mineru_internal",
                llm_provider="vllm_local",
                embedding_model="Snowflake/snowflake-arctic-embed-l-v2.0",
                embedding_dimension=1024,
                audit={"quality_ok": True, "write_scope": "file_batch"},
            )
        ],
    )
    manager.close()
    manager.fail_job_files(
        "job_1",
        "graph",
        "worker_1",
        ["file_2"],
        {"message": "bad"},
    )
    manager.close()
    manager.complete_job_if_file_queue_drained("job_1", "graph", {"files_processed": 1})
    manager.close()

    assert complete_connection.cursor_instance.executed_many[0][0] == (
        build_complete_job_file_statement()
    )
    assert complete_connection.cursor_instance.executed_many[0][1][0] == [
        9,
        '{"chunks":2,"nodes":7}',
        "written",
        "mineru_internal",
        "vllm_local",
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        1024,
        (
            '{"file_id":"file_1","metadata":{"quality_ok":true,'
            '"write_scope":"file_batch"},"providers":{"embedding_dimension":1024,'
            '"embedding_model":"Snowflake/snowflake-arctic-embed-l-v2.0",'
            '"llm":"vllm_local","ocr":"mineru_internal"},"row_counts":{"chunks":2,'
            '"nodes":7},"rows_written":9,"stage":"written"}'
        ),
        "job_1",
        "graph",
        "file_1",
        "worker_1",
    ]
    assert complete_connection.cursor_instance.executed_many[0][2] == {"num_statements": 1}
    assert fail_connection.cursor_instance.executed_many[0][0] == build_fail_job_file_statement()
    assert fail_connection.cursor_instance.executed_many[0][2] == {"num_statements": 1}
    assert drain_connection.cursor_instance.executed[0][0] == (
        build_complete_job_if_file_queue_drained_statement()
    )


def test_snowflake_job_manager_heartbeats_claimed_files() -> None:
    connection = FakeConnection(rowcount=1)
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    manager.heartbeat_job_files("job_1", "graph", "worker_1", ["file_1", "file_2"], 900)

    assert connection.cursor_instance.executed[0] == (
        build_heartbeat_job_files_statement(900, 2),
        ["job_1", "graph", "worker_1", "file_1", "file_2"],
    )
    assert connection.committed


def test_snowflake_job_manager_updates_claimed_file_progress() -> None:
    connection = FakeConnection()
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    manager.update_job_file_progress(
        "job_1",
        "graph",
        "worker_1",
        ["file_1", "file_2"],
        "graph_extraction:started",
    )

    assert connection.cursor_instance.executed[0] == (
        build_update_job_file_progress_statement(2),
        ["graph_extraction:started", "job_1", "graph", "worker_1", "file_1", "file_2"],
    )
    assert connection.committed


def test_snowflake_job_file_progress_sink_targets_claimed_rows() -> None:
    manager = _RecordingProgressManager()
    sink = SnowflakeJobFileProgressSink(
        cast(Any, manager),
        "job_1",
        "graph",
        "worker_1",
        ["file_1", "file_2"],
    )

    sink.emit(
        ProgressEvent(
            job_id="job_1",
            graph_id="graph",
            stage="ocr",
            status="started",
            file_id="file_1",
        )
    )
    sink.emit(
        ProgressEvent(
            job_id="job_1",
            graph_id="graph",
            stage="merge",
            status="completed",
        )
    )
    sink.emit(
        ProgressEvent(
            job_id="job_1",
            graph_id="graph",
            stage="ocr",
            status="started",
            file_id="unclaimed",
        )
    )

    # STAGE names a stage, not a stage-and-status pair. The app groups file rows
    # by this column and merges the job row's live counts onto the stage of the
    # same name, so a composite value renders one stage as two rows.
    assert manager.updates == [
        ("job_1", "graph", "worker_1", ["file_1"], "ocr"),
        ("job_1", "graph", "worker_1", ["file_1", "file_2"], "merge"),
    ]
    # Every event also lands on the job row, the unclaimed file's included: the
    # job-level stage is graph-wide and does not depend on who owns the file.
    assert [(p["stage"], p["status"]) for p in manager.progress_payloads] == [
        ("ocr", "started"),
        ("merge", "completed"),
        ("ocr", "started"),
    ]


def test_claim_statement_rejects_non_positive_lease() -> None:
    with pytest.raises(ValueError, match="lease_seconds"):
        build_claim_job_statement(0)


def test_claim_job_files_statement_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="lease_seconds"):
        build_claim_job_files_statement(0, 100)
    with pytest.raises(ValueError, match="batch_size"):
        build_claim_job_files_statement(900, 0)


def test_job_file_claim_sql_uses_parent_job_state_and_claim_token() -> None:
    claim_sql = build_claim_job_files_statement(900, 25)
    select_sql = build_select_claimed_job_files_statement(25)
    drain_sql = build_complete_job_if_file_queue_drained_statement()

    assert "STAGE = ?" in claim_sql
    assert "AND EXISTS (SELECT 1 FROM KG_JOB" in claim_sql
    assert "STATUS = 'RUNNING'" in claim_sql
    # Picked in the subquery and re-checked by the UPDATE, so a row taken in
    # between is not claimed twice.
    assert claim_sql.count("STATUS = 'CLAIMED' AND LEASE_UNTIL < CURRENT_TIMESTAMP()") == 2
    assert "AND WORKER_ID = ? AND STAGE = ?" in select_sql
    assert "STATUS IN ('QUEUED', 'CLAIMED')" in drain_sql
    assert "'failed_files'" in drain_sql
    assert "'FAILED', 'SUCCEEDED'" in drain_sql
    assert "target.STATUS NOT IN ('SUCCEEDED', 'FAILED')" in build_mark_job_running_statement()
    assert "LEASE_EXPIRES_AT >= CURRENT_TIMESTAMP()" in build_complete_job_statement()
    assert "LEASE_UNTIL >= CURRENT_TIMESTAMP()" in build_complete_job_file_statement()


def test_heartbeat_statements_reject_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="lease_seconds"):
        build_heartbeat_job_statement(0)
    with pytest.raises(ValueError, match="lease_seconds"):
        build_heartbeat_job_files_statement(0, 1)
    with pytest.raises(ValueError, match="file_count"):
        build_heartbeat_job_files_statement(900, 0)
    with pytest.raises(ValueError, match="file_count"):
        build_update_job_file_progress_statement(0)


def test_retry_statement_respects_the_attempt_budget() -> None:
    """A poisoned document must not be retried forever by repeating the command."""

    statement = build_retry_failed_job_files_statement(3)

    assert "STATUS = 'QUEUED'" in statement
    assert "STATUS = 'FAILED'" in statement
    assert "COALESCE(ATTEMPTS, 0) < 3" in statement
    assert "ERROR = NULL" in statement
    assert "LEASE_UNTIL = NULL" in statement


def test_retry_statement_rejects_a_meaningless_budget() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        build_retry_failed_job_files_statement(0)


def test_job_reset_targets_settled_jobs_only() -> None:
    """Claiming accepts PENDING or FAILED, so a SUCCEEDED job must be reopened.

    A drained run marks the job SUCCEEDED. Without this reset a requeued file
    queue would never be claimed, because the file claim requires a RUNNING job.
    """

    statement = build_reset_job_for_retry_statement()

    assert "STATUS = 'PENDING'" in statement
    assert "STATUS IN ('SUCCEEDED', 'FAILED')" in statement
    assert "LEASE_OWNER = NULL" in statement


def test_retry_requeues_failed_files_and_reports_exhausted_rows() -> None:
    connection = FakeConnection(
        result_sets=[
            [("FAILED", 10)],
            [("QUEUED", 8), ("FAILED", 2)],
        ]
    )
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    result = manager.retry_failed_job_files("job_1", "graph_1", 3)

    assert result.files_requeued == 8
    assert result.files_exhausted == 2
    assert connection.committed
    statements = [sql for sql, _ in connection.cursor_instance.executed]
    assert any("STATUS = 'QUEUED'" in sql for sql in statements)
    assert any("STATUS = 'PENDING'" in sql for sql in statements)


def test_queue_status_counts_are_reported_per_status() -> None:
    connection = FakeConnection(result_sets=[[("DONE", 7), ("FAILED", 3)]])
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    assert manager.count_job_files_by_status("job_1", "graph_1") == {"DONE": 7, "FAILED": 3}


class _RecordingProgressManager:
    """Job manager stand-in that records graph-level progress writes."""

    def __init__(self, fail: bool = False) -> None:
        self.progress_payloads: list[dict[str, object]] = []
        self.updates: list[tuple[object, ...]] = []
        self.fail = fail

    def update_job_progress(
        self,
        job_id: str,
        graph_id: str,
        lease_owner: str,
        progress: dict[str, object],
    ) -> None:
        if self.fail:
            raise RuntimeError("warehouse unavailable")
        self.progress_payloads.append(progress)

    def update_job_file_progress(self, *args: object, **kwargs: object) -> None:
        self.updates.append(args)


def _extraction_event(status: str, completed: int) -> ProgressEvent:
    return ProgressEvent(
        job_id="job",
        graph_id="graph",
        stage="graph_extraction",
        status=status,
        counts={"batches_completed": completed, "batches_total": 21},
    )


def test_progress_writes_are_throttled_but_keep_every_transition() -> None:
    """Progress reporting must not become a real share of a run's query load.

    A stage can emit per batch or per candidate. Dropping a stage transition,
    though, would leave a finished stage rendered as still running.
    """

    manager = _RecordingProgressManager()
    sink = SnowflakeJobFileProgressSink(
        cast(Any, manager),
        job_id="job",
        graph_id="graph",
        worker_id="worker",
        claimed_file_ids=["f1"],
    )

    sink.emit(_extraction_event("progress", 1))
    sink.emit(_extraction_event("progress", 2))
    sink.emit(_extraction_event("completed", 21))

    recorded = [(p["stage"], p["status"]) for p in manager.progress_payloads]
    assert recorded == [
        ("graph_extraction", "progress"),
        ("graph_extraction", "completed"),
    ]
    assert manager.progress_payloads[-1]["counts"] == {
        "batches_completed": 21,
        "batches_total": 21,
    }


def test_a_failed_progress_write_never_ends_the_run() -> None:
    """The lease heartbeat is authoritative for liveness, not observability."""

    manager = _RecordingProgressManager(fail=True)
    sink = SnowflakeJobFileProgressSink(
        cast(Any, manager),
        job_id="job",
        graph_id="graph",
        worker_id="worker",
        claimed_file_ids=["f1"],
    )

    sink.emit(_extraction_event("progress", 1))

    assert manager.progress_payloads == []


def test_progress_is_recorded_for_a_run_that_takes_no_job_lease() -> None:
    """A file-queue run leaves KG_JOB.LEASE_OWNER null.

    Requiring the lease owner to match would make the write a no-op rather than
    an error, so nothing would surface and the app would keep reporting
    file-level zeros while the worker is plainly busy.
    """

    statement = build_job_progress_statement()

    assert "LEASE_OWNER IS NULL OR LEASE_OWNER = ?" in statement


def test_file_stage_and_job_progress_agree_on_one_stage_name() -> None:
    """The two halves the run page joins must name the same stage identically.

    The app groups KG_JOB_FILE rows by STAGE to derive each stage's state, then
    merges the live counts from KG_JOB.PROGRESS onto the stage of the same name.
    A file column storing ``graph_extraction:progress`` could never match, so a
    single running stage would be drawn twice: a stale file-derived row at 0/9
    titled "Graph Extraction:Progress", and a live duplicate.
    """

    manager = _RecordingProgressManager()
    sink = SnowflakeJobFileProgressSink(
        cast(Any, manager),
        job_id="job",
        graph_id="graph",
        worker_id="worker",
        claimed_file_ids=["f1"],
    )

    sink.emit(_extraction_event("progress", 3))

    file_stages = {update[-1] for update in manager.updates}
    payload_stages = {str(payload["stage"]) for payload in manager.progress_payloads}
    assert file_stages == payload_stages == {"graph_extraction"}


class _FailingCursor(FakeCursor):
    """Lose the session partway through a claim, after ``fail_after`` statements."""

    def __init__(self, fail_after: int) -> None:
        super().__init__()
        self.fail_after = fail_after

    def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
        *,
        timeout: int | None = None,
    ) -> object:
        super().execute(sql, params, timeout=timeout)
        if len(self.executed) > self.fail_after:
            raise RuntimeError("connector lost the session")
        return None


def test_job_claim_opens_a_transaction_before_its_first_statement() -> None:
    """The insert, the lease claim, and the confirming read must commit as one unit."""

    connection = FakeConnection([("job_1", "RUNNING", "worker_1")])
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    result = manager.claim_job("job_1", "graph", "worker_1", 900, {"safe": True})

    assert result.claimed is True
    assert connection.log[0] == ("autocommit", False)
    assert ("commit", None) in connection.log
    assert connection.log[-1] == ("autocommit", True)


def test_file_claim_opens_a_transaction_before_its_first_statement() -> None:
    """Claiming file rows and reading them back must commit as one unit."""

    connection = FakeConnection()
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    assert manager.claim_job_files("job_1", "graph", "worker_1", 900, 4) == []
    assert connection.log[0] == ("autocommit", False)
    assert ("commit", None) in connection.log
    assert connection.log[-1] == ("autocommit", True)


def test_a_failed_claim_confirmation_releases_the_rows_it_claimed() -> None:
    """A read that fails after the claim must not strand rows on an unaware worker."""

    connection = FakeConnection(cursor=_FailingCursor(fail_after=2))
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    with pytest.raises(RuntimeError):
        manager.claim_job_files("job_1", "graph", "worker_1", 900, 4)

    assert connection.log[0] == ("autocommit", False)
    assert ("rollback", None) in connection.log
    assert ("commit", None) not in connection.log


def test_snowflake_job_manager_redacts_the_run_report_before_persisting_it() -> None:
    """The completed-job report is readable by every role granted the schema."""

    connection = FakeConnection(rowcount=1)
    manager = SnowflakeJobManager(CONFIG, connector_factory=lambda **_: connection)

    manager.complete_job(
        "job_1",
        "worker_1",
        {"config": {"llm": {"api_key": "sk-secret", "endpoint": "https://llm.example/?api_key=x"}}},
    )
    manager.close()

    params = connection.cursor_instance.executed[0][1]
    assert params is not None
    persisted = str(params[0])
    assert "sk-secret" not in persisted
    assert "api_key=x" not in persisted
    assert "https://llm.example/?api_key=***" in persisted
