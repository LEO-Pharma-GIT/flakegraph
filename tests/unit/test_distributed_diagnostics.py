"""Verify bounded operator diagnostics without invoking provider libraries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kg_processor.application.distributed_diagnostics import (
    run_status_payload,
    worker_ready_payload,
)
from kg_processor.application.distributed_planner import distributed_processing_config_digest
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import (
    RunDefinition,
    RunStatus,
    RunSummary,
    TaskCount,
    TaskStage,
    TaskStatus,
)


def test_worker_ready_event_exposes_compatibility_without_connection_details() -> None:
    """Keep routine pod logs actionable while excluding secrets and endpoints."""

    settings = _settings()
    payload = worker_ready_payload(
        settings,
        "extract-1",
        {TaskStage.EXTRACT_ENTITY_WINDOW, TaskStage.EXTRACT_RELATION_WINDOW},
    )

    assert payload["event"] == "worker_ready"
    assert payload["config_digest"] == distributed_processing_config_digest(settings)
    assert payload["stages"] == ["extract_entity_window", "extract_relation_window"]
    assert payload["providers"]["llm"] == {"provider": "fake", "model": "test-model"}
    serialized = str(payload)
    assert "secret-value" not in serialized
    assert "https://private.example" not in serialized


def test_status_identifies_incompatible_configuration_and_stalled_queue() -> None:
    """Explain why queued work cannot advance without scanning individual tasks."""

    settings = _settings()
    updated_at = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
    summary = _summary(
        config_digest="another-digest",
        status=RunStatus.QUEUED,
        updated_at=updated_at,
        counts=[TaskCount(stage=TaskStage.PREPARE_DOCUMENT, status=TaskStatus.QUEUED, count=7)],
    )

    payload = run_status_payload(
        summary,
        settings,
        now=updated_at + timedelta(minutes=6),
    )

    diagnostics = payload["diagnostics"]
    assert diagnostics["state"] == "attention_required"
    assert diagnostics["config_digest"]["matches"] is False
    assert diagnostics["queued_tasks"] == 7
    assert [warning["code"] for warning in diagnostics["warnings"]] == [
        "CONFIG_DIGEST_MISMATCH",
        "QUEUED_WORK_NOT_ADVANCING",
    ]


def test_stalled_queued_work_names_the_digest_a_worker_has_to_report() -> None:
    """Point at eligibility, not only capacity, when nothing claims the work.

    Queued work that never starts looks the same whether the fleet is busy or
    whether no worker's configuration matches the run. Told only to check worker
    health, an operator inspects workers that are running perfectly and are
    simply ineligible, and the run's digest can then only be read out of the
    coordination database by hand.
    """

    settings = _settings()
    updated_at = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
    summary = _summary(
        # The run's own configuration, so the only warning left is the stall.
        config_digest=distributed_processing_config_digest(settings),
        status=RunStatus.QUEUED,
        updated_at=updated_at,
        counts=[TaskCount(stage=TaskStage.PREPARE_DOCUMENT, status=TaskStatus.QUEUED, count=4)],
    )

    payload = run_status_payload(summary, settings, now=updated_at + timedelta(minutes=6))

    warnings = {warning["code"]: warning for warning in payload["diagnostics"]["warnings"]}
    assert list(warnings) == ["QUEUED_WORK_NOT_ADVANCING"]
    remediation = warnings["QUEUED_WORK_NOT_ADVANCING"]["remediation"]
    assert distributed_processing_config_digest(settings) in remediation
    assert "worker_ready" in remediation


def test_status_identifies_stalled_later_stage_while_run_remains_running() -> None:
    """Detect a queued finalizer after earlier tasks put the run in running state."""

    settings = _settings()
    updated_at = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
    summary = _summary(
        config_digest=distributed_processing_config_digest(settings),
        status=RunStatus.RUNNING,
        updated_at=updated_at,
        counts=[
            TaskCount(
                stage=TaskStage.FINALIZE_GRAPH,
                status=TaskStatus.QUEUED,
                count=1,
            )
        ],
    )

    payload = run_status_payload(
        summary,
        settings,
        now=updated_at + timedelta(minutes=6),
    )

    assert payload["diagnostics"]["state"] == "attention_required"
    assert payload["diagnostics"]["warnings"][0]["code"] == "QUEUED_WORK_NOT_ADVANCING"


def test_status_reports_active_matching_run_without_warning() -> None:
    """Avoid alarming operators while a compatible worker is making progress."""

    settings = _settings()
    now = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
    summary = _summary(
        config_digest=distributed_processing_config_digest(settings),
        status=RunStatus.RUNNING,
        updated_at=now,
        counts=[
            TaskCount(
                stage=TaskStage.EXTRACT_ENTITY_WINDOW,
                status=TaskStatus.RUNNING,
                count=2,
            )
        ],
    )

    diagnostics = run_status_payload(summary, settings, now=now)["diagnostics"]

    assert diagnostics["state"] == "processing"
    assert diagnostics["running_tasks"] == 2
    assert diagnostics["warnings"] == []


def test_status_names_the_stages_a_fleet_upgrade_left_this_run_without() -> None:
    """A fleet at another digest claims nothing from the run and asks for no workers.

    The run does not fail; it waits for a worker that will never come. Only a
    stage with work still open matters: one the run has finished with can be
    served at any digest.
    """

    settings = _settings()
    now = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
    digest = distributed_processing_config_digest(settings)
    summary = _summary(
        config_digest=digest,
        status=RunStatus.RUNNING,
        updated_at=now,
        counts=[
            TaskCount(stage=TaskStage.PREPARE_DOCUMENT, status=TaskStatus.SUCCEEDED, count=4),
            TaskCount(stage=TaskStage.EXTRACT_ENTITY_WINDOW, status=TaskStatus.QUEUED, count=2),
        ],
        fleet={
            TaskStage.PREPARE_DOCUMENT.value: "digest-after-upgrade",
            TaskStage.EXTRACT_ENTITY_WINDOW.value: "digest-after-upgrade",
            TaskStage.FINALIZE_GRAPH.value: digest,
        },
    )

    diagnostics = run_status_payload(summary, settings, now=now)["diagnostics"]

    assert diagnostics["state"] == "attention_required"
    assert [item["code"] for item in diagnostics["warnings"]] == ["FLEET_DIGEST_MISMATCH"]
    assert "extract_entity_window" in diagnostics["warnings"][0]["message"]
    assert "prepare_document" not in diagnostics["warnings"][0]["message"]
    # Both configurations are named: either side can be the stale one.
    assert "digest-after" in diagnostics["warnings"][0]["message"]
    assert digest[:12] in diagnostics["warnings"][0]["message"]
    assert "cancel it and resubmit" in diagnostics["warnings"][0]["remediation"]


def test_a_fleet_at_the_runs_digest_raises_no_upgrade_warning() -> None:
    settings = _settings()
    now = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
    digest = distributed_processing_config_digest(settings)
    summary = _summary(
        config_digest=digest,
        status=RunStatus.RUNNING,
        updated_at=now,
        counts=[
            TaskCount(stage=TaskStage.EXTRACT_ENTITY_WINDOW, status=TaskStatus.RUNNING, count=2)
        ],
        fleet={TaskStage.EXTRACT_ENTITY_WINDOW.value: digest},
    )

    assert run_status_payload(summary, settings, now=now)["diagnostics"]["warnings"] == []


def test_status_identifies_running_work_that_stopped_heartbeating() -> None:
    """Surface a vanished worker while preserving automatic lease recovery."""

    settings = _settings()
    updated_at = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
    summary = _summary(
        config_digest=distributed_processing_config_digest(settings),
        status=RunStatus.RUNNING,
        updated_at=updated_at,
        counts=[
            TaskCount(
                stage=TaskStage.FINALIZE_GRAPH,
                status=TaskStatus.RUNNING,
                count=1,
            )
        ],
    )

    payload = run_status_payload(
        summary,
        settings,
        now=updated_at + timedelta(minutes=6),
    )

    assert payload["diagnostics"]["state"] == "attention_required"
    assert payload["diagnostics"]["warnings"] == [
        {
            "code": "ACTIVE_WORK_NOT_ADVANCING",
            "message": (
                "1 running tasks have not reported progress or a heartbeat for 360 seconds."
            ),
            "remediation": (
                "Check worker and node health. The durable lease is reclaimed automatically "
                "after it expires; completed prerequisites are not repeated."
            ),
        }
    ]


def test_failed_status_preserves_one_redacted_root_error() -> None:
    """Keep terminal failures actionable without expanding every task payload."""

    settings = _settings()
    now = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
    error = {
        "task_id": "run-1-task-7",
        "error_type": "RuntimeError",
        "error_message": "provider unavailable",
    }
    summary = _summary(
        config_digest=distributed_processing_config_digest(settings),
        status=RunStatus.FAILED,
        updated_at=now,
        counts=[
            TaskCount(
                stage=TaskStage.EXTRACT_RELATION_WINDOW,
                status=TaskStatus.FAILED,
                count=1,
            )
        ],
        error=error,
    )

    payload = run_status_payload(summary, settings, now=now)

    assert payload["error"] == error
    assert payload["diagnostics"]["state"] == "terminal"


def test_terminal_status_carries_no_claim_warning_under_another_config() -> None:
    """Keep completed status useful even when inspected with another config."""

    settings = _settings()
    now = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
    summary = _summary(
        config_digest="historical-digest",
        status=RunStatus.SUCCEEDED,
        updated_at=now,
        counts=[TaskCount(stage=TaskStage.FINALIZE_GRAPH, status=TaskStatus.SUCCEEDED, count=1)],
    )

    payload = run_status_payload(summary, settings, now=now + timedelta(minutes=5))

    assert payload["diagnostics"]["state"] == "terminal"
    assert payload["diagnostics"]["config_digest"]["matches"] is False
    assert payload["diagnostics"]["warnings"] == []


def _settings() -> Settings:
    """Return a deterministic provider profile containing private connection data."""

    return Settings.load(
        overrides={
            "ocr": {"provider": "builtin_text"},
            "llm": {
                "provider": "fake",
                "model": "test-model",
                "endpoint": "https://private.example/v1",
                "api_key": "secret-value",
            },
            "embedding": {"provider": "hash", "model": "test-embedding", "dimension": 8},
        }
    )


def _summary(
    *,
    config_digest: str,
    status: RunStatus,
    updated_at: datetime,
    counts: list[TaskCount],
    error: dict[str, str] | None = None,
    fleet: dict[str, str] | None = None,
) -> RunSummary:
    """Construct a bounded run response for one diagnostics scenario."""

    return RunSummary(
        run=RunDefinition(
            id="run-1",
            graph_id="graph-1",
            config={},
            config_digest=config_digest,
            status=status,
        ),
        task_counts=counts,
        total_tasks=sum(item.count for item in counts),
        created_at=updated_at - timedelta(minutes=1),
        updated_at=updated_at,
        error=error,
        fleet_config_digests=fleet or {},
    )
