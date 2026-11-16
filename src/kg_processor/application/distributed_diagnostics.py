# SPDX-License-Identifier: Apache-2.0
"""Build bounded, secret-free diagnostics for distributed FlakeGraph operators."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from kg_processor.application.distributed_planner import distributed_processing_config_digest
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import RunStatus, RunSummary, TaskStage, TaskStatus

_STALLED_AFTER_SECONDS = 300.0
_TERMINAL_RUN_STATUSES = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


def worker_ready_payload(
    settings: Settings,
    worker_id: str,
    stages: set[TaskStage],
) -> dict[str, Any]:
    """Describe one worker's compatibility contract without exposing credentials.

    The event is intended for ordinary container logs and automated health tooling.
    It includes semantic provider identities and the exact digest used for task
    claims, but deliberately omits endpoints, tokens, filesystem paths, and the
    redacted full configuration.
    """

    return {
        "event": "worker_ready",
        "worker_id": worker_id,
        "config_digest": distributed_processing_config_digest(settings),
        "stages": sorted(stage.value for stage in stages),
        "providers": {
            "ocr": {"provider": settings.ocr.provider},
            "llm": {"provider": settings.llm.provider, "model": settings.llm.model},
            "embedding": {
                "provider": settings.embedding.provider,
                "model": settings.embedding.model,
            },
        },
        "lease_seconds": settings.distributed.lease_seconds,
        "poll_interval_seconds": settings.distributed.poll_interval_seconds,
        "slots": settings.distributed.worker_slots,
    }


def run_status_payload(
    result: RunSummary,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Add actionable compatibility and liveness diagnostics to run status.

    The payload is constant-size: it uses only durable task aggregates and
    timestamps, so status never needs a worker registry or an expensive task
    scan to identify the common cases of incompatible workers and queued work
    that is no longer advancing.
    """

    observed_at = _as_utc(now or datetime.now(UTC))
    seconds_since_update = max(
        0.0,
        (observed_at - _as_utc(result.updated_at)).total_seconds(),
    )
    local_digest = distributed_processing_config_digest(settings)
    digest_matches = local_digest == result.run.config_digest
    queued_tasks, running_tasks = _active_task_counts(result)
    warnings: list[dict[str, str]] = []

    if not digest_matches and result.run.status not in _TERMINAL_RUN_STATUSES:
        warnings.append(
            {
                "code": "CONFIG_DIGEST_MISMATCH",
                "message": "This configuration cannot claim tasks from the run.",
                "remediation": (
                    "Use the run's provider, model, ontology, prompt, and graph settings, "
                    "then confirm worker_ready reports the run config_digest."
                ),
            }
        )
    # Workers at another digest do not claim this run's tasks, and do not ask
    # for workers to do so: the demand signal counts only work the fleet can
    # take, so the run does not fail, it waits for nothing. Either side can be
    # the stale one - a pool left on an older release, or a run submitted from
    # a client on one.
    left_behind = _stages_served_at_another_digest(result)
    if left_behind and result.run.status not in _TERMINAL_RUN_STATUSES:
        fleet = sorted({result.fleet_config_digests[stage][:12] for stage in left_behind})
        warnings.append(
            {
                "code": "FLEET_DIGEST_MISMATCH",
                "message": (
                    f"The workers serving {', '.join(left_behind)} run configuration "
                    f"{', '.join(fleet)}, and this run {result.run.config_digest[:12]}; "
                    "no worker will claim its remaining tasks."
                ),
                "remediation": (
                    "Deploy one release to every worker pool and to the client that submits "
                    "runs; then retry the run, or cancel it and resubmit it from that client."
                ),
            }
        )
    # Queued work that never starts has two very different causes that look
    # identical from here: no spare capacity, or no worker whose configuration
    # matches this run. Naming only the first sends an operator to inspect
    # workers that are perfectly healthy and simply ineligible.
    if (
        result.run.status not in _TERMINAL_RUN_STATUSES
        and queued_tasks > 0
        and running_tasks == 0
        and seconds_since_update >= _STALLED_AFTER_SECONDS
    ):
        warnings.append(
            {
                "code": "QUEUED_WORK_NOT_ADVANCING",
                "message": (
                    f"{queued_tasks} tasks have remained queued without run activity for "
                    f"{int(seconds_since_update)} seconds."
                ),
                "remediation": (
                    "A worker only claims tasks whose run config_digest equals its own, so "
                    f"first confirm one is running this run's {result.run.config_digest}: it is "
                    "reported in each worker's worker_ready log line at startup. Workers that "
                    "are healthy but report a different digest will never claim this run. "
                    "Otherwise eligible workers are busy or unhealthy; completed work is "
                    "durable and the run resumes when capacity returns."
                ),
            }
        )
    running_stalled_after = max(
        _STALLED_AFTER_SECONDS,
        settings.distributed.lease_seconds / 2,
    )
    if (
        result.run.status not in _TERMINAL_RUN_STATUSES
        and running_tasks > 0
        and seconds_since_update >= running_stalled_after
    ):
        warnings.append(
            {
                "code": "ACTIVE_WORK_NOT_ADVANCING",
                "message": (
                    f"{running_tasks} running tasks have not reported progress or a heartbeat "
                    f"for {int(seconds_since_update)} seconds."
                ),
                "remediation": (
                    "Check worker and node health. The durable lease is reclaimed automatically "
                    "after it expires; completed prerequisites are not repeated."
                ),
            }
        )

    if result.run.status in _TERMINAL_RUN_STATUSES:
        state = "terminal"
    elif warnings:
        state = "attention_required"
    elif running_tasks > 0:
        state = "processing"
    else:
        state = "waiting"

    payload = result.model_dump(mode="json")
    payload["diagnostics"] = {
        "state": state,
        "config_digest": {
            "run": result.run.config_digest,
            "local": local_digest,
            "matches": digest_matches,
        },
        "queued_tasks": queued_tasks,
        "running_tasks": running_tasks,
        "seconds_since_update": round(seconds_since_update, 3),
        "warnings": warnings,
    }
    return payload


def _active_task_counts(result: RunSummary) -> tuple[int, int]:
    """Return queued and running task totals from the bounded stage aggregates."""

    queued = sum(item.count for item in result.task_counts if item.status == TaskStatus.QUEUED)
    running = sum(item.count for item in result.task_counts if item.status == TaskStatus.RUNNING)
    return queued, running


def _stages_served_at_another_digest(result: RunSummary) -> list[str]:
    """Name the stages with open work whose fleet declared a different digest."""

    open_stages = {
        item.stage.value
        for item in result.task_counts
        if item.status in (TaskStatus.QUEUED, TaskStatus.RUNNING) and item.count
    }
    return sorted(
        stage
        for stage, digest in result.fleet_config_digests.items()
        if stage in open_stages and digest != result.run.config_digest
    )


def _as_utc(value: datetime) -> datetime:
    """Normalize persisted naive or aware timestamps for reliable age arithmetic."""

    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
