"""Fast contract tests for PostgreSQL plan validation helpers."""

from __future__ import annotations

import pytest

from kg_processor.adapters.distributed.postgres import PostgresDistributedStore, _validate_acyclic
from kg_processor.domain.distributed import TaskDefinition, TaskStage


def test_cycle_is_rejected_before_tasks_reach_postgres() -> None:
    """Keep a malformed plan from creating workers that wait forever."""

    run_id = "cyclic-plan"
    first = TaskDefinition(
        id="first",
        run_id=run_id,
        stage=TaskStage.PREPARE_DOCUMENT,
        scope_id="first-file",
        dependency_ids=["second"],
    )
    second = TaskDefinition(
        id="second",
        run_id=run_id,
        stage=TaskStage.PREPARE_DOCUMENT,
        scope_id="second-file",
        dependency_ids=["first"],
    )

    with pytest.raises(ValueError, match="must be acyclic"):
        _validate_acyclic([first, second])


def test_recent_run_list_rejects_unbounded_limits_before_connecting() -> None:
    """Keep the operator-facing history query bounded independently of corpus size."""

    store = PostgresDistributedStore("postgresql://example.invalid/flakegraph")

    with pytest.raises(ValueError, match="between 1 and 500"):
        store.list_runs(0)
    with pytest.raises(ValueError, match="between 1 and 500"):
        store.list_runs(501)
