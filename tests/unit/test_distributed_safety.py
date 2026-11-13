"""Contracts for distributed planning and publication safety."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from kg_processor.adapters.distributed.postgres import _SCHEMA_STATEMENTS
from kg_processor.application.distributed_planner import _graph_output_payload
from kg_processor.application.distributed_worker import DistributedWorker
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import (
    ArtifactKind,
    ArtifactRef,
    PublicationLease,
    StoredArtifact,
    TaskStage,
)
from kg_processor.domain.finalization import GraphDatasetManifest
from kg_processor.domain.graph import GraphWriteBatch


def test_finalizer_rejects_arbitrary_environment_credential_references(tmp_path: Path) -> None:
    """A queued task cannot ask a worker to reveal an unrelated environment secret."""

    settings = _settings(tmp_path)
    settings.writer.provider = "snowflake_bulk"
    settings.snowflake.database = "DB"
    settings.snowflake.schema_name = "GRAPH"
    settings.snowflake.bulk_stage = "@DB.GRAPH.LOAD_STAGE"
    settings.snowflake.password_environment_variable = "AWS_SECRET_ACCESS_KEY"

    with pytest.raises(ValueError, match="dedicated password credential slot"):
        _graph_output_payload(settings)


def test_worker_drains_fenced_publication_before_claiming_another_task(tmp_path: Path) -> None:
    """Final task completion queues publication instead of exposing Snowflake inline."""

    manifest = GraphDatasetManifest(
        run_id="run-1",
        graph_id="graph-1",
        engine="spark",
        tables={},
    )
    artifact = StoredArtifact(
        ref=ArtifactRef(
            id="artifact-1",
            run_id="run-1",
            kind=ArtifactKind.GRAPH_RESULT,
            media_type="application/vnd.flakegraph.graph-manifest+json",
            checksum="abc",
            size_bytes=len(manifest.model_dump_json()),
        ),
        payload=manifest.model_dump_json().encode(),
    )
    store = _PublicationStore(artifact)
    publisher = _Publisher()
    worker = DistributedWorker(
        _settings(tmp_path),
        "finalizer-1",
        {TaskStage.FINALIZE_GRAPH},
        pipeline=cast(Any, object()),
        task_store=cast(Any, store),
        artifact_store=cast(Any, store),
        manifest_publisher=publisher,
    )

    result = worker.process_one()

    assert result.succeeded is True
    assert store.claim_task_calls == 0
    assert publisher.payloads[0]["publication"] == {
        "id": "publication-1",
        "generation": 17,
        "attempt": 1,
    }


class _PublicationStore:
    """Small structural fake for the optional fenced publication capability."""

    def __init__(self, artifact: StoredArtifact) -> None:
        self.artifact = artifact
        self.claim_task_calls = 0
        self.claimed = False

    def claim_publication(
        self,
        worker_id: str,
        lease_duration: timedelta,
    ) -> PublicationLease | None:
        del lease_duration
        if self.claimed:
            return None
        self.claimed = True
        return PublicationLease(
            id="publication-1",
            run_id="run-1",
            artifact_id=self.artifact.ref.id,
            task_payload={"output": {"provider": "snowflake_bulk"}},
            worker_id=worker_id,
            attempt=1,
            generation=17,
            lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )

    def publish_claimed(self, publication_id: str, worker_id: str, publish: Any) -> None:
        publish(self.claim_publication_lease(publication_id, worker_id))

    def heartbeat_publication(
        self,
        _publication_id: str,
        _worker_id: str,
        _lease_duration: timedelta,
    ) -> None:
        return None

    def claim_publication_lease(
        self,
        publication_id: str,
        worker_id: str,
    ) -> PublicationLease:
        return PublicationLease(
            id=publication_id,
            run_id="run-1",
            artifact_id=self.artifact.ref.id,
            task_payload={"output": {"provider": "snowflake_bulk"}},
            worker_id=worker_id,
            attempt=1,
            generation=17,
            lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )

    def fail_publication(self, *args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs))

    def claim_task(self, *args: object, **kwargs: object) -> None:
        self.claim_task_calls += 1

    def get(self, artifact_id: str) -> StoredArtifact:
        assert artifact_id == self.artifact.ref.id
        return self.artifact


class _Publisher:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def stage_batch_publication(
        self, batch: GraphWriteBatch, task_payload: Mapping[str, object]
    ) -> AbstractContextManager[Callable[[], None]]:
        """This fixture covers manifest publication exclusively."""

        raise AssertionError("A manifest command cannot stage a batch")

    def publish(self, manifest: GraphDatasetManifest, task_payload: Any) -> None:
        assert manifest.run_id == "run-1"
        self.payloads.append(dict(task_payload))


def _settings(tmp_path: Path) -> Settings:
    return Settings.load(
        env={},
        overrides={
            "job": {"graph_id": "graph-1"},
            "files": {"input_path": str(tmp_path)},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "api_key": "test"},
            "embedding": {"provider": "hash", "dimension": 8},
            "writer": {"provider": "local_artifacts", "output_path": str(tmp_path)},
            "cache": {"provider": "none"},
            "distributed": {"database_url": "postgresql://example/test"},
        },
    )


def test_demand_excludes_work_no_worker_can_claim() -> None:
    """Asking for workers to do unclaimable work holds a pool at its ceiling.

    A worker claims only a run carrying its own configuration digest, so runs
    created under a superseded configuration can never be claimed. A demand
    view with no digest predicate goes on counting them, which holds a
    KEDA-driven pool at its maximum against work nobody can take.
    """

    statements = "\n".join(_SCHEMA_STATEMENTS)

    assert "CREATE TABLE IF NOT EXISTS flakegraph_worker_fleet" in statements
    assert "FROM flakegraph_worker_fleet AS fleet" in statements
    assert "fleet.config_digest <> run.config_digest" in statements
    # Expressed as NOT EXISTS, so a stage with no recorded fleet is still
    # counted and a first install can scale from zero before a worker has run.
    assert "AND NOT EXISTS (" in statements


def test_the_served_configuration_does_not_expire() -> None:
    """A liveness window here would break scaling up from zero.

    The view asks what this fleet serves, not who is alive. Expiring the record
    would erase the answer exactly when a pool has scaled to zero, which is the
    moment the demand signal has to be right.
    """

    fleet = next(
        statement
        for statement in _SCHEMA_STATEMENTS
        if "CREATE TABLE IF NOT EXISTS flakegraph_worker_fleet" in statement
    )

    assert "stage TEXT PRIMARY KEY" in fleet
    assert "config_digest TEXT NOT NULL" in fleet
    assert "INTERVAL" not in fleet
