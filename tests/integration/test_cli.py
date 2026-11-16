from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from kg_processor import __version__
from kg_processor.adapters.jobs.snowflake import SnowflakeJobManager
from kg_processor.application.distributed_planner import distributed_processing_config_digest
from kg_processor.application.progress import ProgressEvent, ProgressSink
from kg_processor.application.snowflake_access import SnowflakeAccessCheck, SnowflakeAccessReport
from kg_processor.cli import WorkerQueueBlockedError, _run_file_queue_worker, app
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import (
    ArtifactKind,
    ArtifactRef,
    PriorityBand,
    RunDefinition,
    RunSnapshot,
    RunStatus,
    RunSummary,
    StoredArtifact,
    TaskCount,
    TaskDefinition,
    TaskSnapshot,
    TaskStage,
    TaskStatus,
)
from kg_processor.domain.documents import InputFile, SourceListing
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.domain.jobs import JobFileClaim, JobFileResult

runner = CliRunner()


def test_cli_version_option_reports_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"FlakeGraph {__version__}"


def test_cli_config_print_redacts_api_keys(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
llm:
  provider: openai_compatible
  endpoint: https://example.test/v1
  api_key: super-secret
snowflake:
  password: snowflake-password
  oauth_token: bearer-token
  private_key_path: /tmp/private-key.pem
azure_blob:
  connection_string: DefaultEndpointsProtocol=https;AccountKey=secret
  sas_token: sv=secret
generic_http_ocr:
  api_key: ocr-secret
  api_key_header: X-API-Key
  api_key_prefix: ""
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["config", "print", "--config", str(config)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["llm"]["api_key"] == "***"
    assert payload["snowflake"]["password"] == "***"
    assert payload["snowflake"]["oauth_token"] == "***"
    assert payload["snowflake"]["private_key_path"] == "***"
    assert payload["azure_blob"]["connection_string"] == "***"
    assert payload["azure_blob"]["sas_token"] == "***"
    assert payload["generic_http_ocr"]["api_key"] == "***"
    assert payload["generic_http_ocr"]["api_key_header"] == "X-API-Key"
    assert payload["generic_http_ocr"]["api_key_prefix"] == ""


def test_cli_sources_list_names_local_documents_as_json(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "judo.md").write_text("Judo was founded by Kano Jigoro.", encoding="utf-8")
    (corpus / "karate.txt").write_text("Karate comes from Okinawa.", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(f"files:\n  source: local\n  input_path: {corpus}\n", encoding="utf-8")

    result = runner.invoke(app, ["sources", "list", "--config", str(config), "--limit", "1"])

    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert [row["name"] for row in rows] == ["judo.md"]
    assert rows[0]["uri"] == (corpus / "judo.md").resolve().as_uri()
    assert rows[0]["size_bytes"] == 32
    assert rows[0]["checksum"] is None


def test_cli_sources_list_reports_a_backend_refusal_in_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("files:\n  source: local\n  input_path: /tmp\n", encoding="utf-8")

    class _RefusingSource:
        def browse(self, limit: int) -> Iterator[SourceListing]:
            raise RuntimeError(
                "An error occurred (NoSuchBucket): The specified bucket does not exist"
            )
            yield  # pragma: no cover - makes this a generator like the real sources

    monkeypatch.setattr(
        "kg_processor.factories.build_file_source", lambda settings: _RefusingSource()
    )

    result = runner.invoke(app, ["sources", "list", "--config", str(config)])

    assert result.exit_code == 1
    assert "local source could not be listed: An error occurred (NoSuchBucket)" in result.output
    assert "Traceback" not in result.output


def test_cli_sources_list_refuses_a_source_that_can_only_be_fetched(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"files:\n  source: manifest\n  manifest_path: {manifest}\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["sources", "list", "--config", str(config)])

    assert result.exit_code == 2
    assert "manifest sources cannot be listed without fetching" in result.output


def test_cli_config_providers_lists_supported_provider_catalog() -> None:
    result = runner.invoke(app, ["config", "providers", "--kind", "ocr"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    names = {provider["name"] for provider in payload}
    assert {provider["kind"] for provider in payload} == {"ocr"}
    assert {"mineru_internal", "mineru_api", "snowflake_cortex"} <= names
    assert "openai_compatible" not in names


def test_cli_config_providers_rejects_unknown_kind() -> None:
    result = runner.invoke(app, ["config", "providers", "--kind", "unknown"])

    assert result.exit_code != 0
    assert "Unsupported provider kind 'unknown'" in result.output


def test_cli_preflight_and_worker_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
files:
  input_path: {input_dir}
ocr:
  provider: builtin_text
llm:
  provider: fake
  model: fake
embedding:
  provider: hash
  dimension: 8
graph:
  chunk_token_size: 30
  chunk_token_overlap: 5
  gleaning_max_passes: 0
writer:
  output_path: {output_dir}
""",
        encoding="utf-8",
    )

    preflight = runner.invoke(app, ["preflight", "--config", str(config)])
    worker = runner.invoke(app, ["worker", "--config", str(config)])
    inspect = runner.invoke(app, ["inspect", "graph", "--output", str(output_dir)])
    opened_urls: list[str] = []
    monkeypatch.setattr("kg_processor.cli.webbrowser.open", opened_urls.append)
    explorer = runner.invoke(
        app,
        ["inspect", "html", "--output", str(output_dir), "--open"],
    )

    assert preflight.exit_code == 0
    assert worker.exit_code == 0
    assert inspect.exit_code == 0
    assert explorer.exit_code == 0
    report = json.loads(worker.stdout)
    inspection = json.loads(inspect.stdout)
    explorer_result = json.loads(explorer.stdout)
    progress = [json.loads(line) for line in worker.stderr.splitlines() if line.strip()]
    assert report["files_processed"] == 1
    assert inspection["tables"]["nodes"] > 0
    assert inspection["trace_events"] > 0
    assert explorer_result["self_contained"] is True
    assert explorer_result["counts"]["nodes"] > 0
    assert Path(explorer_result["html_path"]).is_file()
    assert opened_urls == [Path(explorer_result["html_path"]).resolve().as_uri()]
    assert any(event["stage"] == "ocr" and event["status"] == "completed" for event in progress)
    assert any(event["stage"] == "write" and event["status"] == "completed" for event in progress)


def test_cli_worker_can_force_rich_terminal_progress(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    config = _write_local_config(tmp_path / "config.yaml", input_dir, output_dir)

    result = runner.invoke(
        app,
        ["worker", "--config", str(config), "--progress", "rich"],
    )

    assert result.exit_code == 0
    assert result.stdout == ""
    assert "FlakeGraph" in result.stderr
    assert "Discover files" in result.stderr
    assert "Extract graph" in result.stderr
    assert "Graph ready" in result.stderr
    assert "Artifacts" in result.stderr


def test_cli_worker_can_disable_progress_without_hiding_final_json(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    config = _write_local_config(tmp_path / "config.yaml", input_dir, output_dir)

    result = runner.invoke(
        app,
        ["worker", "--config", str(config), "--progress", "none"],
    )

    assert result.exit_code == 0
    assert result.stderr == ""
    assert json.loads(result.stdout)["files_processed"] == 1


def test_cli_compare_reports_stable_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    first_output = tmp_path / "out-first"
    second_output = tmp_path / "out-second"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    first_config = _write_local_config(tmp_path / "first.yaml", input_dir, first_output)
    second_config = _write_local_config(tmp_path / "second.yaml", input_dir, second_output)

    first_worker = runner.invoke(app, ["worker", "--config", str(first_config)])
    second_worker = runner.invoke(app, ["worker", "--config", str(second_config)])
    compare = runner.invoke(
        app,
        ["inspect", "compare", "--left", str(first_output), "--right", str(second_output)],
    )

    assert first_worker.exit_code == 0
    assert second_worker.exit_code == 0
    assert compare.exit_code == 0
    comparison = json.loads(compare.stdout)
    assert comparison["ok"]
    assert any(check["name"] == "nodes_stable_identity" for check in comparison["checks"])


def test_cli_snowflake_ddl_uses_configured_embedding_dimension(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
embedding:
  dimension: 42
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["snowflake", "ddl", "--config", str(config)])

    assert result.exit_code == 0
    assert "VECTOR(FLOAT, 42)" in result.stdout
    assert "CREATE TABLE IF NOT EXISTS KG_GRAPH_METRICS" in result.stdout


def test_distributed_init_declares_what_the_deployed_fleet_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deployment, not only a running worker, records the fleet's digest.

    A pool scaled to zero has no worker to declare a new digest, and the demand
    signal counts only work the declared fleet can take, so the bootstrap hook
    declares it for every stage it deploys.
    """

    settings = Settings.load(overrides={"distributed": {"database_url": "postgresql://fleet"}})
    declared: dict[str, object] = {}

    class FleetStore:
        def initialize(self) -> None:
            declared["initialized"] = True

        def record_served_configuration(self, stages: set[TaskStage], config_digest: str) -> None:
            declared["stages"] = stages
            declared["digest"] = config_digest

    monkeypatch.setattr(
        "kg_processor.factories.build_distributed_store", lambda _settings: FleetStore()
    )
    monkeypatch.setattr("kg_processor.cli.Settings.load", lambda _config: settings)

    result = runner.invoke(
        app,
        [
            "distributed",
            "init",
            "--serve-stage",
            "prepare_document",
            "--serve-stage",
            "finalize_graph",
        ],
    )

    assert result.exit_code == 0, result.output
    assert declared == {
        "initialized": True,
        "stages": {TaskStage.PREPARE_DOCUMENT, TaskStage.FINALIZE_GRAPH},
        "digest": distributed_processing_config_digest(settings),
    }
    payload = json.loads(result.stdout)
    assert payload["served_stages"] == ["finalize_graph", "prepare_document"]
    assert payload["config_digest"] == distributed_processing_config_digest(settings)

    declared.clear()
    result = runner.invoke(app, ["distributed", "init"])

    assert result.exit_code == 0, result.output
    assert declared == {"initialized": True}
    assert json.loads(result.stdout)["served_stages"] == []


def test_distributed_publish_writes_a_finished_graph_through_the_configured_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publishing is a write of the run's final batch, never a second extraction."""

    settings = Settings.load(
        overrides={
            "distributed": {"database_url": "postgresql://fleet"},
            "writer": {"provider": "snowflake_bulk"},
            "snowflake": {"account": "acct", "user": "svc", "bulk_stage": "STAGE"},
        }
    )
    batch = GraphWriteBatch(
        graph_id="graph-1",
        write_scope="graph_snapshot",
        reindex_file_ids=[],
        documents=[],
        pages=[],
        chunks=[],
        nodes=[],
        edges=[],
        evidence=[],
        entity_sources=[],
        communities=[],
        community_findings=[],
        run_report={},
        graph_metrics={},
        extraction_trace=[],
    )
    payload = batch.model_dump_json().encode()
    run = RunDefinition(
        id="run-1", graph_id="graph-1", config={}, config_digest="d", status=RunStatus.SUCCEEDED
    )
    now = datetime.now(UTC)

    class FleetStore:
        def initialize(self) -> None:
            return None

        def get_run(self, run_id: str) -> RunSnapshot:
            assert run_id == "run-1"
            return RunSnapshot(
                run=run,
                tasks=[
                    TaskSnapshot(
                        task=TaskDefinition(
                            id="final",
                            run_id="run-1",
                            stage=TaskStage.FINALIZE_GRAPH,
                            scope_id="graph",
                        ),
                        status=TaskStatus.SUCCEEDED,
                        attempts=1,
                        output_artifact_ids=["artifact-1"],
                    )
                ],
                created_at=now,
                updated_at=now,
            )

        def get(self, artifact_id: str) -> StoredArtifact:
            assert artifact_id == "artifact-1"
            return StoredArtifact(
                ref=ArtifactRef(
                    id=artifact_id,
                    run_id="run-1",
                    kind=ArtifactKind.GRAPH_RESULT,
                    media_type="application/json",
                    checksum="c",
                    size_bytes=len(payload),
                ),
                payload=payload,
            )

    written: list[GraphWriteBatch] = []

    class Writer:
        def write(self, value: GraphWriteBatch) -> None:
            written.append(value)

    monkeypatch.setattr(
        "kg_processor.factories.build_distributed_store", lambda _settings: FleetStore()
    )
    monkeypatch.setattr("kg_processor.factories.build_writer", lambda _settings: Writer())
    monkeypatch.setattr("kg_processor.cli.Settings.load", lambda _config: settings)

    result = runner.invoke(app, ["distributed", "publish", "--run-id", "run-1"])

    assert result.exit_code == 0, result.output
    assert [item.graph_id for item in written] == ["graph-1"]
    assert json.loads(result.stdout) == {
        "run_id": "run-1",
        "graph_id": "graph-1",
        "writer": "snowflake_bulk",
        "nodes": 0,
        "edges": 0,
    }


def test_distributed_status_keeps_the_fleet_warning_when_tasks_are_included(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verdict comes from the aggregates whether or not the tasks are printed."""

    settings = Settings.load(overrides={"distributed": {"database_url": "postgresql://fleet"}})
    now = datetime.now(UTC)
    run = RunDefinition(
        id="run-1",
        graph_id="graph-1",
        config={},
        config_digest=distributed_processing_config_digest(settings),
        status=RunStatus.RUNNING,
    )
    task = TaskDefinition(
        id="task-1", run_id="run-1", stage=TaskStage.EXTRACT_ENTITY_WINDOW, scope_id="w-1"
    )

    class FleetStore:
        def initialize(self) -> None:
            return None

        def get_run_summary(self, run_id: str) -> RunSummary:
            assert run_id == "run-1"
            return RunSummary(
                run=run,
                task_counts=[
                    TaskCount(
                        stage=TaskStage.EXTRACT_ENTITY_WINDOW, status=TaskStatus.QUEUED, count=1
                    )
                ],
                total_tasks=1,
                created_at=now,
                updated_at=now,
                fleet_config_digests={TaskStage.EXTRACT_ENTITY_WINDOW.value: "after-upgrade"},
            )

        def get_run(self, run_id: str) -> RunSnapshot:
            assert run_id == "run-1"
            return RunSnapshot(
                run=run,
                tasks=[TaskSnapshot(task=task, status=TaskStatus.QUEUED, attempts=0)],
                created_at=now,
                updated_at=now,
            )

    monkeypatch.setattr(
        "kg_processor.factories.build_distributed_store", lambda _settings: FleetStore()
    )
    monkeypatch.setattr("kg_processor.cli.Settings.load", lambda _config: settings)

    result = runner.invoke(app, ["distributed", "status", "--run-id", "run-1", "--include-tasks"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [item["code"] for item in payload["diagnostics"]["warnings"]] == [
        "FLEET_DIGEST_MISMATCH"
    ]
    assert [item["task"]["id"] for item in payload["tasks"]] == ["task-1"]


def test_file_queue_worker_drains_claimed_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings.load(
        overrides={
            "job": {
                "job_id": "job",
                "graph_id": "graph",
                "use_file_queue": True,
                "lease_owner": "worker-1",
                "file_batch_size": 1,
            },
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )
    manager = _FakeQueueManager(
        [
            [JobFileClaim(job_id="job", graph_id="graph", file_id="file_1")],
            [JobFileClaim(job_id="job", graph_id="graph", file_id="file_2")],
            [],
        ]
    )

    def fake_build_pipeline(
        _settings: Settings,
        claimed_files: list[JobFileClaim] | None = None,
        progress_sink: ProgressSink | None = None,
        write_scope_override: str | None = None,
        consumption: object | None = None,
    ) -> _FakeQueuePipeline:
        _ = write_scope_override, consumption
        return _FakeQueuePipeline(claimed_files or [], progress_sink)

    monkeypatch.setattr("kg_processor.factories.build_pipeline", fake_build_pipeline)

    summary = _run_file_queue_worker(settings, cast(SnowflakeJobManager, manager))

    assert summary["claimed"] is True
    assert summary["drained"] is True
    assert summary["batches_processed"] == 2
    assert summary["files_processed"] == 2
    assert manager.claim_calls == 3
    assert manager.completed_file_ids == [["file_1"], ["file_2"]]
    assert manager.drained_reports == ["run_file_1", "run_file_2"]
    # STAGE names the stage alone; the row's own STATUS and the job row's
    # progress payload carry the status the app renders.
    assert manager.progress_updates == [
        ("job", "graph", "worker-1", ["file_1"], "ocr"),
        ("job", "graph", "worker-1", ["file_2"], "ocr"),
    ]


def test_file_queue_worker_publishes_snapshot_for_complete_queue_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single complete claim must retain communities and other final artifacts."""

    settings = Settings.load(
        overrides={
            "job": {
                "job_id": "job",
                "graph_id": "graph",
                "use_file_queue": True,
                "lease_owner": "worker-1",
                "file_batch_size": 25,
            },
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )
    manager = _FakeQueueManager(
        [
            [JobFileClaim(job_id="job", graph_id="graph", file_id="file_1")],
            [],
        ],
        owns_entire_queue=True,
    )
    write_scopes: list[str | None] = []

    def fake_build_pipeline(
        _settings: Settings,
        claimed_files: list[JobFileClaim] | None = None,
        progress_sink: ProgressSink | None = None,
        write_scope_override: str | None = None,
        consumption: object | None = None,
    ) -> _FakeQueuePipeline:
        write_scopes.append(write_scope_override)
        return _FakeQueuePipeline(claimed_files or [], progress_sink)

    monkeypatch.setattr("kg_processor.factories.build_pipeline", fake_build_pipeline)

    summary = _run_file_queue_worker(settings, cast(SnowflakeJobManager, manager))

    assert summary["files_processed"] == 1
    assert write_scopes == ["graph_snapshot"]


def test_file_queue_worker_heartbeats_claimed_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(
        overrides={
            "job": {
                "job_id": "job",
                "graph_id": "graph",
                "use_file_queue": True,
                "lease_owner": "worker-1",
                "lease_seconds": 6,
                "file_batch_size": 1,
            },
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )
    manager = _FakeQueueManager(
        [
            [JobFileClaim(job_id="job", graph_id="graph", file_id="file_1")],
            [],
        ]
    )
    heartbeat_intervals: list[float] = []

    def fake_build_pipeline(
        _settings: Settings,
        claimed_files: list[JobFileClaim] | None = None,
        progress_sink: ProgressSink | None = None,
        write_scope_override: str | None = None,
        consumption: object | None = None,
    ) -> _FakeQueuePipeline:
        _ = write_scope_override, consumption
        return _FakeQueuePipeline(claimed_files or [], progress_sink)

    class ImmediateHeartbeat:
        def __init__(self, heartbeat: Callable[[], None], interval_seconds: float) -> None:
            self.heartbeat = heartbeat
            heartbeat_intervals.append(interval_seconds)

        def __enter__(self) -> ImmediateHeartbeat:
            self.heartbeat()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object,
        ) -> None:
            return None

        def raise_if_unhealthy(self) -> None:
            """Model the healthy heartbeat contract used by the worker."""

    monkeypatch.setattr("kg_processor.factories.build_pipeline", fake_build_pipeline)
    monkeypatch.setattr(
        "kg_processor.application.lease_heartbeat.LeaseHeartbeat", ImmediateHeartbeat
    )

    summary = _run_file_queue_worker(settings, cast(SnowflakeJobManager, manager))

    assert summary["files_processed"] == 1
    assert heartbeat_intervals == [2]
    assert manager.heartbeat_file_ids == [["file_1"]]


def test_file_queue_worker_fails_claims_when_pipeline_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(
        overrides={
            "job": {
                "job_id": "job",
                "graph_id": "graph",
                "use_file_queue": True,
                "lease_owner": "worker-1",
                "file_batch_size": 1,
            },
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )
    manager = _FakeQueueManager([[JobFileClaim(job_id="job", graph_id="graph", file_id="file_1")]])

    def failing_build_pipeline(
        _settings: Settings,
        claimed_files: list[JobFileClaim] | None = None,
        progress_sink: ProgressSink | None = None,
        write_scope_override: str | None = None,
        consumption: object | None = None,
    ) -> _FakeQueuePipeline:
        _ = (claimed_files, progress_sink, write_scope_override)
        raise RuntimeError("missing provider credentials")

    monkeypatch.setattr("kg_processor.factories.build_pipeline", failing_build_pipeline)

    with pytest.raises(RuntimeError, match="missing provider credentials"):
        _run_file_queue_worker(settings, cast(SnowflakeJobManager, manager))

    assert manager.failed_file_ids == [["file_1"]]
    assert manager.failed_errors[0]["error_type"] == "RuntimeError"
    assert manager.drained_reports == ["job:failed"]


def test_file_queue_worker_fails_only_pipeline_reported_file_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(
        overrides={
            "job": {
                "job_id": "job",
                "graph_id": "graph",
                "use_file_queue": True,
                "lease_owner": "worker-1",
                "file_batch_size": 2,
            },
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )
    manager = _FakeQueueManager(
        [
            [
                JobFileClaim(job_id="job", graph_id="graph", file_id="good_file"),
                JobFileClaim(job_id="job", graph_id="graph", file_id="bad_file"),
            ],
            [],
        ]
    )

    def fake_build_pipeline(
        _settings: Settings,
        claimed_files: list[JobFileClaim] | None = None,
        progress_sink: ProgressSink | None = None,
        write_scope_override: str | None = None,
        consumption: object | None = None,
    ) -> _PartiallyFailedQueuePipeline:
        _ = write_scope_override, consumption
        return _PartiallyFailedQueuePipeline(claimed_files or [], progress_sink)

    monkeypatch.setattr("kg_processor.factories.build_pipeline", fake_build_pipeline)

    summary = _run_file_queue_worker(settings, cast(SnowflakeJobManager, manager))

    assert summary["claimed"] is True
    assert summary["drained"] is True
    assert summary["batches_processed"] == 1
    assert summary["files_processed"] == 1
    assert manager.completed_file_ids == [["good_file"]]
    assert manager.failed_file_ids == [["bad_file"]]
    assert manager.failed_errors == [{"type": "RuntimeError", "message": "bad PDF"}]


def _write_local_config(path: Path, input_dir: Path, output_dir: Path) -> Path:
    path.write_text(
        f"""
job:
  job_id: stable-test
  graph_id: stable-graph
files:
  input_path: {input_dir}
ocr:
  provider: builtin_text
llm:
  provider: fake
  model: fake
embedding:
  provider: hash
  dimension: 8
graph:
  chunk_token_size: 30
  chunk_token_overlap: 5
  gleaning_max_passes: 0
writer:
  output_path: {output_dir}
""",
        encoding="utf-8",
    )
    return path


class _FakeQueueManager:
    def __init__(
        self,
        claim_batches: list[list[JobFileClaim]],
        *,
        owns_entire_queue: bool = False,
        status_counts: dict[str, int] | None = None,
    ) -> None:
        self.claim_batches = claim_batches
        self.owns_entire_queue = owns_entire_queue
        self.status_counts = status_counts or {}
        self.claim_calls = 0
        self.completed_file_ids: list[list[str]] = []
        self.drained_reports: list[str] = []
        self.heartbeat_file_ids: list[list[str]] = []
        self.progress_updates: list[tuple[str, str, str, list[str], str]] = []
        self.failed_file_ids: list[list[str]] = []
        self.failed_errors: list[dict[str, object]] = []

    def claim_job_files(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> list[JobFileClaim]:
        self.claim_calls += 1
        return self.claim_batches.pop(0)

    def worker_owns_entire_file_queue(
        self,
        _job_id: str,
        _graph_id: str,
        _worker_id: str,
        _file_ids: list[str],
    ) -> bool:
        """Mirror the ownership decision made by the Snowflake queue adapter."""

        return self.owns_entire_queue

    def complete_job_files(
        self,
        _job_id: str,
        _graph_id: str,
        _worker_id: str,
        results: list[JobFileResult],
    ) -> None:
        self.completed_file_ids.append([result.file_id for result in results])

    def count_job_files_by_status(self, _job_id: str, _graph_id: str) -> dict[str, int]:
        return dict(self.status_counts)

    def complete_job_if_file_queue_drained(
        self,
        _job_id: str,
        _graph_id: str,
        report: dict[str, object],
    ) -> None:
        # The drained summary carries no run id when no batch was processed.
        self.drained_reports.append(str(report.get("run_id", "drained")))

    def fail_job_files(
        self,
        _job_id: str,
        _graph_id: str,
        _worker_id: str,
        file_ids: list[str],
        error: dict[str, object],
    ) -> None:
        self.failed_file_ids.append(file_ids)
        self.failed_errors.append(error)

    def heartbeat_job_files(
        self,
        _job_id: str,
        _graph_id: str,
        _worker_id: str,
        file_ids: list[str],
        _lease_seconds: int,
    ) -> None:
        self.heartbeat_file_ids.append(file_ids)

    def update_job_file_progress(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        file_ids: list[str],
        stage: str,
    ) -> None:
        self.progress_updates.append((job_id, graph_id, worker_id, file_ids, stage))


class _FakeQueuePipeline:
    def __init__(
        self,
        claimed_files: list[JobFileClaim],
        progress_sink: ProgressSink | None = None,
    ) -> None:
        self.claimed_files = claimed_files
        self.progress_sink = progress_sink

    def close(self) -> None:
        """Match the lifecycle contract of the real pipeline."""

    def run(self) -> GraphWriteBatch:
        file_ids = [claim.file_id for claim in self.claimed_files]
        if self.progress_sink is not None:
            self.progress_sink.emit(
                ProgressEvent(
                    job_id="job",
                    graph_id="graph",
                    stage="ocr",
                    status="started",
                    file_id=file_ids[0],
                )
            )
        return GraphWriteBatch(
            graph_id="graph",
            write_scope="file_batch",
            reindex_file_ids=file_ids,
            documents=[],
            pages=[],
            chunks=[],
            nodes=[],
            edges=[],
            evidence=[],
            entity_sources=[],
            communities=[],
            community_findings=[],
            run_report={
                "job_id": "job",
                "graph_id": "graph",
                "run_id": f"run_{file_ids[0]}",
                "files_processed": len(file_ids),
            },
            graph_metrics={},
            extraction_trace=[],
        )

    def job_file_results(self, _batch: GraphWriteBatch) -> list[JobFileResult]:
        return [
            JobFileResult(file_id=claim.file_id, rows_written=1) for claim in self.claimed_files
        ]

    def failed_job_file_results(self) -> list[JobFileResult]:
        return []


class _PartiallyFailedQueuePipeline:
    def __init__(
        self,
        claimed_files: list[JobFileClaim],
        progress_sink: ProgressSink | None = None,
    ) -> None:
        self.claimed_files = claimed_files
        self.progress_sink = progress_sink

    def close(self) -> None:
        """Match the lifecycle contract of the real pipeline."""

    def run(self) -> GraphWriteBatch:
        _ = self.progress_sink
        return GraphWriteBatch(
            graph_id="graph",
            write_scope="file_batch",
            reindex_file_ids=["good_file"],
            documents=[],
            pages=[],
            chunks=[],
            nodes=[],
            edges=[],
            evidence=[],
            entity_sources=[],
            communities=[],
            community_findings=[],
            run_report={
                "job_id": "job",
                "graph_id": "graph",
                "run_id": "run_partial",
                "files_processed": 1,
            },
            graph_metrics={},
            extraction_trace=[],
        )

    def job_file_results(self, _batch: GraphWriteBatch) -> list[JobFileResult]:
        return [JobFileResult(file_id="good_file", rows_written=1)]

    def failed_job_file_results(self) -> list[JobFileResult]:
        return [
            JobFileResult(
                file_id="bad_file",
                rows_written=0,
                stage="ocr_failed",
                audit={"error": {"type": "RuntimeError", "message": "bad PDF"}},
            )
        ]


def test_cli_snowflake_deployment_commands_render_from_config(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
runtime:
  runtime: spcs
job:
  job_id: job-123
  graph_id: graph-123
  use_lease: true
  lease_owner: worker-1
files:
  source: snowflake_stage
ocr:
  provider: snowflake_cortex
llm:
  provider: snowflake_cortex
  model: llama3.3-70b
embedding:
  provider: snowflake_cortex
  model: snowflake-arctic-embed-l-v2.0
  dimension: 1024
writer:
  provider: snowflake_bulk
cache:
  provider: snowflake
snowflake:
  account: EXAMPLE_ACCOUNT
  database: KG_DB
  schema: GRAPH
  role: KG_PROCESSOR_ROLE
  warehouse: KG_PROCESSOR_WH
  stage: "@KG_DB.GRAPH.KG_DOCS"
  bulk_stage: "@KG_DB.GRAPH.KG_LOAD_STAGE"
  image_repository: KG_DB.GRAPH.KG_IMAGES
  image_name: flakegraph:latest
  image_digest: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  compute_pool: KG_PROCESSOR_CPU_POOL
  service_name: KG_PROCESSOR_JOB
  service_spec_stage: "@KG_DB.GRAPH.KG_SERVICE_SPECS"
""",
        encoding="utf-8",
    )

    spec = runner.invoke(app, ["snowflake", "service-spec", "--config", str(config)])
    image_reference = runner.invoke(
        app,
        ["snowflake", "image-reference", "--config", str(config)],
    )
    execute_sql = runner.invoke(app, ["snowflake", "execute-job-sql", "--config", str(config)])
    setup_sql = runner.invoke(app, ["snowflake", "setup-sql", "--config", str(config)])
    objects_sql = runner.invoke(app, ["snowflake", "objects-sql", "--config", str(config)])
    kubernetes = runner.invoke(
        app,
        [
            "snowflake",
            "kubernetes-job",
            "--config",
            str(config),
            "--image",
            "registry.example/flakegraph:latest",
            "--secret-name",
            "flakegraph-secrets",
            "--parallelism",
            "2",
            "--gpu-count",
            "1",
        ],
    )

    assert spec.exit_code == 0
    assert image_reference.exit_code == 0
    assert execute_sql.exit_code == 0
    assert setup_sql.exit_code == 0
    assert objects_sql.exit_code == 0
    assert kubernetes.exit_code == 0
    assert (
        "/KG_DB/GRAPH/KG_IMAGES/flakegraph:latest@"
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    ) in spec.stdout
    assert image_reference.stdout.strip() == (
        "/KG_DB/GRAPH/KG_IMAGES/flakegraph:latest@"
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )
    assert "KG_SNOWFLAKE_AUTHENTICATOR: oauth" in spec.stdout
    assert "EXECUTE JOB SERVICE" in execute_sql.stdout
    assert "NAME = KG_DB.GRAPH.KG_PROCESSOR_JOB" in execute_sql.stdout
    assert "FROM @KG_DB.GRAPH.KG_SERVICE_SPECS" in execute_sql.stdout
    assert "CREATE IMAGE REPOSITORY IF NOT EXISTS KG_DB.GRAPH.KG_IMAGES" in setup_sql.stdout
    assert "CREATE TABLE IF NOT EXISTS KG_NODE" in setup_sql.stdout
    assert "CREATE ROLE" not in objects_sql.stdout
    assert "GRANT " not in objects_sql.stdout
    assert "CREATE STAGE IF NOT EXISTS KG_DB.GRAPH.KG_DOCS" in objects_sql.stdout
    assert "CREATE TABLE IF NOT EXISTS KG_NODE" in objects_sql.stdout
    assert "kind: Job" in kubernetes.stdout
    assert "parallelism: 2" in kubernetes.stdout
    assert "name: flakegraph-secrets" in kubernetes.stdout
    assert "nvidia.com/gpu: '1'" in kubernetes.stdout


def test_cli_snowflake_access_check_outputs_report_and_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
snowflake:
  account: EXAMPLE_ACCOUNT
  database: KG_DB
  schema: GRAPH
""",
        encoding="utf-8",
    )

    def fake_access_check(*_args: object, **_kwargs: object) -> SnowflakeAccessReport:
        return SnowflakeAccessReport(
            ok=False,
            checks=[
                SnowflakeAccessCheck(
                    name="target_tables",
                    ok=False,
                    message="Some FlakeGraph tables are missing or not visible",
                    details={"missing": ["KG_NODE"]},
                )
            ],
        )

    monkeypatch.setattr(
        "kg_processor.application.snowflake_access.run_snowflake_access_check", fake_access_check
    )

    result = runner.invoke(app, ["snowflake", "access-check", "--config", str(config)])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["checks"][0]["details"]["missing"] == ["KG_NODE"]


def test_cli_snowflake_submit_discovers_and_queues_stage_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the user-facing handoff from stage discovery to the SPCS queue."""

    config = tmp_path / "config.yaml"
    config.write_text(
        """
runtime:
  runtime: spcs
job:
  job_id: job-123
  graph_id: graph-123
  use_file_queue: true
  lease_owner: worker-1
files:
  source: snowflake_stage
snowflake:
  account: EXAMPLE_ACCOUNT
  database: KG_DB
  schema: GRAPH
  stage: "@KG_DB.GRAPH.KG_DOCS"
  password: never-emit-this
""",
        encoding="utf-8",
    )
    files = [
        InputFile(
            id="file-1",
            path=Path("incoming/a.pdf"),
            source_uri="@KG_DB.GRAPH.KG_DOCS/incoming/a.pdf",
            checksum="checksum-a",
            mime_type="application/pdf",
            size_bytes=123,
        )
    ]

    class FakeSource:
        def list_files(self) -> list[InputFile]:
            return files

    class FakeManager:
        submitted_config: dict[str, object] | None = None

        def submit_job_files(
            self,
            job_id: str,
            graph_id: str,
            submitted_files: list[InputFile],
            submitted_config: dict[str, object],
        ) -> int:
            assert (job_id, graph_id) == ("job-123", "graph-123")
            assert submitted_files == files
            self.submitted_config = submitted_config
            return len(submitted_files)

    manager = FakeManager()
    monkeypatch.setattr("kg_processor.factories.build_file_source", lambda _settings: FakeSource())
    monkeypatch.setattr("kg_processor.factories.build_job_manager", lambda _settings: manager)

    result = runner.invoke(app, ["snowflake", "submit", "--config", str(config)])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "files_queued": 1,
        "graph_id": "graph-123",
        "job_id": "job-123",
    }
    assert manager.submitted_config is not None
    submitted_snowflake = cast(dict[str, object], manager.submitted_config["snowflake"])
    assert submitted_snowflake["password"] == "***"


def _queue_worker_settings() -> Settings:
    return Settings.model_validate(
        {
            "runtime": {"runtime": "spcs"},
            "job": {
                "job_id": "job",
                "graph_id": "graph",
                "use_file_queue": True,
                "lease_owner": "worker-1",
                "file_batch_size": 1,
            },
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )


def test_file_queue_worker_refuses_to_report_success_for_a_wholly_failed_queue() -> None:
    """A crashed run leaves every row FAILED, and nothing is then claimable.

    Draining without work is otherwise indistinguishable from a finished graph
    once the service reports terminal state, so the operator would read a stuck
    run as a successful one.
    """

    manager = _FakeQueueManager([[]], status_counts={"FAILED": 10})

    with pytest.raises(WorkerQueueBlockedError, match="retry"):
        _run_file_queue_worker(_queue_worker_settings(), cast(SnowflakeJobManager, manager))

    assert manager.drained_reports == []


def test_file_queue_worker_still_succeeds_when_the_queue_genuinely_finished() -> None:
    """Starting after every file is DONE remains a normal, successful drain."""

    manager = _FakeQueueManager([[]], status_counts={"DONE": 10})

    summary = _run_file_queue_worker(_queue_worker_settings(), cast(SnowflakeJobManager, manager))

    assert summary["drained"] is True
    assert summary["files_processed"] == 0
    assert summary["queue_status_counts"] == {"DONE": 10}
    assert "blocked_files" not in summary


def test_file_queue_worker_tolerates_a_partially_failed_queue() -> None:
    """Some failures alongside real progress is a completed run, not a blocked one."""

    manager = _FakeQueueManager([[]], status_counts={"DONE": 7, "FAILED": 3})

    summary = _run_file_queue_worker(_queue_worker_settings(), cast(SnowflakeJobManager, manager))

    assert summary["queue_status_counts"] == {"DONE": 7, "FAILED": 3}
    assert "blocked_files" not in summary


def test_distributed_submit_requires_an_owner_before_it_touches_the_fleet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new fleet graph must say whose it is; the refusal comes before any store."""

    monkeypatch.delenv("KG_JOB_OWNER", raising=False)
    monkeypatch.setenv("KG_DISTRIBUTED_DATABASE_URL", "postgresql://unreachable.invalid/fleet")

    def unexpected(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the store must not be built without an owner")

    monkeypatch.setattr("kg_processor.factories.build_distributed_store", unexpected)
    missing = runner.invoke(app, ["distributed", "submit"])
    assert missing.exit_code == 2
    assert "--owner is required" in missing.output

    invalid = runner.invoke(app, ["distributed", "submit", "--owner", "not an address!"])
    assert invalid.exit_code == 2
    assert "sign-in address" in invalid.output


def test_distributed_submit_and_revise_submit_in_the_band_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--priority`` names the run's band; a run that names none is bulk."""

    monkeypatch.setenv("KG_DISTRIBUTED_DATABASE_URL", "postgresql://unreachable.invalid/fleet")
    submitted: list[tuple[bool, PriorityBand]] = []

    class _Store:
        def initialize(self) -> None:
            pass

    class _Summary:
        def model_dump(self, mode: str) -> dict[str, str]:
            return {"mode": mode}

    def submit(
        _planner: object,
        _run_id: str | None = None,
        revision: object | None = None,
        priority_band: PriorityBand = PriorityBand.BULK,
    ) -> _Summary:
        submitted.append((revision is not None, priority_band))
        return _Summary()

    monkeypatch.setattr("kg_processor.factories.build_distributed_store", lambda _: _Store())
    monkeypatch.setattr("kg_processor.factories.build_file_source", lambda _: None)
    monkeypatch.setattr(
        "kg_processor.application.distributed_planner.DistributedRunPlanner.submit", submit
    )
    owner = ["--owner", "owner@example.com"]

    for arguments in (
        ["distributed", "submit", *owner],
        ["distributed", "submit", *owner, "--priority", "interactive"],
        ["distributed", "revise", "--base-run", "run_base", "--priority", "interactive"],
    ):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.output

    assert submitted == [
        (False, PriorityBand.BULK),
        (False, PriorityBand.INTERACTIVE),
        (True, PriorityBand.INTERACTIVE),
    ]
    refused = runner.invoke(app, ["distributed", "submit", *owner, "--priority", "urgent"])
    assert refused.exit_code == 2
