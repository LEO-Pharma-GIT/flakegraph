# SPDX-License-Identifier: Apache-2.0
"""Command-line entry points for local, on-prem, and Snowflake-oriented runs."""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import webbrowser
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Any

import typer
import yaml

from kg_processor import __version__
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import (
    ArtifactKind,
    PriorityBand,
    RevisionRequest,
    TaskStage,
    TaskStatus,
)
from kg_processor.domain.finalization import GraphDatasetManifest
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.domain.jobs import JobFileClaim, JobFileResult

if TYPE_CHECKING:
    from kg_processor.adapters.jobs.snowflake import SnowflakeJobManager
    from kg_processor.application.consumption import ConsumptionCollector
    from kg_processor.application.distributed_worker import WorkerIteration
    from kg_processor.application.graph_dataset import GraphDatasetReader
    from kg_processor.application.lease_heartbeat import LeaseHeartbeat
    from kg_processor.application.pipeline import KgProcessorPipeline
    from kg_processor.application.progress import ProgressSink

APP_DISPLAY_NAME = "FlakeGraph"


class WorkerQueueBlockedError(RuntimeError):
    """Raised when a worker can claim nothing because every file has failed.

    Draining is normally success. When the whole queue is FAILED, however, the
    run did no work, and reporting terminal success would leave an operator
    unable to tell a finished graph from a stuck one.
    """


class WorkerProgressMode(StrEnum):
    """Supported worker progress presentations at the CLI boundary."""

    AUTO = "auto"
    RICH = "rich"
    JSON = "json"
    NONE = "none"


class ReextractFrom(StrEnum):
    """The stage a revision's re-extraction of a kept document starts at."""

    CONTEXT = "context"
    ENTITIES = "entities"
    RELATIONS = "relations"


def _typer_app() -> typer.Typer:
    """Create a command group without Typer's credential-unsafe local dump."""

    # Typer's rich exception renderer can dump frame locals, including credentials
    # stored as ordinary strings by third-party SDKs. FlakeGraph emits its own
    # redacted progress diagnostics, so disable that duplicate unsafe renderer.
    return typer.Typer(
        no_args_is_help=True,
        pretty_exceptions_enable=False,
        pretty_exceptions_show_locals=False,
    )


app = _typer_app()
config_app = _typer_app()
inspect_app = _typer_app()
snowflake_app = _typer_app()
benchmark_app = _typer_app()
distributed_app = _typer_app()
serving_app = _typer_app()
fleet_app = _typer_app()
app.add_typer(config_app, name="config")
app.add_typer(inspect_app, name="inspect")
app.add_typer(snowflake_app, name="snowflake")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(distributed_app, name="distributed")
app.add_typer(serving_app, name="serving")
app.add_typer(fleet_app, name="fleet")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{APP_DISPLAY_NAME} {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            help="Show the installed FlakeGraph version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Register top-level CLI options shared by all subcommands."""

    # The callback owns top-level options that should be available regardless of
    # which subcommand eventually runs. Keeping version handling here also makes
    # container verification independent of any provider configuration.
    _ = version


@config_app.command("print")
def print_config(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Print the resolved redacted configuration as JSON."""

    from kg_processor.application.redaction import redact_sensitive_data

    settings = Settings.load(config)
    _echo_json(redact_sensitive_data(settings.model_dump(mode="json", by_alias=True)))


@config_app.command("providers")
def print_providers(
    kind: Annotated[str | None, typer.Option("--kind")] = None,
) -> None:
    """Print the provider catalog, optionally filtered by provider kind."""

    from kg_processor.config.provider_registry import (
        provider_catalog,
        provider_catalog_for_kind,
        provider_kinds,
    )

    if kind is None:
        _echo_json(provider_catalog())
        return
    if kind not in provider_kinds():
        supported = ", ".join(provider_kinds())
        raise typer.BadParameter(
            f"Unsupported provider kind '{kind}'. Supported kinds: {supported}"
        )
    _echo_json(provider_catalog_for_kind(kind))


@app.command()
def preflight(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    deployment_worker: Annotated[
        bool,
        typer.Option(
            "--deployment-worker",
            help="Validate a deployed worker image while deferring mounted data-path checks.",
        ),
    ] = False,
    orchestrator: Annotated[
        bool,
        typer.Option(
            "--orchestrator",
            help=(
                "Validate a distributed submit host while deferring provider runtime "
                "dependencies to worker images."
            ),
        ),
    ] = False,
) -> None:
    """Validate configuration and runtime prerequisites before a worker run.

    This command is intentionally cheap: it loads the same settings the worker
    will use, but it does not construct OCR, LLM, embedding, cache, or writer
    providers. That lets a user catch missing credentials, unsupported provider
    names, unwritable output paths, and missing local tools before paying for
    OCR/LLM calls or starting a long Snowflake Container Services job.
    """

    from kg_processor.config.preflight import run_preflight

    settings = Settings.load(config)
    result = run_preflight(
        settings,
        deployment_mode=deployment_worker,
        orchestrator_mode=orchestrator,
    )
    # The JSON shape is part of the operator contract: callers can inspect
    # checks/errors directly in CI, local shells, or Snowflake job logs.
    _echo_json(result.model_dump(mode="json"))
    if not result.ok:
        raise typer.Exit(code=1)


@fleet_app.command("profile")
def fleet_profile_command(
    namespace: Annotated[str, typer.Option("--namespace", "-n")] = "flakegraph",
    context: Annotated[str, typer.Option("--context")] = "",
) -> None:
    """Print the processing profile the fleet's workers mount.

    A submit host composes runs against this, because a worker claims only a
    run whose semantic configuration hashes to its own.
    """

    from kg_processor.fleet.kubectl import ClusterTarget
    from kg_processor.fleet.profile import fleet_profile

    _echo_json(fleet_profile(ClusterTarget(context=context), namespace))


@fleet_app.command("preflight")
def fleet_preflight_command(
    config: Annotated[Path, typer.Option("--config", "-c")],
    namespace: Annotated[str, typer.Option("--namespace", "-n")] = "flakegraph",
    context: Annotated[str, typer.Option("--context")] = "",
) -> None:
    """Validate a run against the fleet it is bound for, then the run itself.

    The fleet checks say whether anything will claim the run: worker pools,
    their autoscaling, the profile they mount, the model servers, and the
    credentials the run names. The core preflight then validates the run's own
    settings the way a submit host must. One JSON result carries both.
    """

    from kg_processor.config.preflight import run_preflight
    from kg_processor.fleet.kubectl import ClusterTarget
    from kg_processor.fleet.preflight import fleet_preflight

    settings = Settings.load(config)
    core = run_preflight(settings, orchestrator_mode=True)
    run = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    try:
        fleet = fleet_preflight(run, namespace, ClusterTarget(context=context))
    except Exception as exc:  # noqa: BLE001 - the fleet being unreadable is the finding
        fleet = {"ok": False, "checks": [], "errors": [str(exc)]}
    checks = [*core.checks, *(str(item) for item in fleet["checks"])]
    errors = [*core.errors, *(str(item) for item in fleet["errors"])]
    _echo_json({"ok": not errors, "checks": checks, "errors": errors})
    if errors:
        raise typer.Exit(code=1)


@fleet_app.command("recover")
def fleet_recover_command(
    run_id: Annotated[str, typer.Option("--run-id")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    namespace: Annotated[str, typer.Option("--namespace", "-n")] = "flakegraph",
    context: Annotated[str, typer.Option("--context")] = "",
) -> None:
    """Restart the worker pools a queued run is waiting on, if they are down.

    Task state is never touched: queued work is durable and resumes when a
    pool comes back. A healthy pool is left alone, since it may be busy with
    another graph, and a pool that cannot start for a reason a restart will
    not fix is reported instead.
    """

    from kg_processor.factories import build_distributed_store
    from kg_processor.fleet.kubectl import ClusterTarget
    from kg_processor.fleet.recover import queued_worker_components, recover_workers

    settings = Settings.load(config)
    store = build_distributed_store(settings)
    store.initialize()
    summary = store.get_run_summary(run_id)
    counts = [item.model_dump(mode="json") for item in summary.task_counts]
    message = recover_workers(
        namespace, queued_worker_components(counts), ClusterTarget(context=context)
    )
    _echo_json({"run_id": run_id, "message": message})


sources_app = typer.Typer(help="Look at a run's document source before submitting it.")
app.add_typer(sources_app, name="sources")


@sources_app.command("list")
def sources_list(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=10_000)] = 1000,
) -> None:
    """Name the documents the configured source would offer, without fetching them.

    Prints a JSON array of ``{uri, name, size_bytes, modified_at, checksum}``.
    A source that can only be discovered by fetching is refused rather than
    quietly downloaded.
    """

    from kg_processor.factories import build_file_source
    from kg_processor.ports.file_source import BrowsableFileSource

    settings = Settings.load(config)
    source = build_file_source(settings)
    if not isinstance(source, BrowsableFileSource):
        typer.echo(f"{settings.files.source} sources cannot be listed without fetching", err=True)
        raise typer.Exit(code=2)
    try:
        listing = [item.model_dump(mode="json") for item in source.browse(limit)]
    except Exception as exc:  # noqa: BLE001 - the backend's refusal is the answer here
        # A missing bucket, a wrong endpoint or a rejected credential should
        # read as one line to whoever asked, not as the SDK's stack.
        typer.echo(f"{settings.files.source} source could not be listed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _echo_json(listing)


@distributed_app.command("init")
def distributed_init(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    serve_stages: Annotated[
        list[TaskStage] | None,
        typer.Option(
            "--serve-stage",
            help="A stage the fleet deployed with this configuration serves; repeatable.",
        ),
    ] = None,
) -> None:
    """Create the durable PostgreSQL coordination schema idempotently.

    With ``--serve-stage``, also declare this configuration as what the fleet
    serves for those stages. A deployment runs this at every upgrade because a
    pool scaled to zero has no worker left to declare its new digest, and the
    autoscaler asks for workers only for work the declared fleet can take: with
    a stale declaration nothing would ever scale up to correct it.
    """

    from kg_processor.application.distributed_planner import distributed_processing_config_digest
    from kg_processor.factories import build_distributed_store

    settings = Settings.load(config)
    store = build_distributed_store(settings)
    store.initialize()
    stages = set(serve_stages or [])
    digest = distributed_processing_config_digest(settings)
    if stages:
        store.record_served_configuration(stages, digest)
    _echo_json(
        {
            "initialized": True,
            "config_digest": digest,
            "served_stages": sorted(stage.value for stage in stages),
        }
    )


_OWNER_HELP = (
    "The sign-in address of the person the graph belongs to. Required for a new "
    "graph unless the configuration names job.owner or KG_JOB_OWNER is set."
)
_PRIORITY_HELP = (
    "interactive for a run someone is waiting on: its work is claimed ahead of "
    "every bulk run's. Runs within a band share the workers equally."
)


def _settings_with_owner(config: Path | None, owner: str | None) -> Settings:
    """Load settings, with ``--owner`` taking precedence over the configured owner."""

    try:
        return Settings.load(config, overrides={"job": {"owner": owner}} if owner else None)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None


@distributed_app.command("submit")
def distributed_submit(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    owner: Annotated[str | None, typer.Option("--owner", help=_OWNER_HELP)] = None,
    priority: Annotated[
        PriorityBand, typer.Option("--priority", help=_PRIORITY_HELP)
    ] = PriorityBand.BULK,
) -> None:
    """Discover sources and submit the initial dynamically expanded graph run."""

    from kg_processor.application.distributed_planner import DistributedRunPlanner
    from kg_processor.factories import build_distributed_store, build_file_source

    settings = _settings_with_owner(config, owner)
    if not settings.job.owner:
        typer.echo(f"--owner is required: {_OWNER_HELP}", err=True)
        raise typer.Exit(code=2)
    store = build_distributed_store(settings)
    store.initialize()
    snapshot = DistributedRunPlanner(
        settings,
        build_file_source(settings),
        store,
        store,
    ).submit(run_id, priority_band=priority)
    _echo_json(snapshot.model_dump(mode="json"))


@distributed_app.command("revise")
def distributed_revise(
    base_run: Annotated[str, typer.Option("--base-run", help="A succeeded run of the graph.")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    drop_file: Annotated[
        list[str] | None,
        typer.Option("--drop-file", help="A document id to leave out; repeatable."),
    ] = None,
    add_documents: Annotated[
        bool,
        typer.Option(
            "--add-documents/--no-add-documents",
            help="Discover the configured source and add what the graph lacks.",
        ),
    ] = True,
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild",
            help=(
                "Rebuild the graph even though no document changes - after a "
                "finalization fix. Nothing is extracted again."
            ),
        ),
    ] = False,
    owner: Annotated[
        str | None,
        typer.Option(
            "--owner", help="The graph's owner; defaults to the one its base run recorded."
        ),
    ] = None,
    reextract: Annotated[
        list[str] | None,
        typer.Option(
            "--reextract",
            help=(
                "Kept documents to extract again with the current code and ontology: "
                "a file id, file:<id>, folder:<prefix>, kind:<document|dataset|image>, "
                "or all; repeatable. Their text is never read again."
            ),
        ),
    ] = None,
    reextract_from: Annotated[
        ReextractFrom,
        typer.Option(
            "--from",
            help=(
                "The stage re-extraction starts at: context extracts everything again; "
                "entities keeps the document context; relations keeps the entities too."
            ),
        ),
    ] = ReextractFrom.CONTEXT,
    revalidate: Annotated[
        bool,
        typer.Option(
            "--revalidate",
            help=(
                "Put every other kept document's stored records through the current "
                "validation and verifier, without extracting anything again. Needs "
                "the LLM entity and relation extractors."
            ),
        ),
    ] = False,
    priority: Annotated[
        PriorityBand, typer.Option("--priority", help=_PRIORITY_HELP)
    ] = PriorityBand.BULK,
) -> None:
    """Submit a new version of a graph: the base run's documents, minus drops, plus new ones.

    Kept documents are read from the base run's stage outputs, so the workers
    extract only what is added; the finalizer rebuilds the graph from both and
    the graph's head moves to the new version once it succeeds. ``--reextract``
    and ``--revalidate`` apply newer extraction or validation to documents the
    graph keeps; the documents table says which were reused, revalidated, or
    extracted again, and from which stage.
    """

    from kg_processor.application.distributed_planner import DistributedRunPlanner
    from kg_processor.factories import build_distributed_store, build_file_source

    settings = _settings_with_owner(config, owner)
    store = build_distributed_store(settings)
    store.initialize()
    try:
        snapshot = DistributedRunPlanner(
            settings,
            build_file_source(settings) if add_documents else None,
            store,
            store,
        ).submit(
            run_id,
            RevisionRequest(
                base_run_id=base_run,
                drop_file_ids=list(drop_file or []),
                rebuild=rebuild,
                reextract=list(reextract or []),
                reextract_from=reextract_from.value,
                revalidate=revalidate,
            ),
            priority_band=priority,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    _echo_json(snapshot.model_dump(mode="json"))


@distributed_app.command("worker")
def distributed_worker(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    worker_id: Annotated[str | None, typer.Option("--worker-id")] = None,
    stages: Annotated[list[TaskStage] | None, typer.Option("--stage")] = None,
    once: Annotated[bool, typer.Option("--once")] = False,
) -> None:
    """Claim eligible stages until signalled, or process one poll with ``--once``."""

    from kg_processor.application.distributed_diagnostics import worker_ready_payload
    from kg_processor.application.distributed_worker import DistributedWorker
    from kg_processor.application.progress import JsonLineProgressSink
    from kg_processor.factories import (
        build_distributed_pipeline,
        build_distributed_store,
        build_graph_manifest_publisher,
    )

    settings = Settings.load(config)
    store = build_distributed_store(settings)
    store.initialize()
    effective_worker_id = (
        worker_id or settings.distributed.worker_id or (f"{socket.gethostname()}-{os.getpid()}")
    )
    effective_stages = set(stages or settings.distributed.worker_stages)
    pipeline = build_distributed_pipeline(
        settings,
        effective_stages,
        progress_sink=JsonLineProgressSink(),
    )
    service = DistributedWorker(
        settings,
        effective_worker_id,
        effective_stages,
        pipeline,
        store,
        store,
        build_graph_manifest_publisher(settings, store.blob_store),
        # A run's own ontology executes over this worker's providers.
        pipeline_for=pipeline.for_settings,
    )
    try:
        if once:
            iteration = service.process_one()
            _echo_json(_worker_iteration_payload(iteration))
            if iteration.succeeded is False:
                raise typer.Exit(code=1)
            return

        _echo_json(worker_ready_payload(settings, effective_worker_id, effective_stages))
        stopping = False

        def request_stop(_signum: int, _frame: object) -> None:
            """Stop claiming, and stop once every task in flight has finished."""

            nonlocal stopping
            stopping = True

        previous_sigint = signal.signal(signal.SIGINT, request_stop)
        previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
        try:
            service.run_until_stopped(
                lambda: stopping,
                on_iteration=lambda iteration: (
                    _echo_json(_worker_iteration_payload(iteration)) if iteration.claimed else None
                ),
            )
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
    finally:
        pipeline.close()


@distributed_app.command("status")
def distributed_status(
    run_id: Annotated[str, typer.Option("--run-id")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    include_tasks: Annotated[
        bool,
        typer.Option(
            "--include-tasks",
            help="Include every task payload for diagnosis; output may be very large.",
        ),
    ] = False,
) -> None:
    """Print bounded run progress, with detailed task state only when requested."""

    from kg_processor.application.distributed_diagnostics import run_status_payload
    from kg_processor.factories import build_distributed_store

    settings = Settings.load(config)
    store = build_distributed_store(settings)
    store.initialize()
    payload = run_status_payload(store.get_run_summary(run_id), settings)
    # Diagnostics come from the bounded aggregates either way: the full task
    # list is for reading attempts and leases, not for changing the verdict.
    if include_tasks:
        payload["tasks"] = [task.model_dump(mode="json") for task in store.get_run(run_id).tasks]
    _echo_json(payload)


@distributed_app.command("list")
def distributed_list(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 100,
) -> None:
    """List recent distributed runs without loading task payloads or run configuration."""

    from kg_processor.factories import build_distributed_store

    settings = Settings.load(config)
    store = build_distributed_store(settings)
    store.initialize()
    _echo_json({"runs": [overview.model_dump(mode="json") for overview in store.list_runs(limit)]})


@distributed_app.command("cancel")
def distributed_cancel(
    run_id: Annotated[str, typer.Option("--run-id")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Cancel queued/running tasks without deleting completed audit artifacts."""

    from kg_processor.factories import build_distributed_store

    settings = Settings.load(config)
    store = build_distributed_store(settings)
    store.initialize()
    store.cancel_run(run_id)
    _echo_json(store.get_run_summary(run_id).model_dump(mode="json"))


@distributed_app.command("delete")
def distributed_delete(
    graph_id: Annotated[str, typer.Option("--graph-id")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Delete a graph for good: all its runs, their rows and every stored object.

    Refused while any run of the graph is still queued or running. Deleting a
    graph that is already gone removes nothing and reports zeros.
    """

    from kg_processor.factories import build_distributed_store

    settings = Settings.load(config)
    store = build_distributed_store(settings)
    store.initialize()
    try:
        deletion = store.delete_graph(graph_id)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from None
    _echo_json(deletion.model_dump(mode="json"))


@distributed_app.command("retry")
def distributed_retry(
    run_id: Annotated[str, typer.Option("--run-id")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    skip_file_id: Annotated[
        list[str] | None,
        typer.Option(
            "--skip-file-id",
            help="Quarantine a failed document; repeat for multiple files.",
        ),
    ] = None,
) -> None:
    """Retry stopped work, excluding explicitly quarantined failed documents."""

    from kg_processor.factories import build_distributed_store

    settings = Settings.load(config)
    store = build_distributed_store(settings)
    store.initialize()
    store.retry_run(run_id, skipped_file_ids=skip_file_id)
    _echo_json(store.get_run_summary(run_id).model_dump(mode="json"))


def _final_graph_batch(settings: Settings, run_id: str) -> GraphWriteBatch:
    """The graph a finished run published, read back from its final artifact.

    A completed fleet run leaves exactly one graph-result artifact behind its
    finalize task; on a shared object store it is a manifest of parquet files,
    otherwise the batch itself. Either way it is the same ``GraphWriteBatch``
    every writer accepts, so publishing it elsewhere is a write, not a run.
    """

    result = _final_graph_result(settings, run_id)
    if isinstance(result, GraphWriteBatch):
        return result
    manifest, reader = result
    return reader.read(manifest)


def _final_graph_result(
    settings: Settings,
    run_id: str,
) -> GraphWriteBatch | tuple[GraphDatasetManifest, GraphDatasetReader]:
    """A finished run's final artifact: its batch, or its dataset manifest and a reader."""

    from kg_processor.application.graph_dataset import GraphDatasetReader
    from kg_processor.factories import build_blob_store, build_distributed_store

    store = build_distributed_store(settings)
    store.initialize()
    snapshot = store.get_run(run_id)
    final_tasks = [
        task
        for task in snapshot.tasks
        if task.task.stage == TaskStage.FINALIZE_GRAPH and task.status == TaskStatus.SUCCEEDED
    ]
    if len(final_tasks) != 1 or len(final_tasks[0].output_artifact_ids) != 1:
        raise ValueError("run does not have exactly one completed final graph artifact")
    artifact = store.get(final_tasks[0].output_artifact_ids[0])
    if artifact.ref.kind != ArtifactKind.GRAPH_RESULT:
        raise ValueError("final task output is not a graph result artifact")
    if artifact.ref.media_type == "application/vnd.flakegraph.graph-manifest+json":
        blob_store = build_blob_store(settings)
        if blob_store is None:
            raise ValueError("graph dataset export requires distributed.artifact_uri")
        manifest = GraphDatasetManifest.model_validate_json(artifact.payload)
        return manifest, GraphDatasetReader(blob_store)
    return GraphWriteBatch.model_validate_json(artifact.payload)


@distributed_app.command("export")
def distributed_export(
    run_id: Annotated[str, typer.Option("--run-id")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Materialize a completed fleet graph as standard local inspection artifacts."""

    from kg_processor.factories import build_local_artifacts_writer

    settings = Settings.load(config)
    result = _final_graph_result(settings, run_id)
    writer = build_local_artifacts_writer(output)
    if isinstance(result, GraphWriteBatch):
        writer.write(result)
        graph_id, nodes, edges = result.graph_id, len(result.nodes), len(result.edges)
    else:
        # A published dataset is written table by table: a large fleet graph
        # does not fit in memory as one batch of Python objects.
        manifest, reader = result
        writer.write_dataset(manifest, reader)
        graph_id = manifest.graph_id
        nodes, edges = manifest.table("nodes").row_count, manifest.table("edges").row_count
    _echo_json(
        {
            "run_id": run_id,
            "graph_id": graph_id,
            "output": str(output),
            "nodes": nodes,
            "edges": edges,
        }
    )


@distributed_app.command("publish")
def distributed_publish(
    run_id: Annotated[str, typer.Option("--run-id")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Write a completed fleet graph through the configured writer, without running it again.

    The run's graph already exists; a destination such as Snowflake needs the
    same rows, not a second extraction. The writer named in ``writer.provider``
    (``snowflake_bulk``, ``snowflake_direct`` or ``local_artifacts``) receives
    the run's final batch as the finalizer would have handed it over.
    """

    from kg_processor.factories import build_writer

    settings = Settings.load(config)
    batch = _final_graph_batch(settings, run_id)
    build_writer(settings).write(batch)
    _echo_json(
        {
            "run_id": run_id,
            "graph_id": batch.graph_id,
            "writer": settings.writer.provider,
            "nodes": len(batch.nodes),
            "edges": len(batch.edges),
        }
    )


@app.command()
def worker(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    progress: Annotated[
        WorkerProgressMode,
        typer.Option(
            "--progress",
            help=(
                "Progress output: auto selects the live terminal UI on a TTY and JSON logs "
                "when redirected."
            ),
        ),
    ] = WorkerProgressMode.AUTO,
) -> None:
    """Run the FlakeGraph pipeline locally or as a leased Snowflake worker.

    The worker is the command that actually spends money and CPU/GPU time. It
    reads the configured file source, performs OCR/text extraction, chunks the
    normalized text, asks the LLM for graph facts, validates evidence, merges
    and embeds the graph, creates communities, and writes the final artifacts.
    Local and Snowflake runs share this command so provider differences remain
    isolated behind ports and factories rather than branching through the
    application pipeline.
    """

    from kg_processor.application.progress import error_metadata
    from kg_processor.application.redaction import redact_sensitive_data, redact_sensitive_text
    from kg_processor.factories import build_job_manager, build_pipeline

    settings = Settings.load(config)
    progress_sink = _build_worker_progress_sink(settings, progress)
    job_manager = build_job_manager(settings)
    if job_manager is not None and settings.job.use_file_queue:
        # File-queue mode is the horizontally scalable Snowflake path: each
        # worker repeatedly claims a small batch, processes only those files,
        # and exits cleanly once the queue is drained.
        _start_worker_progress(progress_sink)
        try:
            summary = _run_file_queue_worker(settings, job_manager, progress_sink)
        except Exception as exc:
            _fail_worker_progress(progress_sink, exc)
            raise RuntimeError(redact_sensitive_text(str(exc))) from None
        _finish_worker_progress(progress_sink, summary.get("last_run_report") or summary)
        return
    elif job_manager is not None:
        if not settings.job.lease_owner:
            raise ValueError("job.use_lease requires job.lease_owner")
        # Single-job lease mode prevents two container instances from writing
        # the same graph job at once. We only build expensive providers after a
        # lease is actually claimed.
        claim = job_manager.claim_job(
            settings.job.job_id,
            settings.job.graph_id,
            settings.job.lease_owner,
            settings.job.lease_seconds,
            redact_sensitive_data(settings.model_dump(mode="json", by_alias=True)),
        )
        if not claim.claimed:
            _echo_json(claim.model_dump(mode="json"))
            return
    _start_worker_progress(progress_sink)
    pipeline = build_pipeline(settings, progress_sink=progress_sink)
    try:
        if job_manager is not None and settings.job.lease_owner:
            # Snowflake leases have a finite timeout, so keep heartbeating while
            # OCR, LLM, and embedding work is in progress.
            with _job_heartbeat(settings, job_manager) as heartbeat:
                batch = pipeline.run()
            heartbeat.raise_if_unhealthy()
        else:
            batch = pipeline.run()
    except Exception as exc:
        _fail_worker_progress(progress_sink, exc)
        if job_manager is not None and settings.job.lease_owner:
            job_manager.fail_job(
                settings.job.job_id,
                settings.job.lease_owner,
                error_metadata(exc),
            )
        raise RuntimeError(redact_sensitive_text(str(exc))) from None
    finally:
        pipeline.close()
    if job_manager is not None and settings.job.lease_owner:
        job_manager.complete_job(settings.job.job_id, settings.job.lease_owner, batch.run_report)
    _finish_worker_progress(progress_sink, batch.run_report)


def _run_file_queue_worker(
    settings: Settings,
    job_manager: SnowflakeJobManager,
    progress_sink: ProgressSink | None = None,
) -> dict[str, Any]:
    """Claim and process Snowflake job-file batches until no work remains."""

    from kg_processor.application.consumption import ConsumptionCollector
    from kg_processor.application.progress import error_metadata
    from kg_processor.application.redaction import redact_sensitive_text
    from kg_processor.factories import build_pipeline

    if not settings.job.lease_owner:
        raise ValueError("job.use_file_queue requires job.lease_owner")
    batches_processed = 0
    files_processed = 0
    last_run_report: dict[str, Any] | None = None
    while True:
        claimed_files = job_manager.claim_job_files(
            settings.job.job_id,
            settings.job.graph_id,
            settings.job.lease_owner,
            settings.job.lease_seconds,
            settings.job.file_batch_size,
        )
        if not claimed_files:
            summary = {
                "claimed": batches_processed > 0,
                "drained": True,
                "job_id": settings.job.job_id,
                "graph_id": settings.job.graph_id,
                "batches_processed": batches_processed,
                "files_processed": files_processed,
                "last_run_report": last_run_report,
            }
            # A worker may start after all files have already reached DONE/FAILED,
            # or may drain the final claim concurrently with another worker. Always
            # ask Snowflake to terminally summarize the queue before returning.
            if batches_processed == 0:
                queue_counts = job_manager.count_job_files_by_status(
                    settings.job.job_id,
                    settings.job.graph_id,
                )
                summary["queue_status_counts"] = queue_counts
                blocked = queue_counts.get("FAILED", 0)
                if blocked and not queue_counts.get("DONE", 0):
                    # Draining without claiming anything looks identical to a
                    # successful run once the service reports terminal state. A
                    # queue that is entirely failed is a blocked run, not a
                    # finished one, so it must not be reported as success.
                    summary["blocked_files"] = blocked
                    _echo_json(summary)
                    raise WorkerQueueBlockedError(
                        f"No claimable files remain and {blocked} are FAILED for graph "
                        f"{settings.job.graph_id}. Nothing was processed. Run "
                        "'flakegraph snowflake retry' to requeue them."
                    )
                job_manager.complete_job_if_file_queue_drained(
                    settings.job.job_id,
                    settings.job.graph_id,
                    summary,
                )
            return summary
        # A batch-specific pipeline keeps the shared application logic unaware
        # of Snowflake queue mechanics. The file source sees claimed file ids,
        # while the rest of the pipeline still receives normal documents.
        pipeline: KgProcessorPipeline | None = None
        try:
            complete_graph_claim = job_manager.worker_owns_entire_file_queue(
                settings.job.job_id,
                settings.job.graph_id,
                settings.job.lease_owner,
                [claim.file_id for claim in claimed_files],
            )
            # One collector, shared: the pipeline records into it and the
            # progress sink reads from it, so what the app shows while a run is
            # going is the same tally it will publish when the run ends.
            collector = ConsumptionCollector(settings.job.graph_id, job_id=settings.job.job_id)
            pipeline = build_pipeline(
                settings,
                claimed_files,
                _file_queue_progress_sink(
                    settings,
                    job_manager,
                    claimed_files,
                    progress_sink,
                    collector,
                ),
                write_scope_override=("graph_snapshot" if complete_graph_claim else None),
                consumption=collector,
            )
            with _job_file_heartbeat(settings, job_manager, claimed_files) as heartbeat:
                batch = pipeline.run()
            heartbeat.raise_if_unhealthy()
            completed_results = pipeline.job_file_results(batch)
            failed_results = pipeline.failed_job_file_results()
        except Exception as exc:
            # Persisting the structured failure beside the claimed files makes
            # retry decisions possible without scraping container logs.
            failure = error_metadata(exc)
            job_manager.fail_job_files(
                settings.job.job_id,
                settings.job.graph_id,
                settings.job.lease_owner,
                [claim.file_id for claim in claimed_files],
                failure,
            )
            # The exception exits this worker before the ordinary empty-claim
            # branch can summarize the parent job. This conditional update only
            # becomes terminal when no other worker owns queued or claimed rows.
            job_manager.complete_job_if_file_queue_drained(
                settings.job.job_id,
                settings.job.graph_id,
                {
                    "run_id": f"{settings.job.job_id}:failed",
                    "failed_file_ids": [claim.file_id for claim in claimed_files],
                    "error": failure,
                },
            )
            raise RuntimeError(redact_sensitive_text(str(exc))) from None
        finally:
            if pipeline is not None:
                pipeline.close()
        if completed_results:
            job_manager.complete_job_files(
                settings.job.job_id,
                settings.job.graph_id,
                settings.job.lease_owner,
                completed_results,
            )
        _fail_job_file_results(settings, job_manager, failed_results)
        # Finalize in the same success path that terminally updates the claimed
        # rows. A worker killed before its next empty poll must not leave a fully
        # drained parent job permanently pending.
        job_manager.complete_job_if_file_queue_drained(
            settings.job.job_id,
            settings.job.graph_id,
            batch.run_report,
        )
        batches_processed += 1
        files_processed += int(batch.run_report.get("files_processed", 0))
        last_run_report = batch.run_report


def _fail_job_file_results(
    settings: Settings,
    job_manager: SnowflakeJobManager,
    failed_results: list[JobFileResult],
) -> None:
    """Persist per-file failures collected by the pipeline queue adapter."""

    from kg_processor.application.redaction import redact_sensitive_data, redact_sensitive_text

    if not failed_results:
        return
    if not settings.job.lease_owner:
        raise ValueError("job file failures require job.lease_owner")
    for result in failed_results:
        raw_error = result.audit.get("error")
        error = redact_sensitive_data(
            raw_error
            if isinstance(raw_error, dict)
            else {"message": redact_sensitive_text(str(raw_error))}
        )
        job_manager.fail_job_files(
            settings.job.job_id,
            settings.job.graph_id,
            settings.job.lease_owner,
            [result.file_id],
            error,
        )


def _job_heartbeat(
    settings: Settings,
    job_manager: SnowflakeJobManager,
) -> LeaseHeartbeat:
    from kg_processor.application.lease_heartbeat import LeaseHeartbeat, heartbeat_interval_seconds

    lease_owner = settings.job.lease_owner
    if not lease_owner:
        raise ValueError("job heartbeat requires job.lease_owner")
    return LeaseHeartbeat(
        lambda: job_manager.heartbeat_job(
            settings.job.job_id,
            lease_owner,
            settings.job.lease_seconds,
        ),
        heartbeat_interval_seconds(settings.job.lease_seconds),
    )


def _job_file_heartbeat(
    settings: Settings,
    job_manager: SnowflakeJobManager,
    claimed_files: list[JobFileClaim],
) -> LeaseHeartbeat:
    from kg_processor.application.lease_heartbeat import LeaseHeartbeat, heartbeat_interval_seconds

    lease_owner = settings.job.lease_owner
    if not lease_owner:
        raise ValueError("file heartbeat requires job.lease_owner")
    file_ids = [claim.file_id for claim in claimed_files]
    return LeaseHeartbeat(
        lambda: job_manager.heartbeat_job_files(
            settings.job.job_id,
            settings.job.graph_id,
            lease_owner,
            file_ids,
            settings.job.lease_seconds,
        ),
        heartbeat_interval_seconds(settings.job.lease_seconds),
    )


def _file_queue_progress_sink(
    settings: Settings,
    job_manager: SnowflakeJobManager,
    claimed_files: list[JobFileClaim],
    display_sink: ProgressSink | None = None,
    consumption: ConsumptionCollector | None = None,
) -> ProgressSink:
    from kg_processor.adapters.jobs.snowflake import SnowflakeJobFileProgressSink
    from kg_processor.application.progress import CompositeProgressSink, JsonLineProgressSink

    lease_owner = settings.job.lease_owner
    if not lease_owner:
        raise ValueError("file progress persistence requires job.lease_owner")
    return CompositeProgressSink(
        [
            display_sink or JsonLineProgressSink(),
            SnowflakeJobFileProgressSink(
                job_manager,
                settings.job.job_id,
                settings.job.graph_id,
                lease_owner,
                [claim.file_id for claim in claimed_files],
                (
                    (lambda: consumption.running_totals(settings.consumption.rate_card()))
                    if consumption is not None
                    else None
                ),
            ),
        ]
    )


def _build_worker_progress_sink(
    settings: Settings,
    requested_mode: WorkerProgressMode,
) -> ProgressSink:
    """Select an interactive or machine-readable sink for this process."""

    from kg_processor.adapters.progress import RichTerminalProgressSink, WorkerProgressContext
    from kg_processor.application.progress import JsonLineProgressSink, NullProgressSink

    mode = requested_mode
    if mode == WorkerProgressMode.AUTO:
        mode = WorkerProgressMode.RICH if sys.stdout.isatty() else WorkerProgressMode.JSON
    if mode == WorkerProgressMode.JSON:
        return JsonLineProgressSink()
    if mode == WorkerProgressMode.NONE:
        return NullProgressSink()
    return RichTerminalProgressSink(
        WorkerProgressContext(
            job_id=settings.job.job_id,
            graph_id=settings.job.graph_id,
            ocr_provider=settings.ocr.provider,
            llm_model=settings.llm.model,
            embedding_model=settings.embedding.model,
            writer_provider=settings.writer.provider,
            output_path=settings.writer.output_path,
            extraction_parallelism=settings.graph.extraction_parallelism,
            community_parallelism=settings.graph.finalization_provider_concurrency,
        )
    )


def _start_worker_progress(progress_sink: ProgressSink) -> None:
    """Start live rendering before expensive provider construction."""

    from kg_processor.adapters.progress import RichTerminalProgressSink

    if isinstance(progress_sink, RichTerminalProgressSink):
        progress_sink.start()


def _finish_worker_progress(
    progress_sink: ProgressSink,
    report: dict[str, Any],
) -> None:
    """Print a terminal summary or preserve the existing JSON report contract."""

    from kg_processor.adapters.progress import RichTerminalProgressSink

    if isinstance(progress_sink, RichTerminalProgressSink):
        progress_sink.finish(report)
        return
    _echo_json(report)


def _fail_worker_progress(progress_sink: ProgressSink, exc: Exception) -> None:
    """Close a live terminal safely when provider or pipeline work fails."""

    from kg_processor.adapters.progress import RichTerminalProgressSink

    if isinstance(progress_sink, RichTerminalProgressSink):
        progress_sink.fail(exc)


@inspect_app.command("graph")
def inspect_graph(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("out/kg"),
) -> None:
    """Inspect a local artifact directory and print a review report.

    This is the post-worker sanity check for local and CI runs. It reads the
    Parquet/JSON artifacts written by the local writer, verifies expected table
    schemas, evaluates graph-quality metrics, ranks the most important entities
    and relations, and reports trace/run metadata without re-running OCR or LLM
    extraction.
    """

    from kg_processor.application.inspect import inspect_local_graph

    _echo_json(inspect_local_graph(output))


@inspect_app.command("html")
def inspect_html(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("out/kg"),
    html_path: Annotated[Path | None, typer.Option("--html", help="Generated HTML path.")] = None,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open the generated explorer in a browser."),
    ] = False,
) -> None:
    """Generate a self-contained interactive HTML explorer from local artifacts.

    The export embeds graph data and Plotly.js in one offline file. It includes
    entity, relation, evidence, source, community, quality, and run-metric views
    without invoking OCR, LLM, embedding, or Snowflake providers again.
    """

    from kg_processor.adapters.explorer import StaticHtmlGraphExplorer
    from kg_processor.application.graph_explorer import build_graph_explorer_dataset

    destination = html_path or output / "flakegraph-explorer.html"
    dataset = build_graph_explorer_dataset(output)
    result = StaticHtmlGraphExplorer().write(dataset, destination)
    if open_browser:
        webbrowser.open(destination.resolve().as_uri())
    _echo_json(result.as_dict())


@inspect_app.command("compare")
def inspect_compare(
    left: Annotated[Path, typer.Option("--left")] = Path("out/kg-left"),
    right: Annotated[Path, typer.Option("--right")] = Path("out/kg-right"),
) -> None:
    """Compare two local artifact directories for deterministic parity."""

    from kg_processor.application.inspect import compare_local_graph_artifacts

    result = compare_local_graph_artifacts(left, right)
    _echo_json(result)
    if not result["ok"]:
        raise typer.Exit(code=1)


@inspect_app.command("evaluate")
def inspect_evaluate(
    gold: Annotated[
        Path,
        typer.Option("--gold", help="Curated public gold-graph JSON fixture."),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("out/kg"),
    fail: Annotated[
        bool,
        typer.Option("--fail/--no-fail", help="Exit non-zero when thresholds are missed."),
    ] = True,
) -> None:
    """Measure extracted entity, triple, evidence, and structural quality against gold.

    The command prints machine-readable results and optionally exits non-zero when
    the fixture's explicit acceptance thresholds are missed.
    """

    from kg_processor.application.graph_evaluation import evaluate_graph_artifacts

    result = evaluate_graph_artifacts(output, gold)
    _echo_json(result)
    if fail and not result["ok"]:
        raise typer.Exit(code=1)


@benchmark_app.command("extraction")
def benchmark_extraction(
    config: Annotated[Path, typer.Option("--config", "-c")],
    gold: Annotated[
        Path,
        typer.Option("--gold", help="Curated public gold-graph JSON fixture."),
    ],
    repeats: Annotated[int, typer.Option("--repeats", min=1, max=20)] = 3,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("out/benchmarks"),
) -> None:
    """Run uncached extraction repeatedly and report quality, timing, and stability.

    Each repetition writes an isolated snapshot, evaluates it against gold data, and
    contributes normalized graph signatures to pairwise stability metrics.
    """

    from kg_processor.application.benchmarking import build_benchmark_report
    from kg_processor.application.graph_evaluation import evaluate_graph_artifacts, graph_signatures
    from kg_processor.application.progress import NullProgressSink
    from kg_processor.factories import build_pipeline

    output.mkdir(parents=True, exist_ok=True)
    run_reports: list[dict[str, Any]] = []
    signatures: list[tuple[set[str], set[str]]] = []
    for run_index in range(1, repeats + 1):
        settings = Settings.load(config)
        run_output = output / f"run-{run_index:02d}"
        settings.writer.provider = "local_artifacts"
        settings.writer.output_path = run_output
        settings.cache.provider = "none"
        settings.job.use_lease = False
        settings.job.use_file_queue = False
        settings.job.job_id = f"{settings.job.job_id}-benchmark-{run_index}"
        started = perf_counter()
        pipeline = build_pipeline(settings, progress_sink=NullProgressSink())
        try:
            batch = pipeline.run()
        finally:
            pipeline.close()
        elapsed_seconds = perf_counter() - started
        evaluation = evaluate_graph_artifacts(run_output, gold)
        run_reports.append(
            {
                "run": run_index,
                "output": str(run_output),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "run_report": batch.run_report,
                "evaluation": evaluation,
            }
        )
        signatures.append(graph_signatures(run_output))
    summary = build_benchmark_report(
        config_path=config,
        gold_path=gold,
        repeats=repeats,
        settings=settings,
        runs=run_reports,
        signatures=signatures,
    )
    report_path = output / "benchmark.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _echo_json({**summary, "report_path": str(report_path)})


@snowflake_app.command("ddl")
def snowflake_ddl(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    embedding_dim: Annotated[int | None, typer.Option("--embedding-dim", min=1)] = None,
) -> None:
    """Render canonical Snowflake table DDL."""

    from kg_processor.application.snowflake_schema import render_snowflake_schema_sql

    settings = Settings.load(config)
    dim = embedding_dim if embedding_dim is not None else settings.embedding.dimension
    typer.echo(render_snowflake_schema_sql(dim).strip())


@snowflake_app.command("service-spec")
def snowflake_service_spec(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    config_path: Annotated[str, typer.Option("--config-path")] = "configs/snowflake-cortex.yaml",
) -> None:
    """Render a Snowpark Container Services job spec."""

    from kg_processor.application.snowflake_deployment import render_spcs_service_spec_yaml

    settings = Settings.load(config)
    typer.echo(render_spcs_service_spec_yaml(settings, config_path).strip())


@snowflake_app.command("image-reference")
def snowflake_image_reference(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Render the fully qualified Snowflake image repository reference."""

    from kg_processor.application.snowflake_deployment import render_snowflake_image_reference

    settings = Settings.load(config)
    typer.echo(render_snowflake_image_reference(settings))


@snowflake_app.command("execute-job-sql")
def snowflake_execute_job_sql(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    spec_file: Annotated[str, typer.Option("--spec-file")] = "flakegraph-job.yaml",
    async_execution: Annotated[bool, typer.Option("--async")] = False,
) -> None:
    """Render SQL for executing the SPCS job service."""

    from kg_processor.application.snowflake_deployment import render_execute_job_service_sql

    settings = Settings.load(config)
    typer.echo(render_execute_job_service_sql(settings, spec_file, async_execution).strip())


@snowflake_app.command("kubernetes-job")
def snowflake_kubernetes_job(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    image: Annotated[str, typer.Option("--image")] = "flakegraph:latest",
    config_path: Annotated[
        str,
        typer.Option("--config-path"),
    ] = "configs/onprem-azure-blob-vllm-mineru-oss.yaml",
    secret_name: Annotated[str | None, typer.Option("--secret-name")] = None,
    parallelism: Annotated[int, typer.Option("--parallelism")] = 1,
    completions: Annotated[int | None, typer.Option("--completions")] = None,
    gpu_count: Annotated[int | None, typer.Option("--gpu-count")] = None,
) -> None:
    """Render an on-prem Kubernetes Job manifest using the same settings model."""

    from kg_processor.application.snowflake_deployment import render_kubernetes_job_yaml

    settings = Settings.load(config)
    typer.echo(
        render_kubernetes_job_yaml(
            settings,
            image=image,
            config_path=config_path,
            secret_name=secret_name,
            parallelism=parallelism,
            completions=completions,
            gpu_count=gpu_count,
        ).strip()
    )


@snowflake_app.command("setup-sql")
def snowflake_setup_sql(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    role_name: Annotated[str, typer.Option("--role-name")] = "KG_PROCESSOR_ROLE",
) -> None:
    """Render admin-oriented Snowflake setup SQL including grants."""

    from kg_processor.application.snowflake_deployment import render_snowflake_setup_sql

    settings = Settings.load(config)
    typer.echo(render_snowflake_setup_sql(settings, role_name).strip())


@snowflake_app.command("objects-sql")
def snowflake_objects_sql(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Render schema-object Snowflake setup SQL for lower-privilege execution."""

    from kg_processor.application.snowflake_deployment import render_snowflake_objects_sql

    settings = Settings.load(config)
    typer.echo(render_snowflake_objects_sql(settings).strip())


@snowflake_app.command("access-check")
def snowflake_access_check(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    skip_cortex: Annotated[bool, typer.Option("--skip-cortex")] = False,
) -> None:
    """Check current Snowflake access to required objects and optional Cortex calls."""

    from kg_processor.application.snowflake_access import run_snowflake_access_check

    settings = Settings.load(config)
    report = run_snowflake_access_check(settings, check_cortex=not skip_cortex)
    _echo_json(report.model_dump(mode="json"))
    if not report.ok:
        raise typer.Exit(code=1)


@snowflake_app.command("export")
def snowflake_export(
    graph_id: Annotated[str, typer.Option("--graph-id", help="Graph to materialize.")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("out/kg"),
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    job_id: Annotated[str | None, typer.Option("--job-id", help="Diagnostic job identity.")] = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Diagnostic batch/run identity.")
    ] = None,
) -> None:
    """Materialize a Snowflake-resident graph as local inspection artifacts.

    Snowflake runs write graphs into KG_* tables while inspection and gold
    evaluation read a local parquet snapshot. This bridges the two so a
    Snowflake run can be compared against a gold fixture or a local run.
    """

    from kg_processor.application.snowflake_export import export_snowflake_graph

    settings = Settings.load(config)
    _echo_json(export_snowflake_graph(settings, graph_id, output, job_id=job_id, run_id=run_id))


@snowflake_app.command("retry")
def snowflake_retry(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    max_attempts: Annotated[
        int | None,
        typer.Option("--max-attempts", help="Attempt budget; defaults to distributed setting."),
    ] = None,
) -> None:
    """Return a job's failed files to the queue so a new worker can claim them.

    A run that dies mid-flight leaves its rows FAILED and its job row settled.
    Submission is idempotent and deliberately preserves processing status, so a
    relaunched service would otherwise drain immediately without doing any work.
    Rows past the attempt budget stay failed and are reported as exhausted.
    """

    from kg_processor.factories import build_job_manager

    settings = Settings.load(config)
    if not settings.job.use_file_queue:
        raise ValueError("snowflake retry requires job.use_file_queue=true")
    job_manager = build_job_manager(settings)
    if job_manager is None:
        raise RuntimeError("Snowflake file-queue job manager is not configured")
    budget = max_attempts if max_attempts is not None else settings.distributed.max_attempts
    result = job_manager.retry_failed_job_files(
        settings.job.job_id,
        settings.job.graph_id,
        budget,
    )
    _echo_json(result.model_dump(mode="json"))


@snowflake_app.command("submit")
def snowflake_submit(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Discover staged files and idempotently submit them to the SPCS work queue.

    Submission performs only stage listing and queue metadata writes. It does not
    invoke OCR, LLMs, embeddings, or start a compute pool, so operators can verify
    the exact queued file count before launching a cost-bearing job service.
    """

    from kg_processor.application.redaction import redact_sensitive_data
    from kg_processor.factories import build_file_source, build_job_manager

    settings = Settings.load(config)
    if not settings.job.use_file_queue:
        raise ValueError("snowflake submit requires job.use_file_queue=true")
    job_manager = build_job_manager(settings)
    if job_manager is None:
        raise RuntimeError("Snowflake file-queue job manager is not configured")
    files = build_file_source(settings).list_files()
    queued = job_manager.submit_job_files(
        settings.job.job_id,
        settings.job.graph_id,
        files,
        redact_sensitive_data(settings.model_dump(mode="json", by_alias=True)),
    )
    _echo_json(
        {
            "job_id": settings.job.job_id,
            "graph_id": settings.job.graph_id,
            "files_queued": queued,
        }
    )


@serving_app.command("sidecar")
def serving_sidecar() -> None:
    """Run the authenticating, priority-stamping proxy in front of an engine.

    Configuration comes from ``FLAKEGRAPH_SIDECAR_*`` variables so the container
    needs no arguments and no mounted processing config. The process refuses to
    start on an unsafe sequence limit rather than serving a fleet that will evict
    batch work mid-flight under KV pressure.
    """

    from kg_processor.serving.sidecar import SidecarConfig
    from kg_processor.serving.sidecar import run as run_sidecar

    run_sidecar(SidecarConfig.from_env())


@serving_app.command("ocr-shim")
def serving_ocr_shim() -> None:
    """Run the priority queue and admission control in front of a parsing pool.

    Configuration comes from ``FLAKEGRAPH_OCR_SHIM_*`` variables. The shim holds
    work rather than refusing it, because MinerU answers 409 when busy and the
    pipeline's HTTP transport does not retry.
    """

    from kg_processor.serving.ocr_shim import OcrShimConfig
    from kg_processor.serving.ocr_shim import run as run_ocr_shim

    run_ocr_shim(OcrShimConfig.from_env())


@serving_app.command("balancer")
def serving_balancer() -> None:
    """Keep machines clear and ready for people, and give the rest to batch work.

    Configuration comes from ``FLAKEGRAPH_BALANCER_*`` variables. The process
    holds no state that matters: roles are read from pod labels and occupancy
    from each engine's sidecar every pass, and every cap it sets lapses on its
    own, so stopping this changes the fleet back rather than leaving it
    reserved.
    """

    from kg_processor.serving.balancer import BalancerConfig
    from kg_processor.serving.balancer import run as run_balancer

    run_balancer(BalancerConfig.from_env())


@serving_app.command("sizing")
def serving_sizing(
    kv_heads: Annotated[int, typer.Option("--kv-heads")],
    head_dim: Annotated[int, typer.Option("--head-dim")],
    attention_layers: Annotated[int, typer.Option("--attention-layers")],
    weights_gib: Annotated[float, typer.Option("--weights-gib")],
    device_memory_gib: Annotated[float, typer.Option("--device-memory-gib")],
    max_num_seqs: Annotated[int, typer.Option("--max-num-seqs")],
    kv_cache_dtype: Annotated[str, typer.Option("--kv-cache-dtype")] = "fp8",
    recurrent_layers: Annotated[int, typer.Option("--recurrent-layers")] = 0,
    recurrent_state_bytes_per_layer: Annotated[
        int, typer.Option("--recurrent-state-bytes-per-layer")
    ] = 0,
    recurrent_state_slots_per_sequence: Annotated[
        int, typer.Option("--recurrent-state-slots-per-sequence")
    ] = 1,
    overhead_gib: Annotated[float, typer.Option("--overhead-gib")] = 12.0,
    gpu_memory_utilization: Annotated[float, typer.Option("--gpu-memory-utilization")] = 0.50,
    context_tokens: Annotated[int, typer.Option("--context-tokens")] = 32768,
) -> None:
    """Report whether a sequence limit binds before KV memory does.

    Exits non-zero when it does not, so the same arithmetic can gate a pipeline
    as well as answer an operator sizing a new GPU.
    """

    from kg_processor.serving.sizing import (
        BYTES_PER_GIB,
        DeviceBudget,
        ModelGeometry,
        compute_sizing,
    )

    verdict = compute_sizing(
        ModelGeometry(
            kv_heads=kv_heads,
            head_dim=head_dim,
            attention_layers=attention_layers,
            kv_cache_dtype=kv_cache_dtype,
            weights_bytes=int(weights_gib * BYTES_PER_GIB),
            recurrent_layers=recurrent_layers,
            recurrent_state_bytes_per_layer=recurrent_state_bytes_per_layer,
            recurrent_state_slots_per_sequence=recurrent_state_slots_per_sequence,
        ),
        DeviceBudget(
            device_memory_bytes=int(device_memory_gib * BYTES_PER_GIB),
            gpu_memory_utilization=gpu_memory_utilization,
            overhead_bytes=int(overhead_gib * BYTES_PER_GIB),
        ),
        context_tokens,
        max_num_seqs,
    )
    payload = verdict.model_dump()
    payload["kv_budget_gib"] = round(verdict.kv_budget_bytes / BYTES_PER_GIB, 2)
    _echo_json(payload)
    if not verdict.sequence_limit_binds_first:
        raise typer.Exit(code=1)


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _worker_iteration_payload(iteration: WorkerIteration) -> dict[str, Any]:
    """Convert an immutable worker result into a JSON-safe operator event."""

    return asdict(iteration)
