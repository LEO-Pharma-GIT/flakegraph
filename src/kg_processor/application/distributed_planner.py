# SPDX-License-Identifier: Apache-2.0
"""Build durable, dynamically fanned-out task graphs for FlakeGraph runs."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import batched, chain
from pathlib import Path
from typing import cast
from uuid import uuid4

from kg_processor.application.content_kinds import relative_location, source_roots
from kg_processor.application.extraction_contracts import extraction_contract_fingerprint
from kg_processor.application.ontology import load_ontology
from kg_processor.application.prompt_registry import two_pass_prompt_fingerprints
from kg_processor.application.redaction import redact_sensitive_data, redact_sensitive_text
from kg_processor.application.revision_plan import RevisionPlan, plan_revision
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import (
    ArtifactKind,
    ArtifactRef,
    PriorityBand,
    RevisionRequest,
    RunDefinition,
    RunSummary,
    TaskDefinition,
    TaskStage,
)
from kg_processor.domain.documents import InputFile
from kg_processor.domain.ids import sha256_hex, stable_id
from kg_processor.domain.ontology import OntologyProfile
from kg_processor.ports.artifact_store import ArtifactStore
from kg_processor.ports.file_source import FileSource, IterableFileSource
from kg_processor.ports.task_store import TaskStore

_SOURCE_STAGING_PARALLELISM = 16
_SOURCE_STAGING_MAX_IN_FLIGHT_BYTES = 512 * 1024 * 1024
_SOURCE_STAGING_BATCH_SIZE = 256
_FINALIZER_PASSWORD_SLOT = "KG_SNOWFLAKE_PASSWORD"
_FINALIZER_OAUTH_TOKEN_SLOT = "KG_SNOWFLAKE_OAUTH_TOKEN"


class DistributedRunPlanner:
    """Persist sources and the initial dependency graph for a distributed run.

    Each source starts with one preparation task. Preparation discovers the actual
    extraction windows after OCR and chunking, then publishes those fine-grained
    tasks atomically. Each document's windows converge on one compaction task. The
    final task is a run-wide barrier, so it needs no explicit edge from every
    document and remains constant-size for corpora containing hundreds of thousands
    of files.
    """

    def __init__(
        self,
        settings: Settings,
        file_source: FileSource | None,
        task_store: TaskStore,
        artifact_store: ArtifactStore,
    ) -> None:
        """Retain explicit ports so planning is testable without PostgreSQL or filesystems.

        A revision that only removes documents has no source to discover, so the
        file source is optional there; a run that adds nothing and keeps nothing
        is refused at submit.
        """

        self.settings = snapshot_run_ontology(settings)
        self.file_source = file_source
        self.task_store = task_store
        self.artifact_store = artifact_store

    def _run_owner(self, revision: RevisionRequest | None) -> str:
        """Who the run's graph belongs to, which every fleet run must record.

        The configured ``job.owner`` when there is one; for a revision, else the
        owner its base run recorded. A new graph with neither is refused: the
        console shows a graph only to its owner and those they share it with,
        so a graph nobody owns would be open to nobody.
        """

        if self.settings.job.owner:
            return self.settings.job.owner
        if revision is not None:
            return str(self.task_store.get_run_summary(revision.base_run_id).run.config["owner"])
        raise ValueError(
            "name the graph's owner - the sign-in address of the person it belongs "
            "to - with --owner, job.owner in the configuration, or KG_JOB_OWNER"
        )

    def submit(
        self,
        run_id: str | None = None,
        revision: RevisionRequest | None = None,
        priority_band: PriorityBand = PriorityBand.BULK,
    ) -> RunSummary:
        """Create, populate, and atomically activate one distributed graph run.

        Planning failures cancel the partially-created run. Source artifacts remain
        available for diagnosis until an operator deliberately removes terminal-run
        artifacts; workers can never see tasks until the complete DAG is activated.

        With a ``revision`` the run is a new version of the base run's graph: it
        keeps the base's documents minus the dropped ones by reading their stage
        outputs, and processes only what its source adds.

        ``priority_band`` is how urgently every task of the run is claimed,
        including the work workers discover as it runs.
        """

        owner = self._run_owner(revision)
        plan = self._plan_revision(revision) if revision is not None else None
        files = self._iter_input_files()
        if plan is not None:
            files = plan.without_known(files)
        first_file = next(files, None)
        if first_file is None and plan is None:
            raise ValueError("distributed run source did not contain any supported files")
        if (
            first_file is None
            and plan is not None
            and not plan.changes_anything()
            and not (revision is not None and revision.rebuild)
        ):
            held = f"; the {len(plan.skipped)} offered are already in it" if plan.skipped else ""
            raise ValueError(
                "revision adds no new documents and drops none, "
                f"so the graph would not change{held}"
            )
        validated_files = _iter_unique_file_ids(
            chain((first_file,), files) if first_file is not None else files
        )
        # Discover the full immutable submission before accepting an idempotency
        # key. Spooling metadata retains bounded memory for large source listings;
        # source bytes are still uploaded in bounded batches after validation.
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as file_records:
            file_identities = []
            for file in validated_files:
                file_records.write(file.model_dump_json() + "\n")
                identity = file.model_dump(mode="json", exclude={"path"})
                file_identities.append(sha256_hex(json.dumps(identity, sort_keys=True)))
            file_records.seek(0)
            effective_run_id = run_id or f"run_{uuid4().hex}"
            snapshot = {
                **distributed_processing_config(self.settings),
                "owner": owner,
                "submission_identity": sha256_hex(
                    json.dumps(
                        {
                            "graph_id": self.settings.job.graph_id,
                            "owner": owner,
                            "ontology": self.settings.ontology.model_dump(mode="json"),
                            "output": _graph_output_payload(self.settings),
                            "revision": revision.model_dump(mode="json") if revision else None,
                            "files": sorted(file_identities),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
            }
            definition = RunDefinition(
                id=effective_run_id,
                graph_id=self.settings.job.graph_id,
                config=snapshot,
                config_digest=distributed_processing_config_digest(self.settings),
                priority_band=priority_band,
            )
            if run_id is not None:
                try:
                    existing = self.task_store.get_run_summary(effective_run_id)
                except KeyError:
                    existing = None
                if existing is not None:
                    if (
                        existing.run.graph_id != definition.graph_id
                        or existing.run.config.get("submission_identity")
                        != snapshot["submission_identity"]
                        or existing.run.config_digest != definition.config_digest
                        or existing.run.priority_band != definition.priority_band
                    ):
                        raise ValueError(
                            f"run id {effective_run_id} already exists with different inputs"
                        )
                    # Explicit run IDs are idempotency keys. Returning the existing
                    # summary is safe for active and terminal runs and, critically,
                    # never cancels live work merely because submit was retried.
                    if existing.run.status.value != "planning":
                        return existing
            self.task_store.create_run(definition)
            try:
                tasks = self._iter_initial_tasks(
                    effective_run_id,
                    (InputFile.model_validate_json(line) for line in file_records),
                    plan,
                )
                self.task_store.add_initial_tasks(effective_run_id, tasks)
                self.task_store.activate_run(effective_run_id)
            except Exception:
                # Explicit run IDs are durable planning idempotency keys. PostgreSQL
                # commits bounded COPY batches so an interrupted coordinator can
                # safely replay the deterministic stream and activate the same run.
                try:
                    partial = self.task_store.get_run_summary(effective_run_id)
                except Exception:
                    partial = None
                if run_id is None or partial is None or partial.total_tasks == 0:
                    self.task_store.cancel_run(effective_run_id)
                raise
            # Submission is an operator-facing path, so return bounded counts rather
            # than serializing every task immediately after inserting a large plan.
            return self.task_store.get_run_summary(effective_run_id)

    def _plan_revision(self, revision: RevisionRequest) -> RevisionPlan:
        """Work out what a revision keeps from its base, reprocesses, and must not re-add.

        Revalidation replays stored records through the LLM extractors in place
        of their extraction calls; another extractor has no call to replay, so a
        configuration that uses one is refused before anything is planned.
        """

        extractors = self.settings.extractors
        if revision.revalidate and (
            extractors.entity_provider != "llm" or extractors.relation_provider != "llm"
        ):
            raise ValueError(
                "--revalidate replays stored records through the LLM extractors, but this "
                f"configuration extracts entities with {extractors.entity_provider} and "
                f"relations with {extractors.relation_provider}; re-extract instead"
            )
        plan = plan_revision(
            revision, self.settings.job.graph_id, self.task_store, self.artifact_store
        )
        self._require_folder_entity_of_base(revision)
        return plan

    def _require_folder_entity_of_base(self, revision: RevisionRequest) -> None:
        """Refuse a revision that reads kept documents' folders unlike their extraction did.

        A kept document's folder entity is a mention its document-context pass
        stored, so a revision may change which documents a folder entity leaves
        out but not its type, relation, or level - unless it extracts the
        document context of every kept document again.
        """

        base = self.task_store.get_run_summary(revision.base_run_id).run.config
        stored = _extraction_folder_entity(base.get("ontology"))
        wanted = _extraction_folder_entity(self.settings.ontology.model_dump(mode="json"))
        reextracts_context = revision.reextract_from == "context" and any(
            selector.strip() == "all" for selector in revision.reextract
        )
        if stored == wanted or reextracts_context:
            return
        raise ValueError(
            f"this revision ties documents to their folders as {_describe_folder(wanted)}, "
            f"but base run {revision.base_run_id} extracted them as "
            f"{_describe_folder(stored)}; a kept document's folder entity is stored with "
            "its document context, so extract that again with --reextract all --from "
            "context, or keep the folder entity's type, relation and level and change "
            "only its exclusions"
        )

    def _iter_initial_tasks(
        self,
        run_id: str,
        files: Iterable[InputFile],
        plan: RevisionPlan | None = None,
    ) -> Iterator[TaskDefinition]:
        """Yield stable tasks from bounded, continuously replenished upload batches.

        Task and artifact identities derive from the run and file rather than
        insertion order. Emitting each upload as it completes therefore lets a
        slow object overlap later files without changing retry or graph semantics.
        A revision's reprocessed documents follow the new files, one seed task
        each, once the files have said which kept documents they replace.
        """

        for file_batch in batched(files, _SOURCE_STAGING_BATCH_SIZE):
            for file, source_ref in self._iter_staged_sources(run_id, file_batch):
                yield TaskDefinition(
                    id=stable_id("task", run_id, TaskStage.PREPARE_DOCUMENT.value, file.id),
                    run_id=run_id,
                    stage=TaskStage.PREPARE_DOCUMENT,
                    scope_id=file.id,
                    payload={"source_artifact_id": source_ref.id},
                    max_attempts=self.settings.distributed.max_attempts,
                )
        payload = _graph_output_payload(self.settings)
        # Read after the files streamed past: a file that replaced a kept
        # document has left the kept set by now.
        if plan is not None:
            yield from plan.seed_tasks(run_id, self.settings)
            inherited = plan.inherited
            if inherited:
                payload["inherit"] = [
                    entry.model_dump(mode="json", exclude_defaults=True) for entry in inherited
                ]
            reprocessed = plan.reprocessed()
            if reprocessed:
                payload["reprocessed"] = reprocessed
        yield TaskDefinition(
            id=stable_id("task", run_id, TaskStage.FINALIZE_GRAPH.value),
            run_id=run_id,
            stage=TaskStage.FINALIZE_GRAPH,
            scope_id=self.settings.job.graph_id,
            payload=payload,
            max_attempts=self.settings.distributed.max_attempts,
        )

    def _iter_input_files(self) -> Iterator[InputFile]:
        """Use streaming discovery when an adapter provides deterministic iteration."""

        if self.file_source is None:
            return iter(())
        if isinstance(self.file_source, IterableFileSource):
            return self.file_source.iter_files()
        files = sorted(
            self.file_source.list_files(),
            key=lambda item: (item.source_uri, item.id),
        )
        return iter(files)

    def _iter_staged_sources(
        self,
        run_id: str,
        files: Sequence[InputFile],
    ) -> Iterator[tuple[InputFile, ArtifactRef]]:
        """Yield source/reference pairs as bounded concurrent uploads complete.

        A planning batch contains at most 256 lightweight file records. Queuing
        that bounded metadata is cheap, while only ``workers`` calls can read and
        retain complete source bytes. Completion-order iteration immediately
        replaces a finished upload even when an earlier object remains slow.
        """

        workers = _source_staging_workers(files)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            uploads = {executor.submit(self._store_source, run_id, file): file for file in files}
            for upload in as_completed(uploads):
                yield uploads[upload], upload.result()

    def _store_source(self, run_id: str, file: InputFile) -> ArtifactRef:
        """Read and checksum-verify one file before handing its bytes to workers."""

        payload = file.path.read_bytes()
        checksum = sha256_hex(payload)
        if checksum != file.checksum:
            raise ValueError(
                f"source file changed after discovery: {file.path} "
                f"(expected {file.checksum}, got {checksum})"
            )
        if len(payload) != file.size_bytes:
            raise ValueError(
                f"source file size changed after discovery: {file.path} "
                f"(expected {file.size_bytes}, got {len(payload)})"
            )
        return self.artifact_store.put(
            run_id,
            ArtifactKind.SOURCE_DOCUMENT,
            payload,
            file.mime_type,
            metadata={
                "input_file": {
                    "id": file.id,
                    "source_uri": redact_sensitive_text(file.source_uri),
                    # The worker reads a staged copy under its own settings, so
                    # where the file sits is fixed here, under the run's roots.
                    "location": redact_sensitive_text(
                        file.location
                        or relative_location(file.source_uri, source_roots(self.settings))
                    ),
                    "checksum": file.checksum,
                    "mime_type": file.mime_type,
                    "size_bytes": file.size_bytes,
                    "filename": Path(file.path).name,
                }
            },
            identity_key=file.id,
        )


def _graph_output_payload(settings: Settings) -> dict[str, object]:
    """Return the non-secret destination contract consumed by finalizer workers.

    Worker pools may serve many runs with different Snowflake schemas. Connection
    coordinates therefore travel with the final task, while credentials remain
    named environment references resolved only inside the claiming worker.
    """

    provider = settings.writer.provider
    if provider == "local_artifacts":
        return {"output": {"provider": provider}}
    if provider != "snowflake_bulk":
        raise ValueError(
            "distributed Spark finalization supports local_artifacts or snowflake_bulk output"
        )
    snowflake = settings.snowflake
    _validate_finalizer_credential_slot(
        snowflake.password_environment_variable,
        _FINALIZER_PASSWORD_SLOT,
        "password",
    )
    _validate_finalizer_credential_slot(
        snowflake.oauth_token_environment_variable,
        _FINALIZER_OAUTH_TOKEN_SLOT,
        "OAuth token",
    )
    return {
        "output": {
            "provider": provider,
            "snowflake": {
                "account": snowflake.account,
                "host": snowflake.host,
                "user": snowflake.user,
                "authenticator": snowflake.authenticator,
                "database": snowflake.database,
                "schema": snowflake.schema_name,
                "role": snowflake.role,
                "warehouse": snowflake.warehouse,
                "bulk_stage": snowflake.bulk_stage,
                "password_credential_slot": (
                    "password" if snowflake.password_environment_variable else None
                ),
                "oauth_token_credential_slot": (
                    "oauth_token" if snowflake.oauth_token_environment_variable else None
                ),
                "bulk_target_file_size_mb": snowflake.bulk_target_file_size_mb,
            },
        }
    }


def _validate_finalizer_credential_slot(
    configured_name: str | None,
    dedicated_name: str,
    label: str,
) -> None:
    """Prevent task payloads from turning arbitrary worker environment into secrets."""

    if configured_name is not None and configured_name != dedicated_name:
        raise ValueError(
            f"distributed Snowflake finalizers require the dedicated {label} "
            f"credential slot {dedicated_name}"
        )


def _source_staging_workers(files: Sequence[InputFile]) -> int:
    """Bound upload concurrency by both object count and worst-case memory.

    ``ArtifactStore.put`` accepts immutable bytes, so each active staging worker
    must hold one complete source object while checksumming and uploading it.
    Typical papers retain sixteen-way object-store concurrency; unusually large
    files reduce the pool automatically so a submitter cannot transiently retain
    many gigabytes merely because the corpus contains several large documents.
    """

    largest_file = max(file.size_bytes for file in files)
    byte_limited_workers = max(
        1,
        _SOURCE_STAGING_MAX_IN_FLIGHT_BYTES // max(largest_file, 1),
    )
    return min(_SOURCE_STAGING_PARALLELISM, len(files), byte_limited_workers)


def _iter_unique_file_ids(files: Iterable[InputFile]) -> Iterator[InputFile]:
    """Reject duplicate durable file identities while preserving streaming order."""

    seen: set[str] = set()
    for file in files:
        if file.id in seen:
            raise ValueError(f"duplicate input file id: {file.id}")
        seen.add(file.id)
        yield file


def _extraction_folder_entity(ontology: object) -> dict[str, object] | None:
    """The folder entity a run's ontology extracts with, without its exclusions."""

    profile = ontology.get("profile") if isinstance(ontology, dict) else None
    if not isinstance(profile, dict):
        return None
    folder = OntologyProfile.model_validate(profile).for_extraction().folder_entity
    return folder.model_dump(mode="json") if folder is not None else None


def _describe_folder(folder: dict[str, object] | None) -> str:
    if folder is None:
        return "no folder entity"
    return f"{folder['entity_type']} at level {folder['level']} linked by {folder['relation_type']}"


def snapshot_run_ontology(settings: Settings) -> Settings:
    """Carry an explicitly selected vocabulary by value across worker filesystems.

    An omitted ontology deliberately uses the fleet's mounted default. Explicit
    graph labels are another ontology source and must travel with the run just
    like a profile file, even if they happen to equal the built-in defaults.
    """

    selected = (
        settings.ontology.profile is not None
        or settings.ontology.profile_path is not None
        or bool({"entity_types", "relation_types"} & settings.graph.model_fields_set)
    )
    if not selected:
        return settings
    profile = load_ontology(
        settings.ontology.profile_path,
        settings.graph.entity_types,
        settings.graph.relation_types,
        inline=settings.ontology.profile,
    ).profile
    return settings.model_copy(
        update={
            "ontology": settings.ontology.model_copy(
                update={"profile": profile.model_dump(mode="json"), "profile_path": None}
            )
        }
    )


def distributed_processing_config(settings: Settings) -> dict[str, object]:
    """Return the redacted graph-semantic configuration shared by every worker.

    Worker IDs, leases, polling intervals, source discovery paths, and deployment
    runtime do not affect graph semantics and are intentionally excluded. This lets
    specialized worker pools tune operations without failing compatibility checks.
    """

    full = settings.model_dump(mode="json", by_alias=True)
    snapshot = {
        "graph_id": settings.job.graph_id,
        "ocr": full["ocr"],
        "generic_http_ocr": full["generic_http_ocr"],
        "llm": full["llm"],
        "embedding": full["embedding"],
        "ontology": full["ontology"],
        "extractors": full["extractors"],
        "graph": full["graph"],
        "writer": full["writer"],
        "cache": full["cache"],
        "snowflake": full["snowflake"],
    }
    return cast(dict[str, object], redact_sensitive_data(snapshot))


def distributed_processing_config_digest(settings: Settings) -> str:
    """Hash graph-semantic settings while permitting deployment-local routing.

    The persisted run snapshot retains redacted endpoints and operational settings
    for diagnosis, but those values cannot participate in worker compatibility.
    A production fleet may route extraction workers to node-local replicas while a
    finalizer uses a cluster-wide Service, and heterogeneous workers may use
    different device names or cache paths. Provider and model identities,
    parsing behavior, extraction policy, and graph policy remain part of the
    digest. The ontology travels in the run snapshot rather than constraining
    which otherwise compatible worker may claim it.
    """

    canonical = json.dumps(
        distributed_processing_compatibility_config(settings),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256_hex(canonical)


def distributed_processing_compatibility_config(settings: Settings) -> dict[str, object]:
    """Return the canonical semantic contract that every worker must share.

    Transport addresses, credentials, timeouts, executable paths, batching, cache
    locations, writer destinations, and Snowflake connection details describe how
    a worker reaches infrastructure rather than what graph it should produce. They
    are deliberately absent. So is the ontology: what a graph extracts is chosen
    per run and travels in the run's stored configuration, which every worker
    applies when it claims the run - a worker's own mounted profile is only the
    default for runs that name none. What stays is what the worker *is*: its
    models, parsers, prompts and graph policy.
    """

    full = settings.model_dump(mode="json", by_alias=True)
    compatibility = {
        # Prompt text and response contracts are executable extraction policy,
        # so worker images that differ in them must not contribute to one run
        # under identical YAML. Read from the prompts and contracts themselves rather than
        # from a revision label: the label is a second thing to remember on every
        # edit, and a forgotten bump is invisible — both images claim the same
        # run and mix observations produced under different instructions.
        "extraction_prompts": two_pass_prompt_fingerprints(),
        "extraction_contracts": extraction_contract_fingerprint(),
        "ocr": _without_keys(
            full["ocr"],
            {
                "mineru_api_key",
                "mineru_api_url",
                "mineru_command",
                "mineru_server_url",
                "model_cache_dir",
                "tesseract_command",
                "tesseract_pdf_renderer_command",
                "timeout_seconds",
            },
        ),
        "generic_http_ocr": _without_keys(
            full["generic_http_ocr"],
            {"api_key", "endpoint", "max_response_bytes"},
        ),
        "llm": _without_keys(full["llm"], {"api_key", "endpoint", "timeout_seconds"}),
        "embedding": _without_keys(
            full["embedding"],
            {"api_key", "batch_size", "device", "endpoint"},
        ),
        "extractors": full["extractors"],
        "graph": _without_keys(
            full["graph"],
            {
                "extraction_parallelism",
                "finalization_provider_concurrency",
                # The vocabulary is the run's, like the ontology profile.
                "entity_types",
                "relation_types",
            },
        ),
    }
    return cast(dict[str, object], compatibility)


def _without_keys(value: object, excluded: set[str]) -> dict[str, object]:
    """Copy one serialized settings mapping without deployment-specific fields."""

    if not isinstance(value, dict):
        raise TypeError("serialized settings section must be a mapping")
    return {str(key): item for key, item in value.items() if str(key) not in excluded}
