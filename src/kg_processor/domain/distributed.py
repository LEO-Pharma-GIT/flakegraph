# SPDX-License-Identifier: Apache-2.0
"""Durable run, task, lease, and artifact models for distributed execution.

These records describe FlakeGraph workflow semantics without naming Kubernetes,
PostgreSQL, Snowflake, or a particular worker implementation. Backends may provide
at-least-once delivery, while deterministic task IDs and immutable artifacts make
successful retries idempotent.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskStage(StrEnum):
    """Name independently leased pipeline work that may run on separate workers."""

    PREPARE_DOCUMENT = "prepare_document"
    EXTRACT_DOCUMENT_CONTEXT = "extract_document_context"
    EXTRACT_ENTITY_WINDOW = "extract_entity_window"
    COMPACT_ENTITY_INVENTORY = "compact_entity_inventory"
    EXTRACT_RELATION_WINDOW = "extract_relation_window"
    COMPACT_DOCUMENT = "compact_document"
    # A revision revalidates a kept document in the phases extraction takes:
    # its records are placed by window, each window's entities and then its
    # relations are replayed, and the document is combined. Its entity
    # inventory is compacted by ``COMPACT_ENTITY_INVENTORY``.
    REVALIDATE_DOCUMENT_CONTEXT = "revalidate_document_context"
    REVALIDATE_ENTITY_WINDOW = "revalidate_entity_window"
    REVALIDATE_RELATION_WINDOW = "revalidate_relation_window"
    REVALIDATE_DOCUMENT = "revalidate_document"
    FINALIZE_GRAPH = "finalize_graph"


# The order a run's ready work is claimed in, higher first - the stage ladder,
# which stays within 0-20 so a band's offset keeps bulk and interactive work
# apart. Across runs, a claim first picks the band and then the run with the
# fewest tasks leased in the claiming pool; the ladder decides among the rest.
#
# The compaction stages come first. They make no model call and finish in
# moments, and each one unlocks the next phase of its document: an entity
# inventory releases the document's relation windows, a compacted or
# revalidated document brings its run closer to finalizing. Ranked below the
# model stages they share a pool with, they would wait behind every window of
# every other run - a finished run could sit for hours with nothing left but
# bookkeeping. Among the model stages, earlier ones rank higher so documents
# keep flowing into the pipeline, and a revalidation's phases rank with the
# extraction phases they mirror. Preparation and finalizing have pools of
# their own.
STAGE_PRIORITY: dict[TaskStage, int] = {
    TaskStage.PREPARE_DOCUMENT: 20,
    TaskStage.COMPACT_ENTITY_INVENTORY: 19,
    TaskStage.COMPACT_DOCUMENT: 18,
    TaskStage.REVALIDATE_DOCUMENT: 18,
    TaskStage.EXTRACT_DOCUMENT_CONTEXT: 15,
    TaskStage.REVALIDATE_DOCUMENT_CONTEXT: 15,
    TaskStage.EXTRACT_ENTITY_WINDOW: 10,
    TaskStage.REVALIDATE_ENTITY_WINDOW: 10,
    TaskStage.EXTRACT_RELATION_WINDOW: 4,
    TaskStage.REVALIDATE_RELATION_WINDOW: 4,
    TaskStage.FINALIZE_GRAPH: 0,
}


class PriorityBand(StrEnum):
    """How urgently a run's work is claimed, relative to every other run.

    An interactive run is one someone is waiting on: its ready work is claimed
    ahead of every bulk run's. Within a band, runs share the workers equally.
    """

    INTERACTIVE = "interactive"
    BULK = "bulk"


# Added to the stage ladder for every task of a run in the band. The ladder
# never reaches 1000, so the two bands cannot interleave.
PRIORITY_BAND_OFFSET: dict[PriorityBand, int] = {
    PriorityBand.INTERACTIVE: 1_000,
    PriorityBand.BULK: 0,
}


def task_priority(stage: TaskStage, band: PriorityBand) -> int:
    """The stored priority of a task of ``stage`` in a run of ``band``.

    Workers claim by ``priority DESC``, so a larger number is served first.
    The serving plane inverts this - vLLM serves the lowest value first - and
    the two must not be confused.
    """

    return STAGE_PRIORITY[stage] + PRIORITY_BAND_OFFSET[band]


class RunStatus(StrEnum):
    """Represent the externally visible lifecycle of one graph build."""

    PLANNING = "planning"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(StrEnum):
    """Represent durable task state under at-least-once worker execution."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PublicationStatus(StrEnum):
    """Represent durable delivery of a finalized graph to an external destination."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactKind(StrEnum):
    """Identify immutable payload schemas stored between worker stages."""

    SOURCE_DOCUMENT = "source_document"
    PREPARED_DOCUMENT = "prepared_document"
    DOCUMENT_CONTEXT = "document_context"
    EXTRACTION_WINDOW = "extraction_window"
    EXTRACTED_ENTITY_WINDOW = "extracted_entity_window"
    DOCUMENT_ENTITY_INVENTORY = "document_entity_inventory"
    EXTRACTED_RELATION_WINDOW = "extracted_relation_window"
    EXTRACTED_DOCUMENT = "extracted_document"
    GRAPH_RESULT = "graph_result"


class RunDefinition(BaseModel):
    """Describe a graph run before its task graph becomes executable."""

    id: str
    graph_id: str
    config: dict[str, Any]
    config_digest: str
    status: RunStatus = RunStatus.PLANNING
    priority_band: PriorityBand = PriorityBand.BULK

    @field_validator("id", "graph_id", "config_digest")
    @classmethod
    def identity_fields_must_not_be_blank(cls, value: str) -> str:
        """Reject ambiguous durable identities before database insertion."""

        if not value.strip():
            raise ValueError("distributed run identity fields must not be blank")
        return value


class InheritedDocuments(BaseModel):
    """Documents a run keeps from an earlier run of the same graph.

    A revision re-extracts only what it adds; everything it keeps is read from
    the stage outputs the earlier run already produced, so the finalizer builds
    the graph from both. Entries name concrete runs: a revision of a revision
    inherits from each original run directly rather than through a chain.

    ``file_ids`` are read for every output the run holds for them.
    ``prepared_file_ids`` are read for their prepared text only: their
    extraction is newer - redone by this revision, or by a later run that kept
    the text and extracted it again - so a document's text and its extraction
    may come from two runs, and nothing is copied either way.
    """

    run_id: str
    file_ids: list[str] = Field(default_factory=list)
    prepared_file_ids: list[str] = Field(default_factory=list)

    @field_validator("run_id")
    @classmethod
    def run_id_must_not_be_blank(cls, value: str) -> str:
        """Refuse an inheritance that names no run."""

        if not value.strip():
            raise ValueError("inherited documents must name their run")
        return value

    @model_validator(mode="after")
    def must_keep_something(self) -> InheritedDocuments:
        """Refuse an entry that would read nothing from its run."""

        if not self.file_ids and not self.prepared_file_ids:
            raise ValueError("inherited documents must name at least one document")
        return self

    def reads(self, file_ids: list[str], kind: ArtifactKind) -> bool:
        """Whether an artifact of ``kind`` covering ``file_ids`` is read from this run."""

        covered = set(file_ids)
        if not covered:
            return False
        if covered <= set(self.file_ids):
            return True
        return kind == ArtifactKind.PREPARED_DOCUMENT and covered <= set(self.prepared_file_ids)


ReextractStage = Literal["context", "entities", "relations"]
# What a revision did with each document it holds. A document the revision
# added or replaced is simply extracted; everything else is labelled here.
PROVENANCE_EXTRACTED = "extracted"
PROVENANCE_REUSED = "reused"
PROVENANCE_REVALIDATED = "revalidated"
_SELECTOR_KINDS = ("file", "folder", "kind")
_DOCUMENT_KINDS = ("document", "dataset", "image")


def reextracted_provenance(stage: ReextractStage) -> str:
    """The provenance label of a document extracted again from ``stage`` onward."""

    return f"reextracted:{stage}"


class RevisionRequest(BaseModel):
    """Ask for a new run of a graph that keeps an earlier run's documents.

    ``drop_file_ids`` are removed from what is kept; whatever the file source
    discovers is added, with a document whose bytes the graph already holds
    skipped and one whose identity it holds replaced.

    A kept document is normally reused as it is. ``reextract`` selects kept
    documents to extract again with the current code and ontology from
    ``reextract_from`` onward, reusing their earlier stage outputs before that
    stage; their text is never read again. ``revalidate`` puts every other kept
    document's stored records through the current validation and verification
    without asking the model to extract anything.
    """

    base_run_id: str
    drop_file_ids: list[str] = Field(default_factory=list)
    # Build a new version even though no document changes. The documents are
    # the same, but how a graph is built from them has changed - a finalizer
    # fix - and the only way to apply it to an existing graph is to finalize
    # it again. Nothing is extracted again: kept documents are read from the
    # base run's stage outputs, as for any revision.
    rebuild: bool = False
    # ``file:<id>``, ``folder:<prefix>``, ``kind:<document|dataset|image>``,
    # or ``all``; a bare value is a file id.
    reextract: list[str] = Field(default_factory=list)
    reextract_from: ReextractStage = "context"
    revalidate: bool = False

    @field_validator("reextract")
    @classmethod
    def selectors_must_be_readable(cls, value: list[str]) -> list[str]:
        """Refuse a selector that names no known way of choosing documents."""

        for selector in value:
            kind, text = parse_document_selector(selector)
            if kind == "kind" and text not in _DOCUMENT_KINDS:
                raise ValueError(
                    f"document kind must be one of {', '.join(_DOCUMENT_KINDS)}: {selector}"
                )
            if kind == "file" and not text:
                raise ValueError(f"a document selector must name something: {selector!r}")
        return value


def parse_document_selector(selector: str) -> tuple[str, str]:
    """Split a re-extraction selector into what it matches on and the value.

    ``all`` matches every kept document, and a value without a known prefix is a
    file id, so the common case - naming one document - needs no prefix.
    """

    text = selector.strip()
    if text == "all":
        return "folder", ""
    prefix, separator, rest = text.partition(":")
    if separator and prefix in _SELECTOR_KINDS:
        return prefix, rest.strip()
    return "file", text


def document_selected(selector: str, file_id: str, folder: str | None, kind: str | None) -> bool:
    """Whether a re-extraction selector picks this document.

    A folder selector is a prefix of the folder path matched on whole folder
    names: ``reports`` picks ``reports`` and ``reports/2024``, not ``reports-old``.
    """

    selector_kind, value = parse_document_selector(selector)
    if selector_kind == "file":
        return file_id == value
    if selector_kind == "kind":
        return kind == value
    prefix = value.strip("/")
    folder_path = (folder or "").strip("/")
    return not prefix or folder_path == prefix or folder_path.startswith(prefix + "/")


def document_provenance(payload: Mapping[str, Any]) -> dict[str, str]:
    """Each document's provenance label as a finalizer's payload records it.

    Documents whose extraction a revision reads from an earlier run are reused;
    those it revalidated or extracted again are listed under ``reprocessed``. A
    document absent from the result was extracted by the run itself, as every
    document of a first run is.
    """

    labels: dict[str, str] = {}
    entries = payload.get("inherit")
    for raw in entries if isinstance(entries, list) else []:
        for file_id in InheritedDocuments.model_validate(raw).file_ids:
            labels[file_id] = PROVENANCE_REUSED
    reprocessed = payload.get("reprocessed")
    if isinstance(reprocessed, Mapping):
        for label, file_ids in reprocessed.items():
            for file_id in file_ids if isinstance(file_ids, list) else []:
                labels[str(file_id)] = str(label)
    return labels


class TaskDefinition(BaseModel):
    """Describe one idempotent stage invocation and its dependency barrier.

    A task names no priority of its own. The store gives every task of a run
    ``task_priority(stage, band)`` for the band the run was submitted at, so
    work a worker discovers is claimed in the same band as the rest of its run.
    """

    id: str
    run_id: str
    stage: TaskStage
    scope_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    dependency_ids: list[str] = Field(default_factory=list)
    max_attempts: int = Field(default=3, ge=1)

    @field_validator("id", "run_id", "scope_id")
    @classmethod
    def task_identity_fields_must_not_be_blank(cls, value: str) -> str:
        """Require stable identifiers for leases, retries, and audit events."""

        if not value.strip():
            raise ValueError("distributed task identity fields must not be blank")
        return value


class TaskLease(BaseModel):
    """Carry one claimed task plus completed dependency outputs to a worker."""

    task: TaskDefinition
    worker_id: str
    attempt: int = Field(ge=1)
    claim_token: str = Field(min_length=1)
    lease_expires_at: datetime
    dependency_outputs: dict[str, list[str]] = Field(default_factory=dict)


class PublicationLease(BaseModel):
    """Carry one durable graph-publication command and its fencing generation."""

    id: str
    run_id: str
    artifact_id: str
    task_payload: dict[str, Any]
    worker_id: str
    attempt: int = Field(ge=1)
    generation: int = Field(ge=1)
    lease_expires_at: datetime


class TaskProgress(BaseModel):
    """Describe bounded progress within one otherwise indivisible queue task.

    Most stages fan out into many independently countable tasks. Graph
    finalization is intentionally one run-wide barrier, so it publishes its
    current phase here instead of inventing queue tasks that could execute out of
    order or duplicate graph-wide work.
    """

    phase: str
    phase_index: int = Field(ge=1)
    phase_total: int = Field(ge=1)
    completed: int = Field(default=0, ge=0)
    total: int | None = Field(default=None, ge=1)
    message: str | None = None

    @field_validator("phase")
    @classmethod
    def phase_must_not_be_blank(cls, value: str) -> str:
        """Reject progress records that cannot identify their current work."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("task progress phase must not be blank")
        return normalized

    @model_validator(mode="after")
    def counters_must_be_consistent(self) -> TaskProgress:
        """Keep phase and optional inner-work counters internally consistent."""

        if self.phase_index > self.phase_total:
            raise ValueError("task progress phase_index must not exceed phase_total")
        if self.total is not None and self.completed > self.total:
            raise ValueError("task progress completed must not exceed total")
        return self


class TaskSnapshot(BaseModel):
    """Expose task state for status commands, diagnostics, and progress UIs."""

    task: TaskDefinition
    status: TaskStatus
    attempts: int = Field(ge=0)
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    output_artifact_ids: list[str] = Field(default_factory=list)
    last_error: dict[str, Any] | None = None
    progress: TaskProgress | None = None


class RunSnapshot(BaseModel):
    """Expose a run and all of its tasks as one consistent status response."""

    run: RunDefinition
    tasks: list[TaskSnapshot] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class TaskCount(BaseModel):
    """Aggregate one stage/status bucket without materializing individual tasks."""

    stage: TaskStage
    status: TaskStatus
    count: int = Field(ge=0)
    # Of a queued bucket, how many a worker could claim right now: their
    # dependencies are met and no retry delay is pending. The rest wait on
    # earlier stages, so no pool is short of workers on their account.
    ready: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: TaskProgress | None = None


class ResolvedFailure(BaseModel):
    """A task failure an operator retried past, kept after the retry cleared it.

    Retrying a failed run requeues its failed tasks with a fresh budget and
    clears their error, so without this record a run that later succeeded
    would show no sign that it once stopped, or why.
    """

    task_id: str
    stage: str
    scope_id: str
    attempts: int = Field(ge=0)
    error: dict[str, Any] | None = None
    failed_at: datetime | None = None
    retried_at: datetime


class RunSummary(BaseModel):
    """Expose constant-size run progress suitable for very large task graphs.

    Operators normally need stage progress, not hundreds of thousands of task
    payloads. Detailed snapshots remain available for explicit diagnosis while
    submit, status, cancel, and retry can return this bounded representation. A
    terminal run carries only its single redacted root error, never task payloads.
    """

    run: RunDefinition
    task_counts: list[TaskCount] = Field(default_factory=list)
    total_tasks: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    error: dict[str, Any] | None = None
    # Documents whose preparation gave up and recorded them as unreadable. Their
    # tasks succeed, so the task counts alone cannot say so.
    documents_failed: int = Field(default=0, ge=0)
    # The configuration digest each stage's worker fleet declared at startup.
    # A worker claims only tasks whose run digest equals its own, so a stage
    # served at another digest is one this run's remaining work cannot leave.
    fleet_config_digests: dict[str, str] = Field(default_factory=dict)
    # Failures a retry cleared, most recent first and bounded; the count is of
    # all of them.
    resolved_failures: list[ResolvedFailure] = Field(default_factory=list)
    resolved_failure_count: int = Field(default=0, ge=0)


class RunOverview(BaseModel):
    """Describe one recent run without returning configuration or task payloads."""

    id: str
    graph_id: str
    status: RunStatus
    priority_band: PriorityBand
    task_counts: list[TaskCount] = Field(default_factory=list)
    total_tasks: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class GraphDeletion(BaseModel):
    """What deleting a graph removed: every run of it and all they stored.

    Counts are of what this call removed, so deleting a graph that is already
    gone reports zeros rather than failing.
    """

    graph_id: str
    run_ids: list[str] = Field(default_factory=list)
    tasks: int = Field(default=0, ge=0)
    artifacts: int = Field(default=0, ge=0)
    objects: int = Field(default=0, ge=0)


class ArtifactRef(BaseModel):
    """Identify an immutable content-addressed stage payload."""

    id: str
    run_id: str
    kind: ArtifactKind
    media_type: str
    checksum: str
    size_bytes: int = Field(ge=0)
    storage_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredArtifact(BaseModel):
    """Return artifact metadata and exact uncompressed payload bytes."""

    ref: ArtifactRef
    payload: bytes
