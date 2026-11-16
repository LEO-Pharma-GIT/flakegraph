# SPDX-License-Identifier: Apache-2.0
"""Work out what a revision of a graph keeps, reprocesses, and must not add again.

A graph's documents are held as stage outputs of the runs that processed them.
A document's text (its prepared output) and its extraction may live in two
runs: a revision that extracts a kept document again, or revalidates it,
reads the text where an earlier run left it and stores only the new
extraction. This module reads those outputs from the coordination store once,
at planning, so no worker ever searches an earlier run for them: every task a
revision queues for a kept document names the artifacts it starts from.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from kg_processor.application.content_kinds import content_kind
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import (
    PROVENANCE_REVALIDATED,
    ArtifactKind,
    ArtifactRef,
    InheritedDocuments,
    RevisionRequest,
    RunStatus,
    TaskDefinition,
    TaskStage,
    document_selected,
    parse_document_selector,
    reextracted_provenance,
)
from kg_processor.domain.documents import InputFile
from kg_processor.domain.ids import stable_id
from kg_processor.ports.artifact_store import ArtifactStore
from kg_processor.ports.task_store import TaskStore

# What a document's revalidate_document task stores: the document, its entity
# inventory, its context.
_REVALIDATED_INVENTORY_OUTPUT = 1
_REVALIDATED_CONTEXT_OUTPUT = 2


@dataclass
class HeldDocument:
    """Where a graph's copy of one document lives: its text, and its extraction."""

    extracted_run: str
    prepared_run: str


@dataclass
class DocumentLineage:
    """The stored outputs a reprocessed document starts from, by artifact id.

    ``context``, ``inventory`` and ``windows`` are what a revision that extracts
    only entities or only relations again reuses; a document whose extraction
    was itself revalidated has them from the run whose windows it revalidated.
    """

    prepared: str
    extracted: str
    windows_extracted: int
    context: str | None = None
    inventory: str | None = None
    windows: list[dict[str, Any]] | None = None


@dataclass
class RevisionPlan:
    """What a revision keeps, per document, and how to recognise what it holds.

    ``reprocess`` maps each kept document to be revalidated or extracted again
    to its provenance label; every other held document is reused as it is.
    """

    held: dict[str, HeldDocument]
    checksums: dict[str, str]
    dropped: set[str]
    reprocess: dict[str, str] = field(default_factory=dict)
    lineage: dict[str, DocumentLineage] = field(default_factory=dict)
    skipped: list[InputFile] = field(default_factory=list)
    replaced: list[InputFile] = field(default_factory=list)

    @property
    def inherited(self) -> list[InheritedDocuments]:
        """The held documents as the finalizer reads them, one entry per run.

        A reused document is read from the run holding its extraction, and
        from the run holding its text when that is another. A reprocessed one
        contributes only its text: this run stores its extraction.
        """

        files: dict[str, set[str]] = {}
        texts: dict[str, set[str]] = {}
        for file_id, document in self.held.items():
            if file_id in self.reprocess:
                texts.setdefault(document.prepared_run, set()).add(file_id)
                continue
            files.setdefault(document.extracted_run, set()).add(file_id)
            if document.prepared_run != document.extracted_run:
                texts.setdefault(document.prepared_run, set()).add(file_id)
        return [
            InheritedDocuments(
                run_id=run_id,
                file_ids=sorted(files.get(run_id, set())),
                prepared_file_ids=sorted(texts.get(run_id, set())),
            )
            for run_id in sorted({*files, *texts})
        ]

    def reprocessed(self) -> dict[str, list[str]]:
        """The reprocessed documents still held, grouped by provenance label."""

        grouped: dict[str, list[str]] = {}
        for file_id, label in sorted(self.reprocess.items()):
            if file_id in self.held:
                grouped.setdefault(label, []).append(file_id)
        return grouped

    def changes_anything(self) -> bool:
        """Whether the revision would build a graph different from its base."""

        return bool(self.dropped or self.replaced or self.reprocess)

    def without_known(self, files: Iterator[InputFile]) -> Iterator[InputFile]:
        """Skip files the graph already holds; let one with a held identity replace it.

        Equal bytes under any name are the same document, and re-extracting
        them would only cost the fleet. A file with a held identity but other
        bytes is the document updated, so the old version leaves the kept set,
        and with it any reprocessing of the old version.
        """

        for file in files:
            if file.checksum in self.checksums:
                self.skipped.append(file)
                continue
            if self.held.pop(file.id, None) is not None:
                self.replaced.append(file)
            yield file

    def seed_tasks(self, run_id: str, settings: Settings) -> Iterator[TaskDefinition]:
        """One task per reprocessed document, naming the outputs it starts from.

        Each seed is the stage a document's own processing would reach at that
        point, and fans out exactly as that stage does; it only reads its
        input from an earlier run instead of from a dependency in this one.
        """

        for label, file_ids in self.reprocessed().items():
            for file_id in file_ids:
                stage, payload = _seed(label, self.lineage[file_id])
                yield TaskDefinition(
                    id=stable_id("task", run_id, stage.value, file_id),
                    run_id=run_id,
                    stage=stage,
                    scope_id=file_id,
                    payload={**payload, "provenance": label},
                    max_attempts=settings.distributed.max_attempts,
                )


def _seed(label: str, lineage: DocumentLineage) -> tuple[TaskStage, dict[str, Any]]:
    """The stage and payload that start one reprocessed document."""

    if label == PROVENANCE_REVALIDATED:
        return TaskStage.REVALIDATE_DOCUMENT_CONTEXT, {
            "prepared_artifact_id": lineage.prepared,
            "extracted_artifact_id": lineage.extracted,
            # Carried forward so a later revision can extract this document's
            # relations again from what this one revalidated.
            **({"windows": lineage.windows} if lineage.windows is not None else {}),
        }
    if label == reextracted_provenance("relations"):
        return TaskStage.COMPACT_ENTITY_INVENTORY, {
            "prepared_artifact_id": lineage.prepared,
            "inventory_artifact_id": lineage.inventory,
            "windows": lineage.windows,
        }
    if label == reextracted_provenance("entities"):
        return TaskStage.EXTRACT_DOCUMENT_CONTEXT, {
            "prepared_artifact_id": lineage.prepared,
            "context_artifact_id": lineage.context,
        }
    return TaskStage.EXTRACT_DOCUMENT_CONTEXT, {"prepared_artifact_id": lineage.prepared}


def plan_revision(
    revision: RevisionRequest,
    graph_id: str,
    task_store: TaskStore,
    artifact_store: ArtifactStore,
) -> RevisionPlan:
    """Work out what a revision keeps from its base, and what it reprocesses.

    Only a succeeded run of this graph can be revised: its documents are the
    ones whose stage outputs exist. Their identities and byte checksums let the
    source's files be told apart into new, already held, and replaced. Every
    document chosen for reprocessing has its stored outputs found here, so a
    document that lacks one is refused before the run exists.
    """

    try:
        base = task_store.get_run_summary(revision.base_run_id)
    except KeyError as exc:
        raise ValueError(f"unknown base run: {revision.base_run_id}") from exc
    if base.run.status != RunStatus.SUCCEEDED:
        raise ValueError(
            f"only a succeeded run can be revised; {revision.base_run_id} is "
            f"{base.run.status.value}"
        )
    if base.run.graph_id != graph_id:
        raise ValueError(
            f"base run {revision.base_run_id} built graph {base.run.graph_id}, not {graph_id}"
        )
    held = held_documents(revision.base_run_id, task_store, artifact_store)
    unknown = sorted(set(revision.drop_file_ids) - set(held))
    if unknown:
        raise ValueError(f"documents to drop are not in the graph: {', '.join(unknown)}")
    for file_id in revision.drop_file_ids:
        held.pop(file_id, None)
    sources = _held_sources(held, artifact_store)
    checksums = {
        str(source["checksum"]): file_id
        for file_id, source in sources.items()
        if source.get("checksum")
    }
    plan = RevisionPlan(held=held, checksums=checksums, dropped=set(revision.drop_file_ids))
    reextracted = _selected(revision.reextract, held, sources)
    chosen = {file_id: reextracted_provenance(revision.reextract_from) for file_id in reextracted}
    if revision.revalidate:
        chosen.update(
            {file_id: PROVENANCE_REVALIDATED for file_id in held if file_id not in reextracted}
        )
    if chosen:
        lineage = document_lineage(
            {file_id: held[file_id] for file_id in chosen}, task_store, artifact_store
        )
        _require_lineage(chosen, lineage)
        # A document that yielded no windows - one with no readable text - has
        # nothing to extract or validate, and stays as it is.
        plan.reprocess = {
            file_id: label
            for file_id, label in chosen.items()
            if lineage[file_id].windows_extracted > 0
        }
        plan.lineage = {file_id: lineage[file_id] for file_id in plan.reprocess}
    return plan


def held_documents(
    base_run_id: str,
    task_store: TaskStore,
    artifact_store: ArtifactStore,
) -> dict[str, HeldDocument]:
    """Every document a succeeded run's graph holds, and the runs holding it.

    The run's own outputs come first; what it inherited is read from its
    finalizer, which names every concrete run it read from.
    """

    extracted = {
        file_id: base_run_id
        for ref in artifact_store.list_run_artifacts(base_run_id, {ArtifactKind.EXTRACTED_DOCUMENT})
        for file_id in _file_ids_of(ref)
    }
    prepared = {
        file_id: base_run_id
        for ref in artifact_store.list_run_artifacts(base_run_id, {ArtifactKind.PREPARED_DOCUMENT})
        for file_id in _file_ids_of(ref)
    }
    entries = [
        entry
        for finalizer in task_store.get_stage_tasks(base_run_id, TaskStage.FINALIZE_GRAPH)
        for entry in inherited_documents(finalizer.task.payload)
    ]
    for entry in entries:
        for file_id in entry.file_ids:
            extracted.setdefault(file_id, entry.run_id)
    for entry in entries:
        for file_id in entry.prepared_file_ids:
            prepared.setdefault(file_id, entry.run_id)
    for entry in entries:
        for file_id in entry.file_ids:
            prepared.setdefault(file_id, entry.run_id)
    return {
        file_id: HeldDocument(extracted_run=run_id, prepared_run=prepared.get(file_id, run_id))
        for file_id, run_id in extracted.items()
    }


def document_lineage(
    documents: dict[str, HeldDocument],
    task_store: TaskStore,
    artifact_store: ArtifactStore,
) -> dict[str, DocumentLineage]:
    """Find the stored outputs each document starts from, reading each run once.

    The text is the prepared output of the run holding it. The extraction, its
    context, inventory and windows are the outputs of the run holding the
    extraction: from its own context and inventory tasks, or - for a document
    that run revalidated - from its revalidation task, which carried them.
    """

    prepared: dict[str, str] = {}
    for run_id in sorted({document.prepared_run for document in documents.values()}):
        for ref in artifact_store.list_run_artifacts(run_id, {ArtifactKind.PREPARED_DOCUMENT}):
            for file_id in _file_ids_of(ref):
                document = documents.get(file_id)
                if document is not None and document.prepared_run == run_id:
                    prepared[file_id] = ref.id
    lineage: dict[str, DocumentLineage] = {}
    for run_id in sorted({document.extracted_run for document in documents.values()}):
        own = {file_id for file_id, item in documents.items() if item.extracted_run == run_id}
        contexts = _stage_outputs(task_store, run_id, TaskStage.EXTRACT_DOCUMENT_CONTEXT, own)
        inventories = _stage_outputs(task_store, run_id, TaskStage.COMPACT_ENTITY_INVENTORY, own)
        revalidations = _stage_outputs(task_store, run_id, TaskStage.REVALIDATE_DOCUMENT, own)
        for ref in artifact_store.list_run_artifacts(run_id, {ArtifactKind.EXTRACTED_DOCUMENT}):
            file_ids = _file_ids_of(ref)
            if len(file_ids) != 1 or file_ids[0] not in own or file_ids[0] not in prepared:
                continue
            file_id = file_ids[0]
            entry = DocumentLineage(
                prepared=prepared[file_id],
                extracted=ref.id,
                windows_extracted=ref.metadata["windows"],
            )
            if file_id in revalidations:
                payload, outputs = revalidations[file_id]
                entry.context = outputs[_REVALIDATED_CONTEXT_OUTPUT]
                entry.windows = _windows(payload.get("windows"))
                entry.inventory = outputs[_REVALIDATED_INVENTORY_OUTPUT]
            else:
                context_outputs = contexts.get(file_id, ({}, []))[1]
                entry.context = context_outputs[0] if context_outputs else None
                payload, outputs = inventories.get(file_id, ({}, []))
                entry.windows = _windows(payload.get("windows"))
                entry.inventory = outputs[0] if outputs else None
            lineage[file_id] = entry
    return lineage


def _require_lineage(chosen: dict[str, str], lineage: dict[str, DocumentLineage]) -> None:
    """Refuse to plan reprocessing a document whose stored outputs are incomplete."""

    missing: list[str] = []
    for file_id, label in sorted(chosen.items()):
        entry = lineage.get(file_id)
        if entry is None:
            missing.append(f"{file_id} (no stored text or extraction)")
        elif entry.windows_extracted == 0:
            continue
        elif label == reextracted_provenance("entities") and entry.context is None:
            missing.append(f"{file_id} (no stored document context)")
        elif label == reextracted_provenance("relations") and (
            entry.inventory is None or entry.windows is None
        ):
            missing.append(f"{file_id} (no stored entity inventory and windows)")
    if missing:
        raise ValueError(
            "cannot reprocess documents whose earlier outputs are missing: " + ", ".join(missing)
        )


def _stage_outputs(
    task_store: TaskStore,
    run_id: str,
    stage: TaskStage,
    file_ids: set[str],
) -> dict[str, tuple[dict[str, Any], list[str]]]:
    """Each document's task of one stage in a run: its payload and its outputs."""

    return {
        snapshot.task.scope_id: (snapshot.task.payload, snapshot.output_artifact_ids)
        for snapshot in task_store.get_stage_tasks(run_id, stage)
        if snapshot.task.scope_id in file_ids
    }


def _held_sources(
    held: dict[str, HeldDocument],
    artifact_store: ArtifactStore,
) -> dict[str, dict[str, Any]]:
    """The staged source record of every held document, from the run that read it."""

    sources: dict[str, dict[str, Any]] = {}
    for run_id in sorted({document.prepared_run for document in held.values()}):
        for ref in artifact_store.list_run_artifacts(
            run_id, {ArtifactKind.SOURCE_DOCUMENT}, linked=False
        ):
            input_file = ref.metadata.get("input_file")
            if not isinstance(input_file, dict):
                continue
            file_id = str(input_file.get("id", ""))
            document = held.get(file_id)
            if document is not None and document.prepared_run == run_id:
                sources[file_id] = input_file
    return sources


def _selected(
    selectors: list[str],
    held: dict[str, HeldDocument],
    sources: dict[str, dict[str, Any]],
) -> set[str]:
    """The held documents the re-extraction selectors pick.

    A selector that names a document the graph does not keep, or picks no
    document at all, is refused rather than quietly re-extracting less.
    """

    chosen: set[str] = set()
    for selector in selectors:
        kind, value = parse_document_selector(selector)
        if kind == "file" and value not in held:
            raise ValueError(f"documents to re-extract are not kept in the graph: {value}")
        picked = {
            file_id
            for file_id in held
            if document_selected(
                selector,
                file_id,
                _folder(sources.get(file_id, {})),
                _kind(sources.get(file_id, {})),
            )
        }
        if not picked:
            raise ValueError(f"re-extraction selector picks no kept document: {selector}")
        chosen |= picked
    return chosen


def _folder(source: dict[str, Any]) -> str | None:
    """The folder a staged source sits in, relative to its source root."""

    location = source.get("location")
    return str(location).rpartition("/")[0] if isinstance(location, str) else None


def _kind(source: dict[str, Any]) -> str | None:
    """What a staged source is: a document, a dataset, or an image."""

    if not source:
        return None
    return content_kind(
        str(source.get("mime_type") or ""),
        str(source.get("filename") or source.get("source_uri") or ""),
    )


def inherited_documents(payload: dict[str, Any]) -> list[InheritedDocuments]:
    """Read a finalizer's inherited documents, absent on a run that inherits none."""

    entries = payload.get("inherit")
    if not isinstance(entries, list):
        return []
    return [InheritedDocuments.model_validate(entry) for entry in entries]


def _file_ids_of(ref: ArtifactRef) -> list[str]:
    """The documents a stage artifact covers, from the metadata every stage writes."""

    return [str(item) for item in ref.metadata["file_ids"]]


def _windows(value: object) -> list[dict[str, Any]] | None:
    """A stored window list, as the relation fan-out reads it."""

    if not isinstance(value, list):
        return None
    return [dict(item) for item in value if isinstance(item, dict)]
