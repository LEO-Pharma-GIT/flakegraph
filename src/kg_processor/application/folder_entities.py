# SPDX-License-Identifier: Apache-2.0
"""Tie each document to the entity its folder names, and everything it states to that.

An ontology's ``folder_entity`` says the folder a file is filed in identifies
what the file belongs to. The document-context pass gives each document one
mention of that entity, grounded in the folder line of its location page. The
mention's name is the folder's own name at the definition's level, so every
document of one folder - and of every folder of that name at that level -
names the same entity, however a model would phrase it: the level is the one
whose folder names identify the thing. A model call only describes it, and the
description keeps the folder's whole path.

Finalization links every entity of a document to the document's folder entity
with the folder relation, on the same folder line, and leaves out the documents
the definition excludes. Both finalizers do this from the stored extraction, so
changing the exclusions needs no extraction again.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from kg_processor.application.content_kinds import LOCATION_PAGE_NUMBER
from kg_processor.application.document_normalization import file_display_name
from kg_processor.application.extraction_contracts import folder_entity_request
from kg_processor.application.graph_merge import normalize_entity_name
from kg_processor.application.ordered_map import ordered_map
from kg_processor.application.progress import error_metadata
from kg_processor.application.window_errors import is_systemic_provider_error
from kg_processor.domain.extraction import EntityMention, ExtractionObservations
from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractedRelation
from kg_processor.domain.ids import stable_id
from kg_processor.domain.ontology import (
    FolderEntityDefinition,
    OntologyProfile,
)
from kg_processor.ports.llm import RefusedChatResponseError, StructuredCompletionProvider

FOLDER_EVIDENCE_METHOD = "folder"
FOLDER_ENTITY_STAGE = "folder_entity"
FOLDER_LINK_DESCRIPTION = "Found in a document filed in the folder {folder}."
# The location page's folder line, as location_text writes it.
_FOLDER_LINE = re.compile(r"(?:^|(?<=\s))Folder: ")
# A folder of a path: what lies between its separators.
_FOLDER_SEGMENT = re.compile(r"[^/\\]+")


def is_folder_mention(mention: EntityMention | ExtractedEntity) -> bool:
    """Whether a mention is a document's folder entity rather than something it states."""

    return mention.evidence_method == FOLDER_EVIDENCE_METHOD


def document_location(document: Mapping[str, Any]) -> str:
    """A document's place below its source root, folders and file name."""

    folder = str(document.get("folder") or "").strip("/")
    name = file_display_name(dict(document))
    return f"{folder}/{name}" if folder else name


def excluded_file_ids(
    definition: FolderEntityDefinition, documents: Iterable[Mapping[str, Any]]
) -> set[str]:
    """The documents, by file id, the definition ties to no folder."""

    return {
        str(document.get("file_id"))
        for document in documents
        if definition.excludes(str(document.get("file_id")), document_location(document))
    }


def folder_link_description(folder: str) -> str:
    """What one folder link says, the same for every entity of the folder's documents."""

    return FOLDER_LINK_DESCRIPTION.format(folder=folder)


def folder_entity_mentions(  # noqa: PLR0913 - one context pass's provider settings.
    chunks: Sequence[Chunk],
    documents: Mapping[str, Mapping[str, Any]],
    ontology: OntologyProfile,
    llm: StructuredCompletionProvider,
    *,
    model: str,
    timeout_seconds: int,
    seed: int,
    parallelism: int,
) -> tuple[list[EntityMention], list[dict[str, Any]]]:
    """Give each document one mention of the entity its folder names.

    The mention's name is the last folder of the path at the definition's
    level, and its evidence that folder's name on the location page's folder
    line; the description names the whole path. One call per document
    describes it from the folder, the file name, and the document's opening; a failed,
    refused, or empty answer leaves the description to the folder alone, so
    no document is left untied. A document with no folder at that level, or
    no location page, gets no mention.
    """

    definition = ontology.folder_entity
    if definition is None:
        return [], []
    entity_type = next(
        item for item in ontology.entity_types if item.name == definition.entity_type
    )
    by_document: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.file_id, []).append(chunk)

    def mention_for(
        _index: int, item: tuple[str, list[Chunk]]
    ) -> tuple[EntityMention | None, dict[str, Any]]:
        file_id, document_chunks = item
        document = documents.get(file_id, {})
        event: dict[str, Any] = {"stage": FOLDER_ENTITY_STAGE, "document_id": file_id}
        folder = definition.folder_at_level(document.get("folder"))
        if folder is None:
            return None, {**event, "status": "skipped", "reason": "no_folder_at_level"}
        anchor = _folder_anchor(
            document_chunks, str(document.get("folder") or ""), definition.level
        )
        if isinstance(anchor, str):
            return None, {**event, "status": "skipped", "reason": anchor}
        chunk, start, end = anchor
        opening = next(
            (item for item in document_chunks if item.page_number != LOCATION_PAGE_NUMBER), None
        )
        event["folder"] = folder
        try:
            completion = llm.complete_structured(
                folder_entity_request(
                    folder,
                    file_display_name(dict(document)),
                    entity_type,
                    opening,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    seed=seed,
                )
            )
        except RefusedChatResponseError as exc:
            named: tuple[str, str] | None = None
            event.update({"status": "folder_name", "reason": "refused", **error_metadata(exc)})
        except Exception as exc:
            if is_systemic_provider_error(exc):
                raise
            named = None
            event.update({"status": "folder_name", "reason": "error", **error_metadata(exc)})
        else:
            named = _named(completion.payload)
            event.update(
                {
                    "status": "named" if named else "folder_name",
                    **({"name": named[0]} if named else {"reason": "no_name"}),
                    **completion.provider_metadata,
                }
            )
        mention = EntityMention(
            id=stable_id("folder_entity", chunk.id, folder),
            name=chunk.content[start:end],
            type=definition.entity_type,
            description=_description(folder, definition.entity_type, named),
            source_chunk_id=chunk.id,
            quote=chunk.content[start:end],
            is_document_context=True,
            evidence_method="folder",
            start_offset=start,
            end_offset=end,
        )
        return mention, event

    results = ordered_map(
        list(by_document.items()),
        mention_for,
        parallelism=max(1, parallelism),
    )
    return (
        [mention for mention, _event in results if mention is not None],
        [event for _mention, event in results],
    )


def without_excluded_folder_mentions(
    observations: ExtractionObservations,
    chunks: Sequence[Chunk],
    excluded: set[str],
) -> ExtractionObservations:
    """Leave out the folder mentions of documents tied to no folder."""

    if not excluded:
        return observations
    file_by_chunk = {chunk.id: chunk.file_id for chunk in chunks}
    return observations.model_copy(
        update={
            "entities": [
                mention
                for mention in observations.entities
                if not (
                    is_folder_mention(mention)
                    and file_by_chunk.get(mention.source_chunk_id) in excluded
                )
            ]
        }
    )


def folder_links(
    entities: Sequence[ExtractedEntity],
    chunks_by_id: Mapping[str, Chunk],
    ontology: OntologyProfile,
    excluded: set[str],
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    """Drop excluded documents' folder entities and link every other entity to its folder's.

    Each entity of a document gets one link to the document's folder entity,
    whichever of its mentions in the document is read, with the folder line
    as evidence. An entity resolved into the folder entity itself gets none.
    """

    definition = ontology.folder_entity
    if definition is None:
        return list(entities), []
    relation = ontology.relation(definition.relation_type)
    relation_type = relation.name if relation is not None else definition.relation_type

    def file_of(entity: ExtractedEntity) -> str | None:
        chunk = chunks_by_id.get(entity.source_chunk_id)
        return chunk.file_id if chunk is not None else None

    kept = [
        entity
        for entity in entities
        if not (is_folder_mention(entity) and file_of(entity) in excluded)
    ]
    folder_by_file: dict[str, ExtractedEntity] = {}
    for entity in kept:
        file_id = file_of(entity)
        if is_folder_mention(entity) and file_id is not None:
            folder_by_file.setdefault(file_id, entity)
    links: list[ExtractedRelation] = []
    seen: set[tuple[object, object, str]] = set()
    for entity in kept:
        file_id = file_of(entity)
        folder = folder_by_file.get(file_id) if file_id is not None else None
        if folder is None or is_folder_mention(entity):
            continue
        source, target = _identity(entity), _identity(folder)
        key = (source, target, folder.source_chunk_id)
        if source == target or key in seen:
            continue
        seen.add(key)
        links.append(
            ExtractedRelation(
                source_name=entity.name,
                target_name=folder.name,
                source_component_id=entity.resolved_component_id,
                target_component_id=folder.resolved_component_id,
                source_type=entity.type,
                target_type=folder.type,
                relation_type=relation_type,
                description=folder_link_description(folder.quote or folder.name),
                source_chunk_id=folder.source_chunk_id,
                quote=folder.quote,
                start_offset=folder.start_offset,
                end_offset=folder.end_offset,
                weight=1.0,
                confidence=1.0,
                evidence_method=FOLDER_EVIDENCE_METHOD,
            )
        )
    return kept, links


def _identity(entity: ExtractedEntity) -> object:
    """The canonical node an entity becomes: its resolved component, else name and type."""

    if entity.resolved_component_id:
        return entity.resolved_component_id
    return (normalize_entity_name(entity.name), entity.type)


def _folder_anchor(
    chunks: Sequence[Chunk], folder: str, level: int
) -> tuple[Chunk, int, int] | str:
    """Where the folder's name at ``level`` is written on the location page's folder line.

    The page is read back from its chunks by their offsets, so the line may
    span chunks; the name itself must lie in one, its evidence. Otherwise the
    reason the document gets no folder entity: no location page, none with
    the folder line, or a folder name longer than a chunk or cut by one.
    """

    page = [chunk for chunk in chunks if chunk.page_number == LOCATION_PAGE_NUMBER]
    if not page:
        return "no_location_page"
    text = [" "] * max(chunk.end_offset for chunk in page)
    for chunk in page:
        text[chunk.start_offset : chunk.end_offset] = chunk.content
    line = _FOLDER_LINE.search("".join(text))
    segments = [(match.start(), match.end()) for match in _FOLDER_SEGMENT.finditer(folder)]
    if line is None or len(segments) < level:
        return "no_folder_line"
    start, end = (line.end() + offset for offset in segments[level - 1])
    name = folder[segments[level - 1][0] : segments[level - 1][1]]
    for chunk in page:
        local = start - chunk.start_offset
        if (
            chunk.start_offset <= start
            and end <= chunk.end_offset
            and chunk.content[local : local + len(name)] == name
        ):
            return chunk, local, local + len(name)
    return "folder_name_split"


def _named(payload: Mapping[str, Any]) -> tuple[str, str] | None:
    """The clean name and description a folder answer gives, when it gives a name."""

    name = " ".join(str(payload.get("name") or "").split())
    description = " ".join(str(payload.get("description") or "").split())
    return (name, description) if name else None


def _description(folder: str, entity_type: str, named: tuple[str, str] | None) -> str:
    """The folder entity's description: the model's sentence with its clean name, and the path."""

    if named is None:
        return f"The {entity_type} whose documents are filed in the folder {folder}."
    name, sentence = named
    if not sentence:
        return f"{name}, whose documents are filed in the folder {folder}."
    described = sentence if name.casefold() in sentence.casefold() else f"{name}: {sentence}"
    return f"{described} Its documents are filed in the folder {folder}."
