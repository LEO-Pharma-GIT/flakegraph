# SPDX-License-Identifier: Apache-2.0
"""Serializable contracts exchanged between distributed pipeline stages.

Stage artifacts contain domain data, never provider clients, database handles, or
orchestrator state. The same models can therefore cross PostgreSQL, object storage,
or an in-process local executor without changing graph semantics.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from kg_processor.domain.extraction import (
    ColumnRelation,
    EntityMention,
    ExtractionObservations,
    ExtractionWindow,
    RelationObservation,
)
from kg_processor.domain.graph import Chunk, RelationAssertion


class PreparedDocumentShard(BaseModel):
    """Hold normalized document artifacts produced by one preparation task.

    A shard normally represents one source file. Lists remain part of the contract
    so local execution can use the identical stage with a small document batch.
    Chunk embeddings are intentionally absent until the extraction stage, allowing
    OCR workers to run without loading an embedding model.
    """

    file_ids: list[str]
    files_seen: int = Field(ge=0)
    documents_processed: int = Field(ge=0)
    document_rows: list[dict[str, Any]] = Field(default_factory=list)
    page_rows: list[dict[str, Any]] = Field(default_factory=list)
    block_rows: list[dict[str, Any]] = Field(default_factory=list)
    asset_rows: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    document_context_entities: list[EntityMention] = Field(default_factory=list)
    ocr_cache_hits: int = Field(default=0, ge=0)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class DocumentContextShard(BaseModel):
    """Carry one document's reusable context without repeated prepared content.

    Every extraction window needs focal document entities, but it does not need
    normalized pages, layout blocks, assets, or unrelated chunks. This compact
    contract prevents task fan-out from multiplying document transfer volume and
    retains the exact context trace for auditability.
    """

    file_ids: list[str]
    document_context_entities: list[EntityMention] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    # Each document's first characters, by document id: verification reads a
    # window with the header that names what the document is about.
    document_openings: dict[str, str]


class ExtractionWindowShard(BaseModel):
    """Carry the chunks of one extraction window, which one task extracts.

    The task's dependency on ``DocumentContextShard`` still enforces stage order.
    Separating bounded chunk bytes from reusable context avoids downloading an
    entire prepared document for every window in a long source file.
    """

    file_ids: list[str]
    chunks: list[Chunk]
    # False for a dataset whose profile asks for its entities only: the
    # window's relation task then records the skip instead of calling a model.
    extract_relations: bool = True
    # A catalogue dataset's column relations, which its relation task reads
    # the rows by.
    dataset_columns: list[ColumnRelation] = Field(default_factory=list)


class EntityWindowShard(BaseModel):
    """Hold entity observations from one independently processed text window.

    Only chunk identities are retained because the immutable extraction-window
    artifact already owns source text. This keeps durable fan-out proportional to
    extracted observations instead of copying complete prepared-document payloads.
    """

    file_ids: list[str]
    chunk_ids: list[str]
    entities: list[EntityMention] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class DocumentEntityInventoryShard(BaseModel):
    """Provide one deduplicated document-wide entity vocabulary to relation tasks.

    The inventory is the barrier between two parallel queue waves. Relation windows
    may therefore connect endpoints discovered anywhere in the same document while
    documents and windows remain independently schedulable across a worker fleet.
    """

    file_ids: list[str]
    entities: list[EntityMention] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    chunk_count: int = Field(default=0, ge=0)
    window_count: int = Field(default=0, ge=0)
    # Carried on from the document context, so every relation task has it.
    document_openings: dict[str, str]


class RelationWindowShard(BaseModel):
    """Hold relation observations from one document-inventory-aware window.

    The inventory itself is omitted: every relation task reads the same
    immutable inventory, and document compaction persists it once. Only the
    endpoints a relation response added for things the inventory lacked travel
    here, and compaction adds them to it.
    """

    file_ids: list[str]
    chunk_ids: list[str]
    relations: list[RelationObservation] = Field(default_factory=list)
    held_relations: list[RelationObservation] = Field(default_factory=list)
    entities: list[EntityMention] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class StoredEntity(BaseModel):
    """An entity candidate rebuilt from what a document's extraction stored."""

    name: str
    type: str
    chunk_id: str
    quote: str
    description: str = ""
    confidence: float
    aliases: list[str] = Field(default_factory=list)
    accepted: bool = False

    def candidate(self) -> dict[str, Any]:
        """The record as an entity response would carry it."""

        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "source_chunk_id": self.chunk_id,
            "quote": self.quote,
            "confidence": self.confidence,
            "aliases": list(self.aliases),
        }


class StoredRelation(BaseModel):
    """A relation candidate rebuilt from what a document's extraction stored.

    Endpoints are named rather than identified: ids belong to the mentions of
    the run that stored the record, and revalidation resolves them afresh.
    ``source_id`` and ``target_id`` are the stored mention ids of an accepted
    relation, tried first.
    """

    relation_type: str
    source: str
    source_type: str
    target: str
    target_type: str
    chunk_id: str
    quote: str
    source_surface: str = ""
    target_surface: str = ""
    description: str = ""
    confidence: float
    quantities: list[dict[str, Any]] = Field(default_factory=list)
    assertion: RelationAssertion = "reported"
    accepted: bool = False
    source_id: str | None = None
    target_id: str | None = None


class RevalidationWindow(BaseModel):
    """One window of a revalidated document and the stored records that fall in it.

    The records are replayed as the window's extraction answers. A window of a
    dataset whose profile asks for entities only replays no relations.
    ``stored_mentions`` are the mentions the stored extraction accepted that
    this window's relations name by id, so an endpoint whose mention changed
    id is still found by the name it had.
    """

    window: ExtractionWindow
    entities: list[StoredEntity] = Field(default_factory=list)
    relations: list[StoredRelation] = Field(default_factory=list)
    extract_relations: bool = True
    stored_mentions: list[EntityMention] = Field(default_factory=list)


class RevalidationWindowShard(BaseModel):
    """Carry the one window a revalidation task replays.

    The window brings its own chunks and records, so a window task never
    reads the document's prepared text or its complete stored extraction.
    """

    file_ids: list[str]
    window: RevalidationWindow


class DocumentRevalidationShard(BaseModel):
    """What revalidating a document settles before its windows are replayed.

    The document-context candidates are validated here, once for the whole
    document, and every stored record is placed in the window that holds it.
    ``rejected_records`` lists the records that could not be put to the rules
    at all - a record in a window the current rules skip - under the rule's
    reason; ``reconstruction`` counts how the stored records were rebuilt. Both
    reach the document's ``revalidation`` event when its windows are combined.
    """

    file_ids: list[str]
    document_id: str = ""
    document_context_entities: list[EntityMention] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    rejected_records: list[dict[str, Any]] = Field(default_factory=list)
    reconstruction: dict[str, int] = Field(default_factory=dict)
    window_count: int = Field(default=0, ge=0)
    document_openings: dict[str, str]


class RevalidatedRelationWindowShard(RelationWindowShard):
    """Hold one window's revalidated relations, with what its replay lost.

    Beside what a relation window yields, a revalidated one lists the stored
    relations it could not put to the rules and counts the stored endpoints
    it did and did not find among the document's current entities.
    """

    rejected_records: list[dict[str, Any]] = Field(default_factory=list)
    endpoints_resolved: int = Field(default=0, ge=0)
    endpoints_unresolved: int = Field(default=0, ge=0)


class ExtractedDocumentShard(BaseModel):
    """Hold one prepared shard and its unresolved grounded observations.

    Keeping mention identity unresolved is essential: workers may extract separate
    documents concurrently, but graph-wide resolution must see every mention before
    choosing canonical entities. Local execution nests the complete prepared shard.
    Spark execution stores only its file-level projection because the scalable
    finalizer reads run-scoped prepared artifacts directly, avoiding a second copy
    of every OCR page, block, asset, and chunk in object storage.
    """

    prepared: PreparedDocumentShard
    observations: ExtractionObservations
    trace: list[dict[str, Any]] = Field(default_factory=list)


def combine_extracted_shards(shards: list[ExtractedDocumentShard]) -> ExtractionObservations:
    """Combine worker outputs deterministically before graph-wide resolution.

    Sorting by each shard's first file id removes task completion order from graph
    identity, trace ordering, and persisted output. Duplicate mention and relation
    observations are conservatively removed by the resolution stage itself.
    """

    ordered = sorted(shards, key=_shard_order_key)
    return ExtractionObservations(
        entities=[
            entity
            for shard in ordered
            for entity in [
                *shard.prepared.document_context_entities,
                *shard.observations.entities,
            ]
        ],
        relations=[relation for shard in ordered for relation in shard.observations.relations],
        held_relations=[
            relation for shard in ordered for relation in shard.observations.held_relations
        ],
        trace=[event for shard in ordered for event in shard.observations.trace],
        chunk_count=sum(shard.observations.chunk_count for shard in ordered),
        window_count=sum(shard.observations.window_count for shard in ordered),
    )


def _shard_order_key(shard: ExtractedDocumentShard) -> tuple[str, ...]:
    """Return a stable ordering key even for a defensive empty-file shard."""

    return tuple(sorted(shard.prepared.file_ids))
