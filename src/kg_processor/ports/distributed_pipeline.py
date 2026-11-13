# SPDX-License-Identifier: Apache-2.0
"""Application-stage port consumed by the durable distributed worker."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol

from kg_processor.domain.documents import InputFile
from kg_processor.domain.extraction import ColumnRelation, EntityMention, ExtractionObservations
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.domain.stages import (
    DocumentEntityInventoryShard,
    DocumentRevalidationShard,
    EntityWindowShard,
    ExtractedDocumentShard,
    PreparedDocumentShard,
    RelationWindowShard,
    RevalidatedRelationWindowShard,
    RevalidationWindow,
    RevalidationWindowShard,
)


class DistributedPipeline(Protocol):
    """Expose only the coarse serializable stages a distributed worker may execute."""

    def stage_graph_publication(
        self, batch: GraphWriteBatch, publication_id: str, generation: int
    ) -> AbstractContextManager[Callable[[], None]]:
        """Stage a finalized batch and expose its destination commit separately."""
        ...

    def prepare_documents(
        self,
        files: list[InputFile],
        *,
        record_failures: bool = True,
    ) -> PreparedDocumentShard:
        """OCR, normalize, and chunk an explicit source selection.

        With ``record_failures`` a document that cannot be read is recorded in
        the shard's trace and skipped; without it the failure is raised, for a
        caller that still has attempts left to retry it.
        """
        ...

    def extract_window_entities(
        self,
        prepared: PreparedDocumentShard,
    ) -> EntityWindowShard:
        """Extract unresolved entity mentions from one bounded chunk window."""
        ...

    def extract_window_relations(
        self,
        prepared: PreparedDocumentShard,
        document_entities: list[EntityMention],
        document_openings: dict[str, str] | None = None,
        *,
        dataset_columns: list[ColumnRelation] | None = None,
    ) -> RelationWindowShard:
        """Extract grounded relations against a document-wide entity inventory.

        ``document_openings`` gives each document's first characters, which
        verification reads with every window of it. A catalogue dataset's
        window reads its rows by ``dataset_columns``.
        """
        ...

    def extract_document_context(
        self,
        prepared: PreparedDocumentShard,
    ) -> PreparedDocumentShard:
        """Extract reusable document-context mentions from bounded front matter."""
        ...

    def begin_revalidation(
        self,
        prepared: PreparedDocumentShard,
        observations: ExtractionObservations,
    ) -> tuple[DocumentRevalidationShard, list[RevalidationWindow]]:
        """Validate a document's stored context records and place the rest by window."""
        ...

    def revalidate_entity_window(self, window: RevalidationWindowShard) -> EntityWindowShard:
        """Replay one window's stored entity records through the current rules."""
        ...

    def revalidate_relation_window(
        self,
        window: RevalidationWindowShard,
        inventory: DocumentEntityInventoryShard,
    ) -> RevalidatedRelationWindowShard:
        """Replay and verify one window's stored relations against the document entities."""
        ...

    def finalize_document_shards(
        self,
        shards: list[ExtractedDocumentShard],
        *,
        write: bool = True,
    ) -> GraphWriteBatch:
        """Resolve the complete corpus, validate it, and optionally publish it."""
        ...
