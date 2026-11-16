# SPDX-License-Identifier: Apache-2.0
"""Pipeline cache port.

Cache keys are behavior-based and exclude secrets. That makes local filesystem
caches and Snowflake cache tables interchangeable without letting credential
rotation invalidate deterministic OCR/extraction results.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from kg_processor.domain.documents import ParsedDocument
from kg_processor.domain.graph import ExtractionResult


class OcrCacheKey(BaseModel):
    """Deterministic cache key for normalized OCR documents."""

    id: str
    file_id: str
    checksum: str
    ocr_provider: str
    options_hash: str


class ExtractionCacheKey(BaseModel):
    """Deterministic cache key for LLM extraction results over a chunk batch."""

    id: str
    graph_id: str
    chunk_batch_hash: str
    llm_provider: str
    model: str
    options_hash: str


class EnrichmentCacheKey(BaseModel):
    """Deterministic key for description and community LLM results."""

    id: str
    graph_id: str
    stage: str
    llm_provider: str
    model: str
    options_hash: str


class PipelineCache(Protocol):
    """Cache interface shared by local JSON files and Snowflake tables."""

    def get_ocr_document(self, key: OcrCacheKey) -> ParsedDocument | None:
        """Return a cached OCR document when the exact behavior key is present."""
        ...

    def put_ocr_document(self, key: OcrCacheKey, document: ParsedDocument) -> None:
        """Persist a normalized OCR document under its deterministic cache key."""
        ...

    def get_extraction_result(self, key: ExtractionCacheKey) -> ExtractionResult | None:
        """Return a cached LLM extraction result for an identical chunk batch."""
        ...

    def put_extraction_result(self, key: ExtractionCacheKey, result: ExtractionResult) -> None:
        """Persist an LLM extraction result under its deterministic cache key."""
        ...

    def get_enrichment_result(self, key: EnrichmentCacheKey) -> dict[str, object] | None:
        """Return a cached normalized enrichment result when present."""
        ...

    def put_enrichment_result(
        self,
        key: EnrichmentCacheKey,
        result: dict[str, object],
    ) -> None:
        """Persist one normalized description or community result."""
        ...
