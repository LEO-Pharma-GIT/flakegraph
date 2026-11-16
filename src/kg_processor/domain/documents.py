# SPDX-License-Identifier: Apache-2.0
"""Canonical document models shared by every OCR and file-source provider.

Provider adapters are intentionally forced to normalize into these shapes before
the graph pipeline sees anything. That keeps MinerU, Cortex, Tesseract, and
other OCR services replaceable without creating provider-specific branches in
chunking, extraction, or writers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class InputFile(BaseModel):
    """A locally readable document plus immutable provenance from its source."""

    id: str
    path: Path
    source_uri: str
    # A content hash with one definition across every source, so the same bytes
    # carry the same identity whichever runtime ingested them.
    checksum: str
    mime_type: str
    size_bytes: int
    # Whatever the storage provider reported, when it uses its own algorithm.
    # Retained for provenance and never used as identity.
    provider_checksum: str | None = None
    # The file's folders and name below its source root, fixed where it was
    # discovered: a worker that reads a staged copy knows no source root.
    location: str | None = None


class SourceListing(BaseModel):
    """One object a source would offer, as a listing names it - never fetched."""

    uri: str
    name: str
    size_bytes: int | None = None
    modified_at: str | None = None
    checksum: str | None = None


class LayoutBlock(BaseModel):
    """A normalized layout or text block emitted by an OCR provider."""

    id: str
    page_number: int
    kind: str
    text: str = ""
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedPage(BaseModel):
    """Page-level OCR output used by chunking and audit tables."""

    page_number: int
    markdown: str
    raw_text: str
    blocks: list[LayoutBlock] = Field(default_factory=list)
    detected_language: str | None = None


class ParsedAsset(BaseModel):
    """Non-text artifact detected during OCR, such as figures or extracted images."""

    id: str
    kind: str
    page_number: int | None = None
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Provider-normalized OCR result for one input document."""

    file_id: str
    checksum: str
    source_uri: str
    mime_type: str
    pages: list[ParsedPage]
    assets: list[ParsedAsset] = Field(default_factory=list)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
