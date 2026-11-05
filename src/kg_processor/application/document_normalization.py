# SPDX-License-Identifier: Apache-2.0
"""Convert normalized OCR documents into writer-ready document/page/block rows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from kg_processor.application.content_kinds import (
    LOCATION_PAGE_NUMBER,
    content_kind,
    location_text,
)
from kg_processor.domain.documents import InputFile, ParsedAsset, ParsedDocument, ParsedPage
from kg_processor.domain.ids import stable_id


@dataclass(frozen=True)
class DocumentArtifacts:
    """Writer-ready document rows and trace data produced from OCR output."""

    document: dict[str, Any]
    pages: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    assets: list[dict[str, Any]]
    trace_event: dict[str, Any]


def build_document_artifacts(
    graph_id: str,
    file: InputFile,
    parsed: ParsedDocument,
    ocr_cache_id: str,
    cache_hit: bool,
    location: str | None = None,
) -> DocumentArtifacts:
    """Build every document-level artifact row needed by graph writers."""

    return DocumentArtifacts(
        document=document_row(graph_id, file, parsed, location),
        pages=page_rows(graph_id, file, parsed),
        blocks=block_rows(graph_id, file, parsed),
        assets=asset_rows(graph_id, file, parsed),
        trace_event=ocr_trace_event(file, parsed, ocr_cache_id, cache_hit),
    )


def ocr_trace_event(
    file: InputFile,
    parsed: ParsedDocument,
    ocr_cache_id: str,
    cache_hit: bool,
) -> dict[str, Any]:
    """Build a compact OCR trace event for reports and debugging."""

    return {
        "stage": "ocr",
        "file_id": file.id,
        "cache_id": ocr_cache_id,
        "cache_hit": cache_hit,
        "provider": parsed.provider_metadata.get("provider"),
        "pages": len(parsed.pages),
        "assets": len(parsed.assets),
        "blocks": sum(len(page.blocks) for page in parsed.pages),
    }


# A parser's reference to an image it saw, in markdown or HTML: no text of its own.
_IMAGE_REFERENCE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img\b[^>]*>", re.IGNORECASE)


def image_document(file: InputFile) -> ParsedDocument:
    """An image with no text to read: one document, no pages, the image as its asset."""

    return ParsedDocument(
        file_id=file.id,
        checksum=file.checksum,
        source_uri=file.source_uri,
        mime_type=file.mime_type,
        pages=[],
        assets=[_file_image_asset(file, text=False)],
        provider_metadata={"provider": "none", "reason": "image_without_text"},
    )


def as_image_file(file: InputFile, parsed: ParsedDocument) -> ParsedDocument:
    """An image file's parse, keeping the image itself as the document's asset.

    A parser may answer a photograph with nothing but a reference to the
    picture it saw; that is an image without text. An image with text - a
    label, a screenshot - keeps its pages and gains the file as its asset when
    the parser returned none.
    """

    readable = any(
        _IMAGE_REFERENCE_RE.sub("", page.markdown or page.raw_text).strip() for page in parsed.pages
    )
    if not readable:
        return image_document(file)
    if parsed.assets:
        return parsed
    return parsed.model_copy(update={"assets": [_file_image_asset(file, text=True)]})


def _file_image_asset(file: InputFile, *, text: bool) -> ParsedAsset:
    return ParsedAsset(
        id=stable_id("asset", file.id, "image"),
        kind="image",
        uri=file.source_uri,
        metadata={"text": text},
    )


def file_display_name(document: dict[str, Any]) -> str:
    """The file name a document row points at, for prompts and listings."""

    uri = str(document.get("source_uri", ""))
    parsed = urlparse(uri)
    return PurePosixPath(unquote(parsed.path) if parsed.scheme else uri).name


def with_location_page(parsed: ParsedDocument, location: str) -> ParsedDocument:
    """Put the document's location in front of its pages as page 0.

    Page numbers of the document itself are unchanged; page 0 is the file's
    place in the source, so folder and file names can be read and grounded.
    """

    text = location_text(location)
    page = ParsedPage(page_number=LOCATION_PAGE_NUMBER, markdown=text, raw_text=text)
    return parsed.model_copy(update={"pages": [page, *parsed.pages]})


def document_row(
    graph_id: str,
    file: InputFile,
    parsed: ParsedDocument,
    location: str | None = None,
) -> dict[str, Any]:
    """Build the Snowflake-shaped document row for one parsed file."""

    return {
        "id": file.id,
        "graph_id": graph_id,
        "file_id": file.id,
        "checksum": file.checksum,
        "source_uri": file.source_uri,
        "mime_type": file.mime_type,
        "size_bytes": file.size_bytes,
        "ocr_provider": parsed.provider_metadata.get("provider"),
        # What the file is; a dataset's summary and the files beside it are
        # filled in once extraction has profiled it (content_kinds).
        "kind": content_kind(file.mime_type, file.source_uri),
        "summary": None,
        "related_file_ids": [],
        # The folder the file sits in, relative to the source root.
        "folder": location.rpartition("/")[0] if location is not None else None,
        # How this version of the graph came by the document's extraction; a
        # revision relabels what it reused, revalidated or re-extracted.
        "provenance": "extracted",
    }


def page_rows(graph_id: str, file: InputFile, parsed: ParsedDocument) -> list[dict[str, Any]]:
    """Build Snowflake-shaped page rows for one parsed document."""

    return [
        {
            "id": stable_id("page", graph_id, file.id, page.page_number),
            "graph_id": graph_id,
            "file_id": file.id,
            "page_number": page.page_number,
            "markdown": page.markdown,
            "raw_text": page.raw_text,
            "detected_language": page.detected_language,
        }
        for page in parsed.pages
    ]


def block_rows(graph_id: str, file: InputFile, parsed: ParsedDocument) -> list[dict[str, Any]]:
    """Build Snowflake-shaped OCR block rows for one parsed document."""

    return [
        {
            "id": block.id,
            "graph_id": graph_id,
            "file_id": file.id,
            "page_number": page.page_number,
            "kind": block.kind,
            "text": block.text,
            "bbox": list(block.bbox) if block.bbox is not None else None,
            "metadata": block.metadata,
        }
        for page in parsed.pages
        for block in page.blocks
    ]


def asset_rows(graph_id: str, file: InputFile, parsed: ParsedDocument) -> list[dict[str, Any]]:
    """Build Snowflake-shaped asset rows for one parsed document."""

    return [
        {
            "id": asset.id,
            "graph_id": graph_id,
            "file_id": file.id,
            "kind": asset.kind,
            "page_number": asset.page_number,
            "uri": asset.uri,
            "metadata": asset.metadata,
        }
        for asset in parsed.assets
    ]
