# SPDX-License-Identifier: Apache-2.0
"""Build deterministic, document-scoped windows for graph extraction."""

from __future__ import annotations

from collections import defaultdict

from kg_processor.application.bibliography import is_reference_text
from kg_processor.application.formula import is_display_math_text
from kg_processor.domain.extraction import ExtractionWindow
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ids import stable_id


def build_extraction_windows(
    chunks: list[Chunk],
    target_tokens: int,
    max_chunks: int,
) -> list[ExtractionWindow]:
    """Group contiguous adjacent chunks without mixing source documents.

    A model may use nearby context inside one document, but unrelated files
    must not share a prompt because that would permit unsupported cross-document
    relations. A gap in ``chunk_index`` is also a hard boundary: distributed
    workers may receive a subset after reference-only windows were removed, and
    joining chunks on opposite sides of that gap would reconstruct a different
    logical window than the coordinator originally scheduled.
    """

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    by_document: dict[str, list[Chunk]] = defaultdict(list)
    document_order: list[str] = []
    for chunk in chunks:
        if chunk.document_id not in by_document:
            document_order.append(chunk.document_id)
        by_document[chunk.document_id].append(chunk)

    windows: list[ExtractionWindow] = []
    for document_id in document_order:
        ordered = sorted(by_document[document_id], key=lambda item: item.chunk_index)
        current: list[Chunk] = []
        current_tokens = 0
        for chunk in ordered:
            exceeds_target = current and current_tokens + chunk.token_count > target_tokens
            reaches_chunk_limit = len(current) >= max_chunks
            crosses_index_gap = bool(current) and chunk.chunk_index != current[-1].chunk_index + 1
            if exceeds_target or reaches_chunk_limit or crosses_index_gap:
                windows.append(_window(document_id, current))
                current = []
                current_tokens = 0
            current.append(chunk)
            current_tokens += chunk.token_count
        if current:
            windows.append(_window(document_id, current))
    return windows


def build_document_context_windows(
    chunks: list[Chunk],
    target_tokens: int,
    max_chunks: int,
) -> list[ExtractionWindow]:
    """Select one bounded front-matter window for every source document.

    Document-context extraction needs the title area once, while ordinary body
    windows remain independently queueable. The selection follows source order,
    never crosses a document boundary, and does not split chunks, preserving the
    same provenance records used by later relation evidence.
    """

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    by_document: dict[str, list[Chunk]] = defaultdict(list)
    document_order: list[str] = []
    for chunk in chunks:
        if chunk.document_id not in by_document:
            document_order.append(chunk.document_id)
        by_document[chunk.document_id].append(chunk)

    windows: list[ExtractionWindow] = []
    for document_id in document_order:
        selected: list[Chunk] = []
        selected_tokens = 0
        for chunk in sorted(by_document[document_id], key=lambda item: item.chunk_index):
            if selected and (
                len(selected) >= max_chunks or selected_tokens + chunk.token_count > target_tokens
            ):
                break
            selected.append(chunk)
            selected_tokens += chunk.token_count
        if selected:
            base = _window(document_id, selected)
            windows.append(
                base.model_copy(update={"id": stable_id("document_context_window", document_id)})
            )
    return windows


# Characters of a document's opening shown with every verification request.
DOCUMENT_OPENING_CHARACTERS = 1500


def build_document_openings(
    chunks: list[Chunk],
    max_characters: int = DOCUMENT_OPENING_CHARACTERS,
) -> dict[str, str]:
    """Return the first characters of each document, read from its first chunks.

    A certificate or data sheet names its subject once, at the top, and every
    row below is about it; a window far down the document does not repeat it.
    Chunks are read in order, the overlap of chunks on one page once, until the
    limit - so a short location page is followed by the text it places.
    """

    by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_document[chunk.document_id].append(chunk)
    openings: dict[str, str] = {}
    for document_id, document_chunks in by_document.items():
        parts: list[str] = []
        length = 0
        previous: Chunk | None = None
        for chunk in sorted(document_chunks, key=lambda item: item.chunk_index):
            if length >= max_characters:
                break
            content = chunk.content
            if previous is not None and previous.page_number == chunk.page_number:
                content = content[max(0, previous.end_offset - chunk.start_offset) :]
            elif parts:
                content = "\n" + content
            parts.append(content[: max_characters - length])
            length += len(parts[-1])
            previous = chunk
        openings[document_id] = "".join(parts).strip()
    return openings


def _window(document_id: str, chunks: list[Chunk]) -> ExtractionWindow:
    """Create a stable window from ordered chunks belonging to one document.

    Copying the chunk list prevents later caller mutation, while identity and
    token count remain deterministic inputs to extraction and caching.
    """

    return ExtractionWindow(
        id=stable_id("extraction_window", document_id, *(chunk.id for chunk in chunks)),
        document_id=document_id,
        chunks=list(chunks),
        token_count=sum(chunk.token_count for chunk in chunks),
    )


def window_skip_reason(window: ExtractionWindow) -> str | None:
    """Name why a window holds nothing to extract, or ``None`` when it does.

    A bibliography yields citation metadata, not entities; a page of
    derivations yields symbols whose quotes ground in nothing. Either would
    cost a model call and, for the equations, fail the window outright when
    every record it returned was discarded.
    """

    content = "\n".join(chunk.content for chunk in window.chunks)
    if is_reference_text(content):
        return "bibliography_only"
    if is_display_math_text(content):
        return "display_math_only"
    return None
