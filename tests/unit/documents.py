"""Build the chunks, windows and grounded spans the extraction tests share.

Every default is self-consistent: the offsets cover the content, the token
count is what whitespace splitting gives, and the document id falls back to
the file id the way ``Chunk`` itself does. A test passes only what it asserts.
"""

from __future__ import annotations

from typing import Any

from kg_processor.domain.extraction import ExtractionWindow
from kg_processor.domain.graph import Chunk


def chunk(
    content: str,
    *,
    chunk_id: str = "chunk_1",
    file_id: str = "file_1",
    document_id: str = "",
    start: int = 0,
    content_hash: str = "hash",
    **overrides: Any,
) -> Chunk:
    """Return one chunk whose offsets and token count fit ``content``."""

    fields: dict[str, Any] = {
        "id": chunk_id,
        "file_id": file_id,
        "document_id": document_id,
        "page_number": 1,
        "chunk_index": 0,
        "content": content,
        "start_offset": start,
        "end_offset": start + len(content),
        "token_count": max(1, len(content.split())),
        "content_hash": content_hash,
    }
    return Chunk(**{**fields, **overrides})


def window(*chunks: Chunk, window_id: str = "window_1") -> ExtractionWindow:
    """Wrap one document's chunks in the window an extractor is handed."""

    return ExtractionWindow(
        id=window_id,
        document_id=chunks[0].document_id,
        chunks=list(chunks),
        token_count=sum(item.token_count for item in chunks),
    )


def grounded(content: str, quote: str) -> dict[str, str | int]:
    """Ground a fixture on the real span of ``quote`` so assembly never has to search."""

    start = content.index(quote)
    return {"quote": quote, "start_offset": start, "end_offset": start + len(quote)}
