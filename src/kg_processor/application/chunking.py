# SPDX-License-Identifier: Apache-2.0
"""Deterministic document chunking.

Chunk ids include graph/file/page/chunk position and a text hash so reruns are
stable while still detecting changed OCR output. The chunker keeps page/block
provenance because every extracted graph observation must cite its source text.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right

from kg_processor.domain.documents import LayoutBlock, ParsedDocument
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ids import normalize_text, sha256_hex, stable_id

_TOKEN_RE = re.compile(r"\S+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def chunk_document(
    document: ParsedDocument,
    graph_id: str,
    chunk_token_size: int,
    chunk_token_overlap: int,
) -> list[Chunk]:
    """Split a parsed document into stable, provenance-rich text chunks.

    A chunk whose text repeats an earlier chunk of the document - a sheet copied
    into another, a page printed twice - is left out: it states nothing the
    first did, and every fact it holds keeps the first one as its evidence.
    """

    chunks: list[Chunk] = []
    seen: set[str] = set()
    global_index = 0
    ocr_generation_id = _ocr_generation_id(document)
    document_asset_ids = [asset.id for asset in document.assets if asset.page_number is None]
    for page in document.pages:
        text = page.markdown or page.raw_text
        heading_matches = iter(_HEADING_RE.finditer(text))
        next_heading = next(heading_matches, None)
        section_stack: list[str] = []
        page_asset_ids = [
            asset.id for asset in document.assets if asset.page_number == page.page_number
        ]
        parts = _split_text(text, chunk_token_size, chunk_token_overlap)
        block_ids_by_part = _block_ids_for_parts(page.blocks, text, parts)
        for part_index, ((content, start, end, token_count), block_ids) in enumerate(
            zip(parts, block_ids_by_part, strict=True)
        ):
            while next_heading is not None and next_heading.start() <= start:
                level = len(next_heading.group(1))
                title = " ".join(next_heading.group(2).strip().split())
                if title:
                    section_stack = section_stack[: level - 1]
                    section_stack.append(title)
                next_heading = next(heading_matches, None)
            normalized = normalize_text(content)
            if not normalized:
                continue
            content_hash = sha256_hex(normalized)
            if content_hash in seen:
                continue
            seen.add(content_hash)
            chunks.append(
                Chunk(
                    id=stable_id(
                        "chunk",
                        graph_id,
                        document.file_id,
                        page.page_number,
                        part_index,
                        content_hash,
                    ),
                    graph_id=graph_id,
                    file_id=document.file_id,
                    document_id=document.file_id,
                    page_number=page.page_number,
                    chunk_index=global_index,
                    content=content,
                    start_offset=start,
                    end_offset=end,
                    token_count=token_count,
                    content_hash=content_hash,
                    section_path=list(section_stack),
                    block_ids=block_ids,
                    asset_ids=(
                        [*document_asset_ids, *page_asset_ids]
                        if global_index == 0
                        else page_asset_ids
                    ),
                    ocr_generation_id=ocr_generation_id,
                )
            )
            global_index += 1
    return chunks


def compute_ordered_chunk_hash(chunks: list[Chunk]) -> str:
    """Hash chunk identity, order, and content for extraction cache invalidation.

    A cached extraction result records the chunk ids its entities were grounded
    in. Keying only on content therefore returns entities that cite ids the
    current run does not have whenever identity changes while the text does not —
    re-ingesting the same document through another source, or after its checksum
    definition changes. Every entity then fails chunk grounding and is filtered
    out, which surfaces as a silently empty graph rather than as a cache fault.
    """

    payload = [
        {"chunk_id": chunk.id, "chunk_index": chunk.chunk_index, "content": chunk.content}
        for chunk in chunks
    ]
    return sha256_hex(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _split_text(
    text: str,
    chunk_token_size: int,
    chunk_token_overlap: int,
) -> list[tuple[str, int, int, int]]:
    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return []
    chunks: list[tuple[str, int, int, int]] = []
    step = max(1, chunk_token_size - chunk_token_overlap)
    for token_start in range(0, len(matches), step):
        token_end = min(len(matches), token_start + chunk_token_size)
        selected = matches[token_start:token_end]
        if not selected:
            continue
        start = selected[0].start()
        end = selected[-1].end()
        chunks.append((text[start:end], start, end, len(selected)))
        if token_end == len(matches):
            break
    return chunks


def _block_ids_for_parts(
    blocks: list[LayoutBlock],
    page_text: str,
    parts: list[tuple[str, int, int, int]],
) -> list[list[str]]:
    """Associate blocks with all chunk spans without rescanning blocks per chunk."""

    results: list[list[str]] = [[] for _part in parts]
    if not blocks or not parts:
        return results

    chunk_starts = [part[1] for part in parts]
    chunk_ends = [part[2] for part in parts]
    fallback_results: list[list[str]] = [[] for _part in parts]
    search_offset = 0
    for block in blocks:
        block_start = _metadata_int(block.metadata, "start_offset")
        block_end = _metadata_int(block.metadata, "end_offset")
        if block_start is not None and block_end is not None and block_start < block_end:
            _append_block_overlaps(
                results,
                chunk_starts,
                chunk_ends,
                block.id,
                block_start,
                block_end,
            )

        fallback_span = _find_block_text_span(page_text, block.text, search_offset)
        if fallback_span is None:
            continue
        fallback_start, fallback_end = fallback_span
        search_offset = fallback_end
        _append_block_overlaps(
            fallback_results,
            chunk_starts,
            chunk_ends,
            block.id,
            fallback_start,
            fallback_end,
        )

    return [
        matches or fallback for matches, fallback in zip(results, fallback_results, strict=True)
    ]


def _append_block_overlaps(
    results: list[list[str]],
    chunk_starts: list[int],
    chunk_ends: list[int],
    block_id: str,
    block_start: int,
    block_end: int,
) -> None:
    chunk_index = bisect_right(chunk_ends, block_start)
    while chunk_index < len(chunk_starts) and chunk_starts[chunk_index] < block_end:
        results[chunk_index].append(block_id)
        chunk_index += 1


def _find_block_text_span(page_text: str, block_text: str, start: int) -> tuple[int, int] | None:
    if not block_text:
        return None
    exact_start = page_text.find(block_text, start)
    if exact_start < 0:
        exact_start = page_text.find(block_text)
    if exact_start >= 0:
        return exact_start, exact_start + len(block_text)

    tokens = _TOKEN_RE.findall(block_text)
    if not tokens:
        return None
    whitespace_tolerant = re.compile(r"\s+".join(re.escape(token) for token in tokens))
    match = whitespace_tolerant.search(page_text, start) or whitespace_tolerant.search(page_text)
    return match.span() if match is not None else None


def _metadata_int(metadata: dict[str, object], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _ocr_generation_id(document: ParsedDocument) -> str | None:
    for key in ("ocr_generation_id", "generation_id", "provider_generation_id"):
        value = document.provider_metadata.get(key)
        if value:
            return str(value)
    return None
