# SPDX-License-Identifier: Apache-2.0
"""Deterministic document chunking.

Chunk ids include graph/file/page/chunk position and a text hash so reruns are
stable while still detecting changed OCR output. The chunker keeps page/block
provenance because every extracted graph observation must cite its source text.

Tables and charts are written into page text one row per line. A chunk boundary
never cuts a row: a row read in halves states neither half's facts. A chunk
that begins below a table's header carries the table's caption and column
labels in its section path - context the extraction model reads, not text a
quote can cite - so its rows still say what they measure.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_left, bisect_right

from pydantic import ValidationError

from kg_processor.domain.documents import LayoutBlock, ParsedDocument
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ids import normalize_text, sha256_hex, stable_id
from kg_processor.domain.tables import ChartData, TableData, clean_cell

_TOKEN_RE = re.compile(r"\S+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
# Blocks written into page text one row per line.
_ROW_BLOCK_KINDS = frozenset({"table", "chart"})
# A chunk ends at a row's end only if it keeps at least this share of its
# token budget; a shorter one would leave a sliver before the next chunk.
_MIN_SNAPPED_SHARE = 0.5
# How much of a table's column labels a continuation chunk carries.
_TABLE_CONTEXT_CHARACTERS = 400


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
        regions = _row_regions(page.blocks, text)
        parts = _split_text(text, chunk_token_size, chunk_token_overlap, regions)
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
                    section_path=[*section_stack, *_continued_tables(regions, start)],
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


class _RowRegion:
    """Where a table or chart sits in page text, and what labels its columns."""

    __slots__ = ("context", "end", "start")

    def __init__(self, start: int, end: int, context: str) -> None:
        self.start = start
        self.end = end
        self.context = context


def _row_regions(blocks: list[LayoutBlock], page_text: str) -> list[_RowRegion]:
    """The spans of page text that hold table or chart rows, in page order.

    Only blocks whose offsets point at their own text count: a parser that
    gave no offsets, or offsets the text does not bear out, wrote no rows the
    chunker can keep whole.
    """

    regions: list[_RowRegion] = []
    for block in blocks:
        if block.kind not in _ROW_BLOCK_KINDS:
            continue
        start = _metadata_int(block.metadata, "start_offset")
        end = _metadata_int(block.metadata, "end_offset")
        if start is None or end is None or not 0 <= start < end <= len(page_text):
            continue
        if normalize_text(page_text[start:end]) != normalize_text(block.text):
            continue
        regions.append(_RowRegion(start, end, _table_context(block)))
    return sorted(regions, key=lambda region: region.start)


def _table_context(block: LayoutBlock) -> str:
    """A table's caption and column labels, as one line of context."""

    table: TableData | None = None
    try:
        if isinstance(block.metadata.get("table"), dict):
            table = TableData.model_validate(block.metadata["table"])
        elif isinstance(block.metadata.get("chart"), dict):
            chart = ChartData.model_validate(block.metadata["chart"])
            table = chart.as_table().model_copy(update={"caption": chart.title})
    except ValidationError:
        table = None
    if table is None:
        first_line = block.text.split("\n", 1)[0]
        return clean_cell(first_line)[:_TABLE_CONTEXT_CHARACTERS]
    kind = "Chart" if block.kind == "chart" else "Table"
    caption = clean_cell(table.caption or "")
    columns = " | ".join(clean_cell(label) for label in table.header() if label.strip())
    context = f"{kind}: {caption}" if caption else kind
    if columns:
        context += f" (columns: {columns})"
    return context[:_TABLE_CONTEXT_CHARACTERS]


def _continued_tables(regions: list[_RowRegion], start: int) -> list[str]:
    """Context for the tables a chunk starting at ``start`` begins inside of."""

    return [region.context for region in regions if region.start < start < region.end]


def _split_text(
    text: str,
    chunk_token_size: int,
    chunk_token_overlap: int,
    regions: list[_RowRegion] | None = None,
) -> list[tuple[str, int, int, int]]:
    """Token windows over page text that never end or begin inside a row.

    A window that would end inside a table ends at the last row it holds
    whole, if that keeps most of its budget; one that would begin inside a
    row begins at that row's start. A row longer than a whole window is cut
    where the budget ends, as any long line is.
    """

    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return []
    token_starts = [match.start() for match in matches]
    token_ends = [match.end() for match in matches]
    rows = regions or []
    chunks: list[tuple[str, int, int, int]] = []
    token_start = 0
    while token_start < len(matches):
        token_end = min(len(matches), token_start + chunk_token_size)
        if token_end < len(matches):
            token_end = _end_at_row(text, rows, token_ends, token_start, token_end)
        start = token_starts[token_start]
        end = token_ends[token_end - 1]
        chunks.append((text[start:end], start, end, token_end - token_start))
        if token_end == len(matches):
            break
        following = max(token_start + 1, token_end - chunk_token_overlap)
        token_start = _begin_at_row(text, rows, token_starts, token_start, following)
    return chunks


def _end_at_row(
    text: str,
    regions: list[_RowRegion],
    token_ends: list[int],
    token_start: int,
    token_end: int,
) -> int:
    """Move a window's last token back to the end of the last whole row."""

    end_char = token_ends[token_end - 1]
    region = next((item for item in regions if item.start < end_char < item.end), None)
    if region is None:
        return token_end
    line_break = text.find("\n", end_char, region.end)
    row_end = region.end if line_break == -1 else line_break
    if not text[end_char:row_end].strip():
        return token_end
    previous_break = text.rfind("\n", region.start, end_char)
    if previous_break == -1:
        return token_end
    snapped = bisect_right(token_ends, previous_break)
    if snapped - token_start < max(1, int((token_end - token_start) * _MIN_SNAPPED_SHARE)):
        return token_end
    return snapped


def _begin_at_row(
    text: str,
    regions: list[_RowRegion],
    token_starts: list[int],
    previous_start: int,
    token_start: int,
) -> int:
    """Move a window's first token back to the start of the row it is in."""

    if token_start >= len(token_starts):
        return token_start
    start_char = token_starts[token_start]
    region = next((item for item in regions if item.start < start_char < item.end), None)
    if region is None:
        return token_start
    row_start = text.rfind("\n", region.start, start_char) + 1
    snapped = bisect_left(token_starts, row_start)
    return snapped if snapped > previous_start else token_start


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
