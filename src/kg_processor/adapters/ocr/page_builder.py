# SPDX-License-Identifier: Apache-2.0
"""Assemble a page's text and its layout blocks so their offsets agree.

Chunking cites a block by the character span its ``start_offset`` and
``end_offset`` give in the page markdown, and an evidence quote is a slice of
that same text. Writing the text and recording the spans in one place, as
each block is appended, keeps them exact by construction: there is no later
search that could land on an earlier copy of the same words.
"""

from __future__ import annotations

from typing import Any

from kg_processor.domain.documents import LayoutBlock, ParsedPage
from kg_processor.domain.ids import stable_id

# How blocks are separated in the page text: lines of running text one per
# line, and structured blocks (tables, charts, notes) set off by a blank line
# so a markdown reader sees them as blocks of their own.
LINE_BREAK = "\n"
BLOCK_BREAK = "\n\n"


class PageBuilder:
    """Append blocks to one page, writing their text and offsets together."""

    def __init__(self, file_id: str, page_number: int, *, id_prefix: str = "block") -> None:
        """Start an empty page whose block ids are scoped to ``file_id``."""

        self.file_id = file_id
        self.page_number = page_number
        self.id_prefix = id_prefix
        self._parts: list[str] = []
        self._length = 0
        self.blocks: list[LayoutBlock] = []

    @property
    def is_empty(self) -> bool:
        """Whether nothing has been written yet."""

        return self._length == 0

    def add(
        self,
        kind: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        *,
        separator: str = LINE_BREAK,
        prefix: str = "",
    ) -> LayoutBlock | None:
        """Write ``prefix`` + ``text`` and add a block spanning ``text``.

        The separator is written only between blocks, never before the
        first. The prefix (such as "Figure: ") is page text that belongs to
        no block's span. Nothing is written for empty text.
        """

        if not text:
            return None
        if self._length:
            self._write(separator)
        self._write(prefix)
        start = self._length
        self._write(text)
        return self._block(kind, text, metadata, start, self._length)

    def mark(self, kind: str, metadata: dict[str, Any] | None = None) -> LayoutBlock:
        """Add a block with no text at the current end of the page.

        A picture without alt text has nothing to write, but its block still
        says where on the page it sits.
        """

        return self._block(kind, "", metadata, self._length, self._length)

    def page(self) -> ParsedPage:
        """The page as written: its markdown, raw text and blocks."""

        text = "".join(self._parts)
        return ParsedPage(
            page_number=self.page_number,
            markdown=text,
            raw_text=text,
            blocks=list(self.blocks),
        )

    def _write(self, text: str) -> None:
        if text:
            self._parts.append(text)
            self._length += len(text)

    def _block(
        self,
        kind: str,
        text: str,
        metadata: dict[str, Any] | None,
        start: int,
        end: int,
    ) -> LayoutBlock:
        index = len(self.blocks)
        block = LayoutBlock(
            id=stable_id(self.id_prefix, self.file_id, self.page_number, index, kind, text[:256]),
            page_number=self.page_number,
            kind=kind,
            text=text,
            metadata={**(metadata or {}), "start_offset": start, "end_offset": end},
        )
        self.blocks.append(block)
        return block
