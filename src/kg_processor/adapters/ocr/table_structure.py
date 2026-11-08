# SPDX-License-Identifier: Apache-2.0
"""Turn the cell lists of Office and HTML tables into rectangular rows.

A merged cell is written once in its source and covers several positions.
Each format says so its own way: PowerPoint repeats a continuation cell
(``hMerge``/``vMerge``) in every covered position, Word repeats one only for
rows below (``w:vMerge``) and spans columns in place (``w:gridSpan``), and
HTML writes neither and spans with ``rowspan``/``colspan``. ``build_rows``
reads all three into one grid where a merged cell's text sits in every
position it covers, so each row of ``TableData`` reads on its own.

The HTML table reader here serves both the built-in HTML parser and the
MinerU adapters, whose table blocks carry HTML.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from kg_processor.domain.tables import TableData, TableSource

# Bounds a hostile table cannot push past: spans beyond these are clamped and
# a grid stops growing at this many rows or columns.
MAX_TABLE_SPAN = 1_000
MAX_TABLE_ROWS = 20_000
MAX_TABLE_COLUMNS = 1_000
_WHITESPACE_RE = re.compile(r"[ \t\f\v\r]+")
_HTML_TABLE_START_RE = re.compile(r"<table\b", re.IGNORECASE)


@dataclass(frozen=True)
class GridCell:
    """One cell as its source writes it, before merges are spread out.

    ``continues_left``/``continues_up`` mark a placeholder that only says the
    position belongs to the cell to its left or above it.
    """

    text: str
    column_span: int = 1
    row_span: int = 1
    continues_left: bool = False
    continues_up: bool = False


def build_rows(rows: list[list[GridCell]]) -> list[list[str]]:
    """Lay cells out on a grid, repeating merged text across its span.

    A cell that starts a span fills every position it covers. A continuation
    placeholder takes its position: if a span already filled it, it is only
    consumed; otherwise it copies the cell to its left or above. A position a
    span from an earlier row already holds is skipped when an ordinary cell
    is placed, which is how HTML and Word's column spans read.
    """

    grid: dict[tuple[int, int], str] = {}
    width = 0
    for row_index, row in enumerate(rows[:MAX_TABLE_ROWS]):
        column = 0
        for cell in row:
            span = _clamp(cell.column_span)
            if cell.continues_left or cell.continues_up:
                for offset in range(span):
                    position = (row_index, column + offset)
                    if position not in grid:
                        source = (
                            (row_index - 1, column + offset)
                            if cell.continues_up
                            else (row_index, column + offset - 1)
                        )
                        grid[position] = grid.get(source, "")
                column += span
                continue
            while (row_index, column) in grid:
                column += 1
            if column >= MAX_TABLE_COLUMNS:
                break
            for down in range(_clamp(cell.row_span)):
                for across in range(span):
                    target = (row_index + down, column + across)
                    if target[0] < MAX_TABLE_ROWS and target[1] < MAX_TABLE_COLUMNS:
                        grid.setdefault(target, cell.text)
            column += 1
        width = max(width, min(column, MAX_TABLE_COLUMNS))
    if not grid:
        return []
    height = min(max(row for row, _column in grid) + 1, MAX_TABLE_ROWS)
    width = max(width, max(column for _row, column in grid) + 1)
    return [[grid.get((row, column), "") for column in range(width)] for row in range(height)]


def table_from_rows(
    rows: list[list[str]],
    *,
    header_rows: int,
    caption: str | None = None,
    source: TableSource = "native",
) -> TableData | None:
    """A ``TableData`` from laid-out rows, or None when no cell holds text.

    Header rows are only counted while a data row remains below them: a
    one-row table flagged as having a header row is more likely a row of
    values than a header over nothing.
    """

    if not any(cell.strip() for row in rows for cell in row):
        return None
    header_rows = max(0, min(header_rows, len(rows) - 1))
    return TableData(
        caption=caption.strip() if caption and caption.strip() else None,
        header_rows=header_rows,
        rows=rows,
        source=source,
    )


def contains_html_table(text: str) -> bool:
    """Whether ``text`` holds an HTML ``<table>`` start tag."""

    return bool(_HTML_TABLE_START_RE.search(text))


def table_from_html(html: str, *, source: TableSource = "ocr") -> TableData | None:
    """The first table in an HTML fragment, or None when it has no cells.

    Nested tables are read into the cell that holds them, as text.
    """

    tables = tables_from_html(html, source=source)
    return tables[0] if tables else None


def tables_from_html(html: str, *, source: TableSource = "ocr") -> list[TableData]:
    """Every outermost table in an HTML fragment, in document order."""

    parser = _HtmlTablesParser(source)
    parser.feed(html)
    parser.close()
    return parser.tables


class HtmlTableCollector:
    """Collect one HTML table from the tag events of an enclosing parser.

    The built-in HTML text parser and the fragment reader above both feed
    it, so a table reads the same wherever it is found. Rows inside
    ``thead``, and leading rows made only of ``th`` cells, are header rows.
    A table nested inside a cell is flattened into that cell's text.
    """

    def __init__(self, source: TableSource = "native") -> None:
        """Start an empty table whose cells are read from later events."""

        self.source: TableSource = source
        self._rows: list[list[GridCell]] = []
        self._row_is_header: list[bool] = []
        self._row: list[GridCell] | None = None
        self._row_all_th = True
        self._cell: list[str] | None = None
        self._cell_span = (1, 1)
        self._caption: list[str] | None = None
        self._caption_text: list[str] = []
        self._in_head = False
        self._nested = 0

    def start(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle a start tag seen inside the table."""

        if tag == "table":
            self._nested += 1
            self._text("\n")
            return
        if self._nested:
            self._structural_break(tag)
            return
        if tag == "caption":
            self._caption = []
        elif tag == "thead":
            self._in_head = True
        elif tag in {"tbody", "tfoot"}:
            self._in_head = False
        elif tag == "tr":
            self._finish_row()
            self._row = []
            self._row_all_th = True
        elif tag in {"td", "th"}:
            self._finish_cell()
            if self._row is None:
                self._row = []
                self._row_all_th = True
            self._row_all_th = self._row_all_th and tag == "th"
            attributes = {name.lower(): value or "" for name, value in attrs}
            self._cell_span = (
                _span(attributes.get("colspan")),
                _span(attributes.get("rowspan")),
            )
            self._cell = []
        else:
            self._structural_break(tag)

    def end(self, tag: str) -> bool:
        """Handle an end tag; True once the table itself has closed."""

        if tag == "table":
            if self._nested:
                self._nested -= 1
                self._text("\n")
                return False
            self._finish_row()
            return True
        if self._nested:
            self._structural_break(tag)
            return False
        if tag == "caption" and self._caption is not None:
            self._caption_text.append("".join(self._caption))
            self._caption = None
        elif tag == "thead":
            self._in_head = False
        elif tag == "tr":
            self._finish_row()
        elif tag in {"td", "th"}:
            self._finish_cell()
        else:
            self._structural_break(tag)
        return False

    def data(self, text: str) -> None:
        """Handle text seen inside the table."""

        self._text(text)

    def table(self) -> TableData | None:
        """The table read so far, or None when it holds no text."""

        self._finish_row()
        leading_headers = 0
        for is_header in self._row_is_header:
            if not is_header:
                break
            leading_headers += 1
        caption = _normalize(" ".join(self._caption_text)).replace("\n", " ")
        return table_from_rows(
            build_rows(self._rows),
            header_rows=leading_headers,
            caption=caption or None,
            source=self.source,
        )

    def _text(self, text: str) -> None:
        if self._caption is not None:
            self._caption.append(text)
        elif self._cell is not None:
            self._cell.append(text)

    def _structural_break(self, tag: str) -> None:
        if tag in {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._text("\n")
        elif tag in {"td", "th"}:
            self._text(" ")

    def _finish_cell(self) -> None:
        if self._cell is None or self._row is None:
            self._cell = None
            return
        column_span, row_span = self._cell_span
        self._row.append(
            GridCell(
                text=_normalize("".join(self._cell)), column_span=column_span, row_span=row_span
            )
        )
        self._cell = None

    def _finish_row(self) -> None:
        self._finish_cell()
        if self._row is None:
            return
        if self._row:
            self._rows.append(self._row)
            self._row_is_header.append(self._in_head or self._row_all_th)
        self._row = None


class _HtmlTablesParser(HTMLParser):
    """Find the outermost tables of an HTML fragment and read each one."""

    def __init__(self, source: TableSource) -> None:
        super().__init__(convert_charrefs=True)
        self._source: TableSource = source
        self._collector: HtmlTableCollector | None = None
        self.tables: list[TableData] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open a table, or pass a tag inside one to its collector."""

        normalized = tag.lower()
        if self._collector is not None:
            self._collector.start(normalized, attrs)
        elif normalized == "table":
            self._collector = HtmlTableCollector(self._source)

    def handle_endtag(self, tag: str) -> None:
        """Pass an end tag to the open table, keeping it once it closes."""

        if self._collector is None:
            return
        if self._collector.end(tag.lower()):
            table = self._collector.table()
            if table is not None:
                self.tables.append(table)
            self._collector = None

    def handle_data(self, data: str) -> None:
        """Pass text to the open table."""

        if self._collector is not None:
            self._collector.data(data)

    def close(self) -> None:
        """Keep a table the fragment never closed."""

        super().close()
        if self._collector is not None:
            table = self._collector.table()
            if table is not None:
                self.tables.append(table)
            self._collector = None


def _normalize(text: str) -> str:
    """Collapse spaces within lines and drop blank lines."""

    lines = (_WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _span(value: str | None) -> int:
    try:
        return _clamp(int((value or "1").strip()))
    except ValueError:
        return 1


def _clamp(span: int) -> int:
    return max(1, min(span, MAX_TABLE_SPAN))
