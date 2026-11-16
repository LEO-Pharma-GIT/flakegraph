# SPDX-License-Identifier: Apache-2.0
"""Tables and charts as their sources state them, and how they read as text.

A parser that finds a table or a chart keeps its structure on the layout block
(``metadata["table"]`` or ``metadata["chart"]``) and writes the rendering below
into the page text, so chunking, extraction and evidence quotes work on text as
they always have while the rows stay queryable as data. Every parser renders
through these functions: the same table reads the same way whichever format
it came from, and the text is stable from one run to the next.
"""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, Field

TableSource = Literal["native", "ocr", "vision"]
ChartSource = Literal["native", "vision"]

# A cell is one line of a markdown row: a pipe would end it and a line break
# would end the row.
_CELL_BREAK_RE = re.compile(r"\s*[\r\n]+\s*")
_PERCENT_FORMAT_RE = re.compile(r"(?<!\\)%")
_DECIMALS_RE = re.compile(r"0\.(0+)")
# Plain numbers keep this many significant digits when no format says otherwise.
_SIGNIFICANT_DIGITS = 6
# Whole numbers below this print without a decimal point or exponent.
_LARGEST_PLAIN_INTEGER = 1e15


class TableData(BaseModel):
    """A table's cells as written: header rows first, then data rows.

    Merged cells repeat their text in every position they span, so each row
    reads on its own. ``header_rows`` counts the leading rows that label
    columns rather than state values.
    """

    caption: str | None = None
    header_rows: int = Field(default=0, ge=0)
    rows: list[list[str]] = Field(default_factory=list)
    source: TableSource = "native"

    @property
    def width(self) -> int:
        """The number of columns of the widest row."""

        return max((len(row) for row in self.rows), default=0)

    def header(self) -> list[str]:
        """One label per column, joining stacked header rows top to bottom."""

        width = self.width
        labels: list[str] = []
        for column in range(width):
            parts: list[str] = []
            for row in self.rows[: self.header_rows]:
                cell = row[column].strip() if column < len(row) else ""
                if cell and (not parts or parts[-1] != cell):
                    parts.append(cell)
            labels.append(" / ".join(parts))
        return labels

    def body(self) -> list[list[str]]:
        """The data rows, each padded to the table's width."""

        width = self.width
        return [[*row, *([""] * (width - len(row)))] for row in self.rows[self.header_rows :]]


class ChartSeries(BaseModel):
    """One plotted series: a name and a value per category (or per x value)."""

    name: str = ""
    values: list[float | None] = Field(default_factory=list)
    # Scatter and bubble series plot against numbers of their own.
    x_values: list[float | None] | None = None
    number_format: str | None = None


class ChartData(BaseModel):
    """A chart's plotted data, as the file stores it or as read from a picture.

    ``estimated`` marks values read off an image rather than stored in the
    file: they are approximate and say so wherever they are shown.
    """

    chart_type: str = "chart"
    title: str | None = None
    categories: list[str] = Field(default_factory=list)
    series: list[ChartSeries] = Field(default_factory=list)
    category_axis_title: str | None = None
    value_axis_title: str | None = None
    source: ChartSource = "native"
    estimated: bool = False

    def as_table(self) -> TableData:
        """The chart as a table: a row per category, a column per series."""

        if any(series.x_values is not None for series in self.series):
            return self._scatter_table()
        label = self.category_axis_title or "Category"
        names = (series.name or f"Series {index + 1}" for index, series in enumerate(self.series))
        header = [label, *names]
        length = max(
            [len(self.categories), *(len(series.values) for series in self.series)], default=0
        )
        rows = [header]
        for index in range(length):
            category = self.categories[index] if index < len(self.categories) else str(index + 1)
            rows.append(
                [
                    category,
                    *(
                        format_value(series.values[index], series.number_format)
                        if index < len(series.values)
                        else ""
                        for series in self.series
                    ),
                ]
            )
        return TableData(caption=self.title, header_rows=1, rows=rows, source=self._table_source())

    def _scatter_table(self) -> TableData:
        """A scatter chart as one row per plotted point."""

        x_label = self.category_axis_title or "X"
        y_label = self.value_axis_title or "Y"
        rows = [["Series", x_label, y_label]]
        for index, series in enumerate(self.series):
            name = series.name or f"Series {index + 1}"
            xs = series.x_values or []
            for point, value in enumerate(series.values):
                x_value = xs[point] if point < len(xs) else None
                rows.append(
                    [name, format_value(x_value, None), format_value(value, series.number_format)]
                )
        return TableData(caption=self.title, header_rows=1, rows=rows, source=self._table_source())

    def _table_source(self) -> TableSource:
        return "vision" if self.source == "vision" else "native"


def format_value(value: float | None, number_format: str | None) -> str:
    """Write a plotted number the way its format displays it.

    Percent formats store fractions (0.25 shows as 25%); a format with fixed
    decimals keeps them. Without a format a number keeps a few significant
    digits, and a whole number shows no decimals.
    """

    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    fmt = (number_format or "").split(";", 1)[0]
    if fmt.lower() == "general":
        fmt = ""
    percent = bool(_PERCENT_FORMAT_RE.search(fmt))
    shown = value * 100 if percent else value
    decimals_match = _DECIMALS_RE.search(fmt)
    if decimals_match is None and not ("0" in fmt or "#" in fmt):
        text = _plain_number(shown)
    else:
        decimals = len(decimals_match.group(1)) if decimals_match is not None else 0
        grouping = "," if "," in fmt else ""
        text = f"{shown:{grouping}.{decimals}f}"
    return f"{text}%" if percent else text


def _plain_number(value: float) -> str:
    if float(value).is_integer() and abs(value) < _LARGEST_PLAIN_INTEGER:
        return str(int(value))
    return f"{value:.{_SIGNIFICANT_DIGITS}g}"


def clean_cell(text: str) -> str:
    """One table cell as one line of text, its pipes escaped."""

    return _CELL_BREAK_RE.sub(" ", text.strip()).replace("|", "\\|")


def render_table(table: TableData) -> str:
    """A table as a markdown table, under its caption when it has one.

    A table without header rows gets an empty header, so the first data row
    is never taken for column labels.
    """

    width = table.width
    if width == 0:
        return f"Table: {clean_cell(table.caption)}" if table.caption else ""
    header = table.header() if table.header_rows else [""] * width
    lines: list[str] = []
    if table.caption:
        lines.append(f"Table: {clean_cell(table.caption)}")
    lines.append(_markdown_row(header))
    lines.append(_markdown_row(["---"] * width))
    lines.extend(_markdown_row(row) for row in table.body())
    return "\n".join(lines)


def render_chart(chart: ChartData) -> str:
    """A chart as a line naming it, then its data as a table.

    The line says what kind of chart it is, its title and value axis, and
    whether its values were read from a picture.
    """

    head = f"Chart ({chart.chart_type})"
    if chart.title:
        head += f": {clean_cell(chart.title)}"
    details: list[str] = []
    if chart.value_axis_title:
        details.append(f"values: {clean_cell(chart.value_axis_title)}")
    if chart.estimated:
        details.append("values estimated from an image")
    if details:
        head += f" ({'; '.join(details)})"
    table = chart.as_table().model_copy(update={"caption": None})
    body = render_table(table)
    return f"{head}\n{body}" if body else head


def _markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(clean_cell(cell) for cell in cells) + " |"
