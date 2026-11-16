# SPDX-License-Identifier: Apache-2.0
"""Read the data a DrawingML chart plots from the values its file caches.

PowerPoint, Word and Excel store a chart in the same part, a ``c:chartSpace``
document, and every series keeps a copy of what it plots (``c:strCache`` for
labels, ``c:numCache`` for numbers) beside the reference into the workbook
that produced it. The cached copy is exactly what the file displays, so one
reader recovers a chart's numbers for all three formats without opening the
embedded workbook or looking at a picture of the chart.

Elements are matched by local name, so the transitional and strict OOXML
namespaces read alike.
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta
from xml.etree.ElementTree import Element

from defusedxml import ElementTree

from kg_processor.domain.tables import ChartData, ChartSeries, format_value

# The kinds of plot a plot area can hold, by element name, and what each is
# called in ``ChartData.chart_type``. A bar chart is a "bar" or a "column"
# according to its ``c:barDir``.
_PLOT_TYPES = {
    "barChart": "bar",
    "bar3DChart": "bar",
    "lineChart": "line",
    "line3DChart": "line",
    "pieChart": "pie",
    "pie3DChart": "pie",
    "ofPieChart": "pie",
    "doughnutChart": "doughnut",
    "areaChart": "area",
    "area3DChart": "area",
    "radarChart": "radar",
    "scatterChart": "scatter",
    "bubbleChart": "bubble",
    "stockChart": "stock",
    "surfaceChart": "surface",
    "surface3DChart": "surface",
}
# Plots whose series state an x value per point rather than a category.
_XY_PLOTS = frozenset({"scatterChart", "bubbleChart"})
_CATEGORY_AXES = frozenset({"catAx", "dateAx"})
# Axis positions that run along the bottom or top of the plot: on a chart
# with two value axes (scatter, bubble) the horizontal one carries x.
_HORIZONTAL_AXIS_POSITIONS = frozenset({"b", "t"})
# Levels of a multi-level category axis read from the outermost group in.
_CATEGORY_LEVEL_SEPARATOR = " / "
# Excel counts days from 1899-12-30 (the 1900 system, which keeps Lotus's
# phantom 29 February) or from 1904-01-01 when the workbook says so.
_EPOCH_1900 = date(1899, 12, 30)
_EPOCH_1904 = date(1904, 1, 1)
# The largest serial a date axis can hold: 9999-12-31 in the 1900 system.
_MAX_DATE_SERIAL = 2_958_465
# A number format is a date format when it uses day, month or year codes
# outside quoted text and bracketed sections ("[$-409]", "[Red]").
_FORMAT_LITERAL_RE = re.compile(r'"[^"]*"|\[[^\]]*\]|\\.')
_DATE_CODE_RE = re.compile(r"[dmy]", re.IGNORECASE)
_DAY_CODE_RE = re.compile(r"d", re.IGNORECASE)
_MONTH_CODE_RE = re.compile(r"m", re.IGNORECASE)
# A cached point list longer than this is a hostile or broken part: a chart
# plots far fewer points, and ``ptCount`` sizes the lists built below.
MAX_CHART_POINTS = 100_000


class ChartReadError(ValueError):
    """A chart part that does not hold a readable chart."""


class EmptyChartError(ChartReadError):
    """A well-formed chart that plots no series: there is nothing to read."""


def read_chart(payload: bytes) -> ChartData:
    """Read one chart part into its title, categories, series and axis titles.

    Every plot in the plot area contributes its series, so a combination
    chart (bars with a line over them) keeps them all and is named for each
    kind it draws, such as "bar+line". Raises ``ChartReadError`` when the
    part has no plot area or no series, and lets the XML parser's own errors
    through for a part that is not well formed.
    """

    root = ElementTree.fromstring(payload)
    chart = _child(root, "chart")
    if chart is None:
        raise ChartReadError("chart part has no c:chart element")
    plot_area = _child(chart, "plotArea")
    if plot_area is None:
        raise ChartReadError("chart has no plot area")
    epoch = _EPOCH_1904 if _bool_value(_child(root, "date1904")) else _EPOCH_1900

    kinds: list[str] = []
    series: list[ChartSeries] = []
    categories: list[str] = []
    for plot in plot_area:
        plot_name = _local_name(plot.tag)
        kind = _PLOT_TYPES.get(plot_name)
        if kind is None:
            continue
        if plot_name in {"barChart", "bar3DChart"}:
            kind = "column" if _value(_child(plot, "barDir")) == "col" else "bar"
        if kind not in kinds:
            kinds.append(kind)
        for node in _children(plot, "ser"):
            entry, labels = _read_series(node, xy=plot_name in _XY_PLOTS, epoch=epoch)
            series.append(entry)
            if labels and len(labels) > len(categories):
                categories = labels
    if not series:
        raise EmptyChartError("chart plots no series")

    category_title, value_title = _axis_titles(plot_area)
    title = _chart_title(chart)
    if title is None and not _bool_value(_child(chart, "autoTitleDeleted")) and len(series) == 1:
        # A chart with one series and no title of its own shows that
        # series' name as its title unless the author deleted it.
        title = series[0].name or None
    return ChartData(
        chart_type="+".join(kinds),
        title=title,
        categories=categories,
        series=series,
        category_axis_title=category_title,
        value_axis_title=value_title,
        source="native",
    )


def _read_series(node: Element, *, xy: bool, epoch: date) -> tuple[ChartSeries, list[str] | None]:
    """One series and the category labels it plots against, if it names any.

    A scatter or bubble series plots ``c:yVal`` against numeric ``c:xVal``;
    an x list of text is categories, which is how Excel plots it too.
    """

    name = _series_name(_child(node, "tx"))
    x_values: list[float | None] | None = None
    if xy:
        values, number_format = _numbers(_child(node, "yVal"))
        x_node = _child(node, "xVal")
        labels: list[str] | None = None
        if x_node is not None and _is_numeric_data(x_node):
            x_values, _x_format = _numbers(x_node)
        elif x_node is not None:
            labels = _labels(x_node, epoch)
    else:
        values, number_format = _numbers(_child(node, "val"))
        category_node = _child(node, "cat")
        labels = _labels(category_node, epoch) if category_node is not None else None
    return (
        ChartSeries(name=name, values=values, x_values=x_values, number_format=number_format),
        labels,
    )


def _series_name(tx: Element | None) -> str:
    """A series name from its cached reference or its literal ``c:v``."""

    if tx is None:
        return ""
    reference = _child(tx, "strRef")
    if reference is not None:
        cache = _child(reference, "strCache")
        texts = [text for text in _point_texts(cache) if text] if cache is not None else []
        return " ".join(texts).strip()
    return (_child_text(tx, "v") or "").strip()


def _is_numeric_data(node: Element) -> bool:
    return any(_local_name(child.tag) in {"numRef", "numLit"} for child in node)


def _numbers(node: Element | None) -> tuple[list[float | None], str | None]:
    """The cached numbers of a ``c:val``/``c:yVal``/``c:xVal`` and their format.

    The list is ``ptCount`` long and a point the cache leaves out stays None,
    so every value stays under its own category. A format given on the
    points but not on the cache is used for the whole series.
    """

    if node is None:
        return [], None
    cache = _number_cache(node)
    if cache is None:
        # Text in a value slot is unusual but legal; read what parses.
        labels = _string_cache(node)
        return [_float(text) for text in _point_texts(labels)] if labels is not None else [], None
    number_format = _child_text(cache, "formatCode")
    points = _indexed_points(cache)
    values: list[float | None] = [None] * _point_count(cache, points)
    point_format: str | None = None
    for index, point in points:
        values[index] = _float(_child_text(point, "v"))
        point_format = point_format or point.get("formatCode")
    if not number_format or number_format.lower() == "general":
        number_format = point_format or number_format
    return values, number_format or None


def _labels(node: Element, epoch: date) -> list[str]:
    """Category labels of a ``c:cat``/``c:xVal``, whichever way they are cached."""

    for child in node:
        name = _local_name(child.tag)
        if name in {"multiLvlStrRef", "multiLvlStrCache"}:
            return _multi_level_labels(child)
        if name in {"numRef", "numLit"}:
            return _number_labels(child if name == "numLit" else _child(child, "numCache"), epoch)
    cache = _string_cache(node)
    return _point_texts(cache) if cache is not None else []


def _number_labels(cache: Element | None, epoch: date) -> list[str]:
    """Numeric categories written the way their format shows them.

    Dates are stored as day counts; a date format turns them back into
    dates (ISO, to the precision the format shows) rather than numbers.
    """

    if cache is None:
        return []
    number_format = _child_text(cache, "formatCode") or ""
    points = _indexed_points(cache)
    labels = [""] * _point_count(cache, points)
    for index, point in points:
        value = _float(_child_text(point, "v"))
        labels[index] = _format_category(value, point.get("formatCode") or number_format, epoch)
    return labels


def _format_category(value: float | None, number_format: str, epoch: date) -> str:
    if value is None:
        return ""
    codes = _FORMAT_LITERAL_RE.sub("", number_format)
    if _DATE_CODE_RE.search(codes) and 0 <= value <= _MAX_DATE_SERIAL:
        day = epoch + timedelta(days=math.floor(value))
        if _DAY_CODE_RE.search(codes):
            return day.isoformat()
        if _MONTH_CODE_RE.search(codes):
            return f"{day.year:04d}-{day.month:02d}"
        return f"{day.year:04d}"
    return format_value(value, number_format or None)


def _multi_level_labels(reference: Element) -> list[str]:
    """Join a multi-level axis's labels from the outermost group inward.

    The first ``c:lvl`` holds the innermost labels, one per category; each
    later level names groups, and a group's label sits on the first category
    it covers and holds until the next label.
    """

    cache = (
        reference
        if _local_name(reference.tag) == "multiLvlStrCache"
        else _child(reference, "multiLvlStrCache")
    )
    if cache is None:
        return []
    levels = _children(cache, "lvl")
    if not levels:
        return []
    count = _point_count(cache, _indexed_points(levels[0]))
    joined: list[list[str]] = [[] for _ in range(count)]
    for level in reversed(levels):
        label = ""
        by_index = dict(_indexed_points(level, limit=count))
        for index in range(count):
            point = by_index.get(index)
            if point is not None:
                label = (_child_text(point, "v") or "").strip()
            if label and (not joined[index] or joined[index][-1] != label):
                joined[index].append(label)
    return [_CATEGORY_LEVEL_SEPARATOR.join(parts) for parts in joined]


def _axis_titles(plot_area: Element) -> tuple[str | None, str | None]:
    """The category and value axis titles, if the axes carry any.

    A chart with a category axis takes its title for the categories and the
    first titled value axis for the values. A scatter or bubble chart has
    two value axes; the one along the bottom or top carries x.
    """

    category: str | None = None
    value: str | None = None
    has_category_axis = any(_local_name(axis.tag) in _CATEGORY_AXES for axis in plot_area)
    for axis in plot_area:
        name = _local_name(axis.tag)
        if name not in _CATEGORY_AXES and name != "valAx":
            continue
        title_node = _child(axis, "title")
        title = _title_text(title_node) if title_node is not None else None
        if not title:
            continue
        horizontal = _value(_child(axis, "axPos")) in _HORIZONTAL_AXIS_POSITIONS
        if name in _CATEGORY_AXES or (not has_category_axis and horizontal):
            category = category or title
        else:
            value = value or title
    return category, value


def _chart_title(chart: Element) -> str | None:
    title = _child(chart, "title")
    return _title_text(title) if title is not None else None


def _title_text(title: Element) -> str | None:
    """A title's text: rich text paragraphs, or a cached cell reference."""

    tx = _child(title, "tx")
    if tx is None:
        return None
    rich = _child(tx, "rich")
    if rich is not None:
        paragraphs = []
        for paragraph in rich.iter():
            if _local_name(paragraph.tag) != "p":
                continue
            runs = "".join(
                node.text or "" for node in paragraph.iter() if _local_name(node.tag) == "t"
            )
            if runs.strip():
                paragraphs.append(runs.strip())
        return " ".join(paragraphs) or None
    text = _series_name(tx)
    return text or None


def _number_cache(node: Element) -> Element | None:
    for child in node:
        name = _local_name(child.tag)
        if name == "numLit":
            return child
        if name == "numRef":
            return _child(child, "numCache")
    return None


def _string_cache(node: Element) -> Element | None:
    for child in node:
        name = _local_name(child.tag)
        if name == "strLit":
            return child
        if name == "strRef":
            return _child(child, "strCache")
    return None


def _point_texts(cache: Element | None) -> list[str]:
    if cache is None:
        return []
    points = _indexed_points(cache)
    texts = [""] * _point_count(cache, points)
    for index, point in points:
        texts[index] = (_child_text(point, "v") or "").strip()
    return texts


def _indexed_points(cache: Element, limit: int | None = None) -> list[tuple[int, Element]]:
    """The cache's ``c:pt`` elements with their indexes, skipping bad ones."""

    bound = MAX_CHART_POINTS if limit is None else min(limit, MAX_CHART_POINTS)
    points: list[tuple[int, Element]] = []
    for point in _children(cache, "pt"):
        try:
            index = int(point.get("idx", ""))
        except ValueError:
            continue
        if 0 <= index < bound:
            points.append((index, point))
    return points


def _point_count(cache: Element, points: list[tuple[int, Element]]) -> int:
    """How many points the cache covers: its ``ptCount``, or past its last point."""

    declared = _value(_child(cache, "ptCount"))
    highest = max((index + 1 for index, _point in points), default=0)
    try:
        count = int(declared) if declared is not None else highest
    except ValueError:
        count = highest
    return min(max(count, highest), MAX_CHART_POINTS)


def _float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _bool_value(node: Element | None) -> bool:
    value = _value(node)
    if value is None:
        # A boolean element with no val attribute means true.
        return node is not None
    return value.lower() in {"1", "true"}


def _value(node: Element | None) -> str | None:
    return node.get("val") if node is not None else None


def _child(node: Element, local_name: str) -> Element | None:
    return next((child for child in node if _local_name(child.tag) == local_name), None)


def _children(node: Element, local_name: str) -> list[Element]:
    return [child for child in node if _local_name(child.tag) == local_name]


def _child_text(node: Element, local_name: str) -> str | None:
    child = _child(node, local_name)
    return child.text if child is not None else None


def _local_name(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""
