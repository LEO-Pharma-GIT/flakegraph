# SPDX-License-Identifier: Apache-2.0
"""Read the pictures a document holds with a model that sees images.

A parser keeps what a file stores as text and data; a picture of a chart, a
screenshot of a table or a photographed whiteboard stores neither. After
parsing, each picture worth reading is sent to the graph's model, and what it
reads is written onto the picture's page the way every other content is: a
description and the picture's legible text, and charts and tables in the
shared table rendering, marked as read from an image. From there chunking,
extraction, evidence and search treat it as text with a source.

Most pictures in a document say nothing - logos, icons, dividers, the same
template band on every page - so pictures are chosen before any is sent:
too small, too large, an unreadable format, or repeated across pages, and the
same bytes are read once. What was skipped and why is recorded with the
document, so a reader can see what the graph did not look at.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from kg_processor.application.ordered_map import ordered_map
from kg_processor.application.prompt_registry import get_prompt_template
from kg_processor.domain.documents import LayoutBlock, ParsedAsset, ParsedDocument, ParsedPage
from kg_processor.domain.ids import stable_id
from kg_processor.domain.tables import (
    ChartData,
    ChartSeries,
    TableData,
    clean_cell,
    render_chart,
    render_table,
)
from kg_processor.ports.llm import (
    ImageInput,
    StructuredCompletionProvider,
    StructuredCompletionRequest,
)

logger = logging.getLogger(__name__)

FIGURE_READING_TASK = "figure_reading"
FigureKind = Literal[
    "chart", "table", "diagram", "photo", "screenshot", "text", "decoration", "other"
]
FIGURE_KINDS: tuple[FigureKind, ...] = (
    "chart",
    "table",
    "diagram",
    "photo",
    "screenshot",
    "text",
    "decoration",
    "other",
)
CHART_TYPES = ("bar", "line", "pie", "doughnut", "area", "scatter", "bubble", "radar", "other")
# Formats a vision model's image processor decodes. Vector metafiles (EMF,
# WMF), SVG and TIFF are rasterised by nothing here, so they are skipped.
READABLE_MEDIA_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"}
)
_SENT_MEDIA_TYPE = "image/png"
_JPEG_MEDIA_TYPE = "image/jpeg"
_JPEG_QUALITY = 90

# Bounds on what one reading may hold, so a confused reply stays small.
_MAX_DESCRIPTION_CHARACTERS = 1200
_MAX_TEXT_CHARACTERS = 6000
_MAX_CHARTS = 4
_MAX_TABLES = 4
_MAX_CHART_CATEGORIES = 200
_MAX_CHART_SERIES = 24
_MAX_TABLE_ROWS = 200
_MAX_TABLE_COLUMNS = 30
_MAX_LABEL_CHARACTERS = 300
_MAX_PAGE_CONTEXT_CHARACTERS = 600

SkipReason = Literal[
    "unreadable_format", "too_small", "too_large", "repeated", "duplicate", "over_limit"
]


@dataclass(frozen=True)
class FigurePolicy:
    """Which pictures are read, and with what model and budget."""

    model: str
    timeout_seconds: int
    max_output_tokens: int
    min_side_px: int
    min_bytes: int
    max_bytes: int
    repeat_limit: int
    max_per_document: int
    max_side_px: int
    seed: int
    parallelism: int = 1
    repeat_min_pixels: int = 1_000_000

    def fingerprint(self) -> dict[str, object]:
        """What decides a picture's reading, for cache keys."""

        template = get_prompt_template(FIGURE_READING_TASK)
        return {
            "model": self.model,
            "prompt": template.checksum,
            "contract": figure_reading_schema_fingerprint(),
            "max_side_px": self.max_side_px,
            "max_output_tokens": self.max_output_tokens,
        }


class FigureSeries(BaseModel):
    """One series of a chart read from a picture."""

    name: str = ""
    values: list[float | None] = Field(default_factory=list)


class FigureChart(BaseModel):
    """A chart as the model read it from a picture."""

    chart_type: str = "other"
    title: str = ""
    category_axis_title: str = ""
    value_axis_title: str = ""
    categories: list[str] = Field(default_factory=list)
    series: list[FigureSeries] = Field(default_factory=list)


class FigureTable(BaseModel):
    """A table as the model read it from a picture."""

    caption: str = ""
    header_rows: int = 0
    rows: list[list[str]] = Field(default_factory=list)


class FigureReading(BaseModel):
    """What the model read in one picture."""

    kind: FigureKind = "other"
    description: str = ""
    text: str = ""
    charts: list[FigureChart] = Field(default_factory=list)
    tables: list[FigureTable] = Field(default_factory=list)


ImageLoader = Callable[[ParsedAsset], bytes]
ReadingCache = Callable[[str], Mapping[str, Any] | None]
ReadingStore = Callable[[str, Mapping[str, Any]], None]


@dataclass
class FigureReport:
    """What a document's picture reading did, for its trace."""

    pictures: int = 0
    read: int = 0
    cached: int = 0
    failed: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    kinds: Counter[str] = field(default_factory=Counter)

    def as_metadata(self) -> dict[str, object]:
        """The report as plain JSON for provider metadata and traces."""

        return {
            "pictures": self.pictures,
            "read": self.read,
            "cached": self.cached,
            "failed": self.failed,
            "skipped": dict(sorted(self.skipped.items())),
            "kinds": dict(sorted(self.kinds.items())),
        }


def figure_candidates(
    document: ParsedDocument, policy: FigurePolicy
) -> tuple[list[ParsedAsset], Counter[str]]:
    """The pictures of a document worth reading, in page order, and why others are not.

    A picture is skipped when its format cannot be decoded, when it is too
    small to hold content or too large to send, when it appears on so many
    pages that it is the document's template, or when the same bytes were
    already chosen. What remains past the per-document limit is counted, not
    read.
    """

    images = [asset for asset in document.assets if asset.kind == "image"]
    pages_by_digest: dict[str, set[int]] = {}
    for asset in images:
        digest = _digest(asset)
        if digest:
            pages_by_digest.setdefault(digest, set()).add(asset.page_number or 0)
    chosen: list[ParsedAsset] = []
    seen: set[str] = set()
    skipped: Counter[str] = Counter()
    for asset in sorted(images, key=lambda item: (item.page_number or 0, item.id)):
        reason = _skip_reason(asset, policy, pages_by_digest, seen)
        if reason is None and len(chosen) >= policy.max_per_document:
            reason = "over_limit"
        if reason is not None:
            skipped[reason] += 1
            continue
        seen.add(_digest(asset))
        chosen.append(asset)
    return chosen, skipped


def _skip_reason(
    asset: ParsedAsset,
    policy: FigurePolicy,
    pages_by_digest: Mapping[str, set[int]],
    seen: set[str],
) -> SkipReason | None:
    metadata = asset.metadata
    media_type = str(metadata.get("content_type") or "").lower()
    size = _int(metadata.get("size_bytes"))
    width, height = _int(metadata.get("width_px")), _int(metadata.get("height_px"))
    digest = _digest(asset)
    checks: tuple[tuple[bool, SkipReason], ...] = (
        (media_type not in READABLE_MEDIA_TYPES, "unreadable_format"),
        (size is not None and size < policy.min_bytes, "too_small"),
        (size is not None and size > policy.max_bytes, "too_large"),
        (
            width is not None and height is not None and min(width, height) < policy.min_side_px,
            "too_small",
        ),
        (
            bool(digest)
            and len(pages_by_digest.get(digest, ())) >= policy.repeat_limit
            and not _large(width, height, policy),
            "repeated",
        ),
        (bool(digest) and digest in seen, "duplicate"),
    )
    return next((reason for failed, reason in checks if failed), None)


def _large(width: int | None, height: int | None, policy: FigurePolicy) -> bool:
    """Whether a picture is big enough to carry content even when it repeats.

    A logo or a footer band repeats on every page and is small; a picture as
    large as a page that repeats - a form, a mock-up, a screenshot shown again
    with annotations - still says something, and it is read once, on its first
    page, like any duplicate.
    """

    return width is not None and height is not None and width * height >= policy.repeat_min_pixels


def read_figures(  # noqa: PLR0913 - one document's reading context.
    document: ParsedDocument,
    *,
    llm: StructuredCompletionProvider,
    policy: FigurePolicy,
    load_image: ImageLoader,
    cached_reading: ReadingCache | None = None,
    store_reading: ReadingStore | None = None,
) -> tuple[ParsedDocument, FigureReport]:
    """Read a document's pictures and write what they show onto their pages.

    Each picture's reading is appended to the end of its page, so the offsets
    of everything the parser wrote stay where they were. A picture that cannot
    be loaded or read is counted and left out; it never fails the document.
    """

    candidates, skipped = figure_candidates(document, policy)
    report = FigureReport(
        pictures=sum(1 for asset in document.assets if asset.kind == "image"),
        skipped=skipped,
    )
    if not candidates:
        return _with_report(document, report), report

    def read_one(_index: int, asset: ParsedAsset) -> tuple[FigureReading | None, str]:
        digest = _digest(asset)
        cached = cached_reading(digest) if cached_reading is not None and digest else None
        if cached is not None:
            try:
                return FigureReading.model_validate(cached), "cached"
            except ValidationError:
                pass
        image = _prepared_image(load_image(asset), policy.max_side_px)
        result = llm.complete_structured(_reading_request(document, asset, image, policy))
        reading = FigureReading.model_validate(result.payload)
        if store_reading is not None and digest:
            store_reading(digest, reading.model_dump(mode="json"))
        return reading, "read"

    def contain(
        _index: int, asset: ParsedAsset, exc: Exception
    ) -> tuple[FigureReading | None, str]:
        logger.warning(
            "could not read picture %s of %s (%s: %s)",
            asset.id,
            document.source_uri,
            type(exc).__name__,
            str(exc)[:200],
        )
        return None, "failed"

    # Calls run in parallel; the report is counted from their results here,
    # in one thread, rather than by the calls themselves.
    outcomes = ordered_map(
        candidates, read_one, parallelism=max(1, policy.parallelism), contain=contain
    )
    for _reading, outcome in outcomes:
        if outcome == "cached":
            report.cached += 1
        elif outcome == "read":
            report.read += 1
        else:
            report.failed += 1
    readings = [reading for reading, _outcome in outcomes]
    by_page: dict[int, list[tuple[ParsedAsset, FigureReading]]] = {}
    for asset, reading in zip(candidates, readings, strict=True):
        if reading is None:
            continue
        report.kinds[reading.kind] += 1
        if reading.kind == "decoration":
            continue
        by_page.setdefault(asset.page_number or 0, []).append((asset, reading))
    pages = [
        _with_readings(page, by_page.get(page.page_number, []), document.file_id)
        for page in document.pages
    ]
    updated = document.model_copy(update={"pages": pages})
    return _with_report(updated, report), report


def _with_report(document: ParsedDocument, report: FigureReport) -> ParsedDocument:
    metadata = {**document.provider_metadata, "figure_reading": report.as_metadata()}
    return document.model_copy(update={"provider_metadata": metadata})


def _with_readings(
    page: ParsedPage,
    readings: list[tuple[ParsedAsset, FigureReading]],
    file_id: str,
) -> ParsedPage:
    """Append each reading of a page's pictures to the page, with its blocks."""

    if not readings:
        return page
    markdown = page.markdown
    raw_text = page.raw_text
    blocks = list(page.blocks)
    for asset, reading in readings:
        for kind, text, metadata in _reading_parts(reading):
            separator = "\n\n" if markdown.strip() else ""
            start = len(markdown) + len(separator)
            markdown = f"{markdown}{separator}{text}"
            raw_text = f"{raw_text}{separator if raw_text.strip() else ''}{text}"
            blocks.append(
                LayoutBlock(
                    id=stable_id("block", file_id, page.page_number, asset.id, kind, start),
                    page_number=page.page_number,
                    kind=kind,
                    text=text,
                    metadata={
                        **metadata,
                        "asset_id": asset.id,
                        "source": "vision",
                        "start_offset": start,
                        "end_offset": start + len(text),
                    },
                )
            )
    return page.model_copy(update={"markdown": markdown, "raw_text": raw_text, "blocks": blocks})


def _reading_parts(reading: FigureReading) -> list[tuple[str, str, dict[str, Any]]]:
    """A reading as page text: its description, its text, then its charts and tables."""

    parts: list[tuple[str, str, dict[str, Any]]] = []
    lines: list[str] = []
    description = " ".join(reading.description.split())
    if description:
        lines.append(f"Figure ({reading.kind}): {description}")
    text = "\n".join(line.rstrip() for line in reading.text.strip().splitlines() if line.strip())
    if text:
        lines.append(f"Text in figure:\n{text}")
    if lines:
        parts.append(("figure", "\n".join(lines), {"figure_kind": reading.kind}))
    for read_chart in reading.charts:
        chart = _chart(read_chart)
        if chart.series:
            parts.append(("chart", render_chart(chart), {"chart": chart.model_dump(mode="json")}))
    for read_table in reading.tables:
        table = TableData(
            caption=read_table.caption or None,
            header_rows=max(0, min(read_table.header_rows, len(read_table.rows))),
            rows=[[str(cell) for cell in row] for row in read_table.rows if row],
            source="vision",
        )
        if table.rows:
            parts.append(("table", render_table(table), {"table": table.model_dump(mode="json")}))
    return parts


def _chart(read: FigureChart) -> ChartData:
    return ChartData(
        chart_type=read.chart_type if read.chart_type in CHART_TYPES else "other",
        title=clean_cell(read.title) or None,
        categories=[clean_cell(category) for category in read.categories],
        series=[
            ChartSeries(name=clean_cell(series.name), values=list(series.values))
            for series in read.series
            if series.values
        ],
        category_axis_title=clean_cell(read.category_axis_title) or None,
        value_axis_title=clean_cell(read.value_axis_title) or None,
        source="vision",
        estimated=True,
    )


def _reading_request(
    document: ParsedDocument,
    asset: ParsedAsset,
    image: ImageInput,
    policy: FigurePolicy,
) -> StructuredCompletionRequest:
    template = get_prompt_template(FIGURE_READING_TASK)
    page = next((item for item in document.pages if item.page_number == asset.page_number), None)
    context = {
        "document": document.source_uri.rsplit("/", 1)[-1],
        "page": asset.page_number,
        "alt_text": str(asset.metadata.get("alt_text") or ""),
        "page_text": (page.markdown[:_MAX_PAGE_CONTEXT_CHARACTERS] if page is not None else ""),
    }
    return StructuredCompletionRequest(
        task_name=FIGURE_READING_TASK,
        model=policy.model,
        system=template.text,
        user=json.dumps({"page_context": context}, ensure_ascii=False),
        json_schema=figure_reading_schema(),
        timeout_seconds=policy.timeout_seconds,
        max_tokens=policy.max_output_tokens,
        seed=policy.seed,
        images=[image],
        prompt_metadata={
            "prompt_name": template.name,
            "prompt_revision": template.revision,
            "prompt_sha256": template.checksum,
        },
    )


def figure_reading_schema() -> dict[str, Any]:
    """The strict response contract of one picture reading."""

    label = {"type": "string", "maxLength": _MAX_LABEL_CHARACTERS}
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(FIGURE_KINDS)},
            "description": {"type": "string", "maxLength": _MAX_DESCRIPTION_CHARACTERS},
            "text": {"type": "string", "maxLength": _MAX_TEXT_CHARACTERS},
            "charts": {
                "type": "array",
                "maxItems": _MAX_CHARTS,
                "items": {
                    "type": "object",
                    "properties": {
                        "chart_type": {"type": "string", "enum": list(CHART_TYPES)},
                        "title": label,
                        "category_axis_title": label,
                        "value_axis_title": label,
                        "categories": {
                            "type": "array",
                            "maxItems": _MAX_CHART_CATEGORIES,
                            "items": label,
                        },
                        "series": {
                            "type": "array",
                            "maxItems": _MAX_CHART_SERIES,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": label,
                                    "values": {
                                        "type": "array",
                                        "maxItems": _MAX_CHART_CATEGORIES,
                                        "items": {"type": ["number", "null"]},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "tables": {
                "type": "array",
                "maxItems": _MAX_TABLES,
                "items": {
                    "type": "object",
                    "properties": {
                        "caption": label,
                        "header_rows": {"type": "integer", "minimum": 0},
                        "rows": {
                            "type": "array",
                            "maxItems": _MAX_TABLE_ROWS,
                            "items": {
                                "type": "array",
                                "maxItems": _MAX_TABLE_COLUMNS,
                                "items": label,
                            },
                        },
                    },
                },
            },
        },
    }


def figure_reading_schema_fingerprint() -> str:
    """A stable digest of the response contract, for cache keys."""

    return stable_id(
        "figure_contract", json.dumps(figure_reading_schema(), sort_keys=True), length=16
    )


def _prepared_image(payload: bytes, max_side_px: int) -> ImageInput:
    """Decode a picture, scale it to the longest side allowed, and encode it to send.

    Pillow's decompression-bomb guard stays on: a picture that claims more
    pixels than it allows raises before it is decoded.
    """

    from PIL import Image  # noqa: PLC0415 - only pictures pay for the imaging library.

    with Image.open(io.BytesIO(payload)) as opened:
        opened.seek(0)
        image = opened.convert("RGBA") if opened.mode in {"P", "LA", "PA"} else opened.copy()
    if max(image.size) > max_side_px:
        image.thumbnail((max_side_px, max_side_px))
    buffer = io.BytesIO()
    if image.mode in {"RGBA", "LA"}:
        image.save(buffer, format="PNG", optimize=True)
        media_type = _SENT_MEDIA_TYPE
    else:
        image.convert("RGB").save(buffer, format="JPEG", quality=_JPEG_QUALITY)
        media_type = _JPEG_MEDIA_TYPE
    return ImageInput(
        media_type=media_type, data_base64=base64.b64encode(buffer.getvalue()).decode("ascii")
    )


def _digest(asset: ParsedAsset) -> str:
    return str(asset.metadata.get("sha256") or "")


def _int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
