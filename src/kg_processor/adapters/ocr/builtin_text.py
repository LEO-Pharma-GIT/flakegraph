# SPDX-License-Identifier: Apache-2.0
"""Local parser for text-native documents: text, PDF, Office and HTML.

It reads what a file states without a model or OCR. Office files keep their
structure: slides read title first and then in reading order, tables become
table blocks (``domain.tables``), charts are read from the values their parts
cache (``office_charts``), speaker notes follow their slide, and pictures are
listed as assets whose bytes ``read_image_bytes`` returns on request.
"""

from __future__ import annotations

import codecs
import importlib
import math
import multiprocessing
import re
import subprocess
import sys
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from html.parser import HTMLParser
from itertools import pairwise
from multiprocessing.connection import Connection
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any
from xml.etree.ElementTree import Element

import pypdfium2 as pdfium  # type: ignore[import-untyped]
from defusedxml import ElementTree
from pypdf import PdfReader
from pypdf.errors import LimitReachedError

from kg_processor.adapters.ocr.office_charts import EmptyChartError, read_chart
from kg_processor.adapters.ocr.office_images import (
    content_type_for_member,
    image_facts,
    pdf_image_bytes,
    pdf_page_images,
)
from kg_processor.adapters.ocr.page_builder import BLOCK_BREAK, LINE_BREAK, PageBuilder
from kg_processor.adapters.ocr.table_structure import (
    GridCell,
    HtmlTableCollector,
    build_rows,
    table_from_rows,
)
from kg_processor.domain.documents import (
    InputFile,
    ParsedAsset,
    ParsedDocument,
    ParsedPage,
)
from kg_processor.domain.ids import stable_id
from kg_processor.domain.tables import ChartData, TableData, render_chart, render_table
from kg_processor.ports.ocr import NoTextReadError, OcrOptions, parse_page_range

MAX_OFFICE_ENTRY_BYTES = 25 * 1024 * 1024
MAX_OFFICE_TOTAL_BYTES = 200 * 1024 * 1024
MAX_TEXT_BYTES = 50 * 1024 * 1024
MAX_PDF_SOURCE_BYTES = 200 * 1024 * 1024
MAX_PDFIUM_FALLBACK_MEMORY_BYTES = 1024 * 1024 * 1024
MAX_PDFIUM_FALLBACK_TEXT_BYTES = 50 * 1024 * 1024
MAX_PDFIUM_FALLBACK_PAGES = 10_000
PDFIUM_FALLBACK_TIMEOUT_SECONDS = 120
MAX_PDF_RSS_SAMPLE_FAILURES = 3
MAX_OFFICE_XML_DEPTH = 256
# Two-column detection: a page needs this many text lines to be judged, runs on
# one baseline sit within these points of each other, the gutter is this share
# of the page width, and an interleaved page crosses it at least this often and
# on at least this share of its lines.
_MIN_COLUMN_LINES = 10
_SAME_LINE_POINTS = 2.0
_COLUMN_GUTTER_FRACTION = 0.04
_MIN_INTERLEAVED_CROSSINGS = 6
_INTERLEAVED_LINE_SHARE = 0.3
_OFFICE_READ_CHUNK_BYTES = 1024 * 1024
_ENCODING_SNIFF_BYTES = 4096
_UTF32_MIN_SNIFF_BYTES = 8
_UTF16_MIN_SNIFF_BYTES = 4
_UTF32_ZERO_LANE_RATIO = 0.8
_UTF16_ZERO_LANE_RATIO = 0.6
_UTF16_NONZERO_LANE_RATIO = 0.2
_CONTENT_CHARSET_RE = re.compile(r"\bcharset\s*=\s*([A-Za-z0-9._:-]+)", re.IGNORECASE)
_TEXT_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


class _EncryptedPdfError(ValueError):
    """Identify password-protected PDFs without suppressing parser fallback errors."""


class BuiltinTextOcrProvider:
    """Deterministic local parser for tests and text-native documents.

    This adapter does not attempt image OCR. It is intentionally explicit so
    production jobs can choose MinerU, Tesseract, or Snowflake instead of
    silently falling back to basic text extraction.
    """

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        """Parse text-native formats into pages of offset-bearing blocks.

        Office files keep their tables, charts, speaker notes and pictures as
        structured blocks and assets (see ``_DocumentParts``); how many of
        each were found is recorded in the provider metadata.
        """

        suffix = file.path.suffix.lower()
        layout: dict[str, float] = {}
        parts = _DocumentParts(file)
        if suffix in {".txt", ".md", ".markdown"}:
            pages = [_page_from_text(file, 1, _read_text_document(file.path))]
        elif suffix == ".pdf":
            pages, interleaved_ratio = self._parse_pdf(file.path, file, parts)
            if interleaved_ratio is not None:
                layout["interleaved_column_page_ratio"] = interleaved_ratio
        elif suffix == ".docx":
            pages = self._parse_docx(file.path, file, parts)
        elif suffix in {".html", ".htm"}:
            pages = self._parse_html(file.path, file, parts)
        elif suffix == ".pptx":
            pages = self._parse_pptx(file.path, file, parts)
        elif suffix == ".xlsx":
            pages = self._parse_xlsx(file.path, file, parts)
        else:
            raise ValueError(f"builtin_text OCR does not support {suffix}")
        pages = _select_pages(pages, options.page_range)
        # A scanned PDF has pages and no text layer. Returning it as a parsed
        # document would record the file as successfully OCR'd while it
        # contributes nothing, so the empty result is a failure the caller can
        # route to a parser that reads pixels. A text-native file with nothing
        # in it has nowhere to be routed: it is simply an empty document, and
        # the pipeline records it as one.
        if suffix == ".pdf" and not any(
            page.raw_text.strip() or page.markdown.strip() for page in pages
        ):
            raise NoTextReadError(f"builtin_text extracted no text from {file.source_uri}")
        kept_pages = {page.page_number for page in pages}
        return ParsedDocument(
            file_id=file.id,
            checksum=file.checksum,
            source_uri=file.source_uri,
            mime_type=file.mime_type,
            pages=pages,
            assets=[asset for asset in parts.assets if asset.page_number in kept_pages],
            provider_metadata={
                "provider": "builtin_text",
                "language": options.language,
                "page_range": options.page_range,
                **layout,
                **parts.counts(),
            },
        )

    def read_picture(self, file: InputFile, asset: ParsedAsset) -> bytes:
        """Return the bytes of a picture this parser reported for ``file``.

        Raises ``ValueError`` for an asset another parser produced or one
        that cannot be found in the file, and keeps the parser's size limits.
        """

        if asset.metadata.get("parser") != _ASSET_PARSER:
            raise ValueError(f"asset {asset.id} was not reported by builtin_text")
        return read_image_bytes(file.path, asset)

    def _parse_pdf(
        self, path: Path, file: InputFile, parts: _DocumentParts
    ) -> tuple[list[ParsedPage], float | None]:
        """Extract a PDF with pypdf, falling back to PDFium for unusual fonts.

        Both engines preserve one output page per physical PDF page. PDFium is
        intentionally a text fallback rather than OCR: malformed font metadata
        should not force a text-native document through a slower OCR provider or
        collapse its page provenance.

        The second value is the share of measurable pages whose text layer runs
        across two columns line by line (see ``_interleaved_columns``), or None
        when PDFium read the file and no positions were available.

        Embedded pictures are listed on the pypdf path only, as image assets
        with a figure block that has no text: a PDF picture has no alt text
        the parser could write.
        """

        if path.stat().st_size > MAX_PDF_SOURCE_BYTES:
            raise ValueError(f"PDF exceeds the {MAX_PDF_SOURCE_BYTES} byte source limit")
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted and not reader.decrypt(""):
                raise _EncryptedPdfError(
                    "password-protected PDFs are not supported by builtin_text OCR"
                )
            if len(reader.pages) > MAX_PDFIUM_FALLBACK_PAGES:
                raise ValueError(f"PDF exceeds the {MAX_PDFIUM_FALLBACK_PAGES} page limit")
            pages: list[ParsedPage] = []
            total_text_bytes = 0
            measured_pages = interleaved_pages = 0
            for index, page in enumerate(reader.pages, start=1):
                runs: list[tuple[float, float]] = []
                text = page.extract_text(visitor_text=_text_run_collector(runs)) or ""
                interleaved = _interleaved_columns(runs, float(page.mediabox.width))
                if interleaved is not None:
                    measured_pages += 1
                    interleaved_pages += interleaved
                total_text_bytes += len(text.encode("utf-8"))
                if total_text_bytes > MAX_PDFIUM_FALLBACK_TEXT_BYTES:
                    raise ValueError(
                        f"PDF text exceeds the {MAX_PDFIUM_FALLBACK_TEXT_BYTES} byte limit"
                    )
                segments: list[_Segment] = [_Segment("lines", text)]
                segments.extend(_pdf_picture_segments(parts, page, index))
                pages.append(_page_from_segments(file, index, segments))
            return pages, (interleaved_pages / measured_pages if measured_pages else 0.0)
        except (_EncryptedPdfError, LimitReachedError):
            raise
        except Exception as primary_error:
            parts.discard_assets()
            try:
                return self._parse_pdf_with_pdfium(path, file), None
            except Exception as fallback_error:
                raise RuntimeError(
                    f"builtin_text failed to parse PDF {path}: "
                    f"{type(primary_error).__name__}; PDFium fallback: "
                    f"{type(fallback_error).__name__}"
                ) from fallback_error

    def _parse_pdf_with_pdfium(self, path: Path, file: InputFile) -> list[ParsedPage]:
        """Extract PDFium text in a resource-limited subprocess."""

        page_text = _run_guarded_pdfium_fallback(path)
        return [_page_from_text(file, index, text) for index, text in enumerate(page_text, start=1)]

    def _parse_docx(self, path: Path, file: InputFile, parts: _DocumentParts) -> list[ParsedPage]:
        """Read a Word document as one page: body, then headers and footers.

        Tables become table blocks, and charts and pictures drawn in a
        paragraph follow that paragraph's text.
        """

        with zipfile.ZipFile(path) as archive:
            member_names = _docx_text_member_names(archive)
            _validate_office_archive(path, archive, member_names)
            package = _OfficePackage(path, archive, parts, counted=member_names)
            segments: list[_Segment] = []
            for name in member_names:
                root = ElementTree.fromstring(_read_zip_member(path, archive, name))
                body = next((child for child in root if _local_name(child.tag) == "body"), root)
                context = _DocxContext(package, name, page_number=1)
                segments.extend(_docx_block_segments(body, context))
        return [_page_from_segments(file, 1, segments)]

    def _parse_html(self, path: Path, file: InputFile, parts: _DocumentParts) -> list[ParsedPage]:
        """Read an HTML page's visible text, with its tables as table blocks."""

        parser = _VisibleTextHtmlParser()
        parser.feed(_read_text_document(path, sniff_html_charset=True))
        parser.close()
        segments = parser.segments()
        parts.tables += sum(segment.kind == "table" for segment in segments)
        return [_page_from_segments(file, 1, segments)]

    def _parse_pptx(self, path: Path, file: InputFile, parts: _DocumentParts) -> list[ParsedPage]:
        """Read each slide in presentation order as one page.

        See ``_pptx_slide_segments`` for what a slide page holds and in what
        order.
        """

        pages: list[ParsedPage] = []
        with zipfile.ZipFile(path) as archive:
            slide_names, control_names = _pptx_slide_member_names(path, archive)
            _validate_office_archive(path, archive, [*control_names, *slide_names])
            package = _OfficePackage(path, archive, parts, counted=[*control_names, *slide_names])
            for index, name in enumerate(slide_names, start=1):
                root = ElementTree.fromstring(_read_zip_member(path, archive, name))
                segments = _pptx_slide_segments(package, name, root, index)
                pages.append(_page_from_segments(file, index, segments))
        if not pages:
            raise ValueError(f"PPTX file contains no slides: {path}")
        return pages

    def _parse_xlsx(self, path: Path, file: InputFile, parts: _DocumentParts) -> list[ParsedPage]:
        """Read each worksheet as one page: its cells, then its charts.

        The cell text is exactly what ``_xlsx_sheet_text`` writes; charts and
        pictures the sheet's drawing holds follow it as blocks of their own.
        """

        pages: list[ParsedPage] = []
        with zipfile.ZipFile(path) as archive:
            sheet_names, shared_string_names, control_names = _xlsx_member_names(path, archive)
            readable = [*control_names, *shared_string_names, *sheet_names]
            _validate_office_archive(path, archive, readable)
            package = _OfficePackage(path, archive, parts, counted=readable)
            shared_strings = _xlsx_shared_strings(
                path,
                archive,
                shared_string_names[0] if shared_string_names else None,
            )
            for index, name in enumerate(sheet_names, start=1):
                text = _xlsx_sheet_text(_read_zip_member(path, archive, name), shared_strings)
                segments = [_Segment("lines", text), *_xlsx_drawing_segments(package, name, index)]
                pages.append(_page_from_segments(file, index, segments))
        if not pages:
            raise ValueError(f"XLSX file contains no worksheets: {path}")
        return pages


def _text_run_collector(runs: list[tuple[float, float]]) -> Callable[..., None]:
    """Record where each non-blank text run starts, in content-stream order."""

    def visit(text: str, cm: list[float], tm: list[float], *_: object) -> None:
        if text.strip():
            runs.append(
                (
                    tm[4] * cm[0] + tm[5] * cm[2] + cm[4],
                    tm[4] * cm[1] + tm[5] * cm[3] + cm[5],
                )
            )

    return visit


def _interleaved_columns(runs: list[tuple[float, float]], width: float) -> bool | None:
    """Tell whether a page's text layer interleaves two columns line by line.

    A PDF may draw a two-column page row by row, so the text layer joins each
    left-column line to the right-column line beside it and no sentence
    survives extraction. In content-stream order such a page moves from the
    left half to the right half on the same baseline again and again, where a
    page drawn column by column crosses once. None when the page has too few
    lines to judge.
    """

    lines = {round(y) for _, y in runs}
    if width <= 0 or len(lines) < _MIN_COLUMN_LINES:
        return None
    middle = width / 2
    gutter = width * _COLUMN_GUTTER_FRACTION
    crossings = sum(
        1
        for (x1, y1), (x2, y2) in pairwise(runs)
        if abs(y1 - y2) <= _SAME_LINE_POINTS and x1 < middle - gutter and x2 >= middle
    )
    return crossings >= _MIN_INTERLEAVED_CROSSINGS and crossings >= _INTERLEAVED_LINE_SHARE * len(
        lines
    )


def _run_guarded_pdfium_fallback(path: Path) -> list[str]:
    """Run permissive PDFium parsing outside the worker's memory boundary."""

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_pdfium_fallback_worker,
        args=(str(path), child),
        daemon=True,
    )
    process.start()
    child.close()
    try:
        deadline = monotonic() + PDFIUM_FALLBACK_TIMEOUT_SECONDS
        unreadable_rss_samples = 0
        while not parent.poll(0.05):
            if not process.is_alive():
                break
            resident_bytes = _process_resident_memory_bytes(process.pid)
            if resident_bytes is None:
                unreadable_rss_samples += 1
                if unreadable_rss_samples >= MAX_PDF_RSS_SAMPLE_FAILURES:
                    process.terminate()
                    raise RuntimeError("Unable to enforce the PDFium fallback memory limit")
                continue
            unreadable_rss_samples = 0
            if resident_bytes > MAX_PDFIUM_FALLBACK_MEMORY_BYTES:
                process.terminate()
                raise MemoryError(
                    "PDFium fallback exceeded the "
                    f"{MAX_PDFIUM_FALLBACK_MEMORY_BYTES} byte memory limit"
                )
            if monotonic() >= deadline:
                process.terminate()
                raise TimeoutError(
                    f"PDFium fallback exceeded {PDFIUM_FALLBACK_TIMEOUT_SECONDS} seconds"
                )
        try:
            status, payload = parent.recv()
        except EOFError as exc:
            raise RuntimeError("PDFium fallback worker exited under its resource limit") from exc
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
    if status != "ok":
        raise RuntimeError(str(payload))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise RuntimeError("PDFium fallback worker returned an invalid result")
    return payload


def _pdfium_fallback_worker(path: str, connection: Connection) -> None:
    """Extract bounded page text and return it through a one-way process pipe."""

    try:
        _apply_pdfium_resource_limits()
        document = pdfium.PdfDocument(path)
        try:
            if len(document) > MAX_PDFIUM_FALLBACK_PAGES:
                raise ValueError(
                    f"PDF contains {len(document)} pages, above the "
                    f"{MAX_PDFIUM_FALLBACK_PAGES} page fallback limit"
                )
            pages: list[str] = []
            total_bytes = 0
            for index in range(len(document)):
                page = document[index]
                text_page = page.get_textpage()
                try:
                    text = text_page.get_text_range()
                finally:
                    text_page.close()
                    page.close()
                total_bytes += len(text.encode("utf-8"))
                if total_bytes > MAX_PDFIUM_FALLBACK_TEXT_BYTES:
                    raise ValueError(
                        "PDFium fallback text exceeds the "
                        f"{MAX_PDFIUM_FALLBACK_TEXT_BYTES} byte limit"
                    )
                pages.append(text)
        finally:
            document.close()
        connection.send(("ok", pages))
    except BaseException as exc:  # child process must serialize all parser failures
        with suppress(BaseException):
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _apply_pdfium_resource_limits() -> None:
    """Cap address space and CPU before loading attacker-controlled PDF streams."""

    try:
        resource = importlib.import_module("resource")
    except ModuleNotFoundError as exc:
        raise RuntimeError("PDFium fallback requires OS resource-limit support") from exc
    # Darwin reserves a very large virtual address range before user code starts
    # and rejects practical RLIMIT_AS values. The parent process's resident-memory
    # watchdog remains authoritative on that platform.
    with suppress(OSError, ValueError):
        resource.setrlimit(
            resource.RLIMIT_AS,
            (MAX_PDFIUM_FALLBACK_MEMORY_BYTES, MAX_PDFIUM_FALLBACK_MEMORY_BYTES),
        )
    cpu_limit = max(PDFIUM_FALLBACK_TIMEOUT_SECONDS - 5, 1)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))


def _process_resident_memory_bytes(process_id: int | None) -> int | None:
    """Read one child process's RSS without adding a runtime dependency."""

    if process_id is None:
        return None
    if sys.platform.startswith("linux"):
        try:
            status = Path(f"/proc/{process_id}/status").read_text(encoding="ascii")
        except OSError:
            return None
        match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
        return int(match.group(1)) * 1024 if match else None
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["/bin/ps", "-o", "rss=", "-p", str(process_id)],
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
            return int(result.stdout.strip()) * 1024 if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    return None


def _page_from_text(file: InputFile, page_number: int, text: str) -> ParsedPage:
    """Convert plain text into one page with offset-bearing layout blocks.

    The normalized page text and every block are built from the same rendered
    lines, ensuring evidence offsets point into the exact text later chunked by
    the pipeline.
    """

    return _page_from_segments(file, page_number, [_Segment("lines", text)])


@dataclass
class _Segment:
    """One piece of a page before it is written: text lines or a structure.

    ``lines`` is running text, one block per non-blank line; ``table``,
    ``chart`` and ``notes`` are blocks set off by a blank line; ``figure``
    is a picture, written as "Figure: <alt text>" only when it has alt text.
    """

    kind: str
    text: str = ""
    table: TableData | None = None
    chart: ChartData | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Segments that sit on the page as blocks of their own, set off by blank lines.
_STRUCTURED_SEGMENTS = frozenset({"table", "chart", "notes"})
_FIGURE_PREFIX = "Figure: "
_NOTES_HEADING = "Speaker notes:"


def _page_from_segments(file: InputFile, page_number: int, segments: list[_Segment]) -> ParsedPage:
    """Write a page's segments in order, each block over its exact span.

    Every block's ``start_offset``/``end_offset`` is where the page builder
    wrote its text, so ``page.markdown[start:end] == block.text`` always.
    """

    builder = PageBuilder(file.id, page_number)
    previous_structured = False
    for segment in segments:
        structured = segment.kind in _STRUCTURED_SEGMENTS
        separator = BLOCK_BREAK if structured or previous_structured else LINE_BREAK
        if segment.kind == "lines":
            written = _write_lines(builder, segment.text, separator)
        elif segment.kind == "figure":
            written = _write_figure(builder, segment, separator)
        else:
            block = _structured_block(segment)
            written = block is not None and (
                builder.add(block[0], block[1], block[2], separator=separator) is not None
            )
        if written:
            previous_structured = structured
    return builder.page()


def _write_lines(builder: PageBuilder, text: str, separator: str) -> bool:
    """Write running text one classified block per non-blank line."""

    written = False
    for raw_line in text.lstrip("\ufeff").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        kind, rendered, metadata = _classify_text_line(line)
        builder.add(kind, rendered, metadata, separator=LINE_BREAK if written else separator)
        written = True
    return written


def _write_figure(builder: PageBuilder, segment: _Segment, separator: str) -> bool:
    """Write "Figure: <alt text>", or mark where a picture without alt text sits."""

    alt = " ".join(segment.text.split())
    if not alt:
        builder.mark("figure", segment.metadata)
        return False
    builder.add("figure", alt, segment.metadata, separator=separator, prefix=_FIGURE_PREFIX)
    return True


def _structured_block(segment: _Segment) -> tuple[str, str, dict[str, Any]] | None:
    """The kind, rendered text and metadata of a table, chart or notes segment."""

    if segment.kind == "table" and segment.table is not None:
        table = segment.table
        return "table", render_table(table), {"table": table.model_dump(mode="json")}
    if segment.kind == "chart" and segment.chart is not None:
        chart = segment.chart
        return "chart", render_chart(chart), {"chart": chart.model_dump(mode="json")}
    if segment.kind == "notes":
        return "notes", segment.text, dict(segment.metadata)
    return None


def _classify_text_line(line: str) -> tuple[str, str, dict[str, object]]:
    """Classify one normalized line as heading, table row, list, or paragraph.

    Lightweight structural metadata gives smoke and text-native inputs useful
    layout semantics without pretending to perform visual OCR.
    """

    heading = re.fullmatch(r"(#{1,6})\s+(.+)", line)
    if heading:
        level = len(heading.group(1))
        return "heading", line, {"level": level}
    if "\t" in line:
        cells = [re.sub(r"\s+", " ", value).strip() for value in line.split("\t")]
        rendered = " | ".join(cells)
        return "table_row", rendered, {"cells": cells}
    if re.match(r"^(?:[-*+] |\d+[.)] )", line):
        return "list_item", line, {}
    return "paragraph", re.sub(r"\s+", " ", line), {}


# Pictures are reported as assets that name this parser, so ``read_picture``
# can refuse an asset another parser produced.
_ASSET_PARSER = "builtin_text"
# Reading every picture in a package to hash it decompresses this much at most;
# pictures past it are listed without a hash or pixel size.
MAX_OFFICE_MEDIA_TOTAL_BYTES = 1024 * 1024 * 1024
_RELATIONSHIP_TYPE_SUFFIXES = {
    "chart": "/chart",
    "image": "/image",
    "notes": "/notesSlide",
    "layout": "/slideLayout",
    "master": "/slideMaster",
    "drawing": "/drawing",
    "diagram_data": "/diagramData",
}


class _OfficeBudgetError(ValueError):
    """The aggregate XML limit was reached: the document fails, not one part."""


class _DocumentParts:
    """What one parse found besides text: assets, and counts of structures.

    The counts land in the provider metadata, so a run report can say how
    many charts a corpus held and how many of them could not be read.
    """

    def __init__(self, file: InputFile) -> None:
        """Start an empty record for ``file``."""

        self.file = file
        self.assets: list[ParsedAsset] = []
        self.charts = 0
        self.charts_unreadable = 0
        self.charts_empty = 0
        self.tables = 0
        self.notes_pages = 0
        self.images_unreadable = 0
        self.parts_unreadable = 0

    def add_asset(self, page_number: int, metadata: dict[str, Any]) -> ParsedAsset:
        """Record one picture on ``page_number`` and return its asset."""

        asset = ParsedAsset(
            id=stable_id(
                "asset",
                self.file.id,
                page_number,
                len(self.assets),
                str(metadata.get("member", "")),
            ),
            kind="image",
            page_number=page_number,
            metadata={"parser": _ASSET_PARSER, **metadata},
        )
        self.assets.append(asset)
        return asset

    def discard_assets(self) -> None:
        """Forget assets from a parse that is being redone another way."""

        self.assets.clear()
        self.images_unreadable = 0

    def counts(self) -> dict[str, int]:
        """The non-zero counts, for the provider metadata."""

        counts = {
            "charts": self.charts,
            "charts_unreadable": self.charts_unreadable,
            "charts_empty": self.charts_empty,
            "tables": self.tables,
            "notes_pages": self.notes_pages,
            "images": len(self.assets),
            "images_unreadable": self.images_unreadable,
            "parts_unreadable": self.parts_unreadable,
        }
        return {key: value for key, value in counts.items() if value}


class _OfficePackage:
    """Bounded access to the parts of one open Office package.

    Every XML part read through it counts toward ``MAX_OFFICE_TOTAL_BYTES``
    together with the parts the caller already validated, and every part
    keeps the per-entry limit, so following relationships to charts, notes
    and layouts cannot read past the limits the parser always applied.
    Picture bytes have a budget of their own (``MAX_OFFICE_MEDIA_TOTAL_BYTES``).
    """

    def __init__(
        self,
        path: Path,
        archive: zipfile.ZipFile,
        parts: _DocumentParts,
        *,
        counted: list[str],
    ) -> None:
        """Wrap ``archive``; ``counted`` names parts already in the total."""

        self.path = path
        self.archive = archive
        self.parts = parts
        self.names = set(archive.namelist())
        self._counted: set[str] = set()
        self._total = 0
        for name in counted:
            self._count(name)
        self._relationships: dict[str, dict[str, tuple[str, str]]] = {}
        self._roots: dict[str, Element | None] = {}
        self._media: dict[str, Any] = {}
        self._media_total = 0
        self._content_types: dict[str, str] | None = None

    def xml(self, name: str) -> Element | None:
        """Parse an auxiliary part once; None if missing or unreadable.

        A chart, notes page or layout that cannot be read costs that part,
        not the document. The aggregate limit still fails the document.
        """

        if name in self._roots:
            return self._roots[name]
        root: Element | None = None
        if name in self.names:
            try:
                self._count(name)
                root = ElementTree.fromstring(_read_zip_member(self.path, self.archive, name))
            except _OfficeBudgetError:
                raise
            except Exception:  # noqa: BLE001 - one unreadable part must not fail the document
                self.parts.parts_unreadable += 1
                root = None
        self._roots[name] = root
        return root

    def read_xml_bytes(self, name: str) -> bytes:
        """Read an auxiliary XML part's bytes under the same limits as ``xml``."""

        self._count(name)
        return _read_zip_member(self.path, self.archive, name)

    def relationships(self, source: str) -> dict[str, tuple[str, str]]:
        """A part's relationships by id, empty when it has none or they are unreadable."""

        if source not in self._relationships:
            rels = _relationships_member(source)
            found: dict[str, tuple[str, str]] = {}
            if rels in self.names:
                try:
                    self._count(rels)
                    found = _office_relationships(self.path, self.archive, source, rels)
                except _OfficeBudgetError:
                    raise
                except Exception:  # noqa: BLE001 - unreadable links lose only what they link
                    self.parts.parts_unreadable += 1
            self._relationships[source] = found
        return self._relationships[source]

    def target(self, source: str, relationship_id: str | None, kind: str) -> str | None:
        """The part ``relationship_id`` of ``source`` names, if it is of ``kind``."""

        if not relationship_id:
            return None
        found = self.relationships(source).get(relationship_id)
        if found is None or found[0] not in self.names:
            return None
        member, relationship_type = found
        return member if relationship_type.endswith(_RELATIONSHIP_TYPE_SUFFIXES[kind]) else None

    def first_target(self, source: str, kind: str) -> str | None:
        """The first part of ``kind`` that ``source`` links to."""

        suffix = _RELATIONSHIP_TYPE_SUFFIXES[kind]
        return next(
            (
                member
                for member, relationship_type in self.relationships(source).values()
                if relationship_type.endswith(suffix) and member in self.names
            ),
            None,
        )

    def chart(self, member: str | None) -> ChartData | None:
        """Read one chart part, counting it as read or unreadable."""

        if member is None:
            self.parts.charts_unreadable += 1
            return None
        try:
            chart = read_chart(self.read_xml_bytes(member))
        except _OfficeBudgetError:
            raise
        except EmptyChartError:
            # An empty chart is read correctly; it just plots nothing.
            self.parts.charts_empty += 1
            return None
        except Exception:  # noqa: BLE001 - an unreadable chart must not fail the document
            self.parts.charts_unreadable += 1
            return None
        self.parts.charts += 1
        return chart

    def picture(
        self,
        page_number: int,
        member: str | None,
        *,
        alt_text: str,
        extra: dict[str, Any],
    ) -> _Segment | None:
        """Record a picture part as an asset; its figure segment, or None.

        The picture is read once per package to hash it and read its pixel
        size from its header. One past the entry limit or the media budget is
        listed with its size alone.
        """

        if member is None:
            return None
        metadata: dict[str, Any] = {
            "member": member,
            "content_type": self._content_type(member),
            **self._media_facts(member),
            "alt_text": alt_text or None,
            **extra,
        }
        asset = self.parts.add_asset(page_number, metadata)
        return _Segment("figure", alt_text, metadata={"asset_id": asset.id})

    def _media_facts(self, member: str) -> dict[str, Any]:
        if member in self._media:
            return dict(self._media[member])
        info = self.archive.getinfo(member)
        facts: dict[str, Any] = {
            "sha256": None,
            "size_bytes": info.file_size,
            "width_px": None,
            "height_px": None,
        }
        if (
            info.file_size <= MAX_OFFICE_ENTRY_BYTES
            and self._media_total + info.file_size <= MAX_OFFICE_MEDIA_TOTAL_BYTES
        ):
            self._media_total += info.file_size
            try:
                found = image_facts(_read_zip_member(self.path, self.archive, member))
            except ValueError:
                self.parts.images_unreadable += 1
            else:
                facts.update(
                    sha256=found.sha256,
                    size_bytes=found.size_bytes,
                    width_px=found.width_px,
                    height_px=found.height_px,
                )
        else:
            self.parts.images_unreadable += 1
        self._media[member] = facts
        return dict(facts)

    def _content_type(self, member: str) -> str | None:
        if self._content_types is None:
            self._content_types = {}
            root = self.xml("[Content_Types].xml")
            for node in root if root is not None else []:
                content_type = (node.get("ContentType") or "").strip().lower()
                if not content_type:
                    continue
                if _local_name(node.tag) == "Default" and node.get("Extension"):
                    self._content_types[str(node.get("Extension")).lower()] = content_type
                elif _local_name(node.tag) == "Override" and node.get("PartName"):
                    self._content_types[str(node.get("PartName"))] = content_type
        found = content_type_for_member(member, self._content_types)
        return found.lower() if found else None

    def _count(self, name: str) -> None:
        if name in self._counted or name not in self.names:
            return
        info = self.archive.getinfo(name)
        _validate_zip_member_size(self.path, info)
        self._counted.add(name)
        self._total += info.file_size
        if self._total > MAX_OFFICE_TOTAL_BYTES:
            raise _OfficeBudgetError(
                f"Office file {self.path} contains {self._total} bytes of readable XML, "
                f"above the {MAX_OFFICE_TOTAL_BYTES} byte aggregate limit"
            )


def _relationships_member(source: str) -> str:
    """The relationships part of ``source``: ``dir/_rels/name.rels``."""

    parent = PurePosixPath(source).parent
    rels = PurePosixPath("_rels") / f"{PurePosixPath(source).name}.rels"
    return str(parent / rels) if str(parent) not in {"", "."} else str(rels)


def read_image_bytes(path: Path, asset: ParsedAsset) -> bytes:
    """Return the bytes of a picture ``builtin_text`` reported for ``path``.

    An Office picture is its package part, read under the same per-entry
    limit as every part the parser reads. A PDF picture is decoded from the
    page that draws it and returned as JPEG or PNG, as its ``content_type``
    says. Raises ``ValueError`` when the asset does not locate a picture in
    the file.
    """

    member = asset.metadata.get("member")
    if not isinstance(member, str) or not member:
        raise ValueError(f"asset {asset.id} names no picture")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_image(path, asset)
    if suffix not in {".docx", ".pptx", ".xlsx"}:
        raise ValueError(f"builtin_text reports no pictures in {suffix} files")
    with zipfile.ZipFile(path) as archive:
        if member not in set(archive.namelist()):
            raise ValueError(f"Office file {path} has no part {member}")
        return _read_zip_member(path, archive, member)


def _read_pdf_image(path: Path, asset: ParsedAsset) -> bytes:
    page_number = asset.metadata.get("pdf_page_number")
    names = asset.metadata.get("pdf_xobject_path")
    if (
        not isinstance(page_number, int)
        or not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) for name in names)
    ):
        raise ValueError(f"asset {asset.id} does not locate a PDF image")
    if path.stat().st_size > MAX_PDF_SOURCE_BYTES:
        raise ValueError(f"PDF exceeds the {MAX_PDF_SOURCE_BYTES} byte source limit")
    reader = PdfReader(str(path))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("password-protected PDFs are not supported by builtin_text OCR")
    if not 1 <= page_number <= len(reader.pages):
        raise ValueError(f"PDF has no page {page_number}")
    payload, _content_type = pdf_image_bytes(reader.pages[page_number - 1], tuple(names))
    return payload


def _pdf_picture_segments(parts: _DocumentParts, page: Any, page_number: int) -> list[_Segment]:
    """Image assets for the pictures one PDF page holds, found without decoding.

    A page whose resources cannot be walked loses its pictures, not its text.
    """

    segments: list[_Segment] = []
    try:
        found = list(pdf_page_images(page))
    except Exception:  # noqa: BLE001 - broken resources cost the pictures only
        parts.images_unreadable += 1
        return segments
    for image in found:
        asset = parts.add_asset(
            page_number,
            {
                "member": f"page:{page_number}/xobject:{'/'.join(image.path)}",
                "pdf_page_number": page_number,
                "pdf_xobject_path": list(image.path),
                "content_type": image.content_type,
                "sha256": image.sha256,
                "size_bytes": image.size_bytes,
                "width_px": image.width_px,
                "height_px": image.height_px,
                "alt_text": None,
            },
        )
        segments.append(_Segment("figure", "", metadata={"asset_id": asset.id}))
    return segments


# --- DrawingML shapes (slides, and the drawings of sheets and documents) ---

# Placeholder types that are the slide's title.
_TITLE_PLACEHOLDERS = frozenset({"title", "ctrTitle"})
# Notes-page placeholders that show the slide or page furniture, not notes.
_NOTES_SKIPPED_PLACEHOLDERS = frozenset({"sldImg", "sldNum", "hdr", "ftr", "dt"})
# A master names its placeholders by kind only; these kinds inherit from the
# master's placeholder of the kind on the right.
_MASTER_PLACEHOLDER_KINDS = {
    "ctrTitle": "title",
    "subTitle": "body",
    "obj": "body",
}
# Shapes whose tops lie within this many EMU (0.15 inch) of each other are
# read as one row, left to right: text boxes laid out side by side rarely
# share an exact top, and a strict top-then-left sort would read the one a
# hair higher first even when it stands to the right.
_SAME_ROW_EMU = 137_160
# SmartArt data-model points that hold the diagram's own text.
_SMARTART_TEXT_POINTS = frozenset({"", "node", "asst"})


@dataclass(frozen=True)
class _Transform:
    """Maps a group's child coordinates to slide coordinates (EMU).

    ``x' = scale_x * x + offset_x``; a group scales its children from its
    child extent (``a:chExt``) to its own extent (``a:ext``).
    """

    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def point(self, x: float, y: float) -> tuple[float, float]:
        """A child point in slide coordinates."""

        return self.scale_x * x + self.offset_x, self.scale_y * y + self.offset_y

    def size(self, cx: float, cy: float) -> tuple[float, float]:
        """A child extent in slide coordinates."""

        return self.scale_x * cx, self.scale_y * cy

    def group(self, xfrm: Element | None) -> _Transform:
        """The transform for the children of a group with this ``a:xfrm``."""

        if xfrm is None:
            return self
        off = _xy(_child_by_local_name(xfrm, "off"), "x", "y")
        ext = _xy(_child_by_local_name(xfrm, "ext"), "cx", "cy")
        child_off = _xy(_child_by_local_name(xfrm, "chOff"), "x", "y")
        child_ext = _xy(_child_by_local_name(xfrm, "chExt"), "cx", "cy")
        if off is None:
            return self
        ext = ext or (1.0, 1.0)
        child_off = child_off or (0.0, 0.0)
        child_ext = child_ext or ext
        scale_x = ext[0] / child_ext[0] if child_ext[0] else 1.0
        scale_y = ext[1] / child_ext[1] if child_ext[1] else 1.0
        local = _Transform(
            scale_x, scale_y, off[0] - scale_x * child_off[0], off[1] - scale_y * child_off[1]
        )
        return _Transform(
            self.scale_x * local.scale_x,
            self.scale_y * local.scale_y,
            self.scale_x * local.offset_x + self.offset_x,
            self.scale_y * local.offset_y + self.offset_y,
        )


@dataclass
class _Item:
    """One shape (or group) on a slide, where it sits, and what it says."""

    top: float | None
    left: float | None
    segments: list[_Segment]


@dataclass
class _SlideContext:
    """What reading one slide's shapes needs: its package, part and page."""

    package: _OfficePackage
    member: str
    page_number: int
    placeholders: dict[tuple[str, str], tuple[float, float]]
    title: list[str] = field(default_factory=list)


def _pptx_slide_segments(
    package: _OfficePackage, member: str, root: Element, page_number: int
) -> list[_Segment]:
    """A slide's content in reading order, then its speaker notes.

    The title placeholder comes first as a level-one heading, so chunks of
    the slide carry it as their section. The remaining shapes follow top to
    bottom, then left to right (``_reading_order``); a group is read as one
    unit at its own position, its shapes in the same order inside it.
    """

    context = _SlideContext(package, member, page_number, _layout_placeholders(package, member))
    tree = next((node for node in root.iter() if _local_name(node.tag) == "spTree"), None)
    items = _shape_items(tree, _Transform(), context, depth=0) if tree is not None else []
    segments: list[_Segment] = []
    title = " ".join(line for line in context.title if line)
    if title:
        segments.append(_Segment("lines", f"# {title}"))
    for item in _reading_order(items):
        segments.extend(item.segments)
    notes = _pptx_notes(package, member)
    if notes:
        package.parts.notes_pages += 1
        segments.append(_Segment("notes", "\n".join([_NOTES_HEADING, *notes])))
    return segments


def _reading_order(items: list[_Item]) -> list[_Item]:
    """Positioned items top to bottom, a row left to right; then the rest.

    Items whose tops lie within ``_SAME_ROW_EMU`` of the first item of a row
    are that row. A shape with no position of its own or from its layout is
    read after every positioned one, in document order, since nothing says
    where it stands.
    """

    positioned = sorted(
        (item for item in items if item.top is not None),
        key=lambda item: (item.top or 0.0, item.left or 0.0),
    )
    ordered: list[_Item] = []
    row: list[_Item] = []
    for item in positioned:
        if row and (item.top or 0.0) - (row[0].top or 0.0) > _SAME_ROW_EMU:
            ordered.extend(sorted(row, key=lambda entry: entry.left or 0.0))
            row = []
        row.append(item)
    ordered.extend(sorted(row, key=lambda entry: entry.left or 0.0))
    ordered.extend(item for item in items if item.top is None)
    return ordered


def _shape_items(
    container: Element, transform: _Transform, context: _SlideContext, *, depth: int
) -> list[_Item]:
    """The items of a shape tree or group, with positions in slide EMU."""

    _validate_office_xml_depth(depth)
    items: list[_Item] = []
    for node in container:
        name = _local_name(node.tag)
        if name == "AlternateContent":
            items.extend(_alternate_content_items(node, transform, context, depth=depth + 1))
            continue
        item: _Item | None = None
        if name == "sp":
            item = _text_shape_item(node, transform, context)
        elif name == "pic":
            item = _picture_item(node, transform, context)
        elif name == "graphicFrame":
            item = _graphic_frame_item(node, transform, context, depth=depth + 1)
        elif name == "grpSp":
            properties = _child_by_local_name(node, "grpSpPr")
            xfrm = _child_by_local_name(properties, "xfrm") if properties is not None else None
            children = _shape_items(node, transform.group(xfrm), context, depth=depth + 1)
            segments = [segment for child in _reading_order(children) for segment in child.segments]
            top, left = _position(node, transform, context)
            item = _Item(top, left, segments) if segments else None
        if item is not None and item.segments:
            items.append(item)
    return items


def _alternate_content_items(
    node: Element, transform: _Transform, context: _SlideContext, *, depth: int
) -> list[_Item]:
    """Read the first choice of an ``mc:AlternateContent``, else its fallback.

    The choice holds the shape as newer Office writes it (a chart type, an
    equation); when it yields nothing this parser can read, the fallback
    (usually a picture of the same thing) is read instead.
    """

    for branch in (
        *_children_by_local_name(node, "Choice"),
        *_children_by_local_name(node, "Fallback"),
    ):
        items = _shape_items(branch, transform, context, depth=depth)
        if items:
            return items
    return []


def _text_shape_item(node: Element, transform: _Transform, context: _SlideContext) -> _Item | None:
    body = _child_by_local_name(node, "txBody")
    lines = _drawing_text_lines(body) if body is not None else []
    if not lines:
        return None
    placeholder = _placeholder(node)
    if placeholder is not None and placeholder[0] in _TITLE_PLACEHOLDERS and not context.title:
        context.title = lines
        return None
    top, left = _position(node, transform, context)
    return _Item(top, left, [_Segment("lines", "\n".join(lines))])


def _picture_item(node: Element, transform: _Transform, context: _SlideContext) -> _Item | None:
    segment = _drawing_picture(
        context.package, context.member, context.page_number, node, transform
    )
    if segment is None:
        return None
    top, left = _position(node, transform, context)
    return _Item(top, left, [segment])


def _graphic_frame_item(
    node: Element, transform: _Transform, context: _SlideContext, *, depth: int
) -> _Item | None:
    """A table, chart, SmartArt diagram or embedded object's preview picture."""

    data = next((child for child in node.iter() if _local_name(child.tag) == "graphicData"), None)
    if data is None:
        return None
    segments = _graphic_data_segments(
        context.package, context.member, context.page_number, data, transform, depth=depth
    )
    top, left = _position(node, transform, context)
    return _Item(top, left, segments) if segments else None


def _graphic_data_segments(
    package: _OfficePackage,
    member: str,
    page_number: int,
    data: Element,
    transform: _Transform,
    *,
    depth: int,
) -> list[_Segment]:
    """The segments one ``a:graphicData`` holds, whatever document it is in."""

    _validate_office_xml_depth(depth)
    uri = (data.get("uri") or "").rstrip("/")
    for child in data:
        name = _local_name(child.tag)
        if name == "tbl":
            table = _drawing_table(child)
            if table is None:
                return []
            package.parts.tables += 1
            return [_Segment("table", table=table)]
        if name == "chart" and uri.endswith("/chart"):
            relationship_id = _office_document_relationship_id(child)
            chart = package.chart(package.target(member, relationship_id, "chart"))
            return [_Segment("chart", chart=chart)] if chart is not None else []
        if name == "relIds" and uri.endswith("/diagram"):
            lines = _smartart_lines(package, member, child)
            return [_Segment("lines", "\n".join(lines))] if lines else []
    # An embedded object (OLE) shows a picture of itself; that picture is
    # what the slide displays.
    segments: list[_Segment] = []
    for picture in _descendants(data, "pic", depth=depth + 1):
        segment = _drawing_picture(package, member, page_number, picture, transform)
        if segment is not None:
            segments.append(segment)
    return segments


def _drawing_picture(
    package: _OfficePackage,
    member: str,
    page_number: int,
    node: Element,
    transform: _Transform,
) -> _Segment | None:
    """A ``p:pic``/``xdr:pic``/``pic:pic`` as an asset and its figure segment."""

    blip = next((child for child in node.iter() if _local_name(child.tag) == "blip"), None)
    if blip is None:
        return None
    relationship_id = _attribute_by_local_name(blip, "embed")
    target = package.target(member, relationship_id, "image")
    if target is None:
        return None
    properties = next((child for child in node.iter() if _local_name(child.tag) == "cNvPr"), None)
    extra: dict[str, Any] = {}
    xfrm = _shape_xfrm(node)
    extent = _xy(_child_by_local_name(xfrm, "ext"), "cx", "cy") if xfrm is not None else None
    if extent is not None:
        width, height = transform.size(*extent)
        extra["extent_emu"] = {"cx": round(width), "cy": round(height)}
    return package.picture(page_number, target, alt_text=_alt_text(properties), extra=extra)


def _alt_text(properties: Element | None) -> str:
    """A picture's alt text: its description, else its title."""

    if properties is None:
        return ""
    return " ".join((properties.get("descr") or properties.get("title") or "").split())


def _drawing_table(table: Element) -> TableData | None:
    """A DrawingML ``a:tbl`` with merged cells repeated across their span.

    ``header_rows`` is 1 when the table's ``firstRow`` flag marks its first
    row as a header; without the flag nothing says which rows label columns,
    so none are counted rather than guessing.
    """

    properties = _child_by_local_name(table, "tblPr")
    header_rows = 1 if properties is not None and _truthy(properties.get("firstRow")) else 0
    rows: list[list[GridCell]] = []
    for row in _children_by_local_name(table, "tr"):
        cells: list[GridCell] = []
        for cell in _children_by_local_name(row, "tc"):
            body = _child_by_local_name(cell, "txBody")
            cells.append(
                GridCell(
                    text="\n".join(_drawing_text_lines(body)) if body is not None else "",
                    column_span=_int_attribute(cell, "gridSpan"),
                    row_span=_int_attribute(cell, "rowSpan"),
                    continues_left=_truthy(cell.get("hMerge")),
                    continues_up=_truthy(cell.get("vMerge")),
                )
            )
        rows.append(cells)
    return table_from_rows(build_rows(rows), header_rows=header_rows)


def _drawing_text_lines(body: Element) -> list[str]:
    """A text body's paragraphs, one line each, line breaks kept.

    Runs within a paragraph join without separators, as they display; a
    field (a slide number, a date) reads as its cached text.
    """

    lines: list[str] = []
    for paragraph in body.iter():
        if _local_name(paragraph.tag) != "p":
            continue
        text = _drawing_paragraph_text(paragraph, depth=0)
        lines.extend(line.strip() for line in text.split("\n") if line.strip())
    return lines


def _drawing_paragraph_text(node: Element, *, depth: int) -> str:
    _validate_office_xml_depth(depth)
    parts: list[str] = []
    for child in node:
        name = _local_name(child.tag)
        if name in {"r", "fld"}:
            parts.extend(text.text or "" for text in child if _local_name(text.tag) == "t")
        elif name == "br":
            parts.append("\n")
        elif name == "AlternateContent":
            # An equation's choice is OMML; its fallback is ordinary runs.
            branches = [
                *_children_by_local_name(child, "Fallback"),
                *_children_by_local_name(child, "Choice"),
            ]
            for branch in branches:
                text = _drawing_paragraph_text(branch, depth=depth + 1)
                if text.strip():
                    parts.append(text)
                    break
    return "".join(parts)


def _smartart_lines(package: _OfficePackage, member: str, relationship_ids: Element) -> list[str]:
    """The text of a SmartArt diagram's nodes, from its data part."""

    data_member = package.target(
        member, _attribute_by_local_name(relationship_ids, "dm"), "diagram_data"
    )
    root = package.xml(data_member) if data_member is not None else None
    if root is None:
        return []
    lines: list[str] = []
    for point in root.iter():
        if _local_name(point.tag) != "pt" or (point.get("type") or "") not in _SMARTART_TEXT_POINTS:
            continue
        body = _child_by_local_name(point, "t")
        if body is not None:
            lines.extend(_drawing_text_lines(body))
    return lines


def _pptx_notes(package: _OfficePackage, slide: str) -> list[str]:
    """A slide's speaker notes, one line per paragraph.

    The notes page also shows a picture of the slide, its number and page
    furniture; only its body text, and any text box added to it, is notes.
    """

    member = package.first_target(slide, "notes")
    root = package.xml(member) if member is not None else None
    if root is None:
        return []
    lines: list[str] = []
    for shape in root.iter():
        if _local_name(shape.tag) != "sp":
            continue
        placeholder = _placeholder(shape)
        if placeholder is not None and placeholder[0] in _NOTES_SKIPPED_PLACEHOLDERS:
            continue
        body = _child_by_local_name(shape, "txBody")
        if body is not None:
            lines.extend(_drawing_text_lines(body))
    return lines


def _position(
    node: Element, transform: _Transform, context: _SlideContext
) -> tuple[float | None, float | None]:
    """A shape's top and left in slide EMU, from it or its layout placeholder."""

    xfrm = _shape_xfrm(node)
    offset = _xy(_child_by_local_name(xfrm, "off"), "x", "y") if xfrm is not None else None
    if offset is not None:
        left, top = transform.point(*offset)
        return top, left
    placeholder = _placeholder(node)
    if placeholder is None:
        return None, None
    kind, index = placeholder
    inherited = (
        (context.placeholders.get(("idx", index)) if index else None)
        or context.placeholders.get(("type", kind))
        or context.placeholders.get(("master", _MASTER_PLACEHOLDER_KINDS.get(kind, kind)))
    )
    if inherited is None:
        return None, None
    left, top = inherited
    return top, left


def _layout_placeholders(
    package: _OfficePackage, slide: str
) -> dict[tuple[str, str], tuple[float, float]]:
    """Where the slide's layout and master put each placeholder.

    A placeholder on a slide usually has no position of its own and stands
    where its layout's placeholder of the same index (or kind) stands, or
    else where the master's placeholder of that kind does.
    """

    positions: dict[tuple[str, str], tuple[float, float]] = {}
    layout = package.first_target(slide, "layout")
    master = package.first_target(layout, "master") if layout is not None else None
    for member, keys in ((master, ("master",)), (layout, ("idx", "type"))):
        root = package.xml(member) if member is not None else None
        if root is None:
            continue
        for shape in root.iter():
            placeholder = _placeholder(shape)
            xfrm = _shape_xfrm(shape) if placeholder is not None else None
            offset = _xy(_child_by_local_name(xfrm, "off"), "x", "y") if xfrm is not None else None
            if placeholder is None or offset is None:
                continue
            kind, index = placeholder
            for key in keys:
                if key == "master":
                    positions.setdefault(
                        ("master", _MASTER_PLACEHOLDER_KINDS.get(kind, kind)), offset
                    )
                elif key == "idx" and index:
                    positions.setdefault(("idx", index), offset)
                elif key == "type":
                    positions.setdefault(("type", kind), offset)
    return positions


def _placeholder(node: Element) -> tuple[str, str] | None:
    """A shape's placeholder kind and index, if it is a placeholder.

    A placeholder that names no kind is a content placeholder ("obj").
    """

    properties = next((child for child in node if _local_name(child.tag).startswith("nv")), None)
    application = _child_by_local_name(properties, "nvPr") if properties is not None else None
    placeholder = _child_by_local_name(application, "ph") if application is not None else None
    if placeholder is None:
        return None
    return placeholder.get("type") or "obj", placeholder.get("idx") or ""


def _shape_xfrm(node: Element) -> Element | None:
    """The ``xfrm`` of a shape, picture, graphic frame or group."""

    for child in node:
        name = _local_name(child.tag)
        if name == "xfrm":
            return child
        if name in {"spPr", "grpSpPr"}:
            return _child_by_local_name(child, "xfrm")
    return None


def _xy(node: Element | None, x_name: str, y_name: str) -> tuple[float, float] | None:
    if node is None:
        return None
    try:
        x, y = float(node.get(x_name, "")), float(node.get(y_name, ""))
    except ValueError:
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "on"}


def _int_attribute(node: Element, local_name: str, default: int = 1) -> int:
    value = _attribute_by_local_name(node, local_name)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _descendants(node: Element, local_name: str | frozenset[str], *, depth: int) -> list[Element]:
    """Descendants named ``local_name`` (or any of a set), one branch of each alternative.

    An ``mc:AlternateContent`` states one thing twice; its first branch that
    holds a match is read, so a picture is not counted once per branch.
    """

    _validate_office_xml_depth(depth)
    names = frozenset({local_name}) if isinstance(local_name, str) else local_name
    found: list[Element] = []
    for child in node:
        name = _local_name(child.tag)
        if name in names:
            found.append(child)
        elif name == "AlternateContent":
            for branch in child:
                matches = _descendants(branch, names, depth=depth + 1)
                if matches:
                    found.extend(matches)
                    break
        else:
            found.extend(_descendants(child, names, depth=depth + 1))
    return found


def _child_by_local_name(node: Element | None, local_name: str) -> Element | None:
    if node is None:
        return None
    return next((child for child in node if _local_name(child.tag) == local_name), None)


def _children_by_local_name(node: Element, local_name: str) -> list[Element]:
    return [child for child in node if _local_name(child.tag) == local_name]


# --- XLSX drawings ---


def _xlsx_drawing_segments(package: _OfficePackage, sheet: str, page_number: int) -> list[_Segment]:
    """The charts and pictures a worksheet's drawing holds, top to bottom.

    Anchors are ordered by the cell they start in (row, then column), the
    way a reader scanning the sheet meets them.
    """

    drawing = package.first_target(sheet, "drawing")
    root = package.xml(drawing) if drawing is not None else None
    if drawing is None or root is None:
        return []
    anchored: list[tuple[tuple[float, float], int, list[_Segment]]] = []
    for order, anchor in enumerate(root):
        if not _local_name(anchor.tag).endswith("Anchor"):
            continue
        segments = _xlsx_shape_segments(package, drawing, page_number, anchor, depth=1)
        if segments:
            anchored.append((_anchor_cell(anchor), order, segments))
    anchored.sort(key=lambda entry: (entry[0], entry[1]))
    return [segment for _cell, _order, segments in anchored for segment in segments]


def _xlsx_shape_segments(
    package: _OfficePackage, drawing: str, page_number: int, container: Element, *, depth: int
) -> list[_Segment]:
    """Charts and pictures among an anchor's shapes, through groups and alternatives."""

    _validate_office_xml_depth(depth)
    segments: list[_Segment] = []
    for node in container:
        name = _local_name(node.tag)
        if name == "graphicFrame":
            data = next(
                (child for child in node.iter() if _local_name(child.tag) == "graphicData"), None
            )
            if data is not None:
                segments.extend(
                    _graphic_data_segments(
                        package, drawing, page_number, data, _Transform(), depth=depth + 1
                    )
                )
        elif name == "pic":
            segment = _drawing_picture(package, drawing, page_number, node, _Transform())
            if segment is not None:
                segments.append(segment)
        elif name == "grpSp":
            segments.extend(
                _xlsx_shape_segments(package, drawing, page_number, node, depth=depth + 1)
            )
        elif name == "AlternateContent":
            for branch in node:
                found = _xlsx_shape_segments(package, drawing, page_number, branch, depth=depth + 1)
                if found:
                    segments.extend(found)
                    break
    return segments


def _anchor_cell(anchor: Element) -> tuple[float, float]:
    start = _child_by_local_name(anchor, "from")
    if start is not None:
        row = _child_by_local_name(start, "row")
        column = _child_by_local_name(start, "col")
        try:
            return float((row.text if row is not None else "") or 0), float(
                (column.text if column is not None else "") or 0
            )
        except ValueError:
            return 0.0, 0.0
    position = _xy(_child_by_local_name(anchor, "pos"), "y", "x")
    return position or (0.0, 0.0)


# --- DOCX ---


@dataclass
class _DocxContext:
    """The package and part a Word body is read from, for its relationships."""

    package: _OfficePackage
    member: str
    page_number: int


# Where a Word paragraph draws: DrawingML, or a legacy VML picture.
_DOCX_PICTURE_ELEMENTS = frozenset({"drawing", "imagedata"})
# Word's ``w:tblLook`` packs its flags into a hex ``w:val`` in older files;
# this bit is "first row is a header".
_TBL_LOOK_FIRST_ROW = 0x0020
_HEX_BASE = 16


def _docx_block_segments(
    container: Element, context: _DocxContext, depth: int = 0
) -> list[_Segment]:
    """A Word body's paragraphs, tables, charts and pictures, in order."""

    _validate_office_xml_depth(depth)
    segments: list[_Segment] = []
    for child in container:
        local_name = _local_name(child.tag)
        if local_name == "p":
            text = _docx_inline_text(child).strip()
            if text:
                heading_level = _docx_heading_level(child)
                segments.append(
                    _Segment("lines", f"{'#' * heading_level} {text}" if heading_level else text)
                )
            segments.extend(_docx_drawing_segments(child, context))
        elif local_name == "tbl":
            table = _docx_table(child)
            if table is not None:
                context.package.parts.tables += 1
                segments.append(_Segment("table", table=table))
        elif local_name in {"sdt", "sdtContent"}:
            segments.extend(_docx_block_segments(child, context, depth + 1))
    return segments


def _docx_drawing_segments(paragraph: Element, context: _DocxContext) -> list[_Segment]:
    """Charts and pictures drawn in one paragraph, in document order."""

    segments: list[_Segment] = []
    for node in _descendants(paragraph, _DOCX_PICTURE_ELEMENTS, depth=1):
        name = _local_name(node.tag)
        if name == "drawing":
            properties = next(
                (child for child in node.iter() if _local_name(child.tag) == "docPr"), None
            )
            extent = next(
                (child for child in node.iter() if _local_name(child.tag) == "extent"), None
            )
            data = next(
                (child for child in node.iter() if _local_name(child.tag) == "graphicData"), None
            )
            if data is None:
                continue
            uri = (data.get("uri") or "").rstrip("/")
            if uri.endswith("/chart"):
                segments.extend(
                    _graphic_data_segments(
                        context.package,
                        context.member,
                        context.page_number,
                        data,
                        _Transform(),
                        depth=1,
                    )
                )
                continue
            size = _xy(extent, "cx", "cy")
            for blip in (child for child in data.iter() if _local_name(child.tag) == "blip"):
                target = context.package.target(
                    context.member, _attribute_by_local_name(blip, "embed"), "image"
                )
                segment = context.package.picture(
                    context.page_number,
                    target,
                    alt_text=_alt_text(properties),
                    extra=(
                        {"extent_emu": {"cx": round(size[0]), "cy": round(size[1])}}
                        if size is not None
                        else {}
                    ),
                )
                if segment is not None:
                    segments.append(segment)
        elif name == "imagedata":
            # A legacy VML picture (``w:pict``).
            target = context.package.target(
                context.member, _office_document_relationship_id(node), "image"
            )
            segment = context.package.picture(
                context.page_number, target, alt_text=node.get("title") or "", extra={}
            )
            if segment is not None:
                segments.append(segment)
    return segments


def _docx_table(table: Element) -> TableData | None:
    """A Word table with merged cells repeated across their span.

    Rows marked to repeat as a header on every page (``w:tblHeader``) are
    header rows. Without them, the table style's "first row" flag
    (``w:tblLook``) says the first row is a header, as Word displays it.
    """

    rows: list[list[GridCell]] = []
    repeated_headers = 0
    counting_headers = True
    for row in _docx_flat_children(table, "tr"):
        properties = _child_by_local_name(row, "trPr")
        is_header = properties is not None and _docx_on(
            _child_by_local_name(properties, "tblHeader")
        )
        if counting_headers and is_header:
            repeated_headers += 1
        else:
            counting_headers = False
        cells: list[GridCell] = []
        for cell in _docx_flat_children(row, "tc"):
            cell_properties = _child_by_local_name(cell, "tcPr")
            span = _child_by_local_name(cell_properties, "gridSpan")
            merge = _child_by_local_name(cell_properties, "vMerge")
            horizontal = _child_by_local_name(cell_properties, "hMerge")
            cells.append(
                GridCell(
                    text=_docx_cell_text(cell),
                    column_span=_int_attribute(span, "val") if span is not None else 1,
                    continues_up=merge is not None
                    and (_attribute_by_local_name(merge, "val") or "continue") == "continue",
                    continues_left=horizontal is not None
                    and (_attribute_by_local_name(horizontal, "val") or "continue") == "continue",
                )
            )
        rows.append(cells)
    header_rows = repeated_headers or (1 if _docx_first_row_is_header(table) else 0)
    return table_from_rows(build_rows(rows), header_rows=header_rows)


def _docx_first_row_is_header(table: Element) -> bool:
    properties = _child_by_local_name(table, "tblPr")
    look = _child_by_local_name(properties, "tblLook")
    if look is None:
        return False
    first_row = _attribute_by_local_name(look, "firstRow")
    if first_row is not None:
        return _truthy(first_row)
    try:
        return bool(
            int(_attribute_by_local_name(look, "val") or "0", _HEX_BASE) & _TBL_LOOK_FIRST_ROW
        )
    except ValueError:
        return False


def _docx_on(node: Element | None) -> bool:
    """A Word on/off property: present without ``w:val``, or set to true."""

    if node is None:
        return False
    value = _attribute_by_local_name(node, "val")
    return value is None or _truthy(value)


def _docx_flat_children(node: Element, local_name: str, depth: int = 0) -> Iterator[Element]:
    """Children named ``local_name``, looking through content controls."""

    _validate_office_xml_depth(depth)
    for child in node:
        name = _local_name(child.tag)
        if name == local_name:
            yield child
        elif name in {"sdt", "sdtContent", "customXml"}:
            yield from _docx_flat_children(child, local_name, depth + 1)


def _select_pages(pages: list[ParsedPage], page_range: str | None) -> list[ParsedPage]:
    if page_range is None or not page_range.strip():
        return pages
    selected_numbers = _parse_builtin_page_range(page_range)
    selected = [page for page in pages if page.page_number in selected_numbers]
    if not selected:
        raise ValueError(f"builtin_text page_range '{page_range}' selected no pages")
    return selected


def _parse_builtin_page_range(page_range: str) -> set[int]:
    selected: set[int] = set()
    for start, end in parse_page_range(page_range, "builtin_text"):
        if start is None or end is None:
            # The parser refuses open ends for this provider; this keeps the
            # bounds typed as integers for the arithmetic below.
            raise ValueError(f"Invalid builtin_text page_range '{page_range}': open-ended range")
        if end - start >= MAX_PDFIUM_FALLBACK_PAGES:
            raise ValueError(f"Invalid builtin_text page_range '{page_range}': range is too large")
        selected.update(range(start, end + 1))
    return selected


def _docx_text_member_names(archive: zipfile.ZipFile) -> list[str]:
    """Return only DOCX XML parts read by the text extractor in stable order."""

    names = set(archive.namelist())
    if "word/document.xml" not in names:
        raise ValueError("DOCX file is missing word/document.xml")
    headers = sorted(name for name in names if re.fullmatch(r"word/header\d+\.xml", name))
    footers = sorted(name for name in names if re.fullmatch(r"word/footer\d+\.xml", name))
    return ["word/document.xml", *headers, *footers]


def _validate_office_archive(
    path: Path,
    archive: zipfile.ZipFile,
    member_names: list[str],
) -> None:
    """Bound every Office part that will actually be decompressed, including the total."""

    total_bytes = 0
    seen: set[str] = set()
    for name in member_names:
        if name in seen:
            continue
        seen.add(name)
        info = archive.getinfo(name)
        _validate_zip_member_size(path, info)
        total_bytes += info.file_size
    if total_bytes > MAX_OFFICE_TOTAL_BYTES:
        raise ValueError(
            f"Office file {path} contains {total_bytes} bytes of readable XML, "
            f"above the {MAX_OFFICE_TOTAL_BYTES} byte aggregate limit"
        )


def _docx_heading_level(paragraph: Element) -> int | None:
    properties = next(
        (child for child in paragraph if _local_name(child.tag) == "pPr"),
        None,
    )
    if properties is None:
        return None
    for node in properties:
        if _local_name(node.tag) != "pStyle":
            continue
        style = next(
            (str(value) for key, value in node.attrib.items() if _local_name(key) == "val"),
            "",
        )
        match = re.fullmatch(r"Heading\s*(\d+)", style, re.IGNORECASE)
        if match:
            return min(max(int(match.group(1)), 1), 6)
    return None


def _docx_cell_text(cell: Element) -> str:
    """Render cell paragraphs with boundaries instead of concatenating their runs."""

    paragraphs = _docx_cell_paragraphs(cell)
    return "\n".join(paragraph for paragraph in paragraphs if paragraph)


def _docx_cell_paragraphs(container: Element, depth: int = 0) -> list[str]:
    _validate_office_xml_depth(depth)
    paragraphs: list[str] = []
    for child in container:
        local_name = _local_name(child.tag)
        if local_name == "p":
            paragraphs.append(_docx_inline_text(child).strip())
        elif local_name in {"sdt", "sdtContent"}:
            paragraphs.extend(_docx_cell_paragraphs(child, depth + 1))
    return paragraphs


def _docx_inline_text(node: Element, depth: int = 0) -> str:
    """Render Word inline text while preserving tabs, breaks, and nested paragraphs."""

    _validate_office_xml_depth(depth)
    parts: list[str] = []
    for child in node:
        local_name = _local_name(child.tag)
        if local_name == "t":
            parts.append(child.text or "")
        elif local_name == "tab":
            parts.append("\t")
        elif local_name in {"br", "cr"}:
            parts.append("\n")
        elif local_name == "txbxContent":
            nested = _docx_inline_text(child, depth + 1).strip()
            if nested:
                parts.extend(("\n", nested, "\n"))
        elif local_name == "p":
            nested = _docx_inline_text(child, depth + 1).strip()
            if nested:
                if parts and not parts[-1].endswith((" ", "\t", "\n")):
                    parts.append("\n")
                parts.extend((nested, "\n"))
        else:
            parts.append(_docx_inline_text(child, depth + 1))
    return "".join(parts)


def _read_zip_member(path: Path, archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    _validate_zip_member_size(path, info)
    payload = bytearray()
    with archive.open(info) as member:
        while chunk := member.read(_OFFICE_READ_CHUNK_BYTES):
            payload.extend(chunk)
            if len(payload) > MAX_OFFICE_ENTRY_BYTES:
                raise ValueError(
                    f"Office file {path} contains entry {info.filename} whose actual "
                    f"output exceeds the {MAX_OFFICE_ENTRY_BYTES} byte limit"
                )
    if len(payload) != info.file_size:
        raise ValueError(
            f"Office file {path} entry {info.filename} produced {len(payload)} bytes, "
            f"but its archive metadata declares {info.file_size}"
        )
    return bytes(payload)


def _validate_zip_member_size(path: Path, info: zipfile.ZipInfo) -> None:
    if info.file_size > MAX_OFFICE_ENTRY_BYTES:
        raise ValueError(
            f"Office file {path} contains entry {info.filename} with "
            f"{info.file_size} bytes, above the {MAX_OFFICE_ENTRY_BYTES} byte limit"
        )


class _HtmlCharsetParser(HTMLParser):
    """Capture encoding declarations only from real meta tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.encoding: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record direct and HTTP-equiv charset declarations from meta attributes."""

        if tag.casefold() != "meta" or self.encoding is not None:
            return
        attributes = {name.casefold(): value or "" for name, value in attrs}
        charset = attributes.get("charset", "").strip()
        if charset:
            self.encoding = charset
            return
        if attributes.get("http-equiv", "").casefold() != "content-type":
            return
        match = _CONTENT_CHARSET_RE.search(attributes.get("content", ""))
        if match is not None:
            self.encoding = match.group(1)


class _VisibleTextHtmlParser(HTMLParser):
    """Extract visible, structurally useful HTML text without a browser runtime.

    Script-like content is ignored while headings, lists and paragraph
    boundaries are represented in the same lightweight syntax used by the
    text page classifier. A table is read into a ``TableData`` by the shared
    ``HtmlTableCollector`` and becomes a table block of its own.
    """

    def __init__(self) -> None:
        """Initialize parser state for visible text, open tables, and ignored elements."""

        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._segments: list[_Segment] = []
        self._ignored_depth = 0
        self._table: HtmlTableCollector | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track ignored regions and emit separators for structural start tags.

        Attributes are used only for table cell spans; visible text structure
        is otherwise sufficient for deterministic chunking and avoids
        interpreting scripts or style-dependent presentation.
        """

        normalized = tag.lower()
        if normalized in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif self._ignored_depth:
            return
        elif self._table is not None:
            self._table.start(normalized, attrs)
        elif normalized == "table":
            self._flush_text()
            self._table = HtmlTableCollector("native")
        elif normalized in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")
            self._parts.append(f"{'#' * int(normalized[1])} ")
        elif normalized == "li":
            self._parts.append("\n- ")
        elif normalized in {"br", "p", "div", "section", "article", "tr"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Close ignored regions and preserve boundaries between visible blocks.

        Balanced ignored-depth tracking prevents script or style text from
        leaking into the normalized document even when such tags are nested.
        """

        normalized = tag.lower()
        if normalized in {"script", "style", "noscript"}:
            if self._ignored_depth:
                self._ignored_depth -= 1
        elif self._ignored_depth:
            return
        elif self._table is not None:
            if self._table.end(normalized):
                self._finish_table()
        elif normalized in {
            "p",
            "div",
            "section",
            "article",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Capture text nodes that are not inside ignored tags."""

        if self._ignored_depth:
            return
        if self._table is not None:
            self._table.data(data)
        else:
            self._parts.append(data)

    def close(self) -> None:
        """Finish a table the document never closed."""

        super().close()
        if self._table is not None:
            self._finish_table()

    def segments(self) -> list[_Segment]:
        """The page's text and tables, in document order."""

        self._flush_text()
        return list(self._segments)

    def _finish_table(self) -> None:
        table = self._table.table() if self._table is not None else None
        self._table = None
        if table is not None:
            self._segments.append(_Segment("table", table=table))

    def _flush_text(self) -> None:
        """Turn the text seen since the last table into one lines segment.

        Layout whitespace collapses but paragraph-ish breaks stay, so
        chunking sees human-readable text rather than a browser DOM dump.
        """

        lines = []
        for line in "".join(self._parts).splitlines():
            normalized = re.sub(r"[ ]+", " ", line).strip()
            if normalized:
                lines.append(normalized)
        self._parts = []
        if lines:
            self._segments.append(_Segment("lines", "\n".join(lines)))


def _is_pptx_slide_xml(name: str) -> bool:
    return bool(re.fullmatch(r"ppt/slides/slide\d+\.xml", name))


def _slide_sort_key(name: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", name)
    return int(match.group(1)) if match else 0


def _pptx_slide_member_names(
    path: Path,
    archive: zipfile.ZipFile,
) -> tuple[list[str], list[str]]:
    """Resolve slide parts in presentation order, including nonstandard part names."""

    names = set(archive.namelist())
    presentation_name = "ppt/presentation.xml"
    relationships_name = "ppt/_rels/presentation.xml.rels"
    fallback = sorted(
        (name for name in names if _is_pptx_slide_xml(name)),
        key=_slide_sort_key,
    )
    if presentation_name not in names or relationships_name not in names:
        return fallback, []

    relationships = _office_relationships(
        path,
        archive,
        presentation_name,
        relationships_name,
    )
    root = ElementTree.fromstring(_read_zip_member(path, archive, presentation_name))
    ordered: list[str] = []
    for node in root.iter():
        if _local_name(node.tag) != "sldId":
            continue
        relationship_id = _office_document_relationship_id(node)
        target = relationships.get(relationship_id or "")
        if target and target[1].endswith("/slide") and target[0] in names:
            ordered.append(target[0])
    return list(dict.fromkeys(ordered)) or fallback, [presentation_name, relationships_name]


def _is_xlsx_sheet_xml(name: str) -> bool:
    return bool(re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))


def _xlsx_sheet_sort_key(name: str) -> int:
    match = re.search(r"sheet(\d+)\.xml$", name)
    return int(match.group(1)) if match else 0


def _xlsx_member_names(
    path: Path,
    archive: zipfile.ZipFile,
) -> tuple[list[str], list[str], list[str]]:
    """Resolve worksheet and shared-string parts through workbook relationships."""

    names = set(archive.namelist())
    workbook_name = "xl/workbook.xml"
    relationships_name = "xl/_rels/workbook.xml.rels"
    fallback_sheets = sorted(
        (name for name in names if _is_xlsx_sheet_xml(name)),
        key=_xlsx_sheet_sort_key,
    )
    fallback_shared = ["xl/sharedStrings.xml"] if "xl/sharedStrings.xml" in names else []
    if workbook_name not in names or relationships_name not in names:
        return fallback_sheets, fallback_shared, []

    relationships = _office_relationships(
        path,
        archive,
        workbook_name,
        relationships_name,
    )
    root = ElementTree.fromstring(_read_zip_member(path, archive, workbook_name))
    sheets: list[str] = []
    for node in root.iter():
        if _local_name(node.tag) != "sheet":
            continue
        relationship_id = _office_document_relationship_id(node)
        target = relationships.get(relationship_id or "")
        if target and target[1].endswith("/worksheet") and target[0] in names:
            sheets.append(target[0])
    shared_strings = [
        target
        for target, relationship_type in relationships.values()
        if relationship_type.endswith("/sharedStrings") and target in names
    ]
    return (
        list(dict.fromkeys(sheets)) or fallback_sheets,
        list(dict.fromkeys(shared_strings)) or fallback_shared,
        [workbook_name, relationships_name],
    )


def _office_relationships(
    path: Path,
    archive: zipfile.ZipFile,
    source_member: str,
    relationships_member: str,
) -> dict[str, tuple[str, str]]:
    """Return internal OOXML relationships keyed by their relationship IDs."""

    root = ElementTree.fromstring(_read_zip_member(path, archive, relationships_member))
    relationships: dict[str, tuple[str, str]] = {}
    for node in root:
        if _local_name(node.tag) != "Relationship":
            continue
        if (_attribute_by_local_name(node, "TargetMode") or "").lower() == "external":
            continue
        relationship_id = _attribute_by_local_name(node, "Id")
        target = _attribute_by_local_name(node, "Target")
        relationship_type = _attribute_by_local_name(node, "Type")
        if not relationship_id or not target or not relationship_type:
            continue
        resolved = _resolve_office_relationship_target(source_member, target)
        if resolved is not None:
            relationships[relationship_id] = (resolved, relationship_type)
    return relationships


def _resolve_office_relationship_target(source_member: str, target: str) -> str | None:
    """Resolve an OOXML relationship without allowing traversal outside the archive root."""

    if target.startswith("/"):
        parts = list(PurePosixPath(target.lstrip("/")).parts)
    else:
        parts = list(PurePosixPath(source_member).parent.parts)
        parts.extend(PurePosixPath(target).parts)
    normalized: list[str] = []
    for part in parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized:
                return None
            normalized.pop()
        else:
            normalized.append(part)
    return "/".join(normalized)


def _attribute_by_local_name(node: Element, local_name: str) -> str | None:
    return next(
        (str(value) for key, value in node.attrib.items() if _local_name(key) == local_name),
        None,
    )


def _office_document_relationship_id(node: Element) -> str | None:
    return next(
        (
            str(value)
            for key, value in node.attrib.items()
            if key.endswith("}id") and "officeDocument/2006/relationships" in key
        ),
        None,
    )


def _xlsx_shared_strings(
    path: Path,
    archive: zipfile.ZipFile,
    member_name: str | None,
) -> list[str]:
    if member_name is None:
        return []
    root = ElementTree.fromstring(_read_zip_member(path, archive, member_name))
    return [_xlsx_display_text(item).strip() for item in root if _local_name(item.tag) == "si"]


def _xlsx_sheet_text(xml_payload: bytes, shared_strings: list[str]) -> str:
    """Render a sheet as tab-separated rows whose cells stay under their columns.

    A workbook omits empty cells, so a row with a gap would otherwise shift
    every later cell left, under the wrong header, and a reader pairing cells
    with headers would pair them wrongly. Each cell is placed by its reference
    ("C5" is the third column); a gap is an empty field. Columns empty in every
    row before the first used one are left out, and so is a row's run past the
    table that only repeats what the row already says (_beyond_the_table).
    """

    root = ElementTree.fromstring(xml_payload)
    rows: list[dict[int, str]] = []
    for row in root.iter():
        if _local_name(row.tag) != "row":
            continue
        cells: dict[int, str] = {}
        next_column = 0
        for cell in row:
            if _local_name(cell.tag) != "c":
                continue
            column = _xlsx_column_index(cell.attrib.get("r", ""))
            if column is None:
                column = next_column
            next_column = column + 1
            value = _xlsx_cell_text(cell, shared_strings)
            if value:
                cells[column] = value
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    rows = _beyond_the_table(rows)
    first = min(min(cells) for cells in rows)
    return "\n".join(
        "\t".join(cells.get(column, "") for column in range(first, max(cells) + 1))
        for cells in rows
    )


# The rows that must use a column for it to be part of the table, and how often
# a run past it must repeat one sequence to be read as a copy of that sequence.
_TABLE_ROWS = 2
_REPEATS = 3


def _beyond_the_table(rows: list[dict[int, str]]) -> list[dict[int, str]]:
    """Keep one copy of a sequence a row repeats past the table's last column.

    A row copied across the sheet - out to its last column, sixteen thousand
    cells wide - repeats a few cells thousands of times, and read as text it
    is a line of hundreds of thousands of characters that says what its first
    copy says. The table ends at the last column two rows use; a row's run past
    it that repeats one sequence at least three times keeps the sequence once,
    and any other run is kept whole.
    """

    used: Counter[int] = Counter(column for cells in rows for column in cells)
    shared = [column for column, count in used.items() if count >= _TABLE_ROWS]
    if not shared:
        return rows
    last = max(shared)
    trimmed: list[dict[int, str]] = []
    for cells in rows:
        beyond = sorted(column for column in cells if column > last)
        period = _period([cells[column] for column in beyond])
        if period is None:
            trimmed.append(cells)
            continue
        keep = set(beyond[:period])
        trimmed.append(
            {column: value for column, value in cells.items() if column <= last or column in keep}
        )
    return trimmed


def _period(values: list[str]) -> int | None:
    """The length of the sequence ``values`` repeats at least ``_REPEATS`` times, if any."""

    for period in range(1, len(values) // _REPEATS + 1):
        if all(value == values[index % period] for index, value in enumerate(values)):
            return period
    return None


def _xlsx_column_index(reference: str) -> int | None:
    """The zero-based column of a cell reference such as "AB12", or None without one."""

    letters = ""
    for character in reference:
        if not ("A" <= character.upper() <= "Z"):
            break
        letters += character.upper()
    if not letters:
        return None
    index = 0
    for character in letters:
        index = index * 26 + (ord(character) - ord("A") + 1)
    return index - 1


def _xlsx_cell_text(cell: Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return _xlsx_display_text(cell).strip()
    value = next((child.text or "" for child in cell if _local_name(child.tag) == "v"), "")
    if cell_type == "s" and value.isdigit():
        index = int(value)
        if index < len(shared_strings):
            return shared_strings[index]
        return ""
    return value.strip()


def _xlsx_display_text(node: Element, depth: int = 0) -> str:
    """Render cell text while excluding phonetic pronunciation guides."""

    _validate_office_xml_depth(depth)
    if _local_name(node.tag) == "rPh":
        return ""
    if _local_name(node.tag) == "t":
        return node.text or ""
    return "".join(_xlsx_display_text(child, depth + 1) for child in node)


def _validate_office_xml_depth(depth: int) -> None:
    if depth > MAX_OFFICE_XML_DEPTH:
        raise ValueError(f"Office XML nesting exceeds the {MAX_OFFICE_XML_DEPTH} level limit")


def _read_text_document(path: Path, *, sniff_html_charset: bool = False) -> str:
    """Decode bounded text using BOMs, HTML declarations, and mixed-byte recovery."""

    size_bytes = path.stat().st_size
    if size_bytes > MAX_TEXT_BYTES:
        raise ValueError(
            f"Text file {path} contains {size_bytes} bytes, above the {MAX_TEXT_BYTES} byte limit"
        )
    payload = path.read_bytes()
    encoding = next(
        (name for marker, name in _TEXT_BOMS if payload.startswith(marker)),
        None,
    )
    bom_encoding = encoding is not None
    declared_encoding = False
    if encoding is None and sniff_html_charset:
        parser = _HtmlCharsetParser()
        parser.feed(payload[:8192].decode("ascii", errors="ignore"))
        if parser.encoding is not None:
            encoding = parser.encoding
            declared_encoding = True
    if encoding is None:
        encoding = _bomless_unicode_encoding(payload) or "utf-8-sig"
    try:
        text = payload.decode(encoding)
    except LookupError:
        if not declared_encoding:
            raise ValueError(f"Unsupported text encoding {encoding!r} in {path}") from None
        text = _decode_mixed_utf8(payload)
    except UnicodeDecodeError:
        normalized_encoding = encoding.replace("_", "-").casefold()
        if bom_encoding and normalized_encoding.startswith(("utf-16", "utf-32")):
            text = payload.decode(encoding, errors="replace")
        elif normalized_encoding in {"utf-8", "utf-8-sig"}:
            text = _decode_mixed_utf8(payload)
        else:
            text = _decode_legacy_bytes(payload)
    if "\x00" in text:
        raise ValueError(f"Decoded text file contains NUL characters: {path}")
    return text.lstrip("\ufeff")


def _bomless_unicode_encoding(payload: bytes) -> str | None:
    """Recognize strongly patterned BOM-less UTF-16/32 without guessing prose encodings."""

    sample = payload[:_ENCODING_SNIFF_BYTES]
    if len(sample) >= _UTF32_MIN_SNIFF_BYTES:
        quartets = len(sample) // 4
        if quartets:
            lanes = [sample[index::4].count(0) / quartets for index in range(4)]
            if all(lanes[index] > _UTF32_ZERO_LANE_RATIO for index in (1, 2, 3)):
                return "utf-32-le"
            if all(lanes[index] > _UTF32_ZERO_LANE_RATIO for index in (0, 1, 2)):
                return "utf-32-be"
    if len(sample) >= _UTF16_MIN_SNIFF_BYTES:
        pairs = len(sample) // 2
        even_zero = sample[0::2].count(0) / pairs
        odd_zero = sample[1::2].count(0) / pairs
        if odd_zero > _UTF16_ZERO_LANE_RATIO and even_zero < _UTF16_NONZERO_LANE_RATIO:
            return "utf-16-le"
        if even_zero > _UTF16_ZERO_LANE_RATIO and odd_zero < _UTF16_NONZERO_LANE_RATIO:
            return "utf-16-be"
    return None


def _decode_mixed_utf8(payload: bytes) -> str:
    """Preserve valid UTF-8 spans while decoding isolated legacy bytes as CP1252."""

    decoded = payload.decode("utf-8-sig", errors="surrogateescape")
    return re.sub(
        r"[\udc80-\udcff]+",
        lambda match: _decode_legacy_bytes(
            bytes(ord(character) - 0xDC00 for character in match.group())
        ),
        decoded,
    )


def _decode_legacy_bytes(payload: bytes) -> str:
    """Prefer Windows-1252 typography while retaining every byte via Latin-1."""

    try:
        return payload.decode("cp1252")
    except UnicodeDecodeError:
        return payload.decode("latin-1")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
