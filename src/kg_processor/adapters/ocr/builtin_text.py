# SPDX-License-Identifier: Apache-2.0
"""Simple text/PDF/Office/HTML parser used for deterministic smoke fixtures."""

from __future__ import annotations

import codecs
import importlib
import multiprocessing
import re
import subprocess
import sys
import zipfile
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from html.parser import HTMLParser
from itertools import pairwise
from multiprocessing.connection import Connection
from pathlib import Path, PurePosixPath
from time import monotonic
from xml.etree.ElementTree import Element

import pypdfium2 as pdfium  # type: ignore[import-untyped]
from defusedxml import ElementTree
from pypdf import PdfReader
from pypdf.errors import LimitReachedError

from kg_processor.domain.documents import InputFile, LayoutBlock, ParsedDocument, ParsedPage
from kg_processor.domain.ids import stable_id
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
        """Parse text-native formats into deterministic one-block pages."""

        suffix = file.path.suffix.lower()
        layout: dict[str, float] = {}
        if suffix in {".txt", ".md", ".markdown"}:
            pages = [_page_from_text(file, 1, _read_text_document(file.path))]
        elif suffix == ".pdf":
            pages, interleaved_ratio = self._parse_pdf(file.path, file)
            if interleaved_ratio is not None:
                layout["interleaved_column_page_ratio"] = interleaved_ratio
        elif suffix == ".docx":
            pages = self._parse_docx(file.path, file)
        elif suffix in {".html", ".htm"}:
            pages = self._parse_html(file.path, file)
        elif suffix == ".pptx":
            pages = self._parse_pptx(file.path, file)
        elif suffix == ".xlsx":
            pages = self._parse_xlsx(file.path, file)
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
        return ParsedDocument(
            file_id=file.id,
            checksum=file.checksum,
            source_uri=file.source_uri,
            mime_type=file.mime_type,
            pages=pages,
            provider_metadata={
                "provider": "builtin_text",
                "language": options.language,
                "page_range": options.page_range,
                **layout,
            },
        )

    def _parse_pdf(self, path: Path, file: InputFile) -> tuple[list[ParsedPage], float | None]:
        """Extract a PDF with pypdf, falling back to PDFium for unusual fonts.

        Both engines preserve one output page per physical PDF page. PDFium is
        intentionally a text fallback rather than OCR: malformed font metadata
        should not force a text-native document through a slower OCR provider or
        collapse its page provenance.

        The second value is the share of measurable pages whose text layer runs
        across two columns line by line (see ``_interleaved_columns``), or None
        when PDFium read the file and no positions were available.
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
                pages.append(_page_from_text(file, index, text))
            return pages, (interleaved_pages / measured_pages if measured_pages else 0.0)
        except (_EncryptedPdfError, LimitReachedError):
            raise
        except Exception as primary_error:
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

    def _parse_docx(self, path: Path, file: InputFile) -> list[ParsedPage]:
        with zipfile.ZipFile(path) as archive:
            member_names = _docx_text_member_names(archive)
            _validate_office_archive(path, archive, member_names)
            text = "\n".join(
                part
                for name in member_names
                for part in _docx_xml_text(_read_zip_member(path, archive, name))
            )
        return [_page_from_text(file, 1, text)]

    def _parse_html(self, path: Path, file: InputFile) -> list[ParsedPage]:
        parser = _VisibleTextHtmlParser()
        parser.feed(_read_text_document(path, sniff_html_charset=True))
        return [_page_from_text(file, 1, parser.text())]

    def _parse_pptx(self, path: Path, file: InputFile) -> list[ParsedPage]:
        pages: list[ParsedPage] = []
        with zipfile.ZipFile(path) as archive:
            slide_names, control_names = _pptx_slide_member_names(path, archive)
            _validate_office_archive(path, archive, [*control_names, *slide_names])
            for index, name in enumerate(slide_names, start=1):
                text = _pptx_slide_text(_read_zip_member(path, archive, name))
                pages.append(_page_from_text(file, index, text))
        if not pages:
            raise ValueError(f"PPTX file contains no slides: {path}")
        return pages

    def _parse_xlsx(self, path: Path, file: InputFile) -> list[ParsedPage]:
        pages: list[ParsedPage] = []
        with zipfile.ZipFile(path) as archive:
            sheet_names, shared_string_names, control_names = _xlsx_member_names(path, archive)
            _validate_office_archive(
                path, archive, [*control_names, *shared_string_names, *sheet_names]
            )
            shared_strings = _xlsx_shared_strings(
                path,
                archive,
                shared_string_names[0] if shared_string_names else None,
            )
            for index, name in enumerate(sheet_names, start=1):
                text = _xlsx_sheet_text(_read_zip_member(path, archive, name), shared_strings)
                pages.append(_page_from_text(file, index, text))
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

    text = text.lstrip("\ufeff")
    rendered_lines: list[str] = []
    block_specs: list[tuple[str, str, dict[str, object]]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        kind, rendered, metadata = _classify_text_line(line)
        rendered_lines.append(rendered)
        block_specs.append((kind, rendered, metadata))
    rendered_text = "\n".join(rendered_lines)
    blocks: list[LayoutBlock] = []
    cursor = 0
    for index, (kind, rendered, metadata) in enumerate(block_specs):
        start = rendered_text.find(rendered, cursor)
        end = start + len(rendered)
        cursor = end
        blocks.append(
            LayoutBlock(
                id=stable_id("block", file.id, page_number, index, kind, rendered[:256]),
                page_number=page_number,
                kind=kind,
                text=rendered,
                metadata={**metadata, "start_offset": start, "end_offset": end},
            )
        )
    return ParsedPage(
        page_number=page_number,
        markdown=rendered_text,
        raw_text=rendered_text,
        blocks=blocks,
    )


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


def _docx_xml_text(xml_payload: bytes) -> list[str]:
    """Extract ordered paragraphs, tables, and textbox descendants from one DOCX XML part."""

    root = ElementTree.fromstring(xml_payload)
    body = next((child for child in root if _local_name(child.tag) == "body"), root)
    return _docx_block_parts(body)


def _docx_block_parts(container: Element, depth: int = 0) -> list[str]:
    _validate_office_xml_depth(depth)
    parts: list[str] = []
    for child in container:
        local_name = _local_name(child.tag)
        if local_name == "p":
            text = _docx_inline_text(child).strip()
            if not text:
                continue
            heading_level = _docx_heading_level(child)
            parts.append(f"{'#' * heading_level} {text}" if heading_level else text)
        elif local_name == "tbl":
            for row in child:
                if _local_name(row.tag) != "tr":
                    continue
                cells = [_docx_cell_text(cell) for cell in row if _local_name(cell.tag) == "tc"]
                rendered = [cell for cell in cells if cell]
                if rendered:
                    parts.append("\t".join(rendered))
        elif local_name in {"sdt", "sdtContent"}:
            parts.extend(_docx_block_parts(child, depth + 1))
    return parts


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

    Script-like content is ignored while headings, lists, tables, and paragraph
    boundaries are represented in the same lightweight syntax used by the text
    page classifier.
    """

    def __init__(self) -> None:
        """Initialize parser state for visible text, table cells, and nested ignored elements."""

        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_depth = 0
        self._table_cell_index = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track ignored regions and emit separators for structural start tags.

        Attributes are intentionally unused because visible text structure is
        sufficient for deterministic chunking and avoids interpreting scripts
        or style-dependent presentation.
        """

        normalized = tag.lower()
        if normalized in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif normalized in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")
            self._parts.append(f"{'#' * int(normalized[1])} ")
        elif normalized == "tr":
            self._parts.append("\n")
            self._table_cell_index = 0
        elif normalized in {"td", "th"}:
            if self._table_cell_index:
                self._parts.append("\t")
            self._table_cell_index += 1
        elif normalized == "li":
            self._parts.append("\n- ")
        elif normalized in {"br", "p", "div", "section", "article"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Close ignored regions and preserve boundaries between visible blocks.

        Balanced ignored-depth tracking prevents script or style text from
        leaking into the normalized document even when such tags are nested.
        """

        normalized = tag.lower()
        if normalized in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
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

        if not self._ignored_depth:
            self._parts.append(data)

    def text(self) -> str:
        """Return visible text with normalized spaces and meaningful line breaks.

        This output becomes both page text and block input, so deterministic
        whitespace is important for stable chunk hashes and evidence offsets.
        """

        # Collapse layout whitespace but preserve paragraph-ish breaks so
        # chunking sees human-readable text rather than a browser DOM dump.
        lines = []
        for line in "".join(self._parts).splitlines():
            normalized = re.sub(r"[ ]+", " ", line).strip()
            if normalized:
                lines.append(normalized)
        return "\n".join(lines)


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


def _pptx_slide_text(xml_payload: bytes) -> str:
    root = ElementTree.fromstring(xml_payload)
    values = [
        node.text or "" for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "t" and node.text
    ]
    return "\n".join(value.strip() for value in values if value.strip())


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
