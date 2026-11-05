# SPDX-License-Identifier: Apache-2.0
"""OCR provider port and options.

The options object is broad on purpose: all OCR providers receive one typed
configuration surface, but graph code only consumes the normalized
`ParsedDocument` result and never branches on provider-specific knobs.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from kg_processor.domain.documents import InputFile, ParsedDocument


# File compatibility is part of the provider contract because both preflight
# and composed OCR adapters must make the same routing decision before loading
# a concrete OCR implementation. Providers omitted from this map are treated
# as dynamically capable, which preserves support for configurable HTTP and
# Snowflake adapters whose accepted formats are controlled by the service.
class NoTextReadError(RuntimeError):
    """A parser ran to completion and read no text from any page.

    Distinct from a parser failing, because a caller with another parser
    can route the file on, and a caller holding an earlier, equally empty
    result can keep it: an empty spreadsheet is an empty document, not an
    error, and only the parsers can tell the two apart.
    """


OCR_SUPPORTED_SUFFIXES: dict[str, frozenset[str]] = {
    "builtin_text": frozenset(
        {".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm", ".pptx", ".xlsx"}
    ),
    "tesseract_internal": frozenset(
        {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    ),
    "mineru_api": frozenset(
        {
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".webp",
        }
    ),
    "mineru_internal": frozenset(
        {
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".webp",
        }
    ),
}


class OcrOptions(BaseModel):
    """Provider-neutral OCR options; adapters consume the fields they support."""

    language: str | None = None
    page_range: str | None = None
    timeout_seconds: int = 900
    model_cache_dir: str | None = None
    method: str | None = None
    backend: str | None = None
    effort: str | None = None
    api_url: str | None = None
    api_key: str | None = None
    server_url: str | None = None
    start_page_id: int | None = None
    end_page_id: int | None = None
    formula: bool | None = None
    table: bool | None = None
    image_analysis: bool | None = None
    client_side_output_generation: bool = False
    tesseract_command: str = "tesseract"
    tesseract_pdf_renderer_command: str = "pdftoppm"
    tesseract_dpi: int = 300
    snowflake_parse_mode: str = "OCR"
    snowflake_extract_images: bool = False
    snowflake_page_split: bool = True


PageWindow = tuple[int | None, int | None]

# What each provider's own page filter takes: the first page's number, whether
# several windows may be listed, and whether an end may be left open. Builtin
# text and Tesseract count pages the way readers do; MinerU and Snowflake
# expose index-style ids.
PAGE_RANGE_GRAMMARS: dict[str, tuple[int, bool, bool]] = {
    "builtin_text": (1, True, False),
    "tesseract_internal": (1, False, True),
    "mineru_internal": (0, False, True),
    "mineru_api": (0, False, True),
    "snowflake_cortex": (0, True, True),
}


def parse_page_range(text: str | None, provider: str) -> list[PageWindow]:
    """Read ``page_range`` as inclusive windows in the provider's own numbering.

    Every provider takes the same "3", "2-5" or "1-3,7" spelling, but they
    number pages from zero or one and accept different subsets of it, so the
    grammar is looked up by provider and an error names that provider. An
    open end is ``None``.
    """

    minimum, allow_multiple, allow_open = PAGE_RANGE_GRAMMARS[provider]
    raw = (text or "").strip()
    if not raw:
        return []
    if "," in raw and not allow_multiple:
        raise ValueError(
            f"Invalid {provider} page_range '{raw}': comma-separated ranges are not supported"
        )
    windows: list[PageWindow] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            raise ValueError(f"Invalid {provider} page_range '{raw}': empty range segment")
        if "-" not in token:
            page = _page_index(token, raw, provider, minimum)
            windows.append((page, page))
            continue
        start_raw, end_raw = (piece.strip() for piece in token.split("-", 1))
        if not start_raw and not end_raw:
            raise ValueError(f"Invalid {provider} page_range '{raw}': empty range segment")
        if not allow_open and (not start_raw or not end_raw):
            raise ValueError(
                f"Invalid {provider} page_range '{raw}': open-ended ranges are not supported"
            )
        start = _page_index(start_raw, raw, provider, minimum) if start_raw else None
        end = _page_index(end_raw, raw, provider, minimum) if end_raw else None
        if start is not None and end is not None and end < start:
            raise ValueError(
                f"Invalid {provider} page_range '{raw}': end page is before start page"
            )
        windows.append((start, end))
    return windows


def _page_index(value: str, raw: str, provider: str, minimum: int) -> int:
    numbering = "one-based page numbers" if minimum == 1 else "zero-based page ids"
    try:
        page = int(value)
    except ValueError:
        page = minimum - 1
    if page < minimum:
        raise ValueError(
            f"Invalid {provider} page number '{value}' in page_range '{raw}'. "
            f"{provider} page_range uses {numbering}."
        )
    return page


class OcrProvider(Protocol):
    """Converts an input file into the canonical parsed-document model."""

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        """Parse the source file into pages, blocks, assets, and provider metadata."""
        ...
