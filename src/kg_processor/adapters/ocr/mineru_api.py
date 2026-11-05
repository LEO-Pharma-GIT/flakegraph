# SPDX-License-Identifier: Apache-2.0
"""External MinerU-compatible OCR service adapter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import httpx

from kg_processor.adapters.ocr.generic_http import (
    _json_payload,
    _optional_bool_text,
    _set_if_present,
    raise_for_status,
)
from kg_processor.adapters.ocr.mineru_common import (
    HALF_TURN,
    QUARTER_TURNS,
    first_int,
    first_string,
    mineru_assets_from_payloads,
    renumber_page,
    resolve_page_window,
    text_length,
    unread_pdf_pages,
    write_rotated_page,
)
from kg_processor.domain.documents import (
    InputFile,
    LayoutBlock,
    ParsedAsset,
    ParsedDocument,
    ParsedPage,
)
from kg_processor.domain.ids import stable_id
from kg_processor.ports.ocr import NoTextReadError, OcrOptions

# A parse response carries markdown, layout blocks and inline assets for one
# document; anything past this is a broken or hostile upstream, not a document.
_MAX_RESPONSE_BYTES = 100 * 1024 * 1024


class MineruApiOcrProvider:
    """Calls an external MinerU-compatible `/file_parse` service."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        """Normalize service configuration and create a lazy HTTP client pool."""

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client()

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        """Upload a document to MinerU and normalize markdown, blocks, and assets."""

        result = self._parse_result(file.path, file.mime_type, _form_data(options), options)
        pages = _pages_from_result(file, result)
        assets = _assets_from_result(file, result)
        pages, assets = self._read_turned_pages(file, options, pages, assets)
        pages = _require_text(file, pages)
        return ParsedDocument(
            file_id=file.id,
            checksum=file.checksum,
            source_uri=file.source_uri,
            mime_type=file.mime_type,
            pages=pages,
            assets=assets,
            provider_metadata={
                "provider": "mineru_api",
                "base_url": self.base_url,
                "language": options.language,
                "page_range": options.page_range,
            },
        )

    def _parse_result(
        self,
        path: Path,
        mime_type: str,
        form_data: dict[str, str],
        options: OcrOptions,
    ) -> dict[str, Any]:
        """POST one file to ``/file_parse`` and return its result object."""

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # Stream the response so the configured bound is enforced while bytes
        # arrive rather than after httpx has already buffered an oversized body.
        with (
            path.open("rb") as handle,
            self._client.stream(
                "POST",
                f"{self.base_url}/file_parse",
                files={"files": (path.name, handle, mime_type)},
                headers=headers,
                data=form_data,
                timeout=options.timeout_seconds,
            ) as response,
        ):
            raise_for_status(response, "mineru_api")
            payload = _json_payload(response, _MAX_RESPONSE_BYTES)
        result = _select_result(payload)
        _raise_if_failed(result, path.name)
        return result

    def _read_turned_pages(
        self,
        file: InputFile,
        options: OcrOptions,
        pages: list[ParsedPage],
        assets: list[ParsedAsset],
    ) -> tuple[list[ParsedPage], list[ParsedAsset]]:
        """Recover unread PDF pages within the requested inclusive page window.

        Each selected unread page is retried in both quarter turns; the parse
        with more text wins. A half turn is tried when neither quarter turn
        reads text. Readable pages and pages outside the window are not resent.
        """

        start, end = resolve_page_window(options, provider="mineru_api")
        unread = [
            number
            for number in unread_pdf_pages(file.path, pages)
            if (start is None or number - 1 >= start) and (end is None or number - 1 <= end)
        ]
        if not unread:
            return pages, assets
        form_data = _form_data(options)
        # The turned copy holds one page, so the caller's page window must
        # not be applied to it a second time.
        form_data.pop("start_page_id", None)
        form_data.pop("end_page_id", None)
        by_number = {page.page_number: page for page in pages}
        with tempfile.TemporaryDirectory(prefix="kg-mineru-turn-") as tmp:
            for page_number in unread:
                best: tuple[int, ParsedPage, list[ParsedAsset]] | None = None
                for degrees in (*QUARTER_TURNS, HALF_TURN):
                    if degrees == HALF_TURN and best is not None:
                        break
                    turned = Path(tmp) / f"page-{page_number}-{degrees}.pdf"
                    write_rotated_page(file.path, page_number, degrees, turned)
                    result = self._parse_result(turned, file.mime_type, form_data, options)
                    read = [page for page in _pages_from_result(file, result) if text_length(page)]
                    if not read:
                        continue
                    page = renumber_page(
                        read[0], page_number, degrees=degrees, id_prefix="mineru_api_block"
                    )
                    turned_assets = [
                        asset.model_copy(update={"page_number": page_number})
                        for asset in _assets_from_result(file, result)
                    ]
                    if best is None or text_length(page) > best[0]:
                        best = (text_length(page), page, turned_assets)
                if best is not None:
                    by_number[page_number] = best[1]
                    assets = [*assets, *best[2]]
        return [by_number[number] for number in sorted(by_number)], assets


def _form_data(options: OcrOptions) -> dict[str, str]:
    data = {
        "return_md": "true",
        "return_middle_json": "true",
        "return_content_list": "true",
        "return_images": _bool_text(options.image_analysis is True),
    }
    _set_if_present(data, "lang_list", options.language)
    _set_if_present(data, "backend", options.backend)
    _set_if_present(data, "parse_method", options.method)
    # MinerU's API has no page_range field; FastAPI drops what it does not
    # declare, so the window has to travel as the ids it does take.
    start_page_id, end_page_id = resolve_page_window(options, provider="mineru_api")
    _set_if_present(data, "start_page_id", _optional_int_text(start_page_id))
    _set_if_present(data, "end_page_id", _optional_int_text(end_page_id))
    _set_if_present(data, "formula_enable", _optional_bool_text(options.formula))
    _set_if_present(data, "table_enable", _optional_bool_text(options.table))
    return data


def _select_result(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("MinerU API response must be a JSON object")
    for key in ("result", "data"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    results = payload.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            return first
    if isinstance(results, dict):
        for value in results.values():
            if isinstance(value, dict):
                return value
    return payload


def _raise_if_failed(result: dict[str, Any], source_uri: str) -> None:
    status = str(result.get("status", "")).lower()
    if status in {"failed", "error"}:
        raise RuntimeError(f"MinerU API failed for {source_uri}: {result}")
    if result.get("error"):
        raise RuntimeError(f"MinerU API failed for {source_uri}: {result['error']}")


def _pages_from_result(file: InputFile, result: dict[str, Any]) -> list[ParsedPage]:
    pages = _pages_from_raw_pages(file, result.get("pages"))
    if pages:
        return pages
    middle_json = _maybe_json_object(result.get("middle_json"))
    if middle_json:
        pages = _pages_from_raw_pages(file, middle_json.get("pages"))
        if pages:
            return pages
    # The content list is the parse with its structure intact: every item
    # names its page and what it is. The flat markdown loses both, and a
    # document read from it cites page 1 for everything.
    content_list = result.get("content_list")
    if isinstance(content_list, str):
        content_list = _maybe_json_list(content_list)
    if isinstance(content_list, list):
        pages = _pages_from_content_list(file, content_list)
        if pages:
            return pages
    markdown = first_string(result, ["md_content", "markdown", "md", "content", "text"])
    if markdown:
        return [_page(file, 1, markdown, result)]
    # An answer with none of the three is not a parse; an answer with all of
    # them empty is MinerU reading nothing, which the caller may still recover
    # page by page.
    if not any(key in result for key in ("md_content", "markdown", "md", "content_list", "pages")):
        raise RuntimeError("MinerU API response did not include markdown, content_list, or pages")
    return []


# Layout furniture MinerU itself leaves out of its markdown.
_SILENT_CONTENT_TYPES = frozenset({"page_number", "aside_text", "discarded"})


def _pages_from_content_list(file: InputFile, content_list: list[Any]) -> list[ParsedPage]:
    """Rebuild each page from its typed items, the way MinerU writes its markdown."""

    grouped: dict[int, list[Any]] = {}
    for item in content_list:
        page_index = (
            first_int(item, ["page_idx", "page_index", "index"]) if isinstance(item, dict) else None
        )
        grouped.setdefault((page_index or 0) + 1, []).append(item)
    pages: list[ParsedPage] = []
    for page_number, items in sorted(grouped.items()):
        blocks: list[LayoutBlock] = []
        for index, item in enumerate(items):
            kind = str(item.get("type") or "text") if isinstance(item, dict) else "text"
            if kind in _SILENT_CONTENT_TYPES:
                continue
            text = _content_list_text(item)
            if not text.strip():
                continue
            metadata = (
                {key: value for key, value in item.items() if key not in _CONTENT_TEXT_KEYS}
                if isinstance(item, dict)
                else {}
            )
            blocks.append(
                LayoutBlock(
                    id=stable_id("mineru_api_block", file.id, page_number, index, text[:256]),
                    page_number=page_number,
                    kind=kind,
                    text=text,
                    metadata=metadata,
                )
            )
        if not blocks:
            continue
        text = "\n\n".join(block.text for block in blocks)
        pages.append(
            ParsedPage(page_number=page_number, markdown=text, raw_text=text, blocks=blocks)
        )
    return pages


def _require_text(file: InputFile, pages: list[ParsedPage]) -> list[ParsedPage]:
    """Reject a parse that produced no text on any page.

    A document that reaches the graph with no content contributes nothing and is
    indistinguishable from one that was never processed, so an empty result is an
    error rather than an empty success.
    """

    if not any(page.raw_text.strip() or page.markdown.strip() for page in pages):
        raise NoTextReadError(f"MinerU API returned no text for {file.path.name}")
    return pages


def _assets_from_result(file: InputFile, result: dict[str, Any]) -> list[ParsedAsset]:
    payloads = [result]
    middle_json = _maybe_json_object(result.get("middle_json"))
    if middle_json:
        payloads.append(middle_json)
    return mineru_assets_from_payloads(
        file,
        payloads,
        id_prefix="mineru_api_asset",
    )


def _pages_from_raw_pages(file: InputFile, raw_pages: object) -> list[ParsedPage]:
    if not isinstance(raw_pages, list):
        return []
    pages: list[ParsedPage] = []
    for index, raw_page in enumerate(raw_pages, start=1):
        if isinstance(raw_page, dict):
            page_number = first_int(raw_page, ["page_number", "page", "page_id"])
            if page_number is None:
                page_idx = first_int(raw_page, ["page_idx", "page_index", "index"])
                page_number = page_idx + 1 if page_idx is not None else index
            text = first_string(raw_page, ["markdown", "md", "content", "text", "raw_text"])
            pages.append(_page(file, page_number, text, raw_page))
        else:
            pages.append(_page(file, index, str(raw_page), {}))
    return pages


def _page(file: InputFile, page_number: int, text: str, metadata: dict[str, Any]) -> ParsedPage:
    block = LayoutBlock(
        id=stable_id("mineru_api_block", file.id, page_number, text[:256]),
        page_number=page_number,
        kind=str(metadata.get("type") or metadata.get("kind") or "text"),
        text=text,
        metadata={
            key: value
            for key, value in metadata.items()
            if key not in {"markdown", "md", "content", "text", "raw_text"}
        },
    )
    return ParsedPage(
        page_number=page_number,
        markdown=text,
        raw_text=text,
        blocks=[block],
        detected_language=first_string(metadata, ["language", "detected_language"]) or None,
    )


_PRIMARY_TEXT_KEYS = ("text", "content", "markdown", "md")
_PART_TEXT_KEYS = (
    "list_items",
    "image_caption",
    "image_footnote",
    "table_caption",
    "table_body",
    "table_footnote",
    "chart_caption",
    "chart_footnote",
    "code_caption",
    "code_body",
    "code_footnote",
)
_CONTENT_TEXT_KEYS = _PRIMARY_TEXT_KEYS + _PART_TEXT_KEYS


def _content_list_text(item: object) -> str:
    if isinstance(item, dict):
        values: list[str] = []
        primary = first_string(item, list(_PRIMARY_TEXT_KEYS))
        if primary:
            level = first_int(item, ["text_level"])
            values.append(f"{'#' * level} {primary}" if level else primary)
        for key in _PART_TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, list):
                values.extend(str(part) for part in value if str(part).strip())
            elif value is not None and str(value).strip():
                values.append(str(value))
        return "\n".join(values)
    return str(item)


def _maybe_json_list(value: str) -> list[Any] | None:
    try:
        parsed = json.loads(value)
    except ValueError:
        return None
    return parsed if isinstance(parsed, list) else None


def _maybe_json_object(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _optional_int_text(value: int | None) -> str | None:
    return str(value) if value is not None else None


def _bool_text(value: bool) -> str:
    return str(value).lower()
