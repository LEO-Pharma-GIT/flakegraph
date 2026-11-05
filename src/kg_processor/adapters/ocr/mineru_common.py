# SPDX-License-Identifier: Apache-2.0
"""Shared OCR payload normalization helpers.

Internal MinerU CLI output, MinerU-compatible HTTP services, and Snowflake
Cortex expose similar field and asset shapes but arrive through different
transport paths. Keeping field lookup, asset normalization, and blob redaction
here prevents those adapters from drifting while still letting each adapter own
its page/block response parsing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter

from kg_processor.domain.documents import InputFile, ParsedAsset, ParsedPage
from kg_processor.domain.ids import stable_id
from kg_processor.ports.ocr import OcrOptions, PageWindow, parse_page_range


def resolve_page_window(options: OcrOptions, *, provider: str) -> PageWindow:
    """Return the zero-based inclusive window MinerU takes as start/end page ids.

    Both the CLI and the HTTP API take the same pair, so the explicit ids win
    and the portable ``page_range`` is read the same way for either.
    """

    if options.start_page_id is not None or options.end_page_id is not None:
        return options.start_page_id, options.end_page_id
    windows = parse_page_range(options.page_range, provider)
    return windows[0] if windows else (None, None)


def shape_description(value: object) -> str:
    """Describe a rejected payload by shape without echoing its content."""

    if isinstance(value, dict):
        return f"object with keys {sorted(str(key) for key in value)}"
    return type(value).__name__


_ASSET_COLLECTION_KEYS = ("images", "assets", "media", "figures")
# Keys whose values are raw bytes rather than descriptive metadata. Providers
# disagree on which name they use for the same inline payload, so one document's
# base64 image is spelled ``content`` by one service and ``content_bytes`` by
# another. Redacting all of them keeps a document's bytes out of the provenance
# metadata that traces, caches, and artifacts persist.
_BLOB_METADATA_KEYS = frozenset(
    {
        "base64",
        "bytes",
        "content",
        "content_bytes",
        "data",
        "image_base64",
        "image_data",
    }
)
# The subset that identifies a record as an inline binary asset. ``content`` is
# excluded because a MinerU content-list paragraph carries its text under that
# name and is not an asset.
_ASSET_BLOB_KEYS = _BLOB_METADATA_KEYS - {"content"}


def mineru_assets_from_payloads(
    file: InputFile,
    payloads: list[dict[str, Any]],
    *,
    id_prefix: str,
) -> list[ParsedAsset]:
    """Extract normalized assets from MinerU middle-json/content payloads."""

    assets: list[ParsedAsset] = []
    for payload in payloads:
        _extend_assets(file, assets, payload, id_prefix=id_prefix)
    return _dedupe_assets(assets)


def _extend_assets(
    file: InputFile,
    assets: list[ParsedAsset],
    payload: dict[str, Any],
    *,
    id_prefix: str,
) -> None:
    for key in _ASSET_COLLECTION_KEYS:
        _append_assets_from_collection(
            file,
            assets,
            payload.get(key),
            parent_page_number=None,
            id_prefix=id_prefix,
        )
    raw_pages = payload.get("pages")
    if isinstance(raw_pages, list):
        for page_index, raw_page in enumerate(raw_pages, start=1):
            if not isinstance(raw_page, dict):
                continue
            page_number = first_int(raw_page, ["page_number", "page", "page_id"])
            if page_number is None:
                page_number = page_index
            for key in _ASSET_COLLECTION_KEYS:
                _append_assets_from_collection(
                    file,
                    assets,
                    raw_page.get(key),
                    parent_page_number=page_number,
                    id_prefix=id_prefix,
                )
    content_list = payload.get("content_list")
    if isinstance(content_list, list):
        _append_assets_from_collection(
            file,
            assets,
            content_list,
            parent_page_number=None,
            id_prefix=id_prefix,
        )


def _append_assets_from_collection(
    file: InputFile,
    assets: list[ParsedAsset],
    value: object,
    *,
    parent_page_number: int | None,
    id_prefix: str,
) -> None:
    if not isinstance(value, list):
        return
    for index, item in enumerate(value):
        if isinstance(item, dict) and _looks_like_asset(item):
            assets.append(_asset_from_raw(file, item, index, parent_page_number, id_prefix))


def _looks_like_asset(raw: dict[str, Any]) -> bool:
    kind = first_string(raw, ["type", "kind", "category"]).lower()
    return bool(
        kind in {"image", "figure", "table", "formula", "asset"}
        or first_string(raw, ["uri", "url", "path", "img_path", "image_path"])
        or any(key in raw for key in _ASSET_BLOB_KEYS)
    )


def _asset_from_raw(
    file: InputFile,
    raw: dict[str, Any],
    index: int,
    parent_page_number: int | None,
    id_prefix: str,
) -> ParsedAsset:
    raw_id = first_string(raw, ["id", "asset_id", "image_id", "img_id", "name"])
    page_number = first_int(raw, ["page_number", "page", "page_id"])
    if page_number is None:
        page_number = _page_from_zero_based_index(raw)
    if page_number is None:
        page_number = parent_page_number
    uri = first_string(raw, ["uri", "url", "path", "img_path", "image_path"]) or None
    kind = str(raw.get("type") or raw.get("kind") or "image")
    metadata = redact_blob_metadata(raw)
    return ParsedAsset(
        id=raw_id or stable_id(id_prefix, file.id, page_number, index, kind, uri or ""),
        kind=kind,
        page_number=page_number,
        uri=uri,
        metadata=metadata,
    )


def _page_from_zero_based_index(raw: dict[str, Any]) -> int | None:
    page_index = first_int(raw, ["page_index", "page_idx", "index"])
    return page_index + 1 if page_index is not None else None


def redact_blob_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Replace inline blob fields with their presence and length.

    Asset provenance must describe what was found without carrying the bytes
    themselves into traces, caches, and persisted artifacts.
    """

    metadata: dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in _BLOB_METADATA_KEYS:
            metadata[f"{key}_present"] = bool(value)
            metadata[f"{key}_length"] = len(value) if isinstance(value, str | bytes) else None
        else:
            metadata[key] = value
    return metadata


def _dedupe_assets(assets: list[ParsedAsset]) -> list[ParsedAsset]:
    deduped: list[ParsedAsset] = []
    seen: set[str] = set()
    for asset in assets:
        if asset.id in seen:
            continue
        seen.add(asset.id)
        deduped.append(asset)
    return deduped


def first_string(payload: dict[str, Any], keys: list[str]) -> str:
    """Return the first string-valued field from ``payload`` or an empty string."""

    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def first_int(payload: dict[str, Any], keys: list[str]) -> int | None:
    """Return the first integer-like field from ``payload`` when one exists."""

    for key in keys:
        value = payload.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return None


# The orientations a scanner leaves a page in. A landscape form fed to a
# portrait scanner comes out turned a quarter either way; upside down is rare
# and is tried last, only when neither quarter turn read anything.
QUARTER_TURNS: tuple[int, int] = (90, 270)
HALF_TURN = 180


def unread_pdf_pages(path: Path, pages: Sequence[ParsedPage]) -> list[int]:
    """Return the 1-based numbers of PDF pages the parse produced no text for.

    MinerU's layout model reads nothing from a page whose text runs
    vertically, and says nothing about it: the page is simply absent from the
    result, or present with no blocks. This names those pages so the caller
    can try them turned, which is what a scanned landscape form needs.
    """

    if path.suffix.lower() != ".pdf":
        return []
    try:
        page_count = len(PdfReader(path).pages)
    except Exception:  # noqa: BLE001 - a PDF pypdf cannot open has no pages to turn
        return []
    read = {page.page_number for page in pages if page.raw_text.strip() or page.markdown.strip()}
    return [number for number in range(1, page_count + 1) if number not in read]


def write_rotated_page(source: Path, page_number: int, degrees: int, target: Path) -> None:
    """Write one page of ``source`` as a single-page PDF turned by ``degrees``.

    The turn is recorded as the page's /Rotate, which every renderer applies
    before rasterising, so the page's pixels are untouched and the OCR sees
    the same scan the right way up.
    """

    reader = PdfReader(source)
    writer = PdfWriter()
    page = reader.pages[page_number - 1]
    page.rotate(degrees)
    writer.add_page(page)
    with target.open("wb") as handle:
        writer.write(handle)


def renumber_page(
    page: ParsedPage, page_number: int, *, degrees: int, id_prefix: str
) -> ParsedPage:
    """Renumber a turned page while retaining source identity in its block ids.

    The original block id is scoped to its input file, so identical recovered
    text in different documents remains distinct in canonical block tables.
    """

    blocks = [
        block.model_copy(
            update={
                "id": stable_id(id_prefix, block.id, page_number, degrees, index, block.text[:256]),
                "page_number": page_number,
                "metadata": {**block.metadata, "rotation_degrees": degrees},
            }
        )
        for index, block in enumerate(page.blocks)
    ]
    return page.model_copy(update={"page_number": page_number, "blocks": blocks})


def text_length(page: ParsedPage) -> int:
    """How much a page read, for choosing between orientations."""

    return len(page.raw_text.strip()) or len(page.markdown.strip())
