# SPDX-License-Identifier: Apache-2.0
"""Snowflake Cortex OCR adapter using `AI_PARSE_DOCUMENT` on staged files."""

from __future__ import annotations

import json
from typing import Any, cast

from kg_processor.adapters.ocr.mineru_common import (
    first_int,
    first_string,
    redact_blob_metadata,
    shape_description,
)
from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    ReusableSnowflakeConnections,
    SnowflakeConnectionConfig,
    as_json_object,
    scalar_from_first_row,
    split_stage_uri,
)
from kg_processor.domain.documents import (
    InputFile,
    LayoutBlock,
    ParsedAsset,
    ParsedDocument,
    ParsedPage,
)
from kg_processor.domain.ids import stable_id
from kg_processor.ports.ocr import OcrOptions, parse_page_range

_SNOWFLAKE_PARSE_SQL = "SELECT AI_PARSE_DOCUMENT(TO_FILE(?, ?), PARSE_JSON(?)::OBJECT, TRUE)"
_PAGE_ADDRESSABLE_SUFFIXES = {".docx", ".pdf", ".pptx"}
_CONTENT_KEYS = ["content", "markdown", "text", "raw_text"]


class SnowflakeCortexOcrProvider:
    """Runs Snowflake Cortex `AI_PARSE_DOCUMENT` for staged input files."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        connector_factory: ConnectorFactory | None = None,
    ) -> None:
        self.config = config
        self.connector_factory = connector_factory
        self._connections = ReusableSnowflakeConnections(config, connector_factory)

    def close(self) -> None:
        """Release retained Snowflake sessions."""

        self._connections.close()

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        """Parse a Snowflake stage file and normalize Cortex OCR output."""

        stage, relative_path = split_stage_uri(file.source_uri)
        parse_options = _build_parse_options(file, options)
        with self._connections.cursor() as cursor:
            cast(Any, cursor).execute(
                _SNOWFLAKE_PARSE_SQL,
                [stage, relative_path, json.dumps(parse_options, sort_keys=True)],
                timeout=options.timeout_seconds,
            )
            raw_result = scalar_from_first_row(cursor)

        result = as_json_object(raw_result)
        value, metadata = _extract_value_or_raise(result, file.source_uri)
        pages = _pages_from_value(file, value, file.source_uri)
        # Every page carries one block built from its own text, so block presence
        # says nothing about content. A document that parsed to nothing must fail
        # here rather than be recorded as a successful parse contributing no text.
        if not any(page.raw_text.strip() or page.markdown.strip() for page in pages):
            raise RuntimeError(
                f"Snowflake AI_PARSE_DOCUMENT returned no text for {file.source_uri}"
            )
        assets = _assets_from_value(file, value)
        return ParsedDocument(
            file_id=file.id,
            checksum=file.checksum,
            source_uri=file.source_uri,
            mime_type=file.mime_type,
            pages=pages,
            assets=assets,
            provider_metadata={
                "provider": "snowflake_cortex",
                "stage": stage,
                "relative_path": relative_path,
                "parse_options": parse_options,
                "snowflake_metadata": metadata,
            },
        )


def _build_parse_options(file: InputFile, options: OcrOptions) -> dict[str, object]:
    """Build only the parse options supported by the staged file format.

    Snowflake accepts ``page_split`` and ``page_filter`` for PDF, PowerPoint,
    and Word documents, but rejects those options for text and HTML inputs.
    Markdown files are handled by Cortex as text files. Keeping this decision in
    the adapter lets one stage prefix safely contain all supported document types
    without leaking Snowflake-specific branching into the pipeline.
    """

    parse_options: dict[str, object] = {"mode": options.snowflake_parse_mode}
    supports_pages = file.path.suffix.lower() in _PAGE_ADDRESSABLE_SUFFIXES
    if supports_pages:
        parse_options["page_split"] = options.snowflake_page_split
    if options.snowflake_extract_images:
        parse_options["extract_images"] = True
    page_filter = _page_filter_from_range(options.page_range) if supports_pages else None
    if page_filter:
        parse_options["page_filter"] = page_filter
    return parse_options


def _page_filter_from_range(page_range: str | None) -> list[dict[str, int]] | None:
    # AI_PARSE_DOCUMENT takes half-open windows; an open start begins at 0 and
    # an open end is simply omitted.
    filters: list[dict[str, int]] = []
    for start, end in parse_page_range(page_range, "snowflake_cortex"):
        entry = {"start": start or 0}
        if end is not None:
            entry["end"] = end + 1
        filters.append(entry)
    return filters or None


def _extract_value_or_raise(
    result: dict[str, Any],
    source_uri: str,
) -> tuple[object, dict[str, Any]]:
    error = result.get("error")
    if error:
        raise RuntimeError(f"Snowflake AI_PARSE_DOCUMENT failed for {source_uri}: {error}")
    value = result.get("value", result)
    metadata = result.get("metadata", {})
    return value, metadata if isinstance(metadata, dict) else {}


def _pages_from_value(file: InputFile, value: object, source_uri: str) -> list[ParsedPage]:
    """Map a recognized Cortex response shape onto pages, or reject the response.

    An unrecognized envelope is a provider change, not document text. Serializing
    it into a page would chunk, embed, and extract the response structure itself
    while the run reported success, so the shape is required to be one this
    adapter understands.
    """

    if isinstance(value, dict):
        raw_pages = value.get("pages")
        if isinstance(raw_pages, list):
            return [
                _page_from_raw(file, raw_page, index) for index, raw_page in enumerate(raw_pages)
            ]
        content = first_string(value, _CONTENT_KEYS)
        if content:
            return [_page_from_text(file, 1, content, value)]
    if isinstance(value, str):
        return [_page_from_text(file, 1, value, {})]
    raise RuntimeError(
        f"Snowflake AI_PARSE_DOCUMENT returned an unrecognized response for {source_uri}: "
        f"expected pages or one of {sorted(_CONTENT_KEYS)}, got {shape_description(value)}"
    )


def _page_from_raw(file: InputFile, raw_page: object, index: int) -> ParsedPage:
    if not isinstance(raw_page, dict):
        return _page_from_text(file, index + 1, str(raw_page), {})
    page_index = first_int(raw_page, ["index", "page_index"])
    page_number = first_int(raw_page, ["page_number", "page"])
    if page_number is None:
        page_number = page_index + 1 if page_index is not None else index + 1
    content = first_string(raw_page, _CONTENT_KEYS)
    return _page_from_text(file, page_number, content, raw_page)


def _page_from_text(
    file: InputFile,
    page_number: int,
    text: str,
    metadata: dict[str, Any],
) -> ParsedPage:
    block = LayoutBlock(
        id=stable_id("snowflake_block", file.id, page_number, text[:256]),
        page_number=page_number,
        kind=str(metadata.get("type") or metadata.get("kind") or "page"),
        text=text,
        metadata={key: value for key, value in metadata.items() if key not in _CONTENT_KEYS},
    )
    return ParsedPage(
        page_number=page_number,
        markdown=text,
        raw_text=text,
        blocks=[block],
        detected_language=first_string(metadata, ["language", "detected_language"]) or None,
    )


def _assets_from_value(file: InputFile, value: object) -> list[ParsedAsset]:
    if not isinstance(value, dict):
        return []
    raw_images = value.get("images")
    if not isinstance(raw_images, list):
        return []
    return [
        _asset_from_raw_image(file, raw_image, index)
        for index, raw_image in enumerate(raw_images)
        if isinstance(raw_image, dict)
    ]


def _asset_from_raw_image(file: InputFile, raw_image: dict[str, Any], index: int) -> ParsedAsset:
    raw_id = first_string(raw_image, ["id", "image_id", "name"])
    page_number = first_int(raw_image, ["page_number", "page"])
    page_index = first_int(raw_image, ["page_index", "index"])
    if page_number is None and page_index is not None:
        page_number = page_index + 1
    metadata = redact_blob_metadata(raw_image)
    return ParsedAsset(
        id=raw_id or stable_id("snowflake_asset", file.id, index, metadata),
        kind=str(raw_image.get("type") or raw_image.get("kind") or "image"),
        page_number=page_number,
        uri=first_string(raw_image, ["uri", "url", "path", "stage_uri"]) or None,
        metadata=metadata,
    )
