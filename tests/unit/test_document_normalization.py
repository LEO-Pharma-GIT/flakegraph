from __future__ import annotations

from pathlib import Path

from kg_processor.application.document_normalization import build_document_artifacts
from kg_processor.domain.documents import (
    InputFile,
    LayoutBlock,
    ParsedAsset,
    ParsedDocument,
    ParsedPage,
)


def test_build_document_artifacts_maps_pages_blocks_assets_and_trace() -> None:
    file = InputFile(
        id="file_1",
        path=Path("sample.pdf"),
        source_uri="file:///sample.pdf",
        checksum="abc",
        mime_type="application/pdf",
        size_bytes=123,
    )
    parsed = ParsedDocument(
        file_id="file_1",
        checksum="abc",
        source_uri="file:///sample.pdf",
        mime_type="application/pdf",
        pages=[
            ParsedPage(
                page_number=1,
                markdown="# Title",
                raw_text="Title",
                detected_language="en",
                blocks=[
                    LayoutBlock(
                        id="block_1",
                        page_number=1,
                        kind="heading",
                        text="Title",
                        bbox=(1.0, 2.0, 3.0, 4.0),
                        metadata={"level": 1},
                    )
                ],
            )
        ],
        assets=[
            ParsedAsset(
                id="asset_1",
                kind="image",
                page_number=1,
                uri="file:///figure.png",
                metadata={"caption": "Figure"},
            )
        ],
        provider_metadata={"provider": "test_ocr"},
    )

    artifacts = build_document_artifacts("graph", file, parsed, "ocr_cache_1", True)

    assert artifacts.document == {
        "id": "file_1",
        "graph_id": "graph",
        "file_id": "file_1",
        "checksum": "abc",
        "source_uri": "file:///sample.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 123,
        "ocr_provider": "test_ocr",
        "kind": "document",
        "summary": None,
        "related_file_ids": [],
        "folder": None,
        "provenance": "extracted",
    }
    assert artifacts.pages[0]["page_number"] == 1
    assert artifacts.pages[0]["detected_language"] == "en"
    assert artifacts.blocks == [
        {
            "id": "block_1",
            "graph_id": "graph",
            "file_id": "file_1",
            "page_number": 1,
            "kind": "heading",
            "text": "Title",
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "metadata": {"level": 1},
        }
    ]
    assert artifacts.assets == [
        {
            "id": "asset_1",
            "graph_id": "graph",
            "file_id": "file_1",
            "kind": "image",
            "page_number": 1,
            "uri": "file:///figure.png",
            "metadata": {"caption": "Figure"},
        }
    ]
    assert artifacts.trace_event == {
        "stage": "ocr",
        "file_id": "file_1",
        "cache_id": "ocr_cache_1",
        "cache_hit": True,
        "provider": "test_ocr",
        "pages": 1,
        "assets": 1,
        "blocks": 1,
    }
