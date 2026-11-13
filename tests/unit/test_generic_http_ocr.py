from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from http_fakes import StreamingClient, input_file

from kg_processor.adapters.ocr.generic_http import GenericHttpOcrProvider
from kg_processor.config.settings import GenericHttpOcrSettings
from kg_processor.ports.ocr import OcrOptions


def test_generic_http_ocr_posts_file_and_maps_pages_blocks_and_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "payload": {
                "state": "done",
                "pages": [
                    {
                        "page": "2",
                        "md": "# Page two",
                        "text": "Page two",
                        "lang": "en",
                        "layout": [
                            {
                                "block_id": "b1",
                                "type": "heading",
                                "content": "Page two",
                                "box": [0, 1, 2, 3],
                                "score": "0.94",
                                "attributes": {"font": "bold"},
                            }
                        ],
                    }
                ],
                "media": [
                    {
                        "asset_id": "img1",
                        "asset_type": "image",
                        "href": "https://ocr.example/assets/img1.png",
                        "page": "2",
                        "score": 0.81,
                        "details": {"width": 640, "height": 480},
                    }
                ],
                "warnings": ["low-confidence"],
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = GenericHttpOcrProvider(
        GenericHttpOcrSettings(
            endpoint="https://ocr.example/parse",
            api_key="secret",
            file_field="document",
            result_path="payload",
            status_path="state",
            page_number_path="page",
            markdown_path="md",
            raw_text_path="text",
            detected_language_path="lang",
            blocks_path="layout",
            block_id_path="block_id",
            block_kind_path="type",
            block_text_path="content",
            block_bbox_path="box",
            block_confidence_path="score",
            block_metadata_path="attributes",
            assets_path="media",
            asset_id_path="asset_id",
            asset_kind_path="asset_type",
            asset_uri_path="href",
            asset_page_number_path="page",
            asset_confidence_path="score",
            asset_metadata_path="details",
        )
    )

    document = provider.parse(
        input_file(input_path),
        OcrOptions(language="en", page_range="1-2", formula=True, table=False),
    )

    request = client.requests[0]
    assert request["url"] == "https://ocr.example/parse"
    assert request["headers"] == {"Authorization": "Bearer secret"}
    assert request["data"] == {
        "checksum": "checksum",
        "file_id": "file_1",
        "formula": "true",
        "language": "en",
        "mime_type": "application/pdf",
        "page_range": "1-2",
        "source_uri": str(input_path),
        "table": "false",
    }
    assert "document" in request["files"]
    assert document.pages[0].page_number == 2
    assert document.pages[0].markdown == "# Page two"
    assert document.pages[0].detected_language == "en"
    assert document.pages[0].blocks[0].kind == "heading"
    assert document.pages[0].blocks[0].bbox == (0.0, 1.0, 2.0, 3.0)
    assert document.pages[0].blocks[0].metadata == {"confidence": 0.94, "font": "bold"}
    assert document.assets[0].id == "img1"
    assert document.assets[0].page_number == 2
    assert document.assets[0].metadata == {"confidence": 0.81, "height": 480, "width": 640}
    assert document.provider_metadata["provider"] == "generic_http"
    assert document.provider_metadata["warnings"] == ["low-confidence"]


def test_generic_http_ocr_accepts_result_level_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient({"result": {"markdown": "Alice works at Acme."}})
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = GenericHttpOcrProvider(GenericHttpOcrSettings(endpoint="https://ocr.example/parse"))

    document = provider.parse(input_file(input_path), OcrOptions())

    assert document.pages[0].page_number == 1
    assert document.pages[0].markdown == "Alice works at Acme."
    assert document.pages[0].raw_text == "Alice works at Acme."
    assert document.pages[0].blocks[0].text == "Alice works at Acme."


def test_generic_http_ocr_raises_mapped_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient({"result": {"status": "error", "error": "bad file"}})
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = GenericHttpOcrProvider(GenericHttpOcrSettings(endpoint="https://ocr.example/parse"))

    with pytest.raises(RuntimeError, match="Generic HTTP OCR failed"):
        provider.parse(input_file(input_path), OcrOptions())


def test_generic_http_ocr_rejects_nonempty_pages_without_mapped_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient({"result": {"pages": [{"unmapped": "text"}]}})
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = GenericHttpOcrProvider(GenericHttpOcrSettings(endpoint="https://ocr.example/parse"))

    with pytest.raises(RuntimeError, match="pages did not include mapped text"):
        provider.parse(input_file(input_path), OcrOptions())


def test_generic_http_ocr_preserves_zero_based_page_number(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient({"result": {"pages": [{"page_number": 0, "text": "zero"}]}})
    monkeypatch.setattr(httpx, "Client", client.open)

    document = GenericHttpOcrProvider(
        GenericHttpOcrSettings(
            endpoint="https://ocr.example/parse",
            page_number_path="page_number",
            raw_text_path="text",
        )
    ).parse(input_file(input_path), OcrOptions())

    assert document.pages[0].page_number == 0


def test_generic_http_ocr_rejects_declared_oversized_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        content=b'{"result":{"markdown":"ok"}}', response_headers={"content-length": "1000"}
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = GenericHttpOcrProvider(
        GenericHttpOcrSettings(
            endpoint="https://ocr.example/parse",
            max_response_bytes=10,
        )
    )

    with pytest.raises(RuntimeError, match="above the configured 10 byte limit"):
        provider.parse(input_file(input_path), OcrOptions())


def test_generic_http_ocr_rejects_actual_oversized_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(content=b'{"result":{"markdown":"too large"}}')
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = GenericHttpOcrProvider(
        GenericHttpOcrSettings(
            endpoint="https://ocr.example/parse",
            max_response_bytes=10,
        )
    )

    with pytest.raises(RuntimeError, match="above the configured 10 byte limit"):
        provider.parse(input_file(input_path), OcrOptions())


def test_generic_http_ocr_rejects_invalid_json_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(content=b"not-json")
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = GenericHttpOcrProvider(GenericHttpOcrSettings(endpoint="https://ocr.example/parse"))

    with pytest.raises(ValueError, match="response was not valid JSON"):
        provider.parse(input_file(input_path), OcrOptions())


def test_generic_http_ocr_reports_missing_configured_result_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient({"payload": {}})
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = GenericHttpOcrProvider(
        GenericHttpOcrSettings(
            endpoint="https://ocr.example/parse",
            result_path="payload.result",
        )
    )

    with pytest.raises(ValueError, match="missing result path: payload.result"):
        provider.parse(input_file(input_path), OcrOptions())


def test_generic_http_ocr_normalizes_text_pages_and_primitive_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "result": {
                "pages": [
                    "First page as plain text.",
                    {
                        "blocks": [
                            "Second page heading",
                            {
                                "text": "Second page body",
                                "bbox": [0, 1, "invalid", 3],
                                "confidence": "invalid",
                            },
                        ]
                    },
                ]
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = GenericHttpOcrProvider(GenericHttpOcrSettings(endpoint="https://ocr.example/parse"))

    document = provider.parse(input_file(input_path), OcrOptions())

    assert document.pages[0].markdown == "First page as plain text."
    assert [block.text for block in document.pages[0].blocks] == ["First page as plain text."]
    assert document.pages[0].blocks[0].kind == "text"
    assert document.pages[1].page_number == 2
    assert document.pages[1].markdown == "Second page heading\n\nSecond page body"
    assert [block.text for block in document.pages[1].blocks] == [
        "Second page heading",
        "Second page body",
    ]
    assert document.pages[1].blocks[1].bbox is None
    assert "confidence" not in document.pages[1].blocks[1].metadata
