from __future__ import annotations

import json
from pathlib import Path

import pytest
from snowflake_fakes import CONFIG, FakeConnection

from kg_processor.adapters.ocr.snowflake_cortex import SnowflakeCortexOcrProvider
from kg_processor.domain.documents import InputFile
from kg_processor.ports.ocr import OcrOptions


def test_snowflake_cortex_ocr_parses_stage_file_with_options_and_assets() -> None:
    connection = FakeConnection(
        [
            [
                {
                    "value": {
                        "pages": [
                            {"index": 0, "content": "# Page 1\nAlice"},
                            {"index": 1, "content": "# Page 2\nAcme"},
                        ],
                        "images": [
                            {
                                "image_id": "img-1",
                                "page_index": 0,
                                "image_base64": "abc123",
                                "bbox": [1, 2, 3, 4],
                            }
                        ],
                    },
                    "metadata": {"document_size": 100},
                }
            ]
        ]
    )
    provider = SnowflakeCortexOcrProvider(CONFIG, connector_factory=lambda **_: connection)

    document = provider.parse(
        _input_file("@DB.SCHEMA.DOC_STAGE/docs/a.pdf"),
        OcrOptions(
            page_range="0-2",
            snowflake_parse_mode="LAYOUT",
            snowflake_extract_images=True,
        ),
    )

    sql, params = connection.cursor_instance.executed[0]
    assert sql == "SELECT AI_PARSE_DOCUMENT(TO_FILE(?, ?), PARSE_JSON(?)::OBJECT, TRUE)"
    assert params is not None
    assert params[0] == "@DB.SCHEMA.DOC_STAGE"
    assert params[1] == "docs/a.pdf"
    assert connection.cursor_instance.timeouts == [900]
    parse_options = json.loads(str(params[2]))
    assert parse_options == {
        "extract_images": True,
        "mode": "LAYOUT",
        "page_filter": [{"start": 0, "end": 3}],
        "page_split": True,
    }
    assert [page.page_number for page in document.pages] == [1, 2]
    assert document.pages[0].markdown == "# Page 1\nAlice"
    assert document.assets[0].id == "img-1"
    assert document.assets[0].page_number == 1
    assert document.assets[0].metadata["image_base64_present"] is True
    assert "image_base64" not in document.assets[0].metadata
    assert document.provider_metadata["provider"] == "snowflake_cortex"


def test_snowflake_cortex_ocr_raises_returned_error() -> None:
    connection = FakeConnection([[{"error": {"message": "cannot parse"}}]])
    provider = SnowflakeCortexOcrProvider(CONFIG, connector_factory=lambda **_: connection)

    with pytest.raises(RuntimeError, match="AI_PARSE_DOCUMENT failed"):
        provider.parse(_input_file("@DOC_STAGE/docs/a.pdf"), OcrOptions())


def test_snowflake_cortex_ocr_omits_page_options_for_text_inputs() -> None:
    connection = FakeConnection([[{"value": {"content": "Plain text"}, "metadata": {}}]])
    provider = SnowflakeCortexOcrProvider(CONFIG, connector_factory=lambda **_: connection)

    document = provider.parse(
        _input_file(
            "@DB.SCHEMA.DOC_STAGE/docs/notes.txt",
            path=Path("docs/notes.txt"),
            mime_type="text/plain",
        ),
        OcrOptions(page_range="0-1", snowflake_parse_mode="OCR"),
    )

    _, params = connection.cursor_instance.executed[0]
    assert params is not None
    assert json.loads(str(params[2])) == {"mode": "OCR"}
    assert document.pages[0].raw_text == "Plain text"


def test_snowflake_cortex_ocr_redacts_every_inline_blob_field() -> None:
    """Provider payloads spell one inline image several ways; none may leak."""

    connection = FakeConnection(
        [
            [
                {
                    "value": {
                        "content": "Alice",
                        "images": [
                            {
                                "image_id": "img-1",
                                "bytes": "0011",
                                "content_bytes": "2233",
                                "image_base64": "4455",
                                "caption": "Figure 1",
                            }
                        ],
                    },
                    "metadata": {},
                }
            ]
        ]
    )
    provider = SnowflakeCortexOcrProvider(CONFIG, connector_factory=lambda **_: connection)

    document = provider.parse(_input_file("@DOC_STAGE/docs/a.pdf"), OcrOptions())

    metadata = document.assets[0].metadata
    assert metadata["caption"] == "Figure 1"
    for key in ("bytes", "content_bytes", "image_base64"):
        assert key not in metadata
        assert metadata[f"{key}_present"] is True


def test_snowflake_cortex_ocr_rejects_a_document_that_parsed_to_nothing() -> None:
    """An empty parse contributes nothing to the graph and is not a success."""

    connection = FakeConnection(
        [[{"value": {"pages": [{"index": 0, "content": "   "}]}, "metadata": {}}]]
    )
    provider = SnowflakeCortexOcrProvider(CONFIG, connector_factory=lambda **_: connection)

    with pytest.raises(RuntimeError, match="returned no text"):
        provider.parse(_input_file("@DOC_STAGE/docs/a.pdf"), OcrOptions())


def test_snowflake_cortex_ocr_rejects_an_unrecognized_response_shape() -> None:
    """A provider shape change must fail rather than become the document text.

    Serializing the envelope into a page would chunk, embed, and extract the
    response structure itself while the run reported success.
    """

    connection = FakeConnection([[{"value": {"document": {"blocks": ["Alice"]}}, "metadata": {}}]])
    provider = SnowflakeCortexOcrProvider(CONFIG, connector_factory=lambda **_: connection)

    with pytest.raises(RuntimeError, match="unrecognized response") as error:
        provider.parse(_input_file("@DOC_STAGE/docs/a.pdf"), OcrOptions())

    assert "document" in str(error.value)
    assert "Alice" not in str(error.value)


def _input_file(
    source_uri: str,
    *,
    path: Path = Path("a.pdf"),
    mime_type: str = "application/pdf",
) -> InputFile:
    return InputFile(
        id="file_1",
        path=path,
        source_uri=source_uri,
        checksum="checksum",
        mime_type=mime_type,
        size_bytes=10,
    )
