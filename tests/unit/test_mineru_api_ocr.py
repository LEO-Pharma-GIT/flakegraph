from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from http_fakes import StreamingClient, input_file
from pypdf import PdfWriter

from kg_processor.adapters.ocr import mineru_api
from kg_processor.adapters.ocr.mineru_api import MineruApiOcrProvider
from kg_processor.application.document_normalization import block_rows
from kg_processor.ports.ocr import OcrOptions


def test_mineru_api_ocr_posts_file_parse_and_normalizes_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "results": [
                {
                    "file_name": "sample.pdf",
                    "md_content": "# Title\nAlice works at Acme.",
                    "status": "done",
                }
            ]
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example", api_key="secret")

    document = provider.parse(
        input_file(input_path),
        OcrOptions(
            language="en",
            method="auto",
            backend="pipeline",
            formula=True,
            table=False,
            page_range="1-2",
        ),
    )

    request = client.requests[0]
    assert request["url"] == "https://mineru.example/file_parse"
    assert request["headers"] == {"Authorization": "Bearer secret"}
    assert request["data"] == {
        "backend": "pipeline",
        "formula_enable": "true",
        "end_page_id": "2",
        "lang_list": "en",
        "parse_method": "auto",
        "return_content_list": "true",
        "return_images": "false",
        "return_md": "true",
        "return_middle_json": "true",
        "start_page_id": "1",
        "table_enable": "false",
    }
    assert document.pages[0].markdown == "# Title\nAlice works at Acme."
    assert document.provider_metadata["provider"] == "mineru_api"


def test_mineru_api_ocr_uses_middle_json_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "data": {
                "middle_json": {
                    "pages": [
                        {"page_number": 3, "text": "Page three", "language": "en"},
                    ]
                }
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions())

    assert document.pages[0].page_number == 3
    assert document.pages[0].raw_text == "Page three"
    assert document.pages[0].detected_language == "en"


def test_mineru_api_ocr_accepts_results_dict_keyed_by_file_stem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "results": {
                "sample": {
                    "md_content": "# Sample\nFilename-keyed response.",
                    "status": "done",
                }
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions())

    assert document.pages[0].markdown == "# Sample\nFilename-keyed response."


def test_mineru_api_ocr_ignores_invalid_middle_json_when_markdown_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "data": {
                "md_content": "Fallback markdown",
                "middle_json": "{not-json",
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions())

    assert document.pages[0].markdown == "Fallback markdown"


def test_mineru_api_ocr_preserves_zero_page_and_accepts_page_idx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "data": {
                "pages": [
                    {"page_number": 0, "text": "Zero page"},
                    {"page": "not-a-number", "page_idx": "1", "text": "Second page"},
                ]
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions())

    assert [page.page_number for page in document.pages] == [0, 2]


def test_mineru_api_ocr_normalizes_assets_from_result_and_middle_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "data": {
                "md_content": "Document with assets",
                "images": [
                    {
                        "image_id": "top-image",
                        "type": "image",
                        "page_number": 1,
                        "img_path": "images/top.png",
                        "caption": "Top image",
                    }
                ],
                "middle_json": {
                    "pages": [
                        {
                            "page_number": 2,
                            "text": "Page two",
                            "figures": [
                                {
                                    "type": "figure",
                                    "url": "https://mineru.example/figures/2.png",
                                    "data": "base64-payload",
                                }
                            ],
                        }
                    ]
                },
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions())

    assert [asset.id for asset in document.assets] == [
        "top-image",
        document.assets[1].id,
    ]
    assert document.assets[0].kind == "image"
    assert document.assets[0].page_number == 1
    assert document.assets[0].uri == "images/top.png"
    assert document.assets[0].metadata["caption"] == "Top image"
    assert document.assets[1].kind == "figure"
    assert document.assets[1].page_number == 2
    assert document.assets[1].uri == "https://mineru.example/figures/2.png"
    assert document.assets[1].metadata["data_present"] is True
    assert document.assets[1].metadata["data_length"] == len("base64-payload")
    assert "data" not in document.assets[1].metadata


def test_mineru_api_ocr_deduplicates_assets_from_overlapping_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "result": {
                "md_content": "Document with duplicated image",
                "images": [
                    {"id": "image-1", "kind": "image", "path": "images/1.png"},
                ],
                "pages": [
                    {
                        "page_number": 1,
                        "text": "Page one",
                        "images": [
                            {"id": "image-1", "kind": "image", "path": "images/1.png"},
                        ],
                    }
                ],
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions())

    assert [asset.id for asset in document.assets] == ["image-1"]


def test_mineru_api_ocr_raises_failed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient({"status": "failed", "error": "bad file"})
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    with pytest.raises(RuntimeError, match="MinerU API failed"):
        provider.parse(input_file(input_path), OcrOptions())


def test_mineru_api_ocr_uploads_and_reads_the_body_as_one_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upload handle must stay open for the streamed request and response."""

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient({"data": {"md_content": "Streamed markdown"}})
    monkeypatch.setattr(httpx, "Client", client.open)

    document = MineruApiOcrProvider("https://mineru.example").parse(
        input_file(input_path), OcrOptions()
    )

    request = client.requests[0]
    assert request["method"] == "POST"
    assert request["file_handle_open"] is True
    assert document.pages[0].markdown == "Streamed markdown"


def test_mineru_api_ocr_raises_for_a_rejected_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected upload fails with the parser's own reason, not a status line."""

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient({"detail": "unsupported media type"}, status_code=415)
    monkeypatch.setattr(httpx, "Client", client.open)

    with pytest.raises(httpx.HTTPStatusError, match="HTTP 415: .*unsupported media type"):
        MineruApiOcrProvider("https://mineru.example").parse(input_file(input_path), OcrOptions())


def test_mineru_api_ocr_rejects_a_declared_response_above_the_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized body is refused before it is read."""

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {"data": {"md_content": "Small body"}}, response_headers={"content-length": "10000"}
    )
    monkeypatch.setattr(httpx, "Client", client.open)
    monkeypatch.setattr(mineru_api, "_MAX_RESPONSE_BYTES", 64)

    provider = MineruApiOcrProvider("https://mineru.example")

    with pytest.raises(RuntimeError, match="declared"):
        provider.parse(input_file(input_path), OcrOptions())


def test_mineru_api_ocr_tolerates_non_dict_content_list_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A content list is provider data: a stray scalar must not abort the parse."""

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "data": {
                "content_list": [
                    "loose paragraph",
                    {"page_idx": 1, "text": "Second page paragraph"},
                ]
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    document = MineruApiOcrProvider("https://mineru.example").parse(
        input_file(input_path), OcrOptions()
    )

    assert [page.page_number for page in document.pages] == [1, 2]
    assert document.pages[0].raw_text == "loose paragraph"
    assert document.pages[1].raw_text == "Second page paragraph"


def test_mineru_api_ocr_reads_pages_and_blocks_from_the_content_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The content list keeps each item's page and kind; the markdown keeps neither.

    Read from the flat markdown, a thirty-page scan cited page 1 for every
    quote. Furniture MinerU leaves out of its own markdown stays out here.
    """

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    content_list = [
        {"type": "text", "text": "Long Short-Term Memory", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Recurrent networks forget.", "page_idx": 0},
        {"type": "page_number", "text": "1", "page_idx": 0},
        {"type": "aside_text", "text": "ar 11[0 ::015", "page_idx": 0},
        {"type": "equation", "text": "$$\\alpha$$", "text_format": "latex", "page_idx": 1},
        {
            "type": "table",
            "table_caption": ["Table 1: Error rates"],
            "table_body": "<table><tr><td>0.1</td></tr></table>",
            "table_footnote": [],
            "img_path": "images/t.jpg",
            "page_idx": 1,
        },
    ]
    client = StreamingClient(
        {"data": {"md_content": "# flattened", "content_list": json.dumps(content_list)}}
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    document = MineruApiOcrProvider("https://mineru.example").parse(
        input_file(input_path), OcrOptions()
    )

    assert [page.page_number for page in document.pages] == [1, 2]
    assert document.pages[0].markdown == "# Long Short-Term Memory\n\nRecurrent networks forget."
    assert [block.kind for block in document.pages[0].blocks] == ["text", "text"]
    assert [block.kind for block in document.pages[1].blocks] == ["equation", "table"]
    assert document.pages[1].blocks[1].text == (
        "Table 1: Error rates\n<table><tr><td>0.1</td></tr></table>"
    )
    assert document.pages[1].blocks[1].metadata["img_path"] == "images/t.jpg"
    assert "ar 11" not in document.pages[0].markdown


def _two_blank_pages(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.add_blank_page(width=595, height=842)
    with path.open("wb") as handle:
        writer.write(handle)


def _empty_parse() -> dict[str, object]:
    return {"results": {"doc": {"md_content": "", "content_list": [], "middle_json": ""}}}


def _content(text: str) -> dict[str, object]:
    return {
        "results": {
            "doc": {
                "md_content": text,
                "content_list": [{"type": "text", "text": text, "page_idx": 0}],
            }
        }
    }


def test_mineru_api_ocr_reads_a_page_scanned_on_its_side_by_turning_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A landscape form scanned in portrait reads nothing until it is turned.

    The first pass reads page 1 and nothing on page 2. Page 2 is resent turned
    a quarter each way: 90 degrees reads a little (upside-down garbage reads
    as something), 270 reads the form, so 270 wins and the half turn is never
    tried. Page 1 is never resent.
    """

    input_path = tmp_path / "release-report.pdf"
    _two_blank_pages(input_path)
    first_pass: dict[str, object] = {
        "results": {
            "doc": {
                "md_content": "Page one",
                "content_list": [{"type": "text", "text": "Page one", "page_idx": 0}],
            }
        }
    }
    client = StreamingClient(
        payloads=[first_pass, _content("dxte 15--028 Ma"), _content("Product: POLYSORBATE 80")]
    )
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions(page_range="1-2"))

    assert [page.page_number for page in document.pages] == [1, 2]
    assert document.pages[1].raw_text == "Product: POLYSORBATE 80"
    assert document.pages[1].blocks[0].page_number == 2
    assert document.pages[1].blocks[0].metadata["rotation_degrees"] == 270
    assert len(client.requests) == 3
    turned = [request["files"]["files"][0] for request in client.requests[1:]]
    assert turned == ["page-2-90.pdf", "page-2-270.pdf"]
    # The caller's window applied to the document, not to a one-page copy.
    assert "start_page_id" in client.requests[0]["data"]
    assert "start_page_id" not in client.requests[1]["data"]


def test_mineru_api_ocr_tries_upside_down_only_when_no_quarter_turn_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "upside-down.pdf"
    _two_blank_pages(input_path)
    client = StreamingClient(
        payloads=[
            _empty_parse(),
            _empty_parse(),
            _empty_parse(),
            _content("page one turned"),
            _empty_parse(),
            _empty_parse(),
            _empty_parse(),
        ]
    )
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions())

    assert [page.page_number for page in document.pages] == [1]
    assert document.pages[0].blocks[0].metadata["rotation_degrees"] == 180
    names = [request["files"]["files"][0] for request in client.requests]
    assert names == [
        "upside-down.pdf",
        "page-1-90.pdf",
        "page-1-270.pdf",
        "page-1-180.pdf",
        "page-2-90.pdf",
        "page-2-270.pdf",
        "page-2-180.pdf",
    ]


def test_mineru_api_ocr_still_rejects_a_document_that_reads_nothing_any_way_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "blank.pdf"
    _two_blank_pages(input_path)
    client = StreamingClient(payloads=[_empty_parse()])
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = MineruApiOcrProvider("https://mineru.example")

    with pytest.raises(RuntimeError, match="returned no text"):
        provider.parse(input_file(input_path), OcrOptions())
    assert len(client.requests) == 7


@pytest.mark.parametrize(
    "options,selected",
    [
        (OcrOptions(page_range="1-2"), [2, 3]),
        (OcrOptions(page_range="2-"), [3, 4]),
        (OcrOptions(page_range="-1"), [1, 2]),
        (OcrOptions(page_range="0-3", start_page_id=1, end_page_id=2), [2, 3]),
    ],
)
def test_rotation_recovery_only_uploads_unread_pages_in_the_requested_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: OcrOptions, selected: list[int]
) -> None:
    """A window-honoring service can leave a selected page unread without widening recovery."""

    source = tmp_path / "windowed.pdf"
    pdf = PdfWriter()
    for _ in range(4):
        pdf.add_blank_page(width=100, height=100)
    pdf.write(source)
    first_pass: dict[str, object] = {
        "pages": [{"page_number": selected[0], "text": "Readable selected page"}]
    }
    client = StreamingClient(payloads=[first_pass, _content("Recovered selected page")])
    monkeypatch.setattr(httpx, "Client", client.open)

    parsed = MineruApiOcrProvider("https://mineru.invalid").parse(input_file(source), options)

    assert [page.page_number for page in parsed.pages] == selected
    assert [request["files"]["files"][0] for request in client.requests[1:]] == [
        f"page-{selected[1]}-90.pdf",
        f"page-{selected[1]}-270.pdf",
    ]
    assert all(
        "start_page_id" not in request["data"] and "end_page_id" not in request["data"]
        for request in client.requests[1:]
    )


def test_recovered_block_ids_remain_file_scoped_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical recovered text in distinct source files retains distinct canonical block rows."""

    first_pass: dict[str, object] = {"pages": [{"page_number": 1, "text": "Page one"}]}
    recovered = _content("Identical recovered text")
    client = StreamingClient(payloads=[first_pass, recovered, recovered] * 3)
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = MineruApiOcrProvider("https://mineru.invalid")
    rows = []
    for identity in ("first", "second", "first"):
        source = tmp_path / f"{identity}.pdf"
        _two_blank_pages(source)
        file = input_file(source).model_copy(update={"id": identity, "checksum": identity})
        parsed = provider.parse(file, OcrOptions())
        rows.append(
            next(row for row in block_rows("shared-graph", file, parsed) if row["page_number"] == 2)
        )

    assert rows[0]["id"] != rows[1]["id"]
    assert rows[0]["file_id"] != rows[1]["file_id"]
    assert rows[0]["id"] == rows[2]["id"]
