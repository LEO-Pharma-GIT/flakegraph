"""Pictures are chosen, read by a model, and written onto their pages as text."""

from __future__ import annotations

import base64
import io
from collections.abc import Mapping
from typing import Any

import pytest
from PIL import Image

from kg_processor.application.figure_reading import (
    FigurePolicy,
    figure_candidates,
    read_figures,
)
from kg_processor.domain.documents import ParsedAsset, ParsedDocument, ParsedPage
from kg_processor.domain.tables import ChartData
from kg_processor.ports.llm import (
    LlmCapabilities,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _asset(asset_id: str, page: int, digest: str, **metadata: object) -> ParsedAsset:
    return ParsedAsset(
        id=asset_id,
        kind="image",
        page_number=page,
        metadata={
            "content_type": "image/png",
            "size_bytes": 50_000,
            "width_px": 800,
            "height_px": 600,
            "sha256": digest,
            **metadata,
        },
    )


def _document(assets: list[ParsedAsset], pages: int = 3) -> ParsedDocument:
    return ParsedDocument(
        file_id="file_1",
        checksum="sha",
        source_uri="file:///deck.pptx",
        mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        pages=[
            ParsedPage(page_number=page, markdown=f"# Slide {page}\nBody", raw_text="Body")
            for page in range(1, pages + 1)
        ],
        assets=assets,
    )


def _policy(**overrides: Any) -> FigurePolicy:
    values: dict[str, Any] = {
        "model": "vision-model",
        "timeout_seconds": 30,
        "max_output_tokens": 1024,
        "min_side_px": 96,
        "min_bytes": 2048,
        "max_bytes": 10_000_000,
        "repeat_limit": 3,
        "max_per_document": 50,
        "max_side_px": 512,
        "seed": 7,
    }
    values.update(overrides)
    return FigurePolicy(**values)


class _Reader:
    """A model that answers each picture with a reading chosen by the test."""

    def __init__(self, replies: Mapping[str, dict[str, Any]] | None = None) -> None:
        self.replies = dict(replies or {})
        self.requests: list[StructuredCompletionRequest] = []

    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(supports_images=True)

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        self.requests.append(request)
        reply = self.replies.get("next") or {
            "kind": "chart",
            "description": "Bars of sales by region.",
            "text": "Sales by region\nNorth 40%",
            "charts": [
                {
                    "chart_type": "bar",
                    "title": "Sales by region",
                    "category_axis_title": "",
                    "value_axis_title": "Share %",
                    "categories": ["North", "South"],
                    "series": [{"name": "2025", "values": [40, None]}],
                }
            ],
            "tables": [],
        }
        if reply.get("raise"):
            raise RuntimeError("model unavailable")
        return StructuredCompletionResult(payload=reply)


def test_small_vector_repeated_and_duplicate_pictures_are_not_read() -> None:
    document = _document(
        [
            _asset("chart", 1, "a"),
            _asset("icon", 1, "b", width_px=32, height_px=32),
            _asset("vector", 2, "c", content_type="image/x-emf"),
            _asset("logo1", 1, "d"),
            _asset("logo2", 2, "d"),
            _asset("logo3", 3, "d"),
            _asset("again", 3, "a"),
            _asset("tiny_file", 2, "e", size_bytes=100),
        ]
    )
    chosen, skipped = figure_candidates(document, _policy())

    assert [asset.id for asset in chosen] == ["chart"]
    assert skipped == {"too_small": 2, "unreadable_format": 1, "repeated": 3, "duplicate": 1}


def test_a_reading_is_appended_to_its_page_with_exact_block_offsets() -> None:
    document = _document([_asset("chart", 2, "a")])
    reader = _Reader()
    updated, report = read_figures(
        document, llm=reader, policy=_policy(), load_image=lambda _asset: _png(1200, 800)
    )

    page = updated.pages[1]
    assert page.markdown.startswith("# Slide 2\nBody\n\nFigure (chart): Bars of sales by region.")
    for block in page.blocks:
        start, end = block.metadata["start_offset"], block.metadata["end_offset"]
        assert page.markdown[start:end] == block.text
    chart_block = next(block for block in page.blocks if block.kind == "chart")
    chart = ChartData.model_validate(chart_block.metadata["chart"])
    assert chart.estimated and chart.source == "vision"
    assert "values estimated from an image" in chart_block.text
    assert "| North | 40 |" in chart_block.text
    assert updated.pages[0].markdown == document.pages[0].markdown
    assert report.read == 1 and report.kinds == {"chart": 1}
    assert updated.provider_metadata["figure_reading"]["read"] == 1


def test_pictures_are_scaled_down_before_they_are_sent() -> None:
    reader = _Reader()
    read_figures(
        _document([_asset("chart", 1, "a")]),
        llm=reader,
        policy=_policy(max_side_px=300),
        load_image=lambda _asset: _png(1500, 900),
    )

    (image,) = reader.requests[0].images
    with Image.open(io.BytesIO(base64.b64decode(image.data_base64))) as sent:
        assert max(sent.size) == 300
    assert reader.requests[0].model == "vision-model"


def test_a_decoration_adds_no_text_and_a_failure_does_not_fail_the_document() -> None:
    decoration = _Reader(
        {"next": {"kind": "decoration", "description": "", "text": "", "charts": [], "tables": []}}
    )
    updated, report = read_figures(
        _document([_asset("logo", 1, "a")]),
        llm=decoration,
        policy=_policy(),
        load_image=lambda _asset: _png(400, 400),
    )
    assert updated.pages[0].markdown == "# Slide 1\nBody"
    assert report.kinds == {"decoration": 1}

    failing = _Reader({"next": {"raise": True}})
    updated, report = read_figures(
        _document([_asset("chart", 1, "a")]),
        llm=failing,
        policy=_policy(),
        load_image=lambda _asset: _png(400, 400),
    )
    assert report.failed == 1 and report.read == 0
    assert updated.pages[0].markdown == "# Slide 1\nBody"


def test_a_cached_reading_is_used_without_calling_the_model() -> None:
    store: dict[str, Mapping[str, Any]] = {}
    first = _Reader()
    read_figures(
        _document([_asset("chart", 1, "same")]),
        llm=first,
        policy=_policy(),
        load_image=lambda _asset: _png(400, 400),
        cached_reading=store.get,
        store_reading=store.__setitem__,
    )
    second = _Reader()
    updated, report = read_figures(
        _document([_asset("chart", 2, "same")]),
        llm=second,
        policy=_policy(),
        load_image=lambda _asset: pytest.fail("a cached picture is not loaded"),
        cached_reading=store.get,
        store_reading=store.__setitem__,
    )

    assert second.requests == []
    assert report.cached == 1
    assert "Figure (chart)" in updated.pages[1].markdown


def test_a_table_read_from_a_picture_is_a_table_block() -> None:
    reader = _Reader(
        {
            "next": {
                "kind": "table",
                "description": "A price list.",
                "text": "",
                "charts": [],
                "tables": [
                    {
                        "caption": "Prices",
                        "header_rows": 1,
                        "rows": [["Item", "Price"], ["Widget", "3.50"]],
                    }
                ],
            }
        }
    )
    updated, _report = read_figures(
        _document([_asset("scan", 1, "a")]),
        llm=reader,
        policy=_policy(),
        load_image=lambda _asset: _png(400, 400),
    )

    table = next(block for block in updated.pages[0].blocks if block.kind == "table")
    assert table.metadata["table"]["source"] == "vision"
    assert "| Widget | 3.50 |" in table.text


def test_a_large_picture_that_repeats_is_read_once_on_its_first_page() -> None:
    """A page-sized picture shown again is content, not template."""

    document = _document(
        [
            _asset("form1", 1, "f", width_px=1920, height_px=1080),
            _asset("form2", 2, "f", width_px=1920, height_px=1080),
            _asset("form3", 3, "f", width_px=1920, height_px=1080),
        ]
    )
    chosen, skipped = figure_candidates(document, _policy())

    assert [asset.id for asset in chosen] == ["form1"]
    assert skipped == {"duplicate": 2}
