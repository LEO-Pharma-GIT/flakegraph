"""Each document gets a catalogue card, written from its text and kept as its summary."""

from __future__ import annotations

import json

from kg_processor.application.content_kinds import annotate_documents
from kg_processor.application.document_cards import (
    DOCUMENT_CARD_STAGE,
    DocumentCard,
    card_event,
    card_outline,
    document_card_request,
    document_pages,
    render_card,
)
from kg_processor.domain.graph import Chunk


def _chunk(page: int, index: int, content: str, start: int = 0) -> Chunk:
    return Chunk(
        id=f"chunk_{page}_{index}",
        file_id="file_1",
        page_number=page,
        chunk_index=index,
        content=content,
        start_offset=start,
        end_offset=start + len(content),
        token_count=len(content.split()),
        content_hash=f"hash_{page}_{index}",
    )


def test_overlapping_chunks_of_a_page_are_read_once() -> None:
    page = "Alpha beta gamma delta epsilon"
    chunks = [_chunk(1, 0, page[:16]), _chunk(1, 1, page[11:], start=11)]

    assert document_pages(chunks) == [(1, page)]


def test_a_short_document_is_shown_whole_and_a_long_one_by_page_openings() -> None:
    short = [_chunk(1, 0, "Short page one."), _chunk(2, 1, "Short page two.")]
    assert card_outline(short, 1000) == [
        {"page": 1, "text": "Short page one."},
        {"page": 2, "text": "Short page two."},
    ]

    long_pages = [_chunk(page, page, ("word " * 400).strip()) for page in range(1, 11)]
    outline = card_outline(long_pages, 2000)
    assert [item["page"] for item in outline] == list(range(1, 11))
    assert all(len(item["text"]) <= 202 for item in outline)
    assert outline[0]["text"].endswith(" …")


def test_the_request_names_the_document_and_carries_its_pages() -> None:
    request = document_card_request(
        [_chunk(1, 0, "Annual review of orchard yields.")],
        file_name="review.pdf",
        location="farms/review.pdf",
        max_characters=1000,
        model="model",
        timeout_seconds=30,
        seed=1,
    )

    payload = json.loads(request.user)
    assert payload["file_name"] == "review.pdf"
    assert payload["text"] == [{"page": 1, "text": "Annual review of orchard yields."}]
    assert request.task_name == "document_card"
    assert set(request.json_schema["properties"]) >= {"title", "key_findings", "method"}


def test_a_card_renders_as_labelled_lines_and_skips_empty_fields() -> None:
    card = DocumentCard(
        title="Orchard yields 2025",
        document_type="report",
        about="Yields of three orchards over one season.",
        places=["North valley", "  "],
        key_findings=["Yield rose 12% on irrigated plots.", ""],
    )

    assert render_card(card) == (
        "Title: Orchard yields 2025\n"
        "Type: report\n"
        "About: Yields of three orchards over one season.\n"
        "Places: North valley\n"
        "Key findings:\n"
        "- Yield rose 12% on irrigated plots."
    )


def test_a_document_row_takes_its_card_and_a_dataset_keeps_its_profile() -> None:
    rows = [
        {"file_id": "doc", "kind": "document", "source_uri": "a/report.pdf"},
        {"file_id": "sheet", "kind": "dataset", "source_uri": "a/sheet.csv"},
    ]
    trace = [
        card_event("doc", DocumentCard(title="Report", about="About things.")),
        {"stage": "dataset_profile", "document_id": "sheet", "kind": "other", "summary": "Rows."},
        {"stage": DOCUMENT_CARD_STAGE, "document_id": "sheet", "summary": "ignored"},
    ]

    annotated = {row["file_id"]: row for row in annotate_documents(rows, trace)}
    assert annotated["doc"]["summary"] == "Title: Report\nAbout: About things."
    assert annotated["sheet"]["summary"] == "Rows."
