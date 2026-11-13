# SPDX-License-Identifier: Apache-2.0
"""A catalogue card for every document: what it is, what it covers, what it finds.

A graph answers questions about the things documents mention, but many
questions are about the documents themselves: which ones cover a market, how
many surveyed a population, what each concluded. A card per document answers
those directly - title, type, subject, period, places, organizations, topics,
method and key findings - written once from the document's own text and kept
as the document's summary, where search and question answering can read it.

The card is written from the whole document when it fits the bound, and from
every page's opening otherwise, so a long document's later sections still
reach it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from kg_processor.application.prompt_registry import get_prompt_template
from kg_processor.domain.graph import Chunk
from kg_processor.ports.llm import StructuredCompletionRequest

DOCUMENT_CARD_STAGE = "document_card"
DOCUMENT_CARD_TASK = "document_card"
# A page shows at least this much of its opening, however long the document.
_MIN_PAGE_CHARACTERS = 160
_MAX_TITLE_CHARACTERS = 300
_MAX_FIELD_CHARACTERS = 600
_MAX_ITEM_CHARACTERS = 200
_MAX_FINDING_CHARACTERS = 400
_MAX_LIST_ITEMS = 12
_MAX_TOPICS = 10
_MAX_FINDINGS = 12
DOCUMENT_CARD_MAX_OUTPUT_TOKENS = 2048


class DocumentCard(BaseModel):
    """The card one document's text supports."""

    title: str = ""
    document_type: str = ""
    about: str = ""
    period: str = ""
    places: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    method: str = ""
    key_findings: list[str] = Field(default_factory=list)


def document_pages(chunks: Iterable[Chunk]) -> list[tuple[int, str]]:
    """One document's page texts in order, each overlap between chunks read once."""

    pages: dict[int, list[Chunk]] = {}
    for chunk in chunks:
        pages.setdefault(chunk.page_number, []).append(chunk)
    texts: list[tuple[int, str]] = []
    for page_number in sorted(pages):
        parts: list[str] = []
        covered = -1
        for chunk in sorted(pages[page_number], key=lambda item: item.start_offset):
            if chunk.end_offset <= covered:
                continue
            skip = max(0, covered - chunk.start_offset)
            parts.append(chunk.content[skip:])
            covered = chunk.end_offset
        text = "".join(parts).strip()
        if text:
            texts.append((page_number, text))
    return texts


def card_outline(chunks: Iterable[Chunk], max_characters: int) -> list[dict[str, Any]]:
    """The document as its card is written from: whole pages, or each page's opening.

    A document that fits the bound is shown whole. A longer one gives every
    page an equal share, at least a short opening each, so the card sees
    where the document goes as well as where it starts.
    """

    pages = document_pages(chunks)
    total = sum(len(text) for _page, text in pages)
    if total <= max_characters or not pages:
        return [{"page": page, "text": text} for page, text in pages]
    share = max(_MIN_PAGE_CHARACTERS, max_characters // len(pages))
    return [{"page": page, "text": _opening(text, share)} for page, text in pages]


def _opening(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > limit // 2 else limit].rstrip() + " …"


def document_card_request(  # noqa: PLR0913 - one document's card context.
    chunks: list[Chunk],
    *,
    file_name: str,
    location: str,
    max_characters: int,
    model: str,
    timeout_seconds: int,
    seed: int,
) -> StructuredCompletionRequest:
    """Build the request that writes one document's card."""

    template = get_prompt_template(DOCUMENT_CARD_TASK)
    outline = card_outline(chunks, max_characters)
    payload = {
        "file_name": file_name,
        "location": location,
        "pages": len({chunk.page_number for chunk in chunks}),
        "text": outline,
    }
    return StructuredCompletionRequest(
        task_name=DOCUMENT_CARD_TASK,
        model=model,
        system=template.text,
        user=json.dumps(payload, ensure_ascii=False),
        json_schema=document_card_schema(),
        timeout_seconds=timeout_seconds,
        max_tokens=DOCUMENT_CARD_MAX_OUTPUT_TOKENS,
        seed=seed,
        prompt_metadata={
            "prompt_name": template.name,
            "prompt_revision": template.revision,
            "prompt_sha256": template.checksum,
        },
    )


def document_card_schema() -> dict[str, Any]:
    """The strict response contract of one card."""

    def text(limit: int) -> dict[str, Any]:
        return {"type": "string", "maxLength": limit}

    def items(limit: int, count: int) -> dict[str, Any]:
        return {"type": "array", "maxItems": count, "items": text(limit)}

    return {
        "type": "object",
        "properties": {
            "title": text(_MAX_TITLE_CHARACTERS),
            "document_type": text(_MAX_ITEM_CHARACTERS),
            "about": text(_MAX_FIELD_CHARACTERS),
            "period": text(_MAX_ITEM_CHARACTERS),
            "places": items(_MAX_ITEM_CHARACTERS, _MAX_LIST_ITEMS),
            "organizations": items(_MAX_ITEM_CHARACTERS, _MAX_LIST_ITEMS),
            "topics": items(_MAX_ITEM_CHARACTERS, _MAX_TOPICS),
            "method": text(_MAX_FIELD_CHARACTERS),
            "key_findings": items(_MAX_FINDING_CHARACTERS, _MAX_FINDINGS),
        },
    }


def render_card(card: DocumentCard) -> str:
    """A card as the document's summary text, one labelled line per field."""

    lines: list[str] = []

    def line(label: str, value: str) -> None:
        value = " ".join(value.split())
        if value:
            lines.append(f"{label}: {value}")

    def listed(label: str, values: list[str]) -> None:
        line(label, "; ".join(" ".join(value.split()) for value in values if value.strip()))

    line("Title", card.title)
    line("Type", card.document_type)
    line("About", card.about)
    line("Period", card.period)
    listed("Places", card.places)
    listed("Organizations", card.organizations)
    listed("Topics", card.topics)
    line("Method", card.method)
    findings = [" ".join(item.split()) for item in card.key_findings if item.strip()]
    if findings:
        lines.append("Key findings:")
        lines.extend(f"- {finding}" for finding in findings)
    return "\n".join(lines)


def card_event(document_id: str, card: DocumentCard) -> dict[str, Any]:
    """The trace event that carries a document's card to the document's row."""

    return {
        "stage": DOCUMENT_CARD_STAGE,
        "document_id": document_id,
        "summary": render_card(card),
        "card": card.model_dump(mode="json"),
    }


def document_card_summaries(trace: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Each document's card summary recorded in a trace, by document id."""

    return {
        str(event["document_id"]): str(event["summary"])
        for event in trace
        if event.get("stage") == DOCUMENT_CARD_STAGE
        and event.get("document_id")
        and event.get("summary")
    }
