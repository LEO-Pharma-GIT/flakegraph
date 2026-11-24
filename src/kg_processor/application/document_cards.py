# SPDX-License-Identifier: Apache-2.0
"""A catalogue card for every document: what it is, what it covers, what it finds.

A graph answers questions about the things documents mention, but many
questions are about the documents themselves: which ones cover a market, how
many surveyed a population, what each concluded. A card per document answers
those directly - title, type, subject, date, period, places, organizations and
the part each played, topics, samples, method and key findings - written once
from the document's own text. The card is kept twice on the document's row: as
its summary text, where search and the readers see it, and as a record of typed
fields, where a question can be answered by filtering and counting documents -
which ones a given agency prepared, how many date from a year, which surveyed a
group - rather than by reading every one.

The card is written from the whole document when it fits the bound, and from
every page's opening otherwise, so a long document's later sections still
reach it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, Literal

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
_MAX_SAMPLES = 12
_MAX_DATE_CHARACTERS = 10
DOCUMENT_CARD_MAX_OUTPUT_TOKENS = 2560

OrganizationRole = Literal["author", "commissioner", "subject", "source", "other"]
ORGANIZATION_ROLES: tuple[OrganizationRole, ...] = (
    "author",
    "commissioner",
    "subject",
    "source",
    "other",
)


class CardOrganization(BaseModel):
    """An organization the document names, and the part it plays in the document.

    author - it prepared, wrote or produced the document; commissioner - the
    document was made for it, or it paid for the work; subject - the document
    discusses or evaluates it; source - the document cites it, or a work of
    it, as where information came from; other - any other mention.
    """

    name: str
    role: OrganizationRole = "other"


class CardSample(BaseModel):
    """A group the document's work studied, surveyed or interviewed, and how many as stated."""

    group: str
    count: int | None = None


class DocumentCard(BaseModel):
    """The card one document's text supports."""

    title: str = ""
    document_type: str = ""
    about: str = ""
    # When the document itself was written, issued or presented: YYYY, YYYY-MM or YYYY-MM-DD.
    date: str = ""
    period: str = ""
    places: list[str] = Field(default_factory=list)
    organizations: list[CardOrganization] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    samples: list[CardSample] = Field(default_factory=list)
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

    def records(properties: dict[str, Any], count: int) -> dict[str, Any]:
        return {
            "type": "array",
            "maxItems": count,
            "items": {"type": "object", "properties": properties, "required": list(properties)},
        }

    return {
        "type": "object",
        "properties": {
            "title": text(_MAX_TITLE_CHARACTERS),
            "document_type": text(_MAX_ITEM_CHARACTERS),
            "about": text(_MAX_FIELD_CHARACTERS),
            "date": text(_MAX_DATE_CHARACTERS),
            "period": text(_MAX_ITEM_CHARACTERS),
            "places": items(_MAX_ITEM_CHARACTERS, _MAX_LIST_ITEMS),
            "organizations": records(
                {
                    "name": text(_MAX_ITEM_CHARACTERS),
                    "role": {"type": "string", "enum": list(ORGANIZATION_ROLES)},
                },
                _MAX_LIST_ITEMS,
            ),
            "topics": items(_MAX_ITEM_CHARACTERS, _MAX_TOPICS),
            "samples": records(
                {
                    "group": text(_MAX_ITEM_CHARACTERS),
                    "count": {"type": ["integer", "null"], "minimum": 0},
                },
                _MAX_SAMPLES,
            ),
            "method": text(_MAX_FIELD_CHARACTERS),
            "key_findings": items(_MAX_FINDING_CHARACTERS, _MAX_FINDINGS),
        },
    }


_ROLE_LABELS: tuple[tuple[OrganizationRole, str], ...] = (
    ("author", "Prepared by"),
    ("commissioner", "Prepared for"),
    ("subject", "Organizations discussed"),
    ("source", "Sources cited"),
    ("other", "Other organizations"),
)


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
    line("Date", card.date)
    line("Period", card.period)
    listed("Places", card.places)
    for role, label in _ROLE_LABELS:
        listed(label, [item.name for item in card.organizations if item.role == role])
    listed("Topics", card.topics)
    listed(
        "Sample",
        [
            f"{item.group} ({item.count})" if item.count is not None else item.group
            for item in card.samples
        ],
    )
    line("Method", card.method)
    findings = [" ".join(item.split()) for item in card.key_findings if item.strip()]
    if findings:
        lines.append("Key findings:")
        lines.extend(f"- {finding}" for finding in findings)
    return "\n".join(lines)


def card_event(document_id: str, card: DocumentCard) -> dict[str, Any]:
    """The trace event that carries a document's card to the document's row."""

    record = card.model_dump(mode="json")
    return {
        "stage": DOCUMENT_CARD_STAGE,
        "document_id": document_id,
        "summary": render_card(card),
        "card": record,
        # The same record as JSON text, the form a trace schema and a table column keep.
        "card_json": json.dumps(record, ensure_ascii=False, sort_keys=True),
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


def document_card_records(trace: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Each document's card record recorded in a trace, as JSON text, by document id."""

    return {
        str(event["document_id"]): str(event["card_json"])
        for event in trace
        if event.get("stage") == DOCUMENT_CARD_STAGE
        and event.get("document_id")
        and event.get("card_json")
    }
