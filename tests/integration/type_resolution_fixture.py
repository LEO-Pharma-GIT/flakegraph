"""Three documents whose extraction held a relation for its entity types.

The first document's window read "Jordan" as a person, so "Acme is located in
Jordan" broke the rule that a place is where something is located, and was
held. Other documents read "Jordan" as a place: two of them in the full corpus,
one in the thin corpus. A second held relation, "Acme is located in Bob", has
no other reading anywhere and must stay rejected. Local and Spark finalization
read the same shards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kg_processor.application.rejected_records import rejected_record_key
from kg_processor.config.settings import Settings
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionObservations,
    RelationObservation,
)
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ids import sha256_hex
from kg_processor.domain.stages import ExtractedDocumentShard, PreparedDocumentShard

GRAPH_ID = "type-resolution-graph"
ONTOLOGY_PROFILE: dict[str, Any] = {
    "name": "places",
    "description": "Organizations, the places they are in, and people.",
    "mode": "closed",
    "entity_types": [
        {"name": "ORGANIZATION", "description": "A named organization."},
        {"name": "PLACE", "description": "A named place."},
        {"name": "PERSON", "description": "A named person."},
    ],
    "relation_types": [
        {
            "name": "LOCATED_IN",
            "description": "An organization is located in a place.",
            "source_types": ["ORGANIZATION"],
            "target_types": ["PLACE"],
        },
        {
            "name": "BORDERS",
            "description": "A place borders another place.",
            "source_types": ["PLACE"],
            "target_types": ["PLACE"],
        },
    ],
}
HELD_WINDOW = "window-acme"
# The quote the first document's window gives both held relations.
HELD_TEXT = "Acme is located in Jordan. Acme is located in Bob's town."


def settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Settings both engines finalize the fixture with, type resolution on."""

    return Settings.load(
        env={},
        overrides={
            "job": {"graph_id": GRAPH_ID},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash", "dimension": 8},
            "writer": {"provider": "local_artifacts", "output_path": str(tmp_path / "out")},
            "cache": {"provider": "none"},
            "ontology": {"profile": ONTOLOGY_PROFILE},
            "graph": {"relation_type_resolution": True, "entity_resolution_enabled": False},
            **overrides,
        },
    )


def shards(*, place_documents: int) -> list[ExtractedDocumentShard]:
    """The held document and ``place_documents`` documents reading Jordan as a place."""

    place_texts = ["Jordan borders Syria.", "Travellers crossed Jordan by road."]
    documents = [_held_document()]
    for index, text in enumerate(place_texts[:place_documents], start=1):
        mentions = [_mention(f"jordan-place-{index}", "Jordan", "PLACE", f"chunk-{index}", text)]
        relations: list[RelationObservation] = []
        if "Syria" in text:
            mentions.append(_mention("syria", "Syria", "PLACE", f"chunk-{index}", text))
            relations.append(
                _relation("borders", "jordan-place-1", "syria", "BORDERS", f"chunk-{index}", text)
            )
        documents.append(_shard(f"file-{index}", f"chunk-{index}", text, mentions, relations, []))
    return documents


def held_records() -> list[dict[str, str]]:
    """The two records the first document's window rejected for their entity types."""

    return [
        {
            "kind": "relation",
            "reason": "domain_or_range_violation",
            "name": "LOCATED_IN",
            "source": "Acme",
            "source_type": "ORGANIZATION",
            "target": target,
            "target_type": "PERSON",
            "quote": HELD_TEXT,
            "chunk_id": "chunk-0",
        }
        for target in ("Jordan", "Bob")
    ]


def _held_document() -> ExtractedDocumentShard:
    mentions = [
        _mention("acme", "Acme", "ORGANIZATION", "chunk-0", HELD_TEXT),
        _mention("jordan-person", "Jordan", "PERSON", "chunk-0", HELD_TEXT),
        _mention("bob", "Bob", "PERSON", "chunk-0", HELD_TEXT),
    ]
    held = [
        _relation(
            f"held-{record['target'].lower()}",
            "acme",
            "jordan-person" if record["target"] == "Jordan" else "bob",
            "LOCATED_IN",
            "chunk-0",
            HELD_TEXT,
        ).model_copy(
            update={
                "rejected_record_key": rejected_record_key(
                    HELD_WINDOW, "relation", "domain_or_range_violation", record
                )
            }
        )
        for record in held_records()
    ]
    return _shard("file-0", "chunk-0", HELD_TEXT, mentions, [], held)


def _shard(
    file_id: str,
    chunk_id: str,
    text: str,
    mentions: list[EntityMention],
    relations: list[RelationObservation],
    held: list[RelationObservation],
) -> ExtractedDocumentShard:
    prepared = PreparedDocumentShard(
        file_ids=[file_id],
        files_seen=1,
        documents_processed=1,
        document_rows=[
            {
                "id": f"document-{file_id}",
                "graph_id": GRAPH_ID,
                "file_id": file_id,
                "checksum": sha256_hex(text),
                "source_uri": f"{file_id}.txt",
                "mime_type": "text/plain",
                "size_bytes": len(text),
                "ocr_provider": "builtin_text",
            }
        ],
        chunks=[
            Chunk(
                id=chunk_id,
                graph_id=GRAPH_ID,
                file_id=file_id,
                document_id=file_id,
                page_number=1,
                chunk_index=0,
                content=text,
                start_offset=0,
                end_offset=len(text),
                token_count=len(text.split()),
                content_hash=sha256_hex(text),
            )
        ],
    )
    trace: list[dict[str, Any]] = []
    if held:
        trace.append(
            {
                "stage": "relation_extraction",
                "window_id": HELD_WINDOW,
                "document_id": file_id,
                "chunk_ids": [chunk_id],
                "input_records": len(held),
                "accepted_records": 0,
                "record_actions": {
                    "domain_or_range_violation": len(held),
                    "held_for_type_resolution": len(held),
                },
                "rejected_records": held_records(),
            }
        )
    return ExtractedDocumentShard(
        prepared=prepared,
        observations=ExtractionObservations(
            entities=mentions,
            relations=relations,
            held_relations=held,
            trace=trace,
            chunk_count=1,
            window_count=1,
        ),
    )


def _mention(
    mention_id: str, name: str, entity_type: str, chunk_id: str, text: str
) -> EntityMention:
    start = text.index(name)
    return EntityMention(
        id=mention_id,
        name=name,
        type=entity_type,
        description=f"{name} as the text names it.",
        source_chunk_id=chunk_id,
        quote=name,
        start_offset=start,
        end_offset=start + len(name),
    )


def _relation(
    relation_id: str,
    source_id: str,
    target_id: str,
    relation_type: str,
    chunk_id: str,
    quote: str,
) -> RelationObservation:
    return RelationObservation(
        id=relation_id,
        source_entity_id=source_id,
        target_entity_id=target_id,
        relation_type=relation_type,
        description=quote,
        source_chunk_id=chunk_id,
        quote=quote,
        confidence=0.9,
        start_offset=0,
        end_offset=len(quote),
    )
