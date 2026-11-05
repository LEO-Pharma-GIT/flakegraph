"""Five documents filed in folders read at level 2, ready for finalization.

Two documents sit in folders named "Northwind Pricing Study", one below a
picked root and one below an archive, and the model names that folder two
ways; two sit in "Contract ACME-0042", one of them under a location written
with an empty folder; one sits directly in the picked root and so under no
folder at that level. Each states products and the organization that makes
them; one names an organization no relation reaches. Local and Spark
finalization read the same shards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kg_processor.application.chunking import chunk_document
from kg_processor.application.document_normalization import with_location_page
from kg_processor.application.folder_entities import folder_entity_mentions
from kg_processor.config.settings import Settings
from kg_processor.domain.documents import ParsedDocument, ParsedPage
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionObservations,
    RelationObservation,
)
from kg_processor.domain.ids import sha256_hex
from kg_processor.domain.ontology import OntologyProfile
from kg_processor.domain.stages import ExtractedDocumentShard, PreparedDocumentShard
from kg_processor.ports.llm import (
    LlmCapabilities,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)

GRAPH_ID = "folder-graph"
BASE_PROFILE: dict[str, Any] = {
    "name": "studies",
    "description": "Studies and contracts, and the products their documents state.",
    "mode": "closed",
    "entity_types": [
        {"name": "STUDY", "description": "A research study or contract a folder is filed under."},
        {"name": "PRODUCT", "description": "A product."},
        {"name": "ORGANIZATION", "description": "A company."},
    ],
    "relation_types": [
        {
            "name": "MADE_BY",
            "description": "A product is made or supplied by an organization.",
            "source_types": ["PRODUCT"],
            "target_types": ["ORGANIZATION"],
        }
    ],
}
CORPUS = (
    (
        "file-a",
        "Picked Root/Northwind Pricing Study/report.pdf",
        "Widget Pro is made by Acme Corp.",
    ),
    (
        "file-b",
        "Archive/Northwind Pricing Study/raw/prices.txt",
        "Widget Pro prices rose in May at Initech.",
    ),
    ("file-c", "Picked Root/Contract ACME-0042/terms.pdf", "Gadget Max is made by Globex Ltd."),
    ("file-d", "Picked Root/notes.txt", "Gadget Max is also sold online."),
    # A location written with an empty folder, as some sources list them.
    ("file-e", "Picked Root//Contract ACME-0042/annex.pdf", "Gadget Max ships with an annex."),
)
_NAMES = {
    "report.pdf": "Northwind Pricing Study",
    "prices.txt": "NW Pricing",
    "terms.pdf": "Contract ACME-0042",
}
_ENTITIES = (
    ("Widget Pro", "PRODUCT"),
    ("Gadget Max", "PRODUCT"),
    ("Acme Corp", "ORGANIZATION"),
    ("Globex Ltd", "ORGANIZATION"),
    ("Initech", "ORGANIZATION"),
)


def profile(**folder: Any) -> dict[str, Any]:
    """The ontology with a folder entity of type STUDY at level 2, as ``folder`` sets it."""

    return {**BASE_PROFILE, "folder_entity": {"entity_type": "STUDY", "level": 2, **folder}}


def settings(tmp_path: Path, ontology: dict[str, Any], **overrides: Any) -> Settings:
    """Settings both engines finalize the fixture with."""

    graph = {"entity_resolution_enabled": False, **overrides.pop("graph", {})}
    return Settings.load(
        env={},
        overrides={
            "job": {"graph_id": GRAPH_ID},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash", "dimension": 8},
            "writer": {"provider": "local_artifacts", "output_path": str(tmp_path / "out")},
            "cache": {"provider": "none"},
            "ontology": {"profile": ontology},
            "graph": graph,
            **overrides,
        },
    )


class _NamingLlm:
    """Name each folder as the model would, a little differently per document."""

    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(supports_seed=True, max_output_tokens=1024)

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        name = _NAMES.get(payload["file_name"], "")
        return StructuredCompletionResult(
            payload={"name": name, "description": f"{name} as its documents describe it."}
        )


def shards() -> list[ExtractedDocumentShard]:
    """Every document prepared, its folder mention made, and its text extracted.

    The report's model also reads the study's name off its location page, so
    the study has a text mention beside the folder mention in that chunk.
    """

    ontology = OntologyProfile.model_validate(profile())
    documents: list[ExtractedDocumentShard] = []
    for file_id, location, body in CORPUS:
        parsed = ParsedDocument(
            file_id=file_id,
            checksum=sha256_hex(body),
            source_uri=f"/uploads/run/{location}",
            mime_type="text/plain",
            pages=[ParsedPage(page_number=1, markdown=body, raw_text=body)],
        )
        chunks = chunk_document(with_location_page(parsed, location), GRAPH_ID, 200, 0)
        row = {
            "id": file_id,
            "graph_id": GRAPH_ID,
            "file_id": file_id,
            "checksum": parsed.checksum,
            "source_uri": parsed.source_uri,
            "mime_type": "text/plain",
            "size_bytes": len(body),
            "ocr_provider": "builtin_text",
            "folder": location.rpartition("/")[0],
        }
        folders, trace = folder_entity_mentions(
            chunks,
            {file_id: row},
            ontology,
            _NamingLlm(),
            model="fake",
            timeout_seconds=5,
            seed=7,
            parallelism=1,
        )
        text = chunks[1]
        mentions = [
            EntityMention(
                id=f"{file_id}-{name}",
                name=name,
                type=entity_type,
                description=f"{name} as {file_id} states it.",
                source_chunk_id=text.id,
                quote=name,
                start_offset=text.content.index(name),
                end_offset=text.content.index(name) + len(name),
            )
            for name, entity_type in _ENTITIES
            if name in text.content
        ]
        if file_id == "file-a":
            location_chunk = chunks[0]
            start = location_chunk.content.index("Northwind Pricing Study")
            mentions.append(
                EntityMention(
                    id="file-a-study",
                    name="Northwind Pricing Study",
                    type="STUDY",
                    description="The study.",
                    source_chunk_id=location_chunk.id,
                    quote="Northwind Pricing Study",
                    start_offset=start,
                    end_offset=start + len("Northwind Pricing Study"),
                )
            )
        relations = [
            RelationObservation(
                id=f"{file_id}-made-by",
                source_entity_id=mentions[0].id,
                target_entity_id=mentions[1].id,
                relation_type="MADE_BY",
                description=text.content,
                source_chunk_id=text.id,
                quote=text.content,
                start_offset=0,
                end_offset=len(text.content),
            )
            for _ in [0]
            if "made by" in text.content
        ]
        documents.append(
            ExtractedDocumentShard(
                prepared=PreparedDocumentShard(
                    file_ids=[file_id],
                    files_seen=1,
                    documents_processed=1,
                    document_rows=[row],
                    chunks=chunks,
                    document_context_entities=folders,
                    trace=trace,
                ),
                observations=ExtractionObservations(
                    entities=[*folders, *mentions],
                    relations=relations,
                    chunk_count=len(chunks),
                    window_count=1,
                ),
            )
        )
    return documents
