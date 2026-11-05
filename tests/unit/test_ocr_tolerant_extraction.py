"""Records whose names OCR ran together or misread are kept and labelled ocr_tolerant."""

from __future__ import annotations

import json
from typing import Any

from documents import chunk, window

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.llm_extractors import LlmEntityExtractor, LlmRelationExtractor
from kg_processor.application.two_pass_extraction import _relation_inventory_for_window
from kg_processor.domain.extraction import EntityMention, ExtractionWindow
from kg_processor.domain.ontology import OntologyProfile
from kg_processor.ports.llm import StructuredCompletionRequest, StructuredCompletionResult


class _PayloadLlm(FakeLlmProvider):
    """Answer every extraction call with one fixed list of records."""

    def __init__(self, key: str, records: list[dict[str, object]]) -> None:
        super().__init__()
        self.key = key
        self.records = records

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        return StructuredCompletionResult(payload={self.key: self.records})


def _profile() -> OntologyProfile:
    return OntologyProfile.model_validate(
        {
            "name": "suppliers",
            "description": "Products and the organizations that supply them",
            "entity_types": [
                {
                    "name": "PRODUCT",
                    "description": "A product a document describes",
                    "document_context": True,
                    "document_subject": True,
                },
                {"name": "ORGANIZATION", "description": "A company or institution"},
            ],
            "relation_types": [
                {
                    "name": "SUPPLIES",
                    "description": "An organization supplies another",
                    "source_types": ["ORGANIZATION"],
                    "target_types": ["ORGANIZATION"],
                }
            ],
        }
    )


def _window(content: str) -> ExtractionWindow:
    return window(chunk(content, chunk_id="chunk-1"))


def _entity(name: str, entity_type: str, quote: str) -> dict[str, object]:
    return {
        "name": name,
        "type": entity_type,
        "description": f"{name} is named in the source.",
        "source_chunk_id": "chunk-1",
        "quote": quote,
        "confidence": 1.0,
        "aliases": [],
    }


def _extract_entities(
    content: str, records: list[dict[str, object]]
) -> tuple[list[EntityMention], dict[str, Any]]:
    outcome = LlmEntityExtractor(_PayloadLlm("entities", records)).extract(
        _window(content), _profile(), model="fake", timeout_seconds=30, max_entities=10
    )
    return outcome.entities, outcome.trace


def test_an_entity_whose_name_ocr_ran_together_is_kept_as_ocr_tolerant() -> None:
    content = "Supplier: NorthwindChemicalsLtdHarbourStreet12,Springfield"

    entities, trace = _extract_entities(
        content,
        [
            _entity(
                "Northwind Chemicals Ltd",
                "ORGANIZATION",
                "Northwind Chemicals Ltd, Harbour Street 12, Springfield",
            )
        ],
    )

    [entity] = entities
    assert entity.evidence_method == "ocr_tolerant"
    assert entity.quote == content[entity.start_offset : entity.end_offset]
    assert entity.quote == "NorthwindChemicalsLtdHarbourStreet12,Springfield"
    assert trace["record_actions"] == {"ocr_tolerant_grounding": 1}
    # A kept record is not listed among the rejected.
    assert trace["rejected_records"] == []


def test_text_that_grounds_as_written_keeps_its_method_and_a_miss_is_still_rejected() -> None:
    content = "Supplier: Northwind Chemicals Ltd, Harbour Street 12. Lot s123 released."

    entities, trace = _extract_entities(
        content,
        [
            _entity("Northwind Chemicals Ltd", "ORGANIZATION", "Northwind Chemicals Ltd"),
            _entity("Lot 5123", "ORGANIZATION", "Lot 5123 released"),
        ],
    )

    [entity] = entities
    assert entity.evidence_method == "llm"
    assert "ocr_tolerant_grounding" not in trace["record_actions"]
    assert [(record["reason"], record["name"]) for record in trace["rejected_records"]] == [
        ("ungrounded_quote", "Lot 5123")
    ]


def _extract_subject(
    content: str, name: str, quote: str
) -> tuple[list[EntityMention], dict[str, Any]]:
    outcome = LlmEntityExtractor(
        _PayloadLlm("entities", [_entity(name, "PRODUCT", quote)])
    ).extract_document_context_entities(
        _window(content), _profile(), model="fake", timeout_seconds=30, max_entities=3
    )
    return outcome.entities, outcome.trace


def test_a_document_subject_whose_name_ocr_misread_is_kept_as_ocr_tolerant() -> None:
    content = "ORION 432O4\nTechnical Data Sheet\nRevision 3"

    # The model quotes the text as printed and writes the name as meant.
    entities, trace = _extract_subject(content, "Orion 43204", "ORION 432O4")

    [subject] = entities
    assert subject.is_document_subject is True
    assert subject.evidence_method == "ocr_tolerant"
    assert subject.quote == content[subject.start_offset : subject.end_offset] == "ORION 432O4"
    assert trace["record_actions"] == {"ocr_tolerant_grounding": 1}

    # Or it quotes what it read, and the text is found as OCR printed it.
    entities, _ = _extract_subject(content, "Orion 43204", "Orion 43204 Technical Data Sheet")
    [subject] = entities
    assert subject.evidence_method == "ocr_tolerant"
    assert subject.quote == content[subject.start_offset : subject.end_offset]
    assert subject.quote == "ORION 432O4\nTechnical Data Sheet"

    # A different name is still a mismatch.
    entities, trace = _extract_subject(content, "Orion 43205", "ORION 432O4")
    assert entities == []
    assert trace["record_actions"] == {"document_context_name_mismatch": 1}


def _mention(entity_id: str, name: str) -> EntityMention:
    return EntityMention(
        id=entity_id,
        name=name,
        type="ORGANIZATION",
        description="",
        source_chunk_id="chunk-1",
        quote=name,
    )


def test_a_relation_between_names_ocr_ran_together_is_kept_as_ocr_tolerant() -> None:
    content = "Supplied by NorthwindChemicalsLtdtoAcmeWorkshop on 2 May."
    mentions = [_mention("northwind", "Northwind Chemicals Ltd"), _mention("acme", "Acme Workshop")]
    relation = {
        "source_entity_id": "northwind",
        "target_entity_id": "acme",
        "source_surface": "Northwind Chemicals Ltd",
        "target_surface": "Acme Workshop",
        "relation_type": "SUPPLIES",
        "description": "Northwind Chemicals Ltd supplies Acme Workshop.",
        "source_chunk_id": "chunk-1",
        "quote": "Supplied by Northwind Chemicals Ltd to Acme Workshop",
        "confidence": 1.0,
    }
    extraction_window = _window(content)

    # Both names reach the relation call although neither is in the text as written.
    inventory = _relation_inventory_for_window(mentions, extraction_window)
    assert {entity.id for entity in inventory} == {"northwind", "acme"}
    outcome = LlmRelationExtractor(_PayloadLlm("relations", [relation])).extract(
        extraction_window,
        mentions,
        _profile(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    [observation] = outcome.relations
    assert observation.evidence_method == "ocr_tolerant"
    assert observation.quote == content[observation.start_offset : observation.end_offset]
    assert observation.quote == "Supplied by NorthwindChemicalsLtdtoAcmeWorkshop"
    assert (observation.source_surface, observation.target_surface) == (
        "NorthwindChemicalsLtd",
        "AcmeWorkshop",
    )
    assert outcome.trace["record_actions"]["ocr_tolerant_grounding"] == 1
    assert outcome.trace["rejected_records"] == []


def test_a_located_relation_whose_endpoint_ocr_misread_is_labelled_ocr_tolerant() -> None:
    content = (
        "Northwind Chemicals Ltd was audited in May.\n"
        "The audit covered storage.\nIt covered transport.\nIt covered records.\n"
        "The company supplies Acrne Workshop."
    )
    mentions = [_mention("northwind", "Northwind Chemicals Ltd"), _mention("acme", "Acme Workshop")]

    class _PointingLlm(FakeLlmProvider):
        """Paraphrase the quote, then point at the first and last lines."""

        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            if request.task_name == "relation_grounding":
                payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
                lines = [payload["units"][0]["n"], payload["units"][-1]["n"]]
                decision = {"id": "r0", "supported": True, "lines": lines}
                return StructuredCompletionResult(payload={"decisions": [decision]})
            relation = {
                "source_entity_id": "northwind",
                "target_entity_id": "acme",
                "source_surface": "Northwind Chemicals Ltd",
                "target_surface": "Acme Workshop",
                "relation_type": "SUPPLIES",
                "description": "Northwind Chemicals Ltd supplies Acme Workshop.",
                "source_chunk_id": "chunk-1",
                "quote": "Northwind supplies the workshop",
                "confidence": 1.0,
            }
            return StructuredCompletionResult(payload={"relations": [relation]})

    outcome = LlmRelationExtractor(_PointingLlm(), grounding_fallback=True).extract(
        _window(content), mentions, _profile(), model="fake", timeout_seconds=30, max_relations=10
    )

    [observation] = outcome.relations
    assert observation.evidence_method == "ocr_tolerant"
    assert observation.quote == content
    assert observation.target_surface == "Acrne Workshop"
    assert outcome.trace["record_actions"]["llm_grounding_located"] == 1
    assert outcome.trace["record_actions"]["ocr_tolerant_grounding"] == 1
