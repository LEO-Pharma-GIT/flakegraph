from __future__ import annotations

import json

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.ports.llm import StructuredCompletionRequest


def test_fake_llm_derives_grounded_entities_without_crossing_lines() -> None:
    """The test adapter should derive deterministic records from supplied content."""

    result = FakeLlmProvider().complete_structured(
        _entity_request(
            (
                "Alice Smith works at Acme Corp in Copenhagen.\n"
                "Acme Corp collaborates with Contoso Health.\n"
                "Copenhagen teams evaluate MinerU OCR and Azure OpenAI."
            ),
            ["PERSON", "ORGANIZATION", "CONCEPT"],
        )
    )

    entities = result.payload["entities"]
    assert isinstance(entities, list)
    assert [entity["name"] for entity in entities] == [
        "Alice Smith",
        "Acme Corp",
        "Copenhagen",
        "Contoso Health",
        "MinerU OCR",
        "Azure OpenAI",
    ]
    assert all("\n" not in str(entity["name"]) for entity in entities)
    assert result.provider_metadata["task_name"] == "entity_extraction"


def test_fake_llm_defaults_to_concept_without_an_entity_taxonomy() -> None:
    """An empty taxonomy should still yield valid generic entity records."""

    result = FakeLlmProvider().complete_structured(
        _entity_request("Alice Smith works at Acme Corp.", [])
    )

    entities = result.payload["entities"]
    assert isinstance(entities, list)
    assert {entity["type"] for entity in entities} == {"CONCEPT"}


def _entity_request(content: str, entity_types: list[str]) -> StructuredCompletionRequest:
    """Build the minimal structured task understood by the deterministic adapter."""

    payload = {
        "entity_types": entity_types,
        "chunks": [{"id": "chunk_1", "content": content}],
    }
    return StructuredCompletionRequest(
        task_name="entity_extraction",
        model="fake",
        system="Extract entities.",
        user=f"INPUT_JSON:\n{json.dumps(payload)}",
        json_schema={"type": "object"},
    )
