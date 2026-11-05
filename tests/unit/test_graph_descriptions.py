from __future__ import annotations

from kg_processor.application.graph_descriptions import synthesize_entity_descriptions
from kg_processor.domain.graph import (
    EntitySource,
    Evidence,
    ExtractedEntity,
    GraphNode,
)
from kg_processor.ports.llm import (
    DescriptionMergeRequest,
    DescriptionMergeResult,
)


def test_synthesize_entity_descriptions_updates_nodes_and_entity_sources() -> None:
    node = _node("node_alice", "Alice Smith", "PERSON")
    source = EntitySource(
        id="source_1",
        graph_id="graph",
        node_id=node.id,
        file_id="file_1",
        per_file_description="Old description.",
        mention_count=2,
    )
    provider = _RecordingLlm("Alice Smith is a researcher at Acme Corp.")

    result = synthesize_entity_descriptions(
        [node],
        [source],
        [
            _entity("Alice Smith", "Alice Smith is a researcher."),
            _entity("Alice Smith", "Alice Smith works at Acme Corp."),
        ],
        [_evidence(node.id, "Alice Smith works at Acme Corp in Copenhagen.")],
        provider,
        min_observations=2,
        max_descriptions=8,
        max_evidence=5,
    )

    assert result.merged_count == 1
    assert node.description == "Alice Smith is a researcher at Acme Corp."
    assert source.per_file_description == node.description
    assert result.trace_events == [
        {
            "stage": "description_merge",
            "action": "merged",
            "node_id": "node_alice",
            "entity_name": "Alice Smith",
            "entity_type": "PERSON",
            "observations_available": 2,
            "descriptions_used": 2,
            "evidence_used": 1,
            "description_length": len("Alice Smith is a researcher at Acme Corp."),
            "provider_metadata": {
                "provider": "recording_llm",
                "prompt_name": "entity_description_merge",
            },
        }
    ]
    assert provider.requests == [
        DescriptionMergeRequest(
            entity_name="Alice Smith",
            entity_type="PERSON",
            descriptions=[
                "Alice Smith is a researcher.",
                "Alice Smith works at Acme Corp.",
            ],
            evidence=["Alice Smith works at Acme Corp in Copenhagen."],
        )
    ]


def test_synthesize_entity_descriptions_skips_single_observation() -> None:
    node = _node("node_acme", "Acme Corp", "ORGANIZATION")
    provider = _RecordingLlm("unused")

    result = synthesize_entity_descriptions(
        [node],
        [],
        [_entity("Acme Corp", "Acme Corp is an organization.")],
        [],
        provider,
        min_observations=2,
        max_descriptions=8,
        max_evidence=5,
    )

    assert result.merged_count == 0
    assert result.trace_events == []
    assert provider.requests == []
    assert node.description == "Original description."


class _RecordingLlm:
    def __init__(self, description: str) -> None:
        self.description = description
        self.requests: list[DescriptionMergeRequest] = []

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        self.requests.append(request)
        return DescriptionMergeResult(
            description=self.description,
            provider_metadata={
                "provider": "recording_llm",
                "prompt_name": "entity_description_merge",
            },
        )


def _node(node_id: str, name: str, primary_type: str) -> GraphNode:
    return GraphNode(
        id=node_id,
        graph_id="graph",
        normalized_name=name.lower().replace(" ", ""),
        name=name,
        primary_type=primary_type,
        types=[primary_type],
        description="Original description.",
        source_chunk_ids=["chunk_1"],
    )


def _entity(name: str, description: str) -> ExtractedEntity:
    return ExtractedEntity(
        name=name,
        type="PERSON" if name == "Alice Smith" else "ORGANIZATION",
        description=description,
        source_chunk_id="chunk_1",
    )


def _evidence(subject_id: str, quote: str) -> Evidence:
    return Evidence(
        id="evidence_1",
        graph_id="graph",
        subject_id=subject_id,
        subject_kind="node",
        file_id="file_1",
        chunk_id="chunk_1",
        page_number=1,
        start_offset=0,
        end_offset=len(quote),
        quote=quote,
    )


def test_description_synthesis_keeps_same_named_component_sources_separate() -> None:
    """Same display names must not combine descriptions from different components."""

    first = _node("first", "SAR", "CONCEPT")
    first.source_chunk_ids = ["first-chunk"]
    second = _node("second", "SAR", "CONCEPT")
    second.source_chunk_ids = ["second-chunk"]
    provider = _RecordingLlm("A synthesized description.")
    observations = [
        ExtractedEntity(
            name="SAR",
            type="CONCEPT",
            description=description,
            source_chunk_id=chunk_id,
            resolved_component_id=component,
        )
        for chunk_id, component, descriptions in [
            (
                "first-chunk",
                "first",
                ["Structure activity relationship.", "A chemical relationship."],
            ),
            (
                "second-chunk",
                "second",
                ["Specific absorption rate.", "A measured absorption rate."],
            ),
        ]
        for description in descriptions
    ]
    result = synthesize_entity_descriptions(
        [first, second],
        [],
        observations,
        [],
        provider,
        min_observations=2,
        max_descriptions=8,
        max_evidence=5,
    )
    assert result.merged_count == 2
    assert [request.descriptions for request in provider.requests] == [
        ["Structure activity relationship.", "A chemical relationship."],
        ["Specific absorption rate.", "A measured absorption rate."],
    ]
