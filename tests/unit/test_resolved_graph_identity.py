# SPDX-License-Identifier: Apache-2.0
"""Keep resolver components and source-level loop intent through graph assembly."""

from __future__ import annotations

import json

import pytest
from documents import chunk, grounded

from kg_processor.adapters.embeddings.hash import HashEmbeddingProvider
from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.graph_filter import (
    filter_entities_with_decisions,
    filter_relations_with_decisions,
)
from kg_processor.application.graph_merge import (
    assemble_graph_with_decisions,
    prune_isolated_entities,
)
from kg_processor.application.graph_quality import _self_loop_check
from kg_processor.application.two_pass_extraction import (
    extract_graph_observations,
    resolve_extraction_observations,
)
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionObservations,
    RelationObservation,
)
from kg_processor.domain.graph import Chunk, ExtractionResult
from kg_processor.domain.ids import stable_id
from kg_processor.domain.ontology import (
    EntityTypeDefinition,
    OntologyProfile,
    RelationTypeDefinition,
)
from kg_processor.ports.embeddings import EmbedOptions
from kg_processor.ports.llm import StructuredCompletionRequest, StructuredCompletionResult

_OPTIONS = EmbedOptions(model="hash", dimension=16)


class _NoCalls(FakeLlmProvider):
    """Forbid accidental provider adjudication in deterministic resolution cases."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        raise AssertionError(f"Unexpected provider request: {request.task_name}")


def _resolve(
    observations: ExtractionObservations, ontology: OntologyProfile, enabled: bool = False
) -> ExtractionResult:
    return resolve_extraction_observations(
        observations,
        _NoCalls(),
        HashEmbeddingProvider(),
        _OPTIONS,
        GraphSettings(entity_resolution_enabled=enabled),
        ontology,
        "checksum",
        "test",
        "fake",
        1,
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("same_expansion", [False, True])
def test_resolved_homonyms_keep_component_ids_provenance_and_endpoints(
    enabled: bool, same_expansion: bool
) -> None:
    ontology = OntologyProfile(
        name="homonyms",
        description="Grounded acronym homonyms",
        mode="closed",
        entity_types=[
            EntityTypeDefinition(name=name, description=name)
            for name in ("CONCEPT", "CHEMICAL", "DEVICE")
        ],
        relation_types=[RelationTypeDefinition(name="DESCRIBES", description="describes")],
    )
    chunks: list[Chunk] = []
    mentions: list[EntityMention] = []
    relations: list[RelationObservation] = []
    for index, expansion in enumerate(
        (
            "structure-activity relationship",
            "structure-activity relationship" if same_expansion else "specific absorption rate",
        )
    ):
        target = f"Target{index}"
        text = f"SAR ({expansion}) describes {target}."
        passage = chunk(text, chunk_id=f"chunk-{index}", file_id=f"file-{index}")
        chunks.append(passage)
        mentions.extend(
            [
                EntityMention(
                    id=f"concept-{index}",
                    name="SAR",
                    type="CONCEPT",
                    aliases=[expansion],
                    description=text,
                    source_chunk_id=passage.id,
                    **grounded(text, text),
                ),
                EntityMention(
                    id=f"target-{index}",
                    name=target,
                    type="CHEMICAL" if index == 0 else "DEVICE",
                    description=target,
                    source_chunk_id=passage.id,
                    **grounded(text, target),
                ),
            ]
        )
        relations.append(
            RelationObservation(
                id=f"relation-{index}",
                source_entity_id=f"concept-{index}",
                target_entity_id=f"target-{index}",
                source_surface="SAR",
                target_surface=target,
                relation_type="DESCRIBES",
                description=text,
                source_chunk_id=passage.id,
                **grounded(text, text),
            )
        )
    observations = ExtractionObservations(
        entities=mentions, relations=relations, chunk_count=2, window_count=2
    )
    extraction = _resolve(observations, ontology, enabled)
    by_chunk = {item.id: item for item in chunks}
    entities = filter_entities_with_decisions(extraction.entities, by_chunk).kept
    filtered_relations = filter_relations_with_decisions(
        extraction.relations, entities, by_chunk
    ).kept
    assembly = prune_isolated_entities(
        assemble_graph_with_decisions(
            "graph", chunks, entities, filtered_relations, 10, ontology=ontology
        )
    )
    concepts = [item for item in assembly.nodes if item.primary_type == "CONCEPT"]
    assert len(concepts) == (1 if same_expansion else 2)
    assert len(assembly.edges) == 2
    assert len({edge.source_node_id for edge in assembly.edges}) == len(concepts)
    if same_expansion:
        assert concepts[0].source_chunk_ids == ["chunk-0", "chunk-1"]
        assert concepts[0].id == stable_id("node", "graph", "sar", "CONCEPT")
    else:
        assert {item.id for item in concepts} == {
            stable_id("node", "graph", "sar", "CONCEPT"),
            stable_id("node", "graph", "sar", "CONCEPT", "concept-1"),
        }
        for node in concepts:
            assert len(node.source_chunk_ids) == len(node.aliases) == 1
            chunk_id = node.source_chunk_ids[0]
            source = next(item for item in assembly.entity_sources if item.node_id == node.id)
            assert source.file_id == by_chunk[chunk_id].file_id
            assert set(source.aliases) == {node.name, *node.aliases}
            edge = next(item for item in assembly.edges if item.source_node_id == node.id)
            assert edge.source_chunk_ids == [chunk_id]
            assert all(
                item.chunk_id == chunk_id
                for item in assembly.evidence
                if item.subject_id == node.id
            )
        # Same-name endpoints without a component are ambiguous and must not pick
        # whichever homonym happened to be assembled last.
        ambiguous = extraction.relations[0].model_copy(update={"source_component_id": None})
        rejected = assemble_graph_with_decisions("graph", chunks, entities, [ambiguous], 10)
        assert rejected.edges == []
        assert rejected.decision_reason_counts()["source_node_missing"] == 1

    reversed_extraction = _resolve(
        ExtractionObservations(
            entities=list(reversed(mentions)),
            relations=list(reversed(relations)),
            chunk_count=2,
            window_count=2,
        ),
        ontology,
        enabled,
    )
    reversed_assembly = assemble_graph_with_decisions(
        "graph", chunks, reversed_extraction.entities, reversed_extraction.relations, 10
    )
    assert {node.id for node in assembly.nodes} == {node.id for node in reversed_assembly.nodes}
    assert {edge.id for edge in assembly.edges} == {edge.id for edge in reversed_assembly.edges}


_TEXT = (
    "Alpha activates Alpha kinase through a documented feedback mechanism. "
    "Alpha activates Beta during the same experiment."
)


class _LoopProvider(FakeLlmProvider):
    """Emit a grounded source assertion using two aliases of one accepted mention."""

    def __init__(self, passage: Chunk, loop: bool) -> None:
        self.passage = passage
        self.loop = loop

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        if request.task_name == "entity_extraction":
            return StructuredCompletionResult(
                payload={
                    "entities": [
                        {
                            "name": name,
                            "type": "PROTEIN",
                            "description": "Named protein.",
                            "source_chunk_id": self.passage.id,
                            "quote": name,
                            "confidence": 1,
                            "aliases": ["Alpha kinase"] if name == "Alpha" else [],
                        }
                        for name in ("Alpha", "Beta")
                    ]
                }
            )
        assert request.task_name == "relation_extraction"
        data = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {entity["name"]: entity["id"] for entity in data["entities"]}
        quote = _TEXT.split(". ", 1)[0] + "." if self.loop else _TEXT.split(". ", 1)[1]
        return StructuredCompletionResult(
            payload={
                "relations": [
                    {
                        "source_entity_id": ids["Alpha"],
                        "target_entity_id": ids["Alpha" if self.loop else "Beta"],
                        "source_surface": "Alpha",
                        "target_surface": "Alpha kinase" if self.loop else "Beta",
                        "relation_type": "ACTIVATES",
                        "description": quote,
                        "source_chunk_id": self.passage.id,
                        "quote": quote,
                        "confidence": 1,
                    }
                ]
            }
        )


def _ontology(allowed: bool) -> OntologyProfile:
    return OntologyProfile(
        name="self-regulation",
        description="Protein self regulation",
        mode="closed",
        entity_types=[EntityTypeDefinition(name="PROTEIN", description="Named protein")],
        relation_types=[
            RelationTypeDefinition(
                name="ACTIVATES",
                description="A protein activates a protein",
                source_types=["PROTEIN"],
                target_types=["PROTEIN"],
                allow_self_loop=allowed,
            )
        ],
    )


@pytest.mark.parametrize(
    ("allowed", "loop", "edge_count"), [(True, True, 1), (False, True, 0), (False, False, 1)]
)
def test_explicit_ontology_allowed_self_relations_survive_finalization(
    allowed: bool, loop: bool, edge_count: int
) -> None:
    passage = chunk(_TEXT)
    ontology = _ontology(allowed)
    provider = _LoopProvider(passage, loop)
    settings = GraphSettings(
        gleaning_max_passes=0,
        relation_continuation_max_passes=0,
        relation_empty_retry_min_entities=0,
        verify_relations=False,
        verify_entities=False,
        entity_resolution_enabled=False,
    )
    observations = extract_graph_observations([passage], provider, settings, ontology, "fake", 1)
    extraction = _resolve(observations, ontology)
    assembly = assemble_graph_with_decisions(
        "graph", [passage], extraction.entities, extraction.relations, 10, ontology=ontology
    )
    assert (
        len(observations.relations)
        == len(extraction.relations)
        == len(assembly.edges)
        == edge_count
    )
    if loop and allowed:
        assert extraction.relations[0].source_self_loop
        assert assembly.edges[0].source_node_id == assembly.edges[0].target_node_id
        assert _self_loop_check([edge.model_dump() for edge in assembly.edges], ontology).ok
        # The marker alone never overrides an ontology that forbids loops.
        forbidden = assemble_graph_with_decisions(
            "graph",
            [passage],
            extraction.entities,
            extraction.relations,
            10,
            ontology=_ontology(False),
        )
        assert forbidden.edges == []


@pytest.mark.parametrize("allowed", [False, True])
def test_merging_distinct_source_mentions_never_creates_a_self_loop(allowed: bool) -> None:
    passage = chunk(_TEXT)
    mentions = [
        EntityMention(
            id=entity_id,
            name=name,
            type="PROTEIN",
            description=name,
            source_chunk_id=passage.id,
            aliases=["Alpha kinase"] if name == "Alpha" else [],
            **grounded(_TEXT, name),
        )
        for entity_id, name in (("first", "Alpha"), ("second", "Alpha kinase"))
    ]
    relation = RelationObservation(
        id="relation",
        source_entity_id="first",
        target_entity_id="second",
        source_surface="Alpha",
        target_surface="Alpha kinase",
        relation_type="ACTIVATES",
        description=_TEXT,
        source_chunk_id=passage.id,
        **grounded(_TEXT, _TEXT),
    )
    ontology = _ontology(allowed)
    extraction = _resolve(
        ExtractionObservations(
            entities=mentions, relations=[relation], chunk_count=1, window_count=1
        ),
        ontology,
    )
    assert len(extraction.relations) == 1
    assert not extraction.relations[0].source_self_loop
    assembly = assemble_graph_with_decisions(
        "graph", [passage], extraction.entities, extraction.relations, 10, ontology=ontology
    )
    assert len(assembly.nodes) == 1
    assert assembly.edges == []
    assert assembly.decision_reason_counts()["self_loop_after_entity_merge"] == 1


def test_explicit_loop_is_not_deduplicated_with_a_merge_induced_loop() -> None:
    """A dropped merge artifact must not hide a valid assertion in the same chunk."""

    passage = chunk(_TEXT)
    mentions = [
        EntityMention(
            id=entity_id,
            name=name,
            type="PROTEIN",
            description=name,
            source_chunk_id=passage.id,
            aliases=["Alpha kinase"] if name == "Alpha" else [],
            **grounded(_TEXT, name),
        )
        for entity_id, name in (("first", "Alpha"), ("second", "Alpha kinase"))
    ]
    induced = RelationObservation(
        id="induced",
        source_entity_id="first",
        target_entity_id="second",
        source_surface="Alpha",
        target_surface="Alpha kinase",
        relation_type="ACTIVATES",
        description=_TEXT,
        source_chunk_id=passage.id,
        **grounded(_TEXT, _TEXT),
    )
    explicit = induced.model_copy(update={"id": "explicit", "target_entity_id": "first"})
    ontology = _ontology(True)
    extraction = _resolve(
        ExtractionObservations(
            entities=mentions, relations=[induced, explicit], chunk_count=1, window_count=1
        ),
        ontology,
    )
    assert len(extraction.relations) == 2
    assembly = assemble_graph_with_decisions(
        "graph", [passage], extraction.entities, extraction.relations, 10, ontology=ontology
    )
    assert len(assembly.edges) == 1
    assert assembly.edges[0].source_node_id == assembly.edges[0].target_node_id
    assert assembly.decision_reason_counts()["self_loop_after_entity_merge"] == 1
