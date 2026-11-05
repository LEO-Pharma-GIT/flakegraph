from __future__ import annotations

import pytest
from documents import chunk, grounded

from kg_processor.application.graph_filter import (
    filter_entities_with_decisions,
    filter_relations_with_decisions,
)
from kg_processor.application.graph_merge import (
    GraphAssemblyResult,
    _best_description,
    assemble_graph,
    assemble_graph_with_decisions,
    claimed_only,
    normalize_entity_name,
    normalize_relation_type,
    prune_isolated_entities,
)
from kg_processor.domain.graph import ExtractedEntity, ExtractedRelation, RelationAssertion


def test_normalization_helpers() -> None:
    assert normalize_entity_name("Next.js") == "nextjs"
    assert normalize_entity_name("Müller") == "müller"
    assert normalize_entity_name("Mller") == "mller"
    assert normalize_entity_name("東京") == "東京"
    assert normalize_relation_type(" Located In ") == "located_in"
    assert normalize_relation_type("Works-At") == normalize_relation_type("works at")
    assert _best_description(["equal value B", "equal value A"]) == "equal value B"
    assert _best_description(["equal value A", "equal value B"]) == "equal value B"


def test_assemble_graph_deduplicates_entities_and_tracks_evidence() -> None:
    passage = chunk("Alice Smith works at Acme Corp.")
    entities = [
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice Smith is a person.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "Alice Smith"),
        ),
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice Smith works at Acme Corp.",
            source_chunk_id=passage.id,
            aliases=["A. Smith"],
            **grounded(passage.content, "Alice Smith"),
        ),
        ExtractedEntity(
            name="Acme Corp",
            type="ORGANIZATION",
            description="Acme Corp is an organization.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "Acme Corp"),
        ),
    ]
    relations = [
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Acme Corp",
            source_type="PERSON",
            target_type="ORGANIZATION",
            relation_type="works at",
            description="Alice Smith works at Acme Corp.",
            source_chunk_id=passage.id,
            weight=12,
            **grounded(passage.content, "works at"),
        )
    ]

    filtered_entities = filter_entities_with_decisions(entities, {passage.id: passage}).kept
    filtered_relations = filter_relations_with_decisions(relations, filtered_entities).kept
    nodes, edges, evidence, sources = assemble_graph(
        "graph", [passage], filtered_entities, filtered_relations, relation_weight_max=10
    )

    assert len(nodes) == 2
    assert next(node for node in nodes if node.name == "Alice Smith").aliases == ["A. Smith"]
    assert len(edges) == 1
    assert edges[0].weight == 10
    assert len(evidence) == 3
    assert len(sources) == 2


def test_prune_isolated_entities_removes_dependent_provenance_but_keeps_edges() -> None:
    """Disconnected observations must not survive as expensive graph entities."""

    passage = chunk("Alice works at Acme. Incidental appears separately.")
    entities = [
        ExtractedEntity(
            name="Alice",
            type="PERSON",
            description="A person.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "Alice"),
        ),
        ExtractedEntity(
            name="Acme",
            type="ORGANIZATION",
            description="An organization.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "Acme"),
        ),
        ExtractedEntity(
            name="Incidental",
            type="CONCEPT",
            description="An incidental concept.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "Incidental"),
        ),
    ]
    relation = ExtractedRelation(
        source_name="Alice",
        source_type="PERSON",
        target_name="Acme",
        target_type="ORGANIZATION",
        relation_type="WORKS_AT",
        description="Alice works at Acme.",
        source_chunk_id=passage.id,
        **grounded(passage.content, "Alice works at Acme."),
    )
    assembly = assemble_graph_with_decisions("graph", [passage], entities, [relation], 10)

    pruned = prune_isolated_entities(assembly)

    assert {node.name for node in pruned.nodes} == {"Alice", "Acme"}
    assert len(pruned.edges) == 1
    assert {source.node_id for source in pruned.entity_sources} == {
        node.id for node in pruned.nodes
    }
    assert all(
        item.subject_kind != "node" or item.subject_id in {node.id for node in pruned.nodes}
        for item in pruned.evidence
    )
    assert any(
        decision.reason == "isolated_entity_without_relation" for decision in pruned.decisions
    )


def test_assemble_graph_keeps_same_name_with_distinct_types_addressable() -> None:
    passage = chunk("Jordan teaches grappling at North Hall.")
    entities = [
        ExtractedEntity(
            name="Jordan",
            type="PERSON",
            description="Jordan is a person.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "Jordan"),
        ),
        ExtractedEntity(
            name="Jordan",
            type="ORGANIZATION",
            description="Jordan is also mislabeled as an organization.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "Jordan"),
        ),
        ExtractedEntity(
            name="North Hall",
            type="LOCATION",
            description="North Hall is a location.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "North Hall"),
        ),
    ]
    relations = [
        ExtractedRelation(
            source_name="Jordan",
            target_name="North Hall",
            source_type="PERSON",
            target_type="LOCATION",
            relation_type="teaches at",
            description="Jordan teaches at North Hall.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "teaches grappling at"),
        )
    ]

    nodes, edges, _evidence, _sources = assemble_graph(
        "graph",
        [passage],
        entities,
        relations,
        relation_weight_max=10,
    )

    jordan = next(node for node in nodes if node.name == "Jordan" and node.primary_type == "PERSON")
    assert len(nodes) == 3
    assert jordan.primary_type == "PERSON"
    assert jordan.types == ["PERSON"]
    assert edges[0].source_node_id == jordan.id
    assert jordan.degree == 1


def test_assemble_graph_drops_loop_created_by_entity_normalization() -> None:
    """Do not let punctuation variants become an invalid canonical self-loop."""

    passage = chunk("Wal-Mart renamed Walmart.")
    entities = [
        ExtractedEntity(
            name=name,
            type="ORGANIZATION",
            description=name,
            source_chunk_id=passage.id,
            **grounded(passage.content, name),
        )
        for name in ("Wal-Mart", "Walmart")
    ]
    relation = ExtractedRelation(
        source_name="Wal-Mart",
        target_name="Walmart",
        source_type="ORGANIZATION",
        target_type="ORGANIZATION",
        relation_type="renamed to",
        description=passage.content,
        source_chunk_id=passage.id,
        **grounded(passage.content, "renamed"),
    )

    result = assemble_graph_with_decisions("graph", [passage], entities, [relation], 10)

    assert len(result.nodes) == 1
    assert result.edges == []
    assert result.decision_reason_counts()["self_loop_after_entity_merge"] == 1


def test_assemble_graph_records_merge_decisions() -> None:
    """Ensure graph assembly emits explicit decisions for created and merged observations.

    The trace should explain graph cardinality changes.
    """

    passage = chunk("Alice Smith works at Acme Corp.")
    entities = [
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice Smith is present.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "Alice Smith"),
        ),
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice Smith works at Acme Corp.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "Alice Smith"),
        ),
        ExtractedEntity(
            name="Acme Corp",
            type="ORGANIZATION",
            description="Acme is present.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "Acme Corp"),
        ),
    ]
    relations = [
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Acme Corp",
            source_type="PERSON",
            target_type="ORGANIZATION",
            relation_type="works at",
            description="First observation.",
            source_chunk_id=passage.id,
            weight=7,
            **grounded(passage.content, "works at"),
        ),
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Acme Corp",
            source_type="PERSON",
            target_type="ORGANIZATION",
            relation_type="works at",
            description="Longer duplicate relation observation.",
            source_chunk_id=passage.id,
            weight=7,
            **grounded(passage.content, "works at"),
        ),
        ExtractedRelation(
            source_name="Ghost",
            target_name="Acme Corp",
            source_type="PERSON",
            target_type="ORGANIZATION",
            relation_type="mentions",
            description="Source node is missing.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "Acme Corp"),
        ),
    ]

    result = assemble_graph_with_decisions("graph", [passage], entities, relations, 10)

    assert len(result.nodes) == 2
    assert len(result.edges) == 1
    # Duplicate model output for the same edge/chunk must not amplify weight.
    assert result.edges[0].weight == 7
    assert len(result.edge_observations) == 1
    assert result.decision_reason_counts() == {
        "canonical_edge_created": 1,
        "canonical_node_created": 1,
        "entity_observations_merged": 1,
        "duplicate_edge_observation": 1,
        "source_node_missing": 1,
    }
    merge_events = [decision.to_trace_event() for decision in result.decisions]
    assert any(
        event["kind"] == "entity"
        and event["action"] == "merged"
        and event["reason"] == "entity_observations_merged"
        and event["observation_count"] == 2
        for event in merge_events
    )
    assert any(
        event["kind"] == "relation"
        and event["action"] == "dropped"
        and event["reason"] == "duplicate_edge_observation"
        for event in merge_events
    )
    assert any(
        event["kind"] == "relation"
        and event["action"] == "dropped"
        and event["reason"] == "source_node_missing"
        for event in merge_events
    )


def test_assemble_graph_uses_extracted_quote_spans_for_evidence() -> None:
    passage = chunk("Alice Smith works at Acme Corp.", start=100)
    entities = [
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice Smith is present.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "Alice Smith"),
        ),
        ExtractedEntity(
            name="Acme Corp",
            type="ORGANIZATION",
            description="Acme is present.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "Acme Corp"),
        ),
    ]
    relations = [
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Acme Corp",
            source_type="PERSON",
            target_type="ORGANIZATION",
            relation_type="works at",
            description="Alice works at Acme.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "works at"),
        )
    ]

    nodes, edges, evidence, _sources = assemble_graph(
        "graph",
        [passage],
        entities,
        relations,
        relation_weight_max=10,
    )

    node_by_name = {node.name: node for node in nodes}
    alice_evidence = next(
        row for row in evidence if row.subject_id == node_by_name["Alice Smith"].id
    )
    edge_evidence = next(row for row in evidence if row.subject_id == edges[0].id)
    assert alice_evidence.quote == "Alice Smith"
    assert alice_evidence.start_offset == 100
    assert alice_evidence.end_offset == 111
    assert edge_evidence.quote == "works at"
    assert edge_evidence.start_offset == 112
    assert edge_evidence.end_offset == 120


def test_assemble_graph_refuses_an_entity_without_grounded_offsets() -> None:
    """Every extractor grounds its observations, so assembly treats a missing span as a bug."""

    passage = chunk("Alice Smith works at Acme Corp.")
    entity = ExtractedEntity(
        name="Alice Smith", type="PERSON", description="Ungrounded.", source_chunk_id=passage.id
    )

    with pytest.raises(ValueError, match="no grounded offsets"):
        assemble_graph("graph", [passage], [entity], [], relation_weight_max=10)


def test_filter_entities_applies_confidence_name_quality_and_blocklist() -> None:
    passage = chunk("Alice Smith works at Acme Corp.")
    entities = [
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice is present.",
            source_chunk_id=passage.id,
            confidence=0.95,
        ),
        ExtractedEntity(
            name="Acme Corp",
            type="ORGANIZATION",
            description="Acme is present.",
            source_chunk_id=passage.id,
            confidence=0.2,
        ),
        ExtractedEntity(
            name="Q",
            type="CONCEPT",
            description="Too short.",
            source_chunk_id=passage.id,
            confidence=0.99,
        ),
        ExtractedEntity(
            name="Document",
            type="CONCEPT",
            description="Blocked document furniture.",
            source_chunk_id=passage.id,
            confidence=0.99,
        ),
    ]

    result = filter_entities_with_decisions(
        entities,
        {passage.id: passage},
        min_confidence=0.8,
        min_name_length=2,
        blocklist=["document"],
    )

    assert [entity.name for entity in result.kept] == ["Alice Smith"]
    assert result.dropped_reason_counts() == {
        "blocklisted_entity": 1,
        "low_confidence": 1,
        "name_too_short": 1,
    }
    assert [
        decision.to_trace_event()["reason"]
        for decision in result.decisions
        if decision.action == "dropped"
    ] == ["low_confidence", "name_too_short", "blocklisted_entity"]


def test_filter_entities_accepts_a_grounded_alias_after_canonical_resolution() -> None:
    """Keep a mention when its canonical spelling comes from another source passage."""

    passage = chunk("The paper studies image recognition.")
    entity = ExtractedEntity(
        name="Image classification",
        type="TASK",
        description="A visual recognition task.",
        source_chunk_id=passage.id,
        aliases=["image recognition"],
        confidence=0.99,
    )

    result = filter_entities_with_decisions([entity], {passage.id: passage})

    assert result.kept == [entity]
    assert result.decisions[0].reason == "grounded_alias"


def test_filter_relations_applies_confidence_and_endpoint_grounding() -> None:
    passage = chunk("Alice Smith works at Acme Corp.")
    entities = [
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice is present.",
            source_chunk_id=passage.id,
        ),
        ExtractedEntity(
            name="Acme Corp",
            type="ORGANIZATION",
            description="Acme is present.",
            source_chunk_id=passage.id,
        ),
    ]
    relations = [
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Acme Corp",
            source_type="PERSON",
            target_type="ORGANIZATION",
            relation_type="works_at",
            description="Alice works at Acme.",
            source_chunk_id=passage.id,
            confidence=0.9,
        ),
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Missing Org",
            source_type="PERSON",
            target_type="ORGANIZATION",
            relation_type="works_at",
            description="Ungrounded endpoint.",
            source_chunk_id=passage.id,
            confidence=0.9,
        ),
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Acme Corp",
            source_type="PERSON",
            target_type="ORGANIZATION",
            relation_type="mentions",
            description="Low confidence.",
            source_chunk_id=passage.id,
            confidence=0.2,
        ),
    ]

    result = filter_relations_with_decisions(
        relations,
        entities,
        {passage.id: passage},
        min_confidence=0.8,
        require_endpoint_grounding=True,
    )

    assert [relation.relation_type for relation in result.kept] == ["works_at"]
    assert result.dropped_reason_counts() == {
        "low_confidence": 1,
        "target_entity_missing": 1,
    }


def test_filter_relations_records_endpoint_grounding_reasons() -> None:
    passage = chunk("Alice Smith works at Acme Corp.")
    entities = [
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice is present.",
            source_chunk_id=passage.id,
        ),
        ExtractedEntity(
            name="Copenhagen",
            type="LOCATION",
            description="Known from another accepted observation.",
            source_chunk_id=passage.id,
        ),
    ]
    relations = [
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Copenhagen",
            source_type="PERSON",
            target_type="LOCATION",
            relation_type="located_in",
            description="Endpoint is known but not grounded in this passage.",
            source_chunk_id=passage.id,
        )
    ]

    result = filter_relations_with_decisions(
        relations,
        entities,
        {passage.id: passage},
        require_endpoint_grounding=True,
    )

    assert result.kept == []
    assert result.dropped_reason_counts() == {"target_endpoint_ungrounded": 1}


def test_filter_relations_grounds_verified_local_endpoint_surfaces() -> None:
    """Preserve alias and document-context evidence after canonical resolution."""

    passage = chunk("We evaluate LSTM on the benchmark.")
    entities = [
        ExtractedEntity(
            name="Sequence Learning Paper",
            type="PAPER",
            description="The source paper.",
            source_chunk_id=passage.id,
        ),
        ExtractedEntity(
            name="Long Short-Term Memory",
            type="MODEL",
            description="A recurrent architecture.",
            source_chunk_id=passage.id,
            aliases=["LSTM"],
        ),
    ]
    relation = ExtractedRelation(
        source_name="Sequence Learning Paper",
        target_name="Long Short-Term Memory",
        source_surface="We",
        target_surface="LSTM",
        source_type="PAPER",
        target_type="MODEL",
        relation_type="USES_METHOD",
        description="The paper evaluates LSTM.",
        source_chunk_id=passage.id,
    )

    result = filter_relations_with_decisions(
        [relation],
        entities,
        {passage.id: passage},
        require_endpoint_grounding=True,
    )

    assert result.kept == [relation]
    assert result.dropped_reason_counts() == {}


def test_resolved_alias_remains_a_relation_endpoint_through_assembly() -> None:
    """Keep local relation surfaces after entity resolution canonicalizes a name."""

    passage = chunk("LSTM uses an input gate.")
    entities = [
        ExtractedEntity(
            name="Long Short-Term Memory",
            type="MODEL",
            aliases=["LSTM"],
            description="A recurrent architecture.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "LSTM"),
        ),
        ExtractedEntity(
            name="input gate",
            type="METHOD",
            description="A multiplicative gate.",
            source_chunk_id=passage.id,
            **grounded(passage.content, "input gate"),
        ),
    ]
    relation = ExtractedRelation(
        source_name="LSTM",
        source_type="MODEL",
        target_name="input gate",
        target_type="METHOD",
        relation_type="USES_METHOD",
        description="LSTM uses an input gate.",
        source_chunk_id=passage.id,
        **grounded(passage.content, "uses"),
    )

    filtered = filter_relations_with_decisions([relation], entities)
    assembly = assemble_graph_with_decisions(
        "graph",
        [passage],
        entities,
        filtered.kept,
        relation_weight_max=10,
    )

    assert filtered.kept == [relation]
    assert len(assembly.edges) == 1
    source = next(node for node in assembly.nodes if node.id == assembly.edges[0].source_node_id)
    assert source.name == "Long Short-Term Memory"


def test_ambiguous_resolved_alias_is_not_used_as_a_relation_endpoint() -> None:
    """Avoid assigning a shared acronym to an arbitrary same-typed entity."""

    entities = [
        ExtractedEntity(
            name=name,
            type="MODEL",
            aliases=["SRN"],
            description=name,
            source_chunk_id="chunk",
        )
        for name in ("Simple Recurrent Network", "Scene Representation Network")
    ]
    entities.append(
        ExtractedEntity(
            name="benchmark",
            type="DATASET",
            description="A benchmark.",
            source_chunk_id="chunk",
        )
    )
    relation = ExtractedRelation(
        source_name="SRN",
        source_type="MODEL",
        target_name="benchmark",
        target_type="DATASET",
        relation_type="EVALUATED_ON",
        description="SRN is evaluated on the benchmark.",
        source_chunk_id="chunk",
    )

    result = filter_relations_with_decisions([relation], entities)

    assert result.kept == []
    assert result.dropped_reason_counts() == {"source_entity_missing": 1}


def test_assemble_graph_aggregates_cross_file_assertions_into_one_edge() -> None:
    """Ensure cross-file assertions share topology while preserving source provenance.

    Each assertion must remain independently removable during reindexing.
    """

    first = chunk("Alice works at Acme.", content_hash="one")
    second = chunk("Alice works at Acme.", chunk_id="chunk_2", file_id="file_2", content_hash="two")
    entities = [
        ExtractedEntity(
            name=name,
            type=entity_type,
            description=name,
            source_chunk_id=passage.id,
            **grounded(passage.content, name),
        )
        for passage in (first, second)
        for name, entity_type in (("Alice", "PERSON"), ("Acme", "ORGANIZATION"))
    ]
    relations = [
        ExtractedRelation(
            source_name="Alice",
            target_name="Acme",
            source_type="PERSON",
            target_type="ORGANIZATION",
            relation_type="works_at",
            description=passage.content,
            source_chunk_id=passage.id,
            **grounded(passage.content, "works at"),
        )
        for passage in (first, second)
    ]

    result = assemble_graph_with_decisions(
        "graph", [first, second], entities, relations, relation_weight_max=10
    )

    assert len(result.edges) == 1
    assert result.edges[0].source_file_ids == ["file_1", "file_2"]
    assert result.edges[0].source_chunk_ids == ["chunk_1", "chunk_2"]
    assert result.edges[0].evidence_count == 2
    assert result.edges[0].weight == 2.0
    assert len(result.edge_observations) == 2


def _claim_graph(
    assertions: list[tuple[str, RelationAssertion]],
) -> GraphAssemblyResult:
    """Assemble one Widget-treats-Rust relation stated once per (chunk text, assertion)."""

    passages = {
        text: chunk(text, chunk_id=f"chunk_{index}", content_hash=f"hash-{index}")
        for index, text in enumerate(dict.fromkeys(text for text, _ in assertions))
    }
    entities = [
        ExtractedEntity(
            name=name,
            type=entity_type,
            description=name,
            source_chunk_id=passage.id,
            **grounded(passage.content, name),
        )
        for passage in passages.values()
        for name, entity_type in (("Widget", "PRODUCT"), ("Rust", "CONDITION"))
    ]
    relations = [
        ExtractedRelation(
            source_name="Widget",
            target_name="Rust",
            source_type="PRODUCT",
            target_type="CONDITION",
            relation_type="TREATS",
            description=text,
            source_chunk_id=passages[text].id,
            assertion=assertion,
            **grounded(text, "Widget"),
        )
        for text, assertion in assertions
    ]
    return assemble_graph_with_decisions(
        "graph", list(passages.values()), entities, relations, relation_weight_max=10
    )


def test_an_edge_is_claimed_only_when_every_observation_claims_it() -> None:
    """A claim stays visible as a claim until one source reports the relation."""

    claim = "Widget may be used for removing Rust."
    finding = "Widget removed Rust in every trial."

    claimed = _claim_graph([(claim, "claimed"), ("Widget is useful against Rust.", "claimed")])
    reported = _claim_graph([(claim, "claimed"), (finding, "reported")])

    assert [edge.assertion for edge in claimed.edges] == ["claimed"]
    assert {item.assertion for item in claimed.edge_observations} == {"claimed"}
    assert [edge.assertion for edge in reported.edges] == ["reported"]
    assert sorted(item.assertion for item in reported.edge_observations) == [
        "claimed",
        "reported",
    ]


def test_a_reported_repetition_in_one_chunk_makes_its_observation_reported() -> None:
    """Two assertions of one edge in one chunk keep one observation, reported if either is."""

    text = "Widget may be used for removing Rust, and removed Rust in trials."

    result = _claim_graph([(text, "claimed"), (text, "reported")])

    assert [edge.assertion for edge in result.edges] == ["reported"]
    assert [item.assertion for item in result.edge_observations] == ["reported"]


def test_claimed_only_needs_every_assertion_to_claim() -> None:
    assert claimed_only(["claimed", "claimed"]) == "claimed"
    assert claimed_only(["claimed", "reported"]) == "reported"
    assert claimed_only([]) == "reported"


def test_the_filter_trusts_an_observation_extraction_grounded_to_a_span() -> None:
    """A name extraction grounded by rephrasing is kept; one with no span is checked."""

    passage = chunk("The pH of the syrups was measured after one hour.")
    quote = "The pH of the syrups was measured"
    rephrased = ExtractedEntity(
        name="pH measurement",
        type="ATTRIBUTE",
        description="",
        source_chunk_id=passage.id,
        quote=quote,
        start_offset=0,
        end_offset=len(quote),
    )
    unanchored = rephrased.model_copy(
        update={"name": "density test", "quote": None, "start_offset": None}
    )

    result = filter_entities_with_decisions([rephrased, unanchored], {passage.id: passage})

    assert [entity.name for entity in result.kept] == ["pH measurement"]
    assert [decision.reason for decision in result.decisions] == [
        "grounded_at_extraction",
        "ungrounded_name",
    ]
