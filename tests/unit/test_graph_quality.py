from __future__ import annotations

from kg_processor.application.graph_quality import evaluate_graph_quality_rows
from kg_processor.domain.ontology import (
    EntityTypeDefinition,
    OntologyProfile,
    RelationTypeDefinition,
)


def test_graph_quality_passes_for_grounded_graph() -> None:
    """Confirm a grounded, dimensionally valid graph passes hard quality invariants.

    Warning-only fragmentation must not block persistence.
    """

    result = evaluate_graph_quality_rows(
        nodes=[
            {"id": "node_1", "embedding": [0.1, 0.2]},
            {"id": "node_2", "embedding": [0.3, 0.4]},
        ],
        edges=[
            {
                "id": "edge_1",
                "source_node_id": "node_1",
                "target_node_id": "node_2",
                "embedding": [0.5, 0.6],
            }
        ],
        evidence=[
            {"id": "ev_1", "subject_id": "node_1"},
            {"id": "ev_2", "subject_id": "node_2"},
            {"id": "ev_3", "subject_id": "edge_1"},
        ],
        expected_embedding_dimension=2,
    )

    assert result.ok
    check_results = {check.name: check.ok for check in result.checks}
    assert {
        name: check_results[name]
        for name in (
            "no_orphan_edges",
            "nodes_have_evidence",
            "edges_have_evidence",
            "node_embedding_dimensions",
            "edge_embedding_dimensions",
        )
    } == {
        "no_orphan_edges": True,
        "nodes_have_evidence": True,
        "edges_have_evidence": True,
        "node_embedding_dimensions": True,
        "edge_embedding_dimensions": True,
    }
    assert not next(check for check in result.checks if check.name == "component_ratio").ok


def test_graph_quality_reports_orphans_missing_evidence_and_embedding_mismatches() -> None:
    result = evaluate_graph_quality_rows(
        nodes=[{"id": "node_1", "embedding": [0.1]}],
        edges=[
            {
                "id": "edge_1",
                "source_node_id": "node_1",
                "target_node_id": "missing_node",
                "embedding": None,
            }
        ],
        evidence=[],
        expected_embedding_dimension=2,
    )

    checks = {check.name: check for check in result.checks}
    assert not result.ok
    assert checks["no_orphan_edges"].details == ["edge_1"]
    assert checks["nodes_have_evidence"].details == ["node_1"]
    assert checks["edges_have_evidence"].details == ["edge_1"]
    assert checks["node_embedding_dimensions"].details == ["node_1:1"]
    assert checks["edge_embedding_dimensions"].details == ["edge_1:missing"]


def test_graph_quality_blocks_self_loops_but_not_warning_only_diagnostics() -> None:
    """Distinguish blocking self-loops from non-blocking graph-shape diagnostics.

    Severity, rather than failure alone, controls the aggregate result.
    """

    result = evaluate_graph_quality_rows(
        nodes=[{"id": "node", "embedding": [0.1]}],
        edges=[
            {
                "id": "edge",
                "source_node_id": "node",
                "target_node_id": "node",
                "relation_type": "related_to",
                "embedding": [0.1],
            }
        ],
        evidence=[{"subject_id": "node"}, {"subject_id": "edge"}],
        expected_embedding_dimension=1,
    )

    assert not result.ok
    assert [check.name for check in result.failed_errors()] == ["no_unapproved_self_loops"]


def test_hybrid_ontology_allows_grounded_open_predicates_but_enforces_known_rules() -> None:
    """Keep hybrid extraction and final publication semantics consistent."""

    ontology = OntologyProfile(
        name="hybrid",
        description="Preferred vocabulary with grounded extensions.",
        mode="hybrid",
        entity_types=[
            EntityTypeDefinition(name="PERSON", description="A person."),
            EntityTypeDefinition(name="LOCATION", description="A place."),
        ],
        relation_types=[
            RelationTypeDefinition(
                name="LOCATED_IN",
                description="A person is located in a place.",
                source_types=["PERSON"],
                target_types=["LOCATION"],
            )
        ],
    )
    nodes = [
        {"id": "person", "primary_type": "PERSON", "embedding": [0.1]},
        {"id": "place", "primary_type": "LOCATION", "embedding": [0.2]},
    ]
    evidence = [
        {"subject_id": "person"},
        {"subject_id": "place"},
        {"subject_id": "open-edge"},
    ]
    open_result = evaluate_graph_quality_rows(
        nodes,
        [
            {
                "id": "open-edge",
                "source_node_id": "person",
                "target_node_id": "place",
                "relation_type": "TRAVELLED_TO",
                "embedding": [0.3],
            }
        ],
        evidence,
        1,
        ontology=ontology,
    )

    assert next(check for check in open_result.checks if check.name == "ontology_constraints").ok

    invalid_result = evaluate_graph_quality_rows(
        nodes,
        [
            {
                "id": "open-edge",
                "source_node_id": "place",
                "target_node_id": "person",
                "relation_type": "LOCATED_IN",
                "embedding": [0.3],
            }
        ],
        evidence,
        1,
        ontology=ontology,
    )
    ontology_check = next(
        check for check in invalid_result.checks if check.name == "ontology_constraints"
    )

    assert not ontology_check.ok
    assert ontology_check.count == 2
