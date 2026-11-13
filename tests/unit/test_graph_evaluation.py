from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kg_processor.adapters.writers.local_artifacts import LocalArtifactsWriter
from kg_processor.application.graph_evaluation import (
    GoldDocument,
    GoldEntity,
    GoldEvidenceObservation,
    GoldRelation,
    _entity_metrics,
    _evidence_metrics,
    _missing_triple_reasons,
    _two_hop_recoverability,
    evaluate_graph_artifacts,
    load_gold_graph,
)
from kg_processor.domain.graph import Chunk, Evidence, GraphEdge, GraphNode, GraphWriteBatch


def test_missing_triple_reasons_partition_every_recall_failure() -> None:
    """Separate upstream endpoint loss from direction, predicate, and edge misses."""

    missing = {
        ("missing-a", "related_to", "missing-b"),
        ("missing-source", "related_to", "target"),
        ("source", "related_to", "missing-target"),
        ("source", "reversed_relation", "target"),
        ("source", "expected_relation", "target"),
        ("source", "absent_relation", "other"),
    }
    observed = {
        ("target", "reversed_relation", "source"),
        ("source", "different_relation", "target"),
    }

    reasons = _missing_triple_reasons(
        missing,
        observed,
        matched_gold_ids={"source", "target", "other"},
    )

    assert reasons == {
        "both_endpoints_missing": 1,
        "source_endpoint_missing": 1,
        "target_endpoint_missing": 1,
        "reversed": 1,
        "wrong_predicate": 1,
        "no_edge": 1,
    }


def test_public_graph_evaluation_scores_written_artifacts(tmp_path: Path) -> None:
    """Exercise public artifact evaluation across entities, triples, evidence, and structure.

    A perfect controlled snapshot should pass every threshold.
    """

    output = tmp_path / "graph"
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps(_gold_payload()), encoding="utf-8")
    LocalArtifactsWriter(output).write(_evaluation_batch())

    result = evaluate_graph_artifacts(output, gold)

    assert result["ok"]
    assert result["entities"]["f1"] == 1.0
    assert result["triples"]["f1"] == 1.0
    assert result["evidence"]["support_rate"] == 1.0
    assert result["two_hop_recoverability"] == 1.0
    assert result["quality"]["ok"]


def test_graph_evaluation_names_the_gold_relations_it_misses(tmp_path: Path) -> None:
    """A missed triple is reported by its gold relation id as well as its text."""

    output = tmp_path / "graph"
    gold = tmp_path / "gold.json"
    payload = _gold_payload()
    for index, relation in enumerate(payload["relations"]):
        relation["id"] = f"expected-relation-{index}"
    gold.write_text(json.dumps(payload), encoding="utf-8")
    batch = _evaluation_batch()
    LocalArtifactsWriter(output).write(batch)
    assert evaluate_graph_artifacts(output, gold)["triples"]["missing_relation_ids"] == []

    without_edges = tmp_path / "without-edges"
    LocalArtifactsWriter(without_edges).write(
        batch.model_copy(
            update={
                "edges": [],
                "evidence": [item for item in batch.evidence if item.subject_kind != "edge"],
            }
        )
    )
    result = evaluate_graph_artifacts(without_edges, gold)
    expected = sorted(
        relation.id for relation in load_gold_graph(gold).relations if relation.id is not None
    )
    assert expected and result["triples"]["missing_relation_ids"] == expected


def test_public_graph_evaluation_rejects_wrong_relation_evidence(tmp_path: Path) -> None:
    """Make evidence alignment part of acceptance rather than a display-only metric."""

    output = tmp_path / "graph"
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps(_gold_payload()), encoding="utf-8")
    batch = _evaluation_batch()
    evidence = [
        item.model_copy(update={"quote": "Alice", "start_offset": 0, "end_offset": 5})
        if item.subject_kind == "edge"
        else item
        for item in batch.evidence
    ]
    LocalArtifactsWriter(output).write(batch.model_copy(update={"evidence": evidence}))

    result = evaluate_graph_artifacts(output, gold)

    assert result["evidence"]["support_rate"] == 0.0
    assert result["acceptance"]["evidence_support"] is False
    assert result["ok"] is False


def test_entity_evaluation_matches_persisted_observed_aliases() -> None:
    """Decouple benchmark identity from one resolution-selected display spelling."""

    matches, metrics = _entity_metrics(
        [
            {
                "id": "actual-model",
                "name": "Recurrent sequence model",
                "primary_type": "MODEL",
                "types": ["MODEL"],
                "aliases": ["Long Short-Term Memory", "LSTM"],
            }
        ],
        [GoldEntity(id="gold-model", name="Long Short-Term Memory", type="MODEL")],
    )

    assert matches == {"actual-model": "gold-model"}
    assert metrics["recall"] == 1.0


def test_entity_evaluation_folds_unicode_compatibility_ligatures() -> None:
    """Avoid counting PDF typography as an entity-recall failure."""

    matches, metrics = _entity_metrics(
        [
            {
                "id": "actual-model",
                "name": "uni\ufb01ed neural architecture",
                "primary_type": "MODEL",
                "types": ["MODEL"],
                "aliases": [],
            }
        ],
        [GoldEntity(id="gold-model", name="unified neural architecture", type="MODEL")],
    )

    assert matches == {"actual-model": "gold-model"}
    assert metrics["recall"] == 1.0


def test_reference_scope_ignores_unannotated_entities_but_scores_induced_edges(
    tmp_path: Path,
) -> None:
    """Avoid false positives from a partial gold graph without hiding in-scope errors."""

    output = tmp_path / "graph"
    gold = tmp_path / "gold.json"
    payload = _gold_payload()
    payload["evaluation_scope"] = {
        "entity_coverage": "reference",
        "relation_coverage": "induced",
    }
    payload["relations"].append(
        {
            "source": "alice",
            "target": "acme",
            "relation_type": "OWNS",
            "required": False,
        }
    )
    gold.write_text(json.dumps(payload), encoding="utf-8")
    batch = _evaluation_batch()
    extra_node = _node("paris-node", "Paris", "LOCATION", "chunk")
    extra_edges = [
        _edge("extra-induced", "alice-node", "acme-node", "owns"),
        _edge("extra-unexpected", "alice-node", "acme-node", "manages"),
        _edge("extra-unscored", "acme-node", "paris-node", "located_in"),
    ]
    LocalArtifactsWriter(output).write(
        batch.model_copy(
            update={
                "nodes": [*batch.nodes, extra_node],
                "edges": [*batch.edges, *extra_edges],
            }
        )
    )

    result = evaluate_graph_artifacts(output, gold)

    assert result["entities"]["precision"] is None
    assert result["entities"]["precision_applicable"] is False
    assert result["entities"]["observed_actual"] == 3
    assert result["entities"]["unscored"] == ["Paris"]
    assert result["triples"]["precision"] is None
    assert result["triples"]["precision_applicable"] is False
    assert result["triples"]["recall"] == 1.0
    assert result["triples"]["observed_actual"] == 4
    assert result["triples"]["unscored_count"] == 1


def test_reference_relation_scope_ignores_unlisted_relations_between_gold_entities(
    tmp_path: Path,
) -> None:
    """Score selected required facts without claiming an exhaustive induced subgraph."""

    output = tmp_path / "graph"
    gold = tmp_path / "gold.json"
    payload = _gold_payload()
    payload["evaluation_scope"] = {
        "entity_coverage": "reference",
        "relation_coverage": "reference",
    }
    gold.write_text(json.dumps(payload), encoding="utf-8")
    batch = _evaluation_batch()
    extra_edge = _edge("extra-reference", "alice-node", "acme-node", "manages")
    LocalArtifactsWriter(output).write(
        batch.model_copy(update={"edges": [*batch.edges, extra_edge]})
    )

    result = evaluate_graph_artifacts(output, gold)

    assert result["triples"]["precision"] is None
    assert result["triples"]["precision_applicable"] is False
    assert result["triples"]["recall"] == 1.0
    assert result["triples"]["actual"] == 1
    assert result["triples"]["unexpected"] == []
    assert result["triples"]["unscored_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["entities"].append(
                {"id": "alice", "name": "Alice Smith", "type": "PERSON"}
            ),
            "gold entity IDs must be unique",
        ),
        (
            lambda payload: payload["entities"].append(
                {"id": "other", "name": "Alice", "type": "PERSON"}
            ),
            "gold surface",
        ),
        (
            lambda payload: payload["relations"].append(dict(payload["relations"][0])),
            "gold directed relation triples must be unique",
        ),
    ],
)
def test_gold_graph_rejects_ambiguous_canonical_contracts(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Reject fixture mistakes that would silently distort precision or recall."""

    payload = _gold_payload()
    mutation(payload)
    path = tmp_path / "invalid-gold.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        load_gold_graph(path)


def test_gold_observations_are_document_bound_and_substring_grounded(tmp_path: Path) -> None:
    """Validate exact source observations without involving provider output."""

    payload = _gold_payload()
    payload["documents"] = [{"id": "source", "path": "source.txt"}]
    payload["relations"][0]["observations"] = [
        {
            "document": "source",
            "sentence": "Alice works at Acme.",
            "evidence_contains": "works at",
        }
    ]
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    graph = load_gold_graph(path)

    assert graph.relations[0].observations[0].document == "source"

    payload["relations"][0]["observations"][0]["evidence_contains"] = "employed by"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="must occur"):
        load_gold_graph(path)


def test_gold_graph_allows_same_surface_for_distinct_entity_types(tmp_path: Path) -> None:
    """Distinguish a publication from a model that intentionally shares its title."""

    payload = _gold_payload()
    payload["entities"].extend(
        [
            {"id": "shared-paper", "name": "Shared Name", "type": "PAPER"},
            {"id": "shared-model", "name": "Shared Name", "type": "MODEL"},
        ]
    )
    path = tmp_path / "typed-gold.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    graph = load_gold_graph(path)

    assert {entity.id for entity in graph.entities if entity.name == "Shared Name"} == {
        "shared-paper",
        "shared-model",
    }


def test_two_hop_recoverability_uses_non_gold_intermediate_nodes() -> None:
    """Credit useful short paths even when their intermediate entity is not annotated."""

    edges = [
        {"source_node_id": "actual-a", "target_node_id": "intermediate"},
        {"source_node_id": "intermediate", "target_node_id": "actual-b"},
    ]
    relation = GoldRelation(source="gold-a", target="gold-b", relation_type="RELATED_TO")

    score = _two_hop_recoverability(
        edges,
        {"actual-a": "gold-a", "actual-b": "gold-b"},
        [relation],
    )

    assert score == 1.0


def test_evidence_support_rate_counts_each_gold_relation_once() -> None:
    """Duplicate actual edges cannot push gold evidence support above one."""

    relation = GoldRelation(
        source="alice",
        target="acme",
        relation_type="WORKS_AT",
        observations=[
            GoldEvidenceObservation(
                document="gold-document",
                sentence="Alice works at Acme.",
                evidence_contains="works at",
            )
        ],
    )
    edges = [
        {
            "id": edge_id,
            "source_node_id": "alice-node",
            "target_node_id": "acme-node",
            "relation_type": "WORKS_AT",
        }
        for edge_id in ("edge-1", "edge-2")
    ]
    evidence = [
        {"subject_id": edge_id, "file_id": "file", "quote": "Alice works at Acme."}
        for edge_id in ("edge-1", "edge-2")
    ]

    metrics = _evidence_metrics(
        edges,
        evidence,
        {"alice-node": "alice", "acme-node": "acme"},
        [relation],
        documents=[{"file_id": "file", "checksum": "checksum"}],
        gold_documents=[GoldDocument(id="gold-document", path="paper.pdf", checksum="checksum")],
    )

    assert metrics["supported"] == 1
    assert metrics["support_rate"] == 1.0


def test_evidence_support_requires_annotated_phrase_in_reviewed_document() -> None:
    """Reject both an unrelated source and unrelated wording from the right source."""

    relation = GoldRelation(
        source="alice",
        target="acme",
        relation_type="WORKS_AT",
        evidence_contains=["works at"],
        observations=[
            GoldEvidenceObservation(
                document="gold-document",
                page_number=2,
                sentence="Alice works at Acme.",
                evidence_contains="works at",
            )
        ],
    )
    edges = [
        {
            "id": "edge",
            "source_node_id": "alice-node",
            "target_node_id": "acme-node",
            "relation_type": "WORKS_AT",
        }
    ]
    node_matches = {"alice-node": "alice", "acme-node": "acme"}
    gold_documents = [
        GoldDocument(id="gold-document", path="paper.pdf", checksum="expected-checksum")
    ]
    documents = [
        {"file_id": "right-file", "checksum": "expected-checksum"},
        {"file_id": "wrong-file", "checksum": "different-checksum"},
    ]
    wrong_document = _evidence_metrics(
        edges,
        [{"subject_id": "edge", "file_id": "wrong-file", "page_number": 2, "quote": "works at"}],
        node_matches,
        [relation],
        documents=documents,
        gold_documents=gold_documents,
    )
    unrelated_passage = _evidence_metrics(
        edges,
        [
            {
                "file_id": "right-file",
                "page_number": 1,
                "quote": "Alice is employed by Acme.",
            }
        ],
        node_matches,
        [relation],
        documents=documents,
        gold_documents=gold_documents,
    )
    blank = _evidence_metrics(
        edges,
        [{"subject_id": "edge", "file_id": "right-file", "page_number": 2, "quote": ""}],
        node_matches,
        [relation],
        documents=documents,
        gold_documents=gold_documents,
    )
    supported = _evidence_metrics(
        edges,
        [
            {
                "subject_id": "edge",
                "file_id": "right-file",
                "page_number": 1,
                "quote": "The directory confirms that Alice works at Acme.",
            }
        ],
        node_matches,
        [relation],
        documents=documents,
        gold_documents=gold_documents,
    )

    assert wrong_document["supported"] == 0
    assert unrelated_passage["supported"] == 0
    assert blank["supported"] == 0
    assert supported["supported"] == 1


def test_evidence_support_scores_observation_only_gold_relations() -> None:
    relation = GoldRelation(
        source="alice",
        target="acme",
        relation_type="WORKS_AT",
        observations=[
            GoldEvidenceObservation(
                document="gold-document",
                page_number=1,
                sentence="Alice works at Acme.",
                evidence_contains="works at",
            )
        ],
    )
    metrics = _evidence_metrics(
        [
            {
                "id": "edge",
                "source_node_id": "alice-node",
                "target_node_id": "acme-node",
                "relation_type": "WORKS_AT",
            }
        ],
        [],
        {"alice-node": "alice", "acme-node": "acme"},
        [relation],
        documents=[{"file_id": "file", "checksum": "checksum"}],
        gold_documents=[GoldDocument(id="gold-document", path="paper.pdf", checksum="checksum")],
    )

    assert metrics["gold_relations_with_phrases"] == 1
    assert metrics["supported"] == 0
    assert metrics["support_rate"] == 0.0


def test_evidence_support_tolerates_small_ocr_errors_but_not_paraphrases() -> None:
    """Recognize source transcription damage without weakening semantic evidence."""

    relation = GoldRelation(
        source="paper",
        target="theorem",
        relation_type="INTRODUCES",
        evidence_contains=["approximate continuous functions with arbitrary precision"],
        observations=[
            GoldEvidenceObservation(
                document="gold-document",
                page_number=6,
                sentence="Networks approximate continuous functions with arbitrary precision.",
                evidence_contains="approximate continuous functions with arbitrary precision",
            )
        ],
    )
    edges = [
        {
            "id": "edge",
            "source_node_id": "paper-node",
            "target_node_id": "theorem-node",
            "relation_type": "INTRODUCES",
        }
    ]
    node_matches = {"paper-node": "paper", "theorem-node": "theorem"}
    documents = [{"file_id": "right-file", "checksum": "expected-checksum"}]
    gold_documents = [
        GoldDocument(id="gold-document", path="paper.pdf", checksum="expected-checksum")
    ]

    ocr_transposition = _evidence_metrics(
        edges,
        [
            {
                "subject_id": "edge",
                "file_id": "right-file",
                "quote": "Networks approximate continuous functions wtih arbitrary precision.",
            }
        ],
        node_matches,
        [relation],
        documents=documents,
        gold_documents=gold_documents,
    )
    paraphrase = _evidence_metrics(
        edges,
        [
            {
                "subject_id": "edge",
                "file_id": "right-file",
                "quote": "The theorem can represent a broad family of smooth mappings.",
            }
        ],
        node_matches,
        [relation],
        documents=documents,
        gold_documents=gold_documents,
    )

    assert ocr_transposition["supported"] == 1
    assert paraphrase["supported"] == 0


def _gold_payload() -> dict[str, Any]:
    """Build a tiny gold graph with perfect expected evaluation thresholds.

    The fixture covers typed aliases, relations, and evidence.
    """

    return {
        "name": "employment",
        "description": "Tiny evaluation graph",
        "documents": [{"id": "gold-document", "path": "source.txt", "checksum": "checksum"}],
        "entities": [
            {"id": "alice", "name": "Alice", "type": "PERSON"},
            {"id": "acme", "name": "Acme", "type": "ORGANIZATION"},
        ],
        "relations": [
            {
                "source": "alice",
                "target": "acme",
                "relation_type": "WORKS_AT",
                "observations": [
                    {
                        "document": "gold-document",
                        "sentence": "Alice works at Acme.",
                        "evidence_contains": "works at",
                    }
                ],
            }
        ],
        "thresholds": {
            "entity_precision": 1.0,
            "entity_recall": 1.0,
            "triple_precision": 1.0,
            "triple_recall": 1.0,
            "min_evidence_support": 1.0,
            "max_isolate_ratio": 0.0,
            "min_two_hop_recoverability": 1.0,
        },
    }


def _evaluation_batch() -> GraphWriteBatch:
    """Build a complete writer batch that should exactly satisfy the tiny gold graph."""

    content = "Alice works at Acme."
    chunk = Chunk(
        id="chunk",
        graph_id="graph",
        file_id="file",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=4,
        content_hash="hash",
    )
    nodes = [
        _node("alice-node", "Alice", "PERSON", chunk.id),
        # The display type can differ from another valid merged type. Gold
        # matching must consider the complete canonical type set.
        _node("acme-node", "Acme", "SCHOOL", chunk.id).model_copy(
            update={"types": ["SCHOOL", "ORGANIZATION"]}
        ),
    ]
    edge = GraphEdge(
        id="edge",
        graph_id="graph",
        source_node_id="alice-node",
        target_node_id="acme-node",
        relation_type="works_at",
        description=content,
        weight=1.0,
        source_file_id="file",
        source_file_ids=["file"],
        source_chunk_ids=[chunk.id],
        evidence_count=1,
    )
    evidence = [
        _evidence("alice-evidence", "alice-node", "Alice", 0, 5, chunk.id),
        _evidence("acme-evidence", "acme-node", "Acme", 15, 19, chunk.id),
        _evidence("edge-evidence", edge.id, content, 0, len(content), chunk.id, "edge"),
    ]
    return GraphWriteBatch(
        graph_id="graph",
        documents=[{"id": "file", "file_id": "file", "checksum": "checksum"}],
        pages=[],
        chunks=[chunk],
        nodes=nodes,
        edges=[edge],
        evidence=evidence,
        entity_sources=[],
        communities=[],
        community_findings=[],
        run_report={"job_id": "job", "graph_id": "graph"},
    )


def _node(node_id: str, name: str, entity_type: str, chunk_id: str) -> GraphNode:
    """Create one canonical graph node with deterministic type and source provenance.

    Embeddings are intentionally unnecessary for this evaluation.
    """

    return GraphNode(
        id=node_id,
        graph_id="graph",
        normalized_name=name.casefold(),
        name=name,
        primary_type=entity_type,
        types=[entity_type],
        description=name,
        source_chunk_ids=[chunk_id],
    )


def _edge(edge_id: str, source: str, target: str, relation_type: str) -> GraphEdge:
    """Create a minimal graph edge used to exercise evaluation scope."""

    return GraphEdge(
        id=edge_id,
        graph_id="graph",
        source_node_id=source,
        target_node_id=target,
        relation_type=relation_type,
        description=relation_type,
        weight=1.0,
        source_file_id="file",
        source_file_ids=["file"],
        source_chunk_ids=["chunk"],
        evidence_count=1,
    )


def _evidence(
    evidence_id: str,
    subject_id: str,
    quote: str,
    start: int,
    end: int,
    chunk_id: str,
    subject_kind: str = "node",
) -> Evidence:
    """Create one exact evidence span for a controlled node or edge subject."""

    return Evidence(
        id=evidence_id,
        graph_id="graph",
        subject_id=subject_id,
        subject_kind=subject_kind,
        file_id="file",
        chunk_id=chunk_id,
        page_number=1,
        start_offset=start,
        end_offset=end,
        quote=quote,
    )
