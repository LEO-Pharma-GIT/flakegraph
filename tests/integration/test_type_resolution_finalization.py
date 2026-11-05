"""Local finalization admits a held relation on the types other documents give."""

from __future__ import annotations

from pathlib import Path

import type_resolution_fixture as fixture

from kg_processor.application.rejected_records import rejected_records_from_trace
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.factories import build_pipeline


def _finalize(tmp_path: Path, *, place_documents: int, enabled: bool = True) -> GraphWriteBatch:
    settings = fixture.settings(
        tmp_path, graph={"relation_type_resolution": enabled, "entity_resolution_enabled": False}
    )
    return build_pipeline(settings).finalize_document_shards(
        fixture.shards(place_documents=place_documents), write=False
    )


def _located_in(batch: GraphWriteBatch) -> list[tuple[str, str, str, str]]:
    """Each LOCATED_IN edge as source, target, target type, and evidence method."""

    nodes = {node.id: node for node in batch.nodes}
    methods = {item.subject_id: item.method for item in batch.evidence}
    return [
        (
            nodes[edge.source_node_id].name,
            nodes[edge.target_node_id].name,
            nodes[edge.target_node_id].primary_type,
            methods[edge.id],
        )
        for edge in batch.edges
        if edge.relation_type == "located_in"
    ]


def test_a_held_relation_is_admitted_when_two_documents_give_its_entity_the_allowed_type(
    tmp_path: Path,
) -> None:
    """Jordan is a place in two other documents, so Acme is located in that place.

    The admitted relation is labelled so it can be judged apart, and its
    rejected record is taken back: the rejected records are what stayed
    rejected, and the metrics say how many were held and how many admitted.
    """

    batch = _finalize(tmp_path, place_documents=2)

    assert _located_in(batch) == [("Acme", "Jordan", "PLACE", "type_resolved")]
    assert [(row.reason, row.target) for row in batch.rejected_records] == [
        ("domain_or_range_violation", "Bob")
    ]
    assert batch.graph_metrics["rejected_records"]["records"] == 1
    assert batch.graph_metrics["type_resolution"] == {
        "held_relations": 2,
        "type_resolved_relations": 1,
    }
    # Held relations are never counted among the relations extraction kept.
    assert batch.run_report["relations_extracted"] == 1


def test_one_document_is_not_enough_to_admit_a_held_relation(tmp_path: Path) -> None:
    """A single other reading could be that document's own mistake; nothing is lost."""

    batch = _finalize(tmp_path, place_documents=1)

    assert _located_in(batch) == []
    assert len(batch.rejected_records) == 2
    assert batch.graph_metrics["type_resolution"] == {
        "held_relations": 2,
        "type_resolved_relations": 0,
    }


def test_type_resolution_off_admits_nothing_and_keeps_every_record(tmp_path: Path) -> None:
    batch = _finalize(tmp_path, place_documents=2, enabled=False)

    assert _located_in(batch) == []
    trace = [
        event for shard in fixture.shards(place_documents=2) for event in shard.observations.trace
    ]
    chunks = [
        chunk for shard in fixture.shards(place_documents=2) for chunk in shard.prepared.chunks
    ]
    assert batch.rejected_records == rejected_records_from_trace(trace, chunks, fixture.GRAPH_ID)
    assert batch.graph_metrics["type_resolution"]["type_resolved_relations"] == 0
