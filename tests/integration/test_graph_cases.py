from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from kg_processor.application.inspect import compare_local_graph_artifacts, inspect_local_graph
from kg_processor.application.pipeline import KgProcessorPipeline
from kg_processor.config.settings import Settings
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.factories import (
    build_cache,
    build_embedding_provider,
    build_file_source,
    build_llm_provider,
    build_ocr_provider,
    build_writer,
)

_GOLDEN_FIXTURE = Path("tests/fixtures/graph_cases/repeated_relation.json")


def test_repeated_relation_graph_regression(tmp_path: Path) -> None:
    fixture = json.loads(_GOLDEN_FIXTURE.read_text(encoding="utf-8"))
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / fixture["input"]["filename"]).write_text(
        fixture["input"]["text"],
        encoding="utf-8",
    )

    first_output = tmp_path / "out-first"
    second_output = tmp_path / "out-second"
    first = _run_fixture(fixture, input_dir, first_output)
    second = _run_fixture(fixture, input_dir, second_output)

    _assert_golden_batch(first, fixture["expected"])
    _assert_golden_batch(second, fixture["expected"])
    inspection = inspect_local_graph(first_output)
    assert inspection["quality"]["ok"]
    assert inspection["schema"]["ok"]
    comparison = compare_local_graph_artifacts(first_output, second_output)
    assert comparison["ok"]
    assert {check["name"]: check["ok"] for check in comparison["checks"]}["edges_stable_identity"]


def _run_fixture(
    fixture: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
) -> GraphWriteBatch:
    """Execute one isolated golden fixture run through the complete local pipeline.

    Fresh settings and output paths ensure the parity assertion cannot be satisfied
    by shared cache or artifact state.
    """

    settings = Settings.load(
        overrides={
            "job": {"job_id": "graphrag-golden", "graph_id": "graphrag-golden"},
            "files": {"input_path": input_dir, "location_page": False},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 8},
            "graph": {
                "chunk_token_size": fixture["settings"]["chunk_token_size"],
                "chunk_token_overlap": fixture["settings"]["chunk_token_overlap"],
                "gleaning_max_passes": 0,
            },
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )
    return KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
    ).run()


def _assert_golden_batch(batch: GraphWriteBatch, expected: dict[str, Any]) -> None:
    assert batch.run_report["run_id"] == expected["run_id"]
    assert batch.run_report["ordered_chunk_hash"] == expected["ordered_chunk_hash"]
    for key, value in expected["counts"].items():
        assert batch.run_report[key] == value
        assert batch.graph_metrics["counts"].get(_metric_count_key(key), value) == value

    nodes_by_name = {node.name: node for node in batch.nodes}
    assert sorted(nodes_by_name) == sorted(expected["nodes"])
    entity_sources_by_node_id = {source.node_id: source for source in batch.entity_sources}
    for name, node_expectation in expected["nodes"].items():
        node = nodes_by_name[name]
        assert node.primary_type == node_expectation["primary_type"]
        assert len(node.source_chunk_ids) == node_expectation["source_chunk_count"]
        assert (
            entity_sources_by_node_id[node.id].mention_count
            == node_expectation["entity_source_mention_count"]
        )

    assert len(batch.edges) == 1
    edge = batch.edges[0]
    edge_expectation = expected["edge"]
    assert nodes_by_name[edge_expectation["source"]].id == edge.source_node_id
    assert nodes_by_name[edge_expectation["target"]].id == edge.target_node_id
    assert edge.relation_type == edge_expectation["relation_type"]
    assert edge.weight == edge_expectation["weight"]
    assert len(edge.source_chunk_ids) == edge_expectation["source_chunk_count"]

    evidence_expectation = expected["evidence"]
    evidence_kinds = Counter(evidence.subject_kind for evidence in batch.evidence)
    assert len(batch.evidence) == evidence_expectation["total"]
    assert evidence_kinds["node"] == evidence_expectation["node"]
    assert evidence_kinds["edge"] == evidence_expectation["edge"]
    assert {evidence.quote for evidence in batch.evidence} == set(evidence_expectation["quotes"])

    assert batch.graph_metrics["merge"]["decision_actions"] == expected["merge_decision_actions"]
    assert batch.graph_metrics["merge"]["decision_reasons"] == expected["merge_decision_reasons"]
    assert batch.graph_metrics["dedupe"] == expected["dedupe"]
    assert not batch.graph_metrics["filtering"]["dropped_entities_by_reason"]
    assert not batch.graph_metrics["filtering"]["dropped_relations_by_reason"]

    assert len(batch.communities) == 1
    member_names = sorted(
        node.name for node in batch.nodes if node.id in set(batch.communities[0].member_node_ids)
    )
    assert member_names == expected["community"]["member_names"]
    assert len(batch.community_findings) == expected["community"]["finding_count"]


def _metric_count_key(run_report_key: str) -> str:
    if run_report_key.endswith("_created"):
        return run_report_key.removesuffix("_created")
    return run_report_key
