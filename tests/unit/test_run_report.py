from __future__ import annotations

from documents import chunk, grounded

from kg_processor.application.graph_filter import EntityFilterResult, RelationFilterResult
from kg_processor.application.graph_merge import assemble_graph_with_decisions
from kg_processor.application.graph_quality import evaluate_graph_quality
from kg_processor.application.run_report import (
    RunProviders,
    RunReportRequest,
    build_run_report_artifacts,
)
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.graph import (
    Community,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)


def test_build_run_report_artifacts_counts_providers_cache_and_quality() -> None:
    passage = chunk("Alice works at Acme.", graph_id="graph", embedding=[0.1, 0.2])
    alice = ExtractedEntity(
        name="Alice",
        type="PERSON",
        description="Person",
        source_chunk_id=passage.id,
        **grounded(passage.content, "Alice"),
    )
    acme = ExtractedEntity(
        name="Acme",
        type="ORGANIZATION",
        description="Company",
        source_chunk_id=passage.id,
        **grounded(passage.content, "Acme"),
    )
    relation = ExtractedRelation(
        source_name="Alice",
        target_name="Acme",
        source_type="PERSON",
        target_type="ORGANIZATION",
        relation_type="works at",
        description="Alice works at Acme.",
        source_chunk_id=passage.id,
        **grounded(passage.content, "Alice works at Acme."),
    )
    extraction = ExtractionResult(
        entities=[alice, acme],
        relations=[relation],
        provider_metadata={
            "strategy": "two_pass",
            "chunk_count": 2,
            "batch_count": 2,
            "window_count": 2,
            "entity_mentions": 2,
            "relation_observations": 1,
            "trace": [
                {"stage": "entity_extraction", "record_actions": {"accepted": 2}},
                {
                    "stage": "relation_extraction",
                    "input_records": 2,
                    "record_actions": {"accepted": 1},
                },
                {"stage": "relation_extraction", "input_records": 0},
                {"stage": "relation_verification", "input_records": 1, "supported_records": 1},
            ],
        },
    )
    assembly = assemble_graph_with_decisions(
        "graph",
        [passage],
        extraction.entities,
        extraction.relations,
        relation_weight_max=10.0,
    )
    for node in assembly.nodes:
        node.embedding = [0.1, 0.2]
    for edge in assembly.edges:
        edge.embedding = [0.3, 0.4]
    community = Community(
        id="community_1",
        graph_id="graph",
        stable_key="node_a|node_b",
        level=0,
        title="Work",
        summary="Alice and Acme",
        rating=5.0,
        member_node_ids=[node.id for node in assembly.nodes],
    )
    quality = evaluate_graph_quality(
        assembly.nodes,
        assembly.edges,
        assembly.evidence,
        expected_embedding_dimension=2,
    )

    artifacts = build_run_report_artifacts(
        RunReportRequest(
            job_id="job",
            graph_id="graph",
            write_scope="graph_snapshot",
            file_ids=["file_1"],
            files_seen=1,
            documents_processed=1,
            block_rows=[{"id": "block_1"}],
            asset_rows=[{"id": "asset_1"}],
            chunks=[passage],
            extraction=extraction,
            entities=extraction.entities,
            relations=extraction.relations,
            entity_filter=EntityFilterResult(kept=extraction.entities, decisions=[]),
            relation_filter=RelationFilterResult(kept=extraction.relations, decisions=[]),
            assembly=assembly,
            descriptions_merged=1,
            communities=[community],
            findings=[],
            ocr_cache_hits=1,
            extraction_cache_hit=True,
            providers=RunProviders(
                ocr="builtin_text",
                llm="fake",
                embedding="hash",
                writer="local_artifacts",
                cache="local",
            ),
            embedding_dimension=2,
            graph_settings=GraphSettings(),
            quality_result=quality,
        )
    )

    assert artifacts.run_report["run_id"].startswith("run_")
    assert artifacts.run_report["chunks_created"] == 1
    assert artifacts.run_report["blocks_created"] == 1
    assert artifacts.run_report["assets_created"] == 1
    assert artifacts.run_report["nodes_created"] == 2
    assert artifacts.run_report["edges_created"] == 1
    assert artifacts.graph_metrics["providers"] == {
        "ocr": "builtin_text",
        "llm": "fake",
        "embedding": "hash",
        "writer": "local_artifacts",
        "cache": "local",
    }
    assert artifacts.graph_metrics["cache"] == {
        "ocr_hits": 1,
        "ocr_misses": 0,
        "extraction_hit": True,
    }
    assert artifacts.graph_metrics["extraction"] == {
        "strategy": "two_pass",
        "chunk_count": 2,
        "batch_count": 2,
        "window_count": 2,
        "entity_calls": 1,
        "relation_calls": 2,
        "verification_calls": 1,
        "entity_mentions": 2,
        "relation_observations": 1,
        "relation_candidates": 2,
        "empty_relation_calls": 1,
        "verification": {"candidates": 1, "supported": 1},
        "record_actions": {"accepted": 3},
    }
    assert artifacts.graph_metrics["quality"]["ok"] is True
