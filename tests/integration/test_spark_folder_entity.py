"""Spark finalization ties documents to their folders exactly as local finalization does."""

# PySpark is an optional dependency, so its imports stay inside the guarded
# contract that only runs when the distributed extra and a JVM are installed.
# ruff: noqa: PLC0415

from __future__ import annotations

import os
import subprocess
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import folder_entity_fixture as fixture
import pytest

from kg_processor.adapters.distributed.local_blob import LocalBlobStore
from kg_processor.application.graph_dataset import GraphDatasetReader
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.factories import build_pipeline


def _spark_runtime_available() -> bool:
    """Require both optional Python packages and a working Java runtime."""

    try:
        java = subprocess.run(
            ["java", "-version"], check=False, capture_output=True, timeout=5
        ).returncode
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return java == 0 and find_spec("pyspark") is not None and find_spec("graphframes") is not None


pytestmark = [
    pytest.mark.spark,
    pytest.mark.skipif(
        os.getenv("KG_RUN_SPARK_INTEGRATION") != "1" or not _spark_runtime_available(),
        reason=(
            "Set KG_RUN_SPARK_INTEGRATION=1 with the distributed-spark extra and a Java 17+ "
            "runtime installed to run Spark integration checks."
        ),
    ),
]


@pytest.mark.parametrize(
    ("folder", "graph"),
    [
        ({}, {}),
        ({"excluded_file_ids": ["file-c", "file-e"]}, {}),
        (
            {
                "excluded_paths": [
                    "Picked Root/Northwind Pricing Study/report.pdf",
                    "Picked Root/Contract ACME-0042/annex.pdf",
                ]
            },
            {},
        ),
        ({}, {"drop_isolated_entities": True}),
    ],
    ids=["all", "excluded-by-id", "excluded-by-path", "isolates-dropped"],
)
def test_spark_links_every_entity_to_its_folder_like_local_finalization(
    tmp_path: Path, folder: dict[str, Any], graph: dict[str, Any]
) -> None:
    """Same shards, same nodes, edges, evidence, and no community around a folder.

    Two folders of one name at the level are one study.
    """

    from kg_processor.application.spark_finalization import (
        SparkFinalizationRequest,
        SparkGraphFinalizer,
    )

    ontology = {**fixture.profile(**folder), "mode": "closed"}
    shards = fixture.shards()
    local = build_pipeline(
        fixture.settings(tmp_path / "local", ontology, graph=graph)
    ).finalize_document_shards(shards, write=False)

    root = tmp_path / "artifacts"
    store = LocalBlobStore(root.as_uri())
    store.initialize()
    artifact_ids: set[str] = set()
    for index, shard in enumerate(shards):
        for kind, payload in (("prepared", shard.prepared), ("extracted", shard)):
            name = f"{kind}-{index}"
            store.put(
                f"run/{kind}_document/{name}.json",
                payload.model_dump_json().encode(),
                "application/json",
            )
            artifact_ids.add(name)
    settings = fixture.settings(
        tmp_path / "spark",
        ontology,
        graph=graph,
        runtime={"runtime": "kubernetes"},
        distributed={
            "artifact_uri": root.as_uri(),
            "finalization_engine": "spark",
            "spark_master": "local[2]",
            "spark_executor_instances": 2,
            "spark_executor_cores": 1,
            "spark_executor_memory": "1g",
        },
    )
    manifest = SparkGraphFinalizer(settings).finalize(
        SparkFinalizationRequest(
            run_id="run",
            graph_id=fixture.GRAPH_ID,
            attempt=1,
            artifact_ids=frozenset(artifact_ids),
        )
    )
    spark = GraphDatasetReader(store).read(manifest)

    assert _nodes(spark) == _nodes(local)
    assert _edges(spark) == _edges(local)
    assert _evidence(spark) == _evidence(local)
    assert _observations(spark) == _observations(local)
    assert manifest.metrics["hard_quality_error_count"] == 0
    folders = {node.id for node in local.nodes if node.primary_type == "STUDY"}
    for batch in (local, spark):
        assert not folders & {
            member for community in batch.communities for member in community.member_node_ids
        }
    assert any(edge.relation_type == "appears_in" for edge in local.edges)
    if folder.get("excluded_file_ids"):
        assert len(folders) == 1
    # One study for both folders of that name, whichever document it is read from.
    assert sum(node.name == "Northwind Pricing Study" for node in local.nodes) == 1
    if folder.get("excluded_paths"):
        # The contract keeps its folder through its terms only: the annex,
        # under a location with an empty folder, is excluded by its path.
        contract = next(node for node in local.nodes if node.name == "Contract ACME-0042")
        assert len(contract.source_chunk_ids) == 1
    # Only its folder link reaches Initech: an isolate the filter drops, link and all.
    node_names = {node.name for node in local.nodes}
    assert ("Initech" in node_names) is not graph.get("drop_isolated_entities", False)


def _observations(batch: GraphWriteBatch) -> list[tuple[Any, ...]]:
    return sorted(
        (item.edge_id, item.chunk_id, item.proposed_relation_type)
        for item in batch.edge_observations
    )


def _nodes(batch: GraphWriteBatch) -> list[tuple[Any, ...]]:
    return sorted(
        (node.id, node.name, node.primary_type, node.degree, tuple(sorted(node.source_chunk_ids)))
        for node in batch.nodes
    )


def _edges(batch: GraphWriteBatch) -> list[tuple[Any, ...]]:
    return sorted(
        (
            edge.id,
            edge.source_node_id,
            edge.target_node_id,
            edge.relation_type,
            edge.description,
            edge.evidence_count,
            round(edge.weight, 6),
            tuple(sorted(edge.source_file_ids)),
        )
        for edge in batch.edges
    )


def _evidence(batch: GraphWriteBatch) -> list[tuple[Any, ...]]:
    return sorted(
        (
            item.id,
            item.subject_id,
            item.subject_kind,
            item.method,
            item.quote,
            item.start_offset,
            item.end_offset,
        )
        for item in batch.evidence
    )
