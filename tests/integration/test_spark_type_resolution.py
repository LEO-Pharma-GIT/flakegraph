"""Spark finalization admits held relations exactly as local finalization does."""

# PySpark is an optional dependency, so its imports stay inside the guarded
# contract that only runs when the distributed extra and a JVM are installed.
# ruff: noqa: PLC0415

from __future__ import annotations

import os
import subprocess
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest
import type_resolution_fixture as fixture

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


@pytest.mark.parametrize("place_documents", [2, 1])
def test_spark_admits_the_held_relations_local_finalization_admits(
    tmp_path: Path, place_documents: int
) -> None:
    """Same shards, same edges, evidence methods, rejected rows, and counts."""

    from kg_processor.application.spark_finalization import (
        SparkFinalizationRequest,
        SparkGraphFinalizer,
    )

    shards = fixture.shards(place_documents=place_documents)
    local = build_pipeline(fixture.settings(tmp_path / "local")).finalize_document_shards(
        shards, write=False
    )

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

    assert _edges(spark) == _edges(local)
    assert _edge_evidence(spark) == _edge_evidence(local)
    assert sorted(row.id for row in spark.rejected_records) == sorted(
        row.id for row in local.rejected_records
    )
    assert manifest.metrics["rejected_records"] == local.graph_metrics["rejected_records"]
    assert manifest.metrics["type_resolution"] == local.graph_metrics["type_resolution"]
    assert manifest.metrics["type_resolution"] == {
        "held_relations": 2,
        "type_resolved_relations": 1 if place_documents == 2 else 0,
    }


def _edges(batch: GraphWriteBatch) -> list[tuple[Any, ...]]:
    return sorted(
        (
            edge.id,
            edge.source_node_id,
            edge.target_node_id,
            edge.relation_type,
            edge.evidence_count,
            round(edge.weight, 6),
        )
        for edge in batch.edges
    )


def _edge_evidence(batch: GraphWriteBatch) -> list[tuple[Any, ...]]:
    return sorted(
        (item.id, item.subject_id, item.method, item.quote, item.start_offset, item.end_offset)
        for item in batch.evidence
        if item.subject_kind == "edge"
    )
