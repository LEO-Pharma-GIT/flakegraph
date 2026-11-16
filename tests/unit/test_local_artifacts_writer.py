from __future__ import annotations

import json
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kg_processor.adapters.distributed.local_blob import LocalBlobStore
from kg_processor.adapters.writers import local_artifacts
from kg_processor.adapters.writers.local_artifacts import LocalArtifactsWriter
from kg_processor.application.graph_dataset import GraphDatasetReader
from kg_processor.domain.finalization import (
    DatasetFile,
    DatasetTableManifest,
    GraphDatasetManifest,
)
from kg_processor.domain.graph import (
    Chunk,
    Community,
    CommunityFinding,
    EntitySource,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphWriteBatch,
)


def test_local_artifacts_writer_replaces_complete_snapshots(tmp_path: Path) -> None:
    output_path = tmp_path / "out"
    writer = LocalArtifactsWriter(output_path)

    writer.write(_batch("first"))
    writer.write(_batch("second"))

    report = json.loads((output_path / "run_report.json").read_text(encoding="utf-8"))
    documents = pd.read_parquet(output_path / "documents.parquet")
    blocks = pd.read_parquet(output_path / "blocks.parquet")
    communities = pd.read_parquet(output_path / "communities.parquet")

    assert report["job_id"] == "second"
    assert documents.loc[0, "source_uri"] == "file:///second.txt"
    assert blocks.loc[0, "id"] == "block_1"
    assert communities.loc[0, "rating_explanation"] == "second rating explanation"
    suggested_questions = cast(Any, communities.loc[0, "suggested_questions"])
    assert list(suggested_questions) == ["What is second?"]
    assert not list(tmp_path.glob(".out.tmp-*"))
    assert not list(tmp_path.glob(".out.previous-*"))


def test_local_artifacts_writer_keeps_document_and_page_schema_when_empty(tmp_path: Path) -> None:
    output_path = tmp_path / "out"
    LocalArtifactsWriter(output_path).write(
        _batch("empty").model_copy(update={"documents": [], "pages": []})
    )

    assert list(pd.read_parquet(output_path / "documents.parquet").columns) == [
        "id",
        "graph_id",
        "file_id",
        "checksum",
        "source_uri",
        "mime_type",
        "size_bytes",
        "ocr_provider",
        "kind",
        "summary",
        "related_file_ids",
        "folder",
        "provenance",
    ]
    assert list(pd.read_parquet(output_path / "pages.parquet").columns) == [
        "id",
        "graph_id",
        "file_id",
        "page_number",
        "markdown",
        "raw_text",
        "detected_language",
    ]


def test_local_artifacts_writer_keeps_previous_snapshot_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "out"
    writer = LocalArtifactsWriter(output_path)
    writer.write(_batch("first"))
    original_write_parquet = local_artifacts._write_parquet

    def fail_on_chunks(
        path: Path,
        rows: list[dict[str, object]],
        columns: list[str] | None = None,
    ) -> None:
        if path.name == "chunks.parquet":
            raise RuntimeError("simulated chunk write failure")
        original_write_parquet(path, rows, columns)

    monkeypatch.setattr(local_artifacts, "_write_parquet", fail_on_chunks)

    with pytest.raises(RuntimeError, match="simulated chunk write failure"):
        writer.write(_batch("second"))

    report = json.loads((output_path / "run_report.json").read_text(encoding="utf-8"))
    documents = pd.read_parquet(output_path / "documents.parquet")
    assets = pd.read_parquet(output_path / "assets.parquet")

    assert report["job_id"] == "first"
    assert documents.loc[0, "source_uri"] == "file:///first.txt"
    assert assets.loc[0, "id"] == "asset_1"
    assert not list(tmp_path.glob(".out.tmp-*"))
    assert not list(tmp_path.glob(".out.previous-*"))


def test_local_artifacts_writer_redacts_sensitive_review_artifacts(tmp_path: Path) -> None:
    output_path = tmp_path / "out"
    writer = LocalArtifactsWriter(output_path)
    batch = _batch("secret").model_copy(
        update={
            "run_report": {
                "job_id": "secret",
                "graph_id": "graph",
                "api_key": "report-secret",
            },
            "graph_metrics": {
                "counts": {"nodes": 1},
                "authorization": "Bearer metric-secret",
            },
            "extraction_trace": [
                {
                    "stage": "llm",
                    "provider_metadata": {
                        "provider": "openai_compatible",
                        "api_key": "trace-secret",
                    },
                }
            ],
        }
    )

    writer.write(batch)

    report = json.loads((output_path / "run_report.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_path / "graph_metrics.json").read_text(encoding="utf-8"))
    trace = [
        json.loads(line)
        for line in (output_path / "extraction_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert report["api_key"] == "***"
    assert metrics["authorization"] == "***"
    assert trace[0]["provider_metadata"]["api_key"] == "***"
    assert trace[0]["provider_metadata"]["provider"] == "openai_compatible"


def test_local_artifacts_writer_preserves_schema_for_empty_graph_tables(tmp_path: Path) -> None:
    """Ensure edge-free snapshots still publish complete, inspectable Parquet table schemas.

    Empty rows must not erase column contracts.
    """

    output_path = tmp_path / "out"
    batch = _batch("empty").model_copy(
        update={
            "edges": [],
            "edge_observations": [],
            "communities": [],
            "community_findings": [],
        }
    )

    LocalArtifactsWriter(output_path).write(batch)

    assert set(pd.read_parquet(output_path / "edges.parquet").columns) == set(
        GraphEdge.model_fields
    )
    assert set(pd.read_parquet(output_path / "edge_observations.parquet").columns) >= {
        "id",
        "edge_id",
        "file_id",
        "chunk_id",
        "evidence_id",
    }
    assert set(pd.read_parquet(output_path / "communities.parquet").columns) == set(
        Community.model_fields
    )


def test_local_artifacts_writer_exports_a_dataset_as_the_batch_it_reads_as(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exporting a published dataset table by table gives the files a whole-batch export gives.

    A fleet graph too large to hold as one batch is written a batch of rows at a
    time. One row per batch and one file per row make every table pass through
    the merge of parts; the unranked node comes first, so the rank column has no
    type until a later part gives it one.
    """

    source = _batch("dataset")
    unranked = source.nodes[0].model_copy(update={"id": "node_unranked", "rank": None})
    source = source.model_copy(update={"nodes": [unranked, *source.nodes]})
    store = LocalBlobStore((tmp_path / "store").as_uri())
    store.initialize()
    tables = {}
    for name, _columns, _model in local_artifacts._TABLES:
        files = []
        for index, row in enumerate(getattr(source, name)):
            buffer = BytesIO()
            cast(Callable[..., None], pq.write_table)(
                pa.Table.from_pylist([row if isinstance(row, dict) else row.model_dump()]),
                buffer,
            )
            payload = buffer.getvalue()
            uri = store.put(f"graph/{name}/{index}.parquet", payload, "application/octet-stream")
            files.append(DatasetFile(uri=uri, size_bytes=len(payload)))
        tables[name] = DatasetTableManifest(name=name, row_count=len(files), files=files)
    manifest = GraphDatasetManifest(
        run_id="run", graph_id="graph", engine="spark", tables=tables, metrics={"nodes": 2}
    )
    reader = GraphDatasetReader(store)
    monkeypatch.setattr(local_artifacts, "_DATASET_BATCH_ROWS", 1)

    LocalArtifactsWriter(tmp_path / "whole").write(reader.read(manifest))
    LocalArtifactsWriter(tmp_path / "tables").write_dataset(manifest, reader)

    for name, _columns, _model in local_artifacts._TABLES:
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "tables" / f"{name}.parquet"),
            pd.read_parquet(tmp_path / "whole" / f"{name}.parquet"),
        )
    for report in ("run_report.json", "graph_metrics.json", "extraction_trace.jsonl"):
        assert (tmp_path / "tables" / report).read_text() == (
            tmp_path / "whole" / report
        ).read_text()
    assert pd.read_parquet(tmp_path / "tables" / "nodes.parquet")["rank"].dtype == "float64"
    assert not list((tmp_path / "tables").glob(".*.parts"))


def _batch(label: str) -> GraphWriteBatch:
    chunk = Chunk(
        id=f"chunk_{label}",
        graph_id="graph",
        file_id="file_1",
        document_id="file_1",
        page_number=1,
        chunk_index=0,
        content=f"{label} Alice Smith works at Acme Corp.",
        start_offset=0,
        end_offset=37,
        token_count=7,
        content_hash=f"hash_{label}",
        section_path=["Intro"],
        block_ids=["block_1"],
        asset_ids=["asset_1"],
        ocr_generation_id=f"ocr_{label}",
        embedding=[0.1, 0.2],
    )
    node = GraphNode(
        id=f"node_{label}",
        graph_id="graph",
        normalized_name="alicesmith",
        name="Alice Smith",
        primary_type="PERSON",
        types=["PERSON"],
        description=f"{label} node",
        embedding=[0.1, 0.2],
        source_chunk_ids=[chunk.id],
        degree=1,
        rank=1.0,
    )
    edge = GraphEdge(
        id=f"edge_{label}",
        graph_id="graph",
        source_node_id=node.id,
        target_node_id=f"target_{label}",
        relation_type="works_at",
        description=f"{label} edge",
        weight=1.0,
        source_file_id="file_1",
        source_chunk_ids=[chunk.id],
        embedding=[0.1, 0.2],
    )
    evidence = Evidence(
        id=f"evidence_{label}",
        graph_id="graph",
        subject_id=node.id,
        subject_kind="node",
        file_id="file_1",
        chunk_id=chunk.id,
        page_number=1,
        start_offset=0,
        end_offset=37,
        quote=chunk.content,
    )
    entity_source = EntitySource(
        id=f"source_{label}",
        graph_id="graph",
        node_id=node.id,
        file_id="file_1",
        per_file_description=f"{label} source",
        mention_count=1,
    )
    community = Community(
        id=f"community_{label}",
        graph_id="graph",
        stable_key=f"stable_{label}",
        level=0,
        title=f"{label} community",
        summary=f"{label} summary",
        rating=5.0,
        rating_explanation=f"{label} rating explanation",
        member_node_ids=[node.id],
        suggested_questions=[f"What is {label}?"],
        embedding=[0.1, 0.2],
    )
    finding = CommunityFinding(
        id=f"finding_{label}",
        graph_id="graph",
        community_id=community.id,
        summary=f"{label} finding",
        explanation=f"{label} explanation",
    )
    return GraphWriteBatch(
        graph_id="graph",
        documents=[
            {
                "id": "file_1",
                "graph_id": "graph",
                "file_id": "file_1",
                "checksum": f"checksum_{label}",
                "source_uri": f"file:///{label}.txt",
                "mime_type": "text/plain",
                "size_bytes": 37,
                "ocr_provider": "builtin_text",
            }
        ],
        pages=[
            {
                "id": f"page_{label}",
                "graph_id": "graph",
                "file_id": "file_1",
                "page_number": 1,
                "markdown": chunk.content,
                "raw_text": chunk.content,
                "detected_language": "en",
            }
        ],
        blocks=[
            {
                "id": "block_1",
                "graph_id": "graph",
                "file_id": "file_1",
                "page_number": 1,
                "kind": "paragraph",
                "text": chunk.content,
                "bbox": [0.0, 1.0, 2.0, 3.0],
                "metadata": {"layout": "body"},
            }
        ],
        assets=[
            {
                "id": "asset_1",
                "graph_id": "graph",
                "file_id": "file_1",
                "kind": "image",
                "page_number": 1,
                "uri": f"file:///{label}.png",
                "metadata": {"layout": "figure"},
            }
        ],
        chunks=[chunk],
        nodes=[node],
        edges=[edge],
        evidence=[evidence],
        entity_sources=[entity_source],
        communities=[community],
        community_findings=[finding],
        run_report={"job_id": label, "graph_id": "graph"},
        extraction_trace=[{"stage": "test", "label": label}],
        graph_metrics={"counts": {"nodes": 1}, "quality": {"ok": True}},
    )


def test_staged_publication_requires_commit_and_rejects_older_generation(tmp_path: Path) -> None:
    """A delayed finalizer cannot replace a newer committed destination snapshot."""

    output = tmp_path / "out"
    writer = LocalArtifactsWriter(output)
    writer.write(_batch("original"))
    with writer.stage_publication(_batch("older"), publication_id="older", generation=1) as older:
        assert json.loads((output / "run_report.json").read_text())["job_id"] == "original"
        with writer.stage_publication(
            _batch("newer"), publication_id="newer", generation=2
        ) as newer:
            newer()
        with pytest.raises(RuntimeError, match="newer publication"):
            older()
    assert json.loads((output / "run_report.json").read_text())["job_id"] == "newer"
    assert not list(tmp_path.glob(".out.tmp-*"))


def test_staged_publication_retries_an_interrupted_destination_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reserved generation can resume after replacement fails before completion."""

    output = tmp_path / "out"
    writer = LocalArtifactsWriter(output)
    original = local_artifacts._replace_snapshot
    with writer.stage_publication(_batch("retry"), publication_id="retry", generation=1) as commit:

        def fail_swap(*args: object) -> None:
            raise OSError("interrupted swap")

        monkeypatch.setattr(local_artifacts, "_replace_snapshot", fail_swap)
        with pytest.raises(OSError, match="interrupted"):
            commit()
    monkeypatch.setattr(local_artifacts, "_replace_snapshot", original)
    with writer.stage_publication(_batch("retry"), publication_id="retry", generation=1) as commit:
        commit()
    assert json.loads((output / "run_report.json").read_text())["job_id"] == "retry"
