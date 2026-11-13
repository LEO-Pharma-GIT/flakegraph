from __future__ import annotations

from pathlib import Path

import pytest

from kg_processor.adapters.explorer import StaticHtmlGraphExplorer
from kg_processor.adapters.writers.local_artifacts import LocalArtifactsWriter
from kg_processor.application.graph_explorer import build_graph_explorer_dataset
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


def test_graph_explorer_dataset_preserves_review_data_and_omits_vectors(tmp_path: Path) -> None:
    """Ensure explorer datasets retain review context while removing bulky embedding vectors.

    Every generated layout must cover the same nodes.
    """

    dataset = build_graph_explorer_dataset(_write_explorer_batch(tmp_path))

    assert dataset.graph_id == "graph"
    assert dataset.counts == {
        "documents": 1,
        "chunks": 1,
        "nodes": 2,
        "edges": 1,
        "edge_observations": 0,
        "evidence": 3,
        "communities": 1,
        "findings": 1,
    }
    assert all("embedding" not in node for node in dataset.payload["nodes"])
    assert all("embedding" not in edge for edge in dataset.payload["edges"])
    assert dataset.payload["nodes"][0]["community_ids"] == ["community_1"]
    assert dataset.payload["chunks"][0]["content"].startswith("Alice Smith")
    assert set(dataset.payload["layouts"]) == {"force", "community", "radial"}
    for layout in dataset.payload["layouts"].values():
        assert set(layout) == {"node_alice", "node_acme"}
        assert all(len(position) == 2 for position in layout.values())


def test_static_graph_explorer_writes_safe_atomic_self_contained_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = build_graph_explorer_dataset(_write_explorer_batch(tmp_path))
    html_path = tmp_path / "explorer.html"
    monkeypatch.setattr(
        "kg_processor.adapters.explorer.static_html.get_plotlyjs",
        lambda: "window.Plotly = { react() { return Promise.resolve(); } };",
    )

    result = StaticHtmlGraphExplorer().write(dataset, html_path, title="Graph <Review>")
    rendered = html_path.read_text(encoding="utf-8")

    assert result.html_path == html_path
    assert result.graph_id == "graph"
    assert result.counts["nodes"] == 2
    assert result.size_bytes == html_path.stat().st_size
    assert "<title>Graph &lt;Review&gt;</title>" in rendered
    assert "window.Plotly =" in rendered
    assert 'id="graph-canvas"' in rendered
    assert 'id="community-grid"' in rendered
    assert 'id="quality-checks"' in rendered
    assert "__FLAKEGRAPH_" not in rendered
    assert "</script><script>alert" not in rendered
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert" in rendered
    assert list(tmp_path.glob(".explorer.html.*.tmp")) == []


def test_graph_explorer_reports_missing_graph_tables(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nodes.parquet, edges.parquet"):
        build_graph_explorer_dataset(tmp_path)


def test_static_explorer_allows_template_like_tokens_in_document_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate placeholders before data insertion so source text cannot break export."""

    dataset = build_graph_explorer_dataset(_write_explorer_batch(tmp_path))
    dataset.payload["chunks"][0]["content"] = "Literal __FLAKEGRAPH_APP_JS__ source text"
    monkeypatch.setattr(
        "kg_processor.adapters.explorer.static_html.get_plotlyjs",
        lambda: "window.Plotly = {};",
    )
    destination = tmp_path / "token-source.html"

    StaticHtmlGraphExplorer().write(dataset, destination)

    assert "Literal __FLAKEGRAPH_APP_JS__ source text" in destination.read_text(encoding="utf-8")


def test_static_explorer_rejects_payloads_above_configured_browser_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before writing a static file too large for reliable browser inspection."""

    dataset = build_graph_explorer_dataset(_write_explorer_batch(tmp_path))
    monkeypatch.setattr(
        "kg_processor.adapters.explorer.static_html.get_plotlyjs",
        lambda: "window.Plotly = {};",
    )

    with pytest.raises(ValueError, match="inline limit"):
        StaticHtmlGraphExplorer().write(
            dataset,
            tmp_path / "too-large.html",
            max_payload_bytes=10,
        )


def _write_explorer_batch(tmp_path: Path) -> Path:
    """Write the shared explorer fixture and return its artifact directory."""

    output_path = tmp_path / "artifacts"
    LocalArtifactsWriter(output_path).write(_explorer_batch())
    return output_path


def _explorer_batch() -> GraphWriteBatch:
    source_text = "Alice Smith </script><script>alert(1)</script> works at Acme Corp."
    chunk = Chunk(
        id="chunk_1",
        graph_id="graph",
        file_id="file_1",
        document_id="document_1",
        page_number=1,
        chunk_index=0,
        content=source_text,
        start_offset=0,
        end_offset=len(source_text),
        token_count=9,
        content_hash="content-hash",
        embedding=[0.1, 0.2],
    )
    alice = GraphNode(
        id="node_alice",
        graph_id="graph",
        normalized_name="alice smith",
        name="Alice Smith",
        primary_type="PERSON",
        types=["PERSON"],
        description="Alice works at Acme Corp.",
        embedding=[0.1, 0.2],
        source_chunk_ids=[chunk.id],
        degree=1,
        rank=2.0,
    )
    acme = GraphNode(
        id="node_acme",
        graph_id="graph",
        normalized_name="acme corp",
        name="Acme Corp",
        primary_type="ORGANIZATION",
        types=["ORGANIZATION"],
        description="Acme employs Alice.",
        embedding=[0.2, 0.1],
        source_chunk_ids=[chunk.id],
        degree=1,
        rank=1.0,
    )
    edge = GraphEdge(
        id="edge_works_at",
        graph_id="graph",
        source_node_id=alice.id,
        target_node_id=acme.id,
        relation_type="works_at",
        description="Alice works at Acme Corp.",
        weight=8.0,
        source_file_id="file_1",
        source_chunk_ids=[chunk.id],
        embedding=[0.15, 0.15],
    )
    evidence = [
        Evidence(
            id=f"evidence_{index}",
            graph_id="graph",
            subject_id=subject_id,
            subject_kind=subject_kind,
            file_id="file_1",
            chunk_id=chunk.id,
            page_number=1,
            start_offset=0,
            end_offset=len(source_text),
            quote=source_text,
        )
        for index, (subject_id, subject_kind) in enumerate(
            [(alice.id, "node"), (acme.id, "node"), (edge.id, "edge")],
            start=1,
        )
    ]
    community = Community(
        id="community_1",
        graph_id="graph",
        stable_key="community-key",
        level=0,
        title="Acme network",
        summary="Alice and Acme form an employment community.",
        rating=8.0,
        rating_explanation="The employment relation is strongly grounded.",
        member_node_ids=[alice.id, acme.id],
        suggested_questions=["Who works at Acme?"],
        embedding=[0.1, 0.1],
    )
    return GraphWriteBatch(
        graph_id="graph",
        documents=[
            {
                "id": "document_1",
                "graph_id": "graph",
                "file_id": "file_1",
                "checksum": "checksum",
                "source_uri": "file:///documents/source.txt",
                "mime_type": "text/plain",
                "size_bytes": len(source_text),
            }
        ],
        pages=[
            {
                "id": "page_1",
                "graph_id": "graph",
                "file_id": "file_1",
                "page_number": 1,
                "markdown": source_text,
                "raw_text": source_text,
            }
        ],
        chunks=[chunk],
        nodes=[alice, acme],
        edges=[edge],
        evidence=evidence,
        entity_sources=[
            EntitySource(
                id="source_alice",
                graph_id="graph",
                node_id=alice.id,
                file_id="file_1",
                per_file_description="Alice works at Acme Corp.",
                mention_count=1,
            )
        ],
        communities=[community],
        community_findings=[
            CommunityFinding(
                id="finding_1",
                graph_id="graph",
                community_id=community.id,
                summary="Employment relation",
                explanation="Alice works at Acme Corp.",
            )
        ],
        run_report={"job_id": "job", "graph_id": "graph", "run_id": "run_1"},
        graph_metrics={
            "embedding": {"dimension": 2},
            "providers": {
                "ocr": "builtin_text",
                "llm": "fake",
                "embedding": "hash",
                "writer": "local_artifacts",
            },
        },
    )
