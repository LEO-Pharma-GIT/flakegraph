"""A published graph embedded again with another model."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from kg_processor.adapters.distributed.local_blob import LocalBlobStore
from kg_processor.application.reembed import embedding_text, model_slug, reembed_manifest
from kg_processor.domain.finalization import DatasetFile, DatasetTableManifest, GraphDatasetManifest
from kg_processor.ports.embeddings import EmbedOptions


class _LengthEmbedder:
    """A vector that says which text it came from: [length, model marker]."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        self.texts.extend(texts)
        return [[float(len(text)), float(options.dimension)] for text in texts]


def _parquet(rows: list[dict[str, object]]) -> bytes:
    buffer = io.BytesIO()
    cast(Any, pq.write_table)(pa.Table.from_pylist(rows), buffer)
    return buffer.getvalue()


def test_each_table_is_embedded_from_the_text_the_finalizer_used() -> None:
    assert embedding_text("chunks", {"content": "Lanterns sold well."}) == "Lanterns sold well."
    node = {
        "primary_type": "Product",
        "name": "Lantern",
        "aliases": ["lamp", None],
        "description": "A light.",
    }
    assert embedding_text("nodes", node) == "Product Lantern lamp A light."
    # Spark's concat_ws skips nulls and keeps empty strings.
    assert (
        embedding_text("nodes", {**node, "aliases": [], "description": None}) == "Product Lantern "
    )
    edge = {
        "relation_type": "SOLD_IN",
        "description": None,
        "source_node_id": "n1",
        "target_node_id": "n2",
    }
    assert embedding_text("edges", edge) == "SOLD_IN n1 n2"
    assert (
        embedding_text("communities", {"title": "Harbour", "summary": "Trade."}) == "Harbour Trade."
    )
    assert model_slug("Org/Embedder-4B") == "org-embedder-4b"


def test_vectors_are_replaced_in_new_files_and_the_model_recorded(tmp_path: Path) -> None:
    store = LocalBlobStore((tmp_path / "objects").as_uri())
    chunks_uri = store.put(
        "run_x/graph_dataset/attempt-0001/chunks/part-0.parquet",
        _parquet(
            [
                {"id": "c1", "content": "abc", "embedding": [0.5, 0.5]},
                {"id": "c2", "content": "abcdef", "embedding": [0.1, 0.1]},
            ]
        ),
        "application/vnd.apache.parquet",
    )
    documents_uri = store.put(
        "run_x/graph_dataset/attempt-0001/documents/part-0.parquet",
        _parquet([{"id": "d1"}]),
        "application/vnd.apache.parquet",
    )
    manifest = GraphDatasetManifest(
        run_id="run_x",
        graph_id="graph_x",
        engine="spark_graphframes",
        tables={
            "chunks": DatasetTableManifest(
                name="chunks", row_count=2, files=[DatasetFile(uri=chunks_uri, size_bytes=1)]
            ),
            "documents": DatasetTableManifest(
                name="documents", row_count=1, files=[DatasetFile(uri=documents_uri, size_bytes=1)]
            ),
        },
        metrics={"embedding": {"model": "old/embedder", "dimension": 2}, "nodes": 0},
    )
    embedder = _LengthEmbedder()

    updated = reembed_manifest(
        manifest,
        store,
        embedder,
        EmbedOptions(model="new/embedder", dimension=2, batch_size=1),
        concurrency=2,
    )

    assert embedder.texts == ["abc", "abcdef"]
    new_file = updated.tables["chunks"].files[0]
    assert "reembed-new-embedder/chunks/" in new_file.uri
    rows = cast(Any, pq.read_table)(io.BytesIO(store.get(new_file.uri))).to_pylist()
    assert [(row["id"], row["embedding"]) for row in rows] == [
        ("c1", [3.0, 2.0]),
        ("c2", [6.0, 2.0]),
    ]
    assert new_file.size_bytes == len(store.get(new_file.uri))
    # The old files stay; tables without vectors keep theirs.
    assert cast(Any, pq.read_table)(io.BytesIO(store.get(chunks_uri))).to_pylist()[0][
        "embedding"
    ] == [
        0.5,
        0.5,
    ]
    assert updated.tables["documents"] == manifest.tables["documents"]
    assert updated.metrics["embedding"] == {
        "model": "new/embedder",
        "dimension": 2,
        "reembedded_from": {"model": "old/embedder", "dimension": 2},
    }
    assert updated.metrics["nodes"] == 0
    assert manifest.metrics["embedding"]["model"] == "old/embedder"
