# SPDX-License-Identifier: Apache-2.0
"""Embed a published graph again with another embedding model.

A graph's vectors - of its chunks, entities, relations and communities - are
only comparable with vectors from the model that made them, so moving to a
better embedding model means embedding every published graph again. Nothing
else about a graph changes: its rows, identities and structure stay as the run
left them, and only the vector column of the four tables is recomputed, from
the same text the finalizer embedded.

The new files are written beside the old ones, under
``<run>/graph_dataset/reembed-<model>/``, and the run's graph manifest is
replaced by one naming them, with the model recorded in its metrics. The
previous manifest is kept in the artifact's metadata, so the change can be
undone by restoring it, and the old files stay where they were.
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from kg_processor.application.graph_dataset import _public_s3_uri
from kg_processor.domain.finalization import DatasetFile, DatasetTableManifest, GraphDatasetManifest
from kg_processor.ports.blob_store import BlobStore
from kg_processor.ports.embeddings import EmbeddingProvider, EmbedOptions

# The text each table's vector was made from, as the Spark finalizer builds it
# (spark_finalization.py): Spark's concat_ws skips nulls and keeps empty strings.
REEMBEDDED_TABLES = ("chunks", "nodes", "edges", "communities")


def _join(separator: str, values: Sequence[Any]) -> str:
    return separator.join(str(value) for value in values if value is not None)


def embedding_text(table: str, row: dict[str, Any]) -> str:
    """The text a row of a graph table is embedded from."""

    if table == "chunks":
        return str(row.get("content") or "")
    if table == "nodes":
        aliases = row.get("aliases")
        return _join(
            " ",
            [
                row.get("primary_type"),
                row.get("name"),
                _join(", ", aliases) if aliases is not None else None,
                row.get("description"),
            ],
        )
    if table == "edges":
        return _join(
            " ",
            [
                row.get("relation_type"),
                row.get("description"),
                row.get("source_node_id"),
                row.get("target_node_id"),
            ],
        )
    if table == "communities":
        return _join(" ", [row.get("title"), row.get("summary")])
    raise ValueError(f"no embedding text is defined for table {table!r}")


def model_slug(model: str) -> str:
    """A path segment naming a model."""

    return re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")


def _batches(texts: list[str], size: int) -> Iterator[tuple[int, list[str]]]:
    for start in range(0, len(texts), size):
        yield start, texts[start : start + size]


def embed_texts(
    texts: list[str],
    provider: EmbeddingProvider,
    options: EmbedOptions,
    concurrency: int,
) -> list[list[float]]:
    """Every text's vector, in order, with several provider batches in flight."""

    vectors: list[list[float] | None] = [None] * len(texts)

    def run(batch: tuple[int, list[str]]) -> None:
        start, chunk = batch
        for offset, vector in enumerate(provider.embed(chunk, options)):
            vectors[start + offset] = vector

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(run, _batches(texts, options.batch_size)))
    missing = sum(1 for vector in vectors if vector is None)
    if missing:
        raise RuntimeError(f"{missing} texts came back without a vector")
    return [vector for vector in vectors if vector is not None]


def reembed_manifest(
    manifest: GraphDatasetManifest,
    blob_store: BlobStore,
    provider: EmbeddingProvider,
    options: EmbedOptions,
    *,
    concurrency: int = 8,
    progress: Callable[[str], None] = lambda _message: None,
) -> GraphDatasetManifest:
    """A manifest naming the graph's tables with every vector made by ``options.model``.

    Each file of the four tables is read, its vector column recomputed and the
    file written anew under ``reembed-<model>/``; the other tables keep their
    files. The input manifest is not changed.
    """

    tables = dict(manifest.tables)
    slug = model_slug(options.model)
    for name in REEMBEDDED_TABLES:
        table = manifest.tables.get(name)
        if table is None or not table.files:
            continue
        files: list[DatasetFile] = []
        for index, file in enumerate(table.files):
            frame = cast(Any, pq.read_table)(io.BytesIO(blob_store.get(_public_s3_uri(file.uri))))
            rows = frame.drop_columns(
                ["embedding"] if "embedding" in frame.column_names else []
            ).to_pylist()
            vectors = embed_texts(
                [embedding_text(name, row) for row in rows], provider, options, concurrency
            )
            column = pa.array(vectors, type=pa.list_(pa.float64()))
            if "embedding" in frame.column_names:
                frame = frame.set_column(frame.column_names.index("embedding"), "embedding", column)
            else:
                frame = frame.append_column("embedding", column)
            buffer = io.BytesIO()
            cast(Any, pq.write_table)(frame, buffer, compression="snappy")
            payload = buffer.getvalue()
            key = f"{manifest.run_id}/graph_dataset/reembed-{slug}/{name}/part-{index:05d}.parquet"
            uri = blob_store.put(key, payload, "application/vnd.apache.parquet")
            if file.uri.startswith("s3a://") and uri.startswith("s3://"):
                # Named as the finalizer named its files, for readers that expect it.
                uri = f"s3a://{uri[5:]}"
            files.append(DatasetFile(uri=uri, size_bytes=len(payload)))
            progress(
                f"{manifest.run_id} {name} file {index + 1}/{len(table.files)}: {len(rows)} rows"
            )
        tables[name] = DatasetTableManifest(
            name=table.name,
            schema_version=table.schema_version,
            row_count=table.row_count,
            files=files,
        )
    previous = manifest.metrics.get("embedding")
    metrics = {
        **manifest.metrics,
        "embedding": {
            "model": options.model,
            "dimension": options.dimension,
            "reembedded_from": previous,
        },
    }
    return manifest.model_copy(update={"tables": tables, "metrics": metrics})
