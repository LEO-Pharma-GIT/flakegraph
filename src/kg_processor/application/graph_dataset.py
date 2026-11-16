# SPDX-License-Identifier: Apache-2.0
"""Read a partitioned graph manifest for explicit local export or inspection."""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from kg_processor.domain.finalization import GraphDatasetManifest
from kg_processor.domain.graph import (
    Chunk,
    Community,
    CommunityFinding,
    DiscardedWindow,
    EdgeObservation,
    EntitySource,
    Evidence,
    FailedDocument,
    GraphEdge,
    GraphNode,
    GraphWriteBatch,
    RejectedRecord,
)
from kg_processor.ports.blob_store import BlobStore


class GraphDatasetReader:
    """Materialize an explicitly requested graph version from Parquet partitions.

    Spark publication leaves table files partitioned in shared object storage.
    This reader exists for bounded local export and the static explorer, both of
    which inherently require a finite in-memory selection. Large downstream loads
    should consume the manifest partitions without passing through this reader.
    """

    def __init__(self, blob_store: BlobStore) -> None:
        """Retain the same authenticated object adapter used by distributed workers."""

        self.blob_store = blob_store

    def read(self, manifest: GraphDatasetManifest) -> GraphWriteBatch:
        """Load every table named by a manifest into the portable writer contract."""

        rows = {name: self._table_rows(manifest, name) for name in manifest.tables}
        return GraphWriteBatch(
            graph_id=manifest.graph_id,
            documents=rows["documents"],
            pages=rows["pages"],
            blocks=rows["blocks"],
            assets=rows["assets"],
            chunks=[Chunk.model_validate(row) for row in rows["chunks"]],
            nodes=[GraphNode.model_validate(row) for row in rows["nodes"]],
            edges=[GraphEdge.model_validate(row) for row in rows["edges"]],
            edge_observations=[
                EdgeObservation.model_validate(row) for row in rows["edge_observations"]
            ],
            evidence=[Evidence.model_validate(row) for row in rows["evidence"]],
            entity_sources=[EntitySource.model_validate(row) for row in rows["entity_sources"]],
            communities=[Community.model_validate(row) for row in rows["communities"]],
            community_findings=[
                CommunityFinding.model_validate(row) for row in rows["community_findings"]
            ],
            discarded_windows=[
                DiscardedWindow.model_validate(row) for row in rows["discarded_windows"]
            ],
            rejected_records=[
                RejectedRecord.model_validate(row) for row in rows["rejected_records"]
            ],
            failed_documents=[
                FailedDocument.model_validate(row) for row in rows["failed_documents"]
            ],
            run_report=dataset_run_report(manifest),
            graph_metrics=manifest.metrics,
        )

    def table_batches(
        self,
        manifest: GraphDatasetManifest,
        name: str,
        batch_size: int,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield one logical table's rows in bounded batches, a partition file at a time.

        Exporting a large graph needs every row but never all of them at once:
        a few thousand thousand-dimension embeddings fit in memory, a graph's
        worth of them as Python objects does not.
        """

        for item in manifest.table(name).files:
            parquet_file = cast(Any, pq.ParquetFile)
            parquet = parquet_file(BytesIO(self.blob_store.get(_public_s3_uri(item.uri))))
            for batch in parquet.iter_batches(batch_size=batch_size):
                yield batch.to_pylist()

    def _table_rows(
        self,
        manifest: GraphDatasetManifest,
        name: str,
    ) -> list[dict[str, Any]]:
        """Read and concatenate one logical table's immutable Parquet files."""

        table_manifest = manifest.table(name)
        if not table_manifest.files:
            return []
        read_table = cast(Any, pq.read_table)
        tables = [
            read_table(BytesIO(self.blob_store.get(_public_s3_uri(item.uri))))
            for item in table_manifest.files
        ]
        table = pa.concat_tables(tables, promote_options="default")
        return [dict(row) for row in table.to_pylist()]


def dataset_run_report(manifest: GraphDatasetManifest) -> dict[str, Any]:
    """The run report a published dataset carries: its run, engine, and timings."""

    return {
        "run_id": manifest.run_id,
        "engine": manifest.engine,
        "timings_seconds": manifest.timings_seconds,
    }


def _public_s3_uri(uri: str) -> str:
    """Translate Hadoop's internal S3A scheme back to the artifact-store contract."""

    return f"s3://{uri[6:]}" if uri.startswith("s3a://") else uri
