# SPDX-License-Identifier: Apache-2.0
"""Bounded Snowflake publication for Spark-finalized graph manifests."""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnectionConfig,
    SnowflakeCursor,
    load_snowflake_connector,
    snowflake_transaction,
    stage_path,
)
from kg_processor.adapters.writers.snowflake_bulk import (
    SnowflakeBulkWriter,
    acquire_publication_fence,
    best_effort_remove_stage_prefix,
    build_bulk_load_prefix,
    build_bulk_merge_statement,
    build_copy_into_load_table_statement,
    build_create_load_table_statement,
    build_load_table_name,
    build_put_statement,
    ensure_publication_fence,
    validate_put_result,
    write_bulk_load_files,
)
from kg_processor.adapters.writers.snowflake_direct import (
    TABLE_COLUMNS,
    build_edge_reconciliation_statements,
    build_reindex_delete_statements,
    build_snowflake_rows,
)
from kg_processor.application.snowflake_schema import (
    render_snowflake_schema_sql,
    split_sql_statements,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.finalization import GraphDatasetManifest
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.ports.blob_store import BlobStore

_BYTES_PER_MEBIBYTE = 1024 * 1024
_PARQUET_BATCH_ROWS = 8_192
_FINALIZER_CREDENTIAL_SLOTS = {
    "password": "KG_SNOWFLAKE_PASSWORD",
    "oauth_token": "KG_SNOWFLAKE_OAUTH_TOKEN",
}
_MANIFEST_TABLES = {
    "documents": "KG_DOCUMENT",
    "pages": "KG_PAGE",
    "blocks": "KG_BLOCK",
    "assets": "KG_ASSET",
    "chunks": "KG_CHUNK",
    "nodes": "KG_NODE",
    "edges": "KG_EDGE",
    "edge_observations": "KG_EDGE_OBSERVATION",
    "evidence": "KG_EVIDENCE",
    "entity_sources": "KG_ENTITY_SOURCE",
    "communities": "KG_COMMUNITY",
    "community_findings": "KG_COMMUNITY_FINDING",
    "discarded_windows": "KG_DISCARDED_WINDOW",
    "rejected_records": "KG_REJECTED_RECORD",
    "failed_documents": "KG_FAILED_DOCUMENT",
}


class SnowflakeManifestWriter:
    """Bulk-load a partitioned graph without collecting it on the finalizer.

    Spark limits rows per published Parquet file. This adapter reads and transforms
    one such file at a time, COPYs it into a temporary Snowflake table, then releases
    the Python rows before moving on. The final graph replacement is a single
    transaction after every partition has loaded successfully.
    """

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        embedding_dimension: int,
        bulk_stage: str,
        blob_store: BlobStore,
        *,
        connector_factory: ConnectorFactory | None = None,
        target_file_size_bytes: int = 128 * _BYTES_PER_MEBIBYTE,
        publication_id: str | None = None,
        publication_generation: int | None = None,
    ) -> None:
        """Retain connection, schema, staging, and immutable-object dependencies."""

        if target_file_size_bytes <= 0:
            raise ValueError("target_file_size_bytes must be positive")
        self.config = config
        self.embedding_dimension = embedding_dimension
        self.bulk_stage = bulk_stage
        self.blob_store = blob_store
        self.connector_factory = connector_factory or load_snowflake_connector()
        self.target_file_size_bytes = target_file_size_bytes
        self.publication_id = publication_id
        self.publication_generation = publication_generation

    def write(self, manifest: GraphDatasetManifest) -> None:
        """Stage all manifest partitions and atomically replace one graph snapshot."""

        load_id = (
            f"PUB{self.publication_generation}"
            if self.publication_generation is not None
            else uuid.uuid4().hex[:12].upper()
        )
        metadata_batch = _empty_batch(manifest)
        prefix = build_bulk_load_prefix(metadata_batch, load_id)
        connection = self.connector_factory(**self.config.connect_kwargs())
        cursor = connection.cursor()
        load_tables: dict[str, str] = {}
        try:
            for statement in split_sql_statements(
                render_snowflake_schema_sql(self.embedding_dimension)
            ):
                cursor.execute(statement)
            with tempfile.TemporaryDirectory(prefix="flakegraph-snowflake-manifest-") as tmp:
                root = Path(tmp)
                sequence = 0
                for logical_name, target_table in _MANIFEST_TABLES.items():
                    table = manifest.table(logical_name)
                    if table.row_count == 0:
                        continue
                    for source_file in table.files:
                        sequence += 1
                        source_path = root / f"manifest-{sequence:06d}.parquet"
                        self._download_manifest_file(source_file.uri, source_path)
                        parquet_file = cast(Any, pq.ParquetFile)(source_path)
                        for batch_index, record_batch in enumerate(
                            parquet_file.iter_batches(batch_size=_PARQUET_BATCH_ROWS),
                            start=1,
                        ):
                            rows = record_batch.to_pylist()
                            batch = _batch_with_rows(manifest, logical_name, rows)
                            mapped_rows = build_snowflake_rows(batch)[target_table]
                            self._stage_rows(
                                cursor,
                                root / f"source-{sequence:06d}-{batch_index:06d}",
                                prefix=(f"{prefix}/source-{sequence:06d}-{batch_index:06d}"),
                                load_id=load_id,
                                table_name=target_table,
                                rows=mapped_rows,
                                load_tables=load_tables,
                            )
                        source_path.unlink(missing_ok=True)
                metadata_rows = build_snowflake_rows(metadata_batch)
                for target_table in ("KG_RUN_REPORT", "KG_GRAPH_METRICS"):
                    sequence += 1
                    self._stage_rows(
                        cursor,
                        root / f"metadata-{sequence:06d}",
                        prefix=f"{prefix}/metadata-{sequence:06d}",
                        load_id=load_id,
                        table_name=target_table,
                        rows=metadata_rows[target_table],
                        load_tables=load_tables,
                    )

                self._ensure_publication_fence(cursor)
                # A fence that is not ours means a newer generation has already
                # been delivered; the transaction then ends with nothing in it.
                with snowflake_transaction(connection) as transaction:
                    if self._acquire_publication_fence(transaction, manifest.graph_id):
                        self._replace_graph(transaction, metadata_batch, load_tables)
        finally:
            best_effort_remove_stage_prefix(cursor, connection, self.bulk_stage, prefix)
            cursor.close()
            connection.close()

    def _replace_graph(
        self,
        cursor: SnowflakeCursor,
        metadata_batch: GraphWriteBatch,
        load_tables: Mapping[str, str],
    ) -> None:
        """Swap the staged rows in for the graph's previous snapshot."""

        for sql, params in build_reindex_delete_statements(metadata_batch):
            cursor.execute(sql, params)
        for table_name, load_table_name in load_tables.items():
            cursor.execute(
                build_bulk_merge_statement(
                    table_name,
                    load_table_name,
                    TABLE_COLUMNS[table_name],
                    self.embedding_dimension,
                )
            )
        # KG_EDGE_OBSERVATION rows are the durable per-file support for
        # every edge, so canonical aggregates are rebuilt from them the
        # same way the direct and bulk writers do.
        for sql, params in build_edge_reconciliation_statements(metadata_batch):
            cursor.execute(sql, params)

    def _download_manifest_file(self, uri: str, destination: Path) -> None:
        """Download one partition to disk so Parquet batches stay memory bounded."""

        public_uri = _public_object_uri(uri)
        with destination.open("wb") as output:
            self.blob_store.download_to(public_uri, output)

    def _acquire_publication_fence(self, cursor: Any, graph_id: str) -> bool:
        """Make newer durable generations permanently supersede older deliveries."""

        if self.publication_generation is None:
            return True
        if not self.publication_id:
            raise ValueError("publication generation requires a publication id")
        return acquire_publication_fence(
            cursor, graph_id, self.publication_generation, self.publication_id
        )

    def _ensure_publication_fence(self, cursor: Any) -> None:
        """Create the destination generation ledger before opening the data transaction."""

        if self.publication_generation is not None:
            ensure_publication_fence(cursor)

    def _stage_rows(
        self,
        cursor: Any,
        output_dir: Path,
        *,
        prefix: str,
        load_id: str,
        table_name: str,
        rows: list[dict[str, object]],
        load_tables: dict[str, str],
    ) -> None:
        """Append one bounded row group to a run-scoped Snowflake load table."""

        if not rows:
            return
        load_table = load_tables.get(table_name)
        if load_table is None:
            load_table = build_load_table_name(table_name, load_id)
            cursor.execute(build_create_load_table_statement(load_table, TABLE_COLUMNS[table_name]))
            load_tables[table_name] = load_table
        files = write_bulk_load_files(
            {table_name: rows},
            output_dir,
            self.bulk_stage,
            prefix,
            load_id,
            self.target_file_size_bytes,
        )
        for load_file in files:
            cursor.execute(
                build_put_statement(
                    load_file.local_path,
                    stage_path(self.bulk_stage, prefix),
                )
            )
            validate_put_result(cursor)
            cursor.execute(
                build_copy_into_load_table_statement(load_table, load_file.stage_file_location)
            )


class TaskConfiguredManifestPublisher:
    """Resolve each run's destination while retaining deployment-owned dependencies."""

    def __init__(self, settings: Settings, blob_store: BlobStore) -> None:
        """Retain worker settings and the shared immutable object data plane."""

        self.settings = settings
        self.blob_store = blob_store

    def publish(
        self,
        manifest: GraphDatasetManifest,
        task_payload: Mapping[str, object],
    ) -> None:
        """Build and invoke the selected writer, leaving local manifests at rest."""

        writer = manifest_writer_from_task(self.settings, task_payload, self.blob_store)
        if writer is not None:
            writer.write(manifest)

    def stage_batch_publication(
        self, batch: GraphWriteBatch, task_payload: Mapping[str, object]
    ) -> AbstractContextManager[Callable[[], None]]:
        """Compose only the destination writer, whichever engine computed the batch.

        A Spark worker's stage pipeline deliberately omits local-finalization
        dependencies. Publication commands may have been computed by either
        engine, so their destination cannot depend on that pipeline's writer.
        """

        publication = task_payload.get("publication")
        if not isinstance(publication, Mapping):
            raise ValueError("batch publication requires a durable publication fence")
        publication_id = _required_text(publication, "id")
        generation = int(publication.get("generation") or 0)
        if generation < 1:
            raise ValueError("publication generation must be positive")
        selected = manifest_writer_from_task(self.settings, task_payload, self.blob_store)
        if selected is None:
            from kg_processor.adapters.writers.local_artifacts import (  # noqa: PLC0415
                LocalArtifactsWriter,
            )

            return LocalArtifactsWriter(self.settings.writer.output_path).stage_publication(
                batch, publication_id=publication_id, generation=generation
            )
        return SnowflakeBulkWriter(
            selected.config,
            selected.embedding_dimension,
            selected.bulk_stage,
            connector_factory=selected.connector_factory,
            target_file_size_bytes=selected.target_file_size_bytes,
        ).stage_publication(batch, publication_id=publication_id, generation=generation)


def manifest_writer_from_task(
    settings: Settings,
    payload: Mapping[str, object],
    blob_store: BlobStore,
) -> SnowflakeManifestWriter | None:
    """Build the requested fleet output writer from a non-secret final-task payload."""

    output = payload.get("output")
    if not isinstance(output, Mapping):
        return None
    provider = str(output.get("provider") or "local_artifacts")
    if provider == "local_artifacts":
        return None
    if provider != "snowflake_bulk":
        raise ValueError(f"unsupported distributed graph output provider: {provider}")
    raw = output.get("snowflake")
    if not isinstance(raw, Mapping):
        raise ValueError("snowflake_bulk output is missing Snowflake target settings")
    password = _credential(raw, "password_credential_slot", settings.snowflake.password)
    oauth_token = _credential(raw, "oauth_token_credential_slot", settings.snowflake.oauth_token)
    database = _required_text(raw, "database")
    schema = _required_text(raw, "schema")
    bulk_stage = _required_text(raw, "bulk_stage")
    target_mb = int(raw.get("bulk_target_file_size_mb") or 128)
    publication = payload.get("publication")
    publication_id: str | None = None
    publication_generation: int | None = None
    if isinstance(publication, Mapping):
        publication_id = _required_text(publication, "id")
        publication_generation = int(publication.get("generation") or 0)
        if publication_generation <= 0:
            raise ValueError("publication generation must be positive")
    # The credential comes from the worker's own environment, so the payload must
    # not choose where it is sent. A task naming another host would have the
    # worker authenticate to it with the fleet's Snowflake credential. Only the
    # destination within the authenticated account is taken from the task.
    config = SnowflakeConnectionConfig(
        account=settings.snowflake.account,
        host=settings.snowflake.host,
        user=settings.snowflake.user,
        password=password,
        authenticator=settings.snowflake.authenticator,
        private_key_path=settings.snowflake.private_key_path,
        database=database,
        schema_name=schema,
        role=_optional_text(raw, "role"),
        warehouse=_optional_text(raw, "warehouse"),
        oauth_token=oauth_token,
        oauth_token_path=settings.snowflake.oauth_token_path,
    )
    return SnowflakeManifestWriter(
        config,
        settings.embedding.dimension,
        bulk_stage,
        blob_store,
        target_file_size_bytes=target_mb * _BYTES_PER_MEBIBYTE,
        publication_id=publication_id,
        publication_generation=publication_generation,
    )


def _empty_batch(manifest: GraphDatasetManifest) -> GraphWriteBatch:
    """Create the metadata and delete contract shared by all partition loads.

    A manifest always describes one complete immutable graph version, so the
    batch declares snapshot scope and the load replaces the whole graph.
    """

    return GraphWriteBatch(
        graph_id=manifest.graph_id,
        write_scope="graph_snapshot",
        documents=[],
        pages=[],
        chunks=[],
        nodes=[],
        edges=[],
        evidence=[],
        entity_sources=[],
        communities=[],
        community_findings=[],
        run_report={
            "job_id": manifest.run_id,
            "run_id": manifest.run_id,
            "graph_id": manifest.graph_id,
            "engine": manifest.engine,
            "timings_seconds": manifest.timings_seconds,
        },
        graph_metrics=manifest.metrics,
        relation_weight_max=manifest.relation_weight_max,
    )


def _batch_with_rows(
    manifest: GraphDatasetManifest,
    logical_name: str,
    rows: list[dict[str, Any]],
) -> GraphWriteBatch:
    """Validate one manifest partition through the ordinary writer-ready model."""

    values = _empty_batch(manifest).model_dump(mode="python")
    values[logical_name] = rows
    return GraphWriteBatch.model_validate(values)


def _credential(raw: Mapping[str, object], key: str, fallback: str | None) -> str | None:
    """Resolve one credential reference only inside the claiming worker process."""

    slot = _optional_text(raw, key)
    if not slot:
        return fallback
    try:
        name = _FINALIZER_CREDENTIAL_SLOTS[slot]
    except KeyError as exc:
        raise ValueError(f"unsupported Snowflake finalizer credential slot: {slot}") from exc
    try:
        return os.environ[name]
    except KeyError as exc:
        raise ValueError(
            f"required Snowflake credential environment variable is not set: {name}"
        ) from exc


def _required_text(raw: Mapping[str, object], key: str) -> str:
    """Return one required target field with a destination-focused error."""

    value = _optional_text(raw, key)
    if not value:
        raise ValueError(f"snowflake_bulk output requires Snowflake setting: {key}")
    return value


def _optional_text(raw: Mapping[str, object], key: str) -> str | None:
    """Normalize optional payload scalars while rejecting nested structures."""

    value = raw.get(key)
    return str(value).strip() or None if value is not None else None


def _public_object_uri(uri: str) -> str:
    """Translate Spark's Hadoop S3A URI back to the BlobStore's public contract."""

    return f"s3://{uri[6:]}" if uri.startswith("s3a://") else uri
