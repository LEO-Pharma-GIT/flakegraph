# SPDX-License-Identifier: Apache-2.0
"""Atomic local artifact writer.

Local artifacts are not debug leftovers; they mirror the Snowflake logical table
contract and are written as all-or-nothing snapshots so failed runs do not leave
half-updated review output.
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from kg_processor.application.graph_dataset import GraphDatasetReader, dataset_run_report
from kg_processor.application.output_ownership import (
    assert_replaceable_output,
    write_ownership_manifest,
)
from kg_processor.application.redaction import redact_sensitive_data
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

_BLOCK_COLUMNS = ["id", "graph_id", "file_id", "page_number", "kind", "text", "bbox", "metadata"]
_ASSET_COLUMNS = ["id", "graph_id", "file_id", "kind", "page_number", "uri", "metadata"]
_DOCUMENT_COLUMNS = [
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
_PAGE_COLUMNS = [
    "id",
    "graph_id",
    "file_id",
    "page_number",
    "markdown",
    "raw_text",
    "detected_language",
]
_CHUNK_COLUMNS = list(Chunk.model_fields)
_NODE_COLUMNS = list(GraphNode.model_fields)
_EDGE_COLUMNS = list(GraphEdge.model_fields)
_EDGE_OBSERVATION_COLUMNS = list(EdgeObservation.model_fields)
_EVIDENCE_COLUMNS = list(Evidence.model_fields)
_ENTITY_SOURCE_COLUMNS = list(EntitySource.model_fields)
_COMMUNITY_COLUMNS = list(Community.model_fields)
_COMMUNITY_FINDING_COLUMNS = list(CommunityFinding.model_fields)
_DISCARDED_WINDOW_COLUMNS = list(DiscardedWindow.model_fields)
_REJECTED_RECORD_COLUMNS = list(RejectedRecord.model_fields)
_FAILED_DOCUMENT_COLUMNS = list(FailedDocument.model_fields)
# Every table a snapshot holds: its columns and, for graph tables, the model
# whose JSON form each row takes. Source tables are rows as extraction left them.
# Always pass the domain columns. Pandas cannot infer a schema from an empty
# row list, but an edge-free graph is still a valid artifact set that must
# satisfy the same local/Snowflake contract as a populated graph.
_TABLES: tuple[tuple[str, list[str], type[BaseModel] | None], ...] = (
    ("documents", _DOCUMENT_COLUMNS, None),
    ("pages", _PAGE_COLUMNS, None),
    ("blocks", _BLOCK_COLUMNS, None),
    ("assets", _ASSET_COLUMNS, None),
    ("chunks", _CHUNK_COLUMNS, Chunk),
    ("nodes", _NODE_COLUMNS, GraphNode),
    ("edges", _EDGE_COLUMNS, GraphEdge),
    ("edge_observations", _EDGE_OBSERVATION_COLUMNS, EdgeObservation),
    ("evidence", _EVIDENCE_COLUMNS, Evidence),
    ("entity_sources", _ENTITY_SOURCE_COLUMNS, EntitySource),
    ("communities", _COMMUNITY_COLUMNS, Community),
    ("community_findings", _COMMUNITY_FINDING_COLUMNS, CommunityFinding),
    ("discarded_windows", _DISCARDED_WINDOW_COLUMNS, DiscardedWindow),
    ("rejected_records", _REJECTED_RECORD_COLUMNS, RejectedRecord),
    ("failed_documents", _FAILED_DOCUMENT_COLUMNS, FailedDocument),
)
# Rows per batch when a published dataset is written table by table: enough to
# keep Parquet row groups useful, few enough that embeddings stay small.
_DATASET_BATCH_ROWS = 2_000


class LocalArtifactsWriter:
    """Writes Snowflake-shaped graph artifacts to a local directory atomically."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def write(self, batch: GraphWriteBatch) -> None:
        """Write a snapshot without taking ownership of an unrelated directory.

        Empty destinations are safe to claim. Non-empty destinations must contain
        this writer's ownership manifest, so a typo in ``output_path`` cannot
        delete arbitrary user data.
        """

        self._publish(lambda path: _write_snapshot(path, batch))

    def write_dataset(self, manifest: GraphDatasetManifest, reader: GraphDatasetReader) -> None:
        """Write a published graph dataset table by table, never holding the whole graph.

        A fleet graph can hold more rows than fit in memory as Python objects,
        embeddings above all. Each table passes through in bounded batches with
        the same row conversion ``write`` applies, so both give the same files.
        """

        self._publish(lambda path: _write_dataset_snapshot(path, manifest, reader))

    @contextmanager
    def stage_publication(
        self, batch: GraphWriteBatch, *, publication_id: str, generation: int
    ) -> Iterator[Callable[[], None]]:
        """Prepare a snapshot before a separately fenced, atomic destination swap.

        The generation ledger lives beside the destination so swapping snapshots
        never drops the fence. A filesystem lock serializes commits from workers
        sharing the path; an older publication cannot overwrite a newer one.
        """

        if generation < 1 or not publication_id:
            raise ValueError("publication requires an identity and positive generation")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.output_path.parent / f".{self.output_path.name}.tmp-{uuid.uuid4().hex}"
        ledger_path = self.output_path.parent / f".{self.output_path.name}.publication.json"
        lock_path = self.output_path.parent / f".{self.output_path.name}.publication.lock"
        try:
            temp_path.mkdir()
            _write_snapshot(temp_path, batch)

            def commit() -> None:
                """Swap this prepared snapshot while retaining the monotonic fence."""

                with _publication_path_lock(lock_path):
                    previous = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
                    previous_generation = int(previous.get("generation", 0))
                    if previous_generation > generation:
                        raise RuntimeError("a newer publication already owns the destination")
                    if previous_generation == generation:
                        if previous.get("publication_id") != publication_id:
                            raise RuntimeError("publication generation belongs to another command")
                        if previous.get("committed"):
                            return
                    assert_replaceable_output(self.output_path)
                    # Persisting the fence first makes a crash fail closed: a
                    # late older command can never replace a newer destination.
                    fence = {"generation": generation, "publication_id": publication_id}
                    _write_publication_ledger(ledger_path, {**fence, "committed": False})
                    _replace_snapshot(temp_path, self.output_path)
                    _write_publication_ledger(ledger_path, {**fence, "committed": True})

            yield commit
        finally:
            if temp_path.exists():
                _remove_path(temp_path)

    def _publish(self, fill: Callable[[Path], None]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        assert_replaceable_output(self.output_path)
        temp_path = self.output_path.parent / (f".{self.output_path.name}.tmp-{uuid.uuid4().hex}")
        try:
            temp_path.mkdir(parents=False, exist_ok=False)
            fill(temp_path)
            _replace_snapshot(temp_path, self.output_path)
        finally:
            if temp_path.exists():
                _remove_path(temp_path)


def _write_publication_ledger(path: Path, record: dict[str, Any]) -> None:
    """Replace the destination fence atomically so interrupted writes fail closed."""

    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(record))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _publication_path_lock(path: Path) -> Iterator[None]:
    """Serialize publication swaps across processes without affecting normal exports."""

    with path.open("a+b") as lock:
        if sys.platform == "win32":
            import msvcrt  # noqa: PLC0415 - only available on Windows.

            lock.write(b"\0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # noqa: PLC0415 - only available on Unix.

            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _write_snapshot(output_path: Path, batch: GraphWriteBatch) -> None:
    """Materialize every table and diagnostic file in a temporary snapshot.

    The caller publishes the directory only after this function succeeds, which
    prevents readers from observing a mixture of old and partially written
    artifacts.
    """

    for name, columns, model in _TABLES:
        rows = getattr(batch, name)
        _write_parquet(
            output_path / f"{name}.parquet",
            _dump_models(rows) if model else rows,
            columns=columns,
        )
    _write_reports(output_path, batch.run_report, batch.graph_metrics, batch.extraction_trace)


def _write_dataset_snapshot(
    output_path: Path,
    manifest: GraphDatasetManifest,
    reader: GraphDatasetReader,
) -> None:
    """Materialize a published dataset's tables and reports, one batch at a time."""

    for name, columns, model in _TABLES:
        batches = reader.table_batches(manifest, name, _DATASET_BATCH_ROWS)
        _write_parquet_batches(
            output_path / f"{name}.parquet",
            (
                [model.model_validate(row).model_dump(mode="json") for row in rows]
                if model
                else rows
                for rows in batches
            ),
            columns=columns,
        )
    # A published dataset carries no extraction trace: its windows' records are
    # the discarded-window and rejected-record tables.
    _write_reports(output_path, dataset_run_report(manifest), manifest.metrics, [])


def _write_reports(
    output_path: Path,
    run_report: dict[str, Any],
    graph_metrics: dict[str, Any],
    extraction_trace: list[dict[str, Any]],
) -> None:
    # Run reports and traces are review artifacts. Redacting here protects both
    # normal pipeline output and any tests or custom callers that construct a
    # GraphWriteBatch directly.
    (output_path / "run_report.json").write_text(
        json.dumps(redact_sensitive_data(run_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / "graph_metrics.json").write_text(
        json.dumps(redact_sensitive_data(graph_metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / "extraction_trace.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in redact_sensitive_data(extraction_trace)
        ),
        encoding="utf-8",
    )
    write_ownership_manifest(output_path)


def _replace_snapshot(temp_path: Path, output_path: Path) -> None:
    # The backup path is only kept during the rename window. If replacement
    # fails, the previous complete snapshot is restored before the exception
    # escapes.
    backup_path = output_path.parent / f".{output_path.name}.previous-{uuid.uuid4().hex}"
    backup_created = False
    if output_path.exists():
        output_path.replace(backup_path)
        backup_created = True
    try:
        temp_path.replace(output_path)
    except Exception:
        if backup_created and backup_path.exists() and not output_path.exists():
            backup_path.replace(output_path)
        raise
    finally:
        if backup_path.exists():
            _remove_path(backup_path)


def _dump_models(models: list[Any]) -> list[dict[str, Any]]:
    return [model.model_dump(mode="json") for model in models]


def _write_parquet(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> None:
    normalized_rows = [
        {key: _normalize_parquet_value(value) for key, value in row.items()} for row in rows
    ]
    frame = pd.DataFrame(normalized_rows, columns=columns)
    frame.to_parquet(path, index=False)


def _write_parquet_batches(
    path: Path,
    batches: Iterable[list[dict[str, Any]]],
    columns: list[str],
) -> None:
    """Write row batches as one Parquet file, each batch converted as ``_write_parquet`` does.

    A batch alone cannot fix a column's type (one whose values are all null in
    it has none), so each is staged as a part and the file takes the schema the
    parts agree on once all of them are known.
    """

    parts_path = path.with_name(f".{path.stem}.parts")
    parts_path.mkdir()
    parts = []
    for index, rows in enumerate(batches):
        part = parts_path / f"{index:06d}.parquet"
        _write_parquet(part, rows, columns=columns)
        parts.append(part)
    if not parts:
        _write_parquet(path, [], columns=columns)
    else:
        parquet = cast(Any, pq)
        schema = pa.unify_schemas(
            [parquet.read_schema(part).remove_metadata() for part in parts],
            promote_options="permissive",
        )
        with parquet.ParquetWriter(path, schema) as writer:
            for part in parts:
                writer.write_table(parquet.read_table(part).replace_schema_metadata().cast(schema))
    shutil.rmtree(parts_path)


def _normalize_parquet_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
