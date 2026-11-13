# SPDX-License-Identifier: Apache-2.0
"""Local artifact inspection and parity comparison utilities.

These commands make local artifacts first-class review outputs: they validate
schema, graph quality, ranked summaries, and rerun identity stability before
Snowflake integration is available.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from kg_processor.application.graph_quality import evaluate_graph_quality_rows

_PARQUET_TABLES = [
    "documents",
    "pages",
    "blocks",
    "assets",
    "chunks",
    "nodes",
    "edges",
    "edge_observations",
    "evidence",
    "entity_sources",
    "communities",
    "community_findings",
    "discarded_windows",
    "rejected_records",
    "failed_documents",
]

_REQUIRED_COLUMNS = {
    "documents": {"id", "graph_id", "file_id", "checksum", "source_uri", "mime_type", "size_bytes"},
    "pages": {"id", "graph_id", "file_id", "page_number", "markdown", "raw_text"},
    "blocks": {"id", "graph_id", "file_id", "page_number", "kind", "text", "bbox", "metadata"},
    "assets": {"id", "graph_id", "file_id", "kind", "page_number", "uri", "metadata"},
    "chunks": {
        "id",
        "graph_id",
        "file_id",
        "document_id",
        "page_number",
        "content",
        "content_hash",
        "section_path",
        "block_ids",
        "asset_ids",
        "embedding",
    },
    "nodes": {"id", "name", "primary_type", "description", "degree", "embedding"},
    "edges": {
        "id",
        "source_node_id",
        "target_node_id",
        "relation_type",
        "weight",
        "confidence",
        "source_file_ids",
        "evidence_count",
        "assertion",
        "embedding",
    },
    "edge_observations": {
        "id",
        "edge_id",
        "file_id",
        "chunk_id",
        "weight",
        "confidence",
        "evidence_id",
        "assertion",
    },
    "evidence": {"id", "subject_id", "subject_kind", "file_id", "chunk_id", "quote"},
    "entity_sources": {"id", "node_id", "file_id", "mention_count"},
    "communities": {
        "id",
        "title",
        "summary",
        "rating",
        "rating_explanation",
        "member_node_ids",
        "parent_community_id",
        "child_community_ids",
        "suggested_questions",
        "embedding",
    },
    "community_findings": {"id", "graph_id", "community_id", "summary", "explanation"},
    "discarded_windows": {
        "id",
        "graph_id",
        "document_id",
        "file_id",
        "stage",
        "window_id",
        "chunk_ids",
        "page_start",
        "page_end",
        "extracted_records",
        "record_actions",
        "preview",
    },
    "rejected_records": {
        "id",
        "graph_id",
        "document_id",
        "file_id",
        "window_id",
        "chunk_id",
        "page_number",
        "kind",
        "reason",
        "name",
        "type",
        "source",
        "source_type",
        "target",
        "target_type",
        "quote",
    },
    "failed_documents": {
        "id",
        "graph_id",
        "document_id",
        "file_id",
        "source_uri",
        "error_type",
        "error_message",
    },
}

# Every artifact table is keyed by its stable id; the JSON output still names
# the key columns so a reader can see what a row was matched on.
_STABILITY_KEY_COLUMNS = ("id",)


@dataclass(frozen=True)
class ArtifactTable:
    """In-memory representation of one local artifact table."""

    exists: bool
    columns: list[str]
    rows: list[dict[str, Any]]
    read_error: str | None = None


def inspect_local_graph(output_path: Path) -> dict[str, Any]:
    """Build a review report for a local artifact directory.

    The worker writes a Snowflake-shaped graph to local Parquet/JSON files when
    the local writer is selected. This function treats that directory as a
    reviewable contract: it checks required tables and columns, summarizes run
    metadata, evaluates graph quality, and ranks the most useful graph objects
    without invoking OCR, LLMs, embeddings, or Snowflake.
    """

    # Read every expected table, including missing tables, so schema failures
    # appear in the report instead of becoming file-not-found exceptions.
    artifacts = read_local_graph_artifacts(output_path)
    tables = {name: len(artifact.rows) for name, artifact in artifacts.items()}

    # run_report.json and graph_metrics.json are optional from a filesystem
    # perspective, but when present they give operators the same operational
    # counters that the worker returns on stdout.
    report = _read_json_file(output_path / "run_report.json")
    metrics = _read_json_file(output_path / "graph_metrics.json")

    # The trace file is line-oriented because workers can append progress while
    # long OCR and LLM phases are running. Inspection only needs a quick count
    # to confirm that the run produced traceable extraction events.
    trace_path = output_path / "extraction_trace.jsonl"
    trace_error: str | None = None
    try:
        trace_events = (
            len([line for line in trace_path.read_text(encoding="utf-8").splitlines() if line])
            if trace_path.exists()
            else 0
        )
    except (OSError, UnicodeError) as exc:
        trace_events = 0
        trace_error = f"{type(exc).__name__}: {exc}"
    expected_embedding_dimension = _expected_embedding_dimension(metrics)
    # Quality checks validate graph-internal consistency, such as evidence
    # coverage and embedding dimensions, using the written rows rather than the
    # in-memory objects from the just-finished pipeline.
    quality = evaluate_graph_quality_rows(
        artifacts["nodes"].rows,
        artifacts["edges"].rows,
        artifacts["evidence"].rows,
        expected_embedding_dimension,
        chunks=artifacts["chunks"].rows,
        communities=artifacts["communities"].rows,
    ).model_dump(mode="json")
    schema = _schema_checks(artifacts)
    return {
        "tables": tables,
        "run_report": report,
        "graph_metrics": metrics,
        "quality": quality,
        "schema": schema,
        "top_entities": _top_entities(artifacts["nodes"].rows, artifacts["evidence"].rows),
        "top_relations": _top_relations(
            artifacts["nodes"].rows,
            artifacts["edges"].rows,
            artifacts["evidence"].rows,
        ),
        "community_summaries": _community_summaries(artifacts["communities"].rows),
        "trace_events": trace_events,
        "read_errors": {
            **{
                name: artifact.read_error
                for name, artifact in artifacts.items()
                if artifact.read_error
            },
            **({"extraction_trace": trace_error} if trace_error else {}),
        },
    }


def compare_local_graph_artifacts(left_path: Path, right_path: Path) -> dict[str, Any]:
    """Compare two local artifact directories for stable rerun identity."""

    left_artifacts = read_local_graph_artifacts(left_path)
    right_artifacts = read_local_graph_artifacts(right_path)
    left_report = _read_json_file(left_path / "run_report.json")
    right_report = _read_json_file(right_path / "run_report.json")
    left_metrics = _read_json_file(left_path / "graph_metrics.json")
    right_metrics = _read_json_file(right_path / "graph_metrics.json")
    checks = [
        _ordered_chunk_hash_check(left_report, right_report),
        _quality_ok_check(left_metrics, right_metrics),
        *_artifact_schema_checks(left_artifacts, right_artifacts),
        *_artifact_identity_checks(left_artifacts, right_artifacts),
    ]
    return {
        "ok": all(check["ok"] for check in checks),
        "left": str(left_path),
        "right": str(right_path),
        "checks": checks,
    }


def read_local_graph_artifacts(output_path: Path) -> dict[str, ArtifactTable]:
    """Load every persisted graph table for inspection and presentation adapters."""

    return {
        name: _read_parquet_artifact(output_path / f"{name}.parquet") for name in _PARQUET_TABLES
    }


def _read_parquet_artifact(path: Path) -> ArtifactTable:
    if not path.exists():
        return ArtifactTable(exists=False, columns=[], rows=[])
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError, UnicodeError) as exc:
        return ArtifactTable(
            exists=True,
            columns=[],
            rows=[],
            read_error=f"{type(exc).__name__}: {exc}",
        )
    records = frame.to_dict(orient="records")
    rows = [{str(key): value for key, value in record.items()} for record in records]
    return ArtifactTable(
        exists=True,
        columns=[str(column) for column in frame.columns],
        rows=rows,
    )


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _ordered_chunk_hash_check(
    left_report: dict[str, Any],
    right_report: dict[str, Any],
) -> dict[str, Any]:
    left_hash = left_report.get("ordered_chunk_hash")
    right_hash = right_report.get("ordered_chunk_hash")
    return {
        "name": "ordered_chunk_hash",
        "ok": bool(left_hash) and left_hash == right_hash,
        "left": left_hash,
        "right": right_hash,
    }


def _quality_ok_check(
    left_metrics: dict[str, Any],
    right_metrics: dict[str, Any],
) -> dict[str, Any]:
    left_ok = _metrics_quality_ok(left_metrics)
    right_ok = _metrics_quality_ok(right_metrics)
    return {
        "name": "quality_ok",
        "ok": left_ok is True and right_ok is True,
        "left": left_ok,
        "right": right_ok,
    }


def _metrics_quality_ok(metrics: dict[str, Any]) -> bool | None:
    quality = metrics.get("quality")
    if not isinstance(quality, dict):
        return None
    ok = quality.get("ok")
    return ok if isinstance(ok, bool) else None


def _artifact_identity_checks(
    left_artifacts: dict[str, ArtifactTable],
    right_artifacts: dict[str, ArtifactTable],
) -> list[dict[str, Any]]:
    return [
        _artifact_identity_check(
            table_name,
            left_artifacts[table_name],
            right_artifacts[table_name],
        )
        for table_name in _PARQUET_TABLES
    ]


def _artifact_identity_check(
    table_name: str,
    left: ArtifactTable,
    right: ArtifactTable,
) -> dict[str, Any]:
    key_columns = _STABILITY_KEY_COLUMNS
    missing_columns = sorted(
        (set(key_columns) - set(left.columns)) | (set(key_columns) - set(right.columns))
    )
    left_keys = _row_identity_keys(left.rows, key_columns) if not missing_columns else set()
    right_keys = _row_identity_keys(right.rows, key_columns) if not missing_columns else set()
    missing_from_right = sorted(left_keys - right_keys)
    missing_from_left = sorted(right_keys - left_keys)
    common_value_columns = tuple(sorted(set(left.columns) & set(right.columns)))
    left_values = _row_values_by_identity(left.rows, key_columns, common_value_columns)
    right_values = _row_values_by_identity(right.rows, key_columns, common_value_columns)
    value_mismatches = sorted(
        key for key in left_keys & right_keys if left_values.get(key) != right_values.get(key)
    )
    return {
        "name": f"{table_name}_stable_identity",
        "table": table_name,
        "ok": (
            left.exists
            and right.exists
            and not missing_columns
            and len(left.rows) == len(right.rows)
            and not missing_from_right
            and not missing_from_left
            and not value_mismatches
        ),
        "left_rows": len(left.rows),
        "right_rows": len(right.rows),
        "key_columns": list(key_columns),
        "missing_columns": missing_columns,
        "missing_from_right": missing_from_right[:20],
        "missing_from_left": missing_from_left[:20],
        "value_mismatches": value_mismatches[:20],
    }


def _artifact_schema_checks(
    left_artifacts: dict[str, ArtifactTable],
    right_artifacts: dict[str, ArtifactTable],
) -> list[dict[str, Any]]:
    return [
        _artifact_schema_check(
            table_name,
            left_artifacts[table_name],
            right_artifacts[table_name],
        )
        for table_name in _PARQUET_TABLES
    ]


def _artifact_schema_check(
    table_name: str,
    left: ArtifactTable,
    right: ArtifactTable,
) -> dict[str, Any]:
    required_columns = _REQUIRED_COLUMNS[table_name]
    left_columns = set(left.columns)
    right_columns = set(right.columns)
    left_missing_required = sorted(required_columns - left_columns)
    right_missing_required = sorted(required_columns - right_columns)
    return {
        "name": f"{table_name}_schema",
        "table": table_name,
        "ok": (
            left.exists
            and right.exists
            and not left_missing_required
            and not right_missing_required
        ),
        "left_exists": left.exists,
        "right_exists": right.exists,
        "left_missing_required": left_missing_required,
        "right_missing_required": right_missing_required,
        "left_extra_columns": sorted(left_columns - right_columns),
        "right_extra_columns": sorted(right_columns - left_columns),
    }


def _row_identity_keys(
    rows: list[dict[str, Any]],
    key_columns: tuple[str, ...],
) -> set[str]:
    return {"|".join(str(row.get(column, "")) for column in key_columns) for row in rows}


def _row_values_by_identity(
    rows: list[dict[str, Any]],
    key_columns: tuple[str, ...],
    value_columns: tuple[str, ...],
) -> dict[str, list[str]]:
    """Fingerprint shared row values under the stable identity used for parity."""

    values: dict[str, list[str]] = {}
    for row in rows:
        key = "|".join(str(row.get(column, "")) for column in key_columns)
        fingerprint = json.dumps(
            {column: _json_stable_value(row.get(column)) for column in value_columns},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        values.setdefault(key, []).append(fingerprint)
    return {key: sorted(fingerprints) for key, fingerprints in values.items()}


def _json_stable_value(value: Any) -> Any:
    """Convert array/scalar containers without numpy's truncated string representation."""

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_stable_value(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_stable_value(item())
        except ValueError:
            pass
    if isinstance(value, dict):
        return {str(key): _json_stable_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_stable_value(item) for item in value]
    return value


def _expected_embedding_dimension(metrics: dict[str, Any]) -> int | None:
    embedding = metrics.get("embedding")
    if not isinstance(embedding, dict):
        return None
    dimension = embedding.get("dimension")
    return dimension if isinstance(dimension, int) else None


def _schema_checks(artifacts: dict[str, ArtifactTable]) -> dict[str, Any]:
    checks = []
    for table_name, required_columns in _REQUIRED_COLUMNS.items():
        artifact = artifacts[table_name]
        missing_columns = sorted(required_columns - set(artifact.columns))
        checks.append(
            {
                "table": table_name,
                "exists": artifact.exists,
                "ok": artifact.exists and not missing_columns,
                "missing_columns": missing_columns,
            }
        )
    return {
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
    }


def _top_entities(
    nodes: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    evidence_counts = Counter(str(item.get("subject_id", "")) for item in evidence)
    ranked = sorted(
        nodes,
        key=lambda node: (
            -_float_value(node.get("rank")),
            -_int_value(node.get("degree")),
            str(node.get("name", "")).lower(),
        ),
    )
    return [
        {
            "id": str(node.get("id", "")),
            "name": str(node.get("name", "")),
            "primary_type": str(node.get("primary_type", "")),
            "degree": _int_value(node.get("degree")),
            "rank": _float_value(node.get("rank")),
            "evidence_count": evidence_counts.get(str(node.get("id", "")), 0),
            "source_chunk_count": len(_as_list(node.get("source_chunk_ids"))),
        }
        for node in ranked[:limit]
    ]


def _top_relations(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    names_by_node_id = {str(node.get("id")): str(node.get("name", "")) for node in nodes}
    evidence_counts = Counter(str(item.get("subject_id", "")) for item in evidence)
    ranked = sorted(
        edges,
        key=lambda edge: (
            -_float_value(edge.get("weight")),
            str(edge.get("relation_type", "")),
            str(edge.get("id", "")),
        ),
    )
    return [
        {
            "id": str(edge.get("id", "")),
            "source": names_by_node_id.get(str(edge.get("source_node_id")), ""),
            "target": names_by_node_id.get(str(edge.get("target_node_id")), ""),
            "relation_type": str(edge.get("relation_type", "")),
            "weight": _float_value(edge.get("weight")),
            "evidence_count": evidence_counts.get(str(edge.get("id", "")), 0),
            "source_file_id": str(edge.get("source_file_id", "")),
        }
        for edge in ranked[:limit]
    ]


def _community_summaries(
    communities: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    ranked = sorted(
        communities,
        key=lambda community: (
            -_float_value(community.get("rating")),
            -len(_as_list(community.get("member_node_ids"))),
            str(community.get("title", "")).lower(),
        ),
    )
    return [
        {
            "id": str(community.get("id", "")),
            "title": str(community.get("title", "")),
            "rating": _float_value(community.get("rating")),
            "rating_explanation": str(community.get("rating_explanation", "")),
            "member_count": len(_as_list(community.get("member_node_ids"))),
            "summary": str(community.get("summary", "")),
            "suggested_questions": [
                str(question) for question in _as_list(community.get("suggested_questions"))
            ],
        }
        for community in ranked[:limit]
    ]


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _int_value(value: object) -> int:
    try:
        return int(cast(Any, value)) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _float_value(value: object) -> float:
    try:
        return float(cast(Any, value)) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
