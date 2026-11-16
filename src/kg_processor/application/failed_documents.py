# SPDX-License-Identifier: Apache-2.0
"""Name the source documents that could not be read.

A document that no parser can open is recorded and the run goes on without it.
The record is one trace event, written where parsing failed; every engine turns
those events into ``failed_documents`` rows with the same rule, so one corpus
reports the same unreadable files wherever it was processed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from kg_processor.domain.documents import InputFile
from kg_processor.domain.graph import FailedDocument
from kg_processor.domain.ids import stable_id

FAILED_STATUS = "failed"
PARSE_STAGE = "ocr"


def failed_document_event(
    file: InputFile,
    provider: str | None,
    error: Mapping[str, Any],
) -> dict[str, Any]:
    """The trace event that records one document parsing gave up on.

    Flat, so the Spark finalizer reads it with the same projection as every
    other event in a document's trace.
    """

    return {
        "stage": PARSE_STAGE,
        "status": FAILED_STATUS,
        "file_id": file.id,
        "source_uri": file.source_uri,
        "mime_type": file.mime_type,
        "size_bytes": file.size_bytes,
        "provider": provider,
        "error_type": str(error.get("error_type") or ""),
        "error_message": str(error.get("error_message") or ""),
    }


def is_failed_document_event(event: Mapping[str, Any]) -> bool:
    """A parse event that produced no document."""

    return (
        str(event.get("stage") or "") == PARSE_STAGE
        and str(event.get("status") or "") == FAILED_STATUS
        and bool(event.get("file_id"))
    )


def failed_documents_from_trace(
    trace: Iterable[Mapping[str, Any]],
    graph_id: str,
) -> list[FailedDocument]:
    """Turn the trace's failed parse events into one row per file."""

    rows: dict[str, FailedDocument] = {}
    for event in trace:
        if not is_failed_document_event(event):
            continue
        file_id = str(event["file_id"])
        size = event.get("size_bytes")
        rows[file_id] = FailedDocument(
            id=failed_document_id(graph_id, file_id),
            graph_id=graph_id,
            document_id=file_id,
            file_id=file_id,
            source_uri=str(event.get("source_uri") or ""),
            mime_type=_optional_text(event.get("mime_type")),
            size_bytes=int(size) if size is not None else None,
            provider=_optional_text(event.get("provider")),
            error_type=str(event.get("error_type") or ""),
            error_message=str(event.get("error_message") or ""),
        )
    return sorted(rows.values(), key=lambda row: (row.source_uri, row.file_id))


def failed_document_id(graph_id: str, file_id: str) -> str:
    """One row per file and graph, whichever engine wrote it."""

    return stable_id("failed_document", graph_id, file_id)


def failed_document_metrics(rows: Iterable[FailedDocument]) -> dict[str, Any]:
    """The summary a run report carries: how many files and why."""

    documents = 0
    by_error: dict[str, int] = {}
    for row in rows:
        documents += 1
        name = row.error_type or "unknown"
        by_error[name] = by_error.get(name, 0) + 1
    return {"documents": documents, "by_error_type": dict(sorted(by_error.items()))}


def _optional_text(value: object) -> str | None:
    text = str(value) if value is not None else ""
    return text or None
