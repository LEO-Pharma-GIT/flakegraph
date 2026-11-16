"""The one rule every engine uses to name documents that could not be read."""

from __future__ import annotations

from pathlib import Path

from kg_processor.application.failed_documents import (
    failed_document_event,
    failed_document_id,
    failed_document_metrics,
    failed_documents_from_trace,
    is_failed_document_event,
)
from kg_processor.domain.documents import InputFile
from kg_processor.domain.ids import stable_id


def _file(file_id: str, name: str) -> InputFile:
    return InputFile(
        id=file_id,
        path=Path("/tmp") / name,
        source_uri=f"s3://corpus/{name}",
        checksum="sha256:0",
        mime_type="application/pdf",
        size_bytes=18,
    )


def test_a_failed_parse_becomes_one_row_naming_the_file_and_why() -> None:
    event = failed_document_event(
        _file("file_broken", "broken.pdf"),
        "fallback",
        {"error_type": "RuntimeError", "error_message": "Unsupported file type: txt"},
    )

    (row,) = failed_documents_from_trace([{"stage": "chunking"}, event], "graph")

    assert row.id == stable_id("failed_document", "graph", "file_broken")
    assert row.id == failed_document_id("graph", "file_broken")
    assert (row.document_id, row.file_id) == ("file_broken", "file_broken")
    assert row.source_uri == "s3://corpus/broken.pdf"
    assert (row.mime_type, row.size_bytes, row.provider) == ("application/pdf", 18, "fallback")
    assert (row.error_type, row.error_message) == ("RuntimeError", "Unsupported file type: txt")


def test_a_file_recorded_twice_is_one_row() -> None:
    """A retried batch can carry the same failure again; the graph names it once."""

    event = failed_document_event(_file("file_a", "a.pdf"), None, {"error_type": "OSError"})

    rows = failed_documents_from_trace([event, dict(event)], "graph")

    assert [row.file_id for row in rows] == ["file_a"]


def test_only_a_failed_parse_is_a_failed_document() -> None:
    assert not is_failed_document_event({"stage": "ocr", "file_id": "file_a"})
    assert not is_failed_document_event({"stage": "relation_extraction", "status": "failed"})
    assert not is_failed_document_event({"stage": "ocr", "status": "failed", "file_id": ""})


def test_metrics_count_documents_by_error_type() -> None:
    rows = failed_documents_from_trace(
        [
            failed_document_event(_file("a", "a.pdf"), None, {"error_type": "RuntimeError"}),
            failed_document_event(_file("b", "b.pdf"), None, {"error_type": "RuntimeError"}),
            failed_document_event(_file("c", "c.pdf"), None, {}),
        ],
        "graph",
    )

    assert failed_document_metrics(rows) == {
        "documents": 3,
        "by_error_type": {"RuntimeError": 2, "unknown": 1},
    }
