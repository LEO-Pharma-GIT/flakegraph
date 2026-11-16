from __future__ import annotations

import io
import json
from pathlib import Path

from rich.console import Console

from kg_processor.adapters.progress import RichTerminalProgressSink, WorkerProgressContext
from kg_processor.application.progress import (
    CompositeProgressSink,
    JsonLineProgressSink,
    ProgressEvent,
    error_metadata,
)


def test_json_line_progress_sink_writes_parseable_redirection_safe_records() -> None:
    stream = io.StringIO()
    sink = JsonLineProgressSink(stream)

    sink.emit(
        ProgressEvent(
            job_id="job",
            graph_id="graph",
            stage="ocr",
            status="completed",
            file_id="file_1",
            counts={"pages": 2},
            metadata={"provider": "mineru_internal"},
            elapsed_ms=123,
            timestamp="2026-07-03T00:00:00+00:00",
        )
    )

    payload = json.loads(stream.getvalue())

    assert payload == {
        "counts": {"pages": 2},
        "elapsed_ms": 123,
        "event": "kg_processor.progress",
        "file_id": "file_1",
        "graph_id": "graph",
        "job_id": "job",
        "metadata": {"provider": "mineru_internal"},
        "stage": "ocr",
        "status": "completed",
        "timestamp": "2026-07-03T00:00:00+00:00",
    }


def test_composite_progress_sink_fans_out_events_in_order() -> None:
    first = RecordingSink()
    second = RecordingSink()
    event = ProgressEvent(job_id="job", graph_id="graph", stage="merge", status="started")

    CompositeProgressSink([first, second]).emit(event)

    assert first.events == [event]
    assert second.events == [event]


def test_composite_progress_sink_isolates_one_failing_destination() -> None:
    """A telemetry outage must not stop delivery or the graph pipeline."""

    first = FailingSink()
    second = RecordingSink()
    event = ProgressEvent(job_id="job", graph_id="graph", stage="merge", status="started")
    sink = CompositeProgressSink([first, second])

    sink.emit(event)

    assert second.events == [event]


def test_progress_event_redacts_sensitive_metadata_before_json_serialization() -> None:
    event = ProgressEvent(
        job_id="job",
        graph_id="graph",
        stage="ocr",
        status="completed",
        metadata={
            "provider": "mineru_api",
            "api_key": "ocr-secret",
            "nested": {"authorization": "Bearer token"},
        },
    )

    payload = event.as_log_record()

    assert payload["metadata"] == {
        "provider": "mineru_api",
        "api_key": "***",
        "nested": {"authorization": "***"},
    }


def test_error_metadata_redacts_sas_query_parameters() -> None:
    metadata = error_metadata(
        RuntimeError("download failed for https://storage.example/file.pdf?sv=2024&sig=secret&sp=r")
    )

    assert metadata["error_message"] == (
        "download failed for https://storage.example/file.pdf?sv=***&sig=***&sp=***"
    )


def test_rich_terminal_progress_renders_real_stage_and_batch_counts() -> None:
    stream = io.StringIO()
    now = [100.0]
    sink = RichTerminalProgressSink(
        _terminal_context(),
        console=Console(file=stream, color_system=None, width=110),
        clock=lambda: now[0],
        live_enabled=False,
    )

    sink.emit(ProgressEvent("job", "graph", "file_source", "started"))
    sink.emit(
        ProgressEvent(
            "job",
            "graph",
            "file_source",
            "completed",
            counts={"files_seen": 3},
            elapsed_ms=10,
        )
    )
    sink.emit(
        ProgressEvent(
            "job",
            "graph",
            "ocr",
            "started",
            file_id="file_1",
            metadata={"source_uri": "/documents/martial-arts-history.pdf"},
        )
    )
    sink.emit(
        ProgressEvent(
            "job",
            "graph",
            "ocr",
            "completed",
            file_id="file_1",
            counts={"pages": 4},
            elapsed_ms=1_200,
        )
    )
    sink.emit(ProgressEvent("job", "graph", "chunking", "completed", counts={"chunks": 12}))
    sink.emit(
        ProgressEvent(
            "job",
            "graph",
            "graph_extraction",
            "started",
            counts={"batches_completed": 0, "batches_total": 4},
        )
    )
    sink.emit(
        ProgressEvent(
            "job",
            "graph",
            "graph_extraction",
            "progress",
            counts={
                "batches_completed": 2,
                "batches_total": 4,
                "batch_entities": 7,
                "batch_relations": 5,
            },
        )
    )
    now[0] = 105.0

    sink.console.print(sink.render())
    rendered = stream.getvalue()

    assert "FlakeGraph" in rendered
    assert "qwen3:4b-instruct" in rendered
    assert "1/3 files" in rendered
    assert "4 pages" in rendered
    assert "2/4 LLM batches" in rendered
    assert "7 entities · 5 relations" in rendered
    assert "2 concurrent" in rendered
    assert "Elapsed 5s" in rendered


def test_rich_terminal_progress_renders_resolution_without_losing_extraction_counts() -> None:
    """Ensure resolution detail replaces display text without corrupting extraction counters.

    Both phases share one terminal stage intentionally.
    """

    stream = io.StringIO()
    sink = RichTerminalProgressSink(
        _terminal_context(),
        console=Console(file=stream, color_system=None, width=110),
        live_enabled=False,
    )
    sink.emit(
        ProgressEvent(
            "job",
            "graph",
            "graph_extraction",
            "started",
            counts={"batches_completed": 0, "batches_total": 11},
        )
    )
    sink.emit(
        ProgressEvent(
            "job",
            "graph",
            "graph_extraction",
            "progress",
            counts={
                "batches_completed": 11,
                "batches_total": 11,
                "batch_entities": 120,
                "batch_relations": 45,
            },
        )
    )
    sink.emit(
        ProgressEvent(
            "job",
            "graph",
            "graph_extraction",
            "progress",
            counts={
                "resolution_batches_completed": 1,
                "resolution_batches_total": 2,
                "resolution_candidates": 24,
                "resolution_failed_batches": 0,
            },
            metadata={"phase": "entity_resolution", "parallelism": 2},
        )
    )

    sink.console.print(sink.render())
    rendered = stream.getvalue()

    assert "1/2 resolution batches" in rendered
    assert "24 candidates" in rendered
    assert "2 concurrent" in rendered
    assert sink._states["graph_extraction"].counts["batches_completed"] == 11


def test_rich_terminal_progress_resets_file_counters_between_queue_batches() -> None:
    """A reused Snowflake queue sink should display only the active batch totals."""

    sink = RichTerminalProgressSink(_terminal_context(), live_enabled=False)
    for file_id in ("first", "second"):
        sink.emit(
            ProgressEvent(
                "job",
                "graph",
                "file_source",
                "completed",
                counts={"files_seen": 2},
            )
        )
        sink.emit(ProgressEvent("job", "graph", "ocr", "started", file_id=file_id))
        sink.emit(ProgressEvent("job", "graph", "ocr", "completed", file_id=file_id))
    assert sink._ocr_files_completed == 2

    sink.emit(ProgressEvent("job", "graph", "file_source", "started"))
    sink.emit(
        ProgressEvent(
            "job",
            "graph",
            "file_source",
            "completed",
            counts={"files_seen": 1},
        )
    )
    sink.emit(ProgressEvent("job", "graph", "ocr", "started", file_id="third"))
    sink.emit(ProgressEvent("job", "graph", "ocr", "completed", file_id="third"))

    assert sink._ocr_files_completed == 1
    assert sink._states["ocr"].status == "completed"


def test_rich_terminal_progress_keeps_ocr_failure_visible_after_later_files() -> None:
    """Do not overwrite a failed file with a green aggregate stage status."""

    stream = io.StringIO()
    sink = RichTerminalProgressSink(
        _terminal_context(),
        console=Console(file=stream, color_system=None, width=110),
        live_enabled=False,
    )
    sink.emit(ProgressEvent("job", "graph", "file_source", "completed", counts={"files_seen": 2}))
    sink.emit(ProgressEvent("job", "graph", "ocr", "started", file_id="bad"))
    sink.emit(ProgressEvent("job", "graph", "ocr", "failed", file_id="bad"))
    sink.emit(ProgressEvent("job", "graph", "ocr", "started", file_id="good"))
    sink.emit(ProgressEvent("job", "graph", "ocr", "completed", file_id="good"))

    sink.console.print(sink.render())

    assert sink._states["ocr"].status == "failed"
    assert "1 failed" in stream.getvalue()


def test_rich_terminal_progress_finishes_with_concise_artifact_summary() -> None:
    stream = io.StringIO()
    now = [20.0]
    sink = RichTerminalProgressSink(
        _terminal_context(),
        console=Console(file=stream, color_system=None, width=100),
        clock=lambda: now[0],
        live_enabled=False,
    )
    sink.start()
    now[0] = 85.0

    sink.finish(
        {
            "files_processed": 6,
            "chunks_created": 19,
            "nodes_created": 45,
            "edges_created": 37,
            "communities_created": 22,
        }
    )
    rendered = stream.getvalue()

    assert "Graph ready in 1m 05s" in rendered
    assert "6 files   19 chunks   45 nodes   37 edges   22 communities" in rendered
    assert "Artifacts out/test-graph" in rendered


def test_rich_terminal_progress_redacts_provider_failure_urls() -> None:
    stream = io.StringIO()
    sink = RichTerminalProgressSink(
        _terminal_context(),
        console=Console(file=stream, color_system=None, width=100),
        live_enabled=False,
    )
    sink.emit(ProgressEvent("job", "graph", "graph_extraction", "started"))

    sink.fail(
        RuntimeError("request failed at https://storage.example/input.pdf?sv=2024&sig=secret&sp=r")
    )
    rendered = stream.getvalue()

    assert "FlakeGraph failed" in rendered
    assert "sig=***" in rendered
    assert "secret" not in rendered


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


class FailingSink:
    """Raise one credential-bearing telemetry error for isolation tests."""

    def emit(self, event: ProgressEvent) -> None:
        """Fail without affecting sibling sinks."""

        _ = event
        raise RuntimeError("password=telemetry-secret")


def _terminal_context() -> WorkerProgressContext:
    return WorkerProgressContext(
        job_id="job",
        graph_id="graph",
        ocr_provider="builtin_text",
        llm_model="qwen3:4b-instruct",
        embedding_model="nomic-embed-text",
        writer_provider="local_artifacts",
        output_path=Path("out/test-graph"),
        extraction_parallelism=2,
        community_parallelism=4,
    )


def test_progress_event_redacts_credentials_inside_its_message() -> None:
    """An adapter may pass an exception string through as the message.

    Progress records reach container logs and Snowflake job tables, so a DSN or
    bearer token quoted by a failing client must be masked there too, not only
    in the structured metadata.
    """

    event = ProgressEvent(
        job_id="job",
        graph_id="graph",
        stage="ocr",
        status="failed",
        message="OCR call failed: Authorization: Bearer abc.def.ghi at https://ocr.example/?sig=SECRET",
    )

    record = event.as_log_record()

    assert "SECRET" not in str(record["message"])
    assert "abc.def.ghi" not in str(record["message"])
    assert "OCR call failed" in str(record["message"])
