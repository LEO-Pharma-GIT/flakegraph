# SPDX-License-Identifier: Apache-2.0
"""Structured progress events for long-running worker execution.

Run reports and extraction traces are written at the end of a successful batch.
These progress events are intentionally lighter: they give operators and the
host app live, JSON-parseable signals while OCR, LLM extraction, and writes are
still in flight.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, TextIO

from kg_processor.application.redaction import redact_sensitive_data, redact_sensitive_text


class ProgressSink(Protocol):
    """Destination for live progress events emitted by the pipeline."""

    def emit(self, event: ProgressEvent) -> None:
        """Publish one progress event."""
        ...


@dataclass(frozen=True)
class ProgressEvent:
    """JSON-serializable progress event for logs and Snowflake job rows."""

    job_id: str
    graph_id: str
    stage: str
    status: str
    file_id: str | None = None
    message: str | None = None
    counts: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    elapsed_ms: int | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def as_log_record(self) -> dict[str, object]:
        """Render the event as a redacted log record."""

        record: dict[str, object] = {
            "event": "kg_processor.progress",
            "timestamp": self.timestamp,
            "job_id": self.job_id,
            "graph_id": self.graph_id,
            "stage": self.stage,
            "status": self.status,
        }
        if self.file_id:
            record["file_id"] = self.file_id
        if self.message:
            # Messages are usually stage prose, but an adapter may pass an
            # exception string through, and those can quote a DSN or token.
            record["message"] = redact_sensitive_text(self.message)
        if self.counts:
            record["counts"] = dict(self.counts)
        if self.metadata:
            # Progress metadata is operator-facing and may be emitted to
            # container logs or Snowflake job tables, so redact it at the final
            # serialization boundary even if a provider accidentally adds a key.
            record["metadata"] = redact_sensitive_data(dict(self.metadata))
        if self.elapsed_ms is not None:
            record["elapsed_ms"] = self.elapsed_ms
        return record


class JsonLineProgressSink:
    """Writes progress events as newline-delimited JSON.

    A worker's slots emit from several threads; each line is written and
    flushed whole, so a log reader never sees two events spliced together.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream
        self._lock = threading.Lock()

    def emit(self, event: ProgressEvent) -> None:
        """Write one redacted progress event to the configured stream."""

        line = json.dumps(event.as_log_record(), sort_keys=True) + "\n"
        # Resolve stderr at emit time so test runners and container runtimes can
        # redirect/capture it without constructing the sink after redirection.
        stream = self.stream or sys.stderr
        with self._lock:
            stream.write(line)
            stream.flush()


class NullProgressSink:
    """Discard progress events when an operator explicitly disables output."""

    def emit(self, event: ProgressEvent) -> None:
        """Accept an event without publishing it anywhere."""

        _ = event


class CompositeProgressSink:
    """Fan progress events out to multiple durable or operational sinks.

    The pipeline should not need to know whether progress is being printed to
    container logs, persisted to Snowflake, or both. This small composite keeps
    that wiring at the process edge while preserving one progress event model.
    """

    def __init__(self, sinks: Sequence[ProgressSink]) -> None:
        self.sinks = list(sinks)

    def emit(self, event: ProgressEvent) -> None:
        """Forward one progress event to every configured sink."""

        for sink in self.sinks:
            # Progress is observational and must never terminate OCR/LLM work,
            # so one sink's outage is swallowed and delivery continues to the
            # independent sinks.
            with suppress(Exception):
                sink.emit(event)


def elapsed_ms(started_at: float, finished_at: float) -> int:
    """Return a non-negative rounded duration in milliseconds."""

    return max(0, round((finished_at - started_at) * 1000))


def error_metadata(exc: Exception) -> dict[str, Any]:
    """Return a small structured error payload for progress metadata."""

    return {"error_type": type(exc).__name__, "error_message": redact_sensitive_text(str(exc))}
