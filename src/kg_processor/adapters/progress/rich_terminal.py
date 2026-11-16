# SPDX-License-Identifier: Apache-2.0
"""Rich live terminal presentation for long-running FlakeGraph workers.

The pipeline emits provider-neutral ``ProgressEvent`` values. This adapter owns
all terminal-specific state, labels, colors, and rendering so application code
does not acquire a dependency on Rich or assumptions about interactive shells.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Literal

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from kg_processor.application.progress import ProgressEvent
from kg_processor.application.redaction import redact_sensitive_data, redact_sensitive_text

StageStatus = Literal["pending", "active", "completed", "failed"]


@dataclass(frozen=True)
class WorkerProgressContext:
    """Non-sensitive run metadata displayed above and below the live stages."""

    job_id: str
    graph_id: str
    ocr_provider: str
    llm_model: str
    embedding_model: str
    writer_provider: str
    output_path: Path
    extraction_parallelism: int
    community_parallelism: int


@dataclass(frozen=True)
class _StageSpec:
    """Stable stage key and concise operator-facing label."""

    key: str
    label: str


@dataclass
class _StageState:
    """Mutable display state accumulated from one or more progress events."""

    status: StageStatus = "pending"
    started_at: float | None = None
    elapsed_seconds: float | None = None
    counts: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    latest_file: str | None = None


_STAGES = (
    _StageSpec("file_source", "Discover files"),
    _StageSpec("ocr", "Read documents"),
    _StageSpec("chunking", "Create chunks"),
    _StageSpec("chunk_embeddings", "Embed chunks"),
    _StageSpec("graph_extraction", "Extract graph"),
    _StageSpec("filtering", "Validate evidence"),
    _StageSpec("merge", "Merge graph"),
    _StageSpec("graph_embeddings", "Embed graph"),
    _StageSpec("community_detection", "Build communities"),
    _StageSpec("quality", "Check quality"),
    _StageSpec("write", "Write artifacts"),
)
_STAGE_INDEX = {stage.key: index for index, stage in enumerate(_STAGES)}
_SECONDS_PER_MINUTE = 60


class RichTerminalProgressSink:
    """Render pipeline events as a compact, continuously updating stage view.

    The sink is thread-safe because extraction and community callbacks can be
    produced while provider calls are running concurrently. Rich's refresh
    thread and event updates therefore share one small protected state model.
    """

    def __init__(
        self,
        context: WorkerProgressContext,
        *,
        console: Console | None = None,
        clock: Callable[[], float] = perf_counter,
        live_enabled: bool = True,
    ) -> None:
        """Create a terminal sink without starting screen updates yet."""

        self.context = context
        self.console = console or Console(stderr=True)
        self.clock = clock
        self.live_enabled = live_enabled
        self._states = {stage.key: _StageState() for stage in _STAGES}
        self._lock = RLock()
        self._spinner = Spinner("dots", style="bold cyan")
        self._started_at = self.clock()
        self._live: Live | None = None
        self._started = False
        self._stopped = False
        self._ocr_files_completed = 0
        self._ocr_files_failed = 0

    def start(self) -> None:
        """Start the live refresh loop once provider construction begins."""

        with self._lock:
            if self._started or self._stopped:
                return
            self._started = True
            if not self.live_enabled:
                return
            self._live = Live(
                console=self.console,
                get_renderable=self.render,
                refresh_per_second=8,
                transient=False,
            )
            live = self._live
        live.start()

    def emit(self, event: ProgressEvent) -> None:
        """Apply one structured pipeline event and refresh the terminal view."""

        if event.stage not in self._states:
            return
        self.start()
        with self._lock:
            self._apply_event(event)
            live = self._live
        if live is not None:
            live.refresh()

    def render(self) -> RenderableType:
        """Build the current header, stage table, and aggregate footer."""

        with self._lock:
            return Group(self._header(), self._stage_table(), self._footer())

    def finish(self, report: Mapping[str, Any]) -> None:
        """Stop live rendering and print a concise successful-run summary."""

        with self._lock:
            for state in self._states.values():
                if state.status == "active":
                    state.status = "completed"
            live = self._stop_locked()
        if live is not None:
            live.stop()
        self.console.print(self._success_panel(report))

    def fail(self, exc: Exception) -> None:
        """Stop live rendering and print a redacted failure at the active stage."""

        with self._lock:
            for state in self._states.values():
                if state.status == "active":
                    state.status = "failed"
            live = self._stop_locked()
        if live is not None:
            live.stop()
        message = redact_sensitive_text(str(exc)) or type(exc).__name__
        body = Text()
        body.append("Worker stopped: ", style="bold red")
        body.append(message)
        self.console.print(Panel(body, title="[bold red]FlakeGraph failed[/]", border_style="red"))

    def _stop_locked(self) -> Live | None:
        """Detach the live renderer while the caller holds the state lock."""

        if self._stopped:
            return None
        self._stopped = True
        return self._live

    def _apply_event(self, event: ProgressEvent) -> None:
        """Fold a pipeline event into its aggregate stage state."""

        if event.stage == "file_source" and event.status == "started":
            self._reset_for_next_batch()
        state = self._states[event.stage]
        self._complete_prior_active_stages(event.stage)
        state.metadata.update(redact_sensitive_data(dict(event.metadata)) if event.metadata else {})
        source_uri = event.metadata.get("source_uri")
        if event.file_id and (source_uri or event.status == "started"):
            state.latest_file = _short_file_name(str(source_uri or event.file_id))

        if event.stage == "ocr":
            self._apply_ocr_event(state, event)
        elif event.status == "progress":
            self._apply_incremental_event(state, event)
        else:
            state.counts.update(event.counts)
            self._apply_status(state, event)

    def _complete_prior_active_stages(self, stage_key: str) -> None:
        """Close an implicit stage when the pipeline advances to a later one."""

        current_index = _STAGE_INDEX[stage_key]
        for stage in _STAGES[:current_index]:
            state = self._states[stage.key]
            if state.status == "active":
                state.status = "completed"
                if state.started_at is not None and state.elapsed_seconds is None:
                    state.elapsed_seconds = self.clock() - state.started_at

    def _apply_ocr_event(self, state: _StageState, event: ProgressEvent) -> None:
        """Aggregate repeated per-file OCR events into one document stage."""

        if event.status == "started":
            self._activate(state)
            return
        if event.status == "completed":
            self._ocr_files_completed += 1
            _add_numeric_counts(state.counts, event.counts)
        elif event.status == "failed":
            self._ocr_files_failed += 1
            state.status = "failed"
        if event.elapsed_ms is not None:
            state.elapsed_seconds = (state.elapsed_seconds or 0.0) + event.elapsed_ms / 1000
        files_total = _count(self._states["file_source"].counts, "files_seen")
        files_done = self._ocr_files_completed + self._ocr_files_failed
        if files_total > 0 and files_done >= files_total:
            state.status = "failed" if self._ocr_files_failed else "completed"

    def _reset_for_next_batch(self) -> None:
        """Reset per-pipeline counters when one sink serves consecutive file batches.

        Snowflake queue workers intentionally reuse their terminal sink. A new
        file-discovery start after prior progress marks a separate pipeline batch,
        so OCR totals and stage states must not carry over.
        """

        if all(state.status == "pending" for state in self._states.values()):
            return
        self._states = {stage.key: _StageState() for stage in _STAGES}
        self._ocr_files_completed = 0
        self._ocr_files_failed = 0
        self._started_at = self.clock()

    def _apply_incremental_event(self, state: _StageState, event: ProgressEvent) -> None:
        """Merge an incremental callback without completing its parent stage.

        Extraction and entity resolution share one visual stage but maintain
        separate counters so callbacks from one phase cannot overwrite or inflate
        the progress of the other.
        """

        self._activate(state)
        if event.stage == "graph_extraction":
            if "resolution_batches_total" in event.counts:
                # Resolution follows extraction within the same application
                # stage. Keep its counters separate so 11 completed extraction
                # windows do not become, for example, 2 resolution batches.
                state.counts.update(event.counts)
                return
            state.counts["batches_completed"] = event.counts.get("batches_completed", 0)
            state.counts["batches_total"] = event.counts.get("batches_total", 0)
            state.counts["entities_so_far"] = _count(state.counts, "entities_so_far") + _count(
                event.counts, "batch_entities"
            )
            state.counts["relations_so_far"] = _count(state.counts, "relations_so_far") + _count(
                event.counts, "batch_relations"
            )
            return
        state.counts.update(event.counts)

    def _apply_status(self, state: _StageState, event: ProgressEvent) -> None:
        """Apply a normal started, completed, or failed lifecycle event."""

        if event.status == "started":
            self._activate(state)
        elif event.status == "completed":
            state.status = "completed"
        elif event.status == "failed":
            state.status = "failed"
        if event.elapsed_ms is not None:
            state.elapsed_seconds = event.elapsed_ms / 1000

    def _activate(self, state: _StageState) -> None:
        """Mark a stage active while preserving its first start timestamp."""

        state.status = "active"
        if state.started_at is None:
            state.started_at = self.clock()

    def _header(self) -> RenderableType:
        """Render graph identity and provider choices without exposing credentials."""

        title = Text()
        title.append("Building ", style="dim")
        title.append(self.context.graph_id, style="bold white")
        title.append("  ")
        title.append(self.context.job_id, style="dim")
        providers = Text()
        providers.append("OCR ", style="dim")
        providers.append(self.context.ocr_provider, style="cyan")
        providers.append("   LLM ", style="dim")
        providers.append(self.context.llm_model, style="cyan")
        providers.append("   Embed ", style="dim")
        providers.append(self.context.embedding_model, style="cyan")
        return Panel(
            Group(title, providers),
            title="[bold cyan]FlakeGraph[/]",
            border_style="cyan",
            padding=(0, 1),
        )

    def _stage_table(self) -> Table:
        """Render every pipeline stage with stable columns and live timing."""

        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(width=2, no_wrap=True)
        table.add_column(width=21, no_wrap=True)
        table.add_column(ratio=1, overflow="ellipsis")
        table.add_column(width=9, justify="right", no_wrap=True)
        for spec in _STAGES:
            state = self._states[spec.key]
            row_style = "dim" if state.status == "pending" else ""
            table.add_row(
                self._status_icon(state.status),
                Text(spec.label, style="bold" if state.status == "active" else ""),
                self._stage_detail(spec.key, state),
                Text(
                    self._stage_elapsed(state),
                    style="cyan" if state.status == "active" else "dim",
                ),
                style=row_style,
            )
        return table

    def _status_icon(self, status: StageStatus) -> RenderableType:
        """Return a visually distinct status marker for one stage."""

        if status == "active":
            return self._spinner
        if status == "completed":
            return Text("✓", style="bold green")
        if status == "failed":
            return Text("×", style="bold red")
        return Text("·", style="grey50")

    def _stage_detail(self, stage_key: str, state: _StageState) -> Text:
        """Translate structured counts into concise stage-specific language."""

        renderers: dict[str, Callable[[_StageState], Text]] = {
            "file_source": self._file_source_detail,
            "ocr": self._ocr_detail,
            "chunking": self._chunk_detail,
            "chunk_embeddings": self._chunk_detail,
            "graph_extraction": self._extraction_detail,
            "filtering": self._filter_detail,
            "merge": self._graph_detail,
            "graph_embeddings": self._graph_detail,
            "community_detection": self._community_detail,
            "quality": self._quality_detail,
            "write": self._write_detail,
        }
        renderer = renderers.get(stage_key)
        return renderer(state) if renderer is not None else Text()

    def _file_source_detail(self, state: _StageState) -> Text:
        """Render the number of input files selected for this run."""

        return _parts((_count(state.counts, "files_seen"), "files"))

    def _ocr_detail(self, state: _StageState) -> Text:
        """Render document progress, parsed pages, and the active file name."""

        total = _count(self._states["file_source"].counts, "files_seen")
        done = self._ocr_files_completed + self._ocr_files_failed
        text = _ratio_text(done, total, "files")
        pages = _count(state.counts, "pages")
        if pages:
            text.append(f"   {pages} pages", style="dim")
        if self._ocr_files_failed:
            text.append(f"   {self._ocr_files_failed} failed", style="bold red")
        if state.status == "active" and state.latest_file:
            text.append(f"   {state.latest_file}", style="cyan")
        return text

    def _chunk_detail(self, state: _StageState) -> Text:
        """Render chunk counts for chunk creation and chunk embeddings."""

        return _parts((_count(state.counts, "chunks"), "chunks"))

    def _extraction_detail(self, state: _StageState) -> Text:
        """Render the active extraction or resolution phase with concurrency.

        Resolution takes precedence once its counters appear; otherwise the view
        reports extraction windows and accumulated entity/relation observations.
        """

        counts = state.counts
        resolution_total = _count(counts, "resolution_batches_total")
        if state.status == "active" and resolution_total:
            text = _progress_text(
                _count(counts, "resolution_batches_completed"),
                resolution_total,
                "resolution batches",
            )
            text.append(
                f"   {_count(counts, 'resolution_candidates')} candidates",
                style="dim",
            )
            failed = _count(counts, "resolution_failed_batches")
            if failed:
                text.append(f"   {failed} kept separate", style="yellow")
            parallelism = _count(state.metadata, "parallelism") or 1
            if parallelism > 1:
                text.append(f"   {parallelism} concurrent", style="dim cyan")
            return text
        completed = _count(counts, "batches_completed")
        total = _count(counts, "batches_total")
        if state.status == "active" and total:
            text = _progress_text(completed, total, "LLM batches")
            entities = _count(counts, "entities_so_far")
            relations = _count(counts, "relations_so_far")
            text.append(f"   {entities} entities · {relations} relations", style="dim")
            if self.context.extraction_parallelism > 1:
                text.append(
                    f"   {self.context.extraction_parallelism} concurrent",
                    style="dim cyan",
                )
            return text
        return _entity_relation_text(counts, "entities", "relations")

    def _filter_detail(self, state: _StageState) -> Text:
        """Render grounded entity and relation counts retained by validation."""

        return _entity_relation_text(state.counts, "entities_after", "relations_after")

    def _graph_detail(self, state: _StageState) -> Text:
        """Render canonical graph counts used by merge and graph embeddings."""

        return _node_edge_text(state.counts)

    def _community_detail(self, state: _StageState) -> Text:
        """Render community report sub-progress or final community counts."""

        completed = _count(state.counts, "reports_completed")
        total = _count(state.counts, "reports_total")
        if state.status == "active" and total:
            text = _progress_text(completed, total, "reports")
            if self.context.community_parallelism > 1:
                text.append(
                    f"   {self.context.community_parallelism} concurrent",
                    style="dim cyan",
                )
            return text
        return _parts(
            (_count(state.counts, "communities"), "communities"),
            (_count(state.counts, "findings"), "findings"),
        )

    def _quality_detail(self, state: _StageState) -> Text:
        """Render a plain-language quality-gate result."""

        if state.status == "pending":
            return Text()
        failed = _count(state.counts, "failed_checks")
        return Text("all checks passed" if failed == 0 else f"{failed} checks failed")

    def _write_detail(self, state: _StageState) -> Text:
        """Render the principal graph rows being persisted by the writer."""

        return _parts(
            (_count(state.counts, "nodes"), "nodes"),
            (_count(state.counts, "edges"), "edges"),
            (_count(state.counts, "communities"), "communities"),
        )

    def _stage_elapsed(self, state: _StageState) -> str:
        """Return completed or continuously increasing elapsed stage time."""

        elapsed = state.elapsed_seconds
        if state.status == "active" and state.started_at is not None:
            elapsed = self.clock() - state.started_at
        return _format_duration(elapsed) if elapsed is not None else ""

    def _footer(self) -> RenderableType:
        """Render aggregate run duration and the most useful graph counters."""

        elapsed = _format_duration(self.clock() - self._started_at)
        file_counts = self._states["file_source"].counts
        chunk_counts = self._states["chunking"].counts
        graph_counts = self._states["merge"].counts
        text = Text()
        text.append("  Elapsed ", style="dim")
        text.append(elapsed, style="bold")
        for value, label in (
            (_count(file_counts, "files_seen"), "files"),
            (_count(chunk_counts, "chunks"), "chunks"),
            (_count(graph_counts, "nodes"), "nodes"),
            (_count(graph_counts, "edges"), "edges"),
        ):
            if value:
                text.append(f"   {value} {label}", style="dim")
        return text

    def _success_panel(self, report: Mapping[str, Any]) -> Panel:
        """Render the final human-readable summary for an interactive run."""

        body = Text()
        body.append("Graph ready", style="bold green")
        body.append(f" in {_format_duration(self.clock() - self._started_at)}\n", style="dim")
        body.append(
            f"{_count(report, 'files_processed')} files   "
            f"{_count(report, 'chunks_created')} chunks   "
            f"{_count(report, 'nodes_created')} nodes   "
            f"{_count(report, 'edges_created')} edges   "
            f"{_count(report, 'communities_created')} communities"
        )
        destination = (
            str(self.context.output_path)
            if self.context.writer_provider == "local_artifacts"
            else self.context.writer_provider
        )
        body.append("\nArtifacts ", style="dim")
        body.append(destination, style="cyan")
        return Panel(body, border_style="green", padding=(0, 1))


def _count(values: Mapping[str, object], key: str) -> int:
    """Read a numeric event counter while rejecting booleans and strings."""

    value = values.get(key, 0)
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0


def _add_numeric_counts(target: dict[str, object], values: Mapping[str, object]) -> None:
    """Add numeric values from repeated per-file events into aggregate counts."""

    for key in values:
        target[key] = _count(target, key) + _count(values, key)


def _parts(*items: tuple[int, str]) -> Text:
    """Join non-zero count/label pairs into a muted compact detail string."""

    text = Text()
    for value, label in items:
        if not value:
            continue
        if text:
            text.append(" · ", style="dim")
        text.append(f"{value} {label}", style="dim")
    return text


def _ratio_text(completed: int, total: int, label: str) -> Text:
    """Render a completed/total counter when the total is known."""

    if total:
        return Text(f"{completed}/{total} {label}", style="dim")
    return Text(f"{completed} {label}", style="dim") if completed else Text()


def _progress_text(completed: int, total: int, label: str) -> Text:
    """Render a fixed-width mini progress bar and exact batch/report counts."""

    width = 12
    filled = min(width, round(width * completed / total)) if total else 0
    text = Text("━" * filled, style="cyan")
    text.append("━" * (width - filled), style="grey30")
    text.append(f"  {completed}/{total} {label}")
    return text


def _entity_relation_text(
    counts: Mapping[str, object],
    entity_key: str,
    relation_key: str,
) -> Text:
    """Render entity and relation counts shared by extraction and filtering."""

    return _parts(
        (_count(counts, entity_key), "entities"),
        (_count(counts, relation_key), "relations"),
    )


def _node_edge_text(counts: Mapping[str, object]) -> Text:
    """Render graph node, edge, and optional evidence counts."""

    return _parts(
        (_count(counts, "nodes"), "nodes"),
        (_count(counts, "edges"), "edges"),
        (_count(counts, "evidence"), "evidence"),
    )


def _short_file_name(value: str, max_length: int = 38) -> str:
    """Keep the current source readable without allowing it to widen the table."""

    name = value.rstrip("/").rsplit("/", maxsplit=1)[-1]
    return name if len(name) <= max_length else f"…{name[-(max_length - 1) :]}"


def _format_duration(seconds: float | None) -> str:
    """Format elapsed seconds compactly for stable terminal columns."""

    if seconds is None:
        return ""
    total_seconds = max(0, round(seconds))
    if total_seconds < 1:
        return "<1s"
    if total_seconds < _SECONDS_PER_MINUTE:
        return f"{total_seconds}s"
    minutes, remaining_seconds = divmod(total_seconds, _SECONDS_PER_MINUTE)
    if minutes < _SECONDS_PER_MINUTE:
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, _SECONDS_PER_MINUTE)
    return f"{hours}h {remaining_minutes:02d}m"
