# SPDX-License-Identifier: Apache-2.0
"""Collect what a run consumed while it runs.

Only real provider work is recorded. A cache hit returns the same artifact
without calling anybody, so recording it would inflate spend for work that was
never billed and would make a second run of an unchanged corpus look as expensive
as the first — destroying the comparison the cache exists to enable.

Collection is thread-safe because extraction fans windows out concurrently.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping, Sequence

from kg_processor.config.provider_registry import provider_profile
from kg_processor.domain.consumption import (
    ConsumptionEvent,
    ConsumptionTotals,
    Locality,
    RateCard,
    TokenUsage,
    group_by,
    summarize,
)

# Providers that run on infrastructure we own. Their calls are recorded exactly
# like hosted ones — the tokens are what make the avoided cost measurable — but
# they accrue avoided rather than billed cost. Read off the registry rather
# than listed again here, so a provider cannot be billed for want of a copy.
_LOCAL_PROFILES = frozenset({"open_source", "local", "test"})


def locality_for(provider: str) -> Locality:
    """Classify a provider by where its work actually runs."""

    return Locality.LOCAL if provider_profile(provider) in _LOCAL_PROFILES else Locality.HOSTED


class ConsumptionCollector:
    """Accumulate consumption events for one run."""

    def __init__(
        self,
        graph_id: str,
        job_id: str = "",
        *,
        run_id: str = "",
        task_id: str = "",
        task_attempt: int = 0,
        sink: Callable[[ConsumptionEvent], None] | None = None,
    ) -> None:
        self.graph_id = graph_id
        self.job_id = job_id
        self.run_id = run_id
        self.task_id = task_id
        self.task_attempt = task_attempt
        self._sink = sink
        self._event_ids: set[str] = set()
        self._events: list[ConsumptionEvent] = []
        self._lock = threading.Lock()

    def record(
        self,
        *,
        stage: str,
        operation: str,
        provider: str,
        model: str,
        usage: TokenUsage | None = None,
        pages: int = 0,
        file_id: str | None = None,
    ) -> None:
        """Record one unit of provider work that actually executed."""

        event = ConsumptionEvent(
            graph_id=self.graph_id,
            job_id=self.job_id,
            run_id=self.run_id,
            task_id=self.task_id,
            task_attempt=self.task_attempt,
            stage=stage,
            operation=operation,
            provider=provider,
            model=model,
            locality=locality_for(provider),
            usage=usage or TokenUsage(),
            pages=pages,
            file_id=file_id,
        )
        self.accept(event)

    def accept(self, event: ConsumptionEvent) -> None:
        """Persist one measured call before exposing it in the in-process report.

        An immutable event identity lets durable sinks retry delivery without
        counting one provider call twice. A failed sink propagates: completing
        a task without its billed usage would silently lose accounting.
        """

        if self._sink is not None:
            self._sink(event)
        self.merge([event])

    def merge(self, events: Iterable[ConsumptionEvent]) -> None:
        """Merge persisted events once without writing them back to their sink."""

        with self._lock:
            for event in events:
                if event.graph_id != self.graph_id or event.job_id != self.job_id:
                    raise ValueError("consumption event belongs to another graph or run")
                if event.event_id not in self._event_ids:
                    self._events.append(event)
                    self._event_ids.add(event.event_id)

    def bind_sink(self, sink: Callable[[ConsumptionEvent], None]) -> None:
        """Attach the run's durable accounting boundary before executing tasks."""

        self._sink = sink

    def for_task(self, task_id: str, attempt: int) -> ConsumptionCollector:
        """Bind concurrent provider calls to one task attempt and the parent run."""

        return ConsumptionCollector(
            self.graph_id,
            self.job_id,
            run_id=self.run_id,
            task_id=task_id,
            task_attempt=attempt,
            sink=self.accept,
        )

    def record_completion(
        self,
        *,
        stage: str,
        provider: str,
        model: str,
        metadata: Mapping[str, object] | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        """Record one LLM call, reading usage from the result or its metadata."""

        # An all-zero usage means the adapter reported nothing, not that the call
        # was free. Falling back to the provider's own metadata keeps a forgotten
        # ``usage=`` from silently costing nothing — which is indistinguishable
        # from a genuinely free call once it reaches a total.
        resolved = usage if usage and usage.total_tokens else None
        if resolved is None and metadata is not None:
            resolved = TokenUsage.from_provider_payload(metadata.get("usage")) or usage
        self.record(
            stage=stage,
            operation="completion",
            provider=provider,
            model=model,
            usage=resolved,
        )

    @property
    def events(self) -> Sequence[ConsumptionEvent]:
        """Return the events recorded so far."""

        with self._lock:
            return list(self._events)

    def running_totals(self, card: RateCard) -> dict[str, object]:
        """Summarize what the run has spent so far, without its event detail.

        The full report carries every recorded call so the answer can be
        re-derived later. That is the right shape to persist once beside a
        finished graph and the wrong shape to write every few seconds while the
        run is still going, so progress reporting takes the totals alone.
        """

        return dict(summarize(self.events, card).model_dump(mode="json"))

    def report(self, card: RateCard, *, event_limit: int | None = None) -> dict[str, object]:
        """Summarize the run for persistence beside its graph.

        Both the totals and the events are returned. Totals answer "what did this
        graph cost"; the events are what let that answer be re-derived when the
        rate table is corrected, or broken down by stage, provider or model
        without re-running anything. ``event_limit`` bounds serialized detail
        only; every event still participates in every aggregate.
        """

        if event_limit is not None and event_limit < 0:
            raise ValueError("consumption event detail limit must not be negative")
        events = self.events
        totals: ConsumptionTotals = summarize(events, card)
        detail = events if event_limit is None else events[:event_limit]
        report: dict[str, object] = {
            "totals": totals.model_dump(mode="json"),
            "by_stage": {
                name: value.model_dump(mode="json")
                for name, value in group_by(events, "stage", card).items()
            },
            "by_provider": {
                name: value.model_dump(mode="json")
                for name, value in group_by(events, "provider", card).items()
            },
            "by_model": {
                name: value.model_dump(mode="json")
                for name, value in group_by(events, "model", card).items()
            },
            "by_locality": {
                name: value.model_dump(mode="json")
                for name, value in group_by(events, "locality", card).items()
            },
            "events": [event.model_dump(mode="json") for event in detail],
        }
        if event_limit is not None:
            report.update(event_count=len(events), events_truncated=len(detail) < len(events))
        return report
