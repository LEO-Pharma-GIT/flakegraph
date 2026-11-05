# SPDX-License-Identifier: Apache-2.0
"""Record what every LLM call consumed, without touching the call sites.

Metering belongs at the provider boundary rather than in extraction, resolution
and enrichment separately: every one of those paths ends in the same three
methods, and instrumenting each caller would leave whichever path is added next
silently unmetered.

The wrapper is placed *outside* the enrichment cache, so a cached response — which
called nobody — records nothing.

A call that fails after exhausting its repair budget was still billed for every
attempt. Providers carry that usage on the failure, and it is recorded before the
failure is re-raised, so an expensive window is never reported as free.
"""

from __future__ import annotations

from kg_processor.application.consumption import ConsumptionCollector
from kg_processor.domain.consumption import TokenUsage
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    CommunitySummaryResult,
    DescriptionMergeRequest,
    DescriptionMergeResult,
    LlmCapabilities,
    LlmProvider,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)


class MeteredLlmProvider:
    """Delegate to a provider and record the usage of each completed call."""

    def __init__(
        self,
        delegate: LlmProvider,
        collector: ConsumptionCollector,
        *,
        provider: str,
        model: str,
    ) -> None:
        self.delegate = delegate
        self.collector = collector
        self.provider = provider
        self.model = model

    def close(self) -> None:
        """Release the reusable provider client when its owning pipeline closes."""

        close = getattr(self.delegate, "close", None)
        if callable(close):
            close()

    def capabilities(self) -> LlmCapabilities:
        """Delegate capability discovery, which consumes nothing."""

        return self.delegate.capabilities()

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Record one structured extraction call against its own task name.

        The task name is used as the stage so that spend can be attributed to
        entity windows, relation windows and compaction separately rather than
        appearing as one undifferentiated extraction total.
        """

        stage = request.task_name or "extraction"
        model = request.model or self.model
        try:
            result = self.delegate.complete_structured(request)
        except Exception as error:
            self._record_billed_failure(error, stage=stage, model=model)
            raise
        self.collector.record_completion(
            stage=stage,
            provider=self.provider,
            model=model,
            usage=result.usage,
            metadata=result.provider_metadata,
        )
        return result

    def _record_billed_failure(self, error: Exception, *, stage: str, model: str) -> None:
        """Record what a failed call consumed before its failure propagates."""

        usage = getattr(error, "usage", None)
        if not isinstance(usage, TokenUsage) or not usage.total_tokens:
            return
        self.collector.record_completion(
            stage=stage,
            provider=self.provider,
            model=model,
            usage=usage,
            metadata={"outcome": "failed"},
        )

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        """Record one description-merge call."""

        try:
            result = self.delegate.merge_entity_description(request)
        except Exception as error:
            self._record_billed_failure(error, stage="description_merge", model=self.model)
            raise
        self.collector.record_completion(
            stage="description_merge",
            provider=self.provider,
            model=self.model,
            usage=result.usage,
            metadata=result.provider_metadata,
        )
        return result

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        """Record one community-report call."""

        try:
            result = self.delegate.summarize_community(request)
        except Exception as error:
            self._record_billed_failure(error, stage="community_report", model=self.model)
            raise
        self.collector.record_completion(
            stage="community_report",
            provider=self.provider,
            model=self.model,
            usage=result.usage,
            metadata=result.provider_metadata,
        )
        return result
