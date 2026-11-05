# SPDX-License-Identifier: Apache-2.0
"""Durable cache wrapper for post-extraction LLM enrichment calls."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from kg_processor.domain.ids import sha256_hex, stable_id
from kg_processor.ports.cache import EnrichmentCacheKey, PipelineCache
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

_ResultT = TypeVar("_ResultT", bound=BaseModel)


class CachedEnrichmentLlmProvider:
    """Checkpoint normalized enrichment responses without caching extraction calls."""

    def __init__(
        self,
        delegate: LlmProvider,
        cache: PipelineCache,
        *,
        graph_id: str,
        provider: str,
        model: str,
    ) -> None:
        self.delegate = delegate
        self.cache = cache
        self.graph_id = graph_id
        self.provider = provider
        self.model = model

    def capabilities(self) -> LlmCapabilities:
        """Delegate structured-output capability discovery."""

        return self.delegate.capabilities()

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Cache batched strict description/community completions."""

        return self._cached_model_result(
            request.task_name,
            request.model_dump(mode="json"),
            StructuredCompletionResult,
            lambda: self.delegate.complete_structured(request),
        )

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        """Cache a provider-native entity-description merge."""

        return self._cached_model_result(
            "description_merge",
            request.model_dump(mode="json"),
            DescriptionMergeResult,
            lambda: self.delegate.merge_entity_description(request),
        )

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        """Cache a provider-native community report."""

        return self._cached_model_result(
            "community_report",
            request.model_dump(mode="json"),
            CommunitySummaryResult,
            lambda: self.delegate.summarize_community(request),
        )

    def _cached_model_result(
        self,
        stage: str,
        payload: dict[str, Any],
        result_type: type[_ResultT],
        produce: Callable[[], _ResultT],
    ) -> _ResultT:
        options_hash = sha256_hex(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        key = EnrichmentCacheKey(
            id=stable_id(
                "enrichment_cache",
                self.graph_id,
                stage,
                self.provider,
                self.model,
                options_hash,
            ),
            graph_id=self.graph_id,
            stage=stage,
            llm_provider=self.provider,
            model=self.model,
            options_hash=options_hash,
        )
        cached = self.cache.get_enrichment_result(key)
        if cached is not None:
            return result_type.model_validate(cached)
        result = produce()
        self.cache.put_enrichment_result(key, result.model_dump(mode="json"))
        return result
