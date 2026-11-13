"""Tests for durable post-extraction LLM checkpoints."""

from pathlib import Path

from llm_fakes import CountingLlm

from kg_processor.adapters.cache.local_json import LocalJsonCache
from kg_processor.application.enrichment_cache import CachedEnrichmentLlmProvider
from kg_processor.ports.llm import DescriptionMergeRequest


def test_enrichment_wrapper_reuses_description_result_after_restart(tmp_path: Path) -> None:
    cache = LocalJsonCache(tmp_path)
    delegate = CountingLlm()
    request = DescriptionMergeRequest(
        entity_name="Alice",
        entity_type="PERSON",
        descriptions=["Alice", "Alice is an engineer."],
        evidence=["Alice works at Acme."],
    )

    first = CachedEnrichmentLlmProvider(
        delegate,
        cache,
        graph_id="graph",
        provider="fake",
        model="fake",
    ).merge_entity_description(request)
    second = CachedEnrichmentLlmProvider(
        delegate,
        cache,
        graph_id="graph",
        provider="fake",
        model="fake",
    ).merge_entity_description(request)

    assert first == second
    assert delegate.description_calls == 1
