# SPDX-License-Identifier: Apache-2.0
"""LLM-assisted synthesis of canonical entity descriptions.

Descriptions are merged only after observations have survived filtering and
canonicalization. The trace records request sizes and provider metadata so a
operators can connect a node description back to the exact LLM prompt contract.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import batched
from typing import Any

from kg_processor.application.enrichment_batching import (
    DESCRIPTION_BATCH_SIZE,
    merge_description_requests,
)
from kg_processor.application.graph_merge import normalize_entity_name
from kg_processor.domain.graph import EntitySource, Evidence, ExtractedEntity, GraphNode
from kg_processor.ports.llm import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DescriptionMergeProvider,
    DescriptionMergeRequest,
    StructuredCompletionProvider,
)

MAX_DESCRIPTION_LENGTH = 1000


@dataclass(frozen=True)
class DescriptionSynthesisResult:
    """Count and trace events from optional entity-description synthesis."""

    merged_count: int
    trace_events: list[dict[str, Any]]


def synthesize_entity_descriptions(
    nodes: list[GraphNode],
    entity_sources: list[EntitySource],
    observations: list[ExtractedEntity],
    evidence: list[Evidence],
    llm: DescriptionMergeProvider,
    min_observations: int,
    max_descriptions: int,
    max_evidence: int,
    parallelism: int = 1,
    model: str = "default",
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
    seed: int | None = None,
) -> DescriptionSynthesisResult:
    """Use bounded batched LLM calls to synthesize repeated descriptions.

    Several independent entities share one strict provider round trip. Batches are
    collected in node order and applied on the orchestration thread, so concurrency
    changes latency without making trace or graph ordering nondeterministic.
    """

    if parallelism <= 0:
        raise ValueError("description merge parallelism must be positive")

    descriptions_by_key = _descriptions_by_node_key(observations)
    evidence_by_subject = _evidence_by_subject(evidence)
    sources_by_node_id = _sources_by_node_id(entity_sources)
    requests: list[tuple[GraphNode, list[str], list[str], DescriptionMergeRequest]] = []
    for node in nodes:
        # Same-named nodes are told apart by the chunks that support each of
        # them; descriptions keep the order in which they were observed.
        chunk_ids = set(node.source_chunk_ids)
        descriptions = list(
            dict.fromkeys(
                description
                for chunk_id, description in descriptions_by_key.get(
                    (node.normalized_name, node.primary_type), []
                )
                if chunk_id in chunk_ids
            )
        )
        if len(descriptions) < min_observations:
            continue
        descriptions_used = descriptions[:max_descriptions]
        evidence_used = evidence_by_subject.get(node.id, [])[:max_evidence]
        request = DescriptionMergeRequest(
            entity_name=node.name,
            entity_type=node.primary_type,
            descriptions=descriptions_used,
            evidence=evidence_used,
        )
        requests.append((node, descriptions, evidence_used, request))

    batch_size = DESCRIPTION_BATCH_SIZE if isinstance(llm, StructuredCompletionProvider) else 1
    request_batches = [list(items) for items in batched([item[3] for item in requests], batch_size)]
    with ThreadPoolExecutor(max_workers=min(parallelism, len(request_batches) or 1)) as executor:
        batch_results = list(
            executor.map(
                lambda items: merge_description_requests(
                    llm,
                    items,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    seed=seed,
                ),
                request_batches,
            )
        )
    results = [result for batch_result in batch_results for result in batch_result]

    merged_count = 0
    trace_events: list[dict[str, Any]] = []
    for (node, descriptions, evidence_used, request), result in zip(
        requests,
        results,
        strict=True,
    ):
        descriptions_used = request.descriptions
        description = result.description.strip()[:MAX_DESCRIPTION_LENGTH]
        if not description:
            trace_events.append(
                _trace_event(
                    node,
                    "skipped_empty_description",
                    descriptions,
                    descriptions_used,
                    evidence_used,
                    result.provider_metadata,
                    description_length=0,
                )
            )
            continue
        node.description = description
        for source in sources_by_node_id.get(node.id, []):
            source.per_file_description = description
        merged_count += 1
        trace_events.append(
            _trace_event(
                node,
                "merged",
                descriptions,
                descriptions_used,
                evidence_used,
                result.provider_metadata,
                description_length=len(description),
            )
        )
    return DescriptionSynthesisResult(merged_count=merged_count, trace_events=trace_events)


def _trace_event(
    node: GraphNode,
    action: str,
    descriptions: list[str],
    descriptions_used: list[str],
    evidence_used: list[str],
    provider_metadata: dict[str, Any],
    description_length: int,
) -> dict[str, Any]:
    return {
        "stage": "description_merge",
        "action": action,
        "node_id": node.id,
        "entity_name": node.name,
        "entity_type": node.primary_type,
        "observations_available": len(descriptions),
        "descriptions_used": len(descriptions_used),
        "evidence_used": len(evidence_used),
        "description_length": description_length,
        "provider_metadata": provider_metadata,
    }


def _descriptions_by_node_key(
    observations: list[ExtractedEntity],
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    grouped: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for observation in observations:
        description = observation.description.strip()
        if not description:
            continue
        key = (normalize_entity_name(observation.name), observation.type)
        grouped[key].append((observation.source_chunk_id, description))
    return dict(grouped)


def _evidence_by_subject(evidence: list[Evidence]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in evidence:
        quote = item.quote.strip()
        if quote and quote not in grouped[item.subject_id]:
            grouped[item.subject_id].append(quote)
    return dict(grouped)


def _sources_by_node_id(entity_sources: list[EntitySource]) -> dict[str, list[EntitySource]]:
    grouped: dict[str, list[EntitySource]] = defaultdict(list)
    for source in entity_sources:
        grouped[source.node_id].append(source)
    return dict(grouped)
