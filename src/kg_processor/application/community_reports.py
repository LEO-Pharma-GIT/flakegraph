# SPDX-License-Identifier: Apache-2.0
"""Generate grounded community summaries from canonical nodes and edges."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import batched
from typing import Any

from kg_processor.application.enrichment_batching import (
    COMMUNITY_BATCH_SIZE,
    summarize_community_requests,
)
from kg_processor.application.ordered_map import ordered_map
from kg_processor.application.redaction import redact_sensitive_text
from kg_processor.application.window_errors import is_systemic_provider_error
from kg_processor.domain.graph import Community, CommunityFinding, Evidence, GraphEdge, GraphNode
from kg_processor.domain.ids import stable_id
from kg_processor.ports.llm import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    CommunitySummaryProvider,
    CommunitySummaryRequest,
    CommunitySummaryResult,
    StructuredCompletionProvider,
)

DEFAULT_MAX_RELATIONS_PER_COMMUNITY = 25
# Bounded quote budgets keep report prompts manageable while preserving grounded
# context. Ratings are computed from the quotes a report actually received, so
# every engine must apply the same budgets.
MAX_EVIDENCE_QUOTES_PER_RELATION = 2
MAX_EVIDENCE_QUOTES_PER_COMMUNITY = 50
# Follows the relation type of an edge every source only claims, in its
# relation line and evidence lines, so the report can say it is a claim.
CLAIMED_RELATION_MARK = " (claimed)"
CommunityProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class CommunityReportResult:
    """Communities, findings, and trace events generated from graph clusters."""

    communities: list[Community]
    findings: list[CommunityFinding]
    trace_events: list[dict[str, Any]]


@dataclass(frozen=True)
class CommunitySeed:
    """Describe a detected member set and its place in the report hierarchy.

    Stable parent keys are resolved to persisted community IDs only after all
    reports exist, which keeps detection independent from generated narrative.
    """

    member_ids: set[str]
    level: int = 0
    parent_stable_key: str | None = None

    @property
    def stable_key(self) -> str:
        """Derive a reproducible hierarchy key from sorted member IDs and depth."""

        return community_stable_key(self.member_ids, self.level)


def community_stable_key(member_ids: set[str], level: int) -> str:
    """Key a community by its members and depth, the same way on every engine."""

    return stable_id("community_key", level, ",".join(sorted(member_ids)), length=40)


@dataclass(frozen=True)
class _CommunityReportContext:
    """Hold deterministic graph context prepared for one LLM summary request.

    The context retains canonical rows and hierarchy identity alongside the
    bounded text request so generated output can be traced back to graph inputs.
    """

    member_ids: set[str]
    internal_edges: list[GraphEdge]
    internal_edge_count: int
    endpoint_pair_count: int
    mean_edge_confidence: float
    request: CommunitySummaryRequest
    stable_key: str
    level: int
    parent_stable_key: str | None


def generate_community_reports(
    graph_id: str,
    community_seeds: Sequence[CommunitySeed],
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    llm: CommunitySummaryProvider,
    max_relations_per_community: int = DEFAULT_MAX_RELATIONS_PER_COMMUNITY,
    report_parallelism: int = 1,
    report_progress: CommunityProgressCallback | None = None,
    evidence: list[Evidence] | None = None,
    model: str = "default",
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
    seed: int | None = None,
) -> CommunityReportResult:
    """Generate stable community artifacts and LLM-authored narrative findings.

    Community identity and structural ratings are computed from canonical graph
    data rather than model prose. LLM output contributes titles, summaries,
    findings, and questions while provider metadata is retained in trace events.
    """

    # Community ids are based on sorted member ids rather than LLM output. That
    # makes report regeneration safe even when summaries or ratings change.
    node_by_id = {node.id: node for node in nodes}
    contexts = _report_contexts(
        community_seeds,
        node_by_id,
        edges,
        max_relations_per_community,
        evidence or [],
    )
    if report_progress is not None:
        report_progress(0, len(contexts))
    summaries = _summarize_communities(
        contexts,
        llm,
        report_parallelism,
        report_progress,
        model,
        timeout_seconds,
        seed,
    )
    communities: list[Community] = []
    findings: list[CommunityFinding] = []
    trace_events: list[dict[str, Any]] = []
    community_by_stable_key: dict[str, Community] = {}
    for context, summary in summaries:
        rating, rating_explanation = structural_rating(
            len(context.member_ids),
            context.endpoint_pair_count,
            len(context.request.relations),
            len(context.request.evidence_quotes),
            context.mean_edge_confidence,
        )
        community = Community(
            id=stable_id("community", graph_id, context.stable_key),
            graph_id=graph_id,
            stable_key=context.stable_key,
            level=context.level,
            title=summary.title,
            summary=summary.summary,
            rating=rating,
            rating_explanation=rating_explanation,
            member_node_ids=sorted(context.member_ids),
            suggested_questions=summary.suggested_questions,
        )
        communities.append(community)
        community_by_stable_key[context.stable_key] = community
        for finding_index, (finding_summary, explanation) in enumerate(summary.findings):
            findings.append(
                CommunityFinding(
                    id=stable_id("finding", community.id, finding_index, finding_summary),
                    graph_id=graph_id,
                    community_id=community.id,
                    summary=finding_summary,
                    explanation=explanation,
                )
            )
        trace_events.append(
            {
                "stage": "community_report",
                "community_id": community.id,
                "stable_key": context.stable_key,
                "level": community.level,
                "members": len(context.member_ids),
                "relations_used": len(context.internal_edges),
                "total_internal_relations": context.internal_edge_count,
                "evidence_quotes_used": len(context.request.evidence_quotes),
                "findings": len(summary.findings),
                "suggested_questions": len(summary.suggested_questions),
                "provider_metadata": summary.provider_metadata,
            }
        )
    for context in contexts:
        if context.parent_stable_key is None:
            continue
        child = community_by_stable_key.get(context.stable_key)
        parent = community_by_stable_key.get(context.parent_stable_key)
        if child is None or parent is None:
            continue
        child.parent_community_id = parent.id
        parent.child_community_ids.append(child.id)
    for community in communities:
        community.child_community_ids.sort()
    return CommunityReportResult(
        communities=communities,
        findings=findings,
        trace_events=trace_events,
    )


def _report_contexts(
    community_seeds: Sequence[CommunitySeed],
    node_by_id: dict[str, GraphNode],
    edges: list[GraphEdge],
    max_relations_per_community: int,
    evidence: list[Evidence],
) -> list[_CommunityReportContext]:
    """Prepare bounded, deterministic requests for every valid community seed.

    Members are degree-ordered, relations are strength-ordered, and exact evidence
    quotes are attached so summaries can be generated without receiving the full
    graph or unbounded source text.
    """

    contexts: list[_CommunityReportContext] = []
    evidence_by_subject: dict[str, list[Evidence]] = {}
    for evidence_item in evidence:
        evidence_by_subject.setdefault(evidence_item.subject_id, []).append(evidence_item)
    for index, seed in enumerate(community_seeds):
        member_ids = seed.member_ids
        members = _ordered_members(member_ids, node_by_id)
        if not members:
            continue
        all_internal_edges = _internal_edges(member_ids, edges)
        internal_edges = all_internal_edges[:max_relations_per_community]
        request = CommunitySummaryRequest(
            title_seed=_title_seed(members, index),
            members=[_member_line(node) for node in members],
            relations=[_relation_line(edge, node_by_id) for edge in internal_edges],
            evidence_quotes=_evidence_lines(internal_edges, evidence_by_subject),
        )
        stable_key = seed.stable_key
        contexts.append(
            _CommunityReportContext(
                member_ids=member_ids,
                internal_edges=internal_edges,
                internal_edge_count=len(all_internal_edges),
                endpoint_pair_count=len(
                    {
                        tuple(sorted((edge.source_node_id, edge.target_node_id)))
                        for edge in all_internal_edges
                    }
                ),
                mean_edge_confidence=(
                    sum(edge.confidence for edge in all_internal_edges) / len(all_internal_edges)
                    if all_internal_edges
                    else 0.0
                ),
                request=request,
                stable_key=stable_key,
                level=seed.level,
                parent_stable_key=seed.parent_stable_key,
            )
        )
    return contexts


def _summarize_communities(
    contexts: list[_CommunityReportContext],
    llm: CommunitySummaryProvider,
    report_parallelism: int,
    report_progress: CommunityProgressCallback | None,
    model: str,
    timeout_seconds: int,
    seed: int | None,
) -> list[tuple[_CommunityReportContext, CommunitySummaryResult]]:
    """Summarize bounded context batches while returning stable record order.

    Completion callbacks report completed communities rather than provider calls.
    Indexed collection keeps persisted ordering independent from batch latency.
    """

    batch_size = COMMUNITY_BATCH_SIZE if isinstance(llm, StructuredCompletionProvider) else 1
    context_batches = [list(items) for items in batched(contexts, batch_size)]
    completed_reports = 0

    def report(_completed_batches: int, results: list[CommunitySummaryResult]) -> None:
        nonlocal completed_reports
        completed_reports += len(results)
        if report_progress is not None:
            report_progress(completed_reports, len(contexts))

    summaries = ordered_map(
        context_batches,
        lambda _index, batch: _summarize_batch(llm, batch, model, timeout_seconds, seed),
        parallelism=report_parallelism,
        on_complete=report,
    )
    return [
        (context, result)
        for batch, results in zip(context_batches, summaries, strict=True)
        for context, result in zip(batch, results, strict=True)
    ]


def _summarize_batch(
    llm: CommunitySummaryProvider,
    batch: list[_CommunityReportContext],
    model: str,
    timeout_seconds: int,
    seed: int | None,
) -> list[CommunitySummaryResult]:
    """Degrade one failed report batch without hiding auth/configuration errors."""

    try:
        return summarize_community_requests(
            llm,
            [context.request for context in batch],
            model=model,
            timeout_seconds=timeout_seconds,
            seed=seed,
        )
    except Exception as exc:
        if is_systemic_provider_error(exc):
            raise
        error = redact_sensitive_text(str(exc))[:500]
        return [
            CommunitySummaryResult(
                title=context.request.title_seed,
                summary=(
                    "Structural community containing "
                    f"{len(context.member_ids)} entities and "
                    f"{context.internal_edge_count} relations."
                ),
                rating=0.0,
                rating_explanation="Narrative provider unavailable; structural rating retained.",
                findings=[],
                suggested_questions=[],
                provider_metadata={
                    "degraded": True,
                    "error_type": type(exc).__name__,
                    "error": error,
                },
            )
            for context in batch
        ]


def _ordered_members(member_ids: set[str], node_by_id: dict[str, GraphNode]) -> list[GraphNode]:
    members = [node_by_id[node_id] for node_id in member_ids if node_id in node_by_id]
    return sorted(members, key=lambda node: (-node.degree, node.name.lower(), node.id))


def _title_seed(members: list[GraphNode], index: int) -> str:
    if members:
        return members[0].name
    return f"Community {index + 1}"


def _member_line(node: GraphNode) -> str:
    return f"{node.name} [{node.primary_type}] degree={node.degree}: {node.description[:500]}"


def _internal_edges(
    member_ids: set[str],
    edges: list[GraphEdge],
) -> list[GraphEdge]:
    internal = [
        edge
        for edge in edges
        if edge.source_node_id in member_ids and edge.target_node_id in member_ids
    ]
    return sorted(
        internal,
        key=lambda edge: (-edge.weight, edge.relation_type, edge.source_node_id, edge.id),
    )


def relation_label(edge: GraphEdge) -> str:
    """The relation type a report context names, marked when its sources only claim it."""

    return edge.relation_type + (CLAIMED_RELATION_MARK if edge.assertion == "claimed" else "")


def _relation_line(edge: GraphEdge, node_by_id: dict[str, GraphNode]) -> str:
    source = node_by_id.get(edge.source_node_id)
    target = node_by_id.get(edge.target_node_id)
    source_name = source.name if source else edge.source_node_id
    target_name = target.name if target else edge.target_node_id
    return (
        f"{source_name} --{relation_label(edge)}--> {target_name}"
        f" weight={edge.weight}: {edge.description[:500]}"
    )


def _evidence_lines(
    edges: list[GraphEdge],
    evidence_by_subject: dict[str, list[Evidence]],
) -> list[str]:
    """Return bounded exact quotes supporting the strongest internal assertions."""

    lines: list[str] = []
    for edge in edges:
        for evidence in sorted(
            evidence_by_subject.get(edge.id, []),
            key=lambda item: (item.file_id, item.page_number, item.start_offset),
        )[:MAX_EVIDENCE_QUOTES_PER_RELATION]:
            lines.append(f"{relation_label(edge)}: {evidence.quote[:700]}")
    return lines[:MAX_EVIDENCE_QUOTES_PER_COMMUNITY]


def structural_rating(
    member_count: int,
    endpoint_pair_count: int,
    reported_relation_count: int,
    evidence_quote_count: int,
    mean_edge_confidence: float,
) -> tuple[float, str]:
    """Compute a deterministic rating from density, support, and confidence.

    Ratings intentionally exclude model self-assessment, making them comparable
    across providers and reproducible when narrative summaries are regenerated.
    Every engine that publishes communities calls this function so the same graph
    produces the same rating regardless of where it was assembled.

    Density counts distinct member endpoint pairs against the pairs a community of
    this size could contain, and is clamped because parallel predicates and loops
    can supply more asserted pairs than that ceiling while a rating must remain
    within its documented 0-10 range. Evidence coverage is measured against the
    relations actually placed in the report context, the only relations that can
    contribute quotes, so it scores evidence quality rather than community size.
    """

    possible_edges = max(member_count * (member_count - 1) / 2, 1.0)
    density = min(endpoint_pair_count / possible_edges, 1.0)
    evidence_coverage = min(evidence_quote_count / max(reported_relation_count, 1), 1.0)
    confidence = mean_edge_confidence
    rating = round(10.0 * (0.4 * density + 0.35 * evidence_coverage + 0.25 * confidence), 2)
    explanation = (
        f"Structural score from density {density:.2f}, evidence coverage "
        f"{evidence_coverage:.2f}, and mean edge confidence {confidence:.2f}."
    )
    return rating, explanation
