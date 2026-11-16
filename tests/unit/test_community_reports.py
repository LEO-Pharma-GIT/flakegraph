from __future__ import annotations

import threading
import time

from llm_fakes import RecordingCommunityLlm

from kg_processor.application.community_reports import CommunitySeed, generate_community_reports
from kg_processor.domain.graph import Evidence, GraphEdge, GraphNode
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    CommunitySummaryResult,
)


def test_generate_community_reports_uses_grounded_member_and_relation_context() -> None:
    """Verify report prompts use strongest internal relations and grounded member context.

    Weaker edges must not displace bounded evidence.
    """

    nodes = [
        _node("node_alice", "Alice Smith", "PERSON", degree=2),
        _node("node_acme", "Acme Corp", "ORGANIZATION", degree=1),
        _node("node_copenhagen", "Copenhagen", "LOCATION", degree=1),
    ]
    edges = [
        _edge(
            "edge_low",
            "node_acme",
            "node_copenhagen",
            "located_in",
            "Acme Corp is located in Copenhagen.",
            weight=2.0,
        ),
        _edge(
            "edge_high",
            "node_alice",
            "node_acme",
            "works_at",
            "Alice Smith works at Acme Corp.",
            weight=9.0,
        ),
    ]
    provider = RecordingCommunityLlm()

    result = generate_community_reports(
        "graph",
        [CommunitySeed(member_ids={"node_alice", "node_acme", "node_copenhagen"})],
        nodes,
        edges,
        provider,
        max_relations_per_community=1,
    )
    communities = result.communities
    findings = result.findings

    assert len(communities) == 1
    assert len(findings) == 1
    assert communities[0].title == "Alice Smith network"
    assert communities[0].rating == 5.17
    assert communities[0].rating_explanation == (
        "Structural score from density 0.67, evidence coverage 0.00, and mean edge confidence 1.00."
    )
    assert communities[0].suggested_questions == ["Where does Alice Smith work?"]
    assert communities[0].member_node_ids == ["node_acme", "node_alice", "node_copenhagen"]
    assert provider.requests[0].title_seed == "Alice Smith"
    assert provider.requests[0].members[0].startswith("Alice Smith [PERSON] degree=2")
    assert provider.requests[0].relations == [
        "Alice Smith --works_at--> Acme Corp weight=9.0: Alice Smith works at Acme Corp."
    ]
    assert result.trace_events == [
        {
            "stage": "community_report",
            "community_id": communities[0].id,
            "stable_key": communities[0].stable_key,
            "level": 0,
            "members": 3,
            "relations_used": 1,
            "total_internal_relations": 2,
            "evidence_quotes_used": 0,
            "findings": 1,
            "suggested_questions": 1,
            "provider_metadata": {
                "provider": "recording_llm",
                "prompt_name": "community_report",
            },
        }
    ]


def test_a_claimed_relation_is_marked_in_the_report_context_and_a_reported_one_is_not() -> None:
    """The report can tell a relation its sources only claim from one they report."""

    nodes = [
        _node("node_glaze", "Glaze-7", "PRODUCT", degree=2),
        _node("node_hull", "Hull steel", "MATERIAL", degree=1),
        _node("node_deck", "Deck plates", "MATERIAL", degree=1),
    ]
    reported = _edge(
        "edge_reported", "node_glaze", "node_hull", "protects", "Glaze-7 protected hull steel.", 9.0
    )
    claimed = _edge(
        "edge_claimed", "node_glaze", "node_deck", "protects", "Glaze-7 may protect decks.", 8.0
    ).model_copy(update={"assertion": "claimed"})
    evidence = [
        Evidence(
            id=f"evidence_{edge.id}",
            graph_id="graph",
            subject_id=edge.id,
            subject_kind="edge",
            file_id="file_1",
            chunk_id="chunk_1",
            page_number=1,
            start_offset=0,
            end_offset=len(edge.description),
            quote=edge.description,
        )
        for edge in (reported, claimed)
    ]
    provider = RecordingCommunityLlm()

    generate_community_reports(
        "graph",
        [CommunitySeed(member_ids={"node_glaze", "node_hull", "node_deck"})],
        nodes,
        [reported, claimed],
        provider,
        evidence=evidence,
    )

    assert provider.requests[0].relations == [
        "Glaze-7 --protects--> Hull steel weight=9.0: Glaze-7 protected hull steel.",
        "Glaze-7 --protects (claimed)--> Deck plates weight=8.0: Glaze-7 may protect decks.",
    ]
    assert provider.requests[0].evidence_quotes == [
        "protects: Glaze-7 protected hull steel.",
        "protects (claimed): Glaze-7 may protect decks.",
    ]


def test_generate_community_reports_parallel_preserves_input_order() -> None:
    nodes = [
        _node("node_alice", "Alice Smith", "PERSON", degree=2),
        _node("node_bob", "Bob Jones", "PERSON", degree=1),
        _node("node_copenhagen", "Copenhagen", "LOCATION", degree=2),
        _node("node_dojo", "Dojo Archive", "ORGANIZATION", degree=1),
    ]
    edges = [
        _edge(
            "edge_alice_bob",
            "node_alice",
            "node_bob",
            "trains_with",
            "Alice Smith trains with Bob Jones.",
            weight=7.0,
        ),
        _edge(
            "edge_archive",
            "node_copenhagen",
            "node_dojo",
            "hosts",
            "Copenhagen hosts the Dojo Archive.",
            weight=6.0,
        ),
    ]
    provider = _DelayedCommunityLlm({"Alice Smith": 0.04, "Copenhagen": 0.0})
    progress: list[tuple[int, int]] = []

    result = generate_community_reports(
        "graph",
        [
            CommunitySeed(member_ids={"node_alice", "node_bob"}),
            CommunitySeed(member_ids={"node_copenhagen", "node_dojo"}),
        ],
        nodes,
        edges,
        provider,
        report_parallelism=2,
        report_progress=lambda completed, total: progress.append((completed, total)),
    )

    assert provider.max_active_calls > 1
    assert [community.title for community in result.communities] == [
        "Alice Smith network",
        "Copenhagen network",
    ]
    assert [trace["members"] for trace in result.trace_events] == [2, 2]
    assert progress == [(0, 2), (1, 2), (2, 2)]


class _DelayedCommunityLlm(RecordingCommunityLlm):
    def __init__(self, delay_by_title_seed: dict[str, float]) -> None:
        super().__init__()
        self.delay_by_title_seed = delay_by_title_seed
        self._lock = threading.Lock()
        self._active_calls = 0
        self.max_active_calls = 0

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        with self._lock:
            self._active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self._active_calls)
        try:
            time.sleep(self.delay_by_title_seed.get(request.title_seed, 0.0))
            return super().summarize_community(request)
        finally:
            with self._lock:
                self._active_calls -= 1


def _node(node_id: str, name: str, primary_type: str, degree: int) -> GraphNode:
    return GraphNode(
        id=node_id,
        graph_id="graph",
        normalized_name=name.lower().replace(" ", ""),
        name=name,
        primary_type=primary_type,
        types=[primary_type],
        description=f"{name} appears in the source graph.",
        source_chunk_ids=["chunk_1"],
        degree=degree,
        rank=float(degree),
    )


def _edge(
    edge_id: str,
    source_node_id: str,
    target_node_id: str,
    relation_type: str,
    description: str,
    weight: float,
) -> GraphEdge:
    return GraphEdge(
        id=edge_id,
        graph_id="graph",
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        relation_type=relation_type,
        description=description,
        weight=weight,
        source_file_id="file_1",
        source_chunk_ids=["chunk_1"],
    )
