# SPDX-License-Identifier: Apache-2.0
"""Deterministic community detection over canonical weighted graph rows."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Literal

import networkx as nx
from networkx.algorithms.community import louvain_communities

from kg_processor.application.community_reports import (
    CommunityProgressCallback,
    CommunityReportResult,
    CommunitySeed,
    community_stable_key,
    generate_community_reports,
)
from kg_processor.application.folder_entities import FOLDER_EVIDENCE_METHOD
from kg_processor.domain.graph import Evidence, GraphEdge, GraphNode
from kg_processor.ports.llm import DEFAULT_LLM_TIMEOUT_SECONDS, LlmProvider

_MAX_HIERARCHY_LEVEL = 2
_MAX_CO_MENTION_NODES = 30
_CHILD_RESOLUTION_MULTIPLIER = 1.5

CommunityAlgorithm = Literal["louvain", "leiden"]
DEFAULT_COMMUNITY_ALGORITHM: CommunityAlgorithm = "louvain"
# One partition call: the ordered node ids, the ordered (source, target, weight)
# edges, the resolution and the seed, returning sets of node indexes.
_Partitioner = Callable[
    [list[str], list[tuple[str, str, float]], float, int],
    list[set[int]],
]


def detect_communities(
    graph_id: str,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    llm: LlmProvider,
    min_community_size: int,
    report_parallelism: int = 1,
    report_progress: CommunityProgressCallback | None = None,
    evidence: list[Evidence] | None = None,
    deterministic_seed: int = 17,
    max_community_size: int = 50,
    resolution: float = 1.0,
    co_mention_weight: float = 0.05,
    model: str = "default",
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
    algorithm: CommunityAlgorithm = DEFAULT_COMMUNITY_ALGORITHM,
    structural_relation_types: frozenset[str] = frozenset(),
) -> CommunityReportResult:
    """Detect deterministic weighted communities and generate grounded reports.

    Canonical relations provide the primary graph, weak chunk co-mentions help
    stabilize sparse regions, and seeded modularity partitions are converted
    into provider-neutral report requests. Louvain is the default because it
    ships with NetworkX under a permissive licence; Leiden is an opt-in extra
    because its implementation is GPL.

    Edges of a structural type, and a folder entity's folder-line evidence,
    say how the sources are filed rather than what they state, and take no
    part in the partition: a folder linked to everything filed in it would
    otherwise make communities of folders instead of topics.
    """

    if not nodes:
        return CommunityReportResult(communities=[], findings=[], trace_events=[])
    graph: nx.Graph[str] = nx.Graph()
    for node in nodes:
        graph.add_node(node.id)
    for edge in edges:
        if (
            edge.source_node_id != edge.target_node_id
            and edge.relation_type not in structural_relation_types
        ):
            _add_weighted_edge(graph, edge.source_node_id, edge.target_node_id, edge.weight)
    _add_co_mention_projection(
        graph,
        nodes,
        co_mention_weight,
        folder_evidence={
            (item.subject_id, item.chunk_id)
            for item in evidence or []
            if item.subject_kind == "node" and item.method == FOLDER_EVIDENCE_METHOD
        },
    )

    raw_communities = _community_seeds(
        graph,
        min_community_size,
        max_community_size,
        resolution,
        deterministic_seed,
        algorithm=algorithm,
    )
    return generate_community_reports(
        graph_id,
        raw_communities,
        nodes,
        edges,
        llm,
        report_parallelism=report_parallelism,
        report_progress=report_progress,
        evidence=evidence,
        model=model,
        timeout_seconds=timeout_seconds,
        seed=deterministic_seed,
    )


def _community_seeds(
    graph: nx.Graph[str],
    min_community_size: int,
    max_community_size: int,
    resolution: float,
    deterministic_seed: int,
    algorithm: CommunityAlgorithm = DEFAULT_COMMUNITY_ALGORITHM,
) -> list[CommunitySeed]:
    """Build top-level and bounded child community seeds from a weighted graph.

    Edgeless or non-positive graphs produce no communities. Oversized partitions
    are recursively subdivided so report prompts remain useful and bounded.
    """

    if graph.number_of_edges() == 0:
        return []
    total_weight = sum(float(data.get("weight", 1.0)) for _, _, data in graph.edges(data=True))
    if total_weight <= 0:
        return []
    partitioner = _partitioner(algorithm)
    top_level = [
        members
        for members in _partition(graph, resolution, deterministic_seed, partitioner)
        if len(members) >= min_community_size
    ]
    seeds: list[CommunitySeed] = []
    for members in top_level:
        seeds.append(CommunitySeed(member_ids=members))
        _append_child_seeds(
            seeds,
            graph,
            members,
            level=1,
            parent_stable_key=community_stable_key(members, 0),
            min_community_size=min_community_size,
            max_community_size=max_community_size,
            resolution=resolution * _CHILD_RESOLUTION_MULTIPLIER,
            deterministic_seed=deterministic_seed,
            partitioner=partitioner,
        )
    return seeds


def _append_child_seeds(
    seeds: list[CommunitySeed],
    graph: nx.Graph[str],
    members: set[str],
    *,
    level: int,
    parent_stable_key: str,
    min_community_size: int,
    max_community_size: int,
    resolution: float,
    deterministic_seed: int,
    partitioner: _Partitioner,
) -> None:
    """Recursively split oversized communities into reportable child groups.

    Recursion is bounded by hierarchy depth and stops when the partition cannot
    produce more than one proper child, preventing duplicate or unhelpful
    report levels.
    """

    if len(members) <= max_community_size or level > _MAX_HIERARCHY_LEVEL:
        return
    induced = graph.subgraph(members).copy()
    children = [
        child
        for child in _partition(induced, resolution, deterministic_seed + level, partitioner)
        if min_community_size <= len(child) < len(members)
    ]
    if len(children) <= 1:
        return
    for child in children:
        seeds.append(
            CommunitySeed(
                member_ids=child,
                level=level,
                parent_stable_key=parent_stable_key,
            )
        )
        _append_child_seeds(
            seeds,
            induced,
            child,
            level=level + 1,
            parent_stable_key=community_stable_key(child, level),
            min_community_size=min_community_size,
            max_community_size=max_community_size,
            resolution=resolution * _CHILD_RESOLUTION_MULTIPLIER,
            deterministic_seed=deterministic_seed,
            partitioner=partitioner,
        )


def _partition(
    graph: nx.Graph[str],
    resolution: float,
    deterministic_seed: int,
    partitioner: _Partitioner,
) -> list[set[str]]:
    """Run a seeded modularity partition after imposing stable vertex and edge order.

    Both algorithms visit vertices in an order drawn from their seed, so the
    input order has to be fixed before the seed can make the result repeatable.
    Sorting removes that iteration-order variance, and the final communities are
    sorted again to make artifact IDs reproducible.
    """

    node_ids = sorted(graph.nodes)
    ordered_edges = sorted(
        (
            min(source, target),
            max(source, target),
            float(data.get("weight", 1.0)),
        )
        for source, target, data in graph.edges(data=True)
        if source != target
    )
    partition = partitioner(node_ids, ordered_edges, resolution, deterministic_seed)
    communities = [{node_ids[index] for index in community} for community in partition]
    return sorted(communities, key=lambda members: (min(members), len(members)))


def _partitioner(algorithm: CommunityAlgorithm) -> _Partitioner:
    """Resolve the configured partition algorithm, importing Leiden only on request.

    ``igraph`` and ``leidenalg`` are GPL-licensed, so they are an optional extra
    rather than a dependency of the Apache-2.0 package; asking for Leiden without
    them installed fails here with the install hint rather than deep inside a run.
    """

    if algorithm == "louvain":
        return _louvain_partition
    if algorithm == "leiden":
        try:
            import igraph  # noqa: PLC0415
            import leidenalg  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "graph.community_algorithm 'leiden' requires the GPL-licensed igraph and "
                "leidenalg packages; install them with the `leiden` extra "
                "(pip install 'flakegraph[leiden]') or use the default 'louvain'."
            ) from exc

        def _leiden_partition(
            node_ids: list[str],
            ordered_edges: list[tuple[str, str, float]],
            resolution: float,
            deterministic_seed: int,
        ) -> list[set[int]]:
            node_index = {node_id: index for index, node_id in enumerate(node_ids)}
            leiden_graph = igraph.Graph(
                n=len(node_ids),
                edges=[
                    (node_index[source], node_index[target]) for source, target, _ in ordered_edges
                ],
                directed=False,
            )
            partition = leidenalg.find_partition(
                leiden_graph,
                leidenalg.RBConfigurationVertexPartition,
                weights=[weight for _source, _target, weight in ordered_edges],
                resolution_parameter=resolution,
                seed=deterministic_seed,
            )
            return [set(community) for community in partition]

        return _leiden_partition
    raise ValueError(f"Unsupported community algorithm: {algorithm}")


def _louvain_partition(
    node_ids: list[str],
    ordered_edges: list[tuple[str, str, float]],
    resolution: float,
    deterministic_seed: int,
) -> list[set[int]]:
    """Run NetworkX's seeded Louvain over an integer-indexed copy of the graph.

    The copy is rebuilt from the sorted ids so NetworkX's own node order, which
    the seeded shuffle starts from, is the same on every run.
    """

    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    ordered_graph: nx.Graph[int] = nx.Graph()
    ordered_graph.add_nodes_from(range(len(node_ids)))
    ordered_graph.add_weighted_edges_from(
        (node_index[source], node_index[target], weight) for source, target, weight in ordered_edges
    )
    return [
        set(community)
        for community in louvain_communities(
            ordered_graph,
            weight="weight",
            resolution=resolution,
            seed=deterministic_seed,
        )
    ]


def _add_co_mention_projection(
    graph: nx.Graph[str],
    nodes: list[GraphNode],
    weight: float,
    folder_evidence: set[tuple[str, str]] | None = None,
) -> None:
    """Project bounded chunk co-mentions as weak community-only graph edges.

    These links never become knowledge-graph facts. They help sparse but related
    entities cluster while a node cap prevents broad chunks creating dense cliques.
    A node is not a member of a chunk whose evidence row for it is a folder
    line (``folder_evidence``, by node and chunk id), as the Spark finalizer
    leaves those rows out. A node's evidence in a chunk is a folder line only
    when the folder mention is the only mention of it there.
    """

    nodes_by_chunk: dict[str, list[str]] = {}
    for node in nodes:
        for chunk_id in node.source_chunk_ids:
            if (node.id, chunk_id) not in (folder_evidence or set()):
                nodes_by_chunk.setdefault(chunk_id, []).append(node.id)
    for node_ids in nodes_by_chunk.values():
        # Very broad chunks should not create a dense artificial clique.
        bounded_ids = sorted(set(node_ids))[:_MAX_CO_MENTION_NODES]
        for source, target in combinations(bounded_ids, 2):
            _add_weighted_edge(graph, source, target, weight)


def _add_weighted_edge(
    graph: nx.Graph[str],
    source_node_id: str,
    target_node_id: str,
    weight: float,
) -> None:
    """Add an undirected community edge, accumulating repeated graph facts."""

    if graph.has_edge(source_node_id, target_node_id):
        graph[source_node_id][target_node_id]["weight"] += weight
        return
    graph.add_edge(source_node_id, target_node_id, weight=weight)
