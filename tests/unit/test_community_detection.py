from __future__ import annotations

from importlib.util import find_spec

import networkx as nx
import pytest
from pydantic import ValidationError

from kg_processor.application.community_detection import _add_weighted_edge, _community_seeds
from kg_processor.config.settings import GraphSettings, Settings


def test_community_detection_omits_reports_when_total_weight_is_zero() -> None:
    """Ensure a graph with no positive support cannot produce misleading communities.

    Mere node adjacency with zero weight is insufficient.
    """

    graph: nx.Graph[str] = nx.Graph()
    graph.add_nodes_from(["node_a", "node_b"])
    graph.add_edge("node_a", "node_b", weight=0.0)

    assert (
        _community_seeds(
            graph,
            min_community_size=2,
            max_community_size=50,
            resolution=1.0,
            deterministic_seed=17,
        )
        == []
    )


def test_community_detection_accumulates_parallel_edge_weights() -> None:
    graph: nx.Graph[str] = nx.Graph()

    _add_weighted_edge(graph, "node_a", "node_b", 2.0)
    _add_weighted_edge(graph, "node_a", "node_b", 3.5)

    assert graph["node_a"]["node_b"]["weight"] == 5.5


def test_seeded_louvain_partition_is_stable_and_omits_isolated_nodes() -> None:
    """Ensure the default seeded partition is repeatable and omits isolated nodes.

    Louvain is the default so the package carries no GPL dependency. Its
    contract on this fixture: two weakly linked triangles come back as two
    communities, and a node with no support is never reported.
    """

    graph: nx.Graph[str] = nx.Graph()
    graph.add_weighted_edges_from(
        [
            ("a", "b", 5.0),
            ("b", "c", 5.0),
            ("a", "c", 5.0),
            ("d", "e", 5.0),
            ("e", "f", 5.0),
            ("d", "f", 5.0),
            ("c", "d", 0.01),
        ]
    )
    graph.add_node("isolated")

    first = _community_seeds(graph, 2, 50, 1.0, 17)
    second = _community_seeds(graph, 2, 50, 1.0, 17)

    assert first == second
    assert {frozenset(seed.member_ids) for seed in first} == {
        frozenset({"a", "b", "c"}),
        frozenset({"d", "e", "f"}),
    }
    assert all("isolated" not in seed.member_ids for seed in first)


def test_louvain_partition_ignores_node_insertion_order() -> None:
    """The same graph built in a different order must yield the same communities.

    NetworkX's Louvain shuffles from its own node order, so without the sorted
    rebuild two workers that discovered nodes in different orders would publish
    different community ids for identical graphs.
    """

    edges = [
        ("a", "b", 5.0),
        ("b", "c", 5.0),
        ("a", "c", 5.0),
        ("d", "e", 5.0),
        ("e", "f", 5.0),
        ("d", "f", 5.0),
        ("c", "d", 0.01),
    ]
    forward: nx.Graph[str] = nx.Graph()
    forward.add_weighted_edges_from(edges)
    backward: nx.Graph[str] = nx.Graph()
    backward.add_weighted_edges_from(reversed(edges))

    assert _community_seeds(forward, 2, 50, 1.0, 17) == _community_seeds(backward, 2, 50, 1.0, 17)


def test_leiden_is_an_opt_in_extra_with_an_install_hint() -> None:
    """Selecting Leiden without its GPL packages must fail up front and say how to fix it."""

    if find_spec("leidenalg") is not None and find_spec("igraph") is not None:
        pytest.skip("the leiden extra is installed in this environment")
    graph: nx.Graph[str] = nx.Graph()
    graph.add_edge("a", "b", weight=1.0)

    with pytest.raises(ImportError, match=r"flakegraph\[leiden\]"):
        _community_seeds(graph, 2, 50, 1.0, 17, algorithm="leiden")


def test_community_algorithm_defaults_to_louvain_and_rejects_unknown_names() -> None:
    settings = Settings.load(env={})

    assert settings.graph.community_algorithm == "louvain"
    with pytest.raises(ValidationError):
        GraphSettings.model_validate({"community_algorithm": "label_propagation"})
