# SPDX-License-Identifier: Apache-2.0
"""Build the provider-neutral dataset used by the static graph explorer.

The explorer is a presentation of already-written local artifacts. This module
owns artifact validation, JSON-safe normalization, graph indexes, and layouts;
it intentionally knows nothing about HTML, Plotly, browsers, or CLI behavior.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import networkx as nx

from kg_processor.application.inspect import (
    _float_value,
    _int_value,
    inspect_local_graph,
    read_local_graph_artifacts,
)

EXPLORER_SCHEMA_VERSION = 1
_SPRING_LAYOUT_NODE_LIMIT = 1_200
_SMALL_GRAPH_NODE_LIMIT = 300
_OMITTED_FIELDS = {"embedding"}


@dataclass(frozen=True)
class GraphExplorerDataset:
    """Complete JSON-safe payload for one generated graph explorer."""

    payload: dict[str, Any]

    @property
    def graph_id(self) -> str:
        """Return the graph identity displayed by the explorer."""

        return str(self.payload["graph_id"])

    @property
    def counts(self) -> dict[str, int]:
        """Return principal artifact counts for export reporting."""

        raw_counts = self.payload["counts"]
        return {str(key): int(value) for key, value in raw_counts.items()}


def build_graph_explorer_dataset(output_path: Path) -> GraphExplorerDataset:
    """Load local artifacts and build an optimized interactive explorer payload.

    Embedding vectors are omitted because they dominate file size but are not
    directly reviewable. Source chunks, evidence spans, per-file descriptions,
    community narratives, findings, quality checks, and operational metrics are
    retained so every visible graph object remains traceable.
    """

    artifacts = read_local_graph_artifacts(output_path)
    _require_explorer_tables(output_path, artifacts)
    inspection = inspect_local_graph(output_path)
    run_report = _read_json_object(output_path / "run_report.json")
    graph_metrics = _read_json_object(output_path / "graph_metrics.json")

    nodes = _normalized_rows(artifacts["nodes"].rows)
    edges = _normalized_rows(artifacts["edges"].rows)
    edge_observations = _normalized_rows(artifacts["edge_observations"].rows)
    evidence = _normalized_rows(artifacts["evidence"].rows)
    entity_sources = _normalized_rows(artifacts["entity_sources"].rows)
    communities = _normalized_rows(artifacts["communities"].rows)
    findings = _normalized_rows(artifacts["community_findings"].rows)
    documents = _normalized_rows(artifacts["documents"].rows)
    chunks = _normalized_rows(artifacts["chunks"].rows)

    membership = _community_membership(communities)
    for node in nodes:
        node["community_ids"] = membership.get(str(node.get("id", "")), [])

    graph_id = _graph_id(run_report, nodes)
    counts = {
        "documents": len(documents),
        "chunks": len(chunks),
        "nodes": len(nodes),
        "edges": len(edges),
        "edge_observations": len(edge_observations),
        "evidence": len(evidence),
        "communities": len(communities),
        "findings": len(findings),
    }
    payload: dict[str, Any] = {
        "schema_version": EXPLORER_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "graph_id": graph_id,
        "artifact_directory": output_path.name,
        "counts": counts,
        "nodes": nodes,
        "edges": edges,
        "edge_observations": edge_observations,
        "evidence": evidence,
        "entity_sources": entity_sources,
        "communities": communities,
        "community_findings": findings,
        "documents": documents,
        "chunks": chunks,
        "layouts": _graph_layouts(nodes, edges, communities),
        "layout_metadata": {
            "default": "force" if len(nodes) <= _SPRING_LAYOUT_NODE_LIMIT else "community",
            "force_is_approximated": len(nodes) > _SPRING_LAYOUT_NODE_LIMIT,
        },
        "filters": {
            "node_types": _value_counts(nodes, "primary_type"),
            "relation_types": _value_counts(edges, "relation_type"),
        },
        "run_report": _json_value(run_report),
        "graph_metrics": _json_value(graph_metrics),
        "quality": _json_value(inspection["quality"]),
        "schema": _json_value(inspection["schema"]),
        "trace_events": int(inspection["trace_events"]),
    }
    return GraphExplorerDataset(payload=payload)


def _require_explorer_tables(output_path: Path, artifacts: dict[str, Any]) -> None:
    """Fail clearly when the selected directory is not a graph artifact snapshot."""

    missing = [name for name in ("nodes", "edges") if not artifacts[name].exists]
    if missing:
        names = ", ".join(f"{name}.parquet" for name in missing)
        raise ValueError(f"Graph explorer requires {names} in artifact directory: {output_path}")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _normalized_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            str(key): _json_value(value)
            for key, value in row.items()
            if str(key) not in _OMITTED_FIELDS
        }
        for row in rows
    ]


def _json_value(value: object) -> Any:  # noqa: PLR0911 - each supported value shape exits clearly.
    """Convert Parquet, NumPy, datetime, and container values into safe JSON data."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_value(item) for item in value]

    # NumPy scalars and arrays expose item()/tolist() without requiring this
    # application boundary to import NumPy-specific types.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar = item_method()
        except ValueError:
            scalar = value
        if scalar is not value:
            return _json_value(scalar)
    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        return _json_value(tolist_method())
    return str(value)


def _graph_id(run_report: dict[str, Any], nodes: list[dict[str, Any]]) -> str:
    report_graph_id = run_report.get("graph_id")
    if report_graph_id:
        return str(report_graph_id)
    if nodes and nodes[0].get("graph_id"):
        return str(nodes[0]["graph_id"])
    return "graph"


def _community_membership(communities: list[dict[str, Any]]) -> dict[str, list[str]]:
    membership: dict[str, list[str]] = defaultdict(list)
    ordered = sorted(
        communities,
        key=lambda item: (
            -_float_value(item.get("rating")),
            str(item.get("title", "")).lower(),
            str(item.get("id", "")),
        ),
    )
    for community in ordered:
        community_id = str(community.get("id", ""))
        for node_id in _string_list(community.get("member_node_ids")):
            membership[node_id].append(community_id)
    return dict(membership)


def _value_counts(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts = Counter(str(row.get(key, "Unknown")) or "Unknown" for row in rows)
    return [
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
    ]


def _graph_layouts(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    communities: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    graph: nx.Graph[str] = nx.Graph()
    node_ids = [str(node.get("id", "")) for node in nodes if node.get("id")]
    graph.add_nodes_from(node_ids)
    valid_node_ids = set(node_ids)
    for edge in edges:
        source = str(edge.get("source_node_id", ""))
        target = str(edge.get("target_node_id", ""))
        if source in valid_node_ids and target in valid_node_ids:
            graph.add_edge(source, target, weight=max(_float_value(edge.get("weight")), 0.01))

    community = _community_layout(nodes, communities)
    radial = _radial_layout(nodes)
    if len(node_ids) > _SPRING_LAYOUT_NODE_LIMIT:
        force = community
    elif graph.number_of_edges() == 0:
        force = radial
    else:
        iterations = 120 if len(node_ids) <= _SMALL_GRAPH_NODE_LIMIT else 70
        force = _normalize_positions(
            nx.spring_layout(
                graph,
                seed=37,
                weight="weight",
                iterations=iterations,
                scale=1.0,
            )
        )
    return {"force": force, "community": community, "radial": radial}


def _community_layout(
    nodes: list[dict[str, Any]],
    communities: list[dict[str, Any]],
) -> dict[str, list[float]]:
    membership = _community_membership(communities)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        node_id = str(node.get("id", ""))
        community_ids = membership.get(node_id, [])
        groups[community_ids[0] if community_ids else "__unassigned__"].append(node)
    if not groups:
        return {}

    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    positions: dict[str, tuple[float, float]] = {}
    group_count = len(ordered_groups)
    center_radius = 2.7 if group_count > 1 else 0.0
    for group_index, (_group_id, members) in enumerate(ordered_groups):
        center_angle = (2 * math.pi * group_index / group_count) - math.pi / 2
        center_x = center_radius * math.cos(center_angle)
        center_y = center_radius * math.sin(center_angle)
        ordered_members = sorted(
            members,
            key=lambda node: (
                -_int_value(node.get("degree")),
                str(node.get("name", "")).lower(),
                str(node.get("id", "")),
            ),
        )
        member_count = len(ordered_members)
        local_radius = min(1.15, 0.24 + 0.14 * math.sqrt(member_count))
        for member_index, node in enumerate(ordered_members):
            if member_count == 1:
                x, y = center_x, center_y
            else:
                angle = (2 * math.pi * member_index / member_count) + center_angle / 3
                radius = local_radius * (0.72 + 0.28 * ((member_index % 3) / 2))
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
            positions[str(node["id"])] = (x, y)
    return _normalize_positions(positions)


def _radial_layout(nodes: list[dict[str, Any]]) -> dict[str, list[float]]:
    if not nodes:
        return {}
    ordered = sorted(
        nodes,
        key=lambda node: (
            -_int_value(node.get("degree")),
            -_float_value(node.get("rank")),
            str(node.get("name", "")).lower(),
        ),
    )
    positions: dict[str, tuple[float, float]] = {}
    total = len(ordered)
    shells = (
        (ordered[: max(1, math.ceil(total * 0.1))], 0.22),
        (ordered[max(1, math.ceil(total * 0.1)) : math.ceil(total * 0.4)], 0.62),
        (ordered[math.ceil(total * 0.4) :], 1.0),
    )
    for shell_index, (members, radius) in enumerate(shells):
        for index, node in enumerate(members):
            angle = (2 * math.pi * index / max(1, len(members))) - math.pi / 2
            angle += shell_index * 0.24
            positions[str(node["id"])] = (
                radius * math.cos(angle),
                radius * math.sin(angle),
            )
    return _normalize_positions(positions)


def _normalize_positions(positions: dict[str, Any]) -> dict[str, list[float]]:
    if not positions:
        return {}
    converted = {
        node_id: (float(position[0]), float(position[1])) for node_id, position in positions.items()
    }
    extents = [abs(coordinate) for position in converted.values() for coordinate in position]
    max_extent = max([1e-9, *extents])
    return {
        node_id: [round(x / max_extent, 6), round(y / max_extent, 6)]
        for node_id, (x, y) in converted.items()
    }


def _string_list(value: object) -> list[str]:
    normalized = _json_value(value)
    if isinstance(normalized, list):
        return [str(item) for item in normalized]
    if normalized is None:
        return []
    return [str(normalized)]
