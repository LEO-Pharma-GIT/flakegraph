# SPDX-License-Identifier: Apache-2.0
"""Severity-aware graph quality gates evaluated before persistence."""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any, Literal

import networkx as nx
from pydantic import BaseModel, Field

from kg_processor.domain.graph import Chunk, Community, Evidence, GraphEdge, GraphNode
from kg_processor.domain.ontology import OntologyProfile, normalize_ontology_label

_DETAIL_LIMIT = 20
# Shape heuristics are warnings, not gates: legitimate corpora can exceed them.
_MAX_ISOLATE_RATIO = 0.15
_MAX_COMPONENT_RATIO = 0.25
_MAX_RELATION_VOCABULARY_RATIO = 0.50
_MIN_EDGES_FOR_VOCABULARY_WARNING = 5
_MIN_REPORTED_COMMUNITY_SIZE = 2


class GraphQualityCheck(BaseModel):
    """Represent one hard invariant or non-blocking graph diagnostic.

    Details are bounded by the evaluator so large failures remain serializable and
    readable while ``count`` preserves the full failure magnitude.
    """

    name: str
    ok: bool
    severity: Literal["error", "warning"] = "error"
    count: int = 0
    details: list[str] = Field(default_factory=list)


class GraphQualityResult(BaseModel):
    """Aggregate all quality checks and the write-blocking outcome.

    Warning failures remain visible in reports and progress output without making
    an otherwise valid graph impossible to inspect or persist.
    """

    ok: bool
    checks: list[GraphQualityCheck]

    def failed_errors(self) -> list[GraphQualityCheck]:
        """Return only failed quality checks whose error severity is configured to block writes."""

        return [check for check in self.checks if not check.ok and check.severity == "error"]

    def warnings(self) -> list[GraphQualityCheck]:
        """Return non-blocking warnings whose configured thresholds were exceeded.

        Passing warnings and hard errors are excluded.
        """

        return [check for check in self.checks if not check.ok and check.severity == "warning"]


class GraphQualityError(RuntimeError):
    """Signal that one or more configured hard graph invariants failed.

    The complete typed result is retained on the exception for CLI reporting and
    tests, while the message names failed checks for concise logs.
    """

    def __init__(self, result: GraphQualityResult) -> None:
        """Attach the full result and construct a concise operator-facing failure message.

        Failed check names remain visible in logs.
        """

        self.result = result
        failed_checks = [check.name for check in result.failed_errors()]
        message = "Graph quality checks failed"
        if failed_checks:
            message = f"{message}: {', '.join(failed_checks)}"
        super().__init__(message)


def evaluate_graph_quality(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    evidence: list[Evidence],
    expected_embedding_dimension: int | None,
    *,
    chunks: list[Chunk] | None = None,
    communities: list[Community] | None = None,
    ontology: OntologyProfile | None = None,
) -> GraphQualityResult:
    """Evaluate graph quality over typed domain models before persistence.

    Models are converted to the row-oriented evaluator used by both live pipeline
    runs and artifact inspection, keeping quality semantics consistent everywhere.
    """

    return evaluate_graph_quality_rows(
        [node.model_dump(mode="json") for node in nodes],
        [edge.model_dump(mode="json") for edge in edges],
        [item.model_dump(mode="json") for item in evidence],
        expected_embedding_dimension,
        chunks=None if chunks is None else [chunk.model_dump(mode="json") for chunk in chunks],
        communities=None
        if communities is None
        else [item.model_dump(mode="json") for item in communities],
        ontology=ontology,
    )


def evaluate_graph_quality_rows(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    expected_embedding_dimension: int | None,
    *,
    chunks: list[dict[str, Any]] | None = None,
    communities: list[dict[str, Any]] | None = None,
    ontology: OntologyProfile | None = None,
) -> GraphQualityResult:
    """Evaluate structural, grounding, ontology, and embedding diagnostics.

    Referential integrity and evidence coverage are hard errors. Shape heuristics
    such as isolates, fragmentation, vocabulary spread, and singleton communities
    are warnings because legitimate corpora can exceed their general thresholds.
    """

    checks = [
        _orphan_edge_check(nodes, edges),
        _duplicate_edge_check(edges),
        _self_loop_check(edges, ontology),
        _ontology_check(nodes, edges, ontology),
        _evidence_coverage_check("nodes_have_evidence", nodes, evidence),
        _evidence_coverage_check("edges_have_evidence", edges, evidence),
    ]
    if chunks is not None:
        checks.append(_evidence_span_check(chunks, evidence))
    # Structural edges reach whatever their sources hold; read with them, every
    # node of a folder's documents looks connected and the vocabulary small.
    structural = ontology.structural_relation_labels() if ontology is not None else frozenset()
    shape_edges = [
        edge
        for edge in edges
        if normalize_ontology_label(str(edge.get("relation_type", ""))) not in structural
    ]
    checks.extend(
        [
            _isolate_check(nodes, shape_edges, _MAX_ISOLATE_RATIO),
            _component_check(nodes, shape_edges, _MAX_COMPONENT_RATIO),
            _relation_vocabulary_check(shape_edges, _MAX_RELATION_VOCABULARY_RATIO),
            _singleton_community_check(communities or []),
        ]
    )
    if expected_embedding_dimension is not None:
        checks.extend(
            [
                _embedding_dimension_check(
                    "node_embedding_dimensions", nodes, expected_embedding_dimension
                ),
                _embedding_dimension_check(
                    "edge_embedding_dimensions", edges, expected_embedding_dimension
                ),
            ]
        )
    return GraphQualityResult(
        ok=not any(not check.ok and check.severity == "error" for check in checks),
        checks=checks,
    )


def _orphan_edge_check(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> GraphQualityCheck:
    """Find edges whose source or target does not exist in the canonical node set."""

    node_ids = {str(node.get("id")) for node in nodes}
    orphan_edges = [
        str(edge.get("id"))
        for edge in edges
        if str(edge.get("source_node_id")) not in node_ids
        or str(edge.get("target_node_id")) not in node_ids
    ]
    return _check("no_orphan_edges", orphan_edges)


def _duplicate_edge_check(edges: list[dict[str, Any]]) -> GraphQualityCheck:
    """Find duplicate canonical triples after consistent relation-label normalization.

    Source and target direction remains significant.
    """

    keys = [
        (
            str(edge.get("source_node_id")),
            str(edge.get("target_node_id")),
            normalize_ontology_label(str(edge.get("relation_type", ""))),
        )
        for edge in edges
    ]
    duplicates = ["|".join(key) for key, count in Counter(keys).items() if count > 1]
    return _check("no_duplicate_canonical_edges", duplicates)


def _self_loop_check(
    edges: list[dict[str, Any]], ontology: OntologyProfile | None
) -> GraphQualityCheck:
    """Reject canonical self-loops unless their ontology explicitly allows them.

    Unknown relation definitions cannot authorize a loop.
    """

    invalid: list[str] = []
    for edge in edges:
        if str(edge.get("source_node_id")) != str(edge.get("target_node_id")):
            continue
        definition = ontology.relation(str(edge.get("relation_type", ""))) if ontology else None
        if definition is None or not definition.allow_self_loop:
            invalid.append(str(edge.get("id")))
    return _check("no_unapproved_self_loops", invalid)


def _ontology_check(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    ontology: OntologyProfile | None,
) -> GraphQualityCheck:
    """Validate relation vocabulary and endpoint types against a closed ontology.

    Open or absent ontologies intentionally skip this gate, preserving generic
    extraction behavior while closed profiles enforce their declared signatures.
    """

    if ontology is None or ontology.mode == "open":
        return _check("ontology_constraints", [])
    node_type = {str(node.get("id")): str(node.get("primary_type")) for node in nodes}
    definitions_by_label = {
        normalize_ontology_label(label): definition
        for definition in ontology.relation_types
        for label in [definition.name, *definition.aliases]
    }
    invalid: list[str] = []
    for edge in edges:
        edge_id = str(edge.get("id"))
        definition = definitions_by_label.get(
            normalize_ontology_label(str(edge.get("relation_type", "")))
        )
        if definition is None:
            # Hybrid ontologies guide extraction with a preferred vocabulary but
            # deliberately retain other source-grounded predicates. Only a closed
            # ontology treats an unknown predicate as a publication error.
            if ontology.mode == "closed":
                invalid.append(f"{edge_id}:unknown_relation")
            continue
        source_type = node_type.get(str(edge.get("source_node_id")), "")
        target_type = node_type.get(str(edge.get("target_node_id")), "")
        if definition.source_types and source_type not in definition.source_types:
            invalid.append(f"{edge_id}:source_type={source_type}")
        if definition.target_types and target_type not in definition.target_types:
            invalid.append(f"{edge_id}:target_type={target_type}")
    return _check("ontology_constraints", invalid)


def _evidence_coverage_check(
    name: str,
    subjects: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> GraphQualityCheck:
    """Require every node or edge subject to have at least one evidence record."""

    evidence_subject_ids = {str(item.get("subject_id")) for item in evidence}
    missing = [
        str(subject.get("id"))
        for subject in subjects
        if str(subject.get("id")) not in evidence_subject_ids
    ]
    return _check(name, missing)


def _evidence_span_check(
    chunks: list[dict[str, Any]], evidence: list[dict[str, Any]]
) -> GraphQualityCheck:
    """Verify evidence offsets and quotes against their referenced chunk content.

    Persisted offsets are document-global while chunk content is local, so the
    chunk start offset is removed before bounds and exact normalized text checks.
    """

    chunks_by_id = {str(chunk.get("id")): chunk for chunk in chunks}
    invalid: list[str] = []
    for item in evidence:
        evidence_id = str(item.get("id"))
        chunk = chunks_by_id.get(str(item.get("chunk_id")))
        if chunk is None:
            invalid.append(f"{evidence_id}:missing_chunk")
            continue
        try:
            chunk_start = _integer(chunk.get("start_offset"))
            start = _integer(item.get("start_offset")) - chunk_start
            end = _integer(item.get("end_offset")) - chunk_start
        except (TypeError, ValueError):
            invalid.append(f"{evidence_id}:invalid_offsets")
            continue
        content = str(chunk.get("content", ""))
        quote = str(item.get("quote", ""))
        if start < 0 or end <= start or end > len(content):
            invalid.append(f"{evidence_id}:out_of_bounds")
        elif _normalize_text(content[start:end]) != _normalize_text(quote):
            invalid.append(f"{evidence_id}:quote_mismatch")
    return _check("evidence_spans_match_chunks", invalid)


def _isolate_check(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], maximum_ratio: float
) -> GraphQualityCheck:
    """Warn when the proportion of nodes without any canonical edge is excessive."""

    connected_ids = {
        str(endpoint)
        for edge in edges
        for endpoint in (edge.get("source_node_id"), edge.get("target_node_id"))
    }
    isolates = [str(node.get("id")) for node in nodes if str(node.get("id")) not in connected_ids]
    ratio = len(isolates) / max(len(nodes), 1)
    return GraphQualityCheck(
        name="isolate_ratio",
        ok=ratio <= maximum_ratio,
        severity="warning",
        count=len(isolates),
        details=[f"ratio={ratio:.4f}", *isolates[: _DETAIL_LIMIT - 1]],
    )


def _component_check(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], maximum_ratio: float
) -> GraphQualityCheck:
    """Warn when graph fragmentation produces too many connected components relative to nodes."""

    graph: nx.Graph[str] = nx.Graph()
    graph.add_nodes_from(str(node.get("id")) for node in nodes)
    graph.add_edges_from(
        (str(edge.get("source_node_id")), str(edge.get("target_node_id"))) for edge in edges
    )
    components = list(nx.connected_components(graph)) if graph else []
    ratio = len(components) / max(len(nodes), 1)
    return GraphQualityCheck(
        name="component_ratio",
        ok=ratio <= maximum_ratio,
        severity="warning",
        count=len(components),
        details=[f"ratio={ratio:.4f}", *[str(len(item)) for item in components[:19]]],
    )


def _relation_vocabulary_check(
    edges: list[dict[str, Any]], maximum_ratio: float
) -> GraphQualityCheck:
    """Warn when predicate vocabulary grows too quickly relative to edge count.

    Tiny graphs are exempt because one label per edge is natural before enough
    observations exist to reveal unconstrained predicate invention.
    """

    relation_types = {
        normalize_ontology_label(str(edge.get("relation_type", ""))) for edge in edges
    }
    ratio = len(relation_types) / max(len(edges), 1)
    # Tiny graphs naturally have one label per edge; warn only once a vocabulary
    # is large enough to indicate unconstrained predicate invention.
    ok = len(edges) < _MIN_EDGES_FOR_VOCABULARY_WARNING or ratio <= maximum_ratio
    return GraphQualityCheck(
        name="relation_vocabulary_ratio",
        ok=ok,
        severity="warning",
        count=len(relation_types),
        details=[f"ratio={ratio:.4f}", *sorted(relation_types)[: _DETAIL_LIMIT - 1]],
    )


def _singleton_community_check(communities: list[dict[str, Any]]) -> GraphQualityCheck:
    """Warn when reports were generated for communities below the useful size floor."""

    singleton_ids = [
        str(item.get("id"))
        for item in communities
        if len(as_list(item.get("member_node_ids"))) < _MIN_REPORTED_COMMUNITY_SIZE
    ]
    return GraphQualityCheck(
        name="no_singleton_community_reports",
        ok=not singleton_ids,
        severity="warning",
        count=len(singleton_ids),
        details=singleton_ids[:_DETAIL_LIMIT],
    )


def _embedding_dimension_check(
    name: str, rows: list[dict[str, Any]], expected_dimension: int
) -> GraphQualityCheck:
    """Require every persisted embedding to match the configured vector dimension.

    Missing vectors are reported separately from mismatched lengths.
    """

    mismatches = []
    for row in rows:
        embedding = row.get("embedding")
        if embedding is None:
            mismatches.append(f"{row.get('id')}:missing")
        elif len(embedding) != expected_dimension:
            mismatches.append(f"{row.get('id')}:{len(embedding)}")
    return _check(name, mismatches)


def _check(
    name: str,
    details: list[str],
    severity: Literal["error", "warning"] = "error",
) -> GraphQualityCheck:
    """Build one bounded quality result from a complete list of offending details."""

    return GraphQualityCheck(
        name=name,
        ok=not details,
        severity=severity,
        count=len(details),
        details=details[:_DETAIL_LIMIT],
    )


def as_list(value: object) -> list[object]:
    """Normalize list-like values from JSON, Arrow, NumPy, or Pandas artifacts.

    Unsupported scalar values become empty lists.
    """

    if isinstance(value, list | tuple | set):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        return list(converted) if isinstance(converted, list | tuple | set) else []
    return []


def _integer(value: object) -> int:
    """Coerce artifact offsets to integers while rejecting booleans and fractions.

    Numeric strings remain compatible with JSON exports.
    """

    if isinstance(value, bool):
        raise ValueError("boolean is not an offset")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError("value is not an integer offset")


def _normalize_text(value: str) -> str:
    """Normalize case and repeated whitespace for evidence quote comparison.

    Punctuation and word order remain significant.
    """

    return " ".join(unicodedata.normalize("NFC", value).casefold().split())
