# SPDX-License-Identifier: Apache-2.0
"""Canonical graph assembly from grounded extraction observations.

Merge decisions are explicit trace records. Canonical edge ids are independent
of file scope, while edge observations preserve each file/chunk assertion so
reruns and Snowflake MERGE operations remain auditable and idempotent.
"""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from kg_processor.application.type_resolution import TYPE_RESOLVED_METHOD, TypeResolution
from kg_processor.domain.graph import (
    Chunk,
    EdgeObservation,
    EntitySource,
    Evidence,
    ExtractedEntity,
    ExtractedRelation,
    GraphEdge,
    GraphNode,
    RelationAssertion,
)
from kg_processor.domain.ids import stable_id
from kg_processor.domain.ontology import OntologyProfile, normalize_ontology_label

# Persisted relation labels stay within a bounded identifier length so canonical
# predicates remain usable as Snowflake identifiers and stable join keys.
MAX_RELATION_LABEL_LENGTH = 80


@dataclass(frozen=True)
class MergeDecision:
    """Traceable record explaining how one observation affected graph assembly."""

    kind: Literal["entity", "relation"]
    action: Literal["created", "merged", "aggregated", "dropped"]
    reason: str
    graph_id: str
    node_id: str | None = None
    edge_id: str | None = None
    normalized_name: str | None = None
    primary_type: str | None = None
    source_name: str | None = None
    target_name: str | None = None
    relation_type: str | None = None
    source_chunk_id: str | None = None
    source_file_id: str | None = None
    observation_count: int | None = None
    source_chunk_ids: list[str] | None = None
    observed_names: list[str] | None = None
    original_weight: float | None = None
    effective_weight: float | None = None
    prior_weight: float | None = None
    new_weight: float | None = None

    def to_trace_event(self) -> dict[str, object]:
        """Serialize the merge decision into the extraction trace format."""

        event: dict[str, object] = {
            "stage": "merge_decision",
            "kind": self.kind,
            "action": self.action,
            "reason": self.reason,
            "graph_id": self.graph_id,
        }
        _set_if_present(event, "node_id", self.node_id)
        _set_if_present(event, "edge_id", self.edge_id)
        _set_if_present(event, "normalized_name", self.normalized_name)
        _set_if_present(event, "primary_type", self.primary_type)
        _set_if_present(event, "source_name", self.source_name)
        _set_if_present(event, "target_name", self.target_name)
        _set_if_present(event, "relation_type", self.relation_type)
        _set_if_present(event, "source_chunk_id", self.source_chunk_id)
        _set_if_present(event, "source_file_id", self.source_file_id)
        _set_if_present(event, "observation_count", self.observation_count)
        _set_if_present(event, "source_chunk_ids", self.source_chunk_ids)
        _set_if_present(event, "observed_names", self.observed_names)
        _set_if_present(event, "original_weight", self.original_weight)
        _set_if_present(event, "effective_weight", self.effective_weight)
        _set_if_present(event, "prior_weight", self.prior_weight)
        _set_if_present(event, "new_weight", self.new_weight)
        return event


@dataclass(frozen=True)
class GraphAssemblyResult:
    """Bundle canonical graph rows, provenance, and explicit merge decisions.

    Returning diagnostics with persisted artifacts allows run reports to explain
    which observations were created, merged, repaired, or dropped, and which
    held relations were admitted: their count, and the keys of the rejected
    records they no longer are.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    edge_observations: list[EdgeObservation]
    evidence: list[Evidence]
    entity_sources: list[EntitySource]
    decisions: list[MergeDecision]
    type_resolved_relations: int = 0
    type_resolved_keys: frozenset[str] = frozenset()

    def decision_reason_counts(self) -> dict[str, int]:
        """Return sorted counts of merge decision reasons."""

        counter = Counter(decision.reason for decision in self.decisions)
        return dict(sorted(counter.items()))

    def decision_action_counts(self) -> dict[str, int]:
        """Return sorted counts of merge decision actions."""

        counter = Counter(decision.action for decision in self.decisions)
        return dict(sorted(counter.items()))


@dataclass(frozen=True)
class EntityAssembly:
    """Intermediate entity assembly state consumed by relation assembly."""

    nodes: list[GraphNode]
    node_by_name_and_type: dict[tuple[str, str], GraphNode]
    node_by_component_id: dict[str, GraphNode]
    evidence: list[Evidence]
    entity_sources: list[EntitySource]
    decisions: list[MergeDecision]


@dataclass(frozen=True)
class RelationAssembly:
    """Hold canonical relation artifacts before final graph-level reconciliation.

    Edge assertions remain separate from canonical edges so repeated evidence can
    accumulate support without duplicating graph topology.
    """

    edges: list[GraphEdge]
    edge_observations: list[EdgeObservation]
    evidence: list[Evidence]
    decisions: list[MergeDecision]


def prune_isolated_entities(
    assembly: GraphAssemblyResult,
    structural_relation_types: frozenset[str] = frozenset(),
) -> GraphAssemblyResult:
    """Remove canonical entities that participate in no accepted graph relation.

    Window extraction intentionally favors recall and can surface headings,
    citation metadata, or incidental names. Requiring at least one grounded edge
    converts that broad observation inventory into an explorable knowledge graph,
    avoids expensive enrichment for disconnected records, and removes only rows
    whose provenance remains available in extraction traces.

    A structural edge, by its normalized type, states how sources are filed,
    not a relation: it keeps its target when its source is kept and keeps no
    source, and it goes with an entity it no longer joins.
    """

    connected_ids = {
        node_id
        for edge in assembly.edges
        if edge.relation_type not in structural_relation_types
        for node_id in (edge.source_node_id, edge.target_node_id)
    }
    connected_ids |= {
        edge.target_node_id
        for edge in assembly.edges
        if edge.relation_type in structural_relation_types and edge.source_node_id in connected_ids
    }
    removed = [node for node in assembly.nodes if node.id not in connected_ids]
    if not removed:
        return assembly
    edges = [
        edge
        for edge in assembly.edges
        if edge.source_node_id in connected_ids and edge.target_node_id in connected_ids
    ]
    edge_ids = {edge.id for edge in edges}
    nodes = [node for node in assembly.nodes if node.id in connected_ids]
    _assign_degrees(nodes, edges)
    decisions = [
        *assembly.decisions,
        *[
            MergeDecision(
                kind="entity",
                action="dropped",
                reason="isolated_entity_without_relation",
                graph_id=node.graph_id,
                node_id=node.id,
                normalized_name=node.normalized_name,
                primary_type=node.primary_type,
            )
            for node in removed
        ],
    ]
    return replace(
        assembly,
        nodes=nodes,
        edges=edges,
        edge_observations=[item for item in assembly.edge_observations if item.edge_id in edge_ids],
        evidence=[
            item
            for item in assembly.evidence
            if (item.subject_kind != "node" or item.subject_id in connected_ids)
            and (item.subject_kind != "edge" or item.subject_id in edge_ids)
        ],
        entity_sources=[item for item in assembly.entity_sources if item.node_id in connected_ids],
        decisions=decisions,
    )


def normalize_entity_name(name: str) -> str:
    """Normalize entity names into stable keys used for merging.

    Node IDs use this normalized name, with a component disambiguator when
    separate resolved identities share a name and type. Every engine that
    assembles graph rows must call this function rather than reimplement it: NFC
    composition and full case folding have no
    equivalent expression in SQL dialects, and any approximation mints different
    IDs for the same entity and breaks idempotent merges into shared tables.
    """

    # Keep Unicode letters/digits so names such as "Muller", "Müller", and
    # "東京" do not collapse into the same ASCII-only key or disappear entirely.
    normalized = unicodedata.normalize("NFC", name)
    stripped = "".join(char for char in normalized.casefold().strip() if char.isalnum())
    return stripped or normalized.strip().casefold()


def structural_relation_types(ontology: OntologyProfile | None) -> frozenset[str]:
    """The persisted types, as edges carry them, of the ontology's structural relations."""

    if ontology is None:
        return frozenset()
    return frozenset(
        normalize_relation_type(label) for label in ontology.structural_relation_labels()
    )


def claimed_only(assertions: Iterable[RelationAssertion]) -> RelationAssertion:
    """The assertion of an edge, or of one chunk's observation, from its sources' assertions.

    Claimed only when every assertion only claims it: one source that reports
    the relation as a fact or finding makes it reported.
    """

    stated = list(assertions)
    return "claimed" if stated and all(item == "claimed" for item in stated) else "reported"


def normalize_relation_type(relation_type: str) -> str:
    """Normalize relation labels into bounded Snowflake-friendly identifiers.

    Canonical edge IDs are hashed from this value, so it is the single relation
    label rule for every engine. The bound applies wherever a persisted
    ``relation_type`` is produced or re-derived, keeping label identity stable
    across engines even for pathologically long extracted predicates.
    """

    return normalize_ontology_label(relation_type)[:MAX_RELATION_LABEL_LENGTH]


def assemble_graph(
    graph_id: str,
    chunks: list[Chunk],
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
    relation_weight_max: float,
    *,
    ontology: OntologyProfile | None = None,
) -> tuple[list[GraphNode], list[GraphEdge], list[Evidence], list[EntitySource]]:
    """Assemble canonical graph rows without returning merge decisions."""

    result = assemble_graph_with_decisions(
        graph_id,
        chunks,
        entities,
        relations,
        relation_weight_max,
        ontology=ontology,
    )
    return result.nodes, result.edges, result.evidence, result.entity_sources


def assemble_graph_with_decisions(
    graph_id: str,
    chunks: list[Chunk],
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
    relation_weight_max: float,
    *,
    held_relations: Sequence[ExtractedRelation] = (),
    type_resolution: TypeResolution | None = None,
    ontology: OntologyProfile | None = None,
) -> GraphAssemblyResult:
    """Merge grounded observations into canonical graph rows and audit decisions.

    Resolved components retain identity even when their display names coincide;
    observations without a component group by normalized name and type. Relation assembly
    then consolidates repeated triples while retaining per-source assertions,
    exact evidence, bounded weights, and reasons for every dropped observation.
    With ``type_resolution``, held relations the entity nodes admit join the
    relations as ``type_resolved`` observations. An explicit source self-loop
    survives only when ``ontology`` allows its predicate; merged-endpoint loops
    are discarded regardless of that allowance.
    """

    # Entities are grouped before relation assembly so endpoint resolution can
    # use the same canonical node map that will be persisted.
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    grouped_entities: dict[tuple[str, str, str], list[ExtractedEntity]] = defaultdict(list)
    for entity in entities:
        grouped_entities[
            (normalize_entity_name(entity.name), entity.type, entity.resolved_component_id or "")
        ].append(entity)

    entity_assembly = _assemble_entities(graph_id, chunks_by_id, grouped_entities)
    admitted: list[ExtractedRelation] = []
    admitted_keys: list[str] = []
    if type_resolution is not None and held_relations:
        admitted, admitted_keys = _admitted_held_relations(
            held_relations, relations, entity_assembly, chunks_by_id, type_resolution
        )
    relation_assembly = _assemble_relations(
        graph_id,
        chunks_by_id,
        entity_assembly.node_by_name_and_type,
        [*relations, *admitted],
        relation_weight_max,
        entity_assembly.node_by_component_id,
        ontology,
    )
    nodes = sorted(entity_assembly.nodes, key=lambda node: node.id)
    edges = relation_assembly.edges
    _assign_degrees(nodes, edges)
    return GraphAssemblyResult(
        nodes=nodes,
        edges=edges,
        edge_observations=relation_assembly.edge_observations,
        evidence=[*entity_assembly.evidence, *relation_assembly.evidence],
        entity_sources=entity_assembly.entity_sources,
        decisions=[*entity_assembly.decisions, *relation_assembly.decisions],
        type_resolved_relations=len(admitted_keys),
        type_resolved_keys=frozenset(admitted_keys),
    )


def _admitted_held_relations(
    held_relations: Sequence[ExtractedRelation],
    relations: Sequence[ExtractedRelation],
    entity_assembly: EntityAssembly,
    chunks_by_id: dict[str, Chunk],
    type_resolution: TypeResolution,
) -> tuple[list[ExtractedRelation], list[str]]:
    """Admit each held relation its endpoints' other typings allow.

    An endpoint may stay on the node its own mention resolved to, or move to a
    node of the same name and another type attested in at least
    ``min_documents`` documents; at least one endpoint must move. Among the
    typings the rules keep, the fewest moves win, then the best-attested nodes,
    then type names and node ids. Returns the observations to assemble and the
    rejected-record key of every admitted relation. An admitted observation
    whose edge the chunk already supports adds nothing and is not assembled,
    though its relation counts as admitted: the graph states it.

    ``spark_finalization`` makes the same choice over DataFrames; the
    cross-engine tests hold the two to one outcome.
    """

    node_index = entity_assembly.node_by_name_and_type
    documents = {
        node.id: len(
            {
                chunks_by_id[chunk_id].document_id
                for chunk_id in node.source_chunk_ids
                if chunk_id in chunks_by_id
            }
        )
        for node in entity_assembly.nodes
    }
    by_name: dict[str, list[GraphNode]] = defaultdict(list)
    for node in entity_assembly.nodes:
        by_name[node.normalized_name].append(node)

    def candidates(
        name: str, entity_type: str, component_id: str | None
    ) -> list[tuple[int, GraphNode]]:
        own = _relation_endpoint_node(
            node_index, name, entity_type, component_id, entity_assembly.node_by_component_id
        )
        if own is None:
            return []
        return [
            (0, own),
            *(
                (1, node)
                for node in by_name[own.normalized_name]
                if node.primary_type != own.primary_type
                and documents[node.id] >= type_resolution.min_documents
            ),
        ]

    component_by_node = {
        node.id: component for component, node in entity_assembly.node_by_component_id.items()
    }
    # Each choice with the edge and chunk it would support.
    chosen: list[tuple[tuple[str, str, str, str], ExtractedRelation]] = []
    keys: list[str] = []
    for relation in held_relations:
        typings = [
            (
                (
                    source_moved + target_moved,
                    -documents[source.id],
                    -documents[target.id],
                    source.primary_type,
                    target.primary_type,
                    source.id,
                    target.id,
                ),
                source,
                target,
                rule,
            )
            for source_moved, source in candidates(
                relation.source_name, relation.source_type, relation.source_component_id
            )
            for target_moved, target in candidates(
                relation.target_name, relation.target_type, relation.target_component_id
            )
            if source_moved + target_moved
            and source.id != target.id
            and (
                rule := type_resolution.rules.get(
                    (relation.relation_type, source.primary_type, target.primary_type)
                )
            )
            is not None
        ]
        if not typings:
            continue
        _, source, target, (kept, stated) = min(typings, key=lambda typing: typing[0])
        keys.append(relation.rejected_record_key or "")
        edge = (source.id, target.id, normalize_relation_type(kept), relation.source_chunk_id)
        chosen.append(
            (
                edge,
                relation.model_copy(
                    update={
                        "source_name": source.name,
                        "source_component_id": component_by_node.get(source.id),
                        "source_type": source.primary_type,
                        "target_name": target.name,
                        "target_component_id": component_by_node.get(target.id),
                        "target_type": target.primary_type,
                        "relation_type": kept,
                        "proposed_relation_type": stated or relation.proposed_relation_type,
                        "evidence_method": TYPE_RESOLVED_METHOD,
                    }
                ),
            )
        )

    covered: set[tuple[str, str, str, str]] = set()
    for relation in relations:
        start = _relation_endpoint_node(
            node_index,
            relation.source_name,
            relation.source_type,
            relation.source_component_id,
            entity_assembly.node_by_component_id,
        )
        end = _relation_endpoint_node(
            node_index,
            relation.target_name,
            relation.target_type,
            relation.target_component_id,
            entity_assembly.node_by_component_id,
        )
        if start is not None and end is not None:
            covered.add(
                (
                    start.id,
                    end.id,
                    normalize_relation_type(relation.relation_type),
                    relation.source_chunk_id,
                )
            )
    admitted: list[ExtractedRelation] = []
    for edge, relation in sorted(
        chosen,
        key=lambda item: (
            item[0],
            item[1].start_offset or 0,
            item[1].end_offset or 0,
            item[1].rejected_record_key or "",
        ),
    ):
        if edge not in covered:
            covered.add(edge)
            admitted.append(relation)
    return admitted, keys


def _assemble_entities(
    graph_id: str,
    chunks_by_id: dict[str, Chunk],
    grouped_entities: dict[tuple[str, str, str], list[ExtractedEntity]],
) -> EntityAssembly:
    nodes: list[GraphNode] = []
    node_by_component_id: dict[str, GraphNode] = {}
    evidence: list[Evidence] = []
    entity_sources: list[EntitySource] = []
    decisions: list[MergeDecision] = []

    # Match the distributed engine: the best-attested component keeps the plain
    # name/type id, while other homonyms add their stable component root.
    identity_winners: dict[tuple[str, str], str] = {}
    for name, entity_type, component in sorted(
        grouped_entities,
        key=lambda key: (-len({item.source_chunk_id for item in grouped_entities[key]}), key),
    ):
        identity_winners.setdefault((name, entity_type), component)

    for (normalized_name, grouped_type, component), observed in sorted(grouped_entities.items()):
        primary_type = _primary_type(observed)
        display_name = _best_display_name([entity.name for entity in observed])
        description = _best_description([entity.description for entity in observed])
        types = sorted({entity.type for entity in observed})
        aliases = _node_aliases(observed, display_name)
        source_chunk_ids = sorted({entity.source_chunk_id for entity in observed})
        observations_by_chunk: dict[str, ExtractedEntity] = {}
        for observation in observed:
            current = observations_by_chunk.get(observation.source_chunk_id)
            if current is None or _observation_quality_key(observation) < _observation_quality_key(
                current
            ):
                observations_by_chunk[observation.source_chunk_id] = observation
        identity = [graph_id, normalized_name, primary_type]
        if component != identity_winners[(normalized_name, grouped_type)]:
            identity.append(component)
        node = GraphNode(
            id=stable_id("node", *identity),
            graph_id=graph_id,
            normalized_name=normalized_name,
            name=display_name,
            primary_type=primary_type,
            types=types,
            aliases=aliases,
            description=description,
            source_chunk_ids=source_chunk_ids,
        )
        nodes.append(node)
        if component:
            node_by_component_id[component] = node
        decisions.append(
            MergeDecision(
                kind="entity",
                action="merged" if len(observed) > 1 else "created",
                reason=(
                    "entity_observations_merged" if len(observed) > 1 else "canonical_node_created"
                ),
                graph_id=graph_id,
                node_id=node.id,
                normalized_name=normalized_name,
                primary_type=primary_type,
                observation_count=len(observed),
                source_chunk_ids=source_chunk_ids,
                observed_names=sorted({entity.name for entity in observed}),
            )
        )
        for chunk_id in source_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk:
                selected_observation = observations_by_chunk[chunk_id]
                evidence.append(
                    _evidence_for_observation(
                        graph_id,
                        node.id,
                        "node",
                        chunk,
                        selected_observation.quote,
                        selected_observation.start_offset,
                        selected_observation.end_offset,
                        selected_observation.evidence_method,
                    )
                )
        counts_by_file = Counter(
            chunks_by_id[cid].file_id for cid in source_chunk_ids if cid in chunks_by_id
        )
        for file_id, count in sorted(counts_by_file.items()):
            file_descriptions = [
                observation.description
                for observation in observed
                if (
                    (chunk := chunks_by_id.get(observation.source_chunk_id)) is not None
                    and chunk.file_id == file_id
                )
            ]
            entity_sources.append(
                EntitySource(
                    id=stable_id("entity_source", graph_id, node.id, file_id),
                    graph_id=graph_id,
                    node_id=node.id,
                    file_id=file_id,
                    per_file_description=_best_description(file_descriptions),
                    mention_count=count,
                    aliases=_node_aliases(
                        [
                            item
                            for item in observed
                            if item.source_chunk_id in chunks_by_id
                            and chunks_by_id[item.source_chunk_id].file_id == file_id
                        ],
                        "",
                    ),
                )
            )

    return EntityAssembly(
        nodes=nodes,
        node_by_name_and_type=_with_unique_alias_endpoints(nodes),
        node_by_component_id=node_by_component_id,
        evidence=evidence,
        entity_sources=entity_sources,
        decisions=decisions,
    )


def _with_unique_alias_endpoints(
    nodes: list[GraphNode],
) -> dict[tuple[str, str], GraphNode]:
    """Extend the endpoint index with aliases that identify one canonical node.

    Relations are extracted from local text and may therefore use an acronym or
    another accepted surface after entity resolution selected a longer canonical
    name. Aliases are lookup keys only; nodes, IDs, and persisted display names
    remain canonical. A colliding alias is deliberately omitted so it cannot join
    a relation to an arbitrary same-typed entity.
    """

    canonical_candidates: dict[tuple[str, str], dict[str, GraphNode]] = defaultdict(dict)
    for node in nodes:
        for node_type in node.types:
            canonical_candidates[(node.normalized_name, node_type)][node.id] = node
    endpoint_nodes = {
        key: next(iter(candidates.values()))
        for key, candidates in canonical_candidates.items()
        if len(candidates) == 1
    }
    alias_candidates: dict[tuple[str, str], dict[str, GraphNode]] = defaultdict(dict)
    for node in nodes:
        for alias in node.aliases:
            normalized_alias = normalize_entity_name(alias)
            for node_type in node.types:
                alias_candidates[(normalized_alias, node_type)][node.id] = node
    for key, candidates in alias_candidates.items():
        if key not in canonical_candidates and len(candidates) == 1:
            endpoint_nodes[key] = next(iter(candidates.values()))
    return endpoint_nodes


def _node_aliases(observed: list[ExtractedEntity], display_name: str) -> list[str]:
    """Collect deterministic accepted surfaces other than the display name.

    Exact normalized duplicates are removed so casing-only variants do not
    inflate persisted metadata. Every value already passed entity grounding and
    alias validation before graph assembly.
    """

    display_key = normalize_entity_name(display_name)
    by_key: dict[str, str] = {}
    for entity in observed:
        for surface in [entity.name, *entity.aliases]:
            stripped = surface.strip()
            key = normalize_entity_name(stripped)
            if stripped and key and key != display_key:
                by_key.setdefault(key, stripped)
    return [by_key[key] for key in sorted(by_key)]


def _primary_type(observed: list[ExtractedEntity]) -> str:
    counts = Counter(entity.type for entity in observed)
    first_seen = {entity.type: index for index, entity in enumerate(observed)}
    return sorted(
        counts,
        key=lambda entity_type: (-counts[entity_type], first_seen[entity_type]),
    )[0]


def _assemble_relations(  # noqa: PLR0915 - each branch records a distinct audit decision.
    graph_id: str,
    chunks_by_id: dict[str, Chunk],
    node_by_name_and_type: dict[tuple[str, str], GraphNode],
    relations: list[ExtractedRelation],
    relation_weight_max: float,
    node_by_component_id: dict[str, GraphNode],
    ontology: OntologyProfile | None,
) -> RelationAssembly:
    """Assemble grounded relation observations into canonical directed edges.

    Missing endpoints or chunks become explicit drop decisions. Repeated triples
    share one canonical edge while each source assertion and evidence record is
    preserved independently for provenance and reindexing.
    """

    allowed_self_loops = (
        {
            normalize_relation_type(item.name)
            for item in ontology.relation_types
            if item.allow_self_loop
        }
        if ontology is not None
        else set()
    )
    edges_by_key: dict[tuple[str, str, str], GraphEdge] = {}
    edge_chunk_ids: dict[str, set[str]] = {}
    edge_observations: list[EdgeObservation] = []
    evidence: list[Evidence] = []
    decisions: list[MergeDecision] = []
    for relation in relations:
        source = _relation_endpoint_node(
            node_by_name_and_type,
            relation.source_name,
            relation.source_type,
            relation.source_component_id,
            node_by_component_id,
        )
        target = _relation_endpoint_node(
            node_by_name_and_type,
            relation.target_name,
            relation.target_type,
            relation.target_component_id,
            node_by_component_id,
        )
        chunk = chunks_by_id.get(relation.source_chunk_id)
        if not source:
            decisions.append(_dropped_relation_decision(graph_id, relation, "source_node_missing"))
            continue
        if not target:
            decisions.append(_dropped_relation_decision(graph_id, relation, "target_node_missing"))
            continue
        if not chunk:
            decisions.append(_dropped_relation_decision(graph_id, relation, "source_chunk_missing"))
            continue
        relation_type = normalize_relation_type(relation.relation_type)
        if source.id == target.id and not (
            relation.source_self_loop and relation_type in allowed_self_loops
        ):
            # Distinct extracted spellings can resolve to the same canonical node
            # after punctuation and alias normalization. Such a relation was not a
            # source-level self-loop and therefore could not be rejected earlier.
            decisions.append(
                _dropped_relation_decision(graph_id, relation, "self_loop_after_entity_merge")
            )
            continue
        key = (source.id, target.id, relation_type)
        weight = min(max(relation.weight, 0.0), relation_weight_max)
        prior = edges_by_key.get(key)
        if prior is None:
            edge_id = stable_id(
                "edge",
                graph_id,
                source.id,
                target.id,
                relation_type,
            )
            edge = GraphEdge(
                id=edge_id,
                graph_id=graph_id,
                source_node_id=source.id,
                target_node_id=target.id,
                relation_type=relation_type,
                description=relation.description[:1000],
                weight=weight,
                confidence=relation.confidence,
                source_file_id=chunk.file_id,
                source_file_ids=[chunk.file_id],
                source_chunk_ids=[chunk.id],
                evidence_count=1,
                assertion=relation.assertion,
            )
            edges_by_key[key] = edge
            edge_chunk_ids[edge.id] = {chunk.id}
            decisions.append(
                MergeDecision(
                    kind="relation",
                    action="created",
                    reason=(
                        "canonical_edge_created_weight_clamped"
                        if weight != relation.weight
                        else "canonical_edge_created"
                    ),
                    graph_id=graph_id,
                    edge_id=edge.id,
                    source_name=relation.source_name,
                    target_name=relation.target_name,
                    relation_type=relation_type,
                    source_chunk_id=chunk.id,
                    source_file_id=chunk.file_id,
                    original_weight=relation.weight,
                    effective_weight=weight,
                    new_weight=edge.weight,
                )
            )
            edge_evidence = _evidence_for_observation(
                graph_id,
                edge.id,
                "edge",
                chunk,
                relation.quote,
                relation.start_offset,
                relation.end_offset,
                relation.evidence_method,
            )
            evidence.append(edge_evidence)
            edge_observations.append(
                _edge_observation(graph_id, edge, chunk, relation, weight, edge_evidence.id)
            )
        else:
            prior_weight = prior.weight
            new_chunk = chunk.id not in edge_chunk_ids[prior.id]
            if not new_chunk:
                _merge_repeated_observation(
                    edge_observations,
                    prior,
                    stable_id("edge_observation", graph_id, prior.id, chunk.id),
                    relation,
                )
                decisions.append(
                    _dropped_relation_decision(
                        graph_id,
                        relation,
                        "duplicate_edge_observation",
                    )
                )
                continue
            prior.source_chunk_ids.append(chunk.id)
            edge_chunk_ids[prior.id].add(chunk.id)
            prior.source_file_ids.append(chunk.file_id)
            prior.source_file_ids = sorted(set(prior.source_file_ids))
            prior.source_file_id = prior.source_file_ids[0]
            prior.evidence_count += 1
            prior.weight = min(prior.weight + weight, relation_weight_max)
            prior.confidence = max(prior.confidence, relation.confidence)
            prior.assertion = claimed_only((prior.assertion, relation.assertion))
            if (len(relation.description), relation.description) > (
                len(prior.description),
                prior.description,
            ):
                prior.description = relation.description[:1000]
            decisions.append(
                MergeDecision(
                    kind="relation",
                    action="aggregated",
                    reason=(
                        "relation_evidence_aggregated_weight_clamped"
                        if prior_weight + weight > relation_weight_max
                        else "relation_evidence_aggregated"
                    ),
                    graph_id=graph_id,
                    edge_id=prior.id,
                    source_name=relation.source_name,
                    target_name=relation.target_name,
                    relation_type=relation_type,
                    source_chunk_id=chunk.id,
                    source_file_id=chunk.file_id,
                    original_weight=relation.weight,
                    effective_weight=weight,
                    prior_weight=prior_weight,
                    new_weight=prior.weight,
                )
            )
            edge_evidence = _evidence_for_observation(
                graph_id,
                prior.id,
                "edge",
                chunk,
                relation.quote,
                relation.start_offset,
                relation.end_offset,
                relation.evidence_method,
            )
            evidence.append(edge_evidence)
            edge_observations.append(
                _edge_observation(
                    graph_id,
                    prior,
                    chunk,
                    relation,
                    weight,
                    edge_evidence.id,
                )
            )

    edges = sorted(edges_by_key.values(), key=lambda edge: edge.id)
    for edge in edges:
        edge.source_chunk_ids.sort()
        edge.source_file_ids = sorted(set(edge.source_file_ids))
    return RelationAssembly(
        edges=edges,
        edge_observations=sorted(edge_observations, key=lambda item: item.id),
        evidence=sorted(evidence, key=lambda item: item.id),
        decisions=decisions,
    )


def _relation_endpoint_node(
    nodes: dict[tuple[str, str], GraphNode],
    name: str,
    entity_type: str,
    component_id: str | None = None,
    component_nodes: dict[str, GraphNode] | None = None,
) -> GraphNode | None:
    """Resolve a relation endpoint without conflating same-named typed entities."""

    if component_id is not None:
        return (component_nodes or {}).get(component_id)
    return nodes.get((normalize_entity_name(name), entity_type))


def _merge_repeated_observation(
    observations: list[EdgeObservation],
    edge: GraphEdge,
    observation_id: str,
    relation: ExtractedRelation,
) -> None:
    """Fold a repeated assertion into the observation already kept for its chunk.

    Its values are added, and a reported repetition makes the observation and
    its edge reported.
    """

    observation = next((item for item in reversed(observations) if item.id == observation_id), None)
    if observation is None:
        return
    observation.assertion = claimed_only((observation.assertion, relation.assertion))
    edge.assertion = claimed_only((edge.assertion, relation.assertion))
    known = {quantity.text for quantity in observation.quantities}
    observation.quantities.extend(item for item in relation.quantities if item.text not in known)


def _edge_observation(
    graph_id: str,
    edge: GraphEdge,
    chunk: Chunk,
    relation: ExtractedRelation,
    weight: float,
    evidence_id: str,
) -> EdgeObservation:
    """Preserve one supporting assertion independently from its canonical edge.

    This record retains file, chunk, description, weight, confidence, and evidence
    identity even when many observations consolidate into one graph relation.
    """

    return EdgeObservation(
        id=stable_id("edge_observation", graph_id, edge.id, chunk.id),
        graph_id=graph_id,
        edge_id=edge.id,
        file_id=chunk.file_id,
        chunk_id=chunk.id,
        description=relation.description[:1000],
        weight=weight,
        confidence=relation.confidence,
        evidence_id=evidence_id,
        quantities=relation.quantities,
        proposed_relation_type=relation.proposed_relation_type,
        assertion=relation.assertion,
    )


def _best_display_name(names: list[str]) -> str:
    return sorted(names, key=lambda name: (-_display_score(name), len(name), name))[0]


def _display_score(name: str) -> int:
    score = 0
    if any(not char.isalnum() and not char.isspace() for char in name):
        score += 3
    alpha = "".join(char for char in name if char.isalpha())
    if alpha and not alpha.islower() and not alpha.isupper():
        score += 2
    elif alpha and alpha.isupper():
        score += 1
    return score


def _best_description(descriptions: list[str]) -> str:
    return max(descriptions, key=lambda description: (len(description), description), default="")


def _observation_quality_key(observation: ExtractedEntity) -> tuple[bool, int]:
    """Prefer, in a chunk, a mention the text makes to a folder mention, then the
    richest description; the Spark finalizer chooses the same way."""

    return observation.evidence_method == "folder", -len(observation.description)


def _evidence_for_observation(  # noqa: PLR0913 - one observation's evidence context.
    graph_id: str,
    subject_id: str,
    subject_kind: str,
    chunk: Chunk,
    quote: str | None,
    start_offset: int | None,
    end_offset: int | None,
    method: str = "llm",
) -> Evidence:
    evidence_quote, evidence_start, evidence_end = _evidence_quote_and_offsets(
        chunk,
        quote,
        start_offset,
        end_offset,
    )
    return Evidence(
        id=stable_id("evidence", graph_id, subject_id, chunk.id),
        graph_id=graph_id,
        subject_id=subject_id,
        subject_kind=subject_kind,
        file_id=chunk.file_id,
        chunk_id=chunk.id,
        page_number=chunk.page_number,
        start_offset=evidence_start,
        end_offset=evidence_end,
        quote=evidence_quote,
        method=method,
    )


def _evidence_quote_and_offsets(
    chunk: Chunk,
    quote: str | None,
    start_offset: int | None,
    end_offset: int | None,
) -> tuple[str, int, int]:
    # Every extractor grounds its observations; a missing span is a bug, not repairable.
    if start_offset is None or end_offset is None:
        raise ValueError(f"observation in chunk {chunk.id} has no grounded offsets")
    local_quote = quote or chunk.content[start_offset:end_offset]
    return (
        local_quote.strip(),
        chunk.start_offset + start_offset,
        chunk.start_offset + end_offset,
    )


def _assign_degrees(nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
    degree: Counter[str] = Counter()
    for edge in edges:
        degree[edge.source_node_id] += 1
        degree[edge.target_node_id] += 1
    for node in nodes:
        node.degree = degree[node.id]
        node.rank = float(node.degree)


def _dropped_relation_decision(
    graph_id: str,
    relation: ExtractedRelation,
    reason: str,
) -> MergeDecision:
    return MergeDecision(
        kind="relation",
        action="dropped",
        reason=reason,
        graph_id=graph_id,
        source_name=relation.source_name,
        target_name=relation.target_name,
        relation_type=normalize_relation_type(relation.relation_type),
        source_chunk_id=relation.source_chunk_id,
        original_weight=relation.weight,
    )


def _set_if_present(event: dict[str, object], key: str, value: object) -> None:
    if value is not None:
        event[key] = value
