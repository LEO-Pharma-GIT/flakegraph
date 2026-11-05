# SPDX-License-Identifier: Apache-2.0
"""Deterministic grounding and filtering before graph merge.

Filters are intentionally free and explainable: static blocklists, confidence
thresholds, source-chunk grounding, and endpoint grounding run before any graph
rows are created so dropped observations can be traced without mutating state.
"""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from kg_processor.application.extraction_grounding import (
    SurfaceTextIndex,
    build_surface_text_index,
    indexed_surface_occurs,
)
from kg_processor.application.graph_merge import _set_if_present, normalize_entity_name
from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractedRelation

_DEFAULT_BLOCKLIST = [
    "chapter",
    "section",
    "introduction",
    "conclusion",
    "paragraph",
    "document",
    "page",
]


@dataclass(frozen=True)
class FilterDecision:
    """Traceable decision explaining why an observation was kept or dropped."""

    kind: Literal["entity", "relation"]
    action: Literal["kept", "dropped"]
    reason: str
    source_chunk_id: str
    name: str | None = None
    source_name: str | None = None
    target_name: str | None = None
    relation_type: str | None = None
    confidence: float | None = None

    def to_trace_event(self) -> dict[str, object]:
        """Serialize the decision into the extraction trace format."""

        event: dict[str, object] = {
            "stage": "filter_decision",
            "kind": self.kind,
            "action": self.action,
            "reason": self.reason,
            "source_chunk_id": self.source_chunk_id,
        }
        _set_if_present(event, "name", self.name)
        _set_if_present(event, "source_name", self.source_name)
        _set_if_present(event, "target_name", self.target_name)
        _set_if_present(event, "relation_type", self.relation_type)
        _set_if_present(event, "confidence", self.confidence)
        return event


@dataclass(frozen=True)
class EntityFilterResult:
    """Filtered entity observations plus their explainable decisions."""

    kept: list[ExtractedEntity]
    decisions: list[FilterDecision]

    def dropped_reason_counts(self) -> dict[str, int]:
        """Return counts for dropped entity reasons."""

        return _reason_counts(self.decisions)


@dataclass(frozen=True)
class RelationFilterResult:
    """Filtered relation observations plus their explainable decisions."""

    kept: list[ExtractedRelation]
    decisions: list[FilterDecision]

    def dropped_reason_counts(self) -> dict[str, int]:
        """Return counts for dropped relation reasons."""

        return _reason_counts(self.decisions)


@dataclass(frozen=True)
class _EntityEndpointIndex:
    """Separate canonical endpoint names from lower-priority alias candidates."""

    canonical: dict[tuple[str, str], set[tuple[str, str]]]
    aliases: dict[tuple[str, str], set[tuple[str, str]]]


def filter_entities_with_decisions(
    entities: list[ExtractedEntity],
    chunks_by_id: dict[str, Chunk],
    min_confidence: float = 0.0,
    min_name_length: int = 2,
    blocklist: list[str] | None = None,
) -> EntityFilterResult:
    """Filter entity observations while preserving reviewable decisions.

    Identity resolution can replace a locally grounded mention with a canonical
    spelling drawn from another chunk. In that case the original spelling is
    retained as a validated alias, so either the canonical name or one of those
    aliases is sufficient source grounding for this individual observation.
    """

    kept: list[ExtractedEntity] = []
    decisions: list[FilterDecision] = []
    grounding_indexes: dict[str, SurfaceTextIndex] = {}
    effective_blocklist = _DEFAULT_BLOCKLIST if blocklist is None else blocklist
    normalized_blocklist = {normalize_entity_name(value) for value in effective_blocklist}
    for entity in entities:
        if entity.confidence < min_confidence:
            decisions.append(_entity_decision(entity, "dropped", "low_confidence"))
            continue
        if len(entity.name.strip()) < min_name_length and _is_latin_only_name(entity.name):
            decisions.append(_entity_decision(entity, "dropped", "name_too_short"))
            continue
        normalized = normalize_entity_name(entity.name)
        if normalized in normalized_blocklist:
            decisions.append(_entity_decision(entity, "dropped", "blocklisted_entity"))
            continue
        source_chunk = chunks_by_id.get(entity.source_chunk_id)
        if source_chunk is None:
            decisions.append(_entity_decision(entity, "dropped", "missing_source_chunk"))
            continue
        if _grounded_at_extraction(entity.quote, entity.start_offset):
            # Extraction grounded this observation to an exact span under its
            # full contract - rephrased names, located lines - as a fleet run's
            # finalizer trusts it; checking the name again here would undo that.
            kept.append(entity)
            decisions.append(_entity_decision(entity, "kept", "grounded_at_extraction"))
            continue
        grounding_index = grounding_indexes.get(entity.source_chunk_id)
        if grounding_index is None:
            grounding_index = build_surface_text_index(source_chunk.content)
            grounding_indexes[entity.source_chunk_id] = grounding_index
        if indexed_surface_occurs(grounding_index, [entity.name]):
            kept.append(entity)
            decisions.append(_entity_decision(entity, "kept", "grounded"))
        elif indexed_surface_occurs(grounding_index, entity.aliases):
            kept.append(entity)
            decisions.append(_entity_decision(entity, "kept", "grounded_alias"))
        else:
            decisions.append(_entity_decision(entity, "dropped", "ungrounded_name"))
    return EntityFilterResult(kept=kept, decisions=decisions)


def _grounded_at_extraction(quote: str | None, start_offset: int | None) -> bool:
    """Whether extraction already tied the observation to an exact source span."""

    return bool(quote) and start_offset is not None


def _is_latin_only_name(value: str) -> bool:
    """Return whether letters in a name all belong to the Latin script."""

    letters = [character for character in value if character.isalpha()]
    return bool(letters) and all(
        "LATIN" in unicodedata.name(character, "") for character in letters
    )


def filter_relations_with_decisions(
    relations: list[ExtractedRelation],
    entities: list[ExtractedEntity],
    chunks_by_id: dict[str, Chunk] | None = None,
    min_confidence: float = 0.0,
    require_endpoint_grounding: bool = True,
) -> RelationFilterResult:
    """Filter relation observations while preserving reviewable decisions.

    Entity membership uses canonical names because those names determine graph
    identity. Evidence grounding uses the verified local endpoint surfaces when
    available, allowing an acronym, shortened name, or document-context phrase
    to support a relation whose canonical endpoint has a different spelling.
    """

    endpoint_index = _entity_endpoint_index(entities)
    grounding_indexes: dict[str, SurfaceTextIndex] = {}
    kept: list[ExtractedRelation] = []
    decisions: list[FilterDecision] = []
    for relation in relations:
        if relation.confidence < min_confidence:
            decisions.append(_relation_decision(relation, "dropped", "low_confidence"))
            continue
        if not _endpoint_is_known(
            endpoint_index,
            relation.source_name,
            relation.source_type,
        ):
            decisions.append(_relation_decision(relation, "dropped", "source_entity_missing"))
            continue
        if not _endpoint_is_known(
            endpoint_index,
            relation.target_name,
            relation.target_type,
        ):
            decisions.append(_relation_decision(relation, "dropped", "target_entity_missing"))
            continue
        if (
            require_endpoint_grounding
            and chunks_by_id is not None
            and not _grounded_at_extraction(relation.quote, relation.start_offset)
        ):
            source_text = chunks_by_id.get(relation.source_chunk_id)
            if source_text is None:
                decisions.append(_relation_decision(relation, "dropped", "missing_source_chunk"))
                continue
            source_surface = relation.source_surface or relation.source_name
            target_surface = relation.target_surface or relation.target_name
            grounding_index = grounding_indexes.get(relation.source_chunk_id)
            if grounding_index is None:
                grounding_index = build_surface_text_index(source_text.content)
                grounding_indexes[relation.source_chunk_id] = grounding_index
            if not indexed_surface_occurs(grounding_index, [source_surface]):
                decisions.append(
                    _relation_decision(relation, "dropped", "source_endpoint_ungrounded")
                )
                continue
            if not indexed_surface_occurs(grounding_index, [target_surface]):
                decisions.append(
                    _relation_decision(relation, "dropped", "target_endpoint_ungrounded")
                )
                continue
        kept.append(relation)
        decisions.append(_relation_decision(relation, "kept", "grounded"))
    return RelationFilterResult(kept=kept, decisions=decisions)


def _entity_endpoint_index(
    entities: list[ExtractedEntity],
) -> _EntityEndpointIndex:
    """Index canonical names and accepted aliases without losing type identity.

    Entity resolution can replace a relation's local endpoint spelling with a
    canonical name while retaining the local spelling as an alias. The relation
    pass still emits the local spelling, so endpoint validation must recognize
    both surfaces. Values are canonical typed identities rather than booleans so
    a surface shared by two entities remains detectably ambiguous.
    """

    canonical: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    aliases: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for entity in entities:
        identity = (normalize_entity_name(entity.name), entity.type)
        canonical[identity].add(identity)
        for alias in entity.aliases:
            aliases[(normalize_entity_name(alias), entity.type)].add(identity)
    return _EntityEndpointIndex(canonical=dict(canonical), aliases=dict(aliases))


def _endpoint_is_known(
    index: _EntityEndpointIndex,
    name: str,
    entity_type: str,
) -> bool:
    """Return whether a typed relation surface identifies one accepted entity.

    The type is part of the key because two ontology types may legitimately
    share a name.
    """

    key = (normalize_entity_name(name), entity_type)
    canonical = index.canonical.get(key, set())
    return len(canonical or index.aliases.get(key, set())) == 1


def _entity_decision(
    entity: ExtractedEntity,
    action: Literal["kept", "dropped"],
    reason: str,
) -> FilterDecision:
    return FilterDecision(
        kind="entity",
        action=action,
        reason=reason,
        source_chunk_id=entity.source_chunk_id,
        name=entity.name,
        confidence=entity.confidence,
    )


def _relation_decision(
    relation: ExtractedRelation,
    action: Literal["kept", "dropped"],
    reason: str,
) -> FilterDecision:
    return FilterDecision(
        kind="relation",
        action=action,
        reason=reason,
        source_chunk_id=relation.source_chunk_id,
        source_name=relation.source_name,
        target_name=relation.target_name,
        relation_type=relation.relation_type,
        confidence=relation.confidence,
    )


def _reason_counts(decisions: list[FilterDecision]) -> dict[str, int]:
    counter = Counter(decision.reason for decision in decisions if decision.action == "dropped")
    return dict(sorted(counter.items()))
