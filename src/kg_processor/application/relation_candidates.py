# SPDX-License-Identifier: Apache-2.0
"""Deterministic ontology-cue candidates for relation-recall auditing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from kg_processor.application.extraction_grounding import (
    build_surface_text_index,
    indexed_surface_spans,
    sentence_spans,
)
from kg_processor.domain.extraction import EntityMention, ExtractionWindow, RelationObservation
from kg_processor.domain.ids import stable_id
from kg_processor.domain.ontology import OntologyProfile, RelationTypeDefinition

_MIN_RELATION_ENTITIES = 2
_MIN_SHORTENABLE_NAME_TOKENS = 3


@dataclass(frozen=True)
class _GroundedEntity:
    """Bind one inventory entity to an exact sentence-local surface occurrence."""

    entity_id: str
    surface: str
    entity_type: str
    start: int
    end: int


def discover_cue_relation_candidates(
    window: ExtractionWindow,
    entities: list[EntityMention],
    ontology: OntologyProfile,
) -> list[RelationObservation]:
    """Propose typed relation pairs from local entity surfaces and ontology cues.

    Generative extraction can omit one fact from an otherwise productive sentence.
    This bounded audit examines only sentences containing configured predicate cues
    and at least two exact entity surfaces. A unique, asserted cue between typed
    endpoints is independently grounded evidence; ambiguous, modal, negated, and
    possessive constructions remain subject to semantic model verification.
    """

    candidates: list[RelationObservation] = []
    seen: set[tuple[str, str, str, str, int]] = set()
    for chunk in window.chunks:
        for sentence in sentence_spans(chunk.content):
            grounded = _grounded_entities(sentence.quote, entities)
            if len(grounded) < _MIN_RELATION_ENTITIES:
                continue
            for definition in ontology.relation_types:
                for cue in _cue_matches(sentence.quote, definition):
                    for pair_index, (source, target) in enumerate(
                        _pairs_across_cue(grounded, cue, definition)
                    ):
                        # The nearest pair retains the existing deterministic fast
                        # path. More distant pairs commonly arise from coordinated
                        # lists, but may also be incidental co-occurrences, so they
                        # always require independent semantic verification.
                        verification_required = pair_index > 0 or not _is_direct_cue_assertion(
                            sentence.quote,
                            cue,
                            source,
                            target,
                            ontology,
                        )
                        key = (
                            source.entity_id,
                            target.entity_id,
                            definition.name,
                            chunk.id,
                            sentence.start_offset,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(
                            RelationObservation(
                                id=stable_id("cue_relation_observation", *key),
                                source_entity_id=source.entity_id,
                                target_entity_id=target.entity_id,
                                source_surface=source.surface,
                                target_surface=target.surface,
                                relation_type=definition.name,
                                description=sentence.quote,
                                source_chunk_id=chunk.id,
                                quote=sentence.quote,
                                confidence=1.0,
                                evidence_method="ontology_cue",
                                verification_required=verification_required,
                                start_offset=sentence.start_offset,
                                end_offset=sentence.end_offset,
                            )
                        )
    return candidates


def _grounded_entities(
    sentence: str,
    entities: list[EntityMention],
) -> list[_GroundedEntity]:
    """Return every exact local occurrence of each grounded inventory surface."""

    sentence_index = build_surface_text_index(sentence)
    grounded = [
        _GroundedEntity(entity.id, sentence[start:end], entity.type, start, end)
        for entity in entities
        for start, end in indexed_surface_spans(sentence_index, _candidate_surfaces(entity))
    ]
    # Prefer the most specific surface when names overlap. For example, the
    # occurrence ``World Taekwondo`` must not also bind the nested art name
    # ``taekwondo`` as the grammatical subject of the same assertion.
    accepted: list[_GroundedEntity] = []
    for candidate in sorted(
        grounded,
        key=lambda item: (item.start, -(item.end - item.start), item.entity_id),
    ):
        if any(
            candidate.start < existing.end and existing.start < candidate.end
            for existing in accepted
        ):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: (item.start, item.end, item.entity_id))


def _is_direct_cue_assertion(
    sentence: str,
    cue: re.Match[str],
    source: _GroundedEntity,
    target: _GroundedEntity,
    ontology: OntologyProfile,
) -> bool:
    """Return whether a configured cue is safe deterministic evidence.

    An ontology cue acts as a second extractor only when exactly one configured
    predicate matches that source span. Negation, modal language, and possessive
    targets can change the proposition's scope, so those cases continue through
    the LLM verifier instead of receiving deterministic acceptance.
    """

    matching_relations = {
        definition.name
        for definition in ontology.relation_types
        if definition.admits(source.entity_type, target.entity_type)
        for match in _cue_matches(sentence, definition)
        if match.span() == cue.span()
    }
    if len(matching_relations) != 1:
        return False
    context = sentence[max(0, cue.start() - 48) : cue.start()]
    if re.search(
        r"\b(?:not|never|no|could|might|may|would|should|can)\b[^.!?]{0,32}$",
        context,
        flags=re.IGNORECASE,
    ):
        return False
    source_suffix = sentence[source.end : cue.start()]
    continuation = re.match(r"\s+([a-z][\w-]*)", source_suffix)
    if continuation is not None and continuation.group(1).casefold() not in {
        "are",
        "as",
        "authored",
        "became",
        "combines",
        "developed",
        "documented",
        "emphasizes",
        "founded",
        "governed",
        "had",
        "has",
        "have",
        "in",
        "includes",
        "influenced",
        "is",
        "of",
        "occurred",
        "originated",
        "popularized",
        "practiced",
        "preceded",
        "presented",
        "recognized",
        "regulated",
        "taught",
        "uses",
        "was",
        "were",
    }:
        return False
    suffix = sentence[target.end : target.end + 2]
    return not suffix.startswith(("'s", "\N{RIGHT SINGLE QUOTATION MARK}s"))


def _candidate_surfaces(entity: EntityMention) -> list[str]:
    """Return grounded identity, context, and shortened numeric-name surfaces.

    Events, versions, agreements, and products are often referenced by a stable
    text-and-number prefix such as ``Sydney 2000`` after first appearing under a
    longer official name. Numeric identity makes this shortening substantially less
    ambiguous than generic head-noun truncation. A document-context mention may
    additionally use only its ontology-configured discourse phrases. Ontology
    validation guarantees those phrases identify at most one context type, so an
    exact sentence such as ``this paper addresses X`` can recover a missed relation
    without inventing a semantic alias. Semantic verification still judges every
    non-direct resulting candidate.
    """

    surfaces = [
        entity.name,
        *entity.aliases,
        *(entity.contextual_surfaces if entity.is_document_context else []),
    ]
    tokens = re.findall(r"[\w]+", entity.name, flags=re.UNICODE)
    if len(tokens) >= _MIN_SHORTENABLE_NAME_TOKENS:
        for end in range(2, len(tokens)):
            prefix = tokens[:end]
            if any(any(character.isdigit() for character in token) for token in prefix):
                surfaces.append(" ".join(prefix))
    return list(dict.fromkeys(surface for surface in surfaces if surface.strip()))


def _cue_matches(sentence: str, definition: RelationTypeDefinition) -> list[re.Match[str]]:
    """Return every configured predicate occurrence in deterministic source order."""

    matches = [
        match
        for cue in definition.evidence_cues
        for match in re.finditer(cue, sentence, flags=re.IGNORECASE)
    ]
    return sorted(matches, key=lambda match: (match.start(), match.end(), match.group(0)))


def _pairs_across_cue(
    entities: list[_GroundedEntity],
    cue: re.Match[str],
    definition: RelationTypeDefinition,
) -> list[tuple[_GroundedEntity, _GroundedEntity]]:
    """Return type-valid endpoint pairs physically surrounding a predicate cue.

    Requiring the predicate to lie between mentions prevents an unrelated cue later
    in a long sentence from connecting entities on the same side. All cross-cue
    pairs are retained because coordinated subjects and objects independently state
    several facts, such as ``models A and B use method C`` or ``model A was trained
    on datasets B and C``. Distance ordering preserves the nearest-pair fast path;
    callers require semantic verification for every additional pair. When ontology
    types uniquely determine direction they override word order; otherwise ordinary
    left-to-right subject/object order is retained.
    """

    candidates: list[tuple[int, int, str, str, _GroundedEntity, _GroundedEntity]] = []
    left_entities = [entity for entity in entities if entity.end <= cue.start()]
    right_entities = [entity for entity in entities if entity.start >= cue.end()]
    for left in left_entities:
        for right in right_entities:
            if left.entity_id == right.entity_id and not definition.allow_self_loop:
                continue
            oriented = _orient_pair(left, right, definition)
            if oriented is None:
                continue
            gap = cue.start() - left.end + right.start - cue.end()
            width = right.end - left.start
            candidates.append((gap, width, oriented[0].entity_id, oriented[1].entity_id, *oriented))
    candidates.sort(key=lambda item: item[:4])
    return [(item[4], item[5]) for item in candidates]


def _orient_pair(
    left: _GroundedEntity,
    right: _GroundedEntity,
    definition: RelationTypeDefinition,
) -> tuple[_GroundedEntity, _GroundedEntity] | None:
    """Orient a pair using ontology types, falling back to sentence order."""

    forward = definition.admits(left.entity_type, right.entity_type)
    reverse = definition.admits(right.entity_type, left.entity_type)
    if forward:
        return left, right
    if reverse:
        return right, left
    return None
