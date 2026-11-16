# SPDX-License-Identifier: Apache-2.0
"""LLM-backed implementations of the extraction-stage ports.

These adapters translate strict provider payloads into grounded domain records.
Shape errors fail the individual call; semantic errors reject only the affected
record and are counted in the returned trace.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from itertools import batched
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from kg_processor.application.content_kinds import (
    DATASET_KINDS,
    DATASET_PROFILE_STAGE,
    catalogue_columns,
    location_page_name,
)
from kg_processor.application.entity_names import identity_surface_match, validate_identity_aliases
from kg_processor.application.extraction_contracts import (
    EndpointCandidate,
    EntityCandidate,
    EntityVerificationCandidate,
    RelationCandidate,
    VerificationCandidate,
    dataset_profile_request,
    document_context_extraction_request,
    entity_extraction_request,
    entity_grounding_request,
    relation_extraction_request,
    relation_grounding_request,
    relation_relabel_request,
    relation_verification_request,
)
from kg_processor.application.extraction_grounding import (
    MAX_PIECED_QUOTE_CHARACTERS,
    GroundedSpan,
    contains_surface_groups,
    delimited_row_span,
    find_quote,
    ground_evidence,
    ground_evidence_ocr_tolerant,
    name_shares_stem,
    name_words_in_quote,
    ocr_tolerant_surface_occurs,
    ocr_tolerant_surface_spans,
    row_names,
    sentence_spans,
    surface_occurs,
    surface_spans,
)
from kg_processor.application.quantities import is_value_text, parse_quantity
from kg_processor.application.rejected_records import (
    REJECTED_QUOTE_CHARACTERS as _REJECTED_QUOTE_CHARACTERS,
)
from kg_processor.application.rejected_records import (
    candidate_detail as _candidate_detail,
)
from kg_processor.application.rejected_records import rejected_record_key
from kg_processor.application.relation_grounding import relation_evidence_supported
from kg_processor.domain.extraction import (
    EntityExtractionOutcome,
    EntityMention,
    EntityVerificationDecision,
    ExtractionWindow,
    RelationExtractionOutcome,
    RelationObservation,
    VerificationDecision,
    VerificationOutcome,
)
from kg_processor.domain.graph import Chunk, Quantity
from kg_processor.domain.ids import stable_id
from kg_processor.domain.ontology import (
    OntologyProfile,
    RelationTypeDefinition,
    normalize_ontology_label,
)
from kg_processor.ports.llm import StructuredCompletionProvider, StructuredCompletionRequest

_MIN_RELATION_ENTITIES = 2


# The LLM grounding fallback asks about this many relations per call, shows each
# evidence unit up to this many characters, and shows units without markup.
# A name ending in a parenthesized abbreviation of letters: 'Hydrofluoroalkanes (HFAs)'.
_TRAILING_ABBREVIATION_RE = re.compile(r"(.+?)\s*\(\s*([A-Za-z][A-Za-z .&-]{0,11})\s*\)")
_MIN_ABBREVIATION_CAPITALS = 2
_TRADEMARK_MARKS = frozenset({"TM", "SM"})
_GROUNDING_BATCH_SIZE = 40
_MAX_UNIT_CHARACTERS = 400
_MARKUP_RE = re.compile(r"<[^>]+>")


class LlmEntityExtractor:
    """Extract and independently ground entity mentions through an LLM port.

    Provider records are treated as untrusted candidates. Ontology labels,
    aliases, chunk identity, evidence spans, duplicates, and output limits are
    enforced before a mention can enter resolution or graph assembly.
    """

    def __init__(
        self,
        llm: StructuredCompletionProvider,
        seed: int = 17,
        *,
        grounding_fallback: bool = False,
    ) -> None:
        """Store the structured provider and deterministic seed used by all entity requests.

        ``grounding_fallback`` lets one extra call per window point at the lines
        that name entities string grounding could not place.
        """

        self.llm = llm
        self.seed = seed
        self.grounding_fallback = grounding_fallback

    def _locate_entities(
        self,
        window: ExtractionWindow,
        pending: list[_UnlocatedEntity],
        *,
        model: str,
        timeout_seconds: int,
        reasons: Counter[str],
    ) -> tuple[list[EntityMention], list[tuple[_UnlocatedEntity, str]]]:
        return _locate_entities(
            self.llm,
            self.seed,
            window,
            pending,
            model=model,
            timeout_seconds=timeout_seconds,
            reasons=reasons,
        )

    def extract(
        self,
        window: ExtractionWindow,
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        max_entities: int,
        previous_entities: list[EntityMention] | None = None,
    ) -> EntityExtractionOutcome:
        """Return only ontology-valid candidates anchored to exact source text.

        Invalid siblings are counted and discarded independently, allowing one
        local model mistake to remain local while preserving valid observations
        and complete provider provenance in the trace.
        """

        request = entity_extraction_request(
            window,
            ontology,
            model=model,
            timeout_seconds=timeout_seconds,
            max_entities=max_entities,
            seed=self.seed,
            previous_entities=previous_entities,
        )
        return self._extract_from_request(
            window,
            ontology,
            request,
            max_entities=max_entities,
            previous_entities=previous_entities,
            stage="entity_extraction",
            document_context=False,
        )

    def profile_dataset(
        self,
        window: ExtractionWindow,
        ontology: OntologyProfile,
        *,
        file_name: str,
        model: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Say what a dataset is - catalogue, measurements, or other - and summarize it.

        The answer is a trace event: the document's kind steers its extraction
        passes, and finalization gives the document its summary. A catalogue's
        column relations are kept where the ontology can hold them and carried
        to its relation windows; the columns turned away are listed with why.
        """

        completion = self.llm.complete_structured(
            dataset_profile_request(
                window,
                ontology,
                file_name=file_name,
                model=model,
                timeout_seconds=timeout_seconds,
                seed=self.seed,
            )
        )
        kind = completion.payload.get("kind")
        proposed = completion.payload.get("columns")
        columns, rejected = (
            catalogue_columns(proposed if isinstance(proposed, list) else [], ontology)
            if kind == "catalogue"
            else ([], [])
        )
        return {
            "stage": DATASET_PROFILE_STAGE,
            "window_id": window.id,
            **_window_scope(window),
            "kind": kind if kind in DATASET_KINDS else None,
            "summary": str(completion.payload.get("summary") or "").strip(),
            "columns": [column.model_dump(mode="json", exclude_none=True) for column in columns],
            "rejected_columns": rejected,
            **completion.provider_metadata,
        }

    def extract_document_context_entities(
        self,
        window: ExtractionWindow,
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        max_entities: int,
    ) -> EntityExtractionOutcome:
        """Identify reusable focal entities from bounded document front matter.

        Context mentions are ordinary grounded entities with an additional discourse
        role. Stable contextual surfaces let later independent windows resolve
        explicit phrases such as ``this paper`` without making those pronouns global
        identity aliases during graph-wide entity resolution.
        """

        request = document_context_extraction_request(
            window,
            ontology,
            model=model,
            timeout_seconds=timeout_seconds,
            max_entities=max_entities,
            seed=self.seed,
        )
        return self._extract_from_request(
            window,
            ontology,
            request,
            max_entities=max_entities,
            previous_entities=None,
            stage="document_context_extraction",
            document_context=True,
        )

    def _extract_from_request(  # noqa: PLR0912,PLR0915 - each rule is a distinct rejection.
        self,
        window: ExtractionWindow,
        ontology: OntologyProfile,
        request: StructuredCompletionRequest,
        *,
        max_entities: int,
        previous_entities: list[EntityMention] | None,
        stage: str,
        document_context: bool,
    ) -> EntityExtractionOutcome:
        """Ground one strict entity response under ordinary or document context semantics."""

        completion = self.llm.complete_structured(request)
        chunks_by_id = {chunk.id: chunk for chunk in window.chunks}
        valid_types = set(ontology.entity_type_names())
        previous_keys = {
            (_normalize(item.name), item.source_chunk_id) for item in (previous_entities or [])
        }
        seen = set(previous_keys)
        # Read once per response rather than per candidate: the profile is fixed
        # for the whole request and is not hashable, so it cannot be cached.
        definitions = _ontology_definition_keys(ontology)
        subject_types = {item.name for item in ontology.entity_types if item.document_subject}
        # Once relations carry stated values, a value is never an entity of its
        # own - except under a type whose names are codes or numbers.
        identifier_types = (
            {item.name for item in ontology.entity_types if item.identifier}
            if ontology.carries_quantities()
            else None
        )
        reasons: Counter[str] = Counter()
        rejections = _RejectionLog(reasons)
        raw_entities, candidates = _validated_records(
            completion.payload,
            "entities",
            EntityCandidate,
            reasons,
            rejections,
        )
        entities: list[EntityMention] = []
        # Entities string grounding could not place, awaiting the fallback.
        pending: list[_UnlocatedEntity] = []
        for candidate in candidates[:max_entities]:
            before, held = rejections.watch(), len(entities) + len(pending)
            try:
                chunk = chunks_by_id.get(candidate.source_chunk_id)
                if chunk is None:
                    reasons["invalid_chunk"] += 1
                    continue
                if candidate.type not in valid_types:
                    reasons["invalid_entity_type"] += 1
                    continue
                name, abbreviations = _split_abbreviation(candidate.name.strip(), reasons)
                if not name:
                    reasons["blank_name"] += 1
                    continue
                if location_page_name(name, chunk):
                    # The file is the document itself; its name and folders
                    # say where it sits, not what it states.
                    reasons["location_page_name"] += 1
                    continue
                if ontology.identifier_pattern(name) is not None:
                    # A registry number or code identifies an entity; it is kept
                    # as that entity's alias, never as a node of its own.
                    reasons["identifier_as_entity"] += 1
                    continue
                if identifier_types is not None and _value_named(
                    name, candidate.type, identifier_types
                ):
                    reasons["value_as_entity"] += 1
                    continue
                if not document_context and _is_contextual_surface_name(
                    name,
                    candidate.type,
                    ontology,
                ):
                    # Contextual surfaces are discourse pointers, not identities. The
                    # matching document-context mention remains available to relation
                    # extraction, where the same surface can ground a real endpoint.
                    reasons["contextual_surface_as_entity"] += 1
                    continue
                # A name names one thing within a chunk: a second type for it is the
                # model hedging, and the first answer stands.
                key = (_normalize(name), chunk.id)
                if key in seen:
                    reasons["duplicate"] += 1
                    continue
                alias_validation = validate_identity_aliases(
                    name,
                    [*candidate.aliases, *abbreviations],
                    chunk.content,
                    is_identifier=lambda alias: ontology.identifier_pattern(alias) is not None,
                )
                if alias_validation.rejected_count:
                    reasons["untrusted_alias"] += alias_validation.rejected_count
                name_method: Literal["llm", "ocr_tolerant"] | None = "llm"
                if document_context:
                    name_method = _document_context_surface_match(
                        name,
                        candidate.quote,
                        alias_validation.aliases,
                    )
                    if name_method is None:
                        reasons["document_context_name_mismatch"] += 1
                        continue
                surface_groups = (
                    [[candidate.quote]] if document_context else [[name, *alias_validation.aliases]]
                )
                grounded = ground_evidence(chunk.content, candidate.quote, surface_groups)
                if grounded is None and not document_context:
                    # A name the model put in its own words ("pH measurement"
                    # for "the pH was measured") is kept when each of its words
                    # is in the verbatim quote; on a benchmark corpus 88% were true.
                    grounded = _rephrased_name_span(chunk.content, candidate.quote, name)
                if grounded is None:
                    # Last, the text as OCR may have printed it: words run
                    # together or a letter misread.
                    grounded = ground_evidence_ocr_tolerant(
                        chunk.content, candidate.quote, surface_groups
                    )
                if grounded is None and self.grounding_fallback and not document_context:
                    description = _entity_description(candidate.description, definitions)
                    pending.append(
                        _UnlocatedEntity(candidate, name, alias_validation.aliases, description)
                    )
                    continue
                if grounded is None:
                    reasons["ungrounded_quote"] += 1
                    continue
                seen.add(key)
                if grounded.repair:
                    reasons[f"repaired_{grounded.repair}"] += 1
                evidence_method = _string_evidence_method(
                    grounded.ocr_tolerant or name_method == "ocr_tolerant", reasons
                )
                description = _entity_description(candidate.description, definitions)
                if not description and candidate.description.strip():
                    reasons["type_definition_as_description"] += 1
                entities.append(
                    EntityMention(
                        id=stable_id(
                            "entity_mention",
                            window.id,
                            chunk.id,
                            candidate.type,
                            _normalize(name),
                            grounded.start_offset,
                        ),
                        name=name,
                        type=candidate.type,
                        description=description,
                        source_chunk_id=chunk.id,
                        quote=grounded.quote,
                        confidence=candidate.confidence,
                        aliases=_dedupe_aliases(alias_validation.aliases, name),
                        is_document_context=document_context,
                        is_document_subject=document_context and candidate.type in subject_types,
                        contextual_surfaces=(
                            next(
                                (
                                    list(definition.contextual_surfaces)
                                    for definition in ontology.entity_types
                                    if definition.name == candidate.type
                                ),
                                [],
                            )
                            if document_context
                            else []
                        ),
                        start_offset=grounded.start_offset,
                        end_offset=grounded.end_offset,
                        evidence_method=evidence_method,
                    )
                )
            finally:
                if len(entities) + len(pending) == held:
                    rejections.reject(before, _entity_record(candidate))
        if pending:
            located, unfound = self._locate_entities(
                window,
                pending,
                model=request.model,
                timeout_seconds=request.timeout_seconds,
                reasons=reasons,
            )
            for mention in located:
                key = (_normalize(mention.name), mention.source_chunk_id)
                if key not in seen:
                    seen.add(key)
                    entities.append(mention)
            for item, reason in unfound:
                reasons[reason] += 1
                rejections.add(reason, _entity_record(item.candidate))
        return EntityExtractionOutcome(
            entities=entities,
            trace={
                "stage": stage,
                "window_id": window.id,
                **_window_scope(window),
                "input_records": len(raw_entities),
                "accepted_records": len(entities),
                "record_actions": dict(sorted(reasons.items())),
                "rejected_records": rejections.records,
                **completion.provider_metadata,
            },
        )


def _split_abbreviation(name: str, reasons: Counter[str]) -> tuple[str, list[str]]:
    """Split a trailing abbreviation off a name: 'Sodium lauryl sulphate (SLS)'.

    The abbreviation becomes an alias, so the name matches its other mentions
    with or without it. Only letters with at least two capitals count: a mark
    such as (TM) or a grade such as (PEG 400) stays part of the name.
    """

    match = _TRAILING_ABBREVIATION_RE.fullmatch(name)
    if match is None:
        return name, []
    head, abbreviation = match.group(1).strip(), match.group(2).strip()
    capitals = sum(character.isupper() for character in abbreviation)
    if (
        capitals < _MIN_ABBREVIATION_CAPITALS
        or abbreviation.upper() in _TRADEMARK_MARKS
        or len(head) <= len(abbreviation)
    ):
        return name, []
    reasons["abbreviation_as_alias"] += 1
    return head, [abbreviation]


def _value_named(name: str, entity_type: str, identifier_types: set[str]) -> bool:
    """Whether a candidate entity is only a stated value, under a non-identifier type."""

    return entity_type not in identifier_types and is_value_text(name)


def _is_contextual_surface_name(
    name: str,
    entity_type: str,
    ontology: OntologyProfile,
) -> bool:
    """Identify a discourse placeholder emitted as an ordinary entity name.

    Document-context surfaces intentionally let relations in independent windows
    refer back to a focal paper, method, or model through phrases such as ``we``
    and ``our method``. Accepting the same phrases as ordinary entity candidates
    creates competing canonical nodes and diverts those relations away from the
    grounded focal entity. Matching is exact after whitespace and case
    normalization so legitimate names that merely contain a common word remain
    unaffected.
    """

    definition = next(
        (item for item in ontology.entity_types if item.name == entity_type),
        None,
    )
    if definition is None or not definition.document_context:
        return False
    normalized_name = _normalize(name)
    return normalized_name in {_normalize(surface) for surface in definition.contextual_surfaces}


@dataclass(frozen=True)
class _UnlocatedEntity:
    """An entity candidate string grounding could not place, kept for the fallback."""

    candidate: EntityCandidate
    name: str
    aliases: list[str]
    description: str


def _locate_entities(
    llm: StructuredCompletionProvider,
    seed: int,
    window: ExtractionWindow,
    pending: list[_UnlocatedEntity],
    *,
    model: str,
    timeout_seconds: int,
    reasons: Counter[str],
) -> tuple[list[EntityMention], list[tuple[_UnlocatedEntity, str]]]:
    """Ask the model which lines name each entity string grounding missed.

    The window's chunks are shown as numbered units, so an entity cited to the
    wrong chunk can still be found. The chosen units must lie in one chunk and
    share a word stem with the name; the evidence is the exact stretch they
    span - llm_located when it contains the name, llm_confirmed when it states
    the thing in other words ("stable for 4 weeks" for stability).
    """

    units = _numbered_units(window.chunks)
    by_number = {number: (chunk, span) for number, chunk, span, _ in units}
    located: list[EntityMention] = []
    unfound: list[tuple[_UnlocatedEntity, str]] = []
    if not units:
        return located, [(item, "ungrounded_quote") for item in pending]
    for batch in batched(pending, _GROUNDING_BATCH_SIZE):
        request = entity_grounding_request(
            [(number, shown) for number, _, _, shown in units],
            [
                (index, item.name, item.candidate.type, item.candidate.quote)
                for index, item in enumerate(batch)
            ],
            model=model,
            timeout_seconds=timeout_seconds,
            seed=seed,
        )
        decisions = {
            str(decision.get("id")): decision
            for decision in llm.complete_structured(request).payload.get("decisions", [])
            if isinstance(decision, dict)
        }
        for index, item in enumerate(batch):
            decision = decisions.get(f"e{index}", {})
            chosen = [
                by_number[number]
                for number in decision.get("lines", [])
                if isinstance(number, int) and number in by_number
            ]
            if not decision.get("supported") or not chosen:
                unfound.append((item, "llm_grounding_unsupported"))
                continue
            chunk = chosen[0][0]
            spans = [span for owner, span in chosen if owner.id == chunk.id]
            start = min(span.start_offset for span in spans)
            end = max(span.end_offset for span in spans)
            stretch = chunk.content[start:end]
            if surface_occurs(stretch, [item.name, *item.aliases]):
                method: Literal["llm_located", "llm_confirmed"] = "llm_located"
            elif name_shares_stem(item.name, stretch):
                method = "llm_confirmed"
            else:
                unfound.append((item, "llm_grounding_unconfirmed"))
                continue
            reasons[f"entity_{method}"] += 1
            located.append(
                EntityMention(
                    id=stable_id(
                        "entity_mention",
                        window.id,
                        chunk.id,
                        item.candidate.type,
                        _normalize(item.name),
                        start,
                    ),
                    name=item.name,
                    type=item.candidate.type,
                    description=item.description,
                    source_chunk_id=chunk.id,
                    quote=stretch,
                    confidence=item.candidate.confidence,
                    aliases=_dedupe_aliases(item.aliases, item.name),
                    start_offset=start,
                    end_offset=end,
                    evidence_method=method,
                )
            )
    return located, unfound


class LlmRelationExtractor:
    """Extract grounded relations constrained to previously accepted entity IDs.

    The adapter validates ontology domain/range rules, repairs unambiguous
    direction mistakes, requires relation cues and endpoint evidence, and rejects
    self-loops unless the ontology explicitly permits them. With type holding on,
    a statement rejected only for its entity types is still grounded and held:
    one window sees one reading of each entity's type, and finalization, which
    sees them all, decides whether another reading admits it.
    """

    def __init__(
        self,
        llm: StructuredCompletionProvider,
        seed: int = 17,
        *,
        grounding_fallback: bool = False,
        new_endpoints: bool = False,
        relabel: bool = False,
        hold_type_violations: bool = False,
    ) -> None:
        """Store the structured provider and deterministic seed used by relation requests.

        ``grounding_fallback`` lets one extra call per window locate the lines
        that state a relation string grounding could not place. ``new_endpoints``
        lets a response add an endpoint the window's entities lack.
        ``hold_type_violations`` carries statements rejected only for their
        entity types out as held relations. No provider work occurs during
        construction.
        """

        self.llm = llm
        self.seed = seed
        self.grounding_fallback = grounding_fallback
        self.new_endpoints = new_endpoints
        self.relabel = relabel
        self.hold_type_violations = hold_type_violations

    def extract(  # noqa: PLR0912, PLR0915 - branches record distinct rejection reasons.
        self,
        window: ExtractionWindow,
        entities: list[EntityMention],
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        max_relations: int,
        previous_relations: list[RelationObservation] | None = None,
        seed: int | None = None,
    ) -> RelationExtractionOutcome:
        """Ground, type-check, orient, and deduplicate relation candidates.

        Fewer than two accepted entities short-circuits the provider call. For
        larger windows, each malformed or unsupported record is rejected without
        invalidating valid sibling relations from the same response.
        """

        if len(entities) < _MIN_RELATION_ENTITIES:
            return RelationExtractionOutcome(
                trace={
                    "stage": "relation_extraction",
                    "window_id": window.id,
                    **_window_scope(window),
                    "input_records": 0,
                    "accepted_records": 0,
                    "skipped": "fewer_than_two_entities",
                }
            )
        request = relation_extraction_request(
            window,
            entities,
            ontology,
            model=model,
            timeout_seconds=timeout_seconds,
            max_relations=max_relations,
            seed=self.seed if seed is None else seed,
            previous_relations=previous_relations,
            new_endpoints=self.new_endpoints,
        )
        completion = self.llm.complete_structured(request)
        entities_by_id = {entity.id: entity for entity in entities}
        chunks_by_id = {chunk.id: chunk for chunk in window.chunks}
        previous_keys = {
            (
                item.source_entity_id,
                item.target_entity_id,
                item.relation_type,
                item.source_chunk_id,
            )
            for item in (previous_relations or [])
        }
        seen = set(previous_keys)
        reasons: Counter[str] = Counter()
        rejections = _RejectionLog(reasons)
        raw_relations, candidates = _validated_records(
            completion.payload,
            "relations",
            RelationCandidate,
            reasons,
            rejections,
        )
        endpoints = (
            _validated_endpoints(
                completion.payload, window, entities, ontology, reasons, rejections
            )
            if self.new_endpoints
            else {}
        )
        entities_by_id.update(endpoints)
        relations: list[RelationObservation] = []
        unlocated: list[_UnlocatedRelation] = []
        # Statements whose type rules their entities break, awaiting a relabel.
        mislabeled: list[tuple[RelationCandidate, EntityMention, EntityMention]] = []
        # Grounded statements rejected only for their entity types.
        held_relations: list[RelationObservation] = []
        held_seen: set[tuple[str, str, str, str]] = set()

        def consider(  # noqa: PLR0911, PLR0912, PLR0915 - each rule is a distinct rejection.
            candidate: RelationCandidate,
            stated: str | None = None,
            *,
            relabel: bool = self.relabel,
        ) -> None:
            """Validate one candidate; ``stated`` is the type a relabel replaced.

            A candidate whose entity types its relation rejects is, with type
            holding on, grounded under its stated type like any other and held.
            It is counted and recorded as a type violation either way; the checks
            after the type check count nothing more for it, so a held candidate
            that cannot be grounded stays exactly that violation.
            """

            before = rejections.watch()
            placed = len(relations) + len(unlocated) + len(mislabeled) + len(held_relations)
            try:
                source = entities_by_id.get(candidate.source_entity_id)
                target = entities_by_id.get(candidate.target_entity_id)
                chunk = chunks_by_id.get(candidate.source_chunk_id)
                if source is None or target is None:
                    reasons["invalid_endpoint"] += 1
                    return
                definition, restated = _relation_definition(
                    candidate.relation_type, ontology, reasons
                )
                if definition is None:
                    return
                if source.id == target.id and not definition.allow_self_loop:
                    reasons["self_loop"] += 1
                    return
                source, target, direction_repaired = _orient_relation(source, target, definition)
                relabel_next = relabel and stated is None
                admitted, proposed_relation_type = admitted_definition(
                    definition, source, target, ontology, reasons, count_violation=not relabel_next
                )
                if admitted is None and relabel_next:
                    mislabeled.append((candidate, source, target))
                    return
                if admitted is None and not self.hold_type_violations:
                    return
                holding = admitted is None
                tally: Counter[str] = Counter() if holding else reasons
                definition = admitted or definition
                proposed_relation_type = stated or restated or proposed_relation_type
                if direction_repaired:
                    tally["repaired_direction"] += 1
                if chunk is None:
                    tally["invalid_chunk"] += 1
                    return
                endpoint_surfaces = _validated_endpoint_surfaces(candidate, tally)
                if endpoint_surfaces is None:
                    return
                source_surface, target_surface = endpoint_surfaces
                if direction_repaired:
                    # Endpoint surfaces belong to the candidate's original direction.
                    # Keep them aligned with the repaired entity IDs so evidence never
                    # attributes one local mention to the opposite canonical endpoint.
                    source_surface, target_surface = target_surface, source_surface
                key = (source.id, target.id, definition.name, chunk.id)
                if key in (held_seen if holding else seen):
                    tally["duplicate"] += 1
                    return
                required_surface_groups, implicit_document_source = (
                    _relation_required_surface_groups(
                        chunk.content,
                        source,
                        source_surface,
                        target,
                        target_surface,
                        allow_nested=True,
                    )
                )
                grounded = ground_evidence(
                    chunk.content,
                    candidate.quote,
                    required_surface_groups,
                    allow_nested=True,
                ) or ground_evidence_ocr_tolerant(
                    chunk.content,
                    candidate.quote,
                    required_surface_groups,
                    allow_nested=True,
                )
                if grounded is None:
                    tally["ungrounded_quote"] += 1
                    if self.grounding_fallback and not holding:
                        unlocated.append(
                            _UnlocatedRelation(
                                candidate,
                                definition,
                                chunk,
                                source,
                                target,
                                source_surface,
                                target_surface,
                                required_surface_groups,
                                implicit_document_source,
                                key,
                                proposed_relation_type,
                            )
                        )
                    return
                observation = _grounded_observation(
                    window,
                    candidate,
                    definition,
                    chunk,
                    source,
                    target,
                    source_surface,
                    target_surface,
                    grounded,
                    implicit_document_source=implicit_document_source,
                    reasons=tally,
                    proposed_relation_type=proposed_relation_type,
                )
                if observation is None:
                    return
                if holding:
                    record = _relation_record(candidate, entities_by_id)
                    rejections.add("domain_or_range_violation", record)
                    reasons["held_for_type_resolution"] += 1
                    held_seen.add(key)
                    held_relations.append(
                        observation.model_copy(
                            update={
                                "rejected_record_key": rejected_record_key(
                                    window.id, "relation", "domain_or_range_violation", record
                                )
                            }
                        )
                    )
                    return
                seen.add(key)
                relations.append(observation)
            finally:
                if (
                    len(relations) + len(unlocated) + len(mislabeled) + len(held_relations)
                    == placed
                ):
                    rejections.reject(before, _relation_record(candidate, entities_by_id))

        for candidate in candidates[:max_relations]:
            consider(candidate)
        if mislabeled:
            for relabeled, original in self._relabel(
                mislabeled, ontology, model=model, timeout_seconds=timeout_seconds, reasons=reasons
            ):
                if relabeled is None and self.hold_type_violations:
                    consider(original, relabel=False)
                elif relabeled is None:
                    reasons["domain_or_range_violation"] += 1
                    rejections.add(
                        "domain_or_range_violation", _relation_record(original, entities_by_id)
                    )
                else:
                    consider(relabeled, stated=original.relation_type)
        if unlocated:
            located, unfound = self._locate_ungrounded(
                window, unlocated, model=model, timeout_seconds=timeout_seconds, reasons=reasons
            )
            for observation, key in located:
                if key not in seen:
                    seen.add(key)
                    relations.append(observation)
            for item, reason in unfound:
                rejections.add(reason, _relation_record(item.candidate, entities_by_id))
        listed = {entity.id for entity in entities}
        added = {mention.id: mention for mention in endpoints.values() if mention.id not in listed}
        return RelationExtractionOutcome(
            relations=relations,
            held_relations=held_relations,
            entities=list(added.values()),
            trace={
                "stage": "relation_extraction",
                "window_id": window.id,
                **_window_scope(window),
                "input_records": len(raw_relations),
                "accepted_records": len(relations),
                "record_actions": dict(sorted(reasons.items())),
                "rejected_records": rejections.records,
                **completion.provider_metadata,
            },
        )

    def _relabel(
        self,
        mislabeled: list[tuple[RelationCandidate, EntityMention, EntityMention]],
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        reasons: Counter[str],
    ) -> list[tuple[RelationCandidate | None, RelationCandidate]]:
        """Offer each statement the relations its two entities' types allow.

        A statement whose type rules its entities break is often a true fact
        under another relation: "foam TESTED_BY density" is foam
        HAS_QUALITY_ATTRIBUTE density. The model may only choose among the
        relations the ontology allows between the two types, in either
        direction, or none; the relabelled statement then passes every check
        again. None marks a statement no allowed relation states.
        """

        choices: list[tuple[RelationCandidate, list[tuple[str, bool]]]] = []
        for candidate, source, target in mislabeled:
            options = [
                (item.name, reversed_)
                for reversed_, (first, second) in (
                    (False, (source, target)),
                    (True, (target, source)),
                )
                for item in ontology.relation_types
                if item.admits(first.type, second.type)
                and (item.allow_self_loop or first.id != second.id)
            ]
            choices.append((candidate, options))
        decided: list[tuple[RelationCandidate | None, RelationCandidate]] = []
        answerable = [(candidate, options) for candidate, options in choices if options]
        decided.extend((None, candidate) for candidate, options in choices if not options)
        for batch in batched(answerable, _GROUNDING_BATCH_SIZE):
            request = relation_relabel_request(
                [
                    {
                        "id": f"r{index}",
                        "source": candidate.source_surface,
                        "target": candidate.target_surface,
                        "stated_relation": candidate.relation_type,
                        "statement": candidate.description,
                        "quote": candidate.quote[:_REJECTED_QUOTE_CHARACTERS],
                        "options": [
                            {
                                "n": number,
                                "relation_type": name,
                                "direction": "target to source"
                                if reversed_
                                else "source to target",
                                "definition": ontology_relation_description(ontology, name),
                            }
                            for number, (name, reversed_) in enumerate(options)
                        ],
                    }
                    for index, (candidate, options) in enumerate(batch)
                ],
                model=model,
                timeout_seconds=timeout_seconds,
                seed=self.seed,
            )
            decisions = {
                str(decision.get("id")): decision
                for decision in self.llm.complete_structured(request).payload.get("decisions", [])
                if isinstance(decision, dict)
            }
            for index, (candidate, options) in enumerate(batch):
                choice = decisions.get(f"r{index}", {}).get("option")
                if not isinstance(choice, int) or not 0 <= choice < len(options):
                    decided.append((None, candidate))
                    continue
                name, reversed_ = options[choice]
                reasons["relabeled_relation"] += 1
                update: dict[str, object] = {"relation_type": name}
                if reversed_:
                    update |= {
                        "source_entity_id": candidate.target_entity_id,
                        "target_entity_id": candidate.source_entity_id,
                        "source_surface": candidate.target_surface,
                        "target_surface": candidate.source_surface,
                    }
                decided.append((candidate.model_copy(update=update), candidate))
        return decided

    def _confirmed_span(
        self,
        item: _UnlocatedRelation,
        chosen: list[GroundedSpan],
    ) -> tuple[int, int, bool, bool] | None:
        """The stretch the chosen lines ground, whether it had to extend, and how.

        The chosen lines must name both endpoints; otherwise they may extend,
        within the chunk, by a line for each endpoint they lack - the shortest
        such stretch, bounded like a quote in pieces - so a topic and a detail
        several sentences apart can be tied. Only when no stretch names them as
        written are the names read OCR-tolerantly, which the last flag reports.
        """

        for ocr_tolerant in (False, True):
            stretch = self._stretch_naming_endpoints(item, chosen, ocr_tolerant=ocr_tolerant)
            if stretch is not None:
                return *stretch, ocr_tolerant
        return None

    def _stretch_naming_endpoints(
        self,
        item: _UnlocatedRelation,
        chosen: list[GroundedSpan],
        *,
        ocr_tolerant: bool,
    ) -> tuple[int, int, bool] | None:
        """The chosen lines, or their shortest extension, that name both endpoints."""

        start = min(span.start_offset for span in chosen)
        end = max(span.end_offset for span in chosen)
        groups = item.required_surface_groups
        if contains_surface_groups(
            "\n".join(span.quote for span in chosen),
            groups,
            allow_nested=True,
            ocr_tolerant=ocr_tolerant,
        ):
            return start, end, False
        content = item.chunk.content
        if ocr_tolerant and not contains_surface_groups(
            content, groups, allow_nested=True, ocr_tolerant=True
        ):
            # No stretch of the chunk names what the whole chunk does not.
            return None
        units = sentence_spans(content)
        # One line more for each endpoint the chosen lines lack: at most two.
        stretches = sorted(
            {
                (
                    min(start, first.start_offset, second.start_offset),
                    max(end, first.end_offset, second.end_offset),
                )
                for index, first in enumerate(units)
                for second in units[index:]
            },
            key=lambda stretch: (stretch[1] - stretch[0], stretch[0]),
        )
        return next(
            (
                (first, last, True)
                for first, last in stretches
                if last - first <= MAX_PIECED_QUOTE_CHARACTERS
                and contains_surface_groups(
                    content[first:last], groups, allow_nested=True, ocr_tolerant=ocr_tolerant
                )
            ),
            None,
        )

    def _locate_ungrounded(
        self,
        window: ExtractionWindow,
        unlocated: list[_UnlocatedRelation],
        *,
        model: str,
        timeout_seconds: int,
        reasons: Counter[str],
    ) -> tuple[
        list[tuple[RelationObservation, tuple[str, str, str, str]]],
        list[tuple[_UnlocatedRelation, str]],
    ]:
        """Ask the model which lines state each relation string grounding missed.

        Every cited chunk is shown as numbered units. The model may only point
        at units; code then requires both endpoints inside the chosen units of
        the cited chunk, and the evidence is the exact source stretch from the
        first chosen unit to the last. Chosen units that do not name both ends
        become llm_verified evidence, kept only if the verifier supports it.
        Model-written text never becomes evidence.
        """

        chunks = list({item.chunk.id: item.chunk for item in unlocated}.values())
        units = _numbered_units(chunks)
        if not units:
            return [], [(item, "ungrounded_quote") for item in unlocated]
        by_number = {number: (chunk, span) for number, chunk, span, _ in units}
        located: list[tuple[RelationObservation, tuple[str, str, str, str]]] = []
        unfound: list[tuple[_UnlocatedRelation, str]] = []
        for batch in batched(unlocated, _GROUNDING_BATCH_SIZE):
            request = relation_grounding_request(
                [(number, shown) for number, _, _, shown in units],
                [
                    (index, item.source.name, item.definition.name, item.target.name)
                    for index, item in enumerate(batch)
                ],
                model=model,
                timeout_seconds=timeout_seconds,
                seed=self.seed,
            )
            decisions = {
                str(decision.get("id")): decision
                for decision in self.llm.complete_structured(request).payload.get("decisions", [])
                if isinstance(decision, dict)
            }
            for index, item in enumerate(batch):
                decision = decisions.get(f"r{index}", {})
                if not decision.get("supported"):
                    reasons["llm_grounding_unsupported"] += 1
                    unfound.append((item, "llm_grounding_unsupported"))
                    continue
                chosen = [
                    by_number[number][1]
                    for number in decision.get("lines", [])
                    if isinstance(number, int)
                    and number in by_number
                    and by_number[number][0].id == item.chunk.id
                ]
                located_span = self._confirmed_span(item, chosen) if chosen else None
                method: Literal["llm_located", "llm_verified"] = "llm_located"
                if located_span is None and chosen:
                    # Lines that do not name both ends may still state the
                    # relation: a row about the subject a header names once.
                    # The verifier, shown the document's opening, decides.
                    stretch = _chosen_stretch(chosen)
                    located_span = (*stretch, False) if stretch is not None else None
                    method = "llm_verified"
                if located_span is None:
                    reasons["llm_grounding_unconfirmed"] += 1
                    unfound.append((item, "llm_grounding_unconfirmed"))
                    continue
                start, end, extended, ocr_tolerant = located_span
                grounded = GroundedSpan(
                    item.chunk.content[start:end], start, end, ocr_tolerant=ocr_tolerant
                )
                unnamed_in_row = reasons["row_endpoint_unnamed"]
                observation = _grounded_observation(
                    window,
                    item.candidate,
                    item.definition,
                    item.chunk,
                    item.source,
                    item.target,
                    item.source_surface,
                    item.target_surface,
                    grounded,
                    implicit_document_source=item.implicit_document_source,
                    reasons=reasons,
                    evidence_method=method,
                    proposed_relation_type=item.proposed_relation_type,
                )
                if observation is None:
                    unfound.append(
                        (
                            item,
                            "row_endpoint_unnamed"
                            if reasons["row_endpoint_unnamed"] > unnamed_in_row
                            else "ungrounded_quote",
                        )
                    )
                    continue
                if extended:
                    reasons["llm_grounding_extended"] += 1
                if method == "llm_verified":
                    reasons["llm_grounding_to_verifier"] += 1
                reasons["llm_grounding_located"] += 1
                located.append((observation, item.key))
        return located, unfound


def _chosen_stretch(chosen: list[GroundedSpan]) -> tuple[int, int, bool] | None:
    """The source stretch the chosen lines span, when it is short enough to be a quote."""

    start = min(span.start_offset for span in chosen)
    end = max(span.end_offset for span in chosen)
    if end - start > MAX_PIECED_QUOTE_CHARACTERS:
        return None
    return start, end, False


@dataclass(frozen=True)
class _UnlocatedRelation:
    """A relation candidate string grounding could not place, kept for the fallback."""

    candidate: RelationCandidate
    definition: RelationTypeDefinition
    chunk: Chunk
    source: EntityMention
    target: EntityMention
    source_surface: str
    target_surface: str
    required_surface_groups: list[list[str]]
    implicit_document_source: bool
    key: tuple[str, str, str, str]
    proposed_relation_type: str | None = None


def _grounded_observation(  # noqa: PLR0913 - one grounded candidate's full context.
    window: ExtractionWindow,
    candidate: RelationCandidate,
    definition: RelationTypeDefinition,
    chunk: Chunk,
    source: EntityMention,
    target: EntityMention,
    source_surface: str,
    target_surface: str,
    grounded: GroundedSpan,
    *,
    implicit_document_source: bool,
    reasons: Counter[str],
    evidence_method: Literal["llm", "llm_located", "llm_verified"] = "llm",
    proposed_relation_type: str | None = None,
) -> RelationObservation | None:
    """Turn a grounded candidate into an observation.

    None when an endpoint cannot be named inside the evidence, except for
    evidence the verifier must confirm (llm_verified), whose unnamed ends are
    identified by name. Evidence within one delimited row names both ends in
    that row, whatever its method, unless the unnamed one is the document's
    subject.
    String-grounded and LLM-located evidence pass the same surface, row, and cue
    checks, so the two paths cannot drift apart. Evidence found through the
    OCR-tolerant view is labelled ``ocr_tolerant`` whichever path found it.
    """

    if implicit_document_source:
        reasons["implicit_document_source"] += 1
    repaired_source_surface = _grounded_endpoint_surface(
        grounded.quote, source, source_surface, ocr_tolerant=grounded.ocr_tolerant
    )
    repaired_target_surface = _grounded_endpoint_surface(
        grounded.quote, target, target_surface, ocr_tolerant=grounded.ocr_tolerant
    )
    if _row_leaves_endpoint_unnamed(chunk, grounded, (source, target)):
        reasons["row_endpoint_unnamed"] += 1
        return None
    if repaired_source_surface is None and (
        implicit_document_source or evidence_method == "llm_verified"
    ):
        # A provenance-only source has no local mention by definition.
        # Persisting its canonical name identifies the endpoint clearly
        # for semantic verification and downstream evidence inspection.
        repaired_source_surface = source.name
    if repaired_target_surface is None and evidence_method == "llm_verified":
        repaired_target_surface = target.name
    if repaired_source_surface is None or repaired_target_surface is None:
        reasons["unresolved_grounded_surface"] += 1
        return None
    if repaired_source_surface != source_surface:
        reasons["repaired_source_surface"] += 1
    if repaired_target_surface != target_surface:
        reasons["repaired_target_surface"] += 1
    if not relation_evidence_supported(
        grounded.quote,
        [repaired_source_surface],
        [repaired_target_surface],
        definition,
    ):
        # Ontology cues are useful diagnostics, but natural-language
        # predicates have unbounded surface forms. Exact endpoint and
        # evidence grounding plus the independent semantic verifier are
        # the authoritative entailment checks; an incomplete regex list
        # must not silently erase an otherwise valid candidate.
        reasons["cue_not_found"] += 1
    if grounded.repair:
        reasons[f"repaired_{grounded.repair}"] += 1
    method: Literal["llm", "llm_located", "llm_verified", "ocr_tolerant"] = evidence_method
    if grounded.ocr_tolerant:
        reasons["ocr_tolerant_grounding"] += 1
        # Evidence the verifier must confirm keeps saying so; its label is
        # what sends it to the verifier.
        if method != "llm_verified":
            method = "ocr_tolerant"
    quantities = _grounded_quantities(candidate, definition, grounded.quote, reasons)
    return RelationObservation(
        id=stable_id(
            "relation_observation",
            window.id,
            source.id,
            target.id,
            definition.name,
            chunk.id,
            grounded.start_offset,
        ),
        source_entity_id=source.id,
        target_entity_id=target.id,
        source_surface=repaired_source_surface,
        target_surface=repaired_target_surface,
        relation_type=definition.name,
        description=candidate.description.strip(),
        source_chunk_id=chunk.id,
        quote=grounded.quote,
        confidence=candidate.confidence,
        evidence_method=method,
        start_offset=grounded.start_offset,
        end_offset=grounded.end_offset,
        quantities=quantities,
        proposed_relation_type=proposed_relation_type,
        assertion=candidate.assertion,
    )


def _row_leaves_endpoint_unnamed(
    chunk: Chunk,
    grounded: GroundedSpan,
    ends: tuple[EntityMention, EntityMention],
) -> bool:
    """Whether the evidence lies in one delimited row that does not name an endpoint.

    A row states the facts of the things it names, and the rows beside it hold
    near-identical things - another salt of a substance, another route or
    form. A local surface the model gave can match a shorter or neighbouring
    cell, so only an endpoint's own name or alias names it in the row. The
    document's subject alone may go unnamed: a certificate names it once, in
    its header, and every row below is about it.
    """

    row = delimited_row_span(chunk.content, grounded.start_offset, grounded.end_offset)
    if row is None:
        return False
    text = chunk.content[row[0] : row[1]]
    return any(
        not entity.is_document_subject and not row_names(text, [entity.name, *entity.aliases])
        for entity in ends
    )


def _validated_endpoints(  # noqa: PLR0913 - one response's endpoint context.
    payload: dict[str, Any],
    window: ExtractionWindow,
    entities: list[EntityMention],
    ontology: OntologyProfile,
    reasons: Counter[str],
    rejections: _RejectionLog,
) -> dict[str, EntityMention]:
    """The endpoints a relation response added, by the id its relations use.

    Each is checked as an entity would be: an ontology type, a name that is not
    only a value, and a quote in the cited chunk that names it. One already in
    the window's entities is that entity. A valid endpoint is kept as an entity
    even when no relation using it survives: it is a grounded, named thing.
    """

    if payload.get("new_entities") is None:
        return {}
    _, candidates = _validated_records(
        payload, "new_entities", EndpointCandidate, reasons, rejections
    )
    chunks_by_id = {chunk.id: chunk for chunk in window.chunks}
    valid_types = set(ontology.entity_type_names())
    identifier_types = (
        {item.name for item in ontology.entity_types if item.identifier}
        if ontology.carries_quantities()
        else None
    )
    listed = {(_normalize(entity.name), entity.type): entity for entity in entities}
    endpoints: dict[str, EntityMention] = {}
    for candidate in candidates:
        name = candidate.name.strip()
        chunk = chunks_by_id.get(candidate.source_chunk_id)
        record: dict[str, object] = {
            "kind": "entity",
            "name": name,
            "type": candidate.type,
            "quote": candidate.quote[:_REJECTED_QUOTE_CHARACTERS],
            "chunk_id": candidate.source_chunk_id,
        }
        reason = None
        grounded = None
        if chunk is None:
            reason = "invalid_chunk"
        elif candidate.type not in valid_types:
            reason = "invalid_entity_type"
        elif not name:
            reason = "blank_name"
        elif location_page_name(name, chunk):
            reason = "location_page_name"
        elif ontology.identifier_pattern(name) is not None:
            reason = "identifier_as_entity"
        elif identifier_types is not None and _value_named(name, candidate.type, identifier_types):
            reason = "value_as_entity"
        elif (known := listed.get((_normalize(name), candidate.type))) is not None:
            reasons["endpoint_already_listed"] += 1
            endpoints[candidate.id] = known
            continue
        else:
            grounded = ground_evidence(
                chunk.content, candidate.quote, [[name]]
            ) or ground_evidence_ocr_tolerant(chunk.content, candidate.quote, [[name]])
            if grounded is None:
                reason = "ungrounded_quote"
        if reason is not None or grounded is None or chunk is None:
            reasons[reason or "rejected"] += 1
            rejections.add(reason or "rejected", record)
            continue
        reasons["added_endpoint"] += 1
        evidence_method = _string_evidence_method(grounded.ocr_tolerant, reasons)
        endpoints[candidate.id] = EntityMention(
            id=stable_id(
                "entity_mention",
                window.id,
                chunk.id,
                candidate.type,
                _normalize(name),
                grounded.start_offset,
            ),
            name=name,
            type=candidate.type,
            description="",
            source_chunk_id=chunk.id,
            quote=grounded.quote,
            start_offset=grounded.start_offset,
            end_offset=grounded.end_offset,
            evidence_method=evidence_method,
        )
    return endpoints


def _rephrased_name_span(content: str, quote: str, name: str) -> GroundedSpan | None:
    """The verbatim quote as an entity's evidence, when its name rephrases it."""

    span = find_quote(content, quote)
    if span is None or not name_words_in_quote(name, span.quote):
        return None
    return GroundedSpan(span.quote, span.start_offset, span.end_offset, repair="rephrased_name")


def ontology_relation_description(ontology: OntologyProfile, name: str) -> str:
    """A relation type's definition, as the ontology states it."""

    definition = ontology.relation(name)
    return definition.description if definition is not None else ""


def admitted_definition(
    definition: RelationTypeDefinition,
    source: EntityMention,
    target: EntityMention,
    ontology: OntologyProfile,
    reasons: Counter[str],
    *,
    count_violation: bool = True,
) -> tuple[RelationTypeDefinition | None, str | None]:
    """The relation type a statement is kept under, and the type it was stated as.

    A statement whose entity types its own relation does not allow is kept
    under the relation an ontology rewrite names for those types, else under
    the ontology's fallback relation when there is one and it allows them; the
    stated type then travels with the observation. Otherwise the statement is
    rejected.
    """

    if definition.admits(source.type, target.type):
        return definition, None
    rewritten = ontology.rewrite_for(definition.name, source.type, target.type)
    if rewritten is not None:
        reasons["rewritten_relation"] += 1
        return rewritten, definition.name
    fallback = ontology.fallback_definition()
    if fallback is None or not fallback.admits(source.type, target.type):
        if count_violation:
            reasons["domain_or_range_violation"] += 1
        return None, None
    reasons["kept_under_fallback_relation"] += 1
    return fallback, definition.name


def _grounded_quantities(
    candidate: RelationCandidate,
    definition: RelationTypeDefinition,
    evidence: str,
    reasons: Counter[str],
) -> list[Quantity]:
    """Keep the values a relation states.

    Only a relation type that carries quantities keeps them, and only values
    whose text the evidence contains exactly as the model gave it.
    """

    if candidate.quantities and not definition.quantities:
        reasons["quantity_not_carried"] += len(candidate.quantities)
        return []
    quantities: list[Quantity] = []
    for value in candidate.quantities:
        found = find_quote(evidence, value.text) if value.text.strip() else None
        if found is None:
            reasons["ungrounded_quantity"] += 1
            continue
        quantities.append(
            parse_quantity(
                found.quote,
                kind=value.kind.strip() or None,
                unit=value.unit.strip() or None,
                comparator=value.comparator.strip() or None,
            )
        )
    return quantities


def _numbered_units(chunks: list[Chunk]) -> list[tuple[int, Chunk, GroundedSpan, str]]:
    """Number the evidence units of the given chunks for a grounding request.

    Units are the grounding sentences - lines, table rows, sentences - with their
    exact offsets, so any units the model picks map back to source text. The
    text shown drops markup and runs of whitespace.
    """

    units: list[tuple[int, Chunk, GroundedSpan, str]] = []
    for chunk in chunks:
        for span in sentence_spans(chunk.content):
            shown = " ".join(_MARKUP_RE.sub(" ", span.quote).split())
            if shown:
                units.append((len(units) + 1, chunk, span, shown[:_MAX_UNIT_CHARACTERS]))
    return units


def _relation_definition(
    relation_type: str,
    ontology: OntologyProfile,
    reasons: Counter[str],
) -> tuple[RelationTypeDefinition | None, str | None]:
    """Resolve a declared predicate or create a safe open-vocabulary definition.

    Closed profiles reject unknown labels. Open and hybrid profiles retain an
    otherwise grounded predicate with no domain/range restrictions, allowing the
    declared vocabulary to guide extraction without silently erasing useful typing.

    A structural relation's label is never a stated relation's: it is outside
    the vocabulary a statement may use, and one kept is kept under the
    profile's restated relation, with the label returned as the stated type.
    """

    if normalize_ontology_label(relation_type) in ontology.reserved_relation_labels():
        restated = ontology.restated_relation()
        if restated is None:
            reasons["invalid_relation_type"] += 1
            return None, None
        reasons["reserved_relation_type"] += 1
        return restated, normalize_ontology_label(relation_type).upper()
    definition = ontology.relation(relation_type)
    if definition is not None:
        return definition, None
    if ontology.mode == "closed":
        reasons["invalid_relation_type"] += 1
        return None, None
    normalized_type = normalize_ontology_label(relation_type).upper()
    if not normalized_type:
        reasons["blank_relation_type"] += 1
        return None, None
    reasons["open_relation_type"] += 1
    return RelationTypeDefinition(
        name=normalized_type,
        description=("Source-grounded relation discovered outside the declared vocabulary."),
    ), None


def _validated_endpoint_surfaces(
    candidate: RelationCandidate,
    reasons: Counter[str],
) -> tuple[str, str] | None:
    """Validate exact local endpoint mentions supplied with a relation candidate.

    Local surfaces let a canonical inventory entry participate through a grounded
    shortened form in another sentence. Blank or identical surfaces cannot prove
    two distinct endpoints and are rejected before evidence grounding.
    """

    source_surface = candidate.source_surface.strip()
    target_surface = candidate.target_surface.strip()
    if not source_surface or not target_surface:
        reasons["blank_endpoint_surface"] += 1
        return None
    if _normalize(source_surface) == _normalize(target_surface):
        reasons["identical_endpoint_surface"] += 1
        return None
    return source_surface, target_surface


def _relation_grounding_surfaces(entity: EntityMention, proposed_surface: str) -> list[str]:
    """Return exact identity surfaces eligible to repair relation evidence.

    Structured models occasionally assign the correct inventory ID but normalize
    or abbreviate its local surface differently from noisy OCR. The proposed
    surface remains preferred. Canonical names, identity-validated aliases, and
    ontology-approved discourse surfaces provide conservative fallbacks; the
    resulting quote is still an exact source span and still passes independent
    directed-predicate verification before publication.
    """

    identity_surfaces = [entity.name, *entity.aliases]
    surfaces = [proposed_surface, *identity_surfaces]
    if entity.is_document_context:
        contextual_keys = {_normalize(surface) for surface in entity.contextual_surfaces}
        proposed_is_approved = (
            any(identity_surface_match(proposed_surface, surface) for surface in identity_surfaces)
            or _normalize(proposed_surface) in contextual_keys
        )
        surfaces = [
            *([proposed_surface] if proposed_is_approved else []),
            *identity_surfaces,
            *entity.contextual_surfaces,
        ]
    return list(dict.fromkeys(surface.strip() for surface in surfaces if surface.strip()))


def _grounded_endpoint_surface(
    quote: str,
    entity: EntityMention,
    proposed_surface: str,
    *,
    ocr_tolerant: bool = False,
) -> str | None:
    """Return the exact quote substring that grounded one endpoint identity.

    Evidence repair may rely on a canonical name, trusted alias, OCR-tolerant
    spelling, or approved document-context surface when the provider's proposed
    label is absent or normalized. Persisting the actual matched substring keeps
    semantic verification aligned with deterministic grounding. Surface priority
    follows the same ordered inventory used to ground the quote; source offsets
    break ties deterministically within one surface.
    """

    surfaces = _relation_grounding_surfaces(entity, proposed_surface)
    for surface in surfaces:
        spans = surface_spans(quote, [surface])
        if spans:
            start, end = spans[0]
            return quote[start:end]
    # Evidence the OCR-tolerant view grounded names its endpoint as OCR printed
    # it; the surface kept is that exact text of the quote.
    for surface in surfaces if ocr_tolerant else []:
        spans = ocr_tolerant_surface_spans(quote, [surface])
        if spans:
            start, end = spans[0]
            return quote[start:end]
    return None


def _relation_required_surface_groups(
    content: str,
    source: EntityMention,
    source_surface: str,
    target: EntityMention,
    target_surface: str,
    *,
    allow_nested: bool = False,
) -> tuple[list[list[str]], bool]:
    """Build exact endpoint requirements for one directed relation candidate.

    Ordinary endpoints must occur literally in the bounded evidence span. The
    document's subject is the sole exception: an entity the document-context pass
    identified under a type the ontology marks ``document_subject`` (a paper's own
    title, the material a certificate or data sheet covers) is already fixed by
    document provenance,
    so a statement about it need not repeat its name - a paper's authorial
    sentence, or a form field below the header that names the subject. The
    exception applies when no one-to-three-sentence span can ground distinct
    source and target mentions, including a target nested inside the subject's
    title. It remains source-only and never relaxes exact grounding for the
    target. Independent semantic verification must still tell the subject's own
    facts from cited, historical, or unrelated statements.

    The returned flag makes use of this provenance rule explicit in extraction
    traces. If the subject's name or a configured contextual surface does occur,
    the normal two-endpoint grounding contract is retained.
    """

    source_surfaces = _relation_grounding_surfaces(source, source_surface)
    target_surfaces = _relation_grounding_surfaces(target, target_surface)
    explicit_endpoint_span = ground_evidence(
        content,
        "",
        [source_surfaces, target_surfaces],
        allow_nested=allow_nested,
    )
    implicit_document_source = source.is_document_subject and explicit_endpoint_span is None
    if implicit_document_source:
        return [target_surfaces], True
    return [source_surfaces, target_surfaces], False


class LlmRelationVerifier:
    """Apply an independent entailment judgment to grounded relation candidates.

    Verification judges the exact directed predicate, not mere co-occurrence.
    Omitted or malformed decisions become insufficient evidence rather than
    silently approving plausible-sounding triples.
    """

    def __init__(self, llm: StructuredCompletionProvider, seed: int = 17) -> None:
        """Store the structured provider and deterministic seed used by verification requests.

        No provider work occurs during construction.
        """

        self.llm = llm
        self.seed = seed

    def verify(
        self,
        window: ExtractionWindow,
        entities: list[EntityMention],
        relations: list[RelationObservation],
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        document_opening: str = "",
        document_subjects: list[EntityMention] | None = None,
        entities_to_verify: list[EntityMention] | None = None,
    ) -> VerificationOutcome:
        """Return one reconciled decision for every candidate relation and entity.

        Empty input avoids an unnecessary provider call. Missing relation IDs are
        filled with zero-confidence insufficient decisions so downstream filtering
        remains conservative and total. A missing entity verdict is left out: the
        mention is already grounded, and verification only removes what it judges.
        """

        judged = entities_to_verify or []
        if not relations and not judged:
            return VerificationOutcome(
                trace={
                    "stage": "relation_verification",
                    "window_id": window.id,
                    **_window_scope(window),
                    "input_records": 0,
                    "accepted_records": 0,
                }
            )
        request = relation_verification_request(
            window,
            entities,
            relations,
            ontology,
            model=model,
            timeout_seconds=timeout_seconds,
            seed=self.seed,
            document_opening=document_opening,
            document_subjects=document_subjects,
            entities_to_verify=judged,
        )
        completion = self.llm.complete_structured(request)
        reasons: Counter[str] = Counter()
        # A list with nothing to judge is left out of the request's schema.
        payload = {"decisions": [], "entity_decisions": [], **completion.payload}
        _raw_decisions, candidates = _validated_records(
            payload,
            "decisions",
            VerificationCandidate,
            reasons,
        )
        _raw_entity_decisions, entity_candidates = _validated_records(
            payload,
            "entity_decisions",
            EntityVerificationCandidate,
            reasons,
        )
        candidates_by_id = {candidate.relation_id: candidate for candidate in candidates}
        decisions: list[VerificationDecision] = []
        for relation in relations:
            candidate = candidates_by_id.get(relation.id)
            if candidate is None:
                decisions.append(
                    VerificationDecision(
                        relation_id=relation.id,
                        verdict="insufficient",
                        reason="omitted",
                        confidence=0.0,
                    )
                )
                continue
            decisions.append(VerificationDecision.model_validate(candidate.model_dump()))
        judged_ids = {item.id for item in judged}
        entity_decisions = list(
            {
                candidate.entity_id: EntityVerificationDecision.model_validate(
                    candidate.model_dump()
                )
                for candidate in entity_candidates
                if candidate.entity_id in judged_ids
            }.values()
        )
        return VerificationOutcome(
            decisions=decisions,
            entity_decisions=entity_decisions,
            trace={
                "stage": "relation_verification",
                "window_id": window.id,
                **_window_scope(window),
                "input_records": len(relations),
                "supported_records": sum(decision.verdict == "supported" for decision in decisions),
                "entity_input_records": len(judged),
                "entity_specific_records": sum(
                    decision.verdict == "specific" for decision in entity_decisions
                ),
                "record_actions": dict(sorted(reasons.items())),
                **completion.provider_metadata,
            },
        )


def _window_scope(window: ExtractionWindow) -> dict[str, Any]:
    """Name the document and chunks a window covers, for the trace.

    Ids only: a finalizer that has to say which text a window covered joins
    them to the chunk table, and the trace stays proportional to calls, not
    to characters.
    """

    return {
        "document_id": window.document_id,
        "chunk_ids": [chunk.id for chunk in window.chunks],
    }


def _record_list(payload: dict[str, Any], key: str) -> list[object]:
    """Read one required top-level record array before sibling validation.

    A wrong top-level shape indicates a broken structured response and fails the
    call; item-level mistakes are handled separately by ``_validated_records``.
    """

    records = payload.get(key)
    if not isinstance(records, list):
        raise ValueError(f"Structured response field '{key}' must be an array")
    return records


# Counters that note a repair or a detail rather than turning a record away: a
# record that moved only these was kept.
_KEPT_RECORD_NOTES = frozenset(
    {
        "abbreviation_as_alias",
        "added_endpoint",
        "entity_llm_confirmed",
        "entity_llm_located",
        "endpoint_already_listed",
        "held_for_type_resolution",
        "implicit_document_source",
        "kept_under_fallback_relation",
        "llm_grounding_extended",
        "llm_grounding_located",
        "ocr_tolerant_grounding",
        "llm_grounding_to_verifier",
        "relabeled_relation",
        "retyped_entity",
        "rewritten_relation",
        "quantity_not_carried",
        "type_definition_as_description",
        "ungrounded_quantity",
        "untrusted_alias",
    }
)


class _RejectionLog:
    """The records one response proposed and validation turned away, and why.

    A duplicate is counted but not listed: the fact it states is already kept.
    """

    def __init__(self, reasons: Counter[str]) -> None:
        self.reasons = reasons
        self.records: list[dict[str, object]] = []

    def watch(self) -> Counter[str]:
        return Counter(self.reasons)

    def reject(self, before: Counter[str], record: dict[str, object]) -> None:
        """Log a record that was not kept, under the rejection it just counted."""

        reason = next(
            (
                name
                for name, count in self.reasons.items()
                if count > before.get(name, 0)
                and name not in _KEPT_RECORD_NOTES
                and not name.startswith("repaired_")
            ),
            "rejected",
        )
        self.add(reason, record)

    def add(self, reason: str, record: dict[str, object]) -> None:
        if reason != "duplicate":
            self.records.append({"reason": reason, **record})


def _entity_record(candidate: EntityCandidate) -> dict[str, object]:
    return {
        "kind": "entity",
        "name": candidate.name,
        "type": candidate.type,
        "quote": candidate.quote[:_REJECTED_QUOTE_CHARACTERS],
        "chunk_id": candidate.source_chunk_id,
        "candidate": _candidate_detail(
            candidate.quote,
            description=candidate.description,
            confidence=candidate.confidence,
            aliases=candidate.aliases,
        ),
    }


def _relation_record(
    candidate: RelationCandidate,
    entities_by_id: dict[str, EntityMention],
) -> dict[str, object]:
    source = entities_by_id.get(candidate.source_entity_id)
    target = entities_by_id.get(candidate.target_entity_id)
    return {
        "kind": "relation",
        "name": candidate.relation_type,
        "source": source.name if source else candidate.source_surface,
        "source_type": source.type if source else "",
        "target": target.name if target else candidate.target_surface,
        "target_type": target.type if target else "",
        "quote": candidate.quote[:_REJECTED_QUOTE_CHARACTERS],
        "chunk_id": candidate.source_chunk_id,
        "candidate": _candidate_detail(
            candidate.quote,
            description=candidate.description,
            confidence=candidate.confidence,
            source_surface=candidate.source_surface,
            target_surface=candidate.target_surface,
            quantities=[item.model_dump(mode="json") for item in candidate.quantities],
            assertion=candidate.assertion,
        ),
    }


def _raw_record(raw: object) -> dict[str, object]:
    """What can be read of a record that failed the response schema."""

    fields = raw if isinstance(raw, dict) else {}
    relation = "relation_type" in fields
    return {
        "kind": "relation" if relation else "entity",
        "name": str(fields.get("relation_type" if relation else "name") or ""),
        "type": str(fields.get("type") or ""),
        "source": str(fields.get("source_surface") or ""),
        "target": str(fields.get("target_surface") or ""),
        "quote": str(fields.get("quote") or "")[:_REJECTED_QUOTE_CHARACTERS],
        "chunk_id": str(fields.get("source_chunk_id") or ""),
    }


def _validated_records[RecordT: BaseModel](
    payload: dict[str, Any],
    key: str,
    model_type: type[RecordT],
    reasons: Counter[str],
    rejections: _RejectionLog | None = None,
) -> tuple[list[object], list[RecordT]]:
    """Validate sibling records independently so one model error remains local.

    Both raw and accepted records are returned so traces can report complete input
    counts while invalid-schema reasons remain visible in run traces.
    """

    raw_records = _record_list(payload, key)
    records: list[RecordT] = []
    for raw_record in raw_records:
        try:
            records.append(model_type.model_validate(raw_record))
        except ValidationError:
            # JSON-schema constrained local models can still violate a numeric
            # bound. The trace records the drop while valid siblings survive.
            reasons["invalid_schema"] += 1
            if rejections is not None:
                rejections.add("invalid_schema", _raw_record(raw_record))
    return raw_records, records


def _orient_relation(
    source: EntityMention,
    target: EntityMention,
    definition: RelationTypeDefinition,
) -> tuple[EntityMention, EntityMention, bool]:
    """Orient endpoints according to ontology types when reversal is unambiguous.

    The boolean records whether a repair occurred; incompatible pairs retain their
    original order and are rejected by the caller's subsequent type check.
    """

    if definition.admits(source.type, target.type):
        return source, target, False
    if definition.admits(target.type, source.type):
        return target, source, True
    return source, target, False


def _dedupe_aliases(aliases: list[str], name: str) -> list[str]:
    """Return accepted aliases distinct from the canonical mention name.

    Original provider ordering is preserved deterministically.
    """

    seen = {_normalize(name)}
    result: list[str] = []
    for alias in aliases:
        cleaned = alias.strip()
        key = _normalize(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _document_context_surface_match(
    name: str, quote: str, aliases: list[str]
) -> Literal["llm", "ocr_tolerant"] | None:
    """Validate a focal identity against its exact front-matter evidence quote.

    A model may quote a short phrase containing the canonical name rather than
    returning the name alone, and a trusted source-grounded alias may be the
    surface printed by the paper. Exact surface occurrence and conservative
    spelling identity support those cases. Compact equality remains as a narrow
    OCR repair for isolated spaces such as ``M ETHOD``; semantic similarity is
    intentionally absent so a related title cannot become a document-wide actor.
    Last, the name may be in the quote as OCR printed it - a letter misread,
    "Stain1ess steel" - which is returned as ``ocr_tolerant`` so the evidence
    is labelled; None when the name is not in the quote at all.
    """

    compact_name = "".join(character for character in name.casefold() if character.isalnum())
    compact_quote = "".join(character for character in quote.casefold() if character.isalnum())
    if not compact_name:
        return None
    if (
        compact_name == compact_quote
        or surface_occurs(quote, [name, *aliases])
        or identity_surface_match(name, quote)
    ):
        return "llm"
    return "ocr_tolerant" if ocr_tolerant_surface_occurs(quote, [name, *aliases]) else None


def _string_evidence_method(
    ocr_tolerant: bool, reasons: Counter[str]
) -> Literal["llm", "ocr_tolerant"]:
    """The method of evidence string grounding found, counting OCR-tolerant finds.

    Evidence found only through the OCR-tolerant view does not name its subject
    as written, so it carries its own label to be judged apart.
    """

    if not ocr_tolerant:
        return "llm"
    reasons["ocr_tolerant_grounding"] += 1
    return "ocr_tolerant"


def _entity_description(value: str, definitions: frozenset[str]) -> str:
    """Return a description of this entity, dropping one of a category.

    The extraction request carries every ontology type definition, so a model can
    answer the description field with a definition it was just given. That text
    says nothing about the entity, and it says it identically for every entity it
    is repeated for: it inflates resolution similarity between distinct entities,
    is read as a finding in the explorer, and survives description merging, which
    counts distinct descriptions and so never sees more than one. Recording it as
    absent is the honest reading of what the model returned.

    Every definition is compared, not only the one belonging to this entity's own
    type: a model that reaches for the wrong definition, such as the LOCATION
    definition for a CONCEPT, has still described a category rather than the
    entity.
    """

    description = value.strip()
    if not description:
        return ""
    if _definition_key(description) in definitions:
        return ""
    return description


def _ontology_definition_keys(ontology: OntologyProfile) -> frozenset[str]:
    """Return every ontology definition in the form used to recognize one."""

    return frozenset(
        key for key in (_definition_key(item.description) for item in ontology.entity_types) if key
    )


def _definition_key(value: str) -> str:
    """Reduce text to what distinguishes a definition from a description.

    A definition repeated back with a dropped full stop, a doubled space, or a
    non-breaking space is the same category text and carries no more about the
    entity than the exact copy does.
    """

    return " ".join(value.casefold().split()).rstrip(".。 ")


def _normalize(value: str) -> str:
    """Create a case- and whitespace-insensitive key for local record deduplication.

    Punctuation remains significant at this stage.
    """

    return " ".join(value.casefold().split())
