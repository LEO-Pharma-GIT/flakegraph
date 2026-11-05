# SPDX-License-Identifier: Apache-2.0
"""Put a document's stored records through the current validation and verification.

A revision can keep a document's text and extraction and still want the current
rules applied to them: a new identifier pattern, a rewritten relation type, a
more tolerant grounding, a better verifier. Extracting the document again
would pay for every model call a second time. Revalidation asks the model to
extract nothing. Every record the document stored - the ones its extraction
accepted and the ones validation or verification turned away - is rebuilt into
the candidate the model proposed and replayed through the current
extractors as though the model had just returned it. The validation those
candidates meet is therefore exactly the code a new extraction would meet, and
it changes whenever that code does. Verification is the extraction's own,
entity verdicts included, with each document's opening; the grounding
fallbacks and the verifier still call the model, because that is validation.

Revalidation takes the phases extraction takes, so a fleet can split a long
document into short window tasks as it does for extraction: the document's
records are rebuilt, its context validated and the rest placed by window;
each window's entities are replayed; the context and window entities become
the document's entities; each window's relations are replayed against them and
verified; and the windows are combined, in window order, with the entity
verdicts applied to the whole document.

Reconstruction is exact for what a trace stores and approximate for what it
does not, and every approximation is counted in the document's ``revalidation``
event rather than left silent:

* A rejected relation names its endpoints, not the ids they had. Each is
  resolved against the window's current entities by normalised name, alias or
  context surface and, when the record has one, by type - the mention in the
  record's own chunk first. An endpoint no stored entity names, with a type,
  is proposed as an entity of its own, in its relation's chunk, and meets the
  entity rules and entity verification like any other; one without a type
  stays unresolved and the relation is rejected for it.
* Records a window never stored - a window whose extraction call failed, or
  relations a verifier turned away with no verdict recorded - cannot be
  recovered and stay lost; their error events stay in the trace.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from kg_processor.application.content_kinds import (
    dataset_columns,
    dataset_profiles,
    window_filter,
)
from kg_processor.application.extraction_grounding import find_quote
from kg_processor.application.extraction_windows import (
    build_document_context_windows,
    build_document_openings,
    build_extraction_windows,
    window_skip_reason,
)
from kg_processor.application.folder_entities import is_folder_mention
from kg_processor.application.llm_extractors import LlmEntityExtractor, LlmRelationExtractor
from kg_processor.application.ordered_map import ordered_map
from kg_processor.application.rejected_records import REJECTED_QUOTE_CHARACTERS, candidate_detail
from kg_processor.application.relation_candidates import discover_cue_relation_candidates
from kg_processor.application.two_pass_extraction import (
    _dedupe_mentions,
    _dedupe_relations,
    _relation_inventory_for_window,
    _verify_held,
    _verify_window,
    apply_entity_verdicts,
)
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.extraction import (
    EntityExtractionOutcome,
    EntityMention,
    ExtractionObservations,
    ExtractionWindow,
    RelationObservation,
)
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ontology import OntologyProfile, normalize_ontology_label
from kg_processor.domain.stages import (
    DocumentEntityInventoryShard,
    DocumentRevalidationShard,
    EntityWindowShard,
    RevalidatedRelationWindowShard,
    RevalidationWindow,
    RevalidationWindowShard,
    StoredEntity,
    StoredRelation,
)
from kg_processor.ports.extraction import EntityExtractor, RelationExtractor, RelationVerifier
from kg_processor.ports.llm import (
    LlmCapabilities,
    StructuredCompletionProvider,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)

REVALIDATION_STAGE = "revalidation"
_CONTEXT_STAGE = "document_context_extraction"
# Trace events a revalidation replaces: the record-bearing passes and what was
# derived from them. Error events of extraction calls stay, because what a
# failed call never returned is still missing.
SUPERSEDED_STAGES = frozenset(
    {
        _CONTEXT_STAGE,
        "entity_extraction",
        "relation_extraction",
        "relation_inventory",
        "relation_verification",
        "relation_verification_error",
        "held_relation_verification",
        "held_relation_verification_error",
        "verification_cascade",
        "cue_candidate_generation",
        "reference_filter",
        REVALIDATION_STAGE,
    }
)
# Trace events of a document's relation phase. What remains without them is
# the document's trace up to its entity inventory, which a later revision that
# extracts only relations again builds on.
RELATION_PHASE_STAGES = frozenset(
    {
        "relation_inventory",
        "relation_extraction",
        "relation_extraction_error",
        "relation_gleaning_error",
        "relation_verification",
        "relation_verification_error",
        "held_relation_verification",
        "held_relation_verification_error",
        "verification_cascade",
        "cue_candidate_generation",
        REVALIDATION_STAGE,
    }
)
# Trace events of a document's entity windows. Without them and the relation
# phase, what remains is the document's trace up to its context.
ENTITY_WINDOW_STAGES = frozenset(
    {
        "entity_extraction",
        "entity_gleaning_error",
        "cue_entity_audit_error",
        "reference_filter",
        "window_error",
    }
)
# The passes whose rejected records are candidates again.
_REPLAYED_RECORD_STAGES = frozenset(
    {
        _CONTEXT_STAGE,
        "entity_extraction",
        "relation_extraction",
        "relation_verification",
        "relation_verification_error",
        "verification_cascade",
        REVALIDATION_STAGE,
    }
)
# How many admitted records the revalidation event names.
_MAX_LISTED_RECORDS = 200

# The extractors revalidation replays through, built over a given provider: the
# configured ones, so every flag a new extraction runs with applies.
ReplayExtractors = Callable[
    [StructuredCompletionProvider], tuple[EntityExtractor, RelationExtractor]
]


@dataclass
class StoredRecords:
    """Every candidate a document's stored extraction can give back."""

    context: list[StoredEntity] = field(default_factory=list)
    entities: list[StoredEntity] = field(default_factory=list)
    relations: list[StoredRelation] = field(default_factory=list)
    endpoints_proposed: int = 0


def stored_records(
    observations: ExtractionObservations,
    chunks: Iterable[Chunk],
) -> StoredRecords:
    """Rebuild the candidates a document's passes proposed, accepted ones first.

    Accepted mentions and relations come from the observations; rejected ones
    from the trace, whichever pass turned them away. A held relation is
    rebuilt from its rejected record, which it always has. A record that
    failed the response schema is not a candidate - the model never proposed a
    readable record - and a duplicate was never listed, since what it stated is
    kept. Last, an endpoint a relation names that no stored entity does is
    proposed as an entity in the relation's chunk.
    """

    content = {chunk.id: chunk.content for chunk in chunks}
    records = StoredRecords()
    mentions = {mention.id: mention for mention in observations.entities}
    for mention in observations.entities:
        if is_folder_mention(mention):
            # Named by the document's folder, not proposed by the model: it is
            # kept as stored.
            continue
        stored = StoredEntity(
            name=mention.name,
            type=mention.type,
            chunk_id=mention.source_chunk_id,
            quote=mention.quote,
            description=mention.description,
            confidence=mention.confidence,
            aliases=list(mention.aliases),
            accepted=True,
        )
        (records.context if mention.is_document_context else records.entities).append(stored)
    for relation in observations.relations:
        if relation.evidence_method == "ontology_cue":
            # Found by code, not proposed by the model: the current cue rules
            # find it again, or do not.
            continue
        source = mentions.get(relation.source_entity_id)
        target = mentions.get(relation.target_entity_id)
        records.relations.append(
            StoredRelation(
                # The type the model stated, so the current rewrites and relabels
                # see what it said rather than what the rules at extraction made of it.
                relation_type=relation.proposed_relation_type or relation.relation_type,
                source=source.name if source else relation.source_surface or "",
                source_type=source.type if source else "",
                target=target.name if target else relation.target_surface or "",
                target_type=target.type if target else "",
                chunk_id=relation.source_chunk_id,
                quote=relation.quote,
                source_surface=relation.source_surface or "",
                target_surface=relation.target_surface or "",
                description=relation.description,
                confidence=relation.confidence,
                quantities=[
                    {
                        "text": item.text,
                        "kind": item.kind or "",
                        "unit": item.unit or "",
                        "comparator": item.comparator or "",
                    }
                    for item in relation.quantities
                ],
                assertion=relation.assertion,
                accepted=True,
                source_id=relation.source_entity_id,
                target_id=relation.target_entity_id,
            )
        )
    for stage, record in _rejected_records(observations.trace):
        _add_rejected(records, stage, record)
    _propose_endpoints(records, observations.entities, content)
    return records


def _add_rejected(
    records: StoredRecords,
    stage: str,
    record: Mapping[str, Any],
) -> None:
    """Rebuild one rejected record as a candidate of the pass it came from."""

    detail: Mapping[str, Any] = record["candidate"]
    chunk_id = str(record["chunk_id"])
    quote = str(detail.get("quote") or record["quote"])
    confidence = float(detail["confidence"])
    description = str(detail["description"])
    if record["kind"] == "relation":
        records.relations.append(
            StoredRelation(
                relation_type=str(detail.get("proposed_relation_type") or record["name"]),
                source=str(record["source"]),
                source_type=str(record["source_type"]),
                target=str(record["target"]),
                target_type=str(record["target_type"]),
                chunk_id=chunk_id,
                quote=quote,
                source_surface=str(detail["source_surface"]),
                target_surface=str(detail["target_surface"]),
                description=description,
                confidence=confidence,
                quantities=[dict(item) for item in detail["quantities"]],
                assertion=detail["assertion"],
            )
        )
        return
    stored = StoredEntity(
        name=str(record["name"]),
        type=str(record["type"]),
        chunk_id=chunk_id,
        quote=quote,
        description=description,
        confidence=confidence,
        aliases=[str(item) for item in detail["aliases"]],
    )
    (records.context if stage == _CONTEXT_STAGE else records.entities).append(stored)


def _rejected_records(
    trace: Iterable[Mapping[str, Any]],
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """The rejected records of the passes revalidation replays, with their pass.

    A new endpoint a relation response proposed is left out: it returns as an
    endpoint the relation names. So is a record that failed the response
    schema, and one the cue rules proposed, which run again of their own accord.
    """

    for event in trace:
        stage = str(event.get("stage") or "")
        if stage not in _REPLAYED_RECORD_STAGES:
            continue
        for record in event.get("rejected_records") or []:
            if record["reason"] == "invalid_schema":
                continue
            if stage == "relation_extraction" and record["kind"] != "relation":
                continue
            if record["candidate"].get("evidence_method") == "ontology_cue":
                continue
            yield stage, record


def _propose_endpoints(
    records: StoredRecords,
    stored_mentions: list[EntityMention],
    content: Mapping[str, str],
) -> None:
    """Propose each typed endpoint no stored entity answers to, where its relation is.

    The endpoint is the model's own - it named it in a relation - so it is
    offered to the entity rules in the relation's chunk, quoted by the
    relation's evidence when that names it. A name a stored mention answers to
    by alias or context surface is that mention, and is not proposed.
    """

    known = {
        (normalize_ontology_label(item.name), item.type)
        for item in [*records.context, *records.entities]
    }
    surfaces = {key for mention in stored_mentions for key in _surface_keys(mention)}
    for relation in records.relations:
        for name, type_ in (
            (relation.source, relation.source_type),
            (relation.target, relation.target_type),
        ):
            key = normalize_ontology_label(name)
            if not key or not type_ or (key, type_) in known or key in surfaces:
                continue
            chunk = content.get(relation.chunk_id, "")
            if find_quote(chunk, name) is None:
                continue
            known.add((key, type_))
            records.endpoints_proposed += 1
            records.entities.append(
                StoredEntity(
                    name=name,
                    type=type_,
                    chunk_id=relation.chunk_id,
                    quote=relation.quote if find_quote(relation.quote, name) else name,
                    confidence=relation.confidence,
                )
            )


class ReplayedResponse:
    """Answer one kind of extraction request with stored records; pass every other call on.

    The extractor that receives this provider validates the stored records as
    the model's answer. Grounding, relabelling and verification requests still
    reach the model, because they are part of validation.
    """

    def __init__(
        self,
        llm: StructuredCompletionProvider,
        task_name: str,
        payload: dict[str, Any],
    ) -> None:
        """Hold the one answer to give and the provider for everything else."""

        self.llm = llm
        self.task_name = task_name
        self.payload = payload

    def capabilities(self) -> LlmCapabilities:
        """Report the underlying provider's capabilities."""

        return self.llm.capabilities()

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Return the stored records for the replayed task and ask the model otherwise."""

        if request.task_name == self.task_name:
            return StructuredCompletionResult(
                payload=self.payload,
                provider_metadata={"replayed": True, REVALIDATION_STAGE: True},
            )
        return self.llm.complete_structured(request)


@dataclass
class _EndpointStats:
    """How relation endpoints were found."""

    resolved: int = 0
    unresolved: int = 0


@dataclass(frozen=True)
class RevalidationRequest:
    """What one document's revalidation runs with."""

    llm: StructuredCompletionProvider
    extractors: ReplayExtractors
    verifier: RelationVerifier
    settings: GraphSettings
    ontology: OntologyProfile
    model: str
    timeout_seconds: int


def revalidate_observations(
    chunks: list[Chunk],
    observations: ExtractionObservations,
    request: RevalidationRequest,
) -> ExtractionObservations:
    """Apply the current validation and verification to one document's stored records.

    ``chunks`` are the document's prepared chunks, ``observations`` what its
    extraction stored. Windows are rebuilt from the chunks under the current
    settings, as extraction builds them. Document-context candidates are
    validated first, then each window's entity candidates, then its relation
    candidates against the entities that stand; the current cue rules run, the
    verifier judges relations and entities with the document's opening as a
    relation window's verification does, held statements are judged apart, and
    the entity verdicts apply to the whole document. The result replaces the
    stored observations; its trace keeps every event revalidation does not
    supersede and adds one ``revalidation`` event that says what changed.

    This takes every step in one call, replaying ``extraction_parallelism``
    windows at a time. A fleet takes the same steps as separate tasks, one
    window each, and combines their results in window order, so the document
    comes out the same.
    """

    file_ids = sorted({chunk.file_id for chunk in chunks})
    begun, windows = begin_revalidation(file_ids, chunks, observations, request)
    shards = [RevalidationWindowShard(file_ids=file_ids, window=entry) for entry in windows]
    parallelism = request.settings.extraction_parallelism
    inventory = revalidated_inventory(
        begun,
        ordered_map(
            shards,
            lambda _index, shard: revalidate_entity_window(shard, request),
            parallelism=parallelism,
        ),
    )
    return conclude_revalidation(
        observations,
        begun,
        inventory,
        ordered_map(
            shards,
            lambda _index, shard: revalidate_relation_window(shard, inventory, request),
            parallelism=parallelism,
        ),
        request.ontology,
    )


def begin_revalidation(
    file_ids: list[str],
    chunks: list[Chunk],
    observations: ExtractionObservations,
    request: RevalidationRequest,
) -> tuple[DocumentRevalidationShard, list[RevalidationWindow]]:
    """Rebuild a document's stored records, validate its context, and place the rest by window.

    This is the part of revalidation that reads the whole document, and it
    asks the model little: the context pass replays its candidates with no
    grounding fallback. The windows come back apart from the document's result,
    each carrying the records that fall in it, so each can be replayed on its
    own; a window the current rules skip is not among them, and its records are
    listed as lost.
    """

    entity_replay, relation_replay = request.extractors(request.llm)
    if not isinstance(entity_replay, LlmEntityExtractor) or not isinstance(
        relation_replay, LlmRelationExtractor
    ):
        raise ValueError(
            "revalidation replays stored records through the LLM extractors; "
            "this configuration extracts with others"
        )
    records = stored_records(observations, chunks)
    trace: list[dict[str, Any]] = []
    lost: list[dict[str, Any]] = []
    context = [
        *_revalidate_context(chunks, observations.trace, records.context, request, trace, lost),
        *(mention for mention in observations.entities if is_folder_mention(mention)),
    ]
    windows = _windows_with_records(
        records, chunks, observations.trace, request.settings, trace, lost
    )
    for entry in windows:
        named = {
            mention_id
            for relation in entry.relations
            for mention_id in (relation.source_id, relation.target_id)
            if mention_id
        }
        entry.stored_mentions = [item for item in observations.entities if item.id in named]
    begun = DocumentRevalidationShard(
        file_ids=file_ids,
        document_id=chunks[0].document_id if chunks else "",
        document_context_entities=context,
        trace=trace,
        rejected_records=lost,
        reconstruction={
            "context_candidates": len(records.context),
            "entity_candidates": len(records.entities),
            "relation_candidates": len(records.relations),
            "endpoints_proposed": records.endpoints_proposed,
        },
        window_count=len(windows),
        document_openings=build_document_openings(chunks),
    )
    return begun, windows


def revalidate_entity_window(
    shard: RevalidationWindowShard,
    request: RevalidationRequest,
) -> EntityWindowShard:
    """Validate one window's stored entity candidates as the current extractor would."""

    outcome = _revalidate_window_entities(shard.window, request)
    return EntityWindowShard(
        file_ids=shard.file_ids,
        chunk_ids=[chunk.id for chunk in shard.window.window.chunks],
        entities=outcome.entities if outcome is not None else [],
        trace=[outcome.trace] if outcome is not None else [],
    )


def revalidated_inventory(
    begun: DocumentRevalidationShard,
    entity_windows: list[EntityWindowShard],
) -> DocumentEntityInventoryShard:
    """Combine the context and the entity windows, in window order, into the document's entities.

    These are the entities every relation window is offered, as each relation
    window of an extraction is offered its document's inventory.
    """

    entities = {mention.id: mention for mention in begun.document_context_entities}
    for shard in entity_windows:
        entities.update({mention.id: mention for mention in shard.entities})
    return DocumentEntityInventoryShard(
        file_ids=begun.file_ids,
        entities=list(entities.values()),
        trace=[*begun.trace, *(event for shard in entity_windows for event in shard.trace)],
        chunk_count=sum(len(shard.chunk_ids) for shard in entity_windows),
        window_count=begun.window_count,
        document_openings=begun.document_openings,
    )


def revalidate_relation_window(
    shard: RevalidationWindowShard,
    inventory: DocumentEntityInventoryShard,
    request: RevalidationRequest,
) -> RevalidatedRelationWindowShard:
    """Validate and verify one window's stored relations against the document's entities.

    A window of a dataset whose profile asks for entities only replays no
    relations; it still reports, so the document's windows are all accounted for.
    """

    entry = shard.window
    chunk_ids = [chunk.id for chunk in entry.window.chunks]
    if not entry.extract_relations:
        return RevalidatedRelationWindowShard(file_ids=shard.file_ids, chunk_ids=chunk_ids)
    result = _revalidate_window_relations(
        entry,
        inventory.entities,
        inventory.document_openings.get(entry.window.document_id, ""),
        request,
    )
    return RevalidatedRelationWindowShard(
        file_ids=shard.file_ids,
        chunk_ids=chunk_ids,
        relations=result.relations,
        held_relations=result.held,
        entities=result.entities,
        trace=result.trace,
        rejected_records=result.lost,
        endpoints_resolved=result.endpoints.resolved,
        endpoints_unresolved=result.endpoints.unresolved,
    )


def conclude_revalidation(
    before: ExtractionObservations,
    begun: DocumentRevalidationShard,
    inventory: DocumentEntityInventoryShard,
    relation_windows: list[RevalidatedRelationWindowShard],
    ontology: OntologyProfile,
) -> ExtractionObservations:
    """Combine a document's revalidated windows, in window order, and apply the entity verdicts.

    A window's verifier judges only that window's mentions, but a relation in
    any window may use one, so the verdicts apply here, once every window of
    the document is in. ``before`` is what the document's extraction stored:
    its trace events revalidation does not supersede are kept, and the
    ``revalidation`` event compares the result with it.
    """

    entities = {mention.id: mention for mention in inventory.entities}
    relations: list[RelationObservation] = []
    held: list[RelationObservation] = []
    trace: list[dict[str, Any]] = []
    lost = list(begun.rejected_records)
    for shard in relation_windows:
        trace.extend(shard.trace)
        lost.extend(shard.rejected_records)
        relations.extend(shard.relations)
        held.extend(shard.held_relations)
        entities.update({mention.id: mention for mention in shard.entities})
    kept_events = [
        event for event in before.trace if str(event.get("stage") or "") not in SUPERSEDED_STAGES
    ]
    revised = apply_entity_verdicts(
        ExtractionObservations(
            entities=list(entities.values()),
            relations=_dedupe_relations(relations),
            held_relations=_dedupe_relations(held),
            trace=[*kept_events, *inventory.trace, *trace],
            chunk_count=before.chunk_count,
            window_count=before.window_count,
        ),
        ontology,
    )
    summary = _summary(
        before,
        revised,
        {
            **begun.reconstruction,
            "endpoints_resolved": sum(shard.endpoints_resolved for shard in relation_windows),
            "endpoints_unresolved": sum(shard.endpoints_unresolved for shard in relation_windows),
        },
        lost,
        document_id=begun.document_id,
        windows=begun.window_count,
    )
    return revised.model_copy(update={"trace": [*revised.trace, summary]})


def _revalidate_context(
    chunks: list[Chunk],
    stored_trace: list[dict[str, Any]],
    candidates: list[StoredEntity],
    request: RevalidationRequest,
    trace: list[dict[str, Any]],
    lost: list[dict[str, Any]],
) -> list[EntityMention]:
    """Validate the document-context candidates as the context pass does.

    The context window is the document's front matter under the current
    settings. A dataset has no context mentions, its opening is profiled
    instead, and its profile stays as it was.
    """

    if not candidates:
        return []
    profiled = set(dataset_profiles(stored_trace))
    windows = {
        chunk.id: window
        for window in build_document_context_windows(
            chunks,
            request.settings.document_context_tokens,
            request.settings.document_context_max_chunks,
        )
        if window.document_id not in profiled
        for chunk in window.chunks
    }
    by_window: dict[str, tuple[ExtractionWindow, list[StoredEntity]]] = {}
    for candidate in candidates:
        window = windows.get(candidate.chunk_id)
        if window is None:
            lost.append(_entity_loss(candidate, "chunk_outside_context_window"))
            continue
        by_window.setdefault(window.id, (window, []))[1].append(candidate)
    mentions: list[EntityMention] = []
    for window, window_candidates in by_window.values():
        # The context pass builds its extractor with no fallback, and so does this.
        extractor = LlmEntityExtractor(
            ReplayedResponse(
                request.llm,
                _CONTEXT_STAGE,
                {"entities": [item.candidate() for item in window_candidates]},
            ),
            request.settings.deterministic_seed,
        )
        outcome = extractor.extract_document_context_entities(
            window,
            request.ontology,
            model=request.model,
            timeout_seconds=request.timeout_seconds,
            # The stored answer was within the cap when it was given; a record
            # the current rules admit beyond it is not silently cut.
            max_entities=len(window_candidates),
        )
        trace.append(outcome.trace)
        mentions.extend(outcome.entities)
    return mentions


def _windows_with_records(
    records: StoredRecords,
    chunks: list[Chunk],
    stored_trace: list[dict[str, Any]],
    settings: GraphSettings,
    trace: list[dict[str, Any]],
    lost: list[dict[str, Any]],
) -> list[RevalidationWindow]:
    """Rebuild the document's windows and give each the candidates in its chunks.

    A window the current rules skip - a bibliography, or a dataset whose
    profile asks for no entities - takes nothing, and each record it would have
    held is listed in ``lost`` under the rule's reason. A catalogue's windows
    carry its column relations, as its relation windows do.
    """

    allowed = window_filter(stored_trace, catalogue_relations=settings.catalogue_relations)
    columns = dataset_columns(stored_trace)
    by_chunk: dict[str, RevalidationWindow] = {}
    grouped: list[RevalidationWindow] = []
    for window in build_extraction_windows(
        chunks, settings.extraction_window_tokens, settings.max_chunks_per_llm_call
    ):
        entry = RevalidationWindow(
            window=window.model_copy(
                update={"dataset_columns": columns.get(window.document_id, [])}
            )
        )
        grouped.append(entry)
        by_chunk.update({chunk.id: entry for chunk in window.chunks})
    for entity in records.entities:
        if (target := by_chunk.get(entity.chunk_id)) is None:
            lost.append(_entity_loss(entity, "chunk_not_in_document"))
        else:
            target.entities.append(entity)
    for relation in records.relations:
        if (target := by_chunk.get(relation.chunk_id)) is None:
            lost.append(_relation_loss(relation, "chunk_not_in_document"))
        else:
            target.relations.append(relation)
    active: list[RevalidationWindow] = []
    for entry in grouped:
        extract_entities, extract_relations = allowed(entry.window.document_id)
        reason = window_skip_reason(entry.window) or (
            None if extract_entities else "dataset_without_entities"
        )
        if reason is None:
            if not extract_relations:
                lost.extend(
                    _relation_loss(item, "dataset_without_relations") for item in entry.relations
                )
                entry.relations = []
                entry.extract_relations = False
            active.append(entry)
        elif entry.entities or entry.relations:
            lost.extend(_entity_loss(item, reason) for item in entry.entities)
            lost.extend(_relation_loss(item, reason) for item in entry.relations)
            trace.append(
                {
                    "stage": "reference_filter",
                    "window_id": entry.window.id,
                    "skipped": True,
                    "reason": reason,
                }
            )
    return active


def _revalidate_window_entities(
    entry: RevalidationWindow,
    request: RevalidationRequest,
) -> EntityExtractionOutcome | None:
    """Validate a window's stored entity candidates as the current extractor would."""

    if not entry.entities:
        return None
    extractor, _relations = request.extractors(
        ReplayedResponse(
            request.llm,
            "entity_extraction",
            {"entities": [item.candidate() for item in entry.entities]},
        )
    )
    return extractor.extract(
        entry.window,
        request.ontology,
        model=request.model,
        timeout_seconds=request.timeout_seconds,
        max_entities=len(entry.entities),
    )


@dataclass
class _RelationWindowResult:
    """One window's revalidated relations, held statements, added endpoints and trace."""

    relations: list[RelationObservation]
    held: list[RelationObservation]
    entities: list[EntityMention]
    trace: list[dict[str, Any]]
    lost: list[dict[str, Any]]
    endpoints: _EndpointStats


def _revalidate_window_relations(
    entry: RevalidationWindow,
    document_entities: list[EntityMention],
    document_opening: str,
    request: RevalidationRequest,
) -> _RelationWindowResult:
    """Validate a window's relation candidates, add cue relations, then verify as extraction does.

    The window's entities are the ones a relation window would be offered: the
    document's surviving mentions whose names occur here. Verification is the
    relation window's own - relations and this window's entities in one call,
    with the document's opening and subjects - and held statements are judged
    apart from their types.
    """

    window = entry.window
    settings = request.settings
    window_entities = _relation_inventory_for_window(document_entities, window)
    trace: list[dict[str, Any]] = []
    payload, endpoints = relation_payload(
        entry.relations,
        window_entities,
        {mention.id: mention for mention in entry.stored_mentions},
    )
    relations: list[RelationObservation] = []
    held: list[RelationObservation] = []
    added: list[EntityMention] = []
    if payload["relations"]:
        _entities, extractor = request.extractors(
            ReplayedResponse(request.llm, "relation_extraction", payload)
        )
        outcome = extractor.extract(
            window,
            window_entities,
            request.ontology,
            model=request.model,
            timeout_seconds=request.timeout_seconds,
            max_relations=len(payload["relations"]),
        )
        trace.append(outcome.trace)
        relations = outcome.relations
        held = outcome.held_relations
        added = outcome.entities
    lost = (
        [_relation_loss(item, "fewer_than_two_entities") for item in entry.relations]
        if payload["relations"] and trace[-1].get("skipped") == "fewer_than_two_entities"
        else []
    )
    cue_candidates = discover_cue_relation_candidates(window, window_entities, request.ontology)
    before = len(relations)
    relations = _dedupe_relations([*relations, *cue_candidates])
    trace.append(
        {
            "stage": "cue_candidate_generation",
            "window_id": window.id,
            "candidate_records": len(cue_candidates),
            "additional_records": len(relations) - before,
            "direct_evidence_records": sum(
                not relation.verification_required for relation in cue_candidates
            ),
        }
    )
    context = _dedupe_mentions([*window_entities, *added])
    relations = _verify_window(
        window,
        settings,
        request.ontology,
        request.verifier,
        model=request.model,
        timeout_seconds=request.timeout_seconds,
        document_entities=document_entities,
        added=added,
        context=context,
        relations=relations,
        document_opening=document_opening,
        trace=trace,
    )
    held = _verify_held(
        window,
        settings,
        request.ontology,
        request.verifier,
        model=request.model,
        timeout_seconds=request.timeout_seconds,
        context=context,
        held=_dedupe_relations(held),
        document_opening=document_opening,
        trace=trace,
    )
    return _RelationWindowResult(relations, held, added, trace, lost, endpoints)


def relation_payload(
    relations: list[StoredRelation],
    window_entities: list[EntityMention],
    old_mentions: Mapping[str, EntityMention],
) -> tuple[dict[str, Any], _EndpointStats]:
    """Rebuild a relation response from stored relations and the window's entities.

    Each endpoint is the window entity it resolves to. One that resolves to
    nothing keeps an empty id, and the current rules reject the relation for it.
    """

    stats = _EndpointStats()
    by_id = {entity.id: entity for entity in window_entities}

    def endpoint(stored_id: str | None, name: str, type_: str, relation: StoredRelation) -> str:
        if stored_id and stored_id in by_id:
            stats.resolved += 1
            return stored_id
        previous = old_mentions.get(stored_id) if stored_id else None
        if previous is not None:
            name, type_ = previous.name, previous.type
        found = resolve_endpoint(name, type_, relation.chunk_id, window_entities)
        if found is None:
            stats.unresolved += 1
            return ""
        stats.resolved += 1
        return found.id

    candidates = [
        {
            "source_entity_id": endpoint(
                relation.source_id, relation.source, relation.source_type, relation
            ),
            "target_entity_id": endpoint(
                relation.target_id, relation.target, relation.target_type, relation
            ),
            "source_surface": relation.source_surface or relation.source,
            "target_surface": relation.target_surface or relation.target,
            "relation_type": relation.relation_type,
            "description": relation.description,
            "source_chunk_id": relation.chunk_id,
            "quote": relation.quote,
            "confidence": relation.confidence,
            "quantities": relation.quantities,
            "assertion": relation.assertion,
        }
        for relation in relations
    ]
    return {"relations": candidates}, stats


def resolve_endpoint(
    name: str,
    type_: str,
    chunk_id: str,
    entities: list[EntityMention],
) -> EntityMention | None:
    """The entity a stored endpoint name refers to, when exactly one identity fits.

    A name matches an entity's name, an alias, or a document-context surface,
    after normalisation. A stored type picks among identities of that name -
    the entity's type, or the one it was proposed under before verification
    retyped it. When no identity of that name has the type, the name alone
    decides, as for an untyped name: the stored record was stated about the
    thing this run types otherwise, and the current rules judge the relation
    under its type. Two identities that fit a name alone are ambiguous, and
    nothing is resolved. The mention in the record's own chunk is preferred.
    """

    key = normalize_ontology_label(name)
    if not key:
        return None
    named = [entity for entity in entities if key in _surface_keys(entity)]
    typed = [entity for entity in named if type_ in (entity.type, entity.proposed_type)]
    matches = typed if type_ and typed else named
    if not matches:
        return None
    identities = {(normalize_ontology_label(item.name), item.type) for item in matches}
    if len(identities) > 1 and not (type_ and typed):
        return None
    return next((item for item in matches if item.source_chunk_id == chunk_id), matches[0])


def _surface_keys(entity: EntityMention) -> set[str]:
    """Every normalised surface an entity answers to."""

    return {
        normalize_ontology_label(value)
        for value in (entity.name, *entity.aliases, *entity.contextual_surfaces)
        if value.strip()
    }


def _entity_loss(entity: StoredEntity, reason: str) -> dict[str, Any]:
    """A stored entity revalidation could not put to the rules, as a rejected record."""

    return {
        "kind": "entity",
        "reason": reason,
        "name": entity.name,
        "type": entity.type,
        "quote": entity.quote[:REJECTED_QUOTE_CHARACTERS],
        "chunk_id": entity.chunk_id,
        "candidate": candidate_detail(
            entity.quote,
            description=entity.description,
            confidence=entity.confidence,
            aliases=list(entity.aliases),
        ),
    }


def _relation_loss(relation: StoredRelation, reason: str) -> dict[str, Any]:
    """A stored relation revalidation could not put to the rules, as a rejected record."""

    return {
        "kind": "relation",
        "reason": reason,
        "name": relation.relation_type,
        "source": relation.source,
        "source_type": relation.source_type,
        "target": relation.target,
        "target_type": relation.target_type,
        "quote": relation.quote[:REJECTED_QUOTE_CHARACTERS],
        "chunk_id": relation.chunk_id,
        "candidate": candidate_detail(
            relation.quote,
            description=relation.description,
            confidence=relation.confidence,
            source_surface=relation.source_surface,
            target_surface=relation.target_surface,
            quantities=relation.quantities,
            assertion=relation.assertion,
        ),
    }


def _summary(
    before: ExtractionObservations,
    after: ExtractionObservations,
    reconstruction: dict[str, int],
    lost: list[dict[str, Any]],
    *,
    document_id: str,
    windows: int,
) -> dict[str, Any]:
    """Say what revalidation changed, and list the losses the rules did not record.

    Records the replayed passes and the verifier turned away are in their own
    events. This one lists what was lost around them - a record in a window
    the current rules skip - and names each record revalidation admitted that
    the stored extraction had not, with the evidence method that placed it.
    Held statements are counted apart: finalization decides them.
    """

    old_entities = _entity_keys(before.entities)
    new_entities = _entity_keys(after.entities)
    old_relations = _relation_keys(before.relations, before.entities)
    new_relations = _relation_keys(after.relations, after.entities)
    old_held = _relation_keys(before.held_relations, before.entities)
    new_held = _relation_keys(after.held_relations, after.entities)
    admitted: list[dict[str, Any]] = [
        {
            "kind": "entity",
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "chunk_id": entity.source_chunk_id,
            "evidence_method": entity.evidence_method,
        }
        for entity in after.entities
        if _entity_key(entity) not in old_entities
    ]
    names = {entity.id: entity for entity in after.entities}
    for relation in after.relations:
        key = _relation_key(relation, names)
        if key not in old_relations:
            admitted.append(
                {
                    "kind": "relation",
                    "id": relation.id,
                    "name": relation.relation_type,
                    "source": key[0],
                    "target": key[1],
                    "chunk_id": relation.source_chunk_id,
                    "evidence_method": relation.evidence_method,
                }
            )
    return {
        "stage": REVALIDATION_STAGE,
        "document_id": document_id,
        "windows": windows,
        "entities": _changes(old_entities, new_entities),
        "relations": _changes(old_relations, new_relations),
        "held_relations": _changes(old_held, new_held),
        "reconstruction": reconstruction,
        "admitted_records": admitted[:_MAX_LISTED_RECORDS],
        "rejected_records": lost,
    }


def _changes(before: set[Any], after: set[Any]) -> dict[str, int]:
    """How many of a kind of observation were kept, admitted and dropped."""

    return {
        "before": len(before),
        "after": len(after),
        "kept": len(before & after),
        "admitted": len(after - before),
        "dropped": len(before - after),
    }


def _entity_key(entity: EntityMention) -> tuple[str, str, str]:
    """An entity observation's identity across two runs: name, type and chunk."""

    return normalize_ontology_label(entity.name), entity.type, entity.source_chunk_id


def _entity_keys(entities: Iterable[EntityMention]) -> set[tuple[str, str, str]]:
    """The identities of these entity observations."""

    return {_entity_key(entity) for entity in entities}


def _relation_key(
    relation: RelationObservation,
    entities: Mapping[str, EntityMention],
) -> tuple[str, str, str, str]:
    """A relation observation's identity across two runs, by its endpoints' names."""

    source = entities.get(relation.source_entity_id)
    target = entities.get(relation.target_entity_id)
    return (
        normalize_ontology_label(source.name if source else relation.source_surface or ""),
        normalize_ontology_label(target.name if target else relation.target_surface or ""),
        relation.relation_type,
        relation.source_chunk_id,
    )


def _relation_keys(
    relations: Iterable[RelationObservation],
    entities: Iterable[EntityMention],
) -> set[tuple[str, str, str, str]]:
    """The identities of these relation observations."""

    by_id = {entity.id: entity for entity in entities}
    return {_relation_key(relation, by_id) for relation in relations}
