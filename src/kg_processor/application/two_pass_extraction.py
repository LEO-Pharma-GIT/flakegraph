# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral orchestration for high-precision two-pass extraction."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import batched
from typing import Any

from kg_processor.application.bibliography import is_reference_text
from kg_processor.application.entity_resolution import resolve_entity_mentions
from kg_processor.application.extraction_grounding import (
    build_surface_text_index,
    indexed_surface_occurs,
    ocr_tolerant_surface_occurs,
    sentence_spans,
    surface_occurs,
    surface_spans,
)
from kg_processor.application.extraction_results import dedupe_extraction_result
from kg_processor.application.extraction_windows import (
    build_document_openings,
    build_extraction_windows,
    window_skip_reason,
)
from kg_processor.application.folder_entities import is_folder_mention
from kg_processor.application.llm_extractors import (
    LlmEntityExtractor,
    LlmRelationExtractor,
    LlmRelationVerifier,
    admitted_definition,
)
from kg_processor.application.ordered_map import ordered_map
from kg_processor.application.progress import error_metadata
from kg_processor.application.prompt_registry import TWO_PASS_PROMPT_REVISION
from kg_processor.application.redaction import redact_sensitive_text
from kg_processor.application.rejected_records import mention_record, observation_record
from kg_processor.application.relation_candidates import discover_cue_relation_candidates
from kg_processor.application.window_errors import is_systemic_provider_error
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.extraction import (
    ColumnRelation,
    EntityMention,
    EntityVerificationDecision,
    ExtractionObservations,
    ExtractionWindow,
    RelationExtractionOutcome,
    RelationObservation,
    VerificationDecision,
    VerificationOutcome,
)
from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractedRelation, ExtractionResult
from kg_processor.domain.ids import stable_id
from kg_processor.domain.ontology import OntologyProfile, normalize_ontology_label
from kg_processor.ports.embeddings import EmbeddingProvider, EmbedOptions
from kg_processor.ports.extraction import EntityExtractor, RelationExtractor, RelationVerifier
from kg_processor.ports.llm import LlmProvider, TruncatedChatResponseError

WindowProgressCallback = Callable[[int, int, int, int], None]
ResolutionProgressCallback = Callable[[int, int, int, int], None]
_MIN_RELATION_ENTITIES = 2
_MIN_COMPOUND_OCCURRENCES = 2
_RELATION_VERIFICATION_BATCH_SIZE = 32
# Entity verdicts are one short word each; a call judges up to this many.
_ENTITY_VERIFICATION_BATCH_SIZE = 40
# Evidence the model only pointed at is kept only when the verifier supports
# it, whether or not other relations are verified.
_ALWAYS_VERIFIED_METHODS = frozenset({"llm_located", "llm_verified"})


@dataclass(frozen=True)
class _WindowResult:
    """Bundle grounded observations and trace events produced for one window.

    Keeping window output self-contained allows concurrent execution while a
    later deterministic merge restores source order.
    """

    entities: list[EntityMention]
    relations: list[RelationObservation]
    trace: list[dict[str, object]]
    held_relations: list[RelationObservation] = field(default_factory=list)


# Given a document id, whether its windows get (entity, relation) extraction.
WindowFilter = Callable[[str], tuple[bool, bool]]


def extract_graph_two_pass(
    chunks: list[Chunk],
    llm: LlmProvider,
    embeddings: EmbeddingProvider,
    embed_options: EmbedOptions,
    graph_settings: GraphSettings,
    ontology: OntologyProfile,
    ontology_checksum: str,
    ontology_source: str,
    model: str,
    timeout_seconds: int,
    window_progress: WindowProgressCallback | None = None,
    resolution_progress: ResolutionProgressCallback | None = None,
    entity_extractor: EntityExtractor | None = None,
    relation_extractor: RelationExtractor | None = None,
    relation_verifier: RelationVerifier | None = None,
    document_context_entities: list[EntityMention] | None = None,
    document_context_trace: list[dict[str, Any]] | None = None,
    window_filter: WindowFilter | None = None,
    dataset_columns: Mapping[str, list[ColumnRelation]] | None = None,
) -> ExtractionResult:
    """Run the complete grounded two-pass extraction and identity-resolution flow.

    Document-scoped windows extract entities first, relations may reference only
    those accepted IDs, optional verification removes unsupported predicates, and
    conservative resolution maps mentions onto canonical names. The result is
    translated to the established extraction contract for downstream assembly.
    """

    observations = extract_graph_observations(
        chunks,
        llm,
        graph_settings,
        ontology,
        model,
        timeout_seconds,
        window_progress,
        entity_extractor=entity_extractor,
        relation_extractor=relation_extractor,
        relation_verifier=relation_verifier,
        document_context_entities=document_context_entities,
        window_filter=window_filter,
        dataset_columns=dataset_columns,
    )
    if document_context_trace:
        observations = observations.model_copy(
            update={"trace": [*document_context_trace, *observations.trace]}
        )
    return resolve_extraction_observations(
        observations,
        llm,
        embeddings,
        embed_options,
        graph_settings,
        ontology,
        ontology_checksum,
        ontology_source,
        model,
        timeout_seconds,
        resolution_progress,
    )


def extract_graph_observations(
    chunks: list[Chunk],
    llm: LlmProvider,
    graph_settings: GraphSettings,
    ontology: OntologyProfile,
    model: str,
    timeout_seconds: int,
    window_progress: WindowProgressCallback | None = None,
    entity_extractor: EntityExtractor | None = None,
    relation_extractor: RelationExtractor | None = None,
    relation_verifier: RelationVerifier | None = None,
    document_context_entities: list[EntityMention] | None = None,
    window_filter: WindowFilter | None = None,
    dataset_columns: Mapping[str, list[ColumnRelation]] | None = None,
) -> ExtractionObservations:
    """Extract document-wide entities before independently extracting relations.

    Every entity window completes before relation windows begin for its in-process
    shard. This true two-phase boundary lets a relation cite an endpoint discovered
    in another window without serializing model calls inside either phase. Durable
    workers expose the same two phases as separate queue waves.
    """

    entity_observations = extract_entity_observations(
        chunks,
        llm,
        graph_settings,
        ontology,
        model,
        timeout_seconds,
        entity_extractor=entity_extractor,
        document_context_entities=document_context_entities,
        window_progress=window_progress,
        window_filter=window_filter,
    )
    relation_observations = extract_relation_observations(
        chunks,
        entity_observations.entities,
        llm,
        graph_settings,
        ontology,
        model,
        timeout_seconds,
        window_progress=window_progress,
        relation_extractor=relation_extractor,
        relation_verifier=relation_verifier,
        window_filter=window_filter,
        dataset_columns=dataset_columns,
    )
    return apply_entity_verdicts(
        ExtractionObservations(
            entities=_dedupe_mentions(
                [*entity_observations.entities, *relation_observations.entities]
            ),
            relations=relation_observations.relations,
            held_relations=relation_observations.held_relations,
            trace=[*entity_observations.trace, *relation_observations.trace],
            chunk_count=len(chunks),
            window_count=entity_observations.window_count,
        ),
        ontology,
    )


def extract_entity_observations(
    chunks: list[Chunk],
    llm: LlmProvider,
    graph_settings: GraphSettings,
    ontology: OntologyProfile,
    model: str,
    timeout_seconds: int,
    window_progress: WindowProgressCallback | None = None,
    entity_extractor: EntityExtractor | None = None,
    document_context_entities: list[EntityMention] | None = None,
    window_filter: WindowFilter | None = None,
) -> ExtractionObservations:
    """Extract and deduplicate entity mentions without issuing relation calls.

    This function is the first parallel phase shared by local and distributed
    execution. Its output is compact enough to form one immutable per-document
    inventory before the second queue wave starts.
    """

    windows = [
        window
        for window in build_extraction_windows(
            chunks,
            graph_settings.extraction_window_tokens,
            graph_settings.max_chunks_per_llm_call,
        )
        if window_filter is None or window_filter(window.document_id)[0]
    ]
    entity_stage = entity_extractor or LlmEntityExtractor(llm, graph_settings.deterministic_seed)
    subjects_by_document = _mentions_by_document(chunks, document_context_entities or [])
    results = _run_window_stage(
        windows,
        graph_settings,
        lambda window: _extract_window_entities(
            window,
            graph_settings,
            ontology,
            model,
            timeout_seconds,
            entity_stage,
            subjects_by_document.get(window.document_id, []),
        ),
        window_progress,
    )
    return ExtractionObservations(
        entities=_dedupe_mentions([item for result in results for item in result.entities]),
        trace=[event for result in results for event in result.trace],
        chunk_count=len(chunks),
        window_count=len(windows),
    )


def extract_relation_observations(
    chunks: list[Chunk],
    document_entities: list[EntityMention],
    llm: LlmProvider,
    graph_settings: GraphSettings,
    ontology: OntologyProfile,
    model: str,
    timeout_seconds: int,
    window_progress: WindowProgressCallback | None = None,
    relation_extractor: RelationExtractor | None = None,
    relation_verifier: RelationVerifier | None = None,
    window_filter: WindowFilter | None = None,
    dataset_columns: Mapping[str, list[ColumnRelation]] | None = None,
    document_openings: dict[str, str] | None = None,
) -> ExtractionObservations:
    """Extract relations using the complete entity inventory for each document.

    Relation windows remain mutually independent and can be work-stolen across a
    fleet. The document-wide inventory only broadens valid endpoint choices; source
    grounding and verification continue to constrain every emitted observation.
    A catalogue's windows carry its column relations, by document id in
    ``dataset_columns``, so each row is read the way its header says. Openings
    default to those read from ``chunks``; a caller holding only some of a
    document's chunks passes the document's own.
    """

    windows = [
        window.model_copy(
            update={"dataset_columns": (dataset_columns or {}).get(window.document_id, [])}
        )
        for window in build_extraction_windows(
            chunks,
            graph_settings.extraction_window_tokens,
            graph_settings.max_chunks_per_llm_call,
        )
        if window_filter is None or window_filter(window.document_id)[1]
    ]
    relation_stage = relation_extractor or LlmRelationExtractor(
        llm, graph_settings.deterministic_seed
    )
    verifier_stage = relation_verifier or LlmRelationVerifier(
        llm, graph_settings.deterministic_seed
    )
    entities_by_document = _mentions_by_document(chunks, document_entities)
    openings = build_document_openings(chunks) if document_openings is None else document_openings
    results = _run_window_stage(
        windows,
        graph_settings,
        lambda window: _extract_window_relations(
            window,
            graph_settings,
            ontology,
            model,
            timeout_seconds,
            relation_stage,
            verifier_stage,
            entities_by_document.get(window.document_id, []),
            openings.get(window.document_id, ""),
        ),
        window_progress,
    )
    return ExtractionObservations(
        entities=_dedupe_mentions([item for result in results for item in result.entities]),
        relations=_dedupe_relations([item for result in results for item in result.relations]),
        held_relations=_dedupe_relations(
            [item for result in results for item in result.held_relations]
        ),
        trace=[event for result in results for event in result.trace],
        chunk_count=len(chunks),
        window_count=len(windows),
    )


def resolve_extraction_observations(
    observations: ExtractionObservations,
    llm: LlmProvider,
    embeddings: EmbeddingProvider,
    embed_options: EmbedOptions,
    graph_settings: GraphSettings,
    ontology: OntologyProfile,
    ontology_checksum: str,
    ontology_source: str,
    model: str,
    timeout_seconds: int,
    resolution_progress: ResolutionProgressCallback | None = None,
) -> ExtractionResult:
    """Resolve combined observation shards and produce the canonical extraction.

    Callers must combine all shards for one graph before invoking this barrier.
    Resolution decisions and provider traces are retained exactly as they are for
    local runs, making distributed execution auditable and semantically equivalent.
    """

    mentions = _dedupe_mentions(observations.entities)
    relations = _dedupe_relations(observations.relations)
    # Held relations are keyed by id, as the distributed finalizer keys them,
    # so both engines hold the same statements for the same shards.
    held_relations = list({item.id: item for item in observations.held_relations}.values())
    resolution = resolve_entity_mentions(
        mentions,
        enabled=graph_settings.entity_resolution_enabled,
        embeddings=embeddings,
        embed_options=embed_options,
        llm=llm,
        model=model,
        timeout_seconds=timeout_seconds,
        lexical_auto_merge=graph_settings.resolution_lexical_auto_merge,
        embedding_auto_merge=graph_settings.resolution_embedding_auto_merge,
        candidate_threshold=graph_settings.resolution_candidate_threshold,
        embedding_lexical_floor=graph_settings.resolution_embedding_lexical_floor,
        max_candidates_per_mention=graph_settings.resolution_max_candidates_per_mention,
        adjudication_batch_size=graph_settings.resolution_adjudication_batch_size,
        adjudication_parallelism=graph_settings.finalization_provider_concurrency,
        llm_merge_min_confidence=graph_settings.resolution_llm_merge_min_confidence,
        batch_progress=resolution_progress,
        seed=graph_settings.deterministic_seed,
    )
    extraction = _to_extraction_result(
        mentions,
        relations,
        resolution.canonical_name_by_mention_id,
        held_relations,
        component_ids=resolution.component_by_mention_id,
        ontology=ontology,
    )
    extraction.provider_metadata = {
        "schema_revision": TWO_PASS_PROMPT_REVISION,
        "strategy": "two_pass",
        "ontology_name": ontology.name,
        "ontology_source": ontology_source,
        "ontology_checksum": ontology_checksum,
        "chunk_count": observations.chunk_count,
        "batch_count": observations.window_count,
        "window_count": observations.window_count,
        "entity_mentions": len(mentions),
        "relation_observations": len(relations),
        "resolution_decisions": [
            decision.model_dump(mode="json") for decision in resolution.decisions
        ],
        "trace": [
            *observations.trace,
            resolution.trace,
        ],
    }
    return dedupe_extraction_result(extraction)


def _mentions_by_document(
    chunks: list[Chunk],
    mentions: list[EntityMention],
) -> dict[str, list[EntityMention]]:
    """Group mentions by the document of their source chunk.

    Distributed extraction sends one bounded window at a time. A mention grounded
    in a chunk outside the shard, such as a focal entity from front matter, has no
    document here; a single-document shard is unambiguous and inherits it, while
    multi-document callers keep the strict lookup so context never leaks across
    sources.
    """

    document_by_chunk = {chunk.id: chunk.document_id for chunk in chunks}
    document_ids = {chunk.document_id for chunk in chunks}
    by_document: dict[str, list[EntityMention]] = {}
    unassigned: list[EntityMention] = []
    for mention in mentions:
        document_id = document_by_chunk.get(mention.source_chunk_id)
        if document_id is None:
            unassigned.append(mention)
        else:
            by_document.setdefault(document_id, []).append(mention)
    if len(document_ids) == 1 and unassigned:
        by_document.setdefault(next(iter(document_ids)), []).extend(unassigned)
    return by_document


def _run_window_stage(
    windows: list[ExtractionWindow],
    settings: GraphSettings,
    operation: Callable[[ExtractionWindow], _WindowResult],
    progress: WindowProgressCallback | None,
) -> list[_WindowResult]:
    """Run one extraction phase concurrently while preserving source order.

    Progress callbacks follow actual completions, but indexed result collection
    ensures provider latency cannot change deduplication or persisted trace order.
    """

    failures: list[Exception] = []

    def contain(_index: int, window: ExtractionWindow, exc: Exception) -> _WindowResult:
        failures.append(exc)
        return _failed_window_result(window, exc)

    def report(completed: int, result: _WindowResult) -> None:
        if progress:
            progress(completed, len(windows), len(result.entities), len(result.relations))

    results = ordered_map(
        windows,
        lambda _index, window: operation(window),
        parallelism=settings.extraction_parallelism,
        on_complete=report,
        contain=contain,
    )
    if windows and len(failures) == len(windows):
        raise RuntimeError(_all_windows_failed_message(windows, failures)) from failures[0]
    return results


_WINDOW_FAILURE_DETAIL_LIMIT = 300


def _all_windows_failed_message(
    windows: Sequence[ExtractionWindow],
    failures: Sequence[Exception],
) -> str:
    """Name the first cause in the message, not only in the exception chain.

    A worker records this message and discards the chain, so a bare "all
    extraction windows failed" reaches the operator with nothing to act on: the
    provider answered, the pipeline rejected every record, and which of those
    happened is exactly what is missing. Carry a bounded, redacted summary of the
    first failure so the reason survives the trip through the task store.
    """

    first = failures[0]
    detail = redact_sensitive_text(str(first)).strip().replace("\n", " ")
    if len(detail) > _WINDOW_FAILURE_DETAIL_LIMIT:
        detail = detail[:_WINDOW_FAILURE_DETAIL_LIMIT] + "..."
    # A transport error says only that something refused a connection. Which
    # endpoint refused it is the entire question, and the exception carries it.
    target = _failed_request_target(first)
    location = f" while calling {target}" if target else ""
    return (
        f"all {len(windows)} extraction windows failed; "
        f"first cause{location}: {type(first).__name__}: {detail or '<no message>'}"
    )


def _failed_request_target(exc: BaseException) -> str:
    """Return the scheme, host, and path an HTTP failure was aimed at.

    Credentials and query strings are dropped: this ends up in a task record
    that operators read casually, and the useful part is the destination.
    """

    request = getattr(exc, "request", None)
    url = getattr(request, "url", None)
    if url is None:
        return ""
    host = getattr(url, "host", "") or ""
    port = getattr(url, "port", None)
    authority = f"{host}:{port}" if port else host
    return f"{getattr(url, 'scheme', '')}://{authority}{getattr(url, 'path', '')}"


def _failed_window_result(window: ExtractionWindow, exc: Exception) -> _WindowResult:
    """Contain one provider failure while retaining a redacted audit event."""

    return _WindowResult(
        entities=[],
        relations=[],
        trace=[
            {
                "stage": "window_error",
                "window_id": window.id,
                "document_id": window.document_id,
                "error": error_metadata(exc),
            }
        ],
    )


def _extract_window_entities(
    window: ExtractionWindow,
    settings: GraphSettings,
    ontology: OntologyProfile,
    model: str,
    timeout_seconds: int,
    entity_extractor: EntityExtractor,
    document_context_entities: list[EntityMention],
) -> _WindowResult:
    """Run entity extraction and completeness audits for one bounded window.

    Gleaning audits every substantive chunk instead of assuming that one accepted
    record makes a chunk complete. It stops on no output or saturation.
    """

    trace: list[dict[str, object]] = []

    skip_reason = window_skip_reason(window)
    if skip_reason is not None:
        return _WindowResult(
            entities=document_context_entities,
            relations=[],
            trace=[
                {
                    "stage": "reference_filter",
                    "window_id": window.id,
                    "skipped": True,
                    "reason": skip_reason,
                }
            ],
        )

    entity_outcome = entity_extractor.extract(
        window,
        ontology,
        model=model,
        timeout_seconds=timeout_seconds,
        max_entities=settings.max_entities_per_batch,
    )
    local_entities = entity_outcome.entities
    trace.append(entity_outcome.trace)
    for pass_index in range(1, settings.gleaning_max_passes + 1):
        audit_window = _entity_audit_window(
            window,
            settings.gleaning_min_uncovered_tokens,
            pass_index,
        )
        if audit_window is None:
            break
        try:
            entity_gleaned = entity_extractor.extract(
                audit_window,
                ontology,
                model=model,
                timeout_seconds=timeout_seconds,
                max_entities=settings.max_entities_per_batch,
                previous_entities=local_entities,
            )
        except Exception as exc:
            if is_systemic_provider_error(exc):
                raise
            trace.append(
                {
                    "stage": "entity_gleaning_error",
                    "window_id": window.id,
                    "gleaning_pass": pass_index,
                    "error": error_metadata(exc),
                }
            )
            break
        trace.append({**entity_gleaned.trace, "gleaning_pass": pass_index})
        if not entity_gleaned.entities:
            break
        local_entities = _dedupe_mentions([*local_entities, *entity_gleaned.entities])
        if len(entity_gleaned.entities) < settings.gleaning_saturation_threshold:
            break

    cue_entity_audit = _cue_entity_audit_window(window, local_entities, ontology)
    if cue_entity_audit is not None:
        try:
            entity_audited = entity_extractor.extract(
                cue_entity_audit,
                ontology,
                model=model,
                timeout_seconds=timeout_seconds,
                max_entities=settings.max_entities_per_batch,
                previous_entities=local_entities,
            )
        except Exception as exc:
            if is_systemic_provider_error(exc):
                raise
            trace.append(
                {
                    "stage": "cue_entity_audit_error",
                    "window_id": window.id,
                    "error": error_metadata(exc),
                }
            )
        else:
            trace.append({**entity_audited.trace, "cue_entity_audit": True})
            local_entities = _dedupe_mentions([*local_entities, *entity_audited.entities])

    entities = _dedupe_mentions([*document_context_entities, *local_entities])
    return _WindowResult(entities=entities, relations=[], trace=trace)


def _extract_window_relations(  # noqa: PLR0912
    window: ExtractionWindow,
    settings: GraphSettings,
    ontology: OntologyProfile,
    model: str,
    timeout_seconds: int,
    relation_extractor: RelationExtractor,
    verifier: RelationVerifier,
    entities: list[EntityMention],
    document_opening: str = "",
) -> _WindowResult:
    """Extract and verify relations against locally groundable document identities."""

    if window_skip_reason(window) is not None:
        return _WindowResult(entities=[], relations=[], trace=[])
    trace: list[dict[str, object]] = []
    # A folder entity is linked to the document's entities at finalization;
    # no statement of the text is offered it as an endpoint.
    entities = [entity for entity in entities if not is_folder_mention(entity)]

    # Compact repeated observations before selecting identities whose name, alias,
    # or approved context surface occurs in this window. An abbreviation learned
    # elsewhere remains usable, while impossible-to-ground remote endpoints never
    # consume prompt space or endpoint-pair enumeration.
    relation_entities = _relation_inventory_for_window(entities, window)
    trace.append(
        {
            "stage": "relation_inventory",
            "window_id": window.id,
            "observation_records": len(entities),
            "identity_records": len(relation_entities),
            "locally_grounded_records": len(relation_entities),
        }
    )

    # Endpoints a relation response added for things the inventory lacked.
    added: list[EntityMention] = []
    # Grounded statements a call rejected only for their entity types.
    held: list[RelationObservation] = []
    relations = _primary_relations(
        relation_extractor,
        window,
        relation_entities,
        ontology,
        settings,
        model=model,
        timeout_seconds=timeout_seconds,
        trace=trace,
        added=added,
        held=held,
    )
    for pass_index in range(1, settings.gleaning_max_passes + 1):
        audit_window = _relation_audit_window(
            window,
            entities,
            settings.gleaning_min_uncovered_tokens,
            pass_index,
        )
        if audit_window is None:
            break
        audit_entities = _relation_completion_inventory(
            audit_window,
            _relation_inventory_for_window(relation_entities, audit_window),
            ontology,
            relations,
        )
        try:
            relation_gleaned = relation_extractor.extract(
                audit_window,
                audit_entities,
                ontology,
                model=model,
                timeout_seconds=timeout_seconds,
                max_relations=settings.max_relations_per_batch,
                previous_relations=relations,
            )
        except Exception as exc:
            if is_systemic_provider_error(exc):
                raise
            trace.append(
                {
                    "stage": "relation_gleaning_error",
                    "window_id": window.id,
                    "gleaning_pass": pass_index,
                    "error": error_metadata(exc),
                }
            )
            break
        trace.append({**relation_gleaned.trace, "gleaning_pass": pass_index})
        added.extend(relation_gleaned.entities)
        held.extend(relation_gleaned.held_relations)
        if not relation_gleaned.relations:
            break
        relations = _dedupe_relations([*relations, *relation_gleaned.relations])
        if len(relation_gleaned.relations) < settings.gleaning_saturation_threshold:
            break

    cue_candidates = discover_cue_relation_candidates(window, relation_entities, ontology)
    relations_before_cue_audit = len(relations)
    relations = _dedupe_relations([*relations, *cue_candidates])
    trace.append(
        {
            "stage": "cue_candidate_generation",
            "window_id": window.id,
            "candidate_records": len(cue_candidates),
            "additional_records": len(relations) - relations_before_cue_audit,
            "direct_evidence_records": sum(
                not relation.verification_required for relation in cue_candidates
            ),
        }
    )

    relations = _verify_window(
        window,
        settings,
        ontology,
        verifier,
        model=model,
        timeout_seconds=timeout_seconds,
        document_entities=entities,
        added=added,
        context=_dedupe_mentions([*relation_entities, *added]),
        relations=relations,
        document_opening=document_opening,
        trace=trace,
    )
    held = _verify_held(
        window,
        settings,
        ontology,
        verifier,
        model=model,
        timeout_seconds=timeout_seconds,
        context=_dedupe_mentions([*relation_entities, *added]),
        held=_dedupe_relations(held),
        document_opening=document_opening,
        trace=trace,
    )
    return _WindowResult(
        entities=_dedupe_mentions(added), relations=relations, trace=trace, held_relations=held
    )


def _verify_window(  # noqa: PLR0913 - one window's verification context.
    window: ExtractionWindow,
    settings: GraphSettings,
    ontology: OntologyProfile,
    verifier: RelationVerifier,
    *,
    model: str,
    timeout_seconds: int,
    document_entities: list[EntityMention],
    added: list[EntityMention],
    context: list[EntityMention],
    relations: list[RelationObservation],
    document_opening: str,
    trace: list[dict[str, object]],
) -> list[RelationObservation]:
    """Judge a window's relations and its own entities, and keep what is supported.

    Each call carries one batch of relations and one of entities, with the
    document's opening and subjects. Every relation turned away - unsupported,
    below the confidence floor, or in a call that failed - is recorded with its
    reason. A rejected entity is recorded and listed by id, and a retyped one
    listed with both types: relations in any window may use it, so document
    combination applies both (``apply_entity_verdicts``).
    """

    to_verify = [
        relation
        for relation in relations
        if relation.verification_required
        and (settings.verify_relations or relation.evidence_method in _ALWAYS_VERIFIED_METHODS)
    ]
    judged_by_id = (
        _entities_to_judge(window, [*document_entities, *added]) if settings.verify_entities else {}
    )
    if not to_verify and not judged_by_id:
        return relations
    mentions_by_id = {mention.id: mention for mention in [*document_entities, *added]}
    subjects = sorted(
        {
            (mention.name, mention.type): mention
            for mention in document_entities
            if mention.is_document_subject
        }.values(),
        key=lambda mention: (mention.name, mention.type),
    )
    relation_batches = [
        list(batch) for batch in batched(to_verify, _RELATION_VERIFICATION_BATCH_SIZE)
    ]
    entity_batches = [
        list(batch)
        for batch in batched(
            [mentions[0] for mentions in judged_by_id.values()], _ENTITY_VERIFICATION_BATCH_SIZE
        )
    ]
    # Both lists share the calls: the longer one sets how many there are.
    batch_count = max(len(relation_batches), len(entity_batches))
    relation_batches += [[]] * (batch_count - len(relation_batches))
    entity_batches += [[]] * (batch_count - len(entity_batches))
    kept: dict[str, float] = {}

    def verify(
        relation_batch: list[RelationObservation], entity_batch: list[EntityMention]
    ) -> VerificationOutcome:
        return verifier.verify(
            window,
            context,
            relation_batch,
            ontology,
            model=model,
            timeout_seconds=timeout_seconds,
            document_opening=document_opening,
            document_subjects=subjects,
            entities_to_verify=entity_batch,
        )

    def failed(
        relation_batch: list[RelationObservation],
        entity_batch: list[EntityMention],
        exc: Exception,
    ) -> None:
        # A relation that needed a verdict and got none is not kept; an
        # entity that got none keeps what grounding already gave it.
        trace.append(
            {
                "stage": "relation_verification_error",
                "window_id": window.id,
                "document_id": window.document_id,
                "verification_batch_size": len(relation_batch),
                "entity_batch_size": len(entity_batch),
                "error": error_metadata(exc),
                "rejected_records": [
                    observation_record(relation, mentions_by_id, "verification_error")
                    for relation in relation_batch
                ],
            }
        )

    verified = _verify_batches(
        verify, list(zip(relation_batches, entity_batches, strict=True)), failed, trace, window
    )
    for batch_index, (relation_batch, _entities, verification) in enumerate(verified, start=1):
        decisions = {decision.relation_id: decision for decision in verification.decisions}
        rejected: list[dict[str, object]] = []
        for relation in relation_batch:
            decision = decisions.get(relation.id)
            reason = _verification_rejection(decision, settings.verification_min_confidence)
            if reason is not None:
                rejected.append(observation_record(relation, mentions_by_id, reason))
            elif decision is not None:
                kept[relation.id] = min(relation.confidence, decision.confidence)
        entity_verdicts, retyped = _entity_outcomes(
            verification.entity_decisions, judged_by_id, ontology
        )
        actions = dict(verification.trace.get("record_actions") or {})
        if retyped:
            actions["retyped_entity"] = len(retyped)
        trace.append(
            {
                **verification.trace,
                "verification_batch": batch_index,
                "verification_batch_size": len(relation_batch),
                "record_actions": dict(sorted(actions.items())),
                "rejected_records": rejected,
                "entity_verdicts": entity_verdicts,
                "retyped_entities": retyped,
            }
        )
    verified_ids = {relation.id for relation in to_verify}
    return [
        relation.model_copy(update={"confidence": kept[relation.id]})
        if relation.id in kept
        else relation
        for relation in relations
        if relation.id not in verified_ids or relation.id in kept
    ]


def _verify_batches(
    verify: Callable[[list[RelationObservation], list[EntityMention]], VerificationOutcome],
    batches: list[tuple[list[RelationObservation], list[EntityMention]]],
    failed: Callable[[list[RelationObservation], list[EntityMention], Exception], None],
    trace: list[dict[str, object]],
    window: ExtractionWindow,
) -> list[tuple[list[RelationObservation], list[EntityMention], VerificationOutcome]]:
    """Verify each batch, asking again for what a reply did not decide.

    Every decision a batch asks for has to fit one reply, and an identical
    request is cut off identically, so a cut-off batch is split - its relations
    and entities together - down to single items before any of it is given up.
    A reply that leaves relations or entities out is asked once more for just
    those; what a second reply also leaves out stays undecided.
    """

    done: list[tuple[list[RelationObservation], list[EntityMention], VerificationOutcome]] = []
    asked_again: set[str] = set()
    pending = list(reversed(batches))
    while pending:
        relation_batch, entity_batch = pending.pop()
        try:
            outcome = verify(relation_batch, entity_batch)
        except TruncatedChatResponseError as exc:
            halves = _halves(relation_batch, entity_batch)
            if halves is None:
                failed(relation_batch, entity_batch, exc)
                continue
            trace.append(
                _batch_event("relation_verification_split", window, relation_batch, entity_batch)
            )
            pending.extend(reversed(halves))
            continue
        except Exception as exc:
            if is_systemic_provider_error(exc):
                raise
            failed(relation_batch, entity_batch, exc)
            continue
        # A relation the reply skipped comes back as an "omitted" stand-in.
        decided = {
            decision.relation_id for decision in outcome.decisions if decision.reason != "omitted"
        } | {decision.entity_id for decision in outcome.entity_decisions}
        missing_relations = [
            item for item in relation_batch if item.id not in decided and item.id not in asked_again
        ]
        missing_entities = [
            item for item in entity_batch if item.id not in decided and item.id not in asked_again
        ]
        if missing_relations or missing_entities:
            asked_again.update(item.id for item in missing_relations)
            asked_again.update(item.id for item in missing_entities)
            trace.append(
                _batch_event(
                    "relation_verification_reask", window, missing_relations, missing_entities
                )
            )
            pending.append((missing_relations, missing_entities))
            relation_batch = [item for item in relation_batch if item not in missing_relations]
            entity_batch = [item for item in entity_batch if item not in missing_entities]
        done.append((relation_batch, entity_batch, outcome))
    return done


def _halves(
    relations: list[RelationObservation], entities: list[EntityMention]
) -> list[tuple[list[RelationObservation], list[EntityMention]]] | None:
    """A batch's relations and entities in two parts, or None when it is one item."""

    items: list[RelationObservation | EntityMention] = [*relations, *entities]
    if len(items) < _SPLITTABLE_BATCH:
        return None
    half = len(items) // 2
    return [
        (
            [item for item in part if isinstance(item, RelationObservation)],
            [item for item in part if isinstance(item, EntityMention)],
        )
        for part in (items[:half], items[half:])
    ]


def _batch_event(
    stage: str,
    window: ExtractionWindow,
    relations: list[RelationObservation],
    entities: list[EntityMention],
) -> dict[str, object]:
    return {
        "stage": stage,
        "window_id": window.id,
        "document_id": window.document_id,
        "verification_batch_size": len(relations),
        "entity_batch_size": len(entities),
    }


# A batch of fewer items than this cannot be asked for in parts.
_SPLITTABLE_BATCH = 2


def _entities_to_judge(
    window: ExtractionWindow,
    mentions: list[EntityMention],
) -> dict[str, list[EntityMention]]:
    """The window's own mentions to judge, by the id of the one that is shown.

    One verdict per name and type in the window applies to every mention of
    that name here. Document-context mentions are the document's anchors,
    found by their own pass, and are not judged.
    """

    window_chunk_ids = {chunk.id for chunk in window.chunks}
    by_name: dict[tuple[str, str], list[EntityMention]] = {}
    for mention in _dedupe_mentions(mentions):
        if mention.source_chunk_id in window_chunk_ids and not mention.is_document_context:
            key = (normalize_ontology_label(mention.name), mention.type)
            by_name.setdefault(key, []).append(mention)
    return {group[0].id: group for group in by_name.values()}


def _verification_rejection(
    decision: VerificationDecision | None,
    min_confidence: float,
) -> str | None:
    """The reason a relation is turned away by its verdict, or None when it is kept.

    The verifier names which rule of evidence an unsupported relation fails;
    without a named rule the verdict itself is the reason.
    """

    if decision is None:
        return "verification_omitted"
    if decision.verdict == "supported":
        return None if decision.confidence >= min_confidence else "verification_low_confidence"
    if decision.reason != "none":
        return f"verification_{decision.reason}"
    return f"verification_{decision.verdict}"


def _entity_outcomes(
    decisions: list[EntityVerificationDecision],
    judged_by_id: dict[str, list[EntityMention]],
    ontology: OntologyProfile,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """The verdicts to apply per document, and the mentions to retype.

    Verdicts are applied once a document's windows are combined, where every
    relation a mention takes part in is known. A real thing under the wrong type
    is a fact to keep; a verdict naming no other configured type changes nothing.
    """

    verdicts: list[dict[str, str]] = []
    retyped: list[dict[str, str]] = []
    for decision in decisions:
        for mention in judged_by_id.get(decision.entity_id, []):
            if decision.verdict == "specific":
                continue
            if decision.verdict == "wrong_type":
                if _retype_to(decision, mention, ontology):
                    retyped.append(
                        {"mention_id": mention.id, "from": mention.type, "to": decision.type}
                    )
                continue
            verdicts.append({"mention_id": mention.id, "verdict": decision.verdict})
    return verdicts, retyped


# Verdicts a mention outlives by being an end of a relation the document states.
_RELATION_ANCHORED_VERDICTS = frozenset({"generic"})


def _retype_to(
    decision: EntityVerificationDecision,
    mention: EntityMention,
    ontology: OntologyProfile,
) -> bool:
    """Whether a wrong_type verdict names another ontology type to keep the mention under."""

    return (
        decision.verdict == "wrong_type"
        and decision.type != mention.type
        and decision.type in ontology.entity_type_names()
    )


def apply_entity_verdicts(
    observations: ExtractionObservations,
    ontology: OntologyProfile,
) -> ExtractionObservations:
    """Retype and remove mentions as verification judged them, and re-check their relations.

    A window judges only its own mentions, but a relation in any window of the
    document may use one, so the verdicts apply once a document's windows are
    combined. A mention judged generic that is an end of a relation the
    document states keeps its node: the class the source relates something to
    is the fact's end. Every other mention judged not specific is removed with
    its relations. A relation on a retyped mention must still meet its type
    rules under the new type - through a rewrite or the fallback relation when
    its own no longer admits it - or it is rejected. Every mention and relation
    that falls away is recorded.
    """

    verdicts: dict[str, str] = {}
    new_types: dict[str, str] = {}
    for event in observations.trace:
        if event.get("stage") != "relation_verification":
            continue
        for item in event.get("entity_verdicts") or []:
            verdicts[str(item["mention_id"])] = str(item["verdict"])
        for item in event.get("retyped_entities") or []:
            new_types[str(item["mention_id"])] = str(item["to"])
    mentions_by_id = {mention.id: mention for mention in observations.entities}
    related = {
        endpoint
        for relation in observations.relations
        for endpoint in (relation.source_entity_id, relation.target_entity_id)
    }
    anchoring = {
        mention_id
        for mention_id, verdict in verdicts.items()
        if verdict in _RELATION_ANCHORED_VERDICTS and mention_id in related
    }
    rejected_ids = set(verdicts) - anchoring
    retyped = {
        mention_id: mentions_by_id[mention_id].model_copy(
            update={"type": to, "proposed_type": mentions_by_id[mention_id].type}
        )
        for mention_id, to in new_types.items()
        if mention_id in mentions_by_id
        and mention_id not in rejected_ids
        and mentions_by_id[mention_id].type != to
    }
    entities = [
        retyped.get(mention.id, mention)
        for mention in observations.entities
        if mention.id not in rejected_ids
    ]
    if len(entities) == len(observations.entities) and not retyped:
        return observations
    current = {mention.id: mention for mention in entities}
    reasons: Counter[str] = Counter()
    if anchoring & mentions_by_id.keys():
        reasons["kept_relation_end"] = len(anchoring & mentions_by_id.keys())
    kept: list[RelationObservation] = []
    records: list[dict[str, object]] = [
        mention_record(mention, f"verification_{verdicts[mention.id]}")
        for mention in observations.entities
        if mention.id in rejected_ids
    ]
    for relation in observations.relations:
        if {relation.source_entity_id, relation.target_entity_id} & rejected_ids:
            records.append(
                observation_record(relation, mentions_by_id, "verification_endpoint_rejected")
            )
            continue
        if not {relation.source_entity_id, relation.target_entity_id} & retyped.keys():
            kept.append(relation)
            continue
        rechecked = _recheck_relation(relation, current, ontology, reasons)
        if rechecked is None:
            records.append(observation_record(relation, current, "domain_or_range_violation"))
        else:
            kept.append(rechecked)
    cascade = {
        "stage": "verification_cascade",
        "rejected_entities": len(observations.entities) - len(entities),
        "retyped_entities": len(retyped),
        "record_actions": dict(sorted(reasons.items())),
        "rejected_records": records,
    }
    # A held statement on a rejected mention falls away with it; its rejected
    # record already stands. One on a retyped mention stays held: finalization
    # decides its types from the whole corpus.
    held = [
        relation
        for relation in observations.held_relations
        if not {relation.source_entity_id, relation.target_entity_id} & rejected_ids
    ]
    return observations.model_copy(
        update={
            "entities": entities,
            "relations": kept,
            "held_relations": held,
            "trace": [*observations.trace, cascade],
        }
    )


def _recheck_relation(
    relation: RelationObservation,
    mentions_by_id: dict[str, EntityMention],
    ontology: OntologyProfile,
    reasons: Counter[str],
) -> RelationObservation | None:
    """A relation on a retyped mention, kept under the type its ends now admit, or None.

    The relation's type rules are applied as extraction applies them: its own
    type, else a rewrite, else the fallback relation. A relation type outside
    the ontology has no rules to break.
    """

    source = mentions_by_id.get(relation.source_entity_id)
    target = mentions_by_id.get(relation.target_entity_id)
    definition = ontology.relation(relation.relation_type)
    if source is None or target is None or definition is None:
        return relation
    admitted, stated = admitted_definition(definition, source, target, ontology, reasons)
    if admitted is None:
        return None
    if stated is None:
        return relation
    return relation.model_copy(
        update={
            "relation_type": admitted.name,
            "proposed_relation_type": relation.proposed_relation_type or stated,
        }
    )


def _verify_held(  # noqa: PLR0913 - one window's verification context.
    window: ExtractionWindow,
    settings: GraphSettings,
    ontology: OntologyProfile,
    verifier: RelationVerifier,
    *,
    model: str,
    timeout_seconds: int,
    context: list[EntityMention],
    held: list[RelationObservation],
    document_opening: str,
    trace: list[dict[str, object]],
) -> list[RelationObservation]:
    """Keep the held statements the verifier finds stated, judged apart from their types.

    A held statement broke its relation's type rules in this window, and
    finalization decides the types from the whole corpus, so the verifier is
    shown the relations without type rules and judges only whether the text
    states the statement. Held statements are judged in batches of their own
    and none counts among the verifier's verdicts. One turned away already has
    its rejected record, so it is only dropped.
    """

    to_verify = [relation for relation in held if relation.verification_required]
    if not to_verify or not settings.verify_relations:
        return held
    untyped = ontology.model_copy(
        update={
            "relation_types": [
                definition.model_copy(update={"source_types": [], "target_types": []})
                for definition in ontology.relation_types
            ]
        }
    )
    kept: dict[str, float] = {}
    for batch_index, batch in enumerate(
        (list(batch) for batch in batched(to_verify, _RELATION_VERIFICATION_BATCH_SIZE)),
        start=1,
    ):
        try:
            verification = verifier.verify(
                window,
                context,
                batch,
                untyped,
                model=model,
                timeout_seconds=timeout_seconds,
                document_opening=document_opening,
                document_subjects=[],
                entities_to_verify=[],
            )
        except Exception as exc:
            if is_systemic_provider_error(exc):
                raise
            trace.append(
                {
                    "stage": "held_relation_verification_error",
                    "window_id": window.id,
                    "verification_batch": batch_index,
                    "verification_batch_size": len(batch),
                    "error": error_metadata(exc),
                }
            )
            continue
        for decision in verification.decisions:
            if _verification_rejection(decision, settings.verification_min_confidence) is None:
                kept[decision.relation_id] = decision.confidence
        trace.append(
            {
                **verification.trace,
                "stage": "held_relation_verification",
                "verification_batch": batch_index,
                "verification_batch_size": len(batch),
                "rejected_records": [],
            }
        )
    verified_ids = {relation.id for relation in to_verify}
    return [
        relation.model_copy(update={"confidence": min(relation.confidence, kept[relation.id])})
        if relation.id in kept
        else relation
        for relation in held
        if relation.id not in verified_ids or relation.id in kept
    ]


def _primary_relations(
    relation_extractor: RelationExtractor,
    window: ExtractionWindow,
    relation_entities: list[EntityMention],
    ontology: OntologyProfile,
    settings: GraphSettings,
    *,
    model: str,
    timeout_seconds: int,
    trace: list[dict[str, object]],
    added: list[EntityMention],
    held: list[RelationObservation],
) -> list[RelationObservation]:
    """Ask for a window's relations: one call, a retry if it came back empty
    over a large inventory, and continuations while calls fill their limit.

    Endpoints any call adds are appended to ``added`` and offered to the next;
    relations any call holds for their entity types are appended to ``held``.
    """

    relation_outcome = relation_extractor.extract(
        window,
        relation_entities,
        ontology,
        model=model,
        timeout_seconds=timeout_seconds,
        max_relations=settings.max_relations_per_batch,
    )
    trace.append(relation_outcome.trace)
    added.extend(relation_outcome.entities)
    if (
        settings.relation_empty_retry_min_entities
        and len(relation_entities) >= settings.relation_empty_retry_min_entities
        and _proposed_records(relation_outcome.trace) == 0
        and not relation_outcome.trace.get("skipped")
    ):
        # An empty answer over a large inventory is as often a sampling accident
        # as a window that states nothing; one retry with another seed is cheap
        # because an empty answer comes back in seconds.
        retried = _optional_relation_pass(
            relation_extractor,
            window,
            relation_entities,
            ontology,
            model=model,
            timeout_seconds=timeout_seconds,
            max_relations=settings.max_relations_per_batch,
            seed=settings.deterministic_seed + 1,
            label={"empty_retry": True},
            trace=trace,
        )
        if retried is not None:
            relation_outcome = retried
            added.extend(retried.entities)
    relations = relation_outcome.relations
    held.extend(relation_outcome.held_relations)
    last_outcome = relation_outcome
    for continuation in range(1, settings.relation_continuation_max_passes + 1):
        # A call that filled its record limit was cut off, not finished.
        if _proposed_records(last_outcome.trace) < settings.max_relations_per_batch:
            break
        continued = _optional_relation_pass(
            relation_extractor,
            window,
            _dedupe_mentions([*relation_entities, *added]),
            ontology,
            model=model,
            timeout_seconds=timeout_seconds,
            max_relations=settings.max_relations_per_batch,
            previous_relations=relations,
            label={"continuation_pass": continuation},
            trace=trace,
        )
        if continued is not None:
            added.extend(continued.entities)
            held.extend(continued.held_relations)
        if continued is None or not continued.relations:
            break
        relations = _dedupe_relations([*relations, *continued.relations])
        last_outcome = continued
    return relations


def _proposed_records(trace: dict[str, object]) -> int:
    """How many records the model returned in one call, before validation."""

    value = trace.get("input_records")
    return value if isinstance(value, int) else 0


def _optional_relation_pass(
    relation_extractor: RelationExtractor,
    window: ExtractionWindow,
    entities: list[EntityMention],
    ontology: OntologyProfile,
    *,
    model: str,
    timeout_seconds: int,
    max_relations: int,
    label: dict[str, object],
    trace: list[dict[str, object]],
    previous_relations: list[RelationObservation] | None = None,
    seed: int | None = None,
) -> RelationExtractionOutcome | None:
    """Run one extra relation call whose failure costs only its own records.

    The window already has an answer; a provider error here is traced and the
    window keeps what it had, unless the error is systemic.
    """

    try:
        outcome = relation_extractor.extract(
            window,
            entities,
            ontology,
            model=model,
            timeout_seconds=timeout_seconds,
            max_relations=max_relations,
            previous_relations=previous_relations,
            seed=seed,
        )
    except Exception as exc:
        if is_systemic_provider_error(exc):
            raise
        trace.append(
            {
                "stage": "relation_extraction_error",
                "window_id": window.id,
                **label,
                "error": error_metadata(exc),
            }
        )
        return None
    trace.append({**outcome.trace, **label})
    return outcome


def is_reference_only_window(window: ExtractionWindow) -> bool:
    """Identify bibliography-only windows without interpreting source claims.

    PDF chunks can begin mid-bibliography after a heading fell into the previous
    window. Requiring several bracketed or author-style numbered entries avoids
    classifying ordinary prose with one citation as references. Skipping these
    windows prevents citation metadata from becoming entities or unsupported
    ``BUILDS_ON`` relations and avoids unnecessary provider calls.
    """

    return is_reference_text("\n".join(chunk.content for chunk in window.chunks))


def _entity_audit_window(
    window: ExtractionWindow,
    minimum_tokens: int,
    pass_index: int,
) -> ExtractionWindow | None:
    """Build a gleaning subwindow containing every substantive source chunk.

    A chunk that yielded one entity may still contain several missed entities, so
    record presence is not a valid completeness signal. Very small chunks remain
    excluded because they rarely justify another provider call and can encourage
    duplicate low-context observations.
    """

    chunks = [chunk for chunk in window.chunks if chunk.token_count >= minimum_tokens]
    return _subwindow(window, chunks, "entity_glean", pass_index)


def _cue_entity_audit_window(
    window: ExtractionWindow,
    entities: list[EntityMention],
    ontology: OntologyProfile,
) -> ExtractionWindow | None:
    """Focus one entity audit on relation sentences with incomplete endpoints.

    Relation extraction cannot recover a fact whose compound subject was shortened
    to a different known entity. This audit selects only cue-bearing sentences with
    fewer than two grounded entities or a grounded surface that appears to continue
    into a longer lowercase noun phrase. Original chunk IDs are retained so any new
    mention still points to valid source provenance.
    """

    selected_by_chunk: dict[str, list[str]] = {}
    window_text = "\n".join(chunk.content for chunk in window.chunks)
    for chunk in window.chunks:
        for sentence in sentence_spans(chunk.content):
            cue_starts = [
                match.start()
                for definition in ontology.relation_types
                for cue in definition.evidence_cues
                for match in re.finditer(cue, sentence.quote, flags=re.IGNORECASE)
            ]
            if not cue_starts:
                continue
            grounded_entities = [
                entity
                for entity in entities
                if any(
                    surface_spans(sentence.quote, [surface])
                    for surface in [entity.name, *entity.aliases]
                )
            ]
            occurrences = [
                (start, end)
                for entity in grounded_entities
                for surface in [entity.name, *entity.aliases]
                for start, end in surface_spans(sentence.quote, [surface])
            ]
            incomplete_endpoints = (
                len({entity.id for entity in grounded_entities}) < _MIN_RELATION_ENTITIES
            )
            partial_endpoint = any(
                _surface_continues_as_repeated_noun_phrase(
                    sentence.quote,
                    start,
                    end,
                    cue_starts,
                    window_text,
                )
                for start, end in occurrences
            )
            if incomplete_endpoints or partial_endpoint:
                selected_by_chunk.setdefault(chunk.id, []).append(sentence.quote)

    # Keep the original chunk content and offsets. A targeted-sentence copy under
    # the source chunk ID would reset text coordinates, and otherwise valid quotes
    # would point at unrelated source spans after publication.
    chunks = [chunk for chunk in window.chunks if selected_by_chunk.get(chunk.id)]
    if not chunks:
        return None
    return ExtractionWindow(
        id=stable_id("cue_entity_audit_window", window.id),
        document_id=window.document_id,
        chunks=chunks,
        token_count=sum(chunk.token_count for chunk in chunks),
    )


def _surface_continues_as_repeated_noun_phrase(
    sentence: str,
    surface_start: int,
    surface_end: int,
    cue_starts: list[int],
    window_text: str,
) -> bool:
    """Detect a recurring compound that extends a known relation subject surface."""

    following = re.match(r"\s+([a-z][\w-]*)", sentence[surface_end:])
    if following is None:
        return False
    token = following.group(1).casefold()
    grammatical_words = {
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
    }
    next_start = surface_end + following.start(1)
    if token in grammatical_words or not any(next_start < cue for cue in cue_starts):
        return False
    compound = sentence[surface_start:surface_end] + following.group(0)
    repeated_compound = (
        len(
            re.findall(
                rf"(?<!\w){re.escape(compound)}(?!\w)",
                window_text,
                flags=re.IGNORECASE,
            )
        )
        >= _MIN_COMPOUND_OCCURRENCES
    )
    repeated_head_pattern = (
        rf"(?<!\w)(?:a|an|the)\s+{re.escape(token)}(?!\w)"
        + r"|['\u2019]s\s+"
        + re.escape(token)
        + r"(?!\w)"
    )
    repeated_head = re.search(
        repeated_head_pattern,
        window_text,
        flags=re.IGNORECASE,
    )
    return repeated_compound or repeated_head is not None


def _relation_audit_window(
    window: ExtractionWindow,
    entities: list[EntityMention],
    minimum_tokens: int,
    pass_index: int,
) -> ExtractionWindow | None:
    """Build a relation audit from chunks containing two grounded entity mentions.

    Existing relations do not make a chunk complete: one sentence can state many
    independent facts. Requiring two text-grounded entities avoids provider calls
    for chunks that cannot produce a valid non-self relation, while occurrence
    matching includes entities first recognized in an overlapping neighbor chunk.
    """

    chunks = [
        chunk
        for chunk in window.chunks
        if chunk.token_count >= minimum_tokens
        and len(_entities_grounded_in_text(entities, chunk.content)) >= _MIN_RELATION_ENTITIES
    ]
    return _subwindow(window, chunks, "relation_glean", pass_index)


def _entities_grounded_in_window(
    entities: list[EntityMention], window: ExtractionWindow
) -> list[EntityMention]:
    """Return inventory entries whose names or aliases occur in an audit window.

    Selecting by source provenance alone loses endpoints that were first extracted
    from an overlapping chunk but are also stated in the audited text. Exact surface
    occurrence keeps the broader inventory grounded without exposing unrelated
    document entities to the relation model. A name that occurs only as OCR
    printed it - the way an ``ocr_tolerant`` mention was grounded - counts too,
    so such an entity can still take part in relations.
    """

    return [
        entity
        for entity in entities
        if entity.is_document_context
        or any(
            surface_occurs(chunk.content, [entity.name, *entity.aliases]) for chunk in window.chunks
        )
        or any(
            ocr_tolerant_surface_occurs(chunk.content, [entity.name, *entity.aliases])
            for chunk in window.chunks
        )
    ]


def _relation_inventory_for_window(
    entities: list[EntityMention], window: ExtractionWindow
) -> list[EntityMention]:
    """Return the compact set of identities groundable in one relation window.

    Entity extraction deliberately retains repeated source observations for
    provenance and later identity resolution. Relation generation needs identity
    choices rather than every source span, so exposing all repeated mentions adds
    tokens and lets the provider select among multiple IDs with the same name.

    Exact normalized name and ontology type are a conservative identity key. The
    current window's mention supplies the endpoint ID when one exists, preserving
    repeated evidence from different chunks; otherwise the first stable mention is
    used. Aliases, document-context surfaces, and the context flag are unioned
    across observations. An identity established in a different window remains
    available when its name, alias, or approved context surface occurs here.
    Identities with no local surface are omitted because grounding cannot accept
    them; sending them only adds prompt tokens and quadratic endpoint pairs.
    """

    grouped: dict[tuple[str, str], list[EntityMention]] = {}
    for entity in entities:
        if is_folder_mention(entity):
            continue
        grouped.setdefault((normalize_ontology_label(entity.name), entity.type), []).append(entity)

    window_chunk_ids = {chunk.id for chunk in window.chunks}
    inventory: list[EntityMention] = []
    for observations in grouped.values():
        representative = next(
            (entity for entity in observations if entity.source_chunk_id in window_chunk_ids),
            observations[0],
        )
        aliases = list(dict.fromkeys(alias for entity in observations for alias in entity.aliases))
        contextual_surfaces = list(
            dict.fromkeys(
                surface for entity in observations for surface in entity.contextual_surfaces
            )
        )
        inventory.append(
            representative.model_copy(
                update={
                    "aliases": aliases,
                    "is_document_context": any(
                        entity.is_document_context for entity in observations
                    ),
                    "is_document_subject": any(
                        entity.is_document_subject for entity in observations
                    ),
                    "contextual_surfaces": contextual_surfaces,
                }
            )
        )
    return _entities_grounded_in_window(inventory, window)


def _relation_completion_inventory(
    window: ExtractionWindow,
    entities: list[EntityMention],
    ontology: OntologyProfile,
    previous_relations: list[RelationObservation],
) -> list[EntityMention]:
    """Focus a gleaning pass on uncovered cue-compatible endpoint pairs.

    The initial relation pass considers every locally groundable pair. A second
    broad request tends to repeat salient facts, so this completion inventory keeps
    only endpoints participating in a still-uncovered typed pair for which the
    source chunk contains one of that predicate's configured evidence cues. The
    LLM still receives the original chunk and must emit exact endpoint surfaces;
    ordinary grounding and semantic verification remain unchanged.

    Definitions without lexical cues remain eligible in every chunk. Otherwise,
    the presence of an unrelated cue-backed pair could exclude all endpoints for
    layout-defined relations such as publication authorship or affiliation. If no
    guided pair can be formed, the complete local inventory is returned. This
    preserves semantic coverage while keeping common completion requests smaller.
    """

    covered_targets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    covered_sources: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for relation in previous_relations:
        covered_targets[
            (relation.source_chunk_id, relation.relation_type, relation.source_entity_id)
        ].add(relation.target_entity_id)
        covered_sources[
            (relation.source_chunk_id, relation.relation_type, relation.target_entity_id)
        ].add(relation.source_entity_id)
    selected_ids: set[str] = set()
    for chunk in window.chunks:
        local_entities = _entities_grounded_in_text(entities, chunk.content)
        cue_definitions = [
            definition
            for definition in ontology.relation_types
            if not definition.evidence_cues
            or any(
                re.search(cue, chunk.content, flags=re.IGNORECASE)
                for cue in definition.evidence_cues
            )
        ]
        for definition in cue_definitions:
            sources = [
                entity
                for entity in local_entities
                if not definition.source_types or entity.type in definition.source_types
            ]
            targets = [
                entity
                for entity in local_entities
                if not definition.target_types or entity.type in definition.target_types
            ]
            source_ids = {entity.id for entity in sources}
            target_ids = {entity.id for entity in targets}
            for source in sources:
                excluded_self = not definition.allow_self_loop and source.id in target_ids
                candidate_count = len(target_ids) - int(excluded_self)
                blocked_count = sum(
                    target_id in target_ids
                    and (definition.allow_self_loop or target_id != source.id)
                    for target_id in covered_targets[(chunk.id, definition.name, source.id)]
                )
                if candidate_count > blocked_count:
                    selected_ids.add(source.id)
            for target in targets:
                excluded_self = not definition.allow_self_loop and target.id in source_ids
                candidate_count = len(source_ids) - int(excluded_self)
                blocked_count = sum(
                    source_id in source_ids
                    and (definition.allow_self_loop or source_id != target.id)
                    for source_id in covered_sources[(chunk.id, definition.name, target.id)]
                )
                if candidate_count > blocked_count:
                    selected_ids.add(target.id)

    selected = [entity for entity in entities if entity.id in selected_ids]
    return selected if len(selected) >= _MIN_RELATION_ENTITIES else entities


def _entities_grounded_in_text(entities: list[EntityMention], text: str) -> list[EntityMention]:
    """Return mentions grounded by identity or document-context surfaces in text."""

    index = build_surface_text_index(text)
    return [
        entity
        for entity in entities
        if indexed_surface_occurs(index, [entity.name, *entity.aliases])
        or (
            entity.is_document_context and indexed_surface_occurs(index, entity.contextual_surfaces)
        )
    ]


def _subwindow(
    parent: ExtractionWindow,
    chunks: list[Chunk],
    stage: str,
    pass_index: int,
) -> ExtractionWindow | None:
    """Create a stable child window for one targeted gleaning pass.

    Empty chunk selections return ``None`` so orchestration can terminate without
    constructing or tracing a meaningless provider request.
    """

    if not chunks:
        return None
    return ExtractionWindow(
        id=stable_id("extraction_subwindow", parent.id, stage, pass_index),
        document_id=parent.document_id,
        chunks=chunks,
        token_count=sum(chunk.token_count for chunk in chunks),
        dataset_columns=parent.dataset_columns,
    )


def _to_extraction_result(
    mentions: list[EntityMention],
    relations: list[RelationObservation],
    canonical_names: dict[str, str],
    held_relations: list[RelationObservation] | None = None,
    *,
    component_ids: dict[str, str] | None = None,
    ontology: OntologyProfile | None = None,
) -> ExtractionResult:
    """Translate grounded mention observations into the established graph contract.

    Resolution component IDs carry identity independently of display names.
    Resolution-selected names are applied to entities and relation endpoints while
    exact quotes, offsets, confidence, and non-canonical spellings remain available
    for downstream evidence and alias assembly. Held relations are translated the
    same way and kept apart.
    """

    mentions_by_id = {mention.id: mention for mention in mentions}
    entities = [
        ExtractedEntity(
            name=canonical_names[mention.id],
            resolved_component_id=(component_ids or {}).get(mention.id),
            type=mention.type,
            description=mention.description,
            source_chunk_id=mention.source_chunk_id,
            quote=mention.quote,
            start_offset=mention.start_offset,
            end_offset=mention.end_offset,
            confidence=mention.confidence,
            aliases=_canonical_aliases(mention, canonical_names[mention.id]),
            evidence_method=mention.evidence_method,
        )
        for mention in mentions
    ]
    return ExtractionResult(
        entities=entities,
        relations=_extracted_relations(
            relations, mentions_by_id, canonical_names, component_ids, ontology
        ),
        held_relations=_extracted_relations(
            held_relations or [], mentions_by_id, canonical_names, component_ids, ontology
        ),
    )


def _extracted_relations(
    relations: list[RelationObservation],
    mentions_by_id: dict[str, EntityMention],
    canonical_names: dict[str, str],
    component_ids: dict[str, str] | None,
    ontology: OntologyProfile | None,
) -> list[ExtractedRelation]:
    """Carry endpoint components and preserve only ontology-approved source loops."""

    allowed_self_loops = (
        {
            normalize_ontology_label(item.name)
            for item in ontology.relation_types
            if item.allow_self_loop
        }
        if ontology is not None
        else set()
    )
    extracted_relations: list[ExtractedRelation] = []
    for relation in relations:
        source = mentions_by_id.get(relation.source_entity_id)
        target = mentions_by_id.get(relation.target_entity_id)
        if source is None or target is None:
            continue
        source_name = canonical_names[source.id]
        target_name = canonical_names[target.id]
        source_self_loop = source.id == target.id
        if (
            source_self_loop
            and normalize_ontology_label(relation.relation_type) not in allowed_self_loops
        ):
            continue
        extracted_relations.append(
            ExtractedRelation(
                source_name=source_name,
                target_name=target_name,
                source_component_id=(component_ids or {}).get(source.id),
                target_component_id=(component_ids or {}).get(target.id),
                source_self_loop=source_self_loop,
                source_surface=relation.source_surface,
                target_surface=relation.target_surface,
                source_type=source.type,
                target_type=target.type,
                relation_type=relation.relation_type,
                description=relation.description,
                source_chunk_id=relation.source_chunk_id,
                quote=relation.quote,
                start_offset=relation.start_offset,
                end_offset=relation.end_offset,
                weight=max(0.1, relation.confidence),
                confidence=relation.confidence,
                quantities=relation.quantities,
                proposed_relation_type=relation.proposed_relation_type,
                evidence_method=relation.evidence_method,
                assertion=relation.assertion,
                rejected_record_key=relation.rejected_record_key,
            )
        )
    return extracted_relations


def _canonical_aliases(mention: EntityMention, canonical_name: str) -> list[str]:
    """Combine accepted aliases with a non-canonical mention spelling.

    Canonical-equivalent duplicates are removed in deterministic order.
    """

    values = [*mention.aliases]
    if normalize_ontology_label(mention.name) != normalize_ontology_label(canonical_name):
        values.append(mention.name)
    seen = {normalize_ontology_label(canonical_name)}
    result: list[str] = []
    for value in values:
        key = normalize_ontology_label(value)
        if value.strip() and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result


def _dedupe_mentions(mentions: list[EntityMention]) -> list[EntityMention]:
    """Remove repeated grounded mentions without collapsing distinct source spans.

    The first stable observation is retained.
    """

    seen: set[tuple[str, str, str, int | None]] = set()
    result: list[EntityMention] = []
    for mention in mentions:
        key = (
            normalize_ontology_label(mention.name),
            mention.type,
            mention.source_chunk_id,
            mention.start_offset,
        )
        if key not in seen:
            seen.add(key)
            result.append(mention)
    return result


def _dedupe_relations(relations: list[RelationObservation]) -> list[RelationObservation]:
    """Remove repeated directed relation observations within the same source chunk.

    Cross-chunk support remains independent evidence.
    """

    indexes: dict[tuple[str, str, str, str], int] = {}
    result: list[RelationObservation] = []
    for relation in relations:
        key = (
            relation.source_entity_id,
            relation.target_entity_id,
            relation.relation_type,
            relation.source_chunk_id,
        )
        if key not in indexes:
            indexes[key] = len(result)
            result.append(relation)
            continue
        existing_index = indexes[key]
        existing = result[existing_index]
        if existing.verification_required and not relation.verification_required:
            # Deterministic cue support must survive when an LLM independently
            # emitted the same triple first.
            result[existing_index] = existing.model_copy(
                update={
                    "evidence_method": "ontology_cue",
                    "verification_required": False,
                    "confidence": max(existing.confidence, relation.confidence),
                }
            )
    return result
