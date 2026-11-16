# SPDX-License-Identifier: Apache-2.0
"""Conservative entity resolution with auditable deterministic and LLM decisions."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import batched

from kg_processor.application.entity_names import identity_surface_match
from kg_processor.application.extraction_contracts import (
    ResolutionDecisionCandidate,
    ResolutionDecisionCandidateBatch,
    entity_resolution_request,
)
from kg_processor.application.ordered_map import ordered_map
from kg_processor.domain.extraction import (
    EntityMention,
    ResolutionCandidate,
    ResolutionDecision,
)
from kg_processor.ports.embeddings import EmbeddingProvider, EmbedOptions
from kg_processor.ports.llm import StructuredCompletionProvider

MIN_EMBEDDING_FOR_LEXICAL_MERGE = 0.75
_MIN_EXPANDED_IDENTITY_SIMILARITY = 0.80
_SORTED_NEIGHBORHOOD_SIZE = 12
_MIN_PAIR_BLOCK_SIZE = 2
MIN_SHORT_INITIALISM_LENGTH = 2
MAX_SHORT_INITIALISM_LENGTH = 6
ResolutionProgressCallback = Callable[[int, int, int, int], None]


@dataclass(frozen=True)
class EntityResolutionResult:
    """Return component identities, display names, and an auditable decision trail.

    The mappings feed graph assembly, while explicit merge and non-merge records
    explain how deterministic rules and optional LLM adjudication affected identity.
    """

    canonical_name_by_mention_id: dict[str, str]
    decisions: list[ResolutionDecision]
    trace: dict[str, object]
    component_by_mention_id: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _AdjudicationBatchResult:
    """Capture one resolution batch response or its conservative failure state.

    Original candidates are retained so malformed provider output can become
    explicit non-merge decisions rather than dropping resolution evidence.
    """

    index: int
    candidates: list[ResolutionCandidate]
    response: ResolutionDecisionCandidateBatch | None
    provider_metadata: dict[str, object]
    error_type: str | None = None


def resolve_entity_mentions(  # noqa: PLR0912,PLR0915 - branches record distinct rules.
    mentions: list[EntityMention],
    *,
    enabled: bool,
    embeddings: EmbeddingProvider,
    embed_options: EmbedOptions,
    llm: StructuredCompletionProvider,
    model: str,
    timeout_seconds: int,
    lexical_auto_merge: float,
    embedding_auto_merge: float,
    candidate_threshold: float,
    embedding_lexical_floor: float = 0.45,
    max_candidates_per_mention: int = 3,
    adjudication_batch_size: int = 20,
    adjudication_parallelism: int = 1,
    llm_merge_min_confidence: float = 0.90,
    batch_progress: ResolutionProgressCallback | None = None,
    seed: int,
) -> EntityResolutionResult:
    """Resolve same-type mentions without conflating semantic relatedness and identity.

    Exact surfaces merge deterministically, strong lexical and embedding agreement
    can auto-merge, and only a bounded ambiguous neighborhood reaches the LLM.
    Every rule records a decision, while provider failures keep candidates separate.
    """

    union_find = UnionFind([mention.id for mention in mentions])
    mentions_by_id = {mention.id: mention for mention in mentions}
    decisions: list[ResolutionDecision] = []
    pairs = _same_type_pairs(mentions)

    # Exact normalized names and declared aliases are deterministic identity
    # evidence, except when the only overlap is an ambiguous short initialism.
    # Different expanded names can legitimately share labels such as ``SRN``;
    # those pairs require adjudication instead of an irreversible exact merge.
    for left, right in pairs:
        if _deterministic_surface_overlap(left, right):
            conflict = _cluster_initialism_conflict(
                union_find,
                mentions_by_id,
                left.id,
                right.id,
            )
            if not conflict:
                union_find.union(left.id, right.id)
            decisions.append(
                ResolutionDecision(
                    left_id=left.id,
                    right_id=right.id,
                    same_entity=not conflict,
                    canonical_name=_preferred_name([left, right]) if not conflict else None,
                    confidence=1.0,
                    reason=(
                        "transitive_initialism_conflict"
                        if conflict
                        else "exact_normalized_name_or_alias"
                    ),
                )
            )
    # Repeated observations with the same typed canonical name are already linked
    # above. Collapse them before fuzzy work so they neither receive duplicate
    # embeddings nor consume every bounded neighbor slot that a genuine spelling
    # variant needs. Aliases and descriptions are retained on the representative.
    resolution_mentions = _name_representatives(mentions)
    unresolved_pairs = [
        (left, right, _lexical_score(left, right))
        for left, right in _same_type_pairs(resolution_mentions)
        if enabled
        and union_find.find(left.id) != union_find.find(right.id)
        and not _deterministic_surface_overlap(left, right)
    ]

    embedding_scores: dict[tuple[str, str], float] = {}
    if unresolved_pairs:
        vectors = embeddings.embed(
            [_resolution_text(item) for item in resolution_mentions],
            embed_options,
        )
        vector_by_id = {
            mention.id: vector for mention, vector in zip(resolution_mentions, vectors, strict=True)
        }
        embedding_scores = {
            (left.id, right.id): _cosine(vector_by_id[left.id], vector_by_id[right.id])
            for left, right, _score in unresolved_pairs
        }

    eligible_candidates: list[ResolutionCandidate] = []
    for left, right, lexical_score in unresolved_pairs:
        embedding_score = embedding_scores[(left.id, right.id)]
        numeric_mismatch = _has_numeric_mismatch(left, right)
        ambiguous_initialism = _has_ambiguous_initialism_overlap(left, right)
        lexical_merge = (
            lexical_score >= lexical_auto_merge
            and embedding_score >= MIN_EMBEDDING_FOR_LEXICAL_MERGE
            and not numeric_mismatch
            and not ambiguous_initialism
        )
        embedding_merge = (
            embedding_score >= embedding_auto_merge
            and lexical_score >= candidate_threshold
            and not numeric_mismatch
            and not ambiguous_initialism
        )
        if lexical_merge or embedding_merge:
            conflict = _cluster_initialism_conflict(
                union_find,
                mentions_by_id,
                left.id,
                right.id,
            )
            if not conflict:
                union_find.union(left.id, right.id)
            decisions.append(
                ResolutionDecision(
                    left_id=left.id,
                    right_id=right.id,
                    same_entity=not conflict,
                    canonical_name=_preferred_name([left, right]) if not conflict else None,
                    confidence=min(lexical_score, max(embedding_score, 0.0)),
                    reason=(
                        "transitive_initialism_conflict"
                        if conflict
                        else "high_confidence_lexical_and_embedding_match"
                    ),
                )
            )
        elif (
            _is_plausible_identity_candidate(
                left,
                right,
                lexical_score,
                embedding_score,
                candidate_threshold,
                embedding_lexical_floor,
            )
            and not numeric_mismatch
        ):
            eligible_candidates.append(
                ResolutionCandidate(
                    left_id=left.id,
                    right_id=right.id,
                    lexical_score=lexical_score,
                    embedding_score=embedding_score,
                )
            )

    # Embedding similarity is broad in a domain-specific corpus: Karate and
    # Judo may be semantically close while clearly naming different entities.
    # The lexical gate above protects precision; this bounded neighborhood also
    # prevents a dense corpus from producing quadratic LLM adjudication work.
    representative_mentions_by_id = {mention.id: mention for mention in resolution_mentions}
    uncertain = _bounded_candidates(
        eligible_candidates,
        representative_mentions_by_id,
        max_candidates_per_mention,
    )
    adjudication_metadata: dict[str, object] = {}
    failed_adjudication_batches = 0
    if uncertain:
        batch_metadata: list[dict[str, object]] = []
        batches = [list(batch) for batch in batched(uncertain, adjudication_batch_size)]
        batch_results = _adjudicate_batches(
            batches,
            resolution_mentions,
            llm,
            model,
            timeout_seconds,
            seed,
            adjudication_parallelism,
            batch_progress,
        )
        for batch_result in batch_results:
            if batch_result.response is None:
                failed_adjudication_batches += 1
                batch_metadata.append(
                    {
                        "batch_index": batch_result.index,
                        "status": "format_error",
                        "error_type": batch_result.error_type,
                    }
                )
            else:
                batch_metadata.append(
                    {
                        "batch_index": batch_result.index,
                        "status": "completed",
                        **batch_result.provider_metadata,
                    }
                )
            for decision in _batch_decisions(batch_result, llm_merge_min_confidence):
                if decision.same_entity and _cluster_initialism_conflict(
                    union_find,
                    mentions_by_id,
                    decision.left_id,
                    decision.right_id,
                ):
                    decisions.append(
                        decision.model_copy(
                            update={
                                "same_entity": False,
                                "canonical_name": None,
                                "reason": "transitive_initialism_conflict",
                            }
                        )
                    )
                    continue
                if decision.same_entity:
                    union_find.union(decision.left_id, decision.right_id)
                decisions.append(decision)
        adjudication_metadata = {"adjudication_batches": batch_metadata}

    groups: dict[str, list[EntityMention]] = {}
    for mention in mentions:
        groups.setdefault(union_find.find(mention.id), []).append(mention)
    preferred_name_by_root = {root: _preferred_name(group) for root, group in groups.items()}
    canonical_name_by_id = {
        mention.id: preferred_name_by_root[union_find.find(mention.id)] for mention in mentions
    }
    return EntityResolutionResult(
        canonical_name_by_mention_id=canonical_name_by_id,
        component_by_mention_id={mention.id: union_find.find(mention.id) for mention in mentions},
        decisions=decisions,
        trace={
            "stage": "entity_resolution",
            "mentions": len(mentions),
            "resolution_identity_records": len(resolution_mentions),
            "same_type_pairs": len(pairs),
            "candidate_pairs": len(unresolved_pairs),
            "eligible_candidate_pairs": len(eligible_candidates),
            "pruned_candidate_pairs": len(eligible_candidates) - len(uncertain),
            "llm_adjudications": len(uncertain),
            "failed_adjudication_batches": failed_adjudication_batches,
            "merged_pairs": sum(decision.same_entity for decision in decisions),
            "canonical_entities": len(groups),
            **adjudication_metadata,
        },
    )


def adjudicate_resolution_candidates(
    candidates: list[ResolutionCandidate],
    mentions: list[EntityMention],
    *,
    llm: StructuredCompletionProvider,
    model: str,
    timeout_seconds: int,
    batch_size: int,
    minimum_merge_confidence: float,
    seed: int,
) -> list[ResolutionDecision]:
    """Adjudicate one bounded candidate partition for distributed execution.

    Candidate generation and connected components belong to the distributed data
    engine. This provider-neutral function preserves the local prompt, structured
    response reconciliation, confidence policy, and conservative failure behavior
    inside independently retryable executor partitions.
    """

    decisions: list[ResolutionDecision] = []
    batches = [list(batch) for batch in batched(candidates, batch_size)]
    for result in _adjudicate_batches(
        batches,
        mentions,
        llm,
        model,
        timeout_seconds,
        seed,
        parallelism=1,
        progress=None,
    ):
        decisions.extend(_batch_decisions(result, minimum_merge_confidence))
    return decisions


def _batch_decisions(
    result: _AdjudicationBatchResult,
    minimum_confidence: float,
) -> list[ResolutionDecision]:
    """Turn one adjudicated batch into a decision per candidate pair.

    Resolution is conservative enrichment. A malformed batch keeps its
    candidates distinct rather than invalidating grounded extraction.
    """

    if result.response is None:
        return [
            ResolutionDecision(
                left_id=candidate.left_id,
                right_id=candidate.right_id,
                same_entity=False,
                confidence=0.0,
                reason="llm_resolution_format_error",
            )
            for candidate in result.candidates
        ]
    response_by_pair = {
        _pair_key(item.left_id, item.right_id): item for item in result.response.decisions
    }
    return [
        _resolution_decision(
            candidate,
            response_by_pair.get(_pair_key(candidate.left_id, candidate.right_id)),
            minimum_confidence,
        )
        for candidate in result.candidates
    ]


def _resolution_decision(
    candidate: ResolutionCandidate,
    response_item: ResolutionDecisionCandidate | None,
    minimum_confidence: float,
) -> ResolutionDecision:
    """Normalize one adjudication result and enforce the merge confidence floor.

    Missing decisions and low-confidence affirmative answers are converted into
    explicit non-merges so uncertainty cannot silently collapse graph nodes.
    """

    if response_item is None:
        return ResolutionDecision(
            left_id=candidate.left_id,
            right_id=candidate.right_id,
            same_entity=False,
            confidence=0.0,
            reason="llm_omitted_candidate",
        )
    parsed = ResolutionDecision.model_validate(response_item.model_dump())
    if parsed.same_entity and parsed.confidence < minimum_confidence:
        return parsed.model_copy(
            update={
                "same_entity": False,
                "reason": f"llm_merge_below_confidence_threshold: {parsed.reason}",
            }
        )
    return parsed


def _is_plausible_identity_candidate(
    left: EntityMention,
    right: EntityMention,
    lexical_score: float,
    embedding_score: float,
    candidate_threshold: float,
    embedding_lexical_floor: float,
) -> bool:
    """Require name-level evidence before semantic similarity can trigger an LLM call.

    Embeddings measure relatedness, not identity. A lexical floor keeps related
    domain concepts out of resolution while explicit initialisms such as BJJ
    versus Brazilian Jiu-Jitsu remain eligible for careful adjudication.
    """

    return (
        lexical_score >= candidate_threshold
        or _acronym_match(left, right)
        or (embedding_score >= candidate_threshold and lexical_score >= embedding_lexical_floor)
    )


def _bounded_candidates(
    candidates: list[ResolutionCandidate],
    mentions_by_id: dict[str, EntityMention],
    max_per_mention: int,
) -> list[ResolutionCandidate]:
    """Select a deterministic, degree-bounded candidate neighborhood per mention.

    Initialisms rank first, followed by lexical and embedding evidence. Limiting
    both endpoints prevents dense domain corpora from creating quadratic LLM work.
    """

    if max_per_mention <= 0:
        raise ValueError("resolution max candidates per mention must be positive")

    # Initialism matches are scarce and highly useful, so rank them ahead of
    # fuzzy string matches. Remaining ties prefer lexical identity evidence,
    # then embeddings, then stable ids to make reruns byte-for-byte repeatable.
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            not _acronym_match(
                mentions_by_id[candidate.left_id],
                mentions_by_id[candidate.right_id],
            ),
            -candidate.lexical_score,
            -(candidate.embedding_score or 0.0),
            candidate.left_id,
            candidate.right_id,
        ),
    )
    selected: list[ResolutionCandidate] = []
    selected_count: dict[str, int] = {}
    for candidate in ranked:
        if selected_count.get(candidate.left_id, 0) >= max_per_mention:
            continue
        if selected_count.get(candidate.right_id, 0) >= max_per_mention:
            continue
        selected.append(candidate)
        selected_count[candidate.left_id] = selected_count.get(candidate.left_id, 0) + 1
        selected_count[candidate.right_id] = selected_count.get(candidate.right_id, 0) + 1
    return selected


def _adjudicate_batches(
    batches: list[list[ResolutionCandidate]],
    mentions: list[EntityMention],
    llm: StructuredCompletionProvider,
    model: str,
    timeout_seconds: int,
    seed: int,
    parallelism: int,
    progress: ResolutionProgressCallback | None,
) -> list[_AdjudicationBatchResult]:
    """Run independent resolution batches concurrently and restore stable ordering.

    Progress follows actual completion order for responsive UIs, but returned
    results are sorted by batch index so scheduling cannot alter merge or trace order.
    """

    failures = 0
    total_candidates = sum(len(batch) for batch in batches)

    def record(completed: int, result: _AdjudicationBatchResult) -> None:
        """Publish aggregate failure progress as each batch completes.

        The closure updates state only on the orchestration thread.
        """

        nonlocal failures
        failures += result.response is None
        if progress is not None:
            progress(completed, len(batches), total_candidates, failures)

    return ordered_map(
        batches,
        lambda index, batch: _adjudicate_batch(
            index, batch, mentions, llm, model, timeout_seconds, seed
        ),
        parallelism=parallelism,
        on_complete=record,
    )


def _adjudicate_batch(
    index: int,
    candidates: list[ResolutionCandidate],
    mentions: list[EntityMention],
    llm: StructuredCompletionProvider,
    model: str,
    timeout_seconds: int,
    seed: int,
) -> _AdjudicationBatchResult:
    """Execute one strict resolution request with conservative format fallback.

    Validation errors are captured as batch data instead of escaping, allowing the
    caller to preserve candidates as distinct entities and continue the graph run.
    """

    request = entity_resolution_request(
        mentions,
        candidates,
        model=model,
        timeout_seconds=timeout_seconds,
        seed=seed + index,
    )
    try:
        completion = llm.complete_structured(request)
        response = ResolutionDecisionCandidateBatch.model_validate(completion.payload)
    except Exception as exc:
        # Resolution is optional enrichment. Transport failures, rate limits,
        # timeouts, and malformed provider payloads all preserve the conservative
        # default that uncertain mentions remain separate. BaseException is not
        # caught so cancellation and interpreter shutdown still propagate.
        return _AdjudicationBatchResult(
            index=index,
            candidates=candidates,
            response=None,
            provider_metadata={},
            error_type=type(exc).__name__,
        )
    return _AdjudicationBatchResult(
        index=index,
        candidates=candidates,
        response=response,
        provider_metadata=dict(completion.provider_metadata),
    )


class UnionFind:
    """Maintain mention identity clusters with deterministic disjoint-set unions.

    Lexically smaller roots always win, ensuring equivalent decisions produce the
    same canonical grouping regardless of traversal or concurrent response order.
    """

    def __init__(self, values: list[str]) -> None:
        """Initialize every supplied mention ID as its own independent identity cluster.

        No implicit merge occurs during construction.
        """

        self.parent = {value: value for value in values}
        self._members = {value: {value} for value in values}

    def find(self, value: str) -> str:
        """Return a cluster's stable root while applying path compression.

        Compression changes only internal lookup cost; deterministic root selection
        remains governed by ``union``. The walk is iterative because lexical-min
        roots carry no rank, so an adversarial edge order builds a chain as long
        as the bounded component, deeper than the interpreter's recursion limit.
        """

        root = self.parent[value]
        while root != self.parent[root]:
            root = self.parent[root]
        while value != root:
            self.parent[value], value = root, self.parent[value]
        return root

    def union(self, left: str, right: str) -> None:
        """Merge two roots with lexical tie-breaking independent of input order.

        Repeated union calls are idempotent, which permits multiple independent
        identity rules to support the same merge without special coordination.
        """

        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        winner, loser = sorted([left_root, right_root])
        self.parent[loser] = winner
        self._members[winner].update(self._members.pop(loser))

    def members(self, value: str) -> set[str]:
        """Return a copy of the IDs currently assigned to one identity cluster."""

        return set(self._members[self.find(value)])


def _cluster_initialism_conflict(
    union_find: UnionFind,
    mentions_by_id: dict[str, EntityMention],
    left_id: str,
    right_id: str,
) -> bool:
    """Prevent a short initialism from transitively joining full identities.

    Pairwise adjudication cannot see prior union decisions. Without a cluster-level
    guard, ``DA`` can independently match both Dario Amodei and Diogo Almeida and
    collapse the two authors despite no model ever claiming that the full names are
    equivalent. The same failure applies to technical acronyms such as ``SRN``.
    Full surfaces that are already close spelling variants remain mergeable.
    """

    left = mentions_by_id[left_id]
    right = mentions_by_id[right_id]
    _left_initialisms, left_expanded = _partition_identity_surfaces(left)
    _right_initialisms, right_expanded = _partition_identity_surfaces(right)
    if left_expanded & right_expanded:
        return False
    left_mentions = [mentions_by_id[item] for item in union_find.members(left_id)]
    right_mentions = [mentions_by_id[item] for item in union_find.members(right_id)]
    left_initialisms, left_expanded = _cluster_identity_surfaces(left_mentions)
    right_initialisms, right_expanded = _cluster_identity_surfaces(right_mentions)
    if not (left_initialisms & right_initialisms and left_expanded and right_expanded):
        return False
    return not any(
        SequenceMatcher(None, left, right).ratio() >= _MIN_EXPANDED_IDENTITY_SIMILARITY
        for left in left_expanded
        for right in right_expanded
    )


def _cluster_identity_surfaces(
    mentions: list[EntityMention],
) -> tuple[set[str], set[str]]:
    """Collect explicit and derived initialisms plus normalized full surfaces."""

    initialisms: set[str] = set()
    expanded: set[str] = set()
    for mention in mentions:
        for surface in [mention.name, *mention.aliases]:
            normalized = _normalize(surface)
            if not normalized:
                continue
            if _is_short_initialism(surface):
                initialisms.add(normalized)
                continue
            expanded.add(normalized)
            words = re.findall(r"[A-Za-z]+", surface)
            if len(words) > 1:
                initialisms.add("".join(word[0] for word in words).casefold())
    return initialisms, expanded


def _name_representatives(mentions: list[EntityMention]) -> list[EntityMention]:
    """Compact repeated typed names before semantic identity resolution.

    Extraction deliberately preserves one mention per grounded occurrence. Exact
    repetitions are valuable evidence later, but embedding and fuzzy-candidate
    generation need one identity record rather than every occurrence. The stable
    minimum mention ID represents each normalized name while aliases, context
    surfaces, the strongest description, and confidence are combined for richer
    adjudication. Original mentions remain untouched and are used for final graph
    grouping after the representative decisions have been unioned.
    """

    grouped: dict[tuple[str, str], list[EntityMention]] = {}
    for mention in mentions:
        grouped.setdefault((mention.type, _normalize(mention.name)), []).append(mention)

    representatives: list[EntityMention] = []
    for key in sorted(grouped):
        observations = grouped[key]
        representative = min(observations, key=lambda item: item.id)
        aliases = list(
            dict.fromkeys(
                surface
                for observation in observations
                for surface in [observation.name, *observation.aliases]
                if surface != representative.name
            )
        )
        contextual_surfaces = list(
            dict.fromkeys(
                surface
                for observation in observations
                for surface in observation.contextual_surfaces
            )
        )
        description = min(
            (observation.description for observation in observations),
            key=lambda value: (-len(value), value),
        )
        representatives.append(
            representative.model_copy(
                update={
                    "aliases": aliases,
                    "description": description,
                    "confidence": max(observation.confidence for observation in observations),
                    "is_document_context": any(
                        observation.is_document_context for observation in observations
                    ),
                    "contextual_surfaces": contextual_surfaces,
                }
            )
        )
    return representatives


def _same_type_pairs(
    mentions: list[EntityMention],
) -> list[tuple[EntityMention, EntityMention]]:
    """Generate a bounded same-type candidate neighborhood for identity checks.

    Materializing every same-type pair is quadratic and becomes unusable for large
    corpora. A sorted-neighborhood index compares nearby normalized surfaces and
    adds exact alias/acronym blocks. This keeps candidate growth linear in ordinary
    corpora while retaining the high-value lexical pairs used by deterministic and
    LLM-assisted resolution.
    """

    mentions_by_type: dict[str, list[EntityMention]] = {}
    for mention in mentions:
        mentions_by_type.setdefault(mention.type, []).append(mention)

    pairs_by_key: dict[tuple[str, str], tuple[EntityMention, EntityMention]] = {}
    for typed_mentions in mentions_by_type.values():
        ordered = sorted(typed_mentions, key=lambda item: (_normalize(item.name), item.id))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 : index + 1 + _SORTED_NEIGHBORHOOD_SIZE]:
                _add_pair(pairs_by_key, left, right)

        # Exact aliases can sort far from canonical names, so index every accepted
        # surface and connect each group to one representative instead of creating
        # a quadratic clique of duplicate mentions.
        by_surface: dict[str, list[EntityMention]] = {}
        by_initialism: dict[str, list[EntityMention]] = {}
        for mention in typed_mentions:
            for surface in _surface_sets(mention):
                by_surface.setdefault(surface, []).append(mention)
            initialism = _initialism_key(mention)
            if initialism:
                by_initialism.setdefault(initialism, []).append(mention)
        for block in [*by_surface.values(), *by_initialism.values()]:
            if len(block) < _MIN_PAIR_BLOCK_SIZE:
                continue
            representative = min(block, key=lambda item: item.id)
            for other in block:
                if other.id != representative.id:
                    _add_pair(pairs_by_key, representative, other)
    return [pairs_by_key[key] for key in sorted(pairs_by_key)]


def _add_pair(
    pairs: dict[tuple[str, str], tuple[EntityMention, EntityMention]],
    left: EntityMention,
    right: EntityMention,
) -> None:
    """Insert one order-independent mention pair into a deterministic mapping."""

    first, second = sorted([left, right], key=lambda item: item.id)
    pairs[(first.id, second.id)] = (first, second)


def _initialism_key(mention: EntityMention) -> str:
    """Return a shared initialism key for acronym-like names and expanded aliases."""

    keys: list[str] = []
    for surface in [mention.name, *mention.aliases]:
        words = re.findall(r"[\w]+", surface, flags=re.UNICODE)
        if len(words) > 1 and all(word.isalpha() for word in words):
            keys.append("".join(word[0] for word in words).casefold())
        elif (
            len(words) == 1 and words[0].isalpha() and len(words[0]) <= MAX_SHORT_INITIALISM_LENGTH
        ):
            keys.append(words[0].casefold())
    return min(keys, default="")


def _has_numeric_mismatch(left: EntityMention, right: EntityMention) -> bool:
    """Prevent fuzzy matching from merging names with conflicting numeric identity.

    Contract numbers, model generations, dates, and version labels often differ by
    one digit yet score above lexical auto-merge thresholds. Equal numeric token
    sequences remain eligible; any mismatch requires the mentions to stay separate.
    """

    left_numbers = re.findall(r"\d+(?:[.,]\d+)?", left.name)
    right_numbers = re.findall(r"\d+(?:[.,]\d+)?", right.name)
    return bool(left_numbers or right_numbers) and left_numbers != right_numbers


def _acronym_match(left: EntityMention, right: EntityMention) -> bool:
    """Return whether any grounded surface pair forms a valid initialism match.

    Names and accepted aliases participate, allowing forms such as BJJ and
    Brazilian Jiu-Jitsu while retaining conservative identity validation.
    """

    return any(
        identity_surface_match(left_surface, right_surface)
        for left_surface in [left.name, *left.aliases]
        for right_surface in [right.name, *right.aliases]
    )


def _surface_sets(mention: EntityMention) -> set[str]:
    """Collect all normalized, non-empty name and accepted alias surfaces for one mention."""

    return {_normalize(value) for value in [mention.name, *mention.aliases] if value.strip()}


def _deterministic_surface_overlap(left: EntityMention, right: EntityMention) -> bool:
    """Accept exact identity overlap unless it consists of a conflicting initialism.

    Shared expanded surfaces remain deterministic, as does repetition of a bare
    initialism with no competing expansion. When one short uppercase surface maps
    to different expanded names, the pair stays available for conservative LLM
    adjudication instead of being auto-merged.
    """

    common = _surface_sets(left) & _surface_sets(right)
    if not common:
        return False
    ambiguous = _has_ambiguous_initialism_overlap(left, right)
    return not ambiguous


def _has_ambiguous_initialism_overlap(left: EntityMention, right: EntityMention) -> bool:
    """Detect a shared acronym whose available expanded identity surfaces disagree."""

    left_initialisms, left_expanded = _partition_identity_surfaces(left)
    right_initialisms, right_expanded = _partition_identity_surfaces(right)
    return bool(
        left_initialisms & right_initialisms
        and (left_expanded or right_expanded)
        and not (left_expanded & right_expanded)
    )


def _partition_identity_surfaces(mention: EntityMention) -> tuple[set[str], set[str]]:
    """Split accepted identity surfaces into short initialisms and fuller names."""

    initialisms: set[str] = set()
    expanded: set[str] = set()
    for surface in [mention.name, *mention.aliases]:
        normalized = _normalize(surface)
        if not normalized:
            continue
        target = initialisms if _is_short_initialism(surface) else expanded
        target.add(normalized)
    return initialisms, expanded


def _is_short_initialism(surface: str) -> bool:
    """Recognize compact uppercase labels whose expansion may be corpus-dependent."""

    letters = "".join(character for character in surface if character.isalpha())
    return (
        MIN_SHORT_INITIALISM_LENGTH <= len(letters) <= MAX_SHORT_INITIALISM_LENGTH
        and letters.isupper()
    )


def _lexical_score(left: EntityMention, right: EntityMention) -> float:
    """Return the strongest sequence similarity across every accepted surface pairing.

    Names and validated aliases participate equally.
    """

    scores = [
        SequenceMatcher(None, left_surface, right_surface).ratio()
        for left_surface in _surface_sets(left)
        for right_surface in _surface_sets(right)
    ]
    return max(scores, default=0.0)


def _resolution_text(mention: EntityMention) -> str:
    """Build typed descriptive text used to embed one entity-resolution mention.

    Accepted aliases and descriptions provide semantic context.
    """

    aliases = ", ".join(mention.aliases)
    return f"{mention.type}: {mention.name}. Aliases: {aliases}. {mention.description}"


def _cosine(left: list[float], right: list[float]) -> float:
    """Compute bounded cosine similarity while treating either zero vector as unrelated.

    Equal vector lengths are required by strict zipping.
    """

    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def _preferred_name(mentions: list[EntityMention]) -> str:
    """Choose a stable canonical spelling from one resolved mention cluster.

    Frequency takes precedence over descriptiveness, followed by lexical order,
    so repeated evidence and not provider response timing determines display names.
    """

    # Prefer the most frequent normalized surface, then the most informative
    # spelling. Stable lexical tie-breaking keeps reruns byte-identical.
    frequencies: dict[str, int] = {}
    for mention in mentions:
        key = _normalize(mention.name)
        frequencies[key] = frequencies.get(key, 0) + 1
    return min(
        (mention.name for mention in mentions),
        key=lambda name: (-frequencies[_normalize(name)], -len(name), name.casefold()),
    )


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    """Create a stable, order-independent lookup key for an adjudicated mention pair.

    Lexical ordering makes response reconciliation deterministic.
    """

    ordered = sorted([left_id, right_id])
    return ordered[0], ordered[1]


def _normalize(value: str) -> str:
    """Normalize case, spacing, and hyphens for conservative entity name comparison.

    Other punctuation remains identity-significant here.
    """

    return " ".join(unicodedata.normalize("NFC", value).casefold().replace("-", " ").split())
