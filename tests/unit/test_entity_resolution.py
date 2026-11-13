from __future__ import annotations

import json

from documents import grounded

from kg_processor.adapters.embeddings.hash import HashEmbeddingProvider
from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.entity_resolution import UnionFind, resolve_entity_mentions
from kg_processor.application.extraction_contracts import entity_resolution_request
from kg_processor.domain.extraction import EntityMention, ResolutionCandidate
from kg_processor.ports.embeddings import EmbedOptions
from kg_processor.ports.llm import StructuredCompletionRequest, StructuredCompletionResult


class _MalformedResolutionLlm(FakeLlmProvider):
    """Simulate a provider that exhausts its structured-output repair attempt.

    Only identity adjudication is made malformed.
    """

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Fail only resolution tasks while preserving normal fake behavior elsewhere.

        This isolates the conservative fallback contract.
        """

        if request.task_name == "entity_resolution":
            raise ValueError("truncated JSON")
        return super().complete_structured(request)


class _UnavailableResolutionLlm(_MalformedResolutionLlm):
    """Raise a transport-style exception to verify the same conservative fallback."""

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Simulate a provider outage during optional identity adjudication."""

        if request.task_name == "entity_resolution":
            raise RuntimeError("provider unavailable")
        return super().complete_structured(request)


class _RecordingResolutionLlm(FakeLlmProvider):
    """Record resolution adjudication calls while retaining deterministic fake semantics.

    Call counts expose candidate and batching behavior.
    """

    def __init__(self) -> None:
        """Initialize a counter used to assert bounded provider-call behavior.

        The fixture has no hidden global state.
        """

        self.resolution_call_count = 0

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Count resolution tasks before delegating generation to the fake provider.

        Non-resolution calls do not affect the metric.
        """

        if request.task_name == "entity_resolution":
            self.resolution_call_count += 1
        return super().complete_structured(request)


class _MergeResolutionLlm(_RecordingResolutionLlm):
    """Accept every bounded identity candidate for representative-compaction tests."""

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Return one high-confidence identity decision for every supplied pair."""

        if request.task_name != "entity_resolution":
            return super().complete_structured(request)
        self.resolution_call_count += 1
        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        return StructuredCompletionResult(
            payload={
                "decisions": [
                    {
                        "left_id": candidate["left_id"],
                        "right_id": candidate["right_id"],
                        "same_entity": True,
                        "canonical_name": "Plain network",
                        "confidence": 1.0,
                        "reason": "The source surfaces are spelling variants.",
                    }
                    for candidate in payload["candidates"]
                ]
            },
            provider_metadata={"provider": "test"},
        )


class _IdenticalEmbeddingProvider:
    """Make every mention semantically identical so tests isolate lexical candidate gating.

    Any resulting merge would therefore expose a gate failure.
    """

    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        """Return one identical non-zero vector for every supplied mention text.

        Vector dimension still follows production options.
        """

        vector = [1.0, *([0.0] * (options.dimension - 1))]
        return [vector.copy() for _text in texts]


class _OrthogonalEmbeddingProvider:
    """Keep fuzzy-name candidates below the automatic embedding-merge threshold.

    Lexical neighborhood pruning remains independently observable.
    """

    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        """Return deterministic orthogonal vectors that remove semantic similarity from tests.

        Input order selects each basis vector.
        """

        vectors: list[list[float]] = []
        for index, _text in enumerate(texts):
            vector = [0.0] * options.dimension
            vector[index] = 1.0
            vectors.append(vector)
        return vectors


def test_union_find_walks_a_chain_deeper_than_the_recursion_limit() -> None:
    """Edges arriving in descending-id order chain every vertex under the next.

    The Spark bounded-components path hands over components of up to 100,000
    vertices, far past the interpreter's recursion limit, so the root walk
    must not recurse.
    """

    ids = [f"mention_{index:05d}" for index in range(10_000)]
    union_find = UnionFind(ids)
    for index in range(len(ids) - 1, 0, -1):
        union_find.union(ids[index], ids[index - 1])

    assert union_find.find(ids[-1]) == ids[0]
    assert len(union_find.members(ids[-1])) == len(ids)


def test_entity_resolution_merges_exact_declared_aliases_without_llm_adjudication() -> None:
    """Ensure exact accepted aliases merge deterministically without spending an LLM call.

    Unrelated mentions must remain separate.
    """

    mentions = [
        _mention("m1", "Brazilian Jiu-Jitsu", ["BJJ"]),
        _mention("m2", "BJJ", ["Brazilian Jiu-Jitsu"]),
        _mention("m3", "Judo", []),
    ]

    result = resolve_entity_mentions(
        mentions,
        enabled=True,
        embeddings=HashEmbeddingProvider(),
        embed_options=EmbedOptions(model="hash", dimension=8, batch_size=8),
        llm=FakeLlmProvider(),
        model="fake",
        timeout_seconds=30,
        lexical_auto_merge=0.96,
        embedding_auto_merge=0.94,
        candidate_threshold=0.80,
        seed=17,
    )

    assert result.canonical_name_by_mention_id["m1"] == "Brazilian Jiu-Jitsu"
    assert result.canonical_name_by_mention_id["m2"] == "Brazilian Jiu-Jitsu"
    assert result.canonical_name_by_mention_id["m3"] == "Judo"
    assert result.decisions[0].reason == "exact_normalized_name_or_alias"


def test_entity_resolution_does_not_exact_merge_conflicting_acronym_expansions() -> None:
    """Keep distinct expanded identities separate when they share only an acronym."""

    mentions = [
        _mention("elman", "Elman simple recurrent network", ["SRN"]),
        _mention("scene", "SRN", ["Scene Representation Networks"]),
    ]

    result = resolve_entity_mentions(
        mentions,
        enabled=False,
        embeddings=HashEmbeddingProvider(),
        embed_options=EmbedOptions(model="hash", dimension=8, batch_size=8),
        llm=FakeLlmProvider(),
        model="fake",
        timeout_seconds=30,
        lexical_auto_merge=0.96,
        embedding_auto_merge=0.94,
        candidate_threshold=0.80,
        seed=17,
    )

    assert result.canonical_name_by_mention_id == {
        "elman": "Elman simple recurrent network",
        "scene": "SRN",
    }
    assert result.trace["merged_pairs"] == 0


def test_entity_resolution_blocks_transitive_initialism_merges() -> None:
    """Keep two full identities separate when an ambiguous initialism matches both."""

    mentions = [
        _mention("initials", "DA", []).model_copy(update={"type": "PERSON"}),
        _mention("dario", "Dario Amodei", []).model_copy(update={"type": "PERSON"}),
        _mention("diogo", "Diogo Almeida", []).model_copy(update={"type": "PERSON"}),
    ]

    result = resolve_entity_mentions(
        mentions,
        enabled=True,
        embeddings=_OrthogonalEmbeddingProvider(),
        embed_options=EmbedOptions(model="orthogonal", dimension=3, batch_size=3),
        llm=_MergeResolutionLlm(),
        model="fake",
        timeout_seconds=30,
        lexical_auto_merge=1.0,
        embedding_auto_merge=1.0,
        candidate_threshold=0.0,
        embedding_lexical_floor=0.0,
        max_candidates_per_mention=3,
        seed=17,
    )

    canonical = result.canonical_name_by_mention_id
    assert canonical["dario"] != canonical["diogo"]
    assert sum(value == canonical["initials"] for value in canonical.values()) == 2
    assert any(decision.reason == "transitive_initialism_conflict" for decision in result.decisions)


def test_entity_resolution_keeps_candidates_separate_when_adjudication_is_malformed() -> None:
    """Ensure malformed adjudication keeps uncertain mentions separate and records the failure.

    Format errors must never become implicit merges.
    """

    mentions = [
        _mention("m1", "Brazilian Jiu Jitsu", []),
        _mention("m2", "Brazilian Jiu Jitsu Academy", []),
    ]

    result = resolve_entity_mentions(
        mentions,
        enabled=True,
        embeddings=HashEmbeddingProvider(),
        embed_options=EmbedOptions(model="hash", dimension=8, batch_size=8),
        llm=_MalformedResolutionLlm(),
        model="fake",
        timeout_seconds=30,
        lexical_auto_merge=1.0,
        embedding_auto_merge=1.0,
        candidate_threshold=0.5,
        seed=17,
    )

    assert result.canonical_name_by_mention_id == {
        "m1": "Brazilian Jiu Jitsu",
        "m2": "Brazilian Jiu Jitsu Academy",
    }
    assert result.decisions[-1].reason == "llm_resolution_format_error"
    assert result.trace["failed_adjudication_batches"] == 1


def test_entity_resolution_keeps_candidates_separate_when_provider_is_unavailable() -> None:
    """Treat transport outages like malformed optional adjudication, not run failures."""

    mentions = [
        _mention("m1", "Brazilian Jiu Jitsu", []),
        _mention("m2", "Brazilian Jiu Jitsu Academy", []),
    ]

    result = resolve_entity_mentions(
        mentions,
        enabled=True,
        embeddings=HashEmbeddingProvider(),
        embed_options=EmbedOptions(model="hash", dimension=8, batch_size=8),
        llm=_UnavailableResolutionLlm(),
        model="fake",
        timeout_seconds=30,
        lexical_auto_merge=1.0,
        embedding_auto_merge=1.0,
        candidate_threshold=0.5,
        seed=17,
    )

    assert result.trace["failed_adjudication_batches"] == 1
    assert not result.decisions[-1].same_entity


def test_entity_resolution_never_auto_merges_conflicting_numbers() -> None:
    """Keep versioned and numbered entities distinct despite near-identical names."""

    mentions = [
        _mention("m1", "International Contract 2024", []),
        _mention("m2", "International Contract 2025", []),
    ]

    result = resolve_entity_mentions(
        mentions,
        enabled=True,
        embeddings=_IdenticalEmbeddingProvider(),
        embed_options=EmbedOptions(model="identical", dimension=4, batch_size=4),
        llm=FakeLlmProvider(),
        model="fake",
        timeout_seconds=30,
        lexical_auto_merge=0.90,
        embedding_auto_merge=0.90,
        candidate_threshold=0.50,
        seed=17,
    )

    assert result.canonical_name_by_mention_id == {
        "m1": "International Contract 2024",
        "m2": "International Contract 2025",
    }
    assert result.trace["merged_pairs"] == 0


def test_entity_resolution_candidate_generation_is_bounded() -> None:
    """Ensure a dense same-type corpus does not materialize all pair combinations."""

    mentions = [_mention(f"m{index:04d}", f"Entity {index:04d}", []) for index in range(1_000)]

    result = resolve_entity_mentions(
        mentions,
        enabled=False,
        embeddings=HashEmbeddingProvider(),
        embed_options=EmbedOptions(model="hash", dimension=8, batch_size=8),
        llm=FakeLlmProvider(),
        model="fake",
        timeout_seconds=30,
        lexical_auto_merge=0.96,
        embedding_auto_merge=0.94,
        candidate_threshold=0.80,
        seed=17,
    )

    assert result.trace["same_type_pairs"] == 11_922


def test_entity_resolution_compacts_repeated_names_before_fuzzy_adjudication() -> None:
    """Prevent duplicate observations from crowding out a genuine name variant."""

    mentions = [
        *[_mention(f"duplicate-{index}", "Plain network", []) for index in range(5)],
        _mention("variant", "Plain networks", []),
    ]
    llm = _MergeResolutionLlm()

    result = resolve_entity_mentions(
        mentions,
        enabled=True,
        embeddings=_IdenticalEmbeddingProvider(),
        embed_options=EmbedOptions(model="identical", dimension=4, batch_size=4),
        llm=llm,
        model="fake",
        timeout_seconds=30,
        lexical_auto_merge=1.1,
        embedding_auto_merge=1.1,
        candidate_threshold=0.5,
        max_candidates_per_mention=1,
        seed=17,
    )

    assert result.trace["mentions"] == 6
    assert result.trace["resolution_identity_records"] == 2
    assert result.trace["llm_adjudications"] == 1
    assert llm.resolution_call_count == 1
    assert set(result.canonical_name_by_mention_id.values()) == {"Plain network"}


def test_entity_resolution_splits_uncertain_pairs_into_bounded_batches() -> None:
    """Ensure uncertain identity pairs respect configured provider request batch boundaries.

    Every candidate should receive one deterministic judgment.
    """

    mentions = [
        _mention("m1", "Karate Alpha", []),
        _mention("m2", "Karate Beta", []),
        _mention("m3", "Karate Gamma", []),
    ]
    llm = _RecordingResolutionLlm()

    result = resolve_entity_mentions(
        mentions,
        enabled=True,
        embeddings=HashEmbeddingProvider(),
        embed_options=EmbedOptions(model="hash", dimension=8, batch_size=8),
        llm=llm,
        model="fake",
        timeout_seconds=30,
        lexical_auto_merge=1.0,
        embedding_auto_merge=1.0,
        candidate_threshold=0.5,
        adjudication_batch_size=1,
        seed=17,
    )

    assert result.trace["llm_adjudications"] == 3
    assert llm.resolution_call_count == 3
    assert all(not decision.same_entity for decision in result.decisions)


def test_entity_resolution_does_not_treat_semantic_relatedness_as_identity() -> None:
    """Ensure perfect embedding similarity cannot merge lexically unrelated martial arts.

    No adjudication call should be scheduled either.
    """

    mentions = [
        _mention("m1", "Karate", []),
        _mention("m2", "Judo", []),
    ]
    llm = _RecordingResolutionLlm()

    result = resolve_entity_mentions(
        mentions,
        enabled=True,
        embeddings=_IdenticalEmbeddingProvider(),
        embed_options=EmbedOptions(model="identical", dimension=4, batch_size=4),
        llm=llm,
        model="fake",
        timeout_seconds=30,
        lexical_auto_merge=0.96,
        embedding_auto_merge=0.94,
        candidate_threshold=0.80,
        embedding_lexical_floor=0.45,
        seed=17,
    )

    assert result.canonical_name_by_mention_id == {"m1": "Karate", "m2": "Judo"}
    assert result.trace["eligible_candidate_pairs"] == 0
    assert result.trace["llm_adjudications"] == 0
    assert llm.resolution_call_count == 0


def test_entity_resolution_prunes_candidate_neighborhood_and_reports_batch_progress() -> None:
    """Verify candidate pruning, concurrent batches, and aggregate progress together.

    Completion order must not affect final trace counts.
    """

    mentions = [
        _mention("m1", "Karate Alpha", []),
        _mention("m2", "Karate Beta", []),
        _mention("m3", "Karate Gamma", []),
        _mention("m4", "Karate Delta", []),
    ]
    llm = _RecordingResolutionLlm()
    progress: list[tuple[int, int, int, int]] = []

    result = resolve_entity_mentions(
        mentions,
        enabled=True,
        embeddings=_OrthogonalEmbeddingProvider(),
        embed_options=EmbedOptions(model="orthogonal", dimension=4, batch_size=4),
        llm=llm,
        model="fake",
        timeout_seconds=30,
        lexical_auto_merge=1.0,
        embedding_auto_merge=1.0,
        candidate_threshold=0.40,
        max_candidates_per_mention=1,
        adjudication_batch_size=1,
        adjudication_parallelism=2,
        batch_progress=lambda completed, total, candidates, failures: progress.append(
            (completed, total, candidates, failures)
        ),
        seed=17,
    )

    assert result.trace["eligible_candidate_pairs"] == 6
    assert result.trace["llm_adjudications"] == 2
    assert result.trace["pruned_candidate_pairs"] == 4
    assert llm.resolution_call_count == 2
    assert progress[-1] == (2, 2, 2, 0)


def test_entity_resolution_prompt_uses_proportional_bounded_output_budget() -> None:
    """Keep resolution token budgets proportional for small batches and capped for large ones."""

    mentions = [_mention("m1", "Karate Alpha", []), _mention("m2", "Karate Beta", [])]
    candidate = ResolutionCandidate(
        left_id="m1",
        right_id="m2",
        lexical_score=0.75,
        embedding_score=0.85,
    )

    small = entity_resolution_request(
        mentions,
        [candidate],
        model="fake",
        timeout_seconds=30,
        seed=17,
    )
    capped = entity_resolution_request(
        mentions,
        [candidate] * 30,
        model="fake",
        timeout_seconds=30,
        seed=17,
    )

    assert small.max_tokens == 512
    assert capped.max_tokens == 4096


def _mention(mention_id: str, name: str, aliases: list[str]) -> EntityMention:
    """Create a grounded martial-art mention with deterministic identity and evidence fields.

    Only surface and aliases vary between scenarios.
    """

    return EntityMention(
        id=mention_id,
        name=name,
        type="MARTIAL_ART",
        description=name,
        source_chunk_id="chunk",
        aliases=aliases,
        **grounded(name, name),
    )
