"""Behavioral contracts for request-efficient graph enrichment batches."""

from __future__ import annotations

from llm_fakes import CountingLlm

from kg_processor.application.enrichment_batching import (
    merge_description_requests,
    summarize_community_requests,
)
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    CommunitySummaryResult,
    DescriptionMergeRequest,
    DescriptionMergeResult,
    RefusedChatResponseError,
    StructuredCompletionRequest,
    StructuredCompletionResult,
    TruncatedChatResponseError,
)


class _IncompleteBatchLlm(CountingLlm):
    """Return schema-valid but incomplete first records to exercise recovery."""

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        result = super().complete_structured(request)
        records = result.payload.get("results")
        assert isinstance(records, list) and isinstance(records[0], dict)
        if request.task_name == "entity_description_batch_merge":
            records[0]["description"] = ""
        elif request.task_name == "community_report_batch":
            records[0]["findings"] = []
        return result


def test_description_batch_uses_one_structured_call_and_preserves_order() -> None:
    """Reduce transport calls without changing input-to-output correspondence."""

    provider = CountingLlm()
    requests = [
        DescriptionMergeRequest(
            entity_name=f"Entity {index}",
            entity_type="CONCEPT",
            descriptions=["short", f"long description {index}"],
            evidence=[],
        )
        for index in range(16)
    ]

    results = merge_description_requests(
        provider,
        requests,
        model="fake",
        timeout_seconds=30,
        seed=17,
    )

    assert provider.structured_calls == 1
    assert provider.description_calls == 0
    assert [result.description for result in results] == [
        f"long description {index}" for index in range(16)
    ]


def test_community_batch_uses_one_structured_call_for_four_reports() -> None:
    """Batch four independently bounded community contexts in one request."""

    provider = CountingLlm()
    requests = [
        CommunitySummaryRequest(
            title_seed=title,
            members=[f"{title} member"],
            relations=[],
        )
        for title in ("First", "Second", "Third", "Fourth")
    ]

    results = summarize_community_requests(
        provider,
        requests,
        model="fake",
        timeout_seconds=30,
        seed=17,
    )

    assert provider.structured_calls == 1
    assert provider.community_calls == 0
    assert [result.title for result in results] == ["First", "Second", "Third", "Fourth"]


def test_incomplete_batch_records_fall_back_without_replaying_siblings() -> None:
    """Recover empty enrichment records individually while retaining valid rows."""

    provider = _IncompleteBatchLlm()
    descriptions = [
        DescriptionMergeRequest(
            entity_name=f"Entity {index}",
            entity_type="CONCEPT",
            descriptions=[f"description {index}"],
            evidence=[],
        )
        for index in range(2)
    ]
    communities = [
        CommunitySummaryRequest(
            title_seed=f"Community {index}",
            members=[f"member {index}"],
            # Co-mention communities can have grounded member context without an
            # internal relation. Empty findings must still recover through the
            # established single-community path.
            relations=[],
            evidence_quotes=[],
        )
        for index in range(2)
    ]

    description_results = merge_description_requests(
        provider,
        descriptions,
        model="fake",
        timeout_seconds=30,
        seed=17,
    )
    community_results = summarize_community_requests(
        provider,
        communities,
        model="fake",
        timeout_seconds=30,
        seed=17,
    )

    assert provider.structured_calls == 2
    assert provider.description_calls == 1
    assert provider.community_calls == 1
    assert all(result.description for result in description_results)
    assert all(result.summary for result in community_results)


class _DecliningLlm(CountingLlm):
    """Decline every batch, and any single record that names a declined entity."""

    def __init__(self, declined: str) -> None:
        super().__init__()
        self.declined = declined

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        self.structured_calls += 1
        raise RefusedChatResponseError("declined")

    def merge_entity_description(self, request: DescriptionMergeRequest) -> DescriptionMergeResult:
        if request.entity_name == self.declined:
            self.description_calls += 1
            raise RefusedChatResponseError("declined")
        return super().merge_entity_description(request)

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        if request.title_seed == self.declined:
            self.community_calls += 1
            raise RefusedChatResponseError("declined")
        return super().summarize_community(request)


def _description(name: str) -> DescriptionMergeRequest:
    return DescriptionMergeRequest(
        entity_name=name, entity_type="CONCEPT", descriptions=["a", "b"], evidence=[]
    )


def test_a_declined_description_batch_costs_only_the_declined_entity_its_merge() -> None:
    """The others are merged one by one; the declined one keeps what it had."""

    provider = _DecliningLlm("Entity 1")
    requests = [_description(f"Entity {index}") for index in range(3)]

    results = merge_description_requests(
        provider, requests, model="fake", timeout_seconds=30, seed=17
    )

    assert provider.structured_calls == 1
    assert [bool(result.description) for result in results] == [True, False, True]
    assert results[1].provider_metadata == {"refused": True}


def test_a_declined_community_keeps_a_structural_report() -> None:
    provider = _DecliningLlm("Declined")
    requests = [
        CommunitySummaryRequest(title_seed=title, members=["A", "B"], relations=["A -> B"])
        for title in ("Kept", "Declined")
    ]

    results = summarize_community_requests(
        provider, requests, model="fake", timeout_seconds=30, seed=17
    )

    assert results[0].provider_metadata.get("refused") is None
    assert results[1].title == "Declined"
    assert results[1].findings == []
    assert results[1].provider_metadata == {"refused": True}


class _OverlongLlm(CountingLlm):
    """A provider that cannot fit a batch, nor the one record named overlong."""

    def __init__(self, overlong: str) -> None:
        super().__init__()
        self.overlong = overlong

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        self.structured_calls += 1
        raise TruncatedChatResponseError("stopped at the output-token limit")

    def merge_entity_description(self, request: DescriptionMergeRequest) -> DescriptionMergeResult:
        if request.entity_name == self.overlong:
            raise TruncatedChatResponseError("stopped at the output-token limit")
        return super().merge_entity_description(request)

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        if request.title_seed == self.overlong:
            raise TruncatedChatResponseError("stopped at the output-token limit")
        return super().summarize_community(request)


def test_a_cut_off_description_batch_is_merged_one_by_one() -> None:
    """A too-long reply costs only the entity that cannot fit on its own."""

    provider = _OverlongLlm("Entity 1")
    requests = [_description(f"Entity {index}") for index in range(3)]

    results = merge_description_requests(
        provider, requests, model="fake", timeout_seconds=30, seed=17
    )

    assert provider.structured_calls == 1
    assert [bool(result.description) for result in results] == [True, False, True]
    assert results[1].provider_metadata == {"truncated": True}


def test_a_cut_off_community_batch_says_which_report_could_not_fit() -> None:
    provider = _OverlongLlm("Overlong")
    requests = [
        CommunitySummaryRequest(title_seed=title, members=["A", "B"], relations=["A -> B"])
        for title in ("Kept", "Overlong")
    ]

    results = summarize_community_requests(
        provider, requests, model="fake", timeout_seconds=30, seed=17
    )

    assert provider.structured_calls == 1
    assert results[0].provider_metadata.get("truncated") is None
    assert results[1].title == "Overlong"
    assert results[1].findings == []
    assert results[1].rating_explanation == "The model's report ran past the output limit."
    assert results[1].provider_metadata == {"truncated": True}


def test_any_other_failure_still_fails_the_batch() -> None:
    """Only a refusal is optional; a broken provider is an error."""

    class _Broken(CountingLlm):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            raise ValueError("malformed")

    try:
        merge_description_requests(
            _Broken(),
            [_description("A"), _description("B")],
            model="fake",
            timeout_seconds=30,
            seed=17,
        )
    except ValueError as error:
        assert str(error) == "malformed"
    else:
        raise AssertionError("a malformed batch must not be swallowed")
