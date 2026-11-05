"""Stand-ins for the LLM port that remember what they were asked."""

from __future__ import annotations

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    CommunitySummaryResult,
    DescriptionMergeRequest,
    DescriptionMergeResult,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)


class CountingLlm(FakeLlmProvider):
    """Count the calls reaching each port method while answering like the fake."""

    def __init__(self) -> None:
        self.structured_calls = 0
        self.description_calls = 0
        self.community_calls = 0

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        self.structured_calls += 1
        return super().complete_structured(request)

    def merge_entity_description(self, request: DescriptionMergeRequest) -> DescriptionMergeResult:
        self.description_calls += 1
        return super().merge_entity_description(request)

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        self.community_calls += 1
        return super().summarize_community(request)


class RecordingCommunityLlm:
    """Answer every community with the same narrative and keep each request."""

    def __init__(self) -> None:
        self.requests: list[CommunitySummaryRequest] = []

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        self.requests.append(request)
        return CommunitySummaryResult(
            title=f"{request.title_seed} network",
            summary="Grounded summary.",
            rating=8.0,
            rating_explanation="High-weight employment relation.",
            findings=[("Strong connection", request.relations[0])],
            suggested_questions=["Where does Alice Smith work?"],
            provider_metadata={"provider": "recording_llm", "prompt_name": "community_report"},
        )
