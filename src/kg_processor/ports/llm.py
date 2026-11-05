# SPDX-License-Identifier: Apache-2.0
"""LLM provider port for structured extraction and graph enrichment.

LLM adapters submit prompts and normalize structured responses; they do not own
GraphRAG behavior. Provider metadata is returned with every LLM-assisted step so
traces can explain which model/prompt revision produced each graph artifact.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from kg_processor.domain.consumption import TokenUsage

# Keep direct port users aligned with the application settings. These values
# are intentionally conservative across local servers and hosted APIs.
# A provider call that has produced no response for several minutes is more
# likely stalled than productively decoding. Bounding the attempt keeps rare
# tails from serializing large queue or Spark stages; slower local deployments
# can raise ``llm.timeout_seconds`` explicitly without changing adapter behavior.
DEFAULT_LLM_TIMEOUT_SECONDS = 180


class TruncatedChatResponseError(RuntimeError):
    """A provider reply that stopped at the request's output-token limit.

    Regenerating the identical request truncates identically, so a provider
    raises this instead of retrying; a caller that can ask for less - fewer
    records in one call - can split the request instead.
    """


class RefusedChatResponseError(RuntimeError):
    """A provider reply its model or content filter declined to finish.

    The refusal follows from what the request asked about, so every attempt
    has been spent before this is raised. A caller whose answer is optional
    enrichment can go on without it; any other failure stays an error.
    """


class LlmCapabilities(BaseModel):
    """Describe machine-readable provider features used by extraction orchestration.

    Capability checks permit modular fallback or validation without branching on
    concrete provider names.
    """

    supports_seed: bool = False
    max_output_tokens: int = 8192


class StructuredCompletionRequest(BaseModel):
    """Define one provider-neutral strict JSON Schema completion request.

    It carries deterministic generation controls and prompt provenance alongside
    the schema, while adapters own only transport-specific serialization.
    """

    task_name: str
    model: str
    system: str
    user: str
    json_schema: dict[str, Any]
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS
    temperature: float = 0.0
    max_tokens: int = 4096
    seed: int | None = None
    prompt_metadata: dict[str, str] = Field(default_factory=dict)


class StructuredCompletionResult(BaseModel):
    """Return a validated JSON object with transport and model provenance.

    Domain-specific interpretation remains in extraction-stage adapters.
    """

    payload: dict[str, Any]
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)


class CommunitySummaryRequest(BaseModel):
    """Inputs used by an LLM to turn a detected community into report text."""

    title_seed: str
    members: list[str]
    relations: list[str]
    evidence_quotes: list[str] = Field(default_factory=list)


class CommunitySummaryResult(BaseModel):
    """Normalized community summary returned by an LLM provider."""

    title: str
    summary: str
    rating: float = Field(ge=0.0, le=10.0)
    rating_explanation: str = ""
    findings: list[tuple[str, str]]
    suggested_questions: list[str] = Field(default_factory=list)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)


class DescriptionMergeRequest(BaseModel):
    """Inputs used to merge several entity observations into one description."""

    entity_name: str
    entity_type: str
    descriptions: list[str]
    evidence: list[str]


class DescriptionMergeResult(BaseModel):
    """Normalized merged description plus provider trace metadata."""

    description: str
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)


@runtime_checkable
class StructuredCompletionProvider(Protocol):
    """Define the low-level strict-output capability required by modular extraction.

    Runtime-checkability lets orchestration validate custom adapters without
    importing or enumerating concrete implementations.
    """

    def capabilities(self) -> LlmCapabilities:
        """Return structured-output and determinism features supported by the adapter.

        Callers use the result to bound requests and avoid unsupported controls.
        """
        ...

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Execute one strict JSON Schema completion without task-specific semantics.

        The adapter validates transport shape; application stages validate domain meaning.
        """
        ...


class DescriptionMergeProvider(Protocol):
    """Provide canonical descriptions from repeated grounded observations."""

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        """Merge repeated entity descriptions into a canonical node description."""
        ...


class CommunitySummaryProvider(Protocol):
    """Generate narrative reports for detected graph communities."""

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        """Generate a short report for a detected graph community."""
        ...


class LlmProvider(
    StructuredCompletionProvider,
    DescriptionMergeProvider,
    CommunitySummaryProvider,
    Protocol,
):
    """Complete extraction tasks and produce optional graph enrichment text.

    Extraction algorithms live in the application layer and communicate through
    strict JSON Schema requests. Adapters therefore implement one transport-level
    completion method regardless of the model service they connect to.
    """
