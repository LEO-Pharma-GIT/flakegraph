# SPDX-License-Identifier: Apache-2.0
"""Batch independent LLM enrichment records behind provider-neutral contracts.

Entity descriptions and community reports are naturally independent, but sending
one HTTP request per graph row makes transport latency dominate finalization.
This module combines a small number of records into one strict structured request,
restores input order by an opaque record ID, and falls back only for records a
provider omitted. It contains no transport or graph-assembly logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from kg_processor.application.prompt_registry import (
    community_report_batch_prompt,
    entity_description_batch_merge_prompt,
    prompt_metadata,
)
from kg_processor.ports.llm import (
    CommunitySummaryProvider,
    CommunitySummaryRequest,
    CommunitySummaryResult,
    DescriptionMergeProvider,
    DescriptionMergeRequest,
    DescriptionMergeResult,
    RefusedChatResponseError,
    StructuredCompletionProvider,
    StructuredCompletionRequest,
    TruncatedChatResponseError,
)

# Description requests are individually bounded by graph settings. Sixteen
# records keep the strict response below the shared output-token budget while
# amortizing transport latency across enough independent canonical nodes.
DESCRIPTION_BATCH_SIZE = 16
# Community contexts are independently bounded to 50 members, 25 relations, and
# 50 evidence quotes. Four records remain well inside the supported model context
# and take half the request round trips of a two-record pack.
# Every response carries an opaque record ID and falls back independently,
# so one incomplete item cannot discard successful neighbors.
COMMUNITY_BATCH_SIZE = 4
# A batch can still outrun the output budget. The same request would stop at the
# same place, so a refused or cut-off batch is asked for again one record at a time.
_BATCH_NOT_ANSWERED = (RefusedChatResponseError, TruncatedChatResponseError)
_MAX_DESCRIPTION_LENGTH = 1000
_MAX_TITLE_LENGTH = 160


def merge_description_requests(
    llm: DescriptionMergeProvider,
    requests: Sequence[DescriptionMergeRequest],
    *,
    model: str,
    timeout_seconds: int,
    seed: int | None,
) -> list[DescriptionMergeResult]:
    """Merge a bounded request batch while preserving one result per input.

    A single input uses the adapter's established enrichment method. Larger
    batches use the universal strict-completion port, reducing round trips without
    requiring provider-specific batch APIs. Missing or duplicate record IDs are
    recovered individually so batching cannot silently delete node descriptions.
    A batch the model declines is asked again record by record, so one entity it
    will not describe leaves its own observed descriptions in place and costs
    its neighbours nothing.
    """

    if len(requests) <= 1 or not isinstance(llm, StructuredCompletionProvider):
        return [_merge_one(llm, request) for request in requests]
    records = [
        {"record_id": _record_id(index), **request.model_dump(mode="json")}
        for index, request in enumerate(requests)
    ]
    prompt = entity_description_batch_merge_prompt(records)
    try:
        completion = llm.complete_structured(
            StructuredCompletionRequest(
                task_name="entity_description_batch_merge",
                model=model,
                system=prompt.system,
                user=prompt.user,
                json_schema=_description_batch_schema(len(records)),
                timeout_seconds=timeout_seconds,
                temperature=0.0,
                max_tokens=min(8192, max(2048, len(records) * 1024)),
                seed=seed,
                prompt_metadata=prompt_metadata(prompt),
            )
        )
    except _BATCH_NOT_ANSWERED:
        return [_merge_one(llm, request) for request in requests]
    returned = _unique_result_records(completion.payload)
    metadata = {
        **completion.provider_metadata,
        "batch_size": len(records),
        **prompt_metadata(prompt),
    }
    results: list[DescriptionMergeResult] = []
    for index, request in enumerate(requests):
        item = returned.get(_record_id(index))
        if (
            item is None
            or not isinstance(item.get("description"), str)
            or not str(item["description"]).strip()
        ):
            results.append(_merge_one(llm, request))
            continue
        results.append(
            DescriptionMergeResult(
                description=str(item["description"]).strip()[:_MAX_DESCRIPTION_LENGTH],
                provider_metadata=metadata,
            )
        )
    return results


def summarize_community_requests(
    llm: CommunitySummaryProvider,
    requests: Sequence[CommunitySummaryRequest],
    *,
    model: str,
    timeout_seconds: int,
    seed: int | None,
) -> list[CommunitySummaryResult]:
    """Generate bounded community-report batches with per-record recovery.

    Community contexts are larger than description contexts, so the caller uses a
    deliberately small batch. Structural ratings remain computed by graph logic;
    the model-generated rating is normalized only to satisfy the shared result
    contract and is not used to rank the persisted graph. A batch the model
    declines is asked again record by record, and a community it will not
    summarize keeps a structural report.
    """

    if len(requests) <= 1 or not isinstance(llm, StructuredCompletionProvider):
        return [_summarize_one(llm, request) for request in requests]
    records = [
        {"record_id": _record_id(index), **request.model_dump(mode="json")}
        for index, request in enumerate(requests)
    ]
    prompt = community_report_batch_prompt(records)
    try:
        completion = llm.complete_structured(
            StructuredCompletionRequest(
                task_name="community_report_batch",
                model=model,
                system=prompt.system,
                user=prompt.user,
                json_schema=_community_batch_schema(len(records)),
                timeout_seconds=timeout_seconds,
                temperature=0.0,
                max_tokens=8192,
                seed=seed,
                prompt_metadata=prompt_metadata(prompt),
            )
        )
    except _BATCH_NOT_ANSWERED:
        return [_summarize_one(llm, request) for request in requests]
    returned = _unique_result_records(completion.payload)
    metadata = {
        **completion.provider_metadata,
        "batch_size": len(records),
        **prompt_metadata(prompt),
    }
    results: list[CommunitySummaryResult] = []
    for index, request in enumerate(requests):
        item = returned.get(_record_id(index))
        normalized = _community_result(item, request, metadata) if item is not None else None
        results.append(normalized or _summarize_one(llm, request))
    return results


def _merge_one(
    llm: DescriptionMergeProvider, request: DescriptionMergeRequest
) -> DescriptionMergeResult:
    """Merge one entity's descriptions, or leave them as observed if unanswered.

    An empty result is what every caller already treats as no merge, so the
    node keeps the description it had.
    """

    try:
        return llm.merge_entity_description(request)
    except RefusedChatResponseError:
        return DescriptionMergeResult(description="", provider_metadata={"refused": True})
    except TruncatedChatResponseError:
        return DescriptionMergeResult(description="", provider_metadata={"truncated": True})


def _summarize_one(
    llm: CommunitySummaryProvider, request: CommunitySummaryRequest
) -> CommunitySummaryResult:
    """Summarize one community, or give it a structural report if unanswered."""

    try:
        return llm.summarize_community(request)
    except RefusedChatResponseError:
        return _structural_report(
            request, "The model declined to summarize this community.", "refused"
        )
    except TruncatedChatResponseError:
        return _structural_report(
            request, "The model's report ran past the output limit.", "truncated"
        )


def _structural_report(
    request: CommunitySummaryRequest, explanation: str, outcome: str
) -> CommunitySummaryResult:
    """Describe a community without a model's report, and say why it has none."""

    return CommunitySummaryResult(
        title=request.title_seed,
        summary=f"Community of {len(request.members)} listed entities.",
        rating=0.0,
        rating_explanation=explanation,
        findings=[],
        provider_metadata={outcome: True},
    )


def _record_id(index: int) -> str:
    """Return a compact positional identity that carries no graph information."""

    return f"record-{index:04d}"


def _unique_result_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index only well-shaped, uniquely identified structured result records."""

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for raw in raw_results:
        if not isinstance(raw, dict) or not isinstance(raw.get("record_id"), str):
            continue
        record_id = str(raw["record_id"])
        if record_id in indexed:
            duplicates.add(record_id)
        else:
            indexed[record_id] = raw
    for record_id in duplicates:
        indexed.pop(record_id, None)
    return indexed


def _community_result(
    item: dict[str, Any],
    request: CommunitySummaryRequest,
    metadata: dict[str, Any],
) -> CommunitySummaryResult | None:
    """Normalize one complete batch item or request single-record recovery.

    Strict JSON shape alone does not guarantee useful enrichment: a model can
    return blank prose or an empty findings array. Co-mention communities may
    legitimately have no internal relation edge, but their grounded member
    descriptions still support useful findings and the established single-record
    path summarizes them. Rejecting every empty finding set preserves that
    behavior without discarding successful siblings in a batch.
    """

    title = str(item.get("title", request.title_seed)).strip()
    summary = item.get("summary")
    if not title or not isinstance(summary, str) or not summary.strip():
        return None
    raw_findings = item.get("findings", [])
    raw_questions = item.get("suggested_questions", [])
    findings = raw_findings if isinstance(raw_findings, list) else []
    questions = raw_questions if isinstance(raw_questions, list) else []
    normalized_findings = [
        (str(finding.get("summary", "")), str(finding.get("explanation", "")))
        for finding in findings
        if isinstance(finding, dict)
        and str(finding.get("summary", "")).strip()
        and str(finding.get("explanation", "")).strip()
    ]
    if not normalized_findings:
        return None
    rating = item.get("rating", 0.0)
    try:
        normalized_rating = min(10.0, max(0.0, float(rating)))
    except (TypeError, ValueError):
        normalized_rating = 0.0
    return CommunitySummaryResult(
        title=title[:_MAX_TITLE_LENGTH],
        summary=summary.strip(),
        rating=normalized_rating,
        rating_explanation=str(item.get("rating_explanation", "")),
        findings=normalized_findings,
        suggested_questions=[str(question) for question in questions if isinstance(question, str)],
        provider_metadata=metadata,
    )


def _description_batch_schema(record_count: int) -> dict[str, Any]:
    """Return a strict schema bounded to the submitted description record count."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "minItems": record_count,
                "maxItems": record_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "record_id": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["record_id", "description"],
                },
            }
        },
        "required": ["results"],
    }


def _community_batch_schema(record_count: int) -> dict[str, Any]:
    """Return the strict multi-community response schema for one provider call."""

    finding = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "explanation": {"type": "string"},
        },
        "required": ["summary", "explanation"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "minItems": record_count,
                "maxItems": record_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "record_id": {"type": "string"},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "rating": {"type": "number", "minimum": 0, "maximum": 10},
                        "rating_explanation": {"type": "string"},
                        "findings": {"type": "array", "items": finding},
                        "suggested_questions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "record_id",
                        "title",
                        "summary",
                        "rating",
                        "rating_explanation",
                        "findings",
                        "suggested_questions",
                    ],
                },
            }
        },
        "required": ["results"],
    }
