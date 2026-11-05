# SPDX-License-Identifier: Apache-2.0
"""Deterministic LLM provider for tests and dry-run graph shape checks."""

from __future__ import annotations

import json
import re
from typing import Any

from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    CommunitySummaryResult,
    DescriptionMergeRequest,
    DescriptionMergeResult,
    LlmCapabilities,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)

# Restrict multi-token fake entities to horizontal whitespace. Using ``\s``
# would let a line break join the last capitalized term on one line with the
# first capitalized term on the next line, which makes local smoke artifacts
# harder to review even though the provider is only deterministic test tooling.
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]*){0,3}\b")
_STOP_NAMES = {"The", "This", "That", "A", "An", "It", "In", "On", "For", "And"}
_MIN_ENTITY_NAME_LEN = 3
_PERSON_MIN_TOKENS = 2
_ORG_MARKERS = ("Inc", "Corp", "Ltd", "LLC")


class FakeLlmProvider:
    """Provide deterministic graph behavior for tests and credential-free smoke runs.

    The provider derives records from each request instead of replaying canned
    fixtures. It therefore exercises serialization and orchestration contracts
    without claiming to represent real model quality.
    """

    def capabilities(self) -> LlmCapabilities:
        """Advertise strict, deterministic behavior available to test orchestration.

        Seed support is reported because identical inputs always produce the
        same output, allowing concurrency and caching tests to remain repeatable.
        """

        return LlmCapabilities(
            supports_seed=True,
            max_output_tokens=8192,
        )

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Return deterministic records matching the two-pass task contract.

        The fake adapter intentionally derives data from the request payload. It
        therefore exercises prompt serialization and orchestration without
        pretending that canned model responses validate real provider behavior.
        """

        input_payload = _structured_input(request.user)
        if request.task_name.startswith("entity_extraction"):
            payload = _fake_entity_payload(input_payload)
        elif request.task_name.startswith("relation_extraction"):
            payload = _fake_relation_payload(input_payload)
        elif request.task_name.startswith("relation_verification"):
            payload = _fake_verification_payload(input_payload)
        elif request.task_name.startswith("entity_resolution"):
            payload = _fake_resolution_payload(input_payload)
        elif request.task_name == "entity_description_batch_merge":
            payload = _fake_description_batch_payload(input_payload)
        elif request.task_name == "community_report_batch":
            payload = _fake_community_batch_payload(input_payload)
        else:
            payload = _empty_payload_for_schema(request.json_schema)
        return StructuredCompletionResult(
            payload=payload,
            provider_metadata={
                "provider": "fake_llm",
                "model": request.model,
                "task_name": request.task_name,
                **request.prompt_metadata,
            },
        )

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        """Choose the longest provided description as a deterministic merge."""

        description = max(request.descriptions, key=len, default="")
        return DescriptionMergeResult(
            description=description,
            provider_metadata={
                "provider": "fake_llm",
                "model": "fake-description-merge",
            },
        )

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        """Return a deterministic community summary for local smoke tests."""

        members = ", ".join(request.members[:5])
        title = request.title_seed or (request.members[0] if request.members else "Community")
        summary = f"Community around {members}." if members else "Empty community."
        return CommunitySummaryResult(
            title=title[:160],
            summary=summary,
            rating=5.0,
            rating_explanation="The deterministic test provider assigns a neutral score.",
            findings=[("Connected entities", summary)],
            suggested_questions=[f"What connects {title}?"[:160]],
            provider_metadata={
                "provider": "fake_llm",
                "model": "fake-community-summary",
            },
        )


def _extract_names(text: str) -> list[str]:
    names = []
    for match in _ENTITY_RE.findall(text):
        cleaned = match.strip()
        if cleaned in _STOP_NAMES or len(cleaned) < _MIN_ENTITY_NAME_LEN:
            continue
        names.append(cleaned)
    return names


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _infer_type(name: str, allowed: list[str]) -> str:
    if "ORGANIZATION" in allowed and any(token in name for token in _ORG_MARKERS):
        return "ORGANIZATION"
    if "PERSON" in allowed and len(name.split()) >= _PERSON_MIN_TOKENS:
        return "PERSON"
    if not allowed:
        return "CONCEPT"
    return "CONCEPT" if "CONCEPT" in allowed else allowed[0]


def _structured_input(user_prompt: str) -> dict[str, Any]:
    """Read the machine payload appended by the shared prompt builder.

    Returning an empty mapping for prompts without the marker keeps unknown-task
    smoke paths deterministic while malformed marked JSON still fails visibly.
    """

    marker = "INPUT_JSON:\n"
    raw_payload = (
        user_prompt.rsplit(marker, maxsplit=1)[1].strip()
        if marker in user_prompt
        else user_prompt.strip()
    )
    if not raw_payload.startswith("{"):
        return {}
    parsed = json.loads(raw_payload)
    return parsed if isinstance(parsed, dict) else {}


def _fake_description_batch_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Choose the longest supplied description for every batch record."""

    records = payload.get("records", [])
    if not isinstance(records, list):
        return {"results": []}
    results = []
    for record in records:
        if not isinstance(record, dict):
            continue
        descriptions = record.get("descriptions", [])
        candidates = descriptions if isinstance(descriptions, list) else []
        results.append(
            {
                "record_id": str(record.get("record_id", "")),
                "description": max(
                    (str(value) for value in candidates),
                    key=len,
                    default="",
                ),
            }
        )
    return {"results": results}


def _fake_community_batch_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Produce deterministic community reports for batched smoke-test records."""

    records = payload.get("records", [])
    if not isinstance(records, list):
        return {"results": []}
    results = []
    for record in records:
        if not isinstance(record, dict):
            continue
        raw_members = record.get("members", [])
        members = raw_members if isinstance(raw_members, list) else []
        title = str(record.get("title_seed", "Community"))
        member_text = ", ".join(str(value) for value in members[:5])
        summary = f"Community around {member_text}." if member_text else "Empty community."
        results.append(
            {
                "record_id": str(record.get("record_id", "")),
                "title": title[:160],
                "summary": summary,
                "rating": 5.0,
                "rating_explanation": "Deterministic structural test summary.",
                "findings": [{"summary": "Connected entities", "explanation": summary}],
                "suggested_questions": [f"What connects {title}?"[:160]],
            }
        )
    return {"results": results}


def _fake_entity_payload(input_payload: dict[str, Any]) -> dict[str, Any]:
    """Derive grounded entity records from title-cased mentions in supplied chunks.

    The payload mirrors the strict entity-stage schema.
    """

    entity_types = [str(value) for value in input_payload.get("entity_types", [])]
    entities: list[dict[str, Any]] = []
    for chunk in input_payload.get("chunks", []):
        if not isinstance(chunk, dict):
            continue
        content = str(chunk.get("content", ""))
        chunk_id = str(chunk.get("id", ""))
        for name in _ordered_unique(_extract_names(content)):
            entities.append(
                {
                    "name": name,
                    "type": _infer_type(name, entity_types),
                    "description": f"{name} is mentioned in the source text.",
                    "source_chunk_id": chunk_id,
                    "quote": name,
                    "confidence": 1.0,
                    "aliases": [],
                }
            )
    return {"entities": entities}


def _fake_relation_payload(input_payload: dict[str, Any]) -> dict[str, Any]:
    """Connect adjacent fake entities per chunk using an ontology-compatible relation.

    Relation prompts contain a document-wide canonical inventory, so an entity's
    representative ``source_chunk_id`` need not be the chunk currently under
    review. Matching names and aliases against each supplied chunk mirrors that
    production contract and still prevents unsupported cross-chunk edges.
    """

    entities = [item for item in input_payload.get("entities", []) if isinstance(item, dict)]
    relation_types = [str(value) for value in input_payload.get("relation_types", [])]
    relation_type = relation_types[0] if relation_types else "RELATED_TO"
    relations: list[dict[str, Any]] = []
    for chunk in input_payload.get("chunks", []):
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("id", ""))
        content = str(chunk.get("content", ""))
        chunk_entities = sorted(
            (entity for entity in entities if _entity_position(entity, content) is not None),
            key=lambda entity: (
                _entity_position(entity, content),
                str(entity.get("id", "")),
            ),
        )
        for source, target in zip(chunk_entities, chunk_entities[1:], strict=False):
            relations.append(
                {
                    "source_entity_id": str(source.get("id", "")),
                    "target_entity_id": str(target.get("id", "")),
                    "source_surface": str(source.get("name", "")),
                    "target_surface": str(target.get("name", "")),
                    "relation_type": relation_type,
                    "description": (
                        f"{source.get('name', '')} is mentioned near {target.get('name', '')}."
                    ),
                    "source_chunk_id": chunk_id,
                    "quote": content,
                    "confidence": 1.0,
                }
            )
    return {"relations": relations}


def _entity_position(entity: dict[str, Any], content: str) -> int | None:
    """Return the first grounded inventory surface position in one source chunk."""

    raw_aliases = entity.get("aliases", [])
    aliases = raw_aliases if isinstance(raw_aliases, list) else []
    surfaces = [str(entity.get("name", "")), *(str(value) for value in aliases)]
    folded_content = content.casefold()
    positions = [folded_content.find(surface.casefold()) for surface in surfaces if surface]
    grounded = [position for position in positions if position >= 0]
    return min(grounded) if grounded else None


def _fake_verification_payload(input_payload: dict[str, Any]) -> dict[str, Any]:
    """Mark every supplied fake relation supported and entity specific for verifier tests.

    IDs are copied exactly from the request payload.
    """

    return {
        "decisions": [
            {
                "relation_id": str(relation.get("id", "")),
                "verdict": "supported",
                "reason": "none",
                "confidence": 1.0,
            }
            for relation in input_payload.get("relations", [])
            if isinstance(relation, dict)
        ],
        "entity_decisions": [
            {
                "entity_id": str(entity.get("id", "")),
                "verdict": "specific",
                "type": str(entity.get("type", "")),
            }
            for entity in input_payload.get("entities_to_verify", [])
            if isinstance(entity, dict)
        ],
    }


def _fake_resolution_payload(input_payload: dict[str, Any]) -> dict[str, Any]:
    """Keep ambiguous fake candidates distinct after earlier deterministic identity merges.

    This protects smoke graphs from synthetic over-merging.
    """

    return {
        "decisions": [
            {
                "left_id": str(candidate.get("left_id", "")),
                "right_id": str(candidate.get("right_id", "")),
                "same_entity": False,
                "canonical_name": None,
                "confidence": 1.0,
                "reason": "Distinct unless deterministic normalization already merged them.",
            }
            for candidate in input_payload.get("candidates", [])
            if isinstance(candidate, dict)
        ]
    }


def _empty_payload_for_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a minimally valid object for an unrecognized structured task.

    Only top-level array properties can be synthesized without inventing domain
    semantics, so other property shapes are deliberately omitted.
    """

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return {}
    return {
        str(name): []
        for name, definition in properties.items()
        if isinstance(definition, dict) and definition.get("type") == "array"
    }
