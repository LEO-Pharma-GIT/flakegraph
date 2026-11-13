# SPDX-License-Identifier: Apache-2.0
"""Normalization helpers for provider-neutral extraction results."""

from __future__ import annotations

from kg_processor.domain.graph import ExtractedEntity, ExtractedRelation, ExtractionResult


def dedupe_extraction_result(result: ExtractionResult) -> ExtractionResult:
    """Remove duplicate observations while preserving deterministic source order.

    Entity identity includes source chunk and declared type. Relation identity also
    includes its normalized predicate, so repeated evidence from different chunks
    remains available to downstream graph assembly.
    """

    # A folder mention is a document's tie to its folder, kept beside a mention
    # the text makes of the same entity in the same chunk.
    seen_entities: set[tuple[str, str, str, bool]] = set()
    entities: list[ExtractedEntity] = []
    for entity in result.entities:
        entity_key = (
            entity.resolved_component_id or _normalize(entity.name),
            entity.type,
            entity.source_chunk_id,
            entity.evidence_method == "folder",
        )
        if entity_key in seen_entities:
            continue
        seen_entities.add(entity_key)
        entities.append(entity)

    seen_relations: set[tuple[str, str, str, str, bool]] = set()
    relations: list[ExtractedRelation] = []
    for relation in result.relations:
        relation_key = (
            relation.source_component_id or _normalize(relation.source_name),
            relation.target_component_id or _normalize(relation.target_name),
            relation.relation_type.strip().lower(),
            relation.source_chunk_id,
            relation.source_self_loop,
        )
        if relation_key in seen_relations:
            continue
        seen_relations.add(relation_key)
        relations.append(relation)

    return ExtractionResult(
        entities=entities,
        relations=relations,
        held_relations=result.held_relations,
        provider_metadata=result.provider_metadata,
    )


def _normalize(value: str) -> str:
    """Normalize an observation label for case-insensitive identity checks."""

    return " ".join(value.lower().split())
