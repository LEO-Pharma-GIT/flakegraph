# SPDX-License-Identifier: Apache-2.0
"""List every entity and relation validation turned away, and why.

Both engines derive these rows from the extraction trace, as they do the
discarded windows: the local pipeline from its in-process trace, the Spark
finalizer from its trace column. The summary names the relation type rules the
model broke most, so an ontology's author can see which rules cost true facts.

A relation held for type resolution leaves its record here too, so it is not
lost if finalization does not admit it; finalization takes back the records of
the relations it admits, and what remains is what stayed rejected.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from typing import Any

from kg_processor.domain.extraction import EntityMention, RelationObservation
from kg_processor.domain.graph import Chunk, RejectedRecord
from kg_processor.domain.ids import stable_id

# The passes whose rejected records are listed: extraction turns away what does
# not ground, verification what the text does not state, and the cascade the
# relations whose entity verification rejected. A revalidation lists what it
# lost around the passes it replays: records in a window the current rules skip.
REJECTING_STAGES = frozenset(
    {
        "entity_extraction",
        "document_context_extraction",
        "relation_extraction",
        "relation_verification",
        "relation_verification_error",
        "verification_cascade",
        "revalidation",
    }
)
REJECTED_RECORD_FIELDS = (
    "name",
    "type",
    "source",
    "source_type",
    "target",
    "target_type",
    "quote",
    "chunk_id",
)
# How many broken type rules a run report names.
MAX_REPORTED_TYPE_RULES = 25
# Characters of a rejected record's quote a trace carries.
REJECTED_QUOTE_CHARACTERS = 400
# Characters of a rejected candidate's full quote kept for revalidation: the
# longest quote a relation response may carry.
CANDIDATE_QUOTE_CHARACTERS = 1200


def candidate_detail(quote: str, **fields: object) -> dict[str, object]:
    """The rest of a rejected candidate, so a later revalidation can rebuild it whole.

    The listed record is what a reader sees; this is what was proposed beyond
    it. The quote is repeated only when the listed one was cut, and is bounded
    like the response schema bounds it.
    """

    detail = dict(fields)
    if len(quote) > REJECTED_QUOTE_CHARACTERS:
        detail["quote"] = quote[:CANDIDATE_QUOTE_CHARACTERS]
    return detail


def mention_record(mention: EntityMention, reason: str) -> dict[str, object]:
    """A trace's rejected record for a mention that was grounded, then turned away."""

    return {
        "reason": reason,
        "kind": "entity",
        "name": mention.name,
        "type": mention.type,
        "quote": mention.quote[:REJECTED_QUOTE_CHARACTERS],
        "chunk_id": mention.source_chunk_id,
        "candidate": candidate_detail(
            mention.quote,
            description=mention.description,
            confidence=mention.confidence,
            aliases=list(mention.aliases),
            evidence_method=mention.evidence_method,
        ),
    }


def observation_record(
    relation: RelationObservation,
    mentions_by_id: Mapping[str, EntityMention],
    reason: str,
) -> dict[str, object]:
    """A trace's rejected record for a relation that was grounded, then turned away."""

    source = mentions_by_id.get(relation.source_entity_id)
    target = mentions_by_id.get(relation.target_entity_id)
    return {
        "reason": reason,
        "kind": "relation",
        "name": relation.relation_type,
        "source": source.name if source else relation.source_surface or "",
        "source_type": source.type if source else "",
        "target": target.name if target else relation.target_surface or "",
        "target_type": target.type if target else "",
        "quote": relation.quote[:REJECTED_QUOTE_CHARACTERS],
        "chunk_id": relation.source_chunk_id,
        "candidate": candidate_detail(
            relation.quote,
            description=relation.description,
            confidence=relation.confidence,
            source_surface=relation.source_surface or "",
            target_surface=relation.target_surface or "",
            quantities=[
                {
                    "text": item.text,
                    "kind": item.kind or "",
                    "unit": item.unit or "",
                    "comparator": item.comparator or "",
                }
                for item in relation.quantities
            ],
            evidence_method=relation.evidence_method,
            # The type the model stated, when an ontology rule kept it under another.
            proposed_relation_type=relation.proposed_relation_type or "",
            assertion=relation.assertion,
        ),
    }


def rejected_record_key(
    window_id: str,
    kind: str,
    reason: str,
    record: Mapping[str, object],
) -> str:
    """Identify one rejected record apart from the graph it lands in.

    A held relation carries this key out of its window, before any graph id is
    known, so finalization can take back the record of a relation it admits.
    The parts are the row id's parts, read the way the rows read them.
    """

    return stable_id(
        "rejected_record_key",
        window_id,
        kind,
        reason,
        *(str(record.get(field) or "") for field in REJECTED_RECORD_FIELDS),
    )


def rejected_records_from_trace(
    trace: Iterable[Mapping[str, Any]],
    chunks: Iterable[Chunk],
    graph_id: str,
    taken_back: Collection[str] = (),
) -> list[RejectedRecord]:
    """Turn the trace's rejected records into rows, one per distinct record.

    A record rejected again by a later pass over the same text is one row. A
    record whose key is in ``taken_back`` belongs to a held relation the graph
    admitted, so it was not lost and has no row.
    """

    by_id = {chunk.id: chunk for chunk in chunks}
    rows: dict[str, RejectedRecord] = {}
    for event in trace:
        if str(event.get("stage") or "") not in REJECTING_STAGES:
            continue
        window_id = str(event.get("window_id") or "")
        for record in event.get("rejected_records") or []:
            if not isinstance(record, Mapping):
                continue
            values = {field: str(record.get(field) or "") for field in REJECTED_RECORD_FIELDS}
            kind = "relation" if record.get("kind") == "relation" else "entity"
            reason = str(record.get("reason") or "rejected")
            if taken_back and rejected_record_key(window_id, kind, reason, values) in taken_back:
                continue
            chunk = by_id.get(values["chunk_id"])
            document_id = str(event.get("document_id") or (chunk.document_id if chunk else ""))
            row_id = stable_id(
                "rejected_record",
                graph_id,
                window_id,
                kind,
                reason,
                *values.values(),
            )
            rows.setdefault(
                row_id,
                RejectedRecord(
                    id=row_id,
                    graph_id=graph_id,
                    document_id=document_id,
                    file_id=chunk.file_id if chunk else document_id,
                    window_id=window_id,
                    page_number=chunk.page_number if chunk else None,
                    kind=kind,
                    reason=reason,
                    **values,
                ),
            )
    return sorted(
        rows.values(),
        key=lambda row: (row.document_id, row.page_number or 0, row.kind, row.reason, row.id),
    )


def rejected_record_metrics(rows: Iterable[RejectedRecord]) -> dict[str, Any]:
    """The summary a run report carries: what was rejected, why, and which rules."""

    by_kind: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    rules: Counter[tuple[str, str, str]] = Counter()
    documents: set[str] = set()
    for row in rows:
        by_kind[row.kind] += 1
        by_reason[f"{row.kind}:{row.reason}"] += 1
        documents.add(row.document_id)
        if row.kind == "relation" and row.reason == "domain_or_range_violation":
            rules[(row.source_type, row.name, row.target_type)] += 1
    return {
        "records": sum(by_kind.values()),
        "documents": len(documents),
        "by_kind": dict(sorted(by_kind.items())),
        "by_reason": dict(sorted(by_reason.items(), key=lambda item: (-item[1], item[0]))),
        "broken_type_rules": [
            {"source_type": source, "relation_type": relation, "target_type": target, "count": n}
            for (source, relation, target), n in sorted(
                rules.items(), key=lambda item: (-item[1], *item[0])
            )[:MAX_REPORTED_TYPE_RULES]
        ],
    }
