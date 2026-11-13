from __future__ import annotations

from kg_processor.application.rejected_records import (
    rejected_record_metrics,
    rejected_records_from_trace,
)
from kg_processor.domain.graph import Chunk


def _chunk(chunk_id: str, page: int) -> Chunk:
    return Chunk(
        id=chunk_id,
        file_id="file-1",
        document_id="file-1",
        page_number=page,
        chunk_index=0,
        content="text",
        start_offset=0,
        end_offset=4,
        token_count=1,
        content_hash="hash",
    )


def _relation(reason: str, source_type: str = "EQUIPMENT") -> dict[str, object]:
    return {
        "kind": "relation",
        "reason": reason,
        "name": "USES_PROCESS",
        "source": "Granulator G-10",
        "source_type": source_type,
        "target": "wet granulation",
        "target_type": "PROCESS_STEP",
        "quote": "The Granulator G-10 performs wet granulation.",
        "chunk_id": "c2",
    }


def test_every_rejected_record_becomes_one_row_placed_on_its_page() -> None:
    """Rows name the file and page; a record a later pass rejects again is one row."""

    trace = [
        {
            "stage": "relation_extraction",
            "window_id": "w1",
            "rejected_records": [_relation("domain_or_range_violation")],
        },
        # A continuation over the same window turns the same record away again.
        {
            "stage": "relation_extraction",
            "window_id": "w1",
            "rejected_records": [_relation("domain_or_range_violation")],
        },
        {
            "stage": "entity_extraction",
            "window_id": "w1",
            "rejected_records": [
                {
                    "kind": "entity",
                    "reason": "ungrounded_quote",
                    "name": "Acme",
                    "type": "ORGANIZATION",
                    "quote": "Acmes",
                    "chunk_id": "c2",
                }
            ],
        },
        # What the verifier turns away is listed like what extraction does.
        {
            "stage": "relation_verification",
            "window_id": "w1",
            "rejected_records": [_relation("verification_adjacency")],
        },
        # A stage that rejects nothing is not read.
        {
            "stage": "relation_grounding",
            "window_id": "w1",
            "rejected_records": [_relation("unsupported")],
        },
    ]

    rows = rejected_records_from_trace(trace, [_chunk("c2", 7)], "graph")

    assert [(row.kind, row.reason, row.page_number, row.file_id) for row in rows] == [
        ("entity", "ungrounded_quote", 7, "file-1"),
        ("relation", "domain_or_range_violation", 7, "file-1"),
        ("relation", "verification_adjacency", 7, "file-1"),
    ]
    assert len({row.id for row in rows}) == 3


def test_the_summary_names_the_type_rules_that_cost_most_relations() -> None:
    trace = [
        {
            "stage": "relation_extraction",
            "window_id": f"w{index}",
            "rejected_records": [_relation("domain_or_range_violation", source_type)],
        }
        for index, source_type in enumerate(["EQUIPMENT", "EQUIPMENT", "FACILITY"])
    ]

    metrics = rejected_record_metrics(rejected_records_from_trace(trace, [], "graph"))

    assert metrics["records"] == 3
    assert metrics["by_reason"] == {"relation:domain_or_range_violation": 3}
    assert metrics["broken_type_rules"] == [
        {
            "source_type": "EQUIPMENT",
            "relation_type": "USES_PROCESS",
            "target_type": "PROCESS_STEP",
            "count": 2,
        },
        {
            "source_type": "FACILITY",
            "relation_type": "USES_PROCESS",
            "target_type": "PROCESS_STEP",
            "count": 1,
        },
    ]
