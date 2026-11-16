"""A relation rejected only for its entity types is held, and finalization decides.

One window reads each entity under one type. These contracts pin what a window
carries out when it holds a statement, that a held statement never counts as
kept, and the rule table and record keys both finalization engines share.
"""

from __future__ import annotations

import json
from typing import Any

from documents import chunk, window

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.llm_extractors import LlmRelationExtractor
from kg_processor.application.rejected_records import (
    rejected_record_key,
    rejected_records_from_trace,
)
from kg_processor.application.two_pass_extraction import (
    apply_entity_verdicts,
    extract_relation_observations,
)
from kg_processor.application.type_resolution import type_resolution_rules
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.extraction import (
    EntityMention,
    EntityVerificationDecision,
    ExtractionObservations,
    ExtractionWindow,
    RelationExtractionOutcome,
    RelationObservation,
    VerificationDecision,
    VerificationOutcome,
)
from kg_processor.domain.ontology import OntologyProfile
from kg_processor.ports.llm import StructuredCompletionRequest, StructuredCompletionResult

_TEXT = "Acme is located in Jordan. Acme is located in Paris."


def _profile(**extra: Any) -> OntologyProfile:
    return OntologyProfile.model_validate(
        {
            "name": "places",
            "description": "Organizations and the places they are in",
            "entity_types": [
                {"name": "ORGANIZATION", "description": "An organization"},
                {"name": "PLACE", "description": "A place"},
                {"name": "PERSON", "description": "A person"},
            ],
            "relation_types": [
                {
                    "name": "LOCATED_IN",
                    "description": "An organization is located in a place",
                    "source_types": ["ORGANIZATION"],
                    "target_types": ["PLACE"],
                },
                {"name": "RELATED_TO", "description": "Any two things are related"},
            ],
            **extra,
        }
    )


def _mention(mention_id: str, name: str, entity_type: str) -> EntityMention:
    return EntityMention(
        id=mention_id,
        name=name,
        type=entity_type,
        description="",
        source_chunk_id="reference-chunk",
        quote=name,
    )


class _LocatedLlm(FakeLlmProvider):
    """States where Acme is: once in Paris, once in Jordan, once in an unquoted town."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        statements = [
            ("paris", "Paris", "Acme is located in Paris."),
            ("jordan", "Jordan", "Acme is located in Jordan."),
            ("bob", "Bob", "Acme is located in Bob's town."),
        ]
        return StructuredCompletionResult(
            payload={
                "relations": [
                    {
                        "source_entity_id": "acme",
                        "target_entity_id": target,
                        "source_surface": "Acme",
                        "target_surface": surface,
                        "relation_type": "LOCATED_IN",
                        "description": quote,
                        "source_chunk_id": "reference-chunk",
                        "quote": quote,
                        "confidence": 1.0,
                    }
                    for target, surface, quote in statements
                ]
            }
        )


def _located(hold: bool) -> tuple[ExtractionWindow, RelationExtractionOutcome]:
    extraction_window = window(chunk(_TEXT, chunk_id="reference-chunk"))
    mentions = [
        _mention("acme", "Acme", "ORGANIZATION"),
        _mention("paris", "Paris", "PLACE"),
        # This window read Jordan, and Bob, as people.
        _mention("jordan", "Jordan", "PERSON"),
        _mention("bob", "Bob", "PERSON"),
    ]
    outcome = LlmRelationExtractor(_LocatedLlm(), hold_type_violations=hold).extract(
        extraction_window,
        mentions,
        _profile(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )
    return extraction_window, outcome


def test_a_window_holds_a_type_rejected_relation_only_when_holding_is_on() -> None:
    """The held relation is grounded, never kept, and still a rejected record.

    Only a statement grounding can place is held; the one whose quote is not in
    the text stays what it was. Rejection counts are the same either way, so a
    run's losses read the same with the setting on and off.
    """

    _, off = _located(hold=False)
    extraction_window, on = _located(hold=True)

    assert [relation.target_entity_id for relation in off.relations] == ["paris"]
    assert off.held_relations == []
    assert [relation.target_entity_id for relation in on.relations] == ["paris"]
    [held] = on.held_relations
    assert (held.target_entity_id, held.relation_type, held.quote) == (
        "jordan",
        "LOCATED_IN",
        "Acme is located in Jordan.",
    )
    for outcome in (off, on):
        assert outcome.trace["accepted_records"] == 1
        assert outcome.trace["record_actions"]["domain_or_range_violation"] == 2
        assert [
            (record["reason"], record["target"]) for record in outcome.trace["rejected_records"]
        ] == [("domain_or_range_violation", "Jordan"), ("domain_or_range_violation", "Bob")]
    assert "ungrounded_quote" not in on.trace["record_actions"]
    assert on.trace["record_actions"]["held_for_type_resolution"] == 1
    # The key names the record the held relation left, so finalization can take it back.
    record = on.trace["rejected_records"][0]
    assert held.rejected_record_key == rejected_record_key(
        extraction_window.id, "relation", record["reason"], record
    )


def test_a_statement_the_relabel_declines_is_held_like_any_other() -> None:
    class _DecliningLlm(_LocatedLlm):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            if request.task_name == "relation_relabel":
                statements = json.loads(request.user[request.user.index("{") :])["statements"]
                return StructuredCompletionResult(
                    payload={"decisions": [{"id": item["id"], "option": -1} for item in statements]}
                )
            return super().complete_structured(request)

    outcome = LlmRelationExtractor(
        _DecliningLlm(), relabel=True, hold_type_violations=True
    ).extract(
        window(chunk(_TEXT, chunk_id="reference-chunk")),
        [
            _mention("acme", "Acme", "ORGANIZATION"),
            _mention("paris", "Paris", "PLACE"),
            _mention("jordan", "Jordan", "PERSON"),
            _mention("bob", "Bob", "PERSON"),
        ],
        _profile(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert [relation.target_entity_id for relation in outcome.held_relations] == ["jordan"]
    assert outcome.trace["record_actions"]["domain_or_range_violation"] == 2
    assert sorted(record["target"] for record in outcome.trace["rejected_records"]) == [
        "Bob",
        "Jordan",
    ]


def test_taking_back_a_held_relations_record_leaves_what_stayed_rejected() -> None:
    extraction_window, outcome = _located(hold=True)
    trace = [outcome.trace]
    chunks = extraction_window.chunks
    [held] = outcome.held_relations

    everything = rejected_records_from_trace(trace, chunks, "graph")
    remaining = rejected_records_from_trace(
        trace, chunks, "graph", taken_back={held.rejected_record_key or ""}
    )

    assert [row.target for row in everything] == ["Bob", "Jordan"]
    assert [row.target for row in remaining] == ["Bob"]
    assert remaining[0] == next(row for row in everything if row.target == "Bob")


class _HeldAndKeptExtractor:
    """One kept relation and two held ones, from every call."""

    def extract(self, window, entities, ontology, **kwargs):  # type: ignore[no-untyped-def]
        def observation(relation_id: str) -> RelationObservation:
            return RelationObservation(
                id=relation_id,
                source_entity_id="acme",
                target_entity_id=relation_id,
                relation_type="LOCATED_IN",
                description="",
                source_chunk_id="reference-chunk",
                quote=_TEXT,
                start_offset=0,
                end_offset=len(_TEXT),
            )

        return RelationExtractionOutcome(
            relations=[observation("kept")],
            held_relations=[observation("held-supported"), observation("held-refused")],
            trace={"stage": "relation_extraction", "input_records": 3},
        )


class _RecordingVerifier:
    """Supports every relation but the one named to refuse; records each batch."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.typed: list[bool] = []

    def verify(self, window, entities, relations, ontology, **kwargs):  # type: ignore[no-untyped-def]
        self.batches.append([relation.id for relation in relations])
        self.typed.append(
            any(item.source_types or item.target_types for item in ontology.relation_types)
        )
        return VerificationOutcome(
            decisions=[
                VerificationDecision(
                    relation_id=relation.id,
                    verdict="insufficient" if relation.id == "held-refused" else "supported",
                    confidence=0.9,
                )
                for relation in relations
            ],
            entity_decisions=[
                EntityVerificationDecision(
                    entity_id=entity.id, verdict="specific", type=entity.type
                )
                for entity in kwargs.get("entities_to_verify") or []
            ],
            trace={"stage": "relation_verification", "input_records": len(relations)},
        )


def test_held_relations_are_verified_apart_and_never_kept() -> None:
    """A held statement must pass the verifier, in batches of its own, to stay held."""

    verifier = _RecordingVerifier()
    observations = extract_relation_observations(
        [chunk(_TEXT, chunk_id="reference-chunk")],
        [_mention("acme", "Acme", "ORGANIZATION")],
        FakeLlmProvider(),
        GraphSettings(gleaning_max_passes=0),
        _profile(),
        "fake",
        30,
        relation_extractor=_HeldAndKeptExtractor(),
        relation_verifier=verifier,
    )

    assert verifier.batches == [["kept"], ["held-supported", "held-refused"]]
    # A held statement broke its type rules on purpose; the verifier judges
    # only whether the text states it, so it is shown no type rules.
    assert verifier.typed == [True, False]
    assert [relation.id for relation in observations.relations] == ["kept"]
    assert [relation.id for relation in observations.held_relations] == ["held-supported"]
    assert [event["stage"] for event in observations.trace if "verification" in event["stage"]] == [
        "relation_verification",
        "held_relation_verification",
    ]


def test_the_rule_table_keeps_a_typing_under_its_own_rules_or_a_rewrite_only() -> None:
    """The fallback relation never admits a held statement; unrestricted relations have no rows."""

    profile = _profile(
        relation_types=[
            *_profile().model_dump()["relation_types"],
            {
                "name": "BASED_IN",
                "description": "A person is based in a place",
                "source_types": ["PERSON"],
                "target_types": ["PLACE"],
            },
        ],
        relation_rewrites=[
            {"relation": "LOCATED_IN", "source_types": ["PERSON"], "to": "BASED_IN"}
        ],
        fallback_relation="RELATED_TO",
    )

    rules = type_resolution_rules(profile)

    assert rules[("LOCATED_IN", "ORGANIZATION", "PLACE")] == ("LOCATED_IN", None)
    assert rules[("LOCATED_IN", "PERSON", "PLACE")] == ("BASED_IN", "LOCATED_IN")
    assert ("LOCATED_IN", "ORGANIZATION", "PERSON") not in rules
    assert not any(stated == "RELATED_TO" for stated, _, _ in rules)


def test_a_held_statement_falls_away_with_an_endpoint_verification_rejected() -> None:
    """Its rejected record already stands; the held statement itself is dropped."""

    held = RelationObservation(
        id="held",
        source_entity_id="acme",
        target_entity_id="heading",
        relation_type="LOCATED_IN",
        description="",
        source_chunk_id="reference-chunk",
        quote="Acme",
        start_offset=0,
        end_offset=4,
    )
    observations = ExtractionObservations(
        entities=[
            _mention("acme", "Acme", "ORGANIZATION"),
            _mention("heading", "Offices", "PLACE"),
        ],
        relations=[],
        held_relations=[held],
        trace=[
            {
                "stage": "relation_verification",
                "entity_verdicts": [{"mention_id": "heading", "verdict": "heading"}],
            }
        ],
        chunk_count=1,
        window_count=1,
    )

    combined = apply_entity_verdicts(observations, _profile())

    assert [entity.id for entity in combined.entities] == ["acme"]
    assert combined.held_relations == []
