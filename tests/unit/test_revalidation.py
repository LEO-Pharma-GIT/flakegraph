"""Behavior tests for putting stored records through the current validation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
from documents import chunk
from pydantic import BaseModel

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.llm_extractors import LlmEntityExtractor, LlmRelationExtractor
from kg_processor.application.rejected_records import (
    observation_record,
    rejected_record_key,
    rejected_records_from_trace,
)
from kg_processor.application.revalidation import (
    REVALIDATION_STAGE,
    RevalidationRequest,
    _entity_loss,
    _relation_loss,
    begin_revalidation,
    conclude_revalidation,
    relation_payload,
    resolve_endpoint,
    revalidate_entity_window,
    revalidate_observations,
    revalidate_relation_window,
    revalidated_inventory,
    stored_records,
)
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.extraction import (
    ColumnRelation,
    EntityMention,
    EntityVerificationDecision,
    ExtractionObservations,
    ExtractionWindow,
    RelationExtractionOutcome,
    RelationObservation,
    VerificationDecision,
    VerificationOutcome,
)
from kg_processor.domain.ontology import (
    EntityTypeDefinition,
    IdentifierPattern,
    OntologyProfile,
    RelationRewrite,
    RelationTypeDefinition,
)
from kg_processor.domain.stages import RevalidationWindowShard, StoredEntity, StoredRelation
from kg_processor.ports.extraction import EntityExtractor, RelationExtractor
from kg_processor.ports.llm import (
    LlmCapabilities,
    StructuredCompletionProvider,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)

CONTENT = (
    "Alice Smith works at Acme Labs. Acme Labs teaches the Kata method to Bob. "
    "Batch B-12 was recorded."
)
TEACHING = "Acme Labs teaches the Kata method to Bob."
_CONFIDENCE = 0.5


def test_a_record_a_revalidation_lost_comes_back_whole_from_its_trace() -> None:
    """What one revalidation could not replay is a candidate of the next, quote and all."""

    source = chunk(CONTENT, chunk_id="c1")
    entity = StoredEntity(
        name="Kata method",
        type="Method",
        chunk_id="c1",
        quote="A long quote " * 40,
        description="A training method.",
        confidence=0.7,
        aliases=["Kata"],
    )
    relation = StoredRelation(
        **{**_stored("Bob", "Person", "TEACHES", "Kata method", "Method"), "quote": "x" * 500},
        description="Bob teaches it.",
        quantities=[{"text": "two", "kind": "", "unit": "", "comparator": ""}],
        assertion="claimed",
    )

    records = stored_records(
        ExtractionObservations(
            trace=[
                {
                    "stage": "revalidation",
                    "rejected_records": [
                        _entity_loss(entity, "window_skipped"),
                        _relation_loss(relation, "window_skipped"),
                    ],
                }
            ],
            chunk_count=1,
            window_count=1,
        ),
        [source],
    )

    assert records.entities[0] == entity
    assert records.relations == [relation]


def test_stored_records_rebuild_every_pass_candidates_accepted_first() -> None:
    """Accepted observations come first; rejected ones come back from every pass.

    Context records stay context candidates; a verifier's rejection of an
    entity or relation is a candidate again, under the type the model stated;
    a schema failure, a rejected new endpoint and a cue relation are not
    candidates; a record comes back with what was proposed; and a typed
    endpoint no stored entity names is proposed where its relation is.
    """

    source = chunk(CONTENT, chunk_id="c1")
    alice = _mention("m-alice", "Alice Smith", "Person")
    acme = _mention("m-acme", "Acme Labs", "Organization")
    context = _mention("m-doc", "The report", "Person").model_copy(
        update={"is_document_context": True}
    )
    accepted = _relation("r1", alice, acme, "WORKS_AT", "Alice Smith works at Acme Labs.")
    accepted = accepted.model_copy(update={"proposed_relation_type": "EMPLOYED_BY"})
    cue = accepted.model_copy(update={"id": "r2", "evidence_method": "ontology_cue"})
    verified_away = _relation("r3", acme, alice, "EMPLOYS", "Alice Smith works at Acme Labs.")
    verified_away = verified_away.model_copy(update={"proposed_relation_type": "HIRED"})
    trace: list[dict[str, Any]] = [
        {
            "stage": "document_context_extraction",
            "rejected_records": [
                _entity_record("Annual review", "Person", "x", "document_context_name_mismatch")
            ],
        },
        {
            "stage": "entity_extraction",
            "rejected_records": [
                _entity_record("Kata method", "Method", "the Kata method", "invalid_entity_type"),
                _entity_record("Broken", "", "", "invalid_schema"),
                {
                    **_entity_record("Bobby", "Person", "to Bob", "ungrounded_quote"),
                    "candidate": {"description": "A student.", "confidence": 0.7, "aliases": ["B"]},
                },
            ],
        },
        {
            "stage": "relation_extraction",
            "rejected_records": [
                _entity_record("Globex", "Organization", "Globex", "ungrounded_quote"),
                _relation_record("Bob", "Person", "TEACHES", "Kata method", "Method", TEACHING),
            ],
        },
        {
            "stage": "relation_verification",
            "rejected_records": [
                observation_record(verified_away, {"m-alice": alice, "m-acme": acme}, "x")
            ],
        },
    ]

    records = stored_records(
        ExtractionObservations(
            entities=[alice, acme, context],
            relations=[accepted, cue],
            trace=trace,
            chunk_count=1,
            window_count=1,
        ),
        [source],
    )

    assert [(item.name, item.accepted) for item in records.context] == [
        ("The report", True),
        ("Annual review", False),
    ]
    assert [(item.name, item.accepted) for item in records.entities] == [
        ("Alice Smith", True),
        ("Acme Labs", True),
        ("Kata method", False),
        ("Bobby", False),
        ("Bob", False),
    ]
    kata, bobby, bob = records.entities[2:]
    assert (kata.confidence, kata.description, kata.aliases) == (0.5, "", [])
    assert (bobby.confidence, bobby.description, bobby.aliases) == (0.7, "A student.", ["B"])
    # The proposed endpoint is quoted by its relation's evidence.
    assert (bob.type, bob.chunk_id, bob.quote) == ("Person", "c1", TEACHING)
    assert records.endpoints_proposed == 1
    first, teaches, employs = records.relations
    assert (first.relation_type, first.source_id, first.target_id, first.accepted) == (
        "EMPLOYED_BY",
        alice.id,
        acme.id,
        True,
    )
    assert (teaches.source, teaches.target, teaches.accepted) == ("Bob", "Kata method", False)
    assert (employs.relation_type, employs.confidence, employs.source_surface) == (
        "HIRED",
        0.9,
        "Acme Labs",
    )


def test_a_claimed_relation_is_replayed_as_a_claim() -> None:
    """Accepted or rejected, a stored claim comes back to extraction still claimed."""

    source = chunk(CONTENT, chunk_id="c1")
    alice = _mention("m-alice", "Alice Smith", "Person")
    acme = _mention("m-acme", "Acme Labs", "Organization")
    accepted = _relation("r1", alice, acme, "WORKS_AT", "Alice Smith works at Acme Labs.")
    accepted = accepted.model_copy(update={"assertion": "claimed"})
    rejected = accepted.model_copy(update={"id": "r2", "relation_type": "EMPLOYS"})
    trace: list[dict[str, Any]] = [
        {
            "stage": "relation_verification",
            "rejected_records": [
                observation_record(rejected, {"m-alice": alice, "m-acme": acme}, "x")
            ],
        }
    ]

    records = stored_records(
        ExtractionObservations(
            entities=[alice, acme],
            relations=[accepted],
            trace=trace,
            chunk_count=1,
            window_count=1,
        ),
        [source],
    )
    payload, _ = relation_payload(records.relations, [alice, acme], {})

    assert [item.assertion for item in records.relations] == ["claimed", "claimed"]
    assert [item["assertion"] for item in payload["relations"]] == ["claimed", "claimed"]


def test_endpoints_resolve_by_name_alias_or_context_surface_within_type() -> None:
    """An endpoint name finds the identity it names; an ambiguous untyped one finds none."""

    alice = _mention("m1", "Alice Smith", "Person", aliases=["A. Smith"], chunk_id="c2")
    alice_here = _mention("m2", "Alice Smith", "Person", chunk_id="c1")
    report = _mention("m3", "Annual report", "Document").model_copy(
        update={"is_document_context": True, "contextual_surfaces": ["this report"]}
    )
    mercury_planet = _mention("m4", "Mercury", "Planet")
    mercury_metal = _mention("m5", "Mercury", "Element")
    entities = [alice, alice_here, report, mercury_planet, mercury_metal]

    assert resolve_endpoint("alice  smith", "Person", "c1", entities) == alice_here
    assert resolve_endpoint("A. Smith", "", "c9", entities) == alice
    assert resolve_endpoint("This Report", "Document", "c1", entities) == report
    # A type no identity of the name has: the name alone decides, and the
    # current rules then judge the relation under the entity's own type.
    assert resolve_endpoint("Alice Smith", "Organization", "c1", entities) == alice_here
    assert resolve_endpoint("Mercury", "Star", "c1", entities) is None
    assert resolve_endpoint("Mercury", "", "c1", entities) is None
    assert resolve_endpoint("Mercury", "Element", "c1", entities) == mercury_metal


def test_an_endpoint_finds_the_mention_verification_retyped() -> None:
    """A relation stated about a thing is still about it after its type is corrected."""

    retyped = _mention("m1", "Sodium chloride", "Excipient").model_copy(
        update={"proposed_type": "Product"}
    )

    assert resolve_endpoint("SODIUM CHLORIDE", "Product", "c1", [retyped]) == retyped
    assert resolve_endpoint("SODIUM CHLORIDE", "Excipient", "c1", [retyped]) == retyped
    assert resolve_endpoint("SODIUM CHLORIDE", "Active ingredient", "c1", [retyped]) == retyped


def test_a_relation_response_is_rebuilt_against_the_window_entities() -> None:
    """Resolved endpoints take current ids; one that resolves to nothing has none.

    An accepted relation keeps its stored endpoint while that mention still
    stands, and finds it again by name when revalidation gave it a new id.
    """

    acme = _mention("new-acme", "Acme Labs", "Organization")
    kept = _mention("m-alice", "Alice Smith", "Person")
    old = {"old-acme": _mention("old-acme", "Acme Labs", "Organization"), kept.id: kept}
    relations = [
        StoredRelation(
            **_stored("Alice Smith", "Person", "WORKS_AT", "Acme Labs", "Organization"),
            accepted=True,
            source_id=kept.id,
            target_id="old-acme",
        ),
        StoredRelation(**_stored("Alice Smith", "Person", "WORKS_AT", "Globex", "")),
    ]

    payload, stats = relation_payload(relations, [acme, kept], old)

    first, second = payload["relations"]
    assert (first["source_entity_id"], first["target_entity_id"]) == (kept.id, acme.id)
    assert (second["source_entity_id"], second["target_entity_id"]) == (kept.id, "")
    assert second["target_surface"] == "Globex"
    assert (stats.resolved, stats.unresolved) == (3, 1)
    assert set(payload) == {"relations"}


def test_revalidation_applies_the_current_rules_and_verification_without_extraction_calls() -> None:
    """Stored records meet the current ontology and verifier; what changed is recorded.

    The base extraction accepted Alice, Acme and the code B-12, and turned
    away the Kata method (its type was unknown then), a TEACHES statement the
    ontology did not allow for an organisation, and statements about Bob, who
    was not an entity. The current ontology knows the Method type, rewrites an
    organisation's TEACHES to USES, and refuses batch codes as entities. Bob is
    proposed as the endpoint he is. The verifier, shown the document's opening,
    no longer supports what Bob teaches.
    """

    source = chunk(CONTENT, chunk_id="c1")
    alice = _mention("m-alice", "Alice Smith", "Person")
    acme = _mention("m-acme", "Acme Labs", "Organization")
    batch = _mention("m-batch", "B-12", "Method")
    works_at = _relation("r-works", alice, acme, "WORKS_AT", "Alice Smith works at Acme Labs.")
    stored = ExtractionObservations(
        entities=[alice, acme, batch],
        relations=[works_at],
        trace=[
            {"stage": "parse", "file_id": "file_1"},
            {
                "stage": "entity_extraction",
                "window_id": "old-window",
                "rejected_records": [
                    _entity_record(
                        "Kata method", "Method", "the Kata method", "invalid_entity_type"
                    )
                ],
            },
            {
                "stage": "relation_extraction",
                "window_id": "old-window",
                "rejected_records": [
                    _relation_record(
                        "Acme Labs", "Organization", "TEACHES", "Kata method", "Method", TEACHING
                    ),
                    _relation_record("Bob", "", "TEACHES", "Kata method", "Method", TEACHING),
                    _relation_record("Bob", "Person", "TEACHES", "Kata method", "Method", TEACHING),
                ],
            },
            {
                "stage": "relation_verification",
                "window_id": "old-window",
                "entity_verdicts": [],
            },
        ],
        chunk_count=1,
        window_count=1,
    )
    llm = _ValidationOnlyLlm()
    verifier = _ScriptedVerifier(supported={"WORKS_AT", "USES"})

    revised = revalidate_observations([source], stored, _request(llm, verifier, _ontology()))

    assert llm.task_names == []
    assert verifier.openings and all(CONTENT[:40] in opening for opening in verifier.openings)
    assert {(entity.name, entity.type) for entity in revised.entities} == {
        ("Alice Smith", "Person"),
        ("Acme Labs", "Organization"),
        ("Kata method", "Method"),
        ("Bob", "Person"),
    }
    # The verifier judged this window's own entities too.
    assert {item.name for item in verifier.judged_entities} >= {"Alice Smith", "Kata method"}
    assert _triples(revised) == {
        ("Alice Smith", "WORKS_AT", "Acme Labs"),
        ("Acme Labs", "USES", "Kata method"),
    }
    uses = next(item for item in revised.relations if item.relation_type == "USES")
    assert (uses.proposed_relation_type, uses.evidence_method) == ("TEACHES", "llm")

    stages = [event["stage"] for event in revised.trace]
    # Superseded passes are replaced; everything else is kept as it was.
    assert stages[0] == "parse" and "old-window" not in str(revised.trace)
    (summary,) = [event for event in revised.trace if event["stage"] == REVALIDATION_STAGE]
    assert summary["entities"] == {"before": 3, "after": 4, "kept": 2, "admitted": 2, "dropped": 1}
    assert summary["relations"] == {"before": 1, "after": 2, "kept": 1, "admitted": 1, "dropped": 0}
    admitted = {
        (item["kind"], item["name"], item["evidence_method"])
        for item in summary["admitted_records"]
    }
    assert admitted == {
        ("entity", "Kata method", "llm"),
        ("entity", "Bob", "llm"),
        ("relation", "USES", "llm"),
    }
    assert summary["reconstruction"]["endpoints_proposed"] == 1
    assert summary["reconstruction"]["endpoints_unresolved"] == 0
    # Each loss is one row of the rejected-records table, under the rule that took it.
    rows = rejected_records_from_trace(revised.trace, [source], "graph")
    rejected = sorted((row.name, row.reason) for row in rows)
    assert ("B-12", "identifier_as_entity") in rejected
    assert [reason for name, reason in rejected if name == "TEACHES"] == [
        "verification_contradicted"
    ]


def test_entity_verdicts_apply_to_the_whole_revalidated_document() -> None:
    """A mention the verifier rejects falls away, with the relations that used it."""

    source = chunk(CONTENT, chunk_id="c1")
    alice = _mention("m-alice", "Alice Smith", "Person")
    acme = _mention("m-acme", "Acme Labs", "Organization")
    stored = ExtractionObservations(
        entities=[alice, acme],
        relations=[
            _relation("r-works", alice, acme, "WORKS_AT", "Alice Smith works at Acme Labs.")
        ],
        chunk_count=1,
        window_count=1,
    )
    verifier = _ScriptedVerifier(supported={"WORKS_AT"}, rejected={"Acme Labs": "placeholder"})

    revised = revalidate_observations(
        [source], stored, _request(_ValidationOnlyLlm(), verifier, _ontology())
    )

    assert [entity.name for entity in revised.entities] == ["Alice Smith"]
    assert revised.relations == []
    reasons = {row.reason for row in rejected_records_from_trace(revised.trace, [source], "g")}
    assert reasons == {"verification_placeholder", "verification_endpoint_rejected"}


def test_a_statement_rejected_for_its_types_comes_back_held() -> None:
    """With type holding on, a replayed type violation is held for finalization.

    It carries the key of its rejected record, so finalization can take the
    record back when it admits the statement, and it was judged apart from its
    types by the verifier.
    """

    source = chunk(CONTENT, chunk_id="c1")
    acme = _mention("m-acme", "Acme Labs", "Organization")
    kata = _mention("m-kata", "Kata method", "Method")
    stored = ExtractionObservations(
        entities=[acme, kata],
        trace=[
            {
                "stage": "relation_extraction",
                "rejected_records": [
                    _relation_record(
                        "Acme Labs", "Organization", "TEACHES", "Kata method", "Method", TEACHING
                    )
                ],
            }
        ],
        chunk_count=1,
        window_count=1,
    )
    ontology = _ontology().model_copy(update={"relation_rewrites": []})
    verifier = _ScriptedVerifier(supported={"TEACHES"})

    revised = revalidate_observations(
        [source], stored, _request(_ValidationOnlyLlm(), verifier, ontology, hold=True)
    )

    assert revised.relations == []
    (held,) = revised.held_relations
    assert held.relation_type == "TEACHES"
    (event,) = [e for e in revised.trace if e["stage"] == "relation_extraction"]
    (record,) = event["rejected_records"]
    assert record["reason"] == "domain_or_range_violation"
    assert held.rejected_record_key == rejected_record_key(
        event["window_id"], "relation", "domain_or_range_violation", record
    )
    assert any(e["stage"] == "held_relation_verification" for e in revised.trace)
    (summary,) = [e for e in revised.trace if e["stage"] == REVALIDATION_STAGE]
    assert summary["held_relations"]["after"] == 1


def test_document_context_mentions_are_revalidated_by_the_context_rules() -> None:
    """A context record the base's rules turned away is a context mention again.

    Front matter is the context window; the replay meets the context pass's
    own validation, and the admitted mention carries its context role.
    """

    source = chunk("Annual report of Acme Labs. Alice Smith works at Acme Labs.", chunk_id="c1")
    report = _mention("m-report", "Annual report", "Report").model_copy(
        update={"is_document_context": True, "quote": "Annual report"}
    )
    stored = ExtractionObservations(
        entities=[report],
        trace=[
            {
                "stage": "document_context_extraction",
                "window_id": "old-context",
                "rejected_records": [
                    _entity_record(
                        "Acme Labs", "Organization", "Acme Labs", "document_context_name_mismatch"
                    )
                ],
            }
        ],
        chunk_count=1,
        window_count=1,
    )
    ontology = _ontology().model_copy(
        update={
            "entity_types": [
                *_ontology().entity_types,
                EntityTypeDefinition(name="Report", description="A report.", document_context=True),
            ]
        }
    )

    revised = revalidate_observations(
        [source], stored, _request(_ValidationOnlyLlm(), _ScriptedVerifier(set()), ontology)
    )

    context = {entity.name for entity in revised.entities if entity.is_document_context}
    assert context == {"Annual report", "Acme Labs"}
    (event,) = [e for e in revised.trace if e["stage"] == "document_context_extraction"]
    assert event["replayed"] is True and "old-context" not in str(revised.trace)


def test_a_folder_mention_is_kept_as_stored_and_never_replayed() -> None:
    """The folder names the mention, not the model: no rule of the passes reads it again.

    Replayed through the entity rules, a name read off the location page would
    be turned away as where the document sits.
    """

    location = chunk(
        "Location: Northwind Pricing Study/report.pdf\n"
        "Folder: Northwind Pricing Study\n"
        "File: report.pdf",
        chunk_id="c0",
        page_number=0,
    )
    body = chunk("Alice Smith works at Acme Labs.", chunk_id="c1", chunk_index=1)
    start = location.content.index("Northwind Pricing Study\nFile")
    folder = _mention("m-folder", "Northwind Pricing Study", "Organization").model_copy(
        update={
            "source_chunk_id": "c0",
            "quote": "Northwind Pricing Study",
            "start_offset": start,
            "end_offset": start + len("Northwind Pricing Study"),
            "is_document_context": True,
            "evidence_method": "folder",
        }
    )
    stored = ExtractionObservations(entities=[folder], chunk_count=2, window_count=1)

    assert stored_records(stored, [location, body]).context == []
    llm = _ValidationOnlyLlm()
    revised = revalidate_observations(
        [location, body], stored, _request(llm, _ScriptedVerifier(set()), _ontology())
    )

    assert [entity for entity in revised.entities if entity.id == "m-folder"] == [folder]
    assert "document_context_extraction" not in llm.task_names


def test_a_catalogue_window_is_replayed_with_its_column_relations() -> None:
    """Relation windows of a catalogue read its rows by the columns its profile kept."""

    source = chunk(CONTENT, chunk_id="c1")
    alice = _mention("m-alice", "Alice Smith", "Person")
    acme = _mention("m-acme", "Acme Labs", "Organization")
    columns = [
        {"column": "Name", "role": "subject", "entity_type": "Person"},
        {
            "column": "Employer",
            "role": "relation",
            "entity_type": "Organization",
            "relation_type": "WORKS_AT",
            "source": "subject",
        },
    ]
    stored = ExtractionObservations(
        entities=[alice, acme],
        relations=[
            _relation("r-works", alice, acme, "WORKS_AT", "Alice Smith works at Acme Labs.")
        ],
        trace=[
            {
                "stage": "dataset_profile",
                "document_id": "file_1",
                "kind": "catalogue",
                "columns": columns,
            }
        ],
        chunk_count=1,
        window_count=1,
    )
    seen: list[list[ColumnRelation]] = []

    class RecordingRelations(LlmRelationExtractor):
        def extract(
            self, window: ExtractionWindow, *args: Any, **kwargs: Any
        ) -> RelationExtractionOutcome:
            seen.append(window.dataset_columns)
            return super().extract(window, *args, **kwargs)

    def extractors(
        provider: StructuredCompletionProvider,
    ) -> tuple[EntityExtractor, RelationExtractor]:
        return LlmEntityExtractor(provider), RecordingRelations(provider)

    request = _request(_ValidationOnlyLlm(), _ScriptedVerifier({"WORKS_AT"}), _ontology())

    revalidate_observations([source], stored, replace(request, extractors=extractors))

    assert seen == [[ColumnRelation.model_validate(item) for item in columns]]


def test_windows_revalidated_apart_and_combined_match_one_pass_over_the_document() -> None:
    """A fleet's window tasks rebuild the document one pass does.

    The document has four windows. Relations name entities stored in other
    windows, an endpoint is proposed in one window and used in another, a
    statement comes back under a rewrite, and the verifier's verdict on one
    mention applies to the whole document. Every intermediate result crosses
    a JSON boundary as a task's artifact does, and windows finish in reverse
    order, yet the observations and every rejected record are the ones the
    single pass produces.
    """

    texts = {
        "c1": "Alice Smith works at Acme Labs.",
        "c2": TEACHING,
        "c3": "Batch B-12 was recorded by Alice Smith.",
        "c4": "Bob joined Acme Labs as Alice Smith did.",
    }
    chunks = [
        chunk(text, chunk_id=chunk_id, file_id="file_1", document_id="file_1", chunk_index=index)
        for index, (chunk_id, text) in enumerate(texts.items())
    ]
    alice = _mention("m-alice", "Alice Smith", "Person")
    acme = _mention("m-acme", "Acme Labs", "Organization")
    kata = _mention("m-kata", "Kata method", "Method", chunk_id="c2")
    batch = _mention("m-batch", "B-12", "Method", chunk_id="c3")
    stored = ExtractionObservations(
        entities=[alice, acme, kata, batch],
        relations=[_relation("r-works", alice, acme, "WORKS_AT", texts["c1"])],
        trace=[
            {"stage": "parse", "file_id": "file_1"},
            {
                "stage": "relation_extraction",
                "window_id": "old-window",
                "rejected_records": [
                    {
                        **_relation_record(
                            "Acme Labs",
                            "Organization",
                            "TEACHES",
                            "Kata method",
                            "Method",
                            TEACHING,
                        ),
                        "chunk_id": "c2",
                    },
                    {
                        **_relation_record(
                            "Bob", "Person", "TEACHES", "Kata method", "Method", TEACHING
                        ),
                        "chunk_id": "c2",
                    },
                    {
                        **_relation_record(
                            "Bob", "Person", "WORKS_AT", "Acme Labs", "Organization", texts["c4"]
                        ),
                        "chunk_id": "c4",
                    },
                ],
            },
        ],
        chunk_count=4,
        window_count=2,
    )
    request = replace(
        _request(
            _ValidationOnlyLlm(),
            _ScriptedVerifier(
                supported={"WORKS_AT", "USES"}, rejected={"Kata method": "placeholder"}
            ),
            _ontology(),
        ),
        settings=GraphSettings(max_chunks_per_llm_call=1),
    )

    whole = revalidate_observations(chunks, stored, request)

    assert _triples(whole) == {
        ("Alice Smith", "WORKS_AT", "Acme Labs"),
        ("Bob", "WORKS_AT", "Acme Labs"),
    }
    assert "Kata method" not in {entity.name for entity in whole.entities}
    (summary,) = [event for event in whole.trace if event["stage"] == REVALIDATION_STAGE]
    assert summary["windows"] == 4
    begun, windows = begin_revalidation(["file_1"], chunks, stored, request)
    begun, windows = _round_trip(begun), [_round_trip(item) for item in windows]
    shards = [RevalidationWindowShard(file_ids=["file_1"], window=item) for item in windows]
    entity_windows = [
        _round_trip(revalidate_entity_window(item, request)) for item in reversed(shards)
    ][::-1]
    inventory = _round_trip(revalidated_inventory(begun, entity_windows))
    relation_windows = [
        _round_trip(revalidate_relation_window(item, inventory, request))
        for item in reversed(shards)
    ][::-1]

    apart = conclude_revalidation(stored, begun, inventory, relation_windows, _ontology())

    assert apart == whole
    assert rejected_records_from_trace(apart.trace, chunks, "g") == (
        rejected_records_from_trace(whole.trace, chunks, "g")
    )


def test_revalidation_refuses_extractors_it_cannot_replay() -> None:
    """Only the LLM extractors take stored records as their answer."""

    class OtherEntities:
        def extract(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("never called")

    request = _request(_ValidationOnlyLlm(), _ScriptedVerifier(set()), _ontology())
    other = replace(
        request,
        extractors=lambda provider: (
            cast(EntityExtractor, OtherEntities()),
            LlmRelationExtractor(provider),
        ),
    )

    with pytest.raises(ValueError, match="replays stored records through the LLM extractors"):
        revalidate_observations([chunk(CONTENT, chunk_id="c1")], _empty(), other)


def test_rejected_candidates_keep_what_the_model_proposed_for_later_revalidation() -> None:
    """A rejected record carries its description, confidence, aliases and full quote."""

    long_quote = "x " * 300
    window = ExtractionWindow(
        id="w", document_id="file_1", chunks=[chunk(CONTENT, chunk_id="c1")], token_count=20
    )
    llm = _AnsweringLlm(
        {
            "entities": [
                {
                    "name": "Nobody",
                    "type": "Person",
                    "description": "Someone absent.",
                    "source_chunk_id": "c1",
                    "quote": long_quote,
                    "confidence": 0.4,
                    "aliases": ["N."],
                }
            ]
        }
    )

    outcome = LlmEntityExtractor(llm).extract(
        window, _ontology(), model="fake", timeout_seconds=30, max_entities=5
    )

    (record,) = outcome.trace["rejected_records"]
    assert len(record["quote"]) == 400
    assert record["candidate"] == {
        "description": "Someone absent.",
        "confidence": 0.4,
        "aliases": ["N."],
        "quote": long_quote,
    }


def _request(
    llm: StructuredCompletionProvider,
    verifier: _ScriptedVerifier,
    ontology: OntologyProfile,
    *,
    hold: bool = False,
) -> RevalidationRequest:
    """A revalidation over the LLM extractors, as a configuration builds them."""

    def extractors(
        provider: StructuredCompletionProvider,
    ) -> tuple[EntityExtractor, RelationExtractor]:
        return LlmEntityExtractor(provider), LlmRelationExtractor(
            provider, hold_type_violations=hold
        )

    return RevalidationRequest(
        llm=llm,
        extractors=extractors,
        verifier=verifier,
        settings=GraphSettings(),
        ontology=ontology,
        model="fake",
        timeout_seconds=30,
    )


def _ontology() -> OntologyProfile:
    """The current ontology: a Method type, a rewrite, and a batch-code pattern."""

    return OntologyProfile(
        name="revalidation",
        description="People, organisations and methods.",
        mode="closed",
        entity_types=[
            EntityTypeDefinition(name="Person", description="A person."),
            EntityTypeDefinition(name="Organization", description="An organisation."),
            EntityTypeDefinition(name="Method", description="A method."),
        ],
        relation_types=[
            RelationTypeDefinition(
                name="WORKS_AT",
                description="A person works at an organisation.",
                source_types=["Person"],
                target_types=["Organization"],
            ),
            RelationTypeDefinition(
                name="TEACHES",
                description="A person teaches a method.",
                source_types=["Person"],
                target_types=["Method"],
            ),
            RelationTypeDefinition(
                name="USES",
                description="An organisation uses a method.",
                source_types=["Organization"],
                target_types=["Method"],
            ),
        ],
        relation_rewrites=[
            RelationRewrite(relation="TEACHES", source_types=["Organization"], to="USES")
        ],
        identifier_patterns=[IdentifierPattern(name="batch", pattern=r"B-\d+")],
    )


def _round_trip[M: BaseModel](model: M) -> M:
    """The model as the next task reads it back from the artifact a task stored."""

    return type(model).model_validate_json(model.model_dump_json())


def _empty() -> ExtractionObservations:
    """A document that stored nothing."""

    return ExtractionObservations(chunk_count=1, window_count=1)


def _mention(
    mention_id: str,
    name: str,
    type_: str,
    *,
    chunk_id: str = "c1",
    aliases: list[str] | None = None,
) -> EntityMention:
    """A stored mention grounded on its name in the shared content."""

    start = CONTENT.find(name)
    return EntityMention(
        id=mention_id,
        name=name,
        type=type_,
        description="",
        source_chunk_id=chunk_id,
        quote=name,
        confidence=0.9,
        aliases=aliases or [],
        start_offset=max(start, 0),
        end_offset=max(start, 0) + len(name),
    )


def _relation(
    relation_id: str,
    source: EntityMention,
    target: EntityMention,
    relation_type: str,
    quote: str,
) -> RelationObservation:
    """A stored relation between two mentions, on its evidence."""

    return RelationObservation(
        id=relation_id,
        source_entity_id=source.id,
        target_entity_id=target.id,
        source_surface=source.name,
        target_surface=target.name,
        relation_type=relation_type,
        description=f"{source.name} {relation_type} {target.name}.",
        source_chunk_id=source.source_chunk_id,
        quote=quote,
        confidence=0.9,
    )


def _triples(observations: ExtractionObservations) -> set[tuple[str, str, str]]:
    """The relations of a document as named triples."""

    by_id = {entity.id: entity.name for entity in observations.entities}
    return {
        (by_id[item.source_entity_id], item.relation_type, by_id[item.target_entity_id])
        for item in observations.relations
    }


def _entity_record(name: str, type_: str, quote: str, reason: str) -> dict[str, Any]:
    """A rejected entity as a trace lists it by name."""

    return {
        "kind": "entity",
        "reason": reason,
        "name": name,
        "type": type_,
        "quote": quote,
        "chunk_id": "c1",
        "candidate": {"description": "", "confidence": _CONFIDENCE, "aliases": []},
    }


def _relation_record(
    source: str,
    source_type: str,
    relation: str,
    target: str,
    target_type: str,
    quote: str,
    *,
    reason: str = "domain_or_range_violation",
) -> dict[str, Any]:
    """A rejected relation as a trace lists it by name: named endpoints, no ids."""

    return {
        "kind": "relation",
        "reason": reason,
        "name": relation,
        "source": source,
        "source_type": source_type,
        "target": target,
        "target_type": target_type,
        "quote": quote,
        "chunk_id": "c1",
        "candidate": {
            "description": "",
            "confidence": _CONFIDENCE,
            "source_surface": "",
            "target_surface": "",
            "quantities": [],
            "assertion": "reported",
        },
    }


def _stored(
    source: str, source_type: str, relation: str, target: str, target_type: str
) -> dict[str, Any]:
    """The fields of a stored relation, for building one in a test."""

    return {
        "relation_type": relation,
        "source": source,
        "source_type": source_type,
        "target": target,
        "target_type": target_type,
        "chunk_id": "c1",
        "quote": CONTENT,
        "confidence": _CONFIDENCE,
    }


class _ValidationOnlyLlm(FakeLlmProvider):
    """Record every request that reaches the model, and refuse extraction ones."""

    def __init__(self) -> None:
        self.task_names: list[str] = []

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        self.task_names.append(request.task_name)
        pytest.fail(f"revalidation asked the model for {request.task_name}")


class _AnsweringLlm:
    """Answer every structured request with one payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities()

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        del request
        return StructuredCompletionResult(payload=self.payload)


class _ScriptedVerifier:
    """Support relations of some types, contradict the rest, and judge entities.

    Every entity is specific unless ``rejected`` gives its verdict. What each
    call was shown is kept.
    """

    def __init__(self, supported: set[str], rejected: dict[str, str] | None = None) -> None:
        self.supported = supported
        self.rejected = rejected or {}
        self.openings: list[str] = []
        self.judged_entities: list[EntityMention] = []

    def verify(
        self,
        window: ExtractionWindow,
        entities: list[EntityMention],
        relations: list[RelationObservation],
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        document_opening: str = "",
        document_subjects: list[EntityMention] | None = None,
        entities_to_verify: list[EntityMention] | None = None,
    ) -> VerificationOutcome:
        del entities, ontology, model, timeout_seconds, document_subjects
        self.openings.append(document_opening)
        self.judged_entities.extend(entities_to_verify or [])
        return VerificationOutcome(
            decisions=[
                VerificationDecision(
                    relation_id=relation.id,
                    verdict="supported"
                    if relation.relation_type in self.supported
                    else "contradicted",
                    confidence=0.9,
                )
                for relation in relations
            ],
            entity_decisions=[
                EntityVerificationDecision(
                    entity_id=entity.id,
                    verdict=self.rejected.get(entity.name, "specific"),
                )
                for entity in entities_to_verify or []
            ],
            trace={
                "stage": "relation_verification",
                "window_id": window.id,
                "document_id": window.document_id,
            },
        )
