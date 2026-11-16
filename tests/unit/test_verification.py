"""Verification judges each window's relations and entities and records what it turns away."""

from __future__ import annotations

import json
from typing import Any

from documents import chunk, window

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application import two_pass_extraction
from kg_processor.application.extraction_contracts import (
    VerificationCandidateBatch,
    relation_grounding_request,
    relation_verification_request,
)
from kg_processor.application.extraction_windows import build_document_openings
from kg_processor.application.llm_extractors import LlmRelationExtractor
from kg_processor.application.prompt_registry import get_prompt_template
from kg_processor.application.rejected_records import rejected_records_from_trace
from kg_processor.application.two_pass_extraction import (
    apply_entity_verdicts,
    extract_relation_observations,
)
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
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ontology import OntologyProfile
from kg_processor.ports.llm import (
    StructuredCompletionRequest,
    StructuredCompletionResult,
    TruncatedChatResponseError,
)

_OPENING = chunk("Data sheet for Acme Paint\nIssued by Beta Labs", chunk_id="opening")
_BODY = chunk(
    "Colour: blue\nContains titanium white and other pigments\nMade by Gamma Works",
    chunk_id="body",
    chunk_index=1,
    page_number=2,
)


def _profile() -> OntologyProfile:
    return OntologyProfile.model_validate(
        {
            "name": "products",
            "description": "Products, what they contain, and who makes them",
            "mode": "closed",
            "entity_types": [
                {
                    "name": "PRODUCT",
                    "description": "A named product",
                    "document_context": True,
                    "document_subject": True,
                },
                {
                    "name": "MATERIAL",
                    "description": "A named material",
                    "guidance": "A material has a name of its own.",
                    "counter_examples": ["a class of materials"],
                },
                {"name": "ORGANIZATION", "description": "A named organisation"},
            ],
            "relation_types": [
                {
                    "name": "MADE_BY",
                    "description": "A product is made by an organisation",
                    "source_types": ["PRODUCT"],
                    "target_types": ["ORGANIZATION"],
                    "guidance": "The text names the maker of the product.",
                    "counter_examples": ["an organisation line under an author list"],
                },
                {
                    "name": "CONTAINS",
                    "description": "A product contains a material",
                    "source_types": ["PRODUCT"],
                    "target_types": ["MATERIAL"],
                },
                {
                    "name": "SUPPLIES",
                    "description": "An organisation supplies a product",
                    "source_types": ["ORGANIZATION"],
                    "target_types": ["PRODUCT"],
                },
                {
                    "name": "SUPPLIED_BY",
                    "description": "A material is supplied by an organisation",
                    "source_types": ["MATERIAL"],
                    "target_types": ["ORGANIZATION"],
                },
            ],
        }
    )


def _mention(name: str, entity_type: str, source: Chunk, **flags: bool) -> EntityMention:
    return EntityMention(
        id=name,
        name=name,
        type=entity_type,
        description="",
        source_chunk_id=source.id,
        quote=name,
        **flags,
    )


def _inventory() -> list[EntityMention]:
    return [
        _mention(
            "Acme Paint", "PRODUCT", _OPENING, is_document_context=True, is_document_subject=True
        ),
        _mention("Gamma Works", "ORGANIZATION", _BODY),
        _mention("titanium white", "MATERIAL", _BODY),
        _mention("pigments", "MATERIAL", _BODY),
    ]


class _VerifyingLlm(FakeLlmProvider):
    """Propose the body's relations and answer verification from scripted verdicts.

    Relations are keyed by their target's name; an unscripted relation is
    supported and an unscripted entity specific.
    """

    def __init__(
        self,
        relations: list[tuple[str, str, str, str]],
        verdicts: dict[str, tuple[str, str, float]] | None = None,
        entity_verdicts: dict[str, str] | None = None,
        point_at: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.relations = relations
        self.verdicts = verdicts or {}
        self.entity_verdicts = entity_verdicts or {}
        self.point_at = point_at or []
        self.verification_requests: list[StructuredCompletionRequest] = []

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        if request.task_name == "relation_extraction":
            ids = {item["name"]: item["id"] for item in payload["entities"]}
            # The header states nothing here; every relation is in the one body chunk.
            chunk_id = payload["chunks"][0]["id"]
            return StructuredCompletionResult(
                payload={
                    "relations": [
                        {
                            "source_entity_id": ids[source],
                            "target_entity_id": ids[target],
                            "source_surface": source,
                            "target_surface": target,
                            "relation_type": relation_type,
                            "description": quote,
                            "source_chunk_id": chunk_id,
                            "quote": quote,
                            "confidence": 0.9,
                        }
                        for source, relation_type, target, quote in self.relations
                        if source in ids and target in ids and chunk_id != _OPENING.id
                    ]
                }
            )
        if request.task_name == "relation_grounding":
            decisions = [
                {"id": item["id"], "supported": True, "lines": self.point_at}
                for item in payload["relations"]
            ]
            return StructuredCompletionResult(payload={"decisions": decisions})
        if request.task_name == "relation_verification":
            self.verification_requests.append(request)
            decisions = []
            for relation in payload["relations"]:
                verdict, reason, confidence = self.verdicts.get(
                    relation["target_surface"], ("supported", "none", 0.95)
                )
                decisions.append(
                    {
                        "relation_id": relation["id"],
                        "verdict": verdict,
                        "reason": reason,
                        "confidence": confidence,
                    }
                )
            entity_decisions = [
                {
                    "entity_id": item["id"],
                    # "wrong_type:TYPE" scripts a verdict that names the right type.
                    "verdict": self.entity_verdicts.get(item["name"], "specific").split(":")[0],
                    "type": self.entity_verdicts.get(item["name"], ":").partition(":")[2],
                }
                for item in payload["entities_to_verify"]
            ]
            return StructuredCompletionResult(
                payload={"decisions": decisions, "entity_decisions": entity_decisions}
            )
        return super().complete_structured(request)

    def payloads(self) -> list[dict[str, Any]]:
        return [
            json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
            for request in self.verification_requests
        ]


def _settings(**overrides: Any) -> GraphSettings:
    return GraphSettings(
        **{
            "gleaning_max_passes": 0,
            "max_chunks_per_llm_call": 1,
            "extraction_parallelism": 1,
            "relation_empty_retry_min_entities": 0,
            "relation_continuation_max_passes": 0,
            **overrides,
        }
    )


def _extract(
    llm: _VerifyingLlm,
    chunks: list[Chunk] | None = None,
    *,
    inventory: list[EntityMention] | None = None,
    settings: GraphSettings | None = None,
    grounding_fallback: bool = False,
    document_openings: dict[str, str] | None = None,
) -> ExtractionObservations:
    return extract_relation_observations(
        chunks or [_OPENING, _BODY],
        inventory or _inventory(),
        llm,
        settings or _settings(),
        _profile(),
        "fake",
        30,
        relation_extractor=LlmRelationExtractor(llm, grounding_fallback=grounding_fallback),
        document_openings=document_openings,
    )


_BODY_RELATIONS = [
    ("Acme Paint", "MADE_BY", "Gamma Works", "Made by Gamma Works"),
    ("Acme Paint", "CONTAINS", "titanium white", "Contains titanium white and other pigments"),
    ("Acme Paint", "CONTAINS", "pigments", "Contains titanium white and other pigments"),
]


def test_a_verification_request_carries_the_document_opening_and_the_ontology_guidance() -> None:
    """Ontology, then the document, then the window: the shared prefix stays cacheable."""

    llm = _VerifyingLlm(_BODY_RELATIONS)

    _extract(llm)

    [payload] = llm.payloads()
    assert list(payload)[:4] == [
        "relation_type_definitions",
        "entity_type_definitions",
        "document",
        "window_id",
    ]
    assert payload["document"]["opening"].startswith("Data sheet for Acme Paint\nIssued by")
    assert payload["document"]["subjects"] == [{"name": "Acme Paint", "type": "PRODUCT"}]
    made_by = next(
        item for item in payload["relation_type_definitions"] if item["name"] == "MADE_BY"
    )
    assert made_by["guidance"] == "The text names the maker of the product."
    assert made_by["counter_examples"] == ["an organisation line under an author list"]
    material = next(
        item for item in payload["entity_type_definitions"] if item["name"] == "MATERIAL"
    )
    assert material["counter_examples"] == ["a class of materials"]
    # The window's own mentions are judged; the document's subject is not.
    assert [item["name"] for item in payload["entities_to_verify"]] == [
        "Gamma Works",
        "titanium white",
        "pigments",
    ]
    schema = llm.verification_requests[0].json_schema
    assert schema["properties"]["decisions"]["minItems"] == 3
    assert schema["properties"]["decisions"]["maxItems"] == 3
    assert schema["properties"]["entity_decisions"]["minItems"] == 3
    assert schema["properties"]["entity_decisions"]["maxItems"] == 3


def test_a_window_holding_only_part_of_a_document_verifies_with_the_opening_it_is_given() -> None:
    """A fleet task sees one window; the opening travels with the document inventory."""

    llm = _VerifyingLlm(_BODY_RELATIONS)

    _extract(llm, [_BODY], document_openings={_BODY.document_id: "Data sheet for Acme Paint"})

    [payload] = llm.payloads()
    assert payload["document"]["opening"] == "Data sheet for Acme Paint"


def test_a_request_with_nothing_of_one_kind_to_judge_leaves_that_list_out() -> None:
    """An empty id enum is not a schema every provider accepts."""

    extraction_window = window(_BODY)
    request = relation_verification_request(
        extraction_window,
        [],
        [],
        _profile(),
        model="fake",
        timeout_seconds=30,
        seed=17,
        entities_to_verify=[_mention("pigments", "MATERIAL", _BODY)],
    )

    assert set(request.json_schema["properties"]) == {"entity_decisions"}
    assert request.json_schema["required"] == ["entity_decisions"]


def test_a_full_entity_batch_gets_room_for_every_decision() -> None:
    """Each entity decision repeats a long mention id; the answer must not be cut short.

    A model that ran out of tokens mid-answer failed its task on every retry.
    """

    entities = [_mention(f"entity_mention_{index:032x}", "MATERIAL", _BODY) for index in range(40)]
    request = relation_verification_request(
        window(_BODY),
        [],
        [],
        _profile(),
        model="fake",
        timeout_seconds=30,
        seed=17,
        entities_to_verify=entities,
    )
    decision = json.dumps(
        {"entity_id": entities[0].id, "verdict": "specific", "type": ""}, separators=(",", ":")
    )

    # A token covers at least two characters of such JSON, ids included.
    assert request.max_tokens >= 40 * len(decision) // 2


def test_a_full_grounding_batch_gets_room_for_six_long_line_numbers_each() -> None:
    """A tokenizer may spend a token per digit; six four-digit lines must still fit."""

    units = [(number, f"line {number}") for number in range(1000, 1400)]
    statements = [(index, "Acme Paint", "MADE_BY", "Gamma Works") for index in range(40)]
    request = relation_grounding_request(
        units, statements, model="fake", timeout_seconds=30, seed=17
    )
    decision = json.dumps(
        {"id": "r39", "supported": True, "lines": [1394, 1395, 1396, 1397, 1398, 1399]}
    )

    # Digits and punctuation can take a token each: allow one per character.
    assert request.max_tokens >= 40 * len(decision)


def test_every_relation_the_verifier_turns_away_is_recorded_with_its_reason() -> None:
    """Unsupported relations carry the rule of evidence they fail; a weak yes is not a yes."""

    llm = _VerifyingLlm(
        _BODY_RELATIONS,
        verdicts={
            "titanium white": ("insufficient", "composition", 0.9),
            "pigments": ("supported", "none", 0.5),
        },
    )

    observations = _extract(llm)

    assert [(item.relation_type, item.target_entity_id) for item in observations.relations] == [
        ("MADE_BY", "Gamma Works")
    ]
    [event] = [item for item in observations.trace if item["stage"] == "relation_verification"]
    assert [(item["reason"], item["target"]) for item in event["rejected_records"]] == [
        ("verification_composition", "titanium white"),
        ("verification_low_confidence", "pigments"),
    ]
    rows = rejected_records_from_trace(observations.trace, [_OPENING, _BODY], "graph")
    assert sorted(row.reason for row in rows) == [
        "verification_composition",
        "verification_low_confidence",
    ]
    assert {row.page_number for row in rows} == {2}


def test_a_failed_verification_call_records_the_relations_it_could_not_keep() -> None:
    class _FailingVerifier(_VerifyingLlm):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            if request.task_name == "relation_verification":
                raise ValueError("malformed response")
            return super().complete_structured(request)

    observations = _extract(_FailingVerifier(_BODY_RELATIONS))

    assert observations.relations == []
    [event] = [
        item for item in observations.trace if item["stage"] == "relation_verification_error"
    ]
    assert {item["reason"] for item in event["rejected_records"]} == {"verification_error"}
    assert len(event["rejected_records"]) == 3


# Two ends six lines apart: no run of three sentences names both.
_FAR = chunk(
    "Titanium white is the base.\nIt is milled.\nIt is sieved.\nIt is dried.\n"
    "It is packed.\nDelta Minerals ships it.",
    chunk_id="far",
    chunk_index=2,
    page_number=3,
)


def _located(llm: _VerifyingLlm, **settings: Any) -> ExtractionObservations:
    """Extract a relation string grounding cannot place, which the model then points at."""

    return _extract(
        llm,
        [_OPENING, _FAR],
        inventory=[
            _inventory()[0],
            _mention("Titanium white", "MATERIAL", _FAR),
            _mention("Delta Minerals", "ORGANIZATION", _FAR),
        ],
        settings=_settings(**settings),
        grounding_fallback=True,
    )


_LOCATED_RELATION = [
    ("Titanium white", "SUPPLIED_BY", "Delta Minerals", "Delta Minerals supplies titanium white")
]


def test_a_relation_the_model_only_pointed_at_is_verified_even_when_others_are_not() -> None:
    """Located evidence is kept only on the verifier's word."""

    unsupported = _VerifyingLlm(
        _LOCATED_RELATION,
        verdicts={"Delta Minerals": ("insufficient", "adjacency", 0.9)},
        point_at=[6],
    )

    observations = _located(unsupported, verify_relations=False)

    assert observations.relations == []
    [event] = [item for item in observations.trace if item["stage"] == "relation_verification"]
    assert [item["reason"] for item in event["rejected_records"]] == ["verification_adjacency"]

    supported = _VerifyingLlm(_LOCATED_RELATION, point_at=[6])
    [relation] = _located(supported, verify_relations=False).relations
    assert relation.evidence_method == "llm_located"
    assert relation.quote == _FAR.content
    # A quoted relation is not verified when verification is off.
    quoted = _VerifyingLlm(_BODY_RELATIONS[:1])
    assert len(_extract(quoted, settings=_settings(verify_relations=False)).relations) == 1
    assert [len(payload["relations"]) for payload in quoted.payloads()] == [0]


def test_lines_that_do_not_name_both_ends_go_to_the_verifier_instead_of_being_dropped() -> None:
    """A row about a subject its header names once is judged with the document's opening."""

    # The row names neither the product nor its maker as the model has them.
    relations = [("Gamma Works", "MADE_BY", "Acme Paint", "Acme Paint is blue")]
    profile = _profile()
    mentions = [
        _mention("Acme Paint", "PRODUCT", _OPENING),
        _mention("Gamma Works", "ORGANIZATION", _OPENING),
    ]

    class _RowLlm(_VerifyingLlm):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            if request.task_name == "relation_extraction":
                relation = {
                    "source_entity_id": "Acme Paint",
                    "target_entity_id": "Gamma Works",
                    "source_surface": "Acme Paint",
                    "target_surface": "Gamma Works",
                    "relation_type": "MADE_BY",
                    "description": "Acme Paint is made by Gamma Works.",
                    "source_chunk_id": _BODY.id,
                    "quote": "Acme Paint is made by Gamma Works",
                    "confidence": 0.9,
                }
                return StructuredCompletionResult(payload={"relations": [relation]})
            return super().complete_structured(request)

    outcome = LlmRelationExtractor(_RowLlm(relations, point_at=[1]), grounding_fallback=True)
    located = outcome.extract(
        window(_BODY), mentions, profile, model="fake", timeout_seconds=30, max_relations=10
    )

    [relation] = located.relations
    assert relation.evidence_method == "llm_verified"
    assert relation.verification_required is True
    assert relation.quote == "Colour: blue"
    assert (relation.source_surface, relation.target_surface) == ("Acme Paint", "Gamma Works")
    assert located.trace["record_actions"]["llm_grounding_to_verifier"] == 1
    assert located.trace["rejected_records"] == []


def test_a_relation_sent_to_the_verifier_is_kept_only_when_it_is_supported() -> None:
    """The product, named only in the header, is the target of a row that names its supplier."""

    relations = [("Gamma Works", "SUPPLIES", "Acme Paint", "Gamma Works supplies the paint")]
    llm = _VerifyingLlm(relations, point_at=[3])
    kept = _extract(llm, grounding_fallback=True)
    [relation] = kept.relations
    assert relation.evidence_method == "llm_verified"
    assert relation.quote == "Made by Gamma Works"
    [payload] = llm.payloads()
    assert payload["document"]["opening"].startswith("Data sheet for Acme Paint")

    refused = _extract(
        _VerifyingLlm(
            relations, verdicts={"Acme Paint": ("insufficient", "not_stated", 0.9)}, point_at=[3]
        ),
        grounding_fallback=True,
    )
    assert refused.relations == []
    [event] = [item for item in refused.trace if item["stage"] == "relation_verification"]
    assert [item["reason"] for item in event["rejected_records"]] == ["verification_not_stated"]


# A table whose rows each name their own material and supplier.
_ROWS = chunk(
    "Material~Supplier~Grade\nTitanium white~Delta Minerals~Fine\nZinc white~Epsilon Oxides~Coarse",
    chunk_id="rows",
    chunk_index=2,
    page_number=3,
)
_PROSE = chunk(
    "A titanium white grade comprises: a binder; a filler; a dispersant.",
    chunk_id="prose",
    chunk_index=2,
)


def _cited(  # noqa: PLR0913 - one relation as the model states it.
    source_chunk: Chunk,
    source: EntityMention,
    relation_type: str,
    target: EntityMention,
    *,
    point_at: list[int],
    source_surface: str | None = None,
    quote: str | None = None,
) -> RelationExtractionOutcome:
    """Extract one relation the model cites in ``source_chunk``.

    Without a ``quote`` the model's quote is not in the chunk, and when asked
    the model points at the units ``point_at`` numbers.
    """

    relation = {
        "source_entity_id": source.id,
        "target_entity_id": target.id,
        "source_surface": source_surface or source.name,
        "target_surface": target.name,
        "relation_type": relation_type,
        "description": f"{source.name} {relation_type} {target.name}.",
        "source_chunk_id": source_chunk.id,
        "quote": quote or f"{source.name} {relation_type.lower()} {target.name}",
        "confidence": 0.9,
    }

    class _CitingLlm(_VerifyingLlm):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            if request.task_name == "relation_extraction":
                return StructuredCompletionResult(payload={"relations": [relation]})
            return super().complete_structured(request)

    extractor = LlmRelationExtractor(_CitingLlm([], point_at=point_at), grounding_fallback=True)
    return extractor.extract(
        window(source_chunk),
        [source, target],
        _profile(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )


def test_a_row_that_names_both_ends_grounds_their_relation() -> None:
    outcome = _cited(
        _ROWS,
        _mention("Titanium white", "MATERIAL", _ROWS),
        "SUPPLIED_BY",
        _mention("Delta Minerals", "ORGANIZATION", _ROWS),
        point_at=[2],
    )

    [relation] = outcome.relations
    assert relation.quote == "Titanium white~Delta Minerals~Fine"
    assert "row_endpoint_unnamed" not in outcome.trace["record_actions"]


def test_a_row_names_an_end_by_its_name_whatever_its_hyphens_and_case() -> None:
    """The model's surface matches as written; the name matches without punctuation."""

    rows = chunk("Material~Supplier~Grade\nTRIZORVANE CITRATE DIHYDRATE~Delta Minerals~Fine")
    outcome = _cited(
        rows,
        _mention("tri-Zorvane citrate dihydrate", "MATERIAL", rows),
        "SUPPLIED_BY",
        _mention("Delta Minerals", "ORGANIZATION", rows),
        point_at=[],
        source_surface="TRIZORVANE CITRATE DIHYDRATE",
        quote="TRIZORVANE CITRATE DIHYDRATE~Delta Minerals",
    )

    [relation] = outcome.relations
    assert relation.evidence_method == "llm"
    assert relation.quote == "TRIZORVANE CITRATE DIHYDRATE~Delta Minerals"


def test_a_surface_that_matches_the_neighbouring_rows_cell_is_rejected() -> None:
    """The quote grounds as written, but its row is another salt of the same substance."""

    rows = chunk(
        "Material~Supplier~Grade\n"
        "Zorvane phosphate, anhydrous~Delta Minerals~Fine\n"
        "Zorvane phosphate, monohydrate~Epsilon Oxides~Coarse"
    )
    outcome = _cited(
        rows,
        _mention("Zorvane phosphate, monohydrate", "MATERIAL", rows),
        "SUPPLIED_BY",
        _mention("Delta Minerals", "ORGANIZATION", rows),
        point_at=[],
        source_surface="Zorvane phosphate",
        quote="Zorvane phosphate, anhydrous~Delta Minerals",
    )

    assert outcome.relations == []
    assert outcome.trace["record_actions"]["row_endpoint_unnamed"] == 1
    assert [(item["reason"], item["source"]) for item in outcome.trace["rejected_records"]] == [
        ("row_endpoint_unnamed", "Zorvane phosphate, monohydrate")
    ]


def test_a_row_that_names_only_one_end_is_rejected() -> None:
    """The supplier is named elsewhere in the document; the row is another supplier's."""

    outcome = _cited(
        _ROWS,
        _mention("Titanium white", "MATERIAL", _ROWS),
        "SUPPLIED_BY",
        _mention("Theta Colours", "ORGANIZATION", _OPENING),
        point_at=[2],
    )

    assert outcome.relations == []
    actions = outcome.trace["record_actions"]
    assert actions["row_endpoint_unnamed"] == 1
    assert "llm_grounding_to_verifier" not in actions
    assert [(item["reason"], item["target"]) for item in outcome.trace["rejected_records"]] == [
        ("row_endpoint_unnamed", "Theta Colours")
    ]


def test_a_row_about_the_document_subject_may_leave_its_name_out() -> None:
    """A data sheet names its subject once, in its header; each row below is about it."""

    subject = _mention(
        "Acme Paint", "PRODUCT", _OPENING, is_document_context=True, is_document_subject=True
    )

    contains = _cited(
        _ROWS, subject, "CONTAINS", _mention("Titanium white", "MATERIAL", _ROWS), point_at=[]
    )
    [relation] = contains.relations
    assert relation.quote == "Titanium white~Delta Minerals~Fine"
    assert relation.source_surface == "Acme Paint"
    assert contains.trace["record_actions"]["implicit_document_source"] == 1

    supplies = _cited(
        _ROWS, _mention("Delta Minerals", "ORGANIZATION", _ROWS), "SUPPLIES", subject, point_at=[2]
    )
    [relation] = supplies.relations
    assert relation.evidence_method == "llm_verified"
    assert relation.quote == "Titanium white~Delta Minerals~Fine"
    assert relation.target_surface == "Acme Paint"
    assert supplies.trace["rejected_records"] == []


def test_an_html_row_that_leaves_an_end_unnamed_still_goes_to_the_verifier() -> None:
    """An HTML table's subject is often its header, and a cell may list several values."""

    table = chunk("<table><tr><td>Titanium white</td><td>Fine</td></tr></table>", chunk_id="table")
    outcome = _cited(
        table,
        _mention("Titanium white", "MATERIAL", table),
        "SUPPLIED_BY",
        _mention("Theta Colours", "ORGANIZATION", _OPENING),
        point_at=[1],
    )

    [relation] = outcome.relations
    assert relation.evidence_method == "llm_verified"
    assert "row_endpoint_unnamed" not in outcome.trace["record_actions"]


def test_prose_that_leaves_an_end_unnamed_still_goes_to_the_verifier() -> None:
    """Semicolons that divide a sentence's list are not cells."""

    outcome = _cited(
        _PROSE,
        _mention("titanium white", "MATERIAL", _PROSE),
        "SUPPLIED_BY",
        _mention("Theta Colours", "ORGANIZATION", _OPENING),
        point_at=[1],
    )

    [relation] = outcome.relations
    assert relation.evidence_method == "llm_verified"
    assert relation.quote == _PROSE.content
    assert outcome.trace["record_actions"]["llm_grounding_to_verifier"] == 1


def _combined(llm: _VerifyingLlm) -> ExtractionObservations:
    relations = _extract(llm)
    return apply_entity_verdicts(
        ExtractionObservations(
            entities=_inventory(),
            relations=relations.relations,
            trace=relations.trace,
            chunk_count=2,
            window_count=2,
        ),
        _profile(),
    )


def test_a_generic_mention_that_ends_a_stated_relation_keeps_its_node() -> None:
    """'Contains ... pigments' states what the paint contains; the class is the fact's end."""

    combined = _combined(_VerifyingLlm(_BODY_RELATIONS, entity_verdicts={"pigments": "generic"}))

    assert [entity.name for entity in combined.entities] == [
        "Acme Paint",
        "Gamma Works",
        "titanium white",
        "pigments",
    ]
    assert sorted(item.target_entity_id for item in combined.relations) == [
        "Gamma Works",
        "pigments",
        "titanium white",
    ]
    assert rejected_records_from_trace(combined.trace, [_OPENING, _BODY], "graph") == []


def test_a_generic_mention_no_stated_relation_uses_leaves() -> None:
    llm = _VerifyingLlm(
        _BODY_RELATIONS,
        verdicts={"pigments": ("insufficient", "not_stated", 0.9)},
        entity_verdicts={"pigments": "generic"},
    )

    combined = _combined(llm)

    assert [entity.name for entity in combined.entities] == [
        "Acme Paint",
        "Gamma Works",
        "titanium white",
    ]
    rows = rejected_records_from_trace(combined.trace, [_OPENING, _BODY], "graph")
    assert sorted((row.kind, row.reason, row.name) for row in rows) == [
        ("entity", "verification_generic", "pigments"),
        ("relation", "verification_not_stated", "CONTAINS"),
    ]
    # Applied again, nothing more is removed or recorded.
    assert apply_entity_verdicts(combined, _profile()) == combined


def test_a_placeholder_leaves_with_the_relations_that_used_it() -> None:
    """A pointer such as 'the product' is no node, whatever is said about it."""

    combined = _combined(
        _VerifyingLlm(_BODY_RELATIONS, entity_verdicts={"pigments": "placeholder"})
    )

    assert sorted(item.target_entity_id for item in combined.relations) == [
        "Gamma Works",
        "titanium white",
    ]
    rows = rejected_records_from_trace(combined.trace, [_OPENING, _BODY], "graph")
    assert sorted((row.kind, row.reason, row.name) for row in rows) == [
        ("entity", "verification_placeholder", "pigments"),
        ("relation", "verification_endpoint_rejected", "CONTAINS"),
    ]
    assert apply_entity_verdicts(combined, _profile()) == combined


def test_entity_verification_can_be_turned_off() -> None:
    llm = _VerifyingLlm(_BODY_RELATIONS, entity_verdicts={"pigments": "generic"})

    observations = _extract(llm, settings=_settings(verify_entities=False))

    [payload] = llm.payloads()
    assert payload["entities_to_verify"] == []
    assert not any(
        item.get("entity_verdicts")
        for item in observations.trace
        if item["stage"] == "relation_verification"
    )


def test_a_document_opening_reads_its_first_chunks_once_each() -> None:
    """Overlapping chunks of one page are read once; the limit cuts the text."""

    first = chunk("Location: Lab / Trial 3", chunk_id="location", page_number=0)
    second = chunk("Data sheet for Acme Paint. Colour blue.", chunk_id="page-1", chunk_index=1)
    overlapping = chunk(
        "Colour blue. Made by Gamma Works.",
        chunk_id="page-1b",
        chunk_index=2,
        start=len("Data sheet for Acme Paint. "),
    )
    other = chunk("Another document", chunk_id="other", file_id="file_2")

    openings = build_document_openings([overlapping, first, second, other])

    assert openings == {
        "file_1": (
            "Location: Lab / Trial 3\nData sheet for Acme Paint. Colour blue. Made by Gamma Works."
        ),
        "file_2": "Another document",
    }
    assert build_document_openings([first, second], max_characters=10) == {"file_1": "Location:"}


def test_the_opening_is_only_as_long_as_the_limit() -> None:
    long = chunk("x" * 5000, chunk_id="long")

    assert len(build_document_openings([long])["file_1"]) == 1500


def _relation_observation(target: str) -> RelationObservation:
    return RelationObservation(
        id=target,
        source_entity_id="Acme Paint",
        target_entity_id=target,
        relation_type="CONTAINS",
        description="",
        source_chunk_id=_BODY.id,
        quote=_BODY.content,
    )


def test_nothing_is_dropped_when_no_entity_was_rejected() -> None:
    observations = ExtractionObservations(
        entities=_inventory(),
        relations=[_relation_observation("pigments")],
        trace=[{"stage": "relation_verification", "rejected_entity_ids": []}],
        chunk_count=2,
        window_count=2,
    )

    assert apply_entity_verdicts(observations, _profile()) == observations


def test_the_prompt_explains_every_reason_and_verdict_the_schema_allows() -> None:
    """A code the model may answer with, but was never told the meaning of, is a guess."""

    text = get_prompt_template("relation_verification").text
    schema = VerificationCandidateBatch.model_json_schema()["$defs"]
    reasons = schema["VerificationCandidate"]["properties"]["reason"]["enum"]
    verdicts = schema["EntityVerificationCandidate"]["properties"]["verdict"]["enum"]

    assert [reason for reason in reasons if reason not in text] == []
    assert [verdict for verdict in verdicts if verdict not in text] == []


def test_a_real_thing_under_the_wrong_type_is_retyped_not_lost() -> None:
    """The verifier names the right type; the mention keeps its evidence and its stated type."""

    llm = _VerifyingLlm(_BODY_RELATIONS, entity_verdicts={"Gamma Works": "wrong_type:MATERIAL"})

    observations = _extract(llm)

    schema = llm.verification_requests[0].json_schema
    item = schema["$defs"]["EntityVerificationCandidate"]["properties"]["type"]
    # Every verdict names a configured type, so a wrong type always has one to keep.
    assert item["enum"] == ["PRODUCT", "MATERIAL", "ORGANIZATION"]
    assert "type" in schema["$defs"]["EntityVerificationCandidate"]["required"]
    [event] = [item for item in observations.trace if item["stage"] == "relation_verification"]
    assert event["retyped_entities"] == [
        {"mention_id": "Gamma Works", "from": "ORGANIZATION", "to": "MATERIAL"}
    ]
    assert event["record_actions"]["retyped_entity"] == 1
    assert event["entity_verdicts"] == []
    assert event["rejected_records"] == []

    combined = apply_entity_verdicts(
        ExtractionObservations(
            entities=_inventory(),
            relations=observations.relations,
            trace=observations.trace,
            chunk_count=2,
            window_count=2,
        ),
        _profile(),
    )

    retyped = next(entity for entity in combined.entities if entity.id == "Gamma Works")
    assert (retyped.type, retyped.proposed_type) == ("MATERIAL", "ORGANIZATION")
    assert retyped.quote == "Gamma Works"
    # MADE_BY needs an organisation: under its new type the relation breaks its rules.
    assert sorted(item.target_entity_id for item in combined.relations) == [
        "pigments",
        "titanium white",
    ]
    rows = rejected_records_from_trace(combined.trace, [_OPENING, _BODY], "graph")
    assert [(row.kind, row.reason, row.name, row.target_type) for row in rows] == [
        ("relation", "domain_or_range_violation", "MADE_BY", "MATERIAL")
    ]
    assert apply_entity_verdicts(combined, _profile()) == combined


def test_a_relation_on_a_retyped_mention_is_kept_under_the_rewrite_its_new_types_allow() -> None:
    """Relations are re-checked the way extraction checks them: own rules, then a rewrite."""

    profile = OntologyProfile.model_validate(
        {
            **_profile().model_dump(),
            "relation_rewrites": [
                {"relation": "MADE_BY", "target_types": ["MATERIAL"], "to": "CONTAINS"}
            ],
        }
    )
    made_by = RelationObservation(
        id="made-by",
        source_entity_id="Acme Paint",
        target_entity_id="Gamma Works",
        relation_type="MADE_BY",
        description="",
        source_chunk_id=_BODY.id,
        quote="Made by Gamma Works",
    )
    untouched = _relation_observation("pigments")

    combined = apply_entity_verdicts(
        ExtractionObservations(
            entities=_inventory(),
            relations=[made_by, untouched],
            trace=[
                {
                    "stage": "relation_verification",
                    "retyped_entities": [
                        {"mention_id": "Gamma Works", "from": "ORGANIZATION", "to": "MATERIAL"}
                    ],
                }
            ],
            chunk_count=2,
            window_count=2,
        ),
        profile,
    )

    kept = {relation.id: relation for relation in combined.relations}
    assert (kept["made-by"].relation_type, kept["made-by"].proposed_relation_type) == (
        "CONTAINS",
        "MADE_BY",
    )
    assert kept[untouched.id] == untouched
    [cascade] = [item for item in combined.trace if item["stage"] == "verification_cascade"]
    assert cascade["retyped_entities"] == 1
    assert cascade["record_actions"] == {"rewritten_relation": 1}
    assert cascade["rejected_records"] == []


def test_wrong_type_naming_no_other_configured_type_keeps_the_mention() -> None:
    """An unknown type or the type it already has: nothing to move it to, nothing to remove."""

    for verdict in ("wrong_type", "wrong_type:SUBSTANCE", "wrong_type:MATERIAL"):
        llm = _VerifyingLlm(_BODY_RELATIONS, entity_verdicts={"pigments": verdict})

        observations = _extract(llm)

        [event] = [item for item in observations.trace if item["stage"] == "relation_verification"]
        assert (event["retyped_entities"], event["entity_verdicts"]) == ([], [])
        combined = _combined(llm)
        assert "pigments" in [entity.name for entity in combined.entities]
        assert rejected_records_from_trace(combined.trace, [_OPENING, _BODY], "graph") == []


def test_a_thing_no_configured_type_covers_leaves_with_its_relations() -> None:
    llm = _VerifyingLlm(_BODY_RELATIONS, entity_verdicts={"pigments": "out_of_scope"})

    combined = _combined(llm)

    assert "pigments" not in [entity.name for entity in combined.entities]
    rows = rejected_records_from_trace(combined.trace, [_OPENING, _BODY], "graph")
    assert sorted((row.kind, row.reason) for row in rows) == [
        ("entity", "verification_out_of_scope"),
        ("relation", "verification_endpoint_rejected"),
    ]


def _window() -> ExtractionWindow:
    return ExtractionWindow(
        id="window", document_id=_BODY.document_id, chunks=[_OPENING, _BODY], token_count=10
    )


def _answering(
    relations: list[RelationObservation], entities: list[EntityMention]
) -> VerificationOutcome:
    """A reply deciding every relation and entity it was asked about."""

    return VerificationOutcome(
        decisions=[
            VerificationDecision(relation_id=item.id, verdict="supported", confidence=0.9)
            for item in relations
        ],
        entity_decisions=[
            EntityVerificationDecision(entity_id=item.id, verdict="specific", type=item.type)
            for item in entities
        ],
    )


def test_a_batch_cut_off_at_its_output_limit_is_verified_in_halves() -> None:
    """Splitting is what saves the batch: an identical request is cut off identically."""

    relations = [_relation_observation(f"pigment {index}") for index in range(5)]
    entities = [_mention(f"thing {index}", "MATERIAL", _BODY) for index in range(3)]
    calls: list[int] = []
    failures: list[tuple[int, int]] = []
    trace: list[dict[str, object]] = []

    def verify(
        relation_batch: list[RelationObservation], entity_batch: list[EntityMention]
    ) -> VerificationOutcome:
        size = len(relation_batch) + len(entity_batch)
        calls.append(size)
        if size > 2:
            raise TruncatedChatResponseError("stopped at the output-token limit")
        return _answering(relation_batch, entity_batch)

    done = two_pass_extraction._verify_batches(
        verify,
        [(relations, entities)],
        lambda relation_batch, entity_batch, _exc: failures.append(
            (len(relation_batch), len(entity_batch))
        ),
        trace,
        _window(),
    )

    verified = [item for batch, _, _ in done for item in batch]
    judged = [item for _, batch, _ in done for item in batch]
    assert sorted(item.id for item in verified) == sorted(item.id for item in relations)
    assert sorted(item.id for item in judged) == sorted(item.id for item in entities)
    assert failures == []
    assert [event["stage"] for event in trace] == ["relation_verification_split"] * 3


def test_a_single_item_cut_off_at_its_output_limit_is_recorded_as_failed() -> None:
    failures: list[tuple[int, int]] = []

    def verify(
        relation_batch: list[RelationObservation], entity_batch: list[EntityMention]
    ) -> VerificationOutcome:
        raise TruncatedChatResponseError("stopped at the output-token limit")

    done = two_pass_extraction._verify_batches(
        verify,
        [([_relation_observation("pigments")], [])],
        lambda relation_batch, entity_batch, _exc: failures.append(
            (len(relation_batch), len(entity_batch))
        ),
        [],
        _window(),
    )

    assert (done, failures) == ([], [(1, 0)])


def test_what_a_reply_leaves_undecided_is_asked_again_once() -> None:
    """A skipped relation is a lost fact; asking once more for just it usually recovers it."""

    relations = [_relation_observation(f"pigment {index}") for index in range(3)]
    asked: list[list[str]] = []

    def verify(
        relation_batch: list[RelationObservation], entity_batch: list[EntityMention]
    ) -> VerificationOutcome:
        asked.append([item.id for item in relation_batch])
        # The first reply skips the last relation; every later one skips it too.
        return _answering(
            [item for item in relation_batch if item.id != "pigment 2" or len(asked) == 2], []
        )

    trace: list[dict[str, object]] = []
    done = two_pass_extraction._verify_batches(
        verify, [(relations, [])], lambda *_: None, trace, _window()
    )

    assert asked == [["pigment 0", "pigment 1", "pigment 2"], ["pigment 2"]]
    assert [[item.id for item in batch] for batch, _, _ in done] == [
        ["pigment 0", "pigment 1"],
        ["pigment 2"],
    ]
    assert [event["stage"] for event in trace] == ["relation_verification_reask"]


def test_a_relation_left_undecided_twice_is_not_asked_a_third_time() -> None:
    asked: list[list[str]] = []

    def verify(
        relation_batch: list[RelationObservation], entity_batch: list[EntityMention]
    ) -> VerificationOutcome:
        asked.append([item.id for item in relation_batch])
        return VerificationOutcome()

    done = two_pass_extraction._verify_batches(
        verify, [([_relation_observation("pigments")], [])], lambda *_: None, [], _window()
    )

    assert asked == [["pigments"], ["pigments"]]
    # The second reply's batch still lists it, so the caller records it as omitted.
    assert [[item.id for item in batch] for batch, _, _ in done] == [[], ["pigments"]]


def test_a_verification_decision_asks_for_its_verdict_and_nothing_to_read() -> None:
    """Every token a decision carries is generated; prose nobody reads is pure cost."""

    llm = _VerifyingLlm(_BODY_RELATIONS)

    _extract(llm)

    schema = llm.verification_requests[0].json_schema
    item = schema["$defs"]["VerificationCandidate"]["properties"]
    assert sorted(item) == ["confidence", "reason", "relation_id", "verdict"]
