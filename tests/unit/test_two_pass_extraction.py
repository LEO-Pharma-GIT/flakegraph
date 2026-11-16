from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from documents import chunk, window

from kg_processor.adapters.embeddings.hash import HashEmbeddingProvider
from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.content_kinds import LOCATION_PAGE_NUMBER, location_text
from kg_processor.application.extraction_contracts import (
    document_context_extraction_request,
    entity_extraction_request,
    relation_extraction_request,
    relation_verification_request,
)
from kg_processor.application.extraction_windows import window_skip_reason
from kg_processor.application.llm_extractors import LlmEntityExtractor, LlmRelationExtractor
from kg_processor.application.ontology import load_ontology
from kg_processor.application.two_pass_extraction import (
    _WINDOW_FAILURE_DETAIL_LIMIT,
    _all_windows_failed_message,
    _primary_relations,
    _relation_completion_inventory,
    _relation_inventory_for_window,
    _run_window_stage,
    extract_graph_observations,
    extract_graph_two_pass,
    is_reference_only_window,
)
from kg_processor.application.window_errors import is_systemic_provider_error
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionWindow,
    RelationExtractionOutcome,
    RelationObservation,
)
from kg_processor.domain.ontology import OntologyProfile, RelationTypeDefinition
from kg_processor.ports.embeddings import EmbedOptions
from kg_processor.ports.llm import StructuredCompletionRequest, StructuredCompletionResult


class AuthenticationError(RuntimeError):
    """Model a provider-wide credential rejection."""


class _HttpError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = type("Response", (), {"status_code": status_code})()


@pytest.mark.parametrize("status_code", [400, 422])
def test_request_specific_http_errors_are_not_systemic(status_code: int) -> None:
    assert not is_systemic_provider_error(_HttpError(status_code))


@pytest.mark.parametrize("status_code", [401, 403])
def test_auth_http_errors_are_systemic(status_code: int) -> None:
    assert is_systemic_provider_error(_HttpError(status_code))


def test_window_stage_reraises_systemic_authentication_failures() -> None:
    def reject(_window: ExtractionWindow) -> object:
        raise AuthenticationError("invalid API key")

    with pytest.raises(AuthenticationError, match="invalid API key"):
        _run_window_stage(  # noqa: SLF001
            [_window("Alice works at Acme.")],
            GraphSettings(extraction_parallelism=1),
            reject,  # type: ignore[arg-type]
            None,
        )


class _MixedRecordLlm(FakeLlmProvider):
    """Return mixed valid and invalid sibling records across every extraction stage.

    This fixture exercises record-local failure isolation.
    """

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Construct mixed stage payloads while delegating unrelated tasks normally.

        Provider metadata identifies the controlled fixture path.
        """

        if request.task_name == "entity_extraction":
            payload = {
                "entities": [
                    {**_entity("Invalid confidence", "PERSON"), "confidence": -1.0},
                    _entity("Jigoro Kano", "PERSON"),
                    _entity("Kodokan", "SCHOOL"),
                    _entity("Tokyo", "LOCATION"),
                ]
            }
        elif request.task_name == "relation_extraction":
            input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
            ids = {item["name"]: item["id"] for item in input_payload["entities"]}
            quote = "Jigoro Kano founded the Kodokan in Tokyo."
            payload = {
                "relations": [
                    {
                        "source_entity_id": ids["Kodokan"],
                        "target_entity_id": ids["Jigoro Kano"],
                        "source_surface": "Kodokan",
                        "target_surface": "Jigoro Kano",
                        "relation_type": "FOUNDED_BY",
                        "description": "Schema-invalid confidence.",
                        "source_chunk_id": "chunk-1",
                        "quote": quote,
                        "confidence": -1.0,
                    },
                    {
                        # Deliberately reversed: the ontology signature should
                        # repair PERSON -> SCHOOL into SCHOOL -> PERSON.
                        "source_entity_id": ids["Jigoro Kano"],
                        "target_entity_id": ids["Kodokan"],
                        "source_surface": "Jigoro Kano",
                        "target_surface": "Kodokan",
                        "relation_type": "FOUNDED_BY",
                        "description": "Jigoro Kano founded the Kodokan.",
                        "source_chunk_id": "chunk-1",
                        "quote": quote,
                        "confidence": 0.98,
                    },
                    {
                        "source_entity_id": ids["Kodokan"],
                        "target_entity_id": ids["Tokyo"],
                        "source_surface": "Kodokan",
                        "target_surface": "Tokyo",
                        "relation_type": "LOCATED_IN",
                        "description": "The Kodokan was established in Tokyo.",
                        "source_chunk_id": "chunk-1",
                        "quote": quote,
                        "confidence": 0.98,
                    },
                    {
                        "source_entity_id": ids["Kodokan"],
                        "target_entity_id": ids["Kodokan"],
                        "source_surface": "Kodokan",
                        "target_surface": "Kodokan",
                        "relation_type": "FOUNDED_BY",
                        "description": "Invalid reflexive record.",
                        "source_chunk_id": "chunk-1",
                        "quote": quote,
                        "confidence": 0.9,
                    },
                    {
                        "source_entity_id": "missing-id",
                        "target_entity_id": ids["Jigoro Kano"],
                        "source_surface": "Kodokan",
                        "target_surface": "Jigoro Kano",
                        "relation_type": "FOUNDED_BY",
                        "description": "Invalid endpoint record.",
                        "source_chunk_id": "chunk-1",
                        "quote": quote,
                        "confidence": 0.9,
                    },
                ]
            }
        elif request.task_name == "relation_verification":
            input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
            payload = {
                "decisions": [
                    {
                        "relation_id": item["id"],
                        "verdict": "supported",
                        "confidence": 0.97,
                    }
                    for item in input_payload["relations"]
                ]
                + [
                    {
                        "relation_id": input_payload["relations"][0]["id"],
                        "verdict": "supported",
                        "confidence": -1.0,
                    }
                ]
            }
        else:
            return super().complete_structured(request)
        return StructuredCompletionResult(payload=payload, provider_metadata={"provider": "mixed"})


class _CoverageAuditLlm(FakeLlmProvider):
    """Reveal one additional entity and relation only during continuation passes."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return partial first passes and complementary audit-pass records."""

        input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        if request.task_name == "entity_extraction":
            previous = input_payload["previous_entities"]
            names = (
                [("roda", "PRACTICE")]
                if previous
                else [
                    ("Capoeira", "MARTIAL_ART"),
                    ("UNESCO", "ORGANIZATION"),
                ]
            )
            payload = {"entities": [_entity(name, entity_type) for name, entity_type in names]}
        elif request.task_name == "relation_extraction":
            ids = {item["name"]: item["id"] for item in input_payload["entities"]}
            previous = input_payload["previous_relations"]
            if previous:
                payload = {
                    "relations": [
                        _relation(
                            ids["roda"],
                            ids["UNESCO"],
                            "roda",
                            "UNESCO",
                            "RECOGNIZED_BY",
                            "The roda was recognized by UNESCO.",
                        )
                    ]
                }
            else:
                payload = {
                    "relations": [
                        _relation(
                            ids["Capoeira"],
                            ids["roda"],
                            "Capoeira",
                            "roda",
                            "HAS_PRACTICE",
                            "Capoeira includes the roda.",
                        )
                    ]
                }
        elif request.task_name == "relation_verification":
            payload = {
                "decisions": [
                    {
                        "relation_id": item["id"],
                        "verdict": "supported",
                        "confidence": 1.0,
                    }
                    for item in input_payload["relations"]
                ]
            }
        else:
            return super().complete_structured(request)
        return StructuredCompletionResult(payload=payload, provider_metadata={"provider": "audit"})


class _LocalSurfaceLlm(FakeLlmProvider):
    """Map a shortened source mention onto its canonical inventory entity ID."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return one relation whose target surface is shorter than its inventory name."""

        input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in input_payload["entities"]}
        return StructuredCompletionResult(
            payload={
                "relations": [
                    _relation(
                        ids["Capoeira"],
                        ids["Capoeira roda"],
                        "Capoeira",
                        "roda",
                        "HAS_PRACTICE",
                        "Capoeira includes the roda.",
                    )
                ]
            },
            provider_metadata={"provider": "local-surface"},
        )


class _DocumentSubjectRelationLlm(FakeLlmProvider):
    """Connect an explicit document pronoun to its pre-extracted focal paper."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return one context-grounded contribution relation."""

        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in payload["entities"]}
        return StructuredCompletionResult(
            payload={
                "relations": [
                    _relation(
                        ids["Adam: A Method for Stochastic Optimization"],
                        ids["Adam"],
                        "We",
                        "Adam",
                        "INTRODUCES",
                        "We introduce Adam, an algorithm for stochastic optimization.",
                    )
                    | {"source_chunk_id": "reference-chunk"}
                ]
            }
        )


class _ImplicitDocumentSubjectRelationLlm(FakeLlmProvider):
    """Attribute an authorial technical statement through document provenance."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return a paper-to-method relation whose quote omits the paper surface."""

        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in payload["entities"]}
        return StructuredCompletionResult(
            payload={
                "relations": [
                    _relation(
                        ids["Learning representations by back-propagating errors"],
                        ids["Gradient descent"],
                        "source paper",
                        "gradient descent",
                        "USES_METHOD",
                        "To minimize E by gradient descent, it is necessary to "
                        "compute the derivative of E.",
                    )
                    | {"source_chunk_id": "reference-chunk"}
                ]
            }
        )


class _ImplicitMethodSubjectRelationLlm(FakeLlmProvider):
    """Try the provenance exception with a non-paper context entity."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return a method relation whose proposed source surface is absent."""

        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in payload["entities"]}
        return StructuredCompletionResult(
            payload={
                "relations": [
                    _relation(
                        ids["Back-propagation"],
                        ids["Gradient descent"],
                        "the method",
                        "gradient descent",
                        "USES_METHOD",
                        "To minimize E by gradient descent, derivatives are computed efficiently.",
                    )
                    | {"source_chunk_id": "reference-chunk"}
                ]
            }
        )


class _NestedTitleRelationLlm(FakeLlmProvider):
    """Return a paper-to-task claim whose target is nested in the paper title."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Build the overlap case that must rely on document provenance."""

        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in payload["entities"]}
        title = "Understanding the Difficulty of Training Deep Feedforward Neural Networks"
        return StructuredCompletionResult(
            payload={
                "relations": [
                    _relation(
                        ids[title],
                        ids["Training Deep Feedforward Neural Networks"],
                        title,
                        "Training Deep Feedforward Neural Networks",
                        "ADDRESSES",
                        title,
                    )
                    | {"source_chunk_id": "reference-chunk"}
                ]
            }
        )


class _NormalizedEndpointSurfaceLlm(FakeLlmProvider):
    """Return a correct endpoint ID with a shortened surface absent from the OCR text."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Exercise canonical-inventory fallback while preserving an exact source quote."""

        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in payload["entities"]}
        quote = (
            "The Stochastic Gradient Variational Bayes estimator uses the reparameterization trick."
        )
        return StructuredCompletionResult(
            payload={
                "relations": [
                    _relation(
                        ids["Stochastic Gradient Variational Bayes estimator"],
                        ids["reparameterization trick"],
                        "SGVB estimator",
                        "reparameterization trick",
                        "USES_METHOD",
                        quote,
                    )
                    | {"source_chunk_id": "reference-chunk"}
                ]
            }
        )


class _CueRecoveryLlm(FakeLlmProvider):
    """Omit a relation during generation so deterministic cue evidence recovers it."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return entities but deliberately omit the directly asserted relation."""

        input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        if request.task_name == "entity_extraction":
            payload = {
                "entities": [
                    _entity("UFC 1", "EVENT"),
                    _entity("Ultimate Fighting Championship", "ORGANIZATION"),
                ]
            }
        elif request.task_name == "relation_extraction":
            payload = {"relations": []}
        elif request.task_name == "relation_verification":
            names = {item["id"]: item["name"] for item in input_payload["entities"]}
            payload = {
                "decisions": [
                    {
                        "relation_id": item["id"],
                        "verdict": (
                            "supported"
                            if names[item["source_entity_id"]] == "UFC 1"
                            and names[item["target_entity_id"]] == "Ultimate Fighting Championship"
                            and item["relation_type"] == "PART_OF"
                            else "insufficient"
                        ),
                        "confidence": 1.0,
                    }
                    for item in input_payload["relations"]
                ]
            }
        else:
            return super().complete_structured(request)
        return StructuredCompletionResult(payload=payload, provider_metadata={"provider": "cue"})


class _CompoundSubjectAuditLlm(FakeLlmProvider):
    """Miss a compound subject initially and recover it from the focused audit."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return the full compound only when orchestration supplies its cue sentence."""

        input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        if request.task_name == "entity_extraction":
            entities = [
                _entity("Capoeira", "MARTIAL_ART"),
                _entity("UNESCO", "ORGANIZATION"),
            ]
            if str(input_payload["window_id"]).startswith("cue_entity_audit_window"):
                entities = [_entity("capoeira roda", "PRACTICE")]
            payload = {"entities": entities}
        elif request.task_name == "relation_extraction":
            payload = {"relations": []}
        elif request.task_name == "relation_verification":
            payload = {
                "decisions": [
                    {
                        "relation_id": item["id"],
                        "verdict": "insufficient",
                        "confidence": 1.0,
                    }
                    for item in input_payload["relations"]
                ]
            }
        else:
            return super().complete_structured(request)
        return StructuredCompletionResult(
            payload=payload, provider_metadata={"provider": "compound-audit"}
        )


class _IncompleteEndpointAuditLlm(FakeLlmProvider):
    """Recover a relation target after the ordinary entity pass omits it."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Expose HiddenSet only to the cue-focused entity audit."""

        input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        if request.task_name == "document_context_extraction":
            payload: object = {"entities": []}
        elif request.task_name == "entity_extraction":
            entities = [_entity("Method Alpha", "MODEL")]
            if str(input_payload["window_id"]).startswith("cue_entity_audit_window"):
                entities = [_entity("HiddenSet", "DATASET")]
            payload = {"entities": entities}
        elif request.task_name == "relation_extraction":
            ids = {item["name"]: item["id"] for item in input_payload["entities"]}
            payload = (
                {
                    "relations": [
                        _relation(
                            ids["Method Alpha"],
                            ids["HiddenSet"],
                            "Method Alpha",
                            "HiddenSet",
                            "EVALUATED_ON",
                            "Method Alpha was evaluated on HiddenSet.",
                        )
                    ]
                }
                if {"Method Alpha", "HiddenSet"} <= ids.keys()
                else {"relations": []}
            )
        elif request.task_name == "relation_verification":
            payload = {
                "decisions": [
                    {
                        "relation_id": item["id"],
                        "verdict": "supported",
                        "confidence": 1.0,
                    }
                    for item in input_payload["relations"]
                ]
            }
        else:
            return super().complete_structured(request)
        return StructuredCompletionResult(
            payload=payload,
            provider_metadata={"provider": "incomplete-endpoint-audit"},
        )


def test_two_pass_keeps_valid_records_when_siblings_are_invalid() -> None:
    """Ensure record-local validation preserves good records across all extraction stages.

    Rejection reasons and direction repairs must remain auditable.
    """

    passage = chunk("Jigoro Kano founded the Kodokan in Tokyo.", chunk_id="chunk-1")
    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)
    settings = GraphSettings(
        gleaning_max_passes=0,
        entity_resolution_enabled=False,
        verification_min_confidence=0.75,
    )

    result = extract_graph_two_pass(
        [passage],
        _MixedRecordLlm(),
        HashEmbeddingProvider(),
        EmbedOptions(model="hash", dimension=8, batch_size=8),
        settings,
        ontology.profile,
        ontology.checksum,
        ontology.source,
        model="test-model",
        timeout_seconds=30,
    )

    triples = [
        (item.source_name, item.relation_type, item.target_name) for item in result.relations
    ]
    assert triples == [
        ("Kodokan", "FOUNDED_BY", "Jigoro Kano"),
        ("Kodokan", "LOCATED_IN", "Tokyo"),
    ]
    assert result.relations[0].quote == passage.content
    relation_trace = next(
        item for item in result.provider_metadata["trace"] if item["stage"] == "relation_extraction"
    )
    assert relation_trace["record_actions"] == {
        "cue_not_found": 1,
        "invalid_endpoint": 1,
        "invalid_schema": 1,
        "repaired_direction": 1,
        "self_loop": 1,
    }
    entity_trace = next(
        item for item in result.provider_metadata["trace"] if item["stage"] == "entity_extraction"
    )
    assert entity_trace["record_actions"] == {"invalid_schema": 1}
    verification_trace = next(
        item
        for item in result.provider_metadata["trace"]
        if item["stage"] == "relation_verification"
    )
    assert verification_trace["record_actions"] == {"invalid_schema": 1}


def test_gleaning_audits_partially_covered_chunks_for_missed_facts() -> None:
    """Revisit chunks containing some results because presence is not completeness."""

    passage = chunk(
        "Capoeira includes the roda. The roda was recognized by UNESCO.", chunk_id="chunk-1"
    )
    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)
    settings = GraphSettings(
        gleaning_max_passes=1,
        gleaning_min_uncovered_tokens=1,
        entity_resolution_enabled=False,
    )

    result = extract_graph_two_pass(
        [passage],
        _CoverageAuditLlm(),
        HashEmbeddingProvider(),
        EmbedOptions(model="hash", dimension=8, batch_size=8),
        settings,
        ontology.profile,
        ontology.checksum,
        ontology.source,
        model="test-model",
        timeout_seconds=30,
    )

    assert {item.name for item in result.entities} == {"Capoeira", "UNESCO", "roda"}
    assert {
        (item.source_name, item.relation_type, item.target_name) for item in result.relations
    } == {
        ("Capoeira", "HAS_PRACTICE", "roda"),
        ("roda", "RECOGNIZED_BY", "UNESCO"),
    }
    trace = result.provider_metadata["trace"]
    assert sum(item["stage"] == "entity_extraction" for item in trace) == 2
    assert sum(item["stage"] == "relation_extraction" for item in trace) == 2


def test_relation_grounding_accepts_verified_local_endpoint_surfaces() -> None:
    """Ground shortened contextual mentions without weakening canonical inventory IDs."""

    content = "The Capoeira roda is recognized by UNESCO. Capoeira includes the roda."
    extraction_window = _window(content, chunk_id="chunk-1")
    entities = [
        EntityMention(
            id="capoeira",
            name="Capoeira",
            type="MARTIAL_ART",
            description="A martial art.",
            source_chunk_id="chunk-1",
            quote="Capoeira",
        ),
        EntityMention(
            id="roda",
            name="Capoeira roda",
            type="PRACTICE",
            description="The capoeira circle practice.",
            source_chunk_id="chunk-1",
            quote="Capoeira roda",
        ),
    ]
    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)

    outcome = LlmRelationExtractor(_LocalSurfaceLlm()).extract(
        extraction_window,
        entities,
        ontology.profile,
        model="test-model",
        timeout_seconds=30,
        max_relations=10,
    )

    assert len(outcome.relations) == 1
    assert outcome.relations[0].source_surface == "Capoeira"
    assert outcome.relations[0].target_surface == "roda"
    assert outcome.relations[0].quote == "Capoeira includes the roda."


def test_ontology_cues_recover_relations_missed_by_generation() -> None:
    """Retain unambiguous typed cue evidence without a stochastic verifier veto."""

    passage = chunk("UFC 1 was part of the Ultimate Fighting Championship.", chunk_id="chunk-1")
    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)

    result = extract_graph_two_pass(
        [passage],
        _CueRecoveryLlm(),
        HashEmbeddingProvider(),
        EmbedOptions(model="hash", dimension=8, batch_size=8),
        GraphSettings(gleaning_max_passes=0, entity_resolution_enabled=False),
        ontology.profile,
        ontology.checksum,
        ontology.source,
        model="test-model",
        timeout_seconds=30,
    )

    assert [
        (item.source_name, item.relation_type, item.target_name) for item in result.relations
    ] == [("UFC 1", "PART_OF", "Ultimate Fighting Championship")]
    cue_trace = next(
        item
        for item in result.provider_metadata["trace"]
        if item["stage"] == "cue_candidate_generation"
    )
    assert cue_trace["additional_records"] == 1
    assert cue_trace["direct_evidence_records"] == 1


def test_cue_entity_audit_recovers_compound_relation_subject() -> None:
    """Recover a named compound instead of attaching its relation to a substring."""

    passage = chunk(
        "A capoeira roda is a recurring community practice. "
        "The capoeira roda was recognized by UNESCO in 2014.",
        chunk_id="chunk-1",
    )
    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)

    result = extract_graph_two_pass(
        [passage],
        _CompoundSubjectAuditLlm(),
        HashEmbeddingProvider(),
        EmbedOptions(model="hash", dimension=8, batch_size=8),
        GraphSettings(gleaning_max_passes=0, entity_resolution_enabled=False),
        ontology.profile,
        ontology.checksum,
        ontology.source,
        model="test-model",
        timeout_seconds=30,
    )

    assert any(entity.name == "capoeira roda" for entity in result.entities)
    assert [
        (item.source_name, item.relation_type, item.target_name) for item in result.relations
    ] == [("capoeira roda", "RECOGNIZED_BY", "UNESCO")]


def test_cue_entity_audit_recovers_a_completely_missing_endpoint() -> None:
    """Revisit cue sentences whose first entity pass cannot form any relation."""

    passage = chunk("Method Alpha was evaluated on HiddenSet.", chunk_id="chunk-1")
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None)

    result = extract_graph_two_pass(
        [passage],
        _IncompleteEndpointAuditLlm(),
        HashEmbeddingProvider(),
        EmbedOptions(model="hash", dimension=8, batch_size=8),
        GraphSettings(gleaning_max_passes=0, entity_resolution_enabled=False),
        ontology.profile,
        ontology.checksum,
        ontology.source,
        model="test-model",
        timeout_seconds=30,
    )

    assert {entity.name for entity in result.entities} == {"Method Alpha", "HiddenSet"}
    assert [
        (item.source_name, item.relation_type, item.target_name) for item in result.relations
    ] == [("Method Alpha", "EVALUATED_ON", "HiddenSet")]


def test_reference_filter_skips_bibliographies_but_keeps_cited_prose() -> None:
    """Require several reference entries before suppressing provider extraction."""

    bibliography = """References
[19] S. Li et al. Composing simple image descriptions. 2011.
[20] T. Lin et al. Microsoft COCO. 2014.
[21] J. Mao et al. Explain images with neural networks. 2014.
[22] T. Mikolov et al. Word representations. 2013.
"""
    numbered_bibliography = """
3. Konorski, J. Integrative Activity of the Brain. 1967.
4. Logothetis, N. K. Visual object recognition. 1996.
5. Riesenhuber, M. Neural mechanisms of recognition. 2002.
6. Young, M. P. Sparse population coding. 1992.
"""
    author_year_bibliography = """
Muller-Budack, E., Pustu-Iren, K., and Ewerth, R. Geolocation estimation.
In Proceedings of ECCV, 2018.
Murty, S., Koh, P. W., and Liang, P. Expbert. arXiv preprint arXiv:2005.01932, 2020.
Narasimhan, K., Kulkarni, T., and Barzilay, R. Language understanding. arXiv:1506.08941, 2015.
Netzer, Y., Wang, T., Coates, A., and Ng, A. Reading digits. In a conference workshop, 2011.
"""
    cited_prose = "Our method follows prior work [19] but introduces a new objective."
    historical_prose = """
A. Smith described the apparatus used in our study.
B. Jones repeated the experiment with a larger sample.
C. García calibrated the instrument before each trial.
D. Chen independently reviewed every measurement.
The experiments ran from 2018 through 2021.
"""

    assert is_reference_only_window(_window(bibliography))
    assert is_reference_only_window(_window(numbered_bibliography))
    assert is_reference_only_window(_window(author_year_bibliography))
    assert not is_reference_only_window(_window(cited_prose))
    assert not is_reference_only_window(_window(historical_prose))


def test_a_window_of_derivations_is_skipped_but_prose_with_formulas_is_not() -> None:
    """A page of LaTeX holds nothing to ground a quote in.

    A formula recogniser renders an appendix as equations with a sentence of
    connective prose between them; the model's records for such a window are
    all rejected as ungrounded, and retrying the window would fail the run.
    Prose that merely contains formulas keeps its words and is extracted.
    """

    derivation = (
        "$$\\begin{array}{rcl} \\frac{\\partial s_{c_j}(t)}{\\partial w_{lm}} & = & "
        "\\frac{\\partial s_{c_j}(t-1)}{\\partial w_{lm}} + \\frac{\\partial y^{in_j}(t)}"
        "{\\partial w_{lm}} g(net_{c_j}(t)) \\end{array}\\tag{14}$$\n"
        "where the truncated derivatives that need to be stored are\n"
        "$$\\delta_{in,i} \\frac{\\partial \\eta(t)}{\\partial u_{in}} \\approx_{tr} "
        "\\left( n e t _ { \\sigma _ { i } ^ { * } } ( t ) \\right) \\tag{15}$$"
    )
    prose_with_formulas = (
        "Batch Normalization allows much higher learning rates. Each activation is "
        "normalized as $\\hat{x}_i = \\frac{x_i - \\mu}{\\sigma}$, where $\\mu$ is the "
        "mini-batch mean and $\\sigma^2$ its variance; Ioffe and Szegedy report that "
        "the Inception network then trains fourteen times faster."
    )

    assert window_skip_reason(_window(derivation)) == "display_math_only"
    assert window_skip_reason(_window(prose_with_formulas)) is None
    assert window_skip_reason(_window(bibliography_text())) == "bibliography_only"


def bibliography_text() -> str:
    return """References
[19] S. Li et al. Composing simple image descriptions. 2011.
[20] T. Lin et al. Microsoft COCO. 2014.
[21] J. Mao et al. Explain images with neural networks. 2014.
[22] T. Mikolov et al. Word representations. 2013.
"""


def test_relation_extraction_grounds_document_context_through_explicit_pronoun() -> None:
    """Let independent windows attach an authored claim to their source document."""

    content = "We introduce Adam, an algorithm for stochastic optimization."
    extraction_window = _window(content)
    paper = EntityMention(
        id="paper",
        name="Adam: A Method for Stochastic Optimization",
        type="PAPER",
        description="The source paper.",
        source_chunk_id="front-matter",
        quote="Adam: A Method for Stochastic Optimization",
        is_document_context=True,
        contextual_surfaces=["we", "this paper"],
    )
    method = EntityMention(
        id="method",
        name="Adam",
        type="METHOD",
        description="An optimization method.",
        source_chunk_id="reference-chunk",
        quote="Adam",
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = LlmRelationExtractor(_DocumentSubjectRelationLlm()).extract(
        extraction_window,
        [paper, method],
        ontology,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert len(outcome.relations) == 1
    relation = outcome.relations[0]
    assert relation.source_entity_id == paper.id
    assert relation.target_entity_id == method.id
    assert relation.source_surface == "We"
    assert relation.relation_type == "INTRODUCES"


def test_relation_extraction_uses_implicit_source_paper_provenance() -> None:
    """Ground an authorial paper relation without requiring its title in every sentence."""

    content = "To minimize E by gradient descent, it is necessary to compute the derivative of E."
    extraction_window = _window(content)
    paper = EntityMention(
        id="paper",
        name="Learning representations by back-propagating errors",
        type="PAPER",
        description="The source paper.",
        source_chunk_id="front-matter",
        quote="Learning representations by back-propagating errors",
        is_document_context=True,
        is_document_subject=True,
        contextual_surfaces=["we", "this paper"],
    )
    method = EntityMention(
        id="gradient-descent",
        name="Gradient descent",
        type="METHOD",
        description="An optimization method.",
        source_chunk_id="chunk-1",
        quote="gradient descent",
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = LlmRelationExtractor(_ImplicitDocumentSubjectRelationLlm()).extract(
        extraction_window,
        [paper, method],
        ontology,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert len(outcome.relations) == 1
    assert outcome.relations[0].source_entity_id == paper.id
    assert outcome.relations[0].target_entity_id == method.id
    assert outcome.relations[0].source_surface == paper.name
    assert outcome.trace["record_actions"]["implicit_document_source"] == 1
    assert outcome.trace["record_actions"]["repaired_source_surface"] == 1


def test_relation_extraction_uses_provenance_for_target_nested_in_paper_title() -> None:
    """A title that names the task it addresses grounds that claim itself.

    The task is nested in the title; a name that contains the other endpoint's
    name is evidence of their relation, so the title is the quote.
    """

    title = "Understanding the Difficulty of Training Deep Feedforward Neural Networks"
    extraction_window = _window(title)
    paper = EntityMention(
        id="paper",
        name=title,
        type="PAPER",
        description="The source paper.",
        source_chunk_id="reference-chunk",
        quote=title,
        is_document_context=True,
        is_document_subject=True,
        contextual_surfaces=["we", "this paper"],
    )
    task = EntityMention(
        id="task",
        name="Training Deep Feedforward Neural Networks",
        type="TASK",
        description="The training task named by the title.",
        source_chunk_id="reference-chunk",
        quote="Training Deep Feedforward Neural Networks",
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = LlmRelationExtractor(_NestedTitleRelationLlm()).extract(
        extraction_window,
        [paper, task],
        ontology,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert len(outcome.relations) == 1
    assert outcome.relations[0].source_entity_id == paper.id
    assert outcome.relations[0].target_entity_id == task.id
    assert outcome.relations[0].quote == title


def test_relation_extraction_does_not_infer_implicit_non_paper_source() -> None:
    """Keep method and model endpoints subject to exact local surface grounding."""

    content = "To minimize E by gradient descent, derivatives are computed efficiently."
    extraction_window = _window(content)
    source = EntityMention(
        id="back-propagation",
        name="Back-propagation",
        type="METHOD",
        description="A differentiation method.",
        source_chunk_id="front-matter",
        quote="Back-propagation",
        is_document_context=True,
        contextual_surfaces=["the method"],
    )
    target = EntityMention(
        id="gradient-descent",
        name="Gradient descent",
        type="METHOD",
        description="An optimization method.",
        source_chunk_id="chunk-1",
        quote="gradient descent",
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = LlmRelationExtractor(_ImplicitMethodSubjectRelationLlm()).extract(
        extraction_window,
        [source, target],
        ontology,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert outcome.relations == []
    assert outcome.trace["record_actions"]["ungrounded_quote"] == 1


def test_relation_grounding_recovers_normalized_surface_from_inventory() -> None:
    """Use a canonical identity fallback when a model-normalized local label is absent."""

    content = (
        "The Stochastic Gradient Variational Bayes estimator uses the reparameterization trick."
    )
    extraction_window = _window(content)
    source = EntityMention(
        id="sgvb",
        name="Stochastic Gradient Variational Bayes estimator",
        type="METHOD",
        description="A gradient estimator.",
        source_chunk_id="reference-chunk",
        quote="Stochastic Gradient Variational Bayes estimator",
    )
    target = EntityMention(
        id="reparameterization",
        name="reparameterization trick",
        type="METHOD",
        description="A gradient-estimation method.",
        source_chunk_id="reference-chunk",
        quote="reparameterization trick",
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = LlmRelationExtractor(_NormalizedEndpointSurfaceLlm()).extract(
        extraction_window,
        [source, target],
        ontology,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert len(outcome.relations) == 1
    assert outcome.relations[0].quote == content
    assert outcome.relations[0].source_surface == "Stochastic Gradient Variational Bayes estimator"
    assert outcome.trace["record_actions"]["repaired_source_surface"] == 1


def test_single_document_window_inherits_context_grounded_in_another_chunk() -> None:
    """Preserve front-matter entities when distributed workers receive body-only windows.

    A queue task contains a bounded subset of a document, while its reusable paper
    or model context is normally grounded in an earlier front-matter chunk. The
    context must remain in the observation inventory so body pronouns can reference
    it and finalization cannot silently drop the focal source entity.
    """

    extraction_window = _window("The body describes a generally useful optimization method.")
    paper = EntityMention(
        id="paper",
        name="A General Optimization Method",
        type="PAPER",
        description="The source paper.",
        source_chunk_id="front-matter-outside-this-window",
        quote="A General Optimization Method",
        is_document_context=True,
        contextual_surfaces=["we", "this paper"],
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    observations = extract_graph_observations(
        extraction_window.chunks,
        FakeLlmProvider(),
        graph_settings=GraphSettings(
            extraction_parallelism=1, gleaning_max_passes=0, max_chunks_per_llm_call=2
        ),
        ontology=ontology,
        model="fake",
        timeout_seconds=30,
        document_context_entities=[paper],
    )

    assert any(entity.id == paper.id for entity in observations.entities)


class _CrossWindowRelationLlm(FakeLlmProvider):
    """Omit a repeated endpoint locally so the document inventory is required."""

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Return one entity per window and a relation that needs both windows."""

        if request.task_name == "entity_extraction":
            if "Introduces Method Alpha" in request.user:
                entities = [
                    {
                        **_entity("Method Alpha", "METHOD"),
                        "source_chunk_id": "chunk-alpha",
                        "quote": "Introduces Method Alpha.",
                    }
                ]
            else:
                entities = [
                    {
                        **_entity("Method Beta", "METHOD"),
                        "source_chunk_id": "chunk-beta",
                        "quote": "Method Beta builds on Method Alpha.",
                    }
                ]
            payload: object = {"entities": entities}
        elif request.task_name == "relation_extraction":
            input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
            ids = {item["name"]: item["id"] for item in input_payload["entities"]}
            if "Method Alpha" not in ids or "Method Beta" not in ids:
                payload = {"relations": []}
            else:
                payload = {
                    "relations": [
                        {
                            "source_entity_id": ids["Method Beta"],
                            "target_entity_id": ids["Method Alpha"],
                            "source_surface": "Method Beta",
                            "target_surface": "Method Alpha",
                            "relation_type": "BUILDS_ON",
                            "description": "Method Beta builds on Method Alpha.",
                            "source_chunk_id": "chunk-beta",
                            "quote": "Method Beta builds on Method Alpha.",
                            "confidence": 0.99,
                        }
                    ]
                }
        else:
            return super().complete_structured(request)
        return StructuredCompletionResult(
            payload=payload,
            raw_text=json.dumps(payload),
            provider_metadata={"provider": "cross-window-test"},
        )


def test_relation_phase_uses_entities_discovered_in_other_document_windows() -> None:
    """Connect endpoints across windows without serializing either provider phase."""

    first = chunk("Introduces Method Alpha.", chunk_id="chunk-alpha")
    second = chunk("Method Beta builds on Method Alpha.", chunk_id="chunk-beta", chunk_index=1)
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    observations = extract_graph_observations(
        [first, second],
        _CrossWindowRelationLlm(),
        graph_settings=GraphSettings(
            extraction_parallelism=2,
            gleaning_max_passes=0,
            verify_relations=False,
            max_chunks_per_llm_call=1,
        ),
        ontology=ontology,
        model="fake",
        timeout_seconds=30,
    )

    names = {entity.name for entity in observations.entities}
    assert names == {"Method Alpha", "Method Beta"}
    assert len(observations.relations) == 1
    assert observations.relations[0].relation_type == "BUILDS_ON"


def test_relation_inventory_compacts_duplicates_and_omits_ungrounded_entities() -> None:
    """Keep cross-window aliases while excluding identities unusable in this window."""

    extraction_window = _window("Local Method uses its LM abbreviation.")
    duplicate_first = EntityMention(
        id="local-first",
        name="Local Method",
        type="METHOD",
        aliases=[],
        description="The first source observation.",
        source_chunk_id="outside-first",
        quote="Local Method",
    )
    duplicate_second = duplicate_first.model_copy(
        update={
            "id": "local-second",
            "aliases": ["LM"],
            "source_chunk_id": "outside-second",
        }
    )
    remote = EntityMention(
        id="remote",
        name="Remote Dataset",
        type="DATASET",
        description="An endpoint established in another window.",
        source_chunk_id="outside-remote",
        quote="Remote Dataset",
    )

    inventory = _relation_inventory_for_window(
        [remote, duplicate_first, duplicate_second],
        extraction_window,
    )

    assert [entity.id for entity in inventory] == ["local-first"]
    assert inventory[0].aliases == ["LM"]


def test_relation_completion_inventory_keeps_only_uncovered_cue_typed_endpoints() -> None:
    """Focus the second pass without allowing a cue to invent incompatible pairs."""

    extraction_window = _window("Residual Network is trained on ImageNet. Alice Smith observes.")
    entities = [
        EntityMention(
            id="model",
            name="Residual Network",
            type="MODEL",
            description="A trained model.",
            source_chunk_id="reference-chunk",
            quote="Residual Network",
        ),
        EntityMention(
            id="dataset",
            name="ImageNet",
            type="DATASET",
            description="A training dataset.",
            source_chunk_id="reference-chunk",
            quote="ImageNet",
        ),
        EntityMention(
            id="person",
            name="Alice Smith",
            type="PERSON",
            description="An observer unrelated to the training predicate.",
            source_chunk_id="reference-chunk",
            quote="Alice Smith",
        ),
    ]
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    completion = _relation_completion_inventory(extraction_window, entities, ontology, [])

    assert [entity.id for entity in completion] == ["model", "dataset"]


def test_relation_completion_inventory_keeps_endpoints_for_cueless_relations() -> None:
    """Do not let an unrelated lexical cue suppress layout-defined authorship."""

    extraction_window = _window(
        "Research Paper. Alice Smith. Residual Network is trained on ImageNet."
    )
    entities = [
        EntityMention(
            id="paper",
            name="Research Paper",
            type="PAPER",
            description="The focal publication.",
            source_chunk_id="reference-chunk",
            quote="Research Paper",
            is_document_context=True,
        ),
        EntityMention(
            id="author",
            name="Alice Smith",
            type="PERSON",
            description="A front-matter author.",
            source_chunk_id="reference-chunk",
            quote="Alice Smith",
        ),
        EntityMention(
            id="model",
            name="Residual Network",
            type="MODEL",
            description="A trained model.",
            source_chunk_id="reference-chunk",
            quote="Residual Network",
        ),
        EntityMention(
            id="dataset",
            name="ImageNet",
            type="DATASET",
            description="A training dataset.",
            source_chunk_id="reference-chunk",
            quote="ImageNet",
        ),
    ]
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    completion = _relation_completion_inventory(extraction_window, entities, ontology, [])

    assert [entity.id for entity in completion] == ["paper", "author", "model", "dataset"]


def _window(content: str, chunk_id: str = "reference-chunk") -> ExtractionWindow:
    """Build one deterministic extraction window for reference-filter tests."""

    return window(chunk(content, chunk_id=chunk_id))


def _entity(name: str, entity_type: str) -> dict[str, object]:
    """Build one strict entity candidate used by the mixed-record provider fixture.

    Confidence and source fields remain valid by default.
    """

    return {
        "name": name,
        "type": entity_type,
        "description": f"{name} appears in the source.",
        "source_chunk_id": "chunk-1",
        "quote": name,
        "confidence": 1.0,
        "aliases": [],
    }


def _relation(
    source_entity_id: str,
    target_entity_id: str,
    source_surface: str,
    target_surface: str,
    relation_type: str,
    quote: str,
) -> dict[str, object]:
    """Build one strict relation candidate for the coverage-audit fixture."""

    return {
        "source_entity_id": source_entity_id,
        "target_entity_id": target_entity_id,
        "source_surface": source_surface,
        "target_surface": target_surface,
        "relation_type": relation_type,
        "description": quote,
        "source_chunk_id": "chunk-1",
        "quote": quote,
        "confidence": 1.0,
    }


def test_every_window_failing_reports_the_first_cause_not_only_a_count() -> None:
    """Carry the reason in the message, because the chain does not survive.

    A distributed worker records the message and drops the exception chain, so a
    bare "all extraction windows failed" reaches the operator with nothing to act
    on — the provider may have answered perfectly and the pipeline discarded
    every record, and which of those happened is the whole question.
    """

    message = _all_windows_failed_message(
        [window(chunk("a")), window(chunk("b"))],
        [ValueError("no records survived grounding")],
    )

    assert "all 2 extraction windows failed" in message
    assert "ValueError" in message
    assert "no records survived grounding" in message


def test_a_window_failure_detail_cannot_carry_a_document_into_the_task_record() -> None:
    """Bound the detail so a provider echoing its prompt cannot leak the corpus."""

    message = _all_windows_failed_message([window(chunk("a"))], [ValueError("x" * 5000)])

    assert len(message) < _WINDOW_FAILURE_DETAIL_LIMIT + 200
    assert message.endswith("...")


def test_a_transport_failure_names_the_endpoint_that_refused() -> None:
    """Say which host refused, because that is the whole question.

    "Connection refused" alone cannot distinguish a gateway that is rolling from
    a consumer still addressing an engine it is no longer allowed to reach.
    """

    request = httpx.Request("POST", "http://gateway:4000/v1/chat/completions")
    failure = httpx.ConnectError("[Errno 111] Connection refused", request=request)

    message = _all_windows_failed_message([window(chunk("a"))], [failure])

    assert "while calling http://gateway:4000/v1/chat/completions" in message
    assert "ConnectError" in message


def test_relation_prompt_sends_entity_identity_not_the_window_text_again() -> None:
    """The relation pass already has the text; repeating it per entity is a fifth of the prompt.

    Each accepted mention carries the quote it was grounded on and a
    description, and both appear verbatim in ``chunks``. Sending them again
    costs prefill for nothing: leaving them out measures 1.36x the batch
    throughput on the fleet. Identity and the surfaces needed to recognise an
    entity in the text stay.
    """

    text = "Judo influenced Brazilian Jiu-Jitsu."
    source = window(chunk(text, chunk_id="c1"))
    ontology = load_ontology(None, ["MARTIAL_ART"], ["INFLUENCED"]).profile
    mention = EntityMention(
        id="m1",
        name="Judo",
        type="MARTIAL_ART",
        description="A Japanese martial art.",
        source_chunk_id="c1",
        quote=text,
        aliases=["judo"],
    )

    request = relation_extraction_request(
        source,
        [mention],
        ontology,
        model="m",
        timeout_seconds=1,
        max_relations=10,
        seed=17,
        previous_relations=None,
    )
    payload = json.loads(request.user[request.user.index("{") :])

    assert payload["entities"] == [
        {
            "id": "m1",
            "name": "Judo",
            "type": "MARTIAL_ART",
            "source_chunk_id": "c1",
            "aliases": ["judo"],
        }
    ]
    # The text itself is still there, once.
    assert text in json.dumps(payload["chunks"])


def test_entity_and_relation_windows_may_answer_at_the_same_length() -> None:
    """A window dense enough to name many entities is not cut short at 8K tokens."""

    ontology = load_ontology(None, ["MARTIAL_ART"], ["INFLUENCED"]).profile
    source = window(chunk("Judo influenced Brazilian Jiu-Jitsu.", chunk_id="c1"))
    entities = entity_extraction_request(
        source,
        ontology,
        model="m",
        timeout_seconds=1,
        max_entities=10,
        seed=17,
        previous_entities=None,
    )
    relations = relation_extraction_request(
        source,
        [],
        ontology,
        model="m",
        timeout_seconds=1,
        max_relations=10,
        seed=17,
        previous_relations=None,
    )

    assert entities.max_tokens == relations.max_tokens == 16384


def test_extraction_prompts_put_the_unchanging_parts_first() -> None:
    """A prefix cache covers a prompt only to its first differing byte.

    The ontology is identical for every window of a run and the window is not,
    so the ontology has to be serialized first or it can never be cached. This
    pins the order rather than the bytes.
    """

    ontology = load_ontology(None, ["MARTIAL_ART"], ["INFLUENCED"]).profile
    first = window(chunk("Judo influenced Brazilian Jiu-Jitsu.", chunk_id="c1"))
    second = window(chunk("Sambo also influenced it.", chunk_id="c2"), window_id="window_2")

    for source in (first, second):
        entity_request = entity_extraction_request(
            source,
            ontology,
            model="m",
            timeout_seconds=1,
            max_entities=10,
            seed=17,
            previous_entities=None,
        )
        keys = list(json.loads(entity_request.user[entity_request.user.index("{") :]))
        assert keys.index("entity_type_definitions") < keys.index("chunks")
        assert keys.index("entity_types") < keys.index("chunks")

    # And the shared head really is byte-identical across windows.
    def head(request: StructuredCompletionRequest) -> str:
        return request.user[: request.user.index('"document_id"')]

    a = entity_extraction_request(
        first,
        ontology,
        model="m",
        timeout_seconds=1,
        max_entities=10,
        seed=17,
        previous_entities=None,
    )
    b = entity_extraction_request(
        second,
        ontology,
        model="m",
        timeout_seconds=1,
        max_entities=10,
        seed=17,
        previous_entities=None,
    )
    assert head(a) == head(b)
    assert len(head(a)) > 200


class _ScriptedRelationExtractor:
    """Answer relation calls from a script and record each call's seed and history."""

    def __init__(self, answers: list[tuple[int, int]]) -> None:
        self.answers = answers
        self.calls: list[dict[str, object]] = []

    def extract(self, window, entities, ontology, **kwargs):  # type: ignore[no-untyped-def]
        proposed, accepted = self.answers[len(self.calls)]
        self.calls.append(kwargs)
        offset = sum(answer[1] for answer in self.answers[: len(self.calls) - 1])
        relations = [
            RelationObservation(
                id=f"relation-{offset + index}",
                source_entity_id="a",
                target_entity_id=f"b{offset + index}",
                relation_type="RELATED_TO",
                description="",
                source_chunk_id="reference-chunk",
                quote="",
            )
            for index in range(accepted)
        ]
        return RelationExtractionOutcome(
            relations=relations,
            trace={"stage": "relation_extraction", "input_records": proposed},
        )


def _mentions(count: int) -> list[EntityMention]:
    return [
        EntityMention(
            id=f"e{index}",
            name=f"Entity {index}",
            type="CONCEPT",
            description="",
            source_chunk_id="reference-chunk",
            quote=f"Entity {index}",
        )
        for index in range(count)
    ]


def test_an_empty_answer_over_a_large_inventory_is_retried_once_with_another_seed() -> None:
    """One cheap retry separates a sampling accident from a window that states nothing."""

    extractor = _ScriptedRelationExtractor([(0, 0), (3, 3)])
    trace: list[dict[str, object]] = []
    settings = GraphSettings(
        relation_empty_retry_min_entities=8, relation_continuation_max_passes=0
    )

    relations = _primary_relations(  # noqa: SLF001
        extractor,
        _window("text"),
        _mentions(8),
        load_ontology(None, [], None).profile,
        settings,
        model="fake",
        timeout_seconds=30,
        trace=trace,
        added=[],
        held=[],
    )

    assert len(relations) == 3
    assert extractor.calls[1]["seed"] == settings.deterministic_seed + 1
    assert trace[-1]["empty_retry"] is True
    small = _ScriptedRelationExtractor([(0, 0)])
    _primary_relations(  # noqa: SLF001
        small,
        _window("text"),
        _mentions(7),
        load_ontology(None, [], None).profile,
        settings,
        model="fake",
        timeout_seconds=30,
        trace=[],
        added=[],
        held=[],
    )
    assert len(small.calls) == 1


def test_a_call_that_fills_its_limit_is_continued_until_it_does_not() -> None:
    """A full answer was cut off; ask for the rest with what is already known."""

    extractor = _ScriptedRelationExtractor([(4, 4), (4, 4), (2, 2), (9, 9)])
    trace: list[dict[str, object]] = []

    relations = _primary_relations(  # noqa: SLF001
        extractor,
        _window("text"),
        _mentions(3),
        load_ontology(None, [], None).profile,
        GraphSettings(max_relations_per_batch=4, relation_continuation_max_passes=5),
        model="fake",
        timeout_seconds=30,
        trace=trace,
        added=[],
        held=[],
    )

    assert len(relations) == 10
    assert len(extractor.calls) == 3
    previous_counts = []
    for call in extractor.calls[1:]:
        previous = call["previous_relations"]
        assert isinstance(previous, list)
        previous_counts.append(len(previous))
    assert previous_counts == [4, 8]
    assert [event.get("continuation_pass") for event in trace] == [None, 1, 2]


class _PointingGroundingLlm(FakeLlmProvider):
    """Propose a relation with a paraphrased quote, then point at the lines stating it."""

    def __init__(self, point_at: list[int]) -> None:
        super().__init__()
        self.point_at = point_at
        self.grounding_units: list[dict[str, object]] = []

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Answer the relation call and the grounding call; nothing else is asked."""

        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        if request.task_name == "relation_grounding":
            self.grounding_units = payload["units"]
            decision = {"id": "r0", "supported": bool(self.point_at), "lines": self.point_at}
            return StructuredCompletionResult(payload={"decisions": [decision]})
        ids = {item["name"]: item["id"] for item in payload["entities"]}
        relation = _relation(
            ids["nonionic formulation"],
            ids["Span 60"],
            "nonionic formulation",
            "Span 60",
            "RELATED_TO",
            "The nonionic formulation contained Span 60",
        ) | {"source_chunk_id": "reference-chunk"}
        return StructuredCompletionResult(payload={"relations": [relation]})


def _far_apart_endpoints() -> tuple[ExtractionWindow, list[EntityMention]]:
    content = (
        "The nonionic formulation was prepared as follows.\n"
        "Water was heated.\nThe lipid was melted.\nBoth phases were mixed.\n"
        "The mixture was homogenised.\nIt was cooled.\n"
        "Span 60 was the surfactant used."
    )
    mentions = [
        EntityMention(
            id=name,
            name=name,
            type="CONCEPT",
            description="",
            source_chunk_id="reference-chunk",
            quote=name,
        )
        for name in ("nonionic formulation", "Span 60")
    ]
    return _window(content), mentions


def test_llm_grounding_locates_a_relation_string_grounding_missed() -> None:
    """The model points at lines; code confirms the endpoints and keeps exact text."""

    extraction_window, mentions = _far_apart_endpoints()
    llm = _PointingGroundingLlm(point_at=[1, 7])

    outcome = LlmRelationExtractor(llm, grounding_fallback=True).extract(
        extraction_window,
        mentions,
        load_ontology(None, [], None).profile,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    [relation] = outcome.relations
    content = extraction_window.chunks[0].content
    assert relation.evidence_method == "llm_located"
    assert relation.quote == content[relation.start_offset : relation.end_offset]
    assert relation.quote.startswith("The nonionic formulation")
    assert relation.quote.endswith("Span 60 was the surfactant used.")
    assert outcome.trace["record_actions"]["llm_grounding_located"] == 1
    assert [unit["n"] for unit in llm.grounding_units] == [1, 2, 3, 4, 5, 6, 7]
    # A relation the fallback located is in the graph, not among the rejected.
    assert outcome.trace["rejected_records"] == []


def test_llm_grounding_keeps_nothing_the_chosen_lines_do_not_name() -> None:
    """Lines that miss an endpoint, or no lines at all, locate nothing."""

    extraction_window, mentions = _far_apart_endpoints()
    # Lines that name neither endpoint still reach, a line for each, the lines
    # naming both when the stretch stays within bounds.
    reached = LlmRelationExtractor(_PointingGroundingLlm([2, 3]), grounding_fallback=True).extract(
        extraction_window,
        mentions,
        load_ontology(None, [], None).profile,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )
    [relation] = reached.relations
    assert relation.quote.startswith("The nonionic formulation")
    assert reached.trace["record_actions"]["llm_grounding_extended"] == 1
    point_at: list[int]
    for point_at, reason in (([], "llm_grounding_unsupported"),):
        outcome = LlmRelationExtractor(
            _PointingGroundingLlm(point_at), grounding_fallback=True
        ).extract(
            extraction_window,
            mentions,
            load_ontology(None, [], None).profile,
            model="fake",
            timeout_seconds=30,
            max_relations=10,
        )
        assert outcome.relations == []
        assert outcome.trace["record_actions"][reason] == 1
        [rejected] = outcome.trace["rejected_records"]
        assert (rejected["reason"], rejected["source"], rejected["target"]) == (
            reason,
            "nonionic formulation",
            "Span 60",
        )
    off = LlmRelationExtractor(_PointingGroundingLlm([1, 7])).extract(
        extraction_window,
        mentions,
        load_ontology(None, [], None).profile,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )
    assert off.relations == []
    assert [record["reason"] for record in off.trace["rejected_records"]] == ["ungrounded_quote"]


def _quantity_profile() -> OntologyProfile:
    return OntologyProfile.model_validate(
        {
            "name": "values",
            "description": "Materials and their attributes",
            "entity_types": [
                {"name": "MATERIAL", "description": "A material"},
                {"name": "ATTRIBUTE", "description": "A quality attribute"},
                {"name": "BATCH", "description": "A batch", "identifier": True},
            ],
            "relation_types": [
                {
                    "name": "HAS_ATTRIBUTE",
                    "description": "A material has a tested attribute",
                    "source_types": ["MATERIAL"],
                    "target_types": ["ATTRIBUTE"],
                    "quantities": True,
                }
            ],
        }
    )


class _QuantityLlm(FakeLlmProvider):
    """Return one attribute relation carrying a limit and a result, one of them invented."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Answer the relation call with stated and unstated values."""

        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in payload["entities"]}
        relation = _relation(
            ids["Lactose"],
            ids["Water"],
            "Lactose",
            "Water",
            "HAS_ATTRIBUTE",
            "Lactose: Water ≤ 0.5 % Result 0.1 %",
        ) | {
            "source_chunk_id": "reference-chunk",
            "quantities": [
                {"text": "≤ 0.5 %", "kind": "limit", "unit": "%", "comparator": "≤"},
                {"text": "0.1 %", "kind": "result", "unit": "%", "comparator": ""},
                {"text": "2.0 %", "kind": "result", "unit": "%", "comparator": ""},
            ],
        }
        return StructuredCompletionResult(payload={"relations": [relation]})


def test_a_relation_keeps_the_values_its_evidence_states() -> None:
    """Each kept value occurs in the evidence; the invented one is dropped."""

    mentions = [
        EntityMention(
            id=name,
            name=name,
            type=kind,
            description="",
            source_chunk_id="reference-chunk",
            quote=name,
        )
        for name, kind in (("Lactose", "MATERIAL"), ("Water", "ATTRIBUTE"))
    ]
    outcome = LlmRelationExtractor(_QuantityLlm()).extract(
        _window("Lactose: Water ≤ 0.5 % Result 0.1 %"),
        mentions,
        _quantity_profile(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    [relation] = outcome.relations
    assert [(item.text, item.kind, item.high) for item in relation.quantities] == [
        ("≤ 0.5 %", "limit", 0.5),
        ("0.1 %", "result", 0.1),
    ]
    assert outcome.trace["record_actions"]["ungrounded_quantity"] == 1


def _claim_profile() -> OntologyProfile:
    return OntologyProfile.model_validate(
        {
            "name": "claims",
            "description": "Coatings and the surfaces they protect",
            "entity_types": [
                {"name": "COATING", "description": "A coating"},
                {"name": "SURFACE", "description": "A surface"},
            ],
            "relation_types": [
                {
                    "name": "PROTECTS",
                    "description": "A coating protects a surface",
                    "source_types": ["COATING"],
                    "target_types": ["SURFACE"],
                }
            ],
        }
    )


class _ClaimLlm(FakeLlmProvider):
    """State one relation as a finding and another only as a possible use."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in payload["entities"]}
        found = _relation(
            ids["Glaze-7"],
            ids["hull steel"],
            "Glaze-7",
            "hull steel",
            "PROTECTS",
            "Glaze-7 protected hull steel for a year",
        ) | {"source_chunk_id": "reference-chunk"}
        claimed = _relation(
            ids["Glaze-7"],
            ids["deck plates"],
            "Glaze-7",
            "deck plates",
            "PROTECTS",
            "Glaze-7 may be used to protect deck plates",
        ) | {"source_chunk_id": "reference-chunk", "assertion": "claimed"}
        return StructuredCompletionResult(payload={"relations": [found, claimed]})


def test_a_relation_the_text_only_claims_is_kept_and_marked_claimed() -> None:
    """A stated use is a relation the text states, kept apart from a reported finding."""

    text = "Glaze-7 protected hull steel for a year. Glaze-7 may be used to protect deck plates."
    mentions = [
        EntityMention(
            id=name,
            name=name,
            type=kind,
            description="",
            source_chunk_id="reference-chunk",
            quote=name,
        )
        for name, kind in (
            ("Glaze-7", "COATING"),
            ("hull steel", "SURFACE"),
            ("deck plates", "SURFACE"),
        )
    ]
    outcome = LlmRelationExtractor(_ClaimLlm()).extract(
        _window(text),
        mentions,
        _claim_profile(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert {item.target_entity_id: item.assertion for item in outcome.relations} == {
        "hull steel": "reported",
        "deck plates": "claimed",
    }


def test_the_relation_contract_asks_for_every_relation_assertion() -> None:
    request = relation_extraction_request(
        _window("Glaze-7 may be used to protect deck plates."),
        [],
        _claim_profile(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
        seed=7,
        previous_relations=None,
    )

    item = request.json_schema["properties"]["relations"]["items"]
    if "$ref" in item:
        item = request.json_schema["$defs"][item["$ref"].rsplit("/", 1)[1]]
    assert item["properties"]["assertion"]["enum"] == ["reported", "claimed"]
    assert "assertion" in item["required"]
    assert "claimed" in request.system


def test_the_verifier_sees_a_claim_as_a_claim_and_may_support_it() -> None:
    """A claimed relation is judged on whether the text makes the claim, not on proof."""

    text = "Glaze-7 may be used to protect deck plates."
    claim = RelationObservation(
        id="r-claim",
        source_entity_id="glaze",
        target_entity_id="deck",
        relation_type="PROTECTS",
        description="Glaze-7 may protect deck plates.",
        source_chunk_id="reference-chunk",
        quote=text,
        assertion="claimed",
    )

    request = relation_verification_request(
        _window(text), [], [claim], _claim_profile(), model="fake", timeout_seconds=30, seed=7
    )

    payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
    assert [item["assertion"] for item in payload["relations"]] == ["claimed"]
    assert "a claim is never insufficient for being unproven" in " ".join(request.system.split())


def test_values_become_entities_only_under_identifier_types_once_relations_carry_them() -> None:
    """A limit is not an entity where relations carry values; a batch number still is."""

    class _ValueEntityLlm(FakeLlmProvider):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            entities = [
                _entity("≤ 0.5 %", "ATTRIBUTE") | {"source_chunk_id": "reference-chunk"},
                _entity("0004417720", "BATCH") | {"source_chunk_id": "reference-chunk"},
            ]
            return StructuredCompletionResult(payload={"entities": entities})

    outcome = LlmEntityExtractor(_ValueEntityLlm()).extract(
        _window("Batch 0004417720 Water ≤ 0.5 %"),
        _quantity_profile(),
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )

    assert [entity.name for entity in outcome.entities] == ["0004417720"]
    assert outcome.trace["record_actions"]["value_as_entity"] == 1
    assert [
        (record["kind"], record["reason"], record["name"], record["type"])
        for record in outcome.trace["rejected_records"]
    ] == [("entity", "value_as_entity", "≤ 0.5 %", "ATTRIBUTE")]


def test_a_trailing_abbreviation_becomes_an_alias_of_the_name() -> None:
    """'Sodium lauryl sulphate (SLS)' is the name with an alias; a mark or grade is not."""

    class _AbbreviatingLlm(FakeLlmProvider):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            entities = [
                _entity("Sodium lauryl sulphate (SLS)", "MATERIAL")
                | {"source_chunk_id": "reference-chunk"},
                _entity("Pemulen (TM)", "MATERIAL") | {"source_chunk_id": "reference-chunk"},
                _entity("Polyethylene glycol (PEG 400)", "MATERIAL")
                | {"source_chunk_id": "reference-chunk"},
            ]
            return StructuredCompletionResult(payload={"entities": entities})

    outcome = LlmEntityExtractor(_AbbreviatingLlm()).extract(
        _window(
            "Sodium lauryl sulphate (SLS) 1 %, Pemulen (TM) 0.2 % and "
            "Polyethylene glycol (PEG 400) 5 %"
        ),
        _quantity_profile(),
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )

    assert [(entity.name, entity.aliases) for entity in outcome.entities] == [
        ("Sodium lauryl sulphate", ["SLS"]),
        ("Pemulen (TM)", []),
        ("Polyethylene glycol (PEG 400)", []),
    ]
    assert outcome.trace["record_actions"]["abbreviation_as_alias"] == 1


def test_a_name_in_a_chunk_keeps_the_first_type_it_is_given() -> None:
    """The same name typed twice in one chunk is one thing: the second type is a hedge."""

    class _HedgingLlm(FakeLlmProvider):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            entities = [
                _entity("Carbopol 980", "MATERIAL") | {"source_chunk_id": "reference-chunk"},
                _entity("Carbopol 980", "ATTRIBUTE") | {"source_chunk_id": "reference-chunk"},
            ]
            return StructuredCompletionResult(payload={"entities": entities})

    outcome = LlmEntityExtractor(_HedgingLlm()).extract(
        _window("Carbopol 980 was dispersed in water."),
        _quantity_profile(),
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )

    assert [(entity.name, entity.type) for entity in outcome.entities] == [
        ("Carbopol 980", "MATERIAL")
    ]
    assert outcome.trace["record_actions"]["duplicate"] == 1


def test_every_rejected_relation_is_listed_with_its_reason_but_no_duplicate() -> None:
    """What validation turned away is reported record by record; a repeat is not a loss."""

    mentions = [
        EntityMention(
            id=entity_id,
            name=name,
            type=entity_type,
            description="",
            source_chunk_id="reference-chunk",
            quote=name,
        )
        for entity_id, name, entity_type in (
            ("water", "Water", "MATERIAL"),
            ("ph", "pH", "ATTRIBUTE"),
            ("lot", "B-17", "BATCH"),
        )
    ]
    kept = _relation("water", "ph", "Water", "pH", "HAS_ATTRIBUTE", "Water pH 7")

    class _MixedRelationLlm(FakeLlmProvider):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            relations = [
                kept,
                kept,
                _relation("lot", "ph", "B-17", "pH", "HAS_ATTRIBUTE", "B-17 pH 7"),
                _relation("water", "water", "Water", "Water", "HAS_ATTRIBUTE", "Water pH 7"),
            ]
            return StructuredCompletionResult(
                payload={
                    "relations": [
                        relation | {"source_chunk_id": "reference-chunk"} for relation in relations
                    ]
                }
            )

    outcome = LlmRelationExtractor(_MixedRelationLlm()).extract(
        _window("Water pH 7. B-17 pH 7."),
        mentions,
        _quantity_profile(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert len(outcome.relations) == 1
    assert outcome.trace["record_actions"]["duplicate"] == 1
    assert [
        (
            record["reason"],
            record["source"],
            record["source_type"],
            record["name"],
            record["target"],
        )
        for record in outcome.trace["rejected_records"]
    ] == [
        ("domain_or_range_violation", "B-17", "BATCH", "HAS_ATTRIBUTE", "pH"),
        ("self_loop", "Water", "MATERIAL", "HAS_ATTRIBUTE", "Water"),
    ]


def test_a_relation_request_names_what_each_entity_type_may_start_and_end() -> None:
    """The type rules, read from the two entities the model wants to relate."""

    mentions = [
        EntityMention(
            id=entity_id,
            name=name,
            type=entity_type,
            description="",
            source_chunk_id="reference-chunk",
            quote=name,
        )
        for entity_id, name, entity_type in (
            ("water", "Water", "MATERIAL"),
            ("ph", "pH", "ATTRIBUTE"),
        )
    ]

    request = relation_extraction_request(
        _window("Water pH 7."),
        mentions,
        _quantity_profile(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
        seed=1,
        previous_relations=None,
    )

    payload = json.loads(request.user[request.user.index("{") :])
    assert payload["relations_by_entity_type"] == {
        "ATTRIBUTE": {"may_start": [], "may_end": ["HAS_ATTRIBUTE"]},
        "MATERIAL": {"may_start": ["HAS_ATTRIBUTE"], "may_end": []},
    }


def test_an_ontology_fallback_relation_keeps_a_statement_its_own_rules_reject() -> None:
    """The edge takes the fallback type; the type the model stated stays on the observation."""

    profile = _quantity_profile().model_copy(
        update={
            "relation_types": [
                *_quantity_profile().relation_types,
                RelationTypeDefinition(name="RELATED_TO", description="Any stated connection."),
            ],
            "fallback_relation": "RELATED_TO",
        }
    )
    mentions = [
        EntityMention(
            id=entity_id,
            name=name,
            type=entity_type,
            description="",
            source_chunk_id="reference-chunk",
            quote=name,
        )
        for entity_id, name, entity_type in (("lot", "B-17", "BATCH"), ("ph", "pH", "ATTRIBUTE"))
    ]

    class _BatchAttributeLlm(FakeLlmProvider):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            relation = _relation("lot", "ph", "B-17", "pH", "HAS_ATTRIBUTE", "B-17 pH 7")
            return StructuredCompletionResult(
                payload={"relations": [relation | {"source_chunk_id": "reference-chunk"}]}
            )

    outcome = LlmRelationExtractor(_BatchAttributeLlm()).extract(
        _window("B-17 pH 7."),
        mentions,
        profile,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    [relation] = outcome.relations
    assert (relation.relation_type, relation.proposed_relation_type) == (
        "RELATED_TO",
        "HAS_ATTRIBUTE",
    )
    assert outcome.trace["record_actions"]["kept_under_fallback_relation"] == 1
    assert outcome.trace["rejected_records"] == []


def test_a_fallback_relation_must_be_one_of_the_ontology_relations() -> None:
    with pytest.raises(ValueError, match="fallback relation MISSING is not defined"):
        _quantity_profile().model_validate(
            {**_quantity_profile().model_dump(), "fallback_relation": "MISSING"}
        )


def _rewrite_profile() -> OntologyProfile:
    """Batches may state attributes only through BATCH_ATTRIBUTE; a rewrite sends them there."""

    base = _quantity_profile()
    return OntologyProfile.model_validate(
        {
            **base.model_dump(),
            "relation_types": [
                *base.model_dump()["relation_types"],
                {
                    "name": "BATCH_ATTRIBUTE",
                    "description": "A batch was measured for an attribute",
                    "source_types": ["BATCH"],
                    "target_types": ["ATTRIBUTE"],
                },
            ],
            "relation_rewrites": [
                {"relation": "HAS_ATTRIBUTE", "source_types": ["BATCH"], "to": "BATCH_ATTRIBUTE"}
            ],
        }
    )


def test_an_ontology_rewrite_keeps_a_statement_under_the_relation_its_types_allow() -> None:
    """The edge takes the rewritten type; the type the model stated stays on the observation."""

    mentions = [
        EntityMention(
            id=entity_id,
            name=name,
            type=entity_type,
            description="",
            source_chunk_id="reference-chunk",
            quote=name,
        )
        for entity_id, name, entity_type in (
            ("lot", "B-17", "BATCH"),
            ("ph", "pH", "ATTRIBUTE"),
            ("water", "Water", "ATTRIBUTE"),
        )
    ]

    class _AttributeLlm(FakeLlmProvider):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            relations = [
                _relation("lot", "ph", "B-17", "pH", "HAS_ATTRIBUTE", "B-17 pH 7"),
                # No rewrite covers an attribute as a source: still rejected.
                _relation("water", "ph", "Water", "pH", "HAS_ATTRIBUTE", "Water pH 7"),
            ]
            return StructuredCompletionResult(
                payload={
                    "relations": [
                        item | {"source_chunk_id": "reference-chunk"} for item in relations
                    ]
                }
            )

    outcome = LlmRelationExtractor(_AttributeLlm()).extract(
        _window("B-17 pH 7. Water pH 7."),
        mentions,
        _rewrite_profile(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    [relation] = outcome.relations
    assert (relation.relation_type, relation.proposed_relation_type) == (
        "BATCH_ATTRIBUTE",
        "HAS_ATTRIBUTE",
    )
    assert outcome.trace["record_actions"]["rewritten_relation"] == 1
    assert [record["reason"] for record in outcome.trace["rejected_records"]] == [
        "domain_or_range_violation"
    ]


@pytest.mark.parametrize(
    ("rewrite", "message"),
    [
        ({"relation": "MISSING", "to": "BATCH_ATTRIBUTE"}, "names an undefined relation"),
        (
            {"relation": "HAS_ATTRIBUTE", "source_types": ["MATERIAL"], "to": "BATCH_ATTRIBUTE"},
            "does not allow: MATERIAL",
        ),
        ({"relation": "HAS_ATTRIBUTE", "source_types": ["LOT"], "to": "BATCH_ATTRIBUTE"}, "LOT"),
    ],
)
def test_an_ontology_rewrite_must_join_defined_relations_that_admit_it(
    rewrite: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        OntologyProfile.model_validate(
            {**_rewrite_profile().model_dump(), "relation_rewrites": [rewrite]}
        )


def test_an_identifier_is_an_alias_of_what_it_identifies_never_an_entity() -> None:
    """A registry number named as an entity is refused; given as an alias, it is kept."""

    profile = OntologyProfile.model_validate(
        {
            **_quantity_profile().model_dump(),
            "identifier_patterns": [{"name": "CAS number", "pattern": r"\d{2,7}-\d{2}-\d"}],
        }
    )

    class _RegistryLlm(FakeLlmProvider):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            entities = [
                _entity("1111-22-3", "MATERIAL"),
                _entity("Northwindol", "MATERIAL") | {"aliases": ["1111-22-3"]},
            ]
            return StructuredCompletionResult(
                payload={
                    "entities": [item | {"source_chunk_id": "reference-chunk"} for item in entities]
                }
            )

    outcome = LlmEntityExtractor(_RegistryLlm()).extract(
        _window("Northwindol (1111-22-3) was tested for 100 weeks."),
        profile,
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )

    [entity] = outcome.entities
    assert (entity.name, entity.aliases) == ("Northwindol", ["1111-22-3"])
    assert [(record["reason"], record["name"]) for record in outcome.trace["rejected_records"]] == [
        ("identifier_as_entity", "1111-22-3")
    ]


def test_a_relation_may_add_a_grounded_endpoint_the_entities_lack() -> None:
    """A named thing the entity pass missed becomes an entity when a relation needs it."""

    water = EntityMention(
        id="water",
        name="Water",
        type="MATERIAL",
        description="",
        source_chunk_id="reference-chunk",
        quote="Water",
    )

    class _EndpointLlm(FakeLlmProvider):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            endpoint = {"source_chunk_id": "reference-chunk"}
            return StructuredCompletionResult(
                payload={
                    "relations": [
                        _relation("water", "new-1", "Water", "pH", "HAS_ATTRIBUTE", "Water pH 7")
                        | endpoint
                    ],
                    "new_entities": [
                        {"id": "new-1", "name": "pH", "type": "ATTRIBUTE", "quote": "Water pH 7"}
                        | endpoint,
                        {"id": "new-2", "name": "Water", "type": "MATERIAL", "quote": "Water"}
                        | endpoint,
                        {"id": "new-3", "name": "Nitrogen", "type": "ATTRIBUTE", "quote": "N2"}
                        | endpoint,
                    ],
                }
            )

    off = LlmRelationExtractor(_EndpointLlm()).extract(
        _window("Water pH 7."),
        [water],
        _quantity_profile(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )
    # Off by default: an endpoint list is ignored and its relation has no end.
    assert (off.entities, off.relations) == ([], [])

    outcome = LlmRelationExtractor(_EndpointLlm(), new_endpoints=True).extract(
        _window("Water pH 7."),
        [water, water.model_copy(update={"id": "spare", "name": "Spare", "quote": "Spare"})],
        _quantity_profile(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    [added] = outcome.entities
    assert (added.name, added.type, added.quote) == ("pH", "ATTRIBUTE", "Water pH 7")
    [relation] = outcome.relations
    assert (relation.source_entity_id, relation.target_entity_id) == ("water", added.id)
    actions = outcome.trace["record_actions"]
    assert (actions["added_endpoint"], actions["endpoint_already_listed"]) == (1, 1)
    assert [
        (record["kind"], record["reason"], record["name"])
        for record in outcome.trace["rejected_records"]
    ] == [("entity", "ungrounded_quote", "Nitrogen")]


def test_llm_grounding_may_extend_to_the_line_naming_the_other_endpoint() -> None:
    """Chosen lines that name one end reach, within the chunk, the nearest line naming the other."""

    extraction_window, mentions = _far_apart_endpoints()
    outcome = LlmRelationExtractor(
        _PointingGroundingLlm(point_at=[7]), grounding_fallback=True
    ).extract(
        extraction_window,
        mentions,
        load_ontology(None, [], None).profile,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )
    [relation] = outcome.relations
    assert relation.quote.startswith("The nonionic formulation")
    assert relation.quote.endswith("Span 60 was the surfactant used.")
    assert outcome.trace["record_actions"]["llm_grounding_extended"] == 1


def test_an_entity_whose_name_rephrases_its_verbatim_quote_is_kept() -> None:
    class _RephrasingLlm(FakeLlmProvider):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            entity = _entity("pH measurement", "ATTRIBUTE") | {
                "source_chunk_id": "reference-chunk",
                "quote": "The pH of the syrups was measured",
            }
            return StructuredCompletionResult(payload={"entities": [entity]})

    outcome = LlmEntityExtractor(_RephrasingLlm()).extract(
        _window("The pH of the syrups was measured after one hour."),
        _quantity_profile(),
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )
    [entity] = outcome.entities
    assert (entity.name, entity.quote) == ("pH measurement", "The pH of the syrups was measured")
    assert outcome.trace["record_actions"]["repaired_rephrased_name"] == 1


def test_a_relabel_keeps_a_rule_breaking_statement_under_an_allowed_relation() -> None:
    """The model may only choose among relations the two types allow, or none."""

    profile = _quantity_profile().model_copy(
        update={
            "relation_types": [
                *_quantity_profile().relation_types,
                RelationTypeDefinition(
                    name="BATCH_OF",
                    description="A batch of a material.",
                    source_types=["BATCH"],
                    target_types=["MATERIAL"],
                ),
            ]
        }
    )
    mentions = [
        EntityMention(
            id=entity_id,
            name=name,
            type=entity_type,
            description="",
            source_chunk_id="reference-chunk",
            quote=name,
        )
        for entity_id, name, entity_type in (
            ("lot", "B-17", "BATCH"),
            ("water", "Water", "MATERIAL"),
            ("ph", "pH", "ATTRIBUTE"),
        )
    ]

    class _RelabelLlm(FakeLlmProvider):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            if request.task_name == "relation_relabel":
                statements = json.loads(request.user[request.user.index("{") :])["statements"]
                return StructuredCompletionResult(
                    payload={
                        "decisions": [
                            {"id": item["id"], "option": 0 if item["target"] == "Water" else -1}
                            for item in statements
                        ]
                    }
                )
            relations = [
                _relation("lot", "water", "B-17", "Water", "HAS_ATTRIBUTE", "B-17 Water"),
                _relation("lot", "ph", "B-17", "pH", "HAS_ATTRIBUTE", "B-17 pH 7"),
            ]
            return StructuredCompletionResult(
                payload={
                    "relations": [
                        relation | {"source_chunk_id": "reference-chunk"} for relation in relations
                    ]
                }
            )

    outcome = LlmRelationExtractor(_RelabelLlm(), relabel=True).extract(
        _window("B-17 Water. B-17 pH 7."),
        mentions,
        profile,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    [relation] = outcome.relations
    assert (relation.relation_type, relation.proposed_relation_type) == (
        "BATCH_OF",
        "HAS_ATTRIBUTE",
    )
    assert outcome.trace["record_actions"]["relabeled_relation"] == 1
    assert [
        (record["reason"], record["target"]) for record in outcome.trace["rejected_records"]
    ] == [("domain_or_range_violation", "pH")]


def test_the_entity_fallback_keeps_a_thing_the_lines_state_in_other_words() -> None:
    class _PointingEntityLlm(FakeLlmProvider):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            if request.task_name == "entity_grounding":
                return StructuredCompletionResult(
                    payload={
                        "decisions": [
                            {"id": "e0", "supported": True, "lines": [1]},
                            {"id": "e1", "supported": False, "lines": []},
                        ]
                    }
                )
            entities = [
                _entity("stability", "ATTRIBUTE")
                | {"source_chunk_id": "reference-chunk", "quote": "Studies show stability"},
                _entity("viscosity", "ATTRIBUTE")
                | {"source_chunk_id": "reference-chunk", "quote": "high viscosity"},
            ]
            return StructuredCompletionResult(payload={"entities": entities})

    outcome = LlmEntityExtractor(_PointingEntityLlm(), grounding_fallback=True).extract(
        _window("The emulsion remains stable after 6 months."),
        _quantity_profile(),
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )

    [entity] = outcome.entities
    assert (entity.name, entity.evidence_method) == ("stability", "llm_confirmed")
    assert entity.quote == "The emulsion remains stable after 6 months."
    assert [(record["reason"], record["name"]) for record in outcome.trace["rejected_records"]] == [
        ("llm_grounding_unsupported", "viscosity")
    ]


def test_extraction_requests_carry_the_ontologys_guidance_and_counter_examples() -> None:
    """Guidance reaches every extraction call; identifier patterns only when named."""

    base = _quantity_profile().model_dump()
    profile = OntologyProfile.model_validate(
        {
            **base,
            "entity_types": [
                {
                    **base["entity_types"][0],
                    "document_context": True,
                    "document_subject": True,
                    "guidance": "The material a header names.",
                    "counter_examples": ["a supplier's brand line"],
                },
                *base["entity_types"][1:],
            ],
            "relation_types": [
                {
                    **base["relation_types"][0],
                    "guidance": "A row of the header's material.",
                    "counter_examples": ["a method code in a column"],
                }
            ],
            "identifier_patterns": [{"name": "Item code", "pattern": r"ITM-\d+"}],
        }
    )
    source = _window("Water pH 7.")

    def payload(request: StructuredCompletionRequest) -> dict[str, Any]:
        return dict(json.loads(request.user[request.user.index("{") :]))

    entity = payload(
        entity_extraction_request(
            source,
            profile,
            model="m",
            timeout_seconds=1,
            max_entities=10,
            seed=17,
            previous_entities=None,
        )
    )
    context = payload(
        document_context_extraction_request(
            source, profile, model="m", timeout_seconds=1, max_entities=10, seed=17
        )
    )
    relation = payload(
        relation_extraction_request(
            source,
            [],
            profile,
            model="m",
            timeout_seconds=1,
            max_relations=10,
            seed=17,
            previous_relations=None,
        )
    )

    for definitions in (entity["entity_type_definitions"], context["document_context_types"]):
        material = definitions[0]
        assert material["guidance"] == "The material a header names."
        assert material["counter_examples"] == ["a supplier's brand line"]
    [has_attribute] = relation["relation_type_definitions"]
    assert has_attribute["guidance"] == "A row of the header's material."
    assert has_attribute["counter_examples"] == ["a method code in a column"]
    assert entity["identifier_patterns"] == [{"name": "Item code", "pattern": r"ITM-\d+"}]
    plain = entity_extraction_request(
        source,
        _quantity_profile(),
        model="m",
        timeout_seconds=1,
        max_entities=10,
        seed=17,
        previous_entities=None,
    )
    assert "identifier_patterns" not in payload(plain)


def _location_window() -> ExtractionWindow:
    """A document's location page and its first page."""

    return window(
        chunk(
            location_text("Plant 2/Lab A/Mixer M-4 log.pdf"),
            chunk_id="location",
            page_number=LOCATION_PAGE_NUMBER,
        ),
        chunk("Mixer M-4 log.pdf lists the lactose charged.", chunk_id="page-1"),
    )


def test_a_file_or_folder_path_read_off_the_location_page_is_not_an_entity() -> None:
    """The file is the document itself; a name a folder gives is still kept."""

    class _LocationLlm(FakeLlmProvider):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            on_page = [
                _entity("File: Mixer M-4 log.pdf", "MATERIAL"),
                _entity("Mixer M-4 log.pdf", "MATERIAL"),
                _entity("Plant 2/Lab A", "MATERIAL"),
                _entity("Lab A", "MATERIAL"),
            ]
            in_text = _entity("Mixer M-4 log.pdf", "MATERIAL") | {"source_chunk_id": "page-1"}
            return StructuredCompletionResult(
                payload={
                    "entities": [
                        *(item | {"source_chunk_id": "location"} for item in on_page),
                        in_text,
                    ]
                }
            )

    outcome = LlmEntityExtractor(_LocationLlm()).extract(
        _location_window(),
        _quantity_profile(),
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )

    assert [(entity.name, entity.source_chunk_id) for entity in outcome.entities] == [
        ("Lab A", "location"),
        ("Mixer M-4 log.pdf", "page-1"),
    ]
    assert outcome.trace["record_actions"]["location_page_name"] == 3
    assert [(record["reason"], record["name"]) for record in outcome.trace["rejected_records"]] == [
        ("location_page_name", "File: Mixer M-4 log.pdf"),
        ("location_page_name", "Mixer M-4 log.pdf"),
        ("location_page_name", "Plant 2/Lab A"),
    ]


def test_a_relation_cannot_add_the_documents_own_file_as_an_endpoint() -> None:
    """An endpoint named by the location page's file line is recorded and refused."""

    lactose = EntityMention(
        id="lactose",
        name="lactose",
        type="MATERIAL",
        description="",
        source_chunk_id="page-1",
        quote="lactose",
    )

    class _FileEndpointLlm(FakeLlmProvider):
        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            return StructuredCompletionResult(
                payload={
                    "relations": [],
                    "new_entities": [
                        {
                            "id": "new-1",
                            "name": "Mixer M-4 log.pdf",
                            "type": "ATTRIBUTE",
                            "source_chunk_id": "location",
                            "quote": "File: Mixer M-4 log.pdf",
                        }
                    ],
                }
            )

    outcome = LlmRelationExtractor(_FileEndpointLlm(), new_endpoints=True).extract(
        _location_window(),
        [lactose, lactose.model_copy(update={"id": "spare", "name": "Spare", "quote": "Spare"})],
        _quantity_profile(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert outcome.entities == []
    assert [(record["reason"], record["name"]) for record in outcome.trace["rejected_records"]] == [
        ("location_page_name", "Mixer M-4 log.pdf")
    ]
