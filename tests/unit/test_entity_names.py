from __future__ import annotations

from pathlib import Path

from documents import chunk, window

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.entity_names import (
    identity_surface_match,
    normalize_entity_surface,
    validate_identity_aliases,
)
from kg_processor.application.llm_extractors import LlmEntityExtractor
from kg_processor.application.ontology import load_ontology
from kg_processor.domain.extraction import ExtractionWindow
from kg_processor.domain.ontology import OntologyProfile
from kg_processor.ports.llm import StructuredCompletionRequest, StructuredCompletionResult


class _EntityPayloadLlm(FakeLlmProvider):
    """Return one controlled entity payload through the complete real extractor path.

    This isolates alias validation without bypassing orchestration.
    """

    def __init__(self, entity: dict[str, object]) -> None:
        """Store the exact candidate payload that the entity stage should validate.

        No normalization occurs inside the fixture.
        """

        self.entity = entity

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Intercept entity tasks while delegating unrelated tasks to the fake provider.

        This preserves realistic behavior outside the test focus.
        """

        if request.task_name in {"entity_extraction", "document_context_extraction"}:
            return StructuredCompletionResult(payload={"entities": [self.entity]})
        return super().complete_structured(request)


def test_identity_alias_validation_accepts_initialism_and_rejects_related_art() -> None:
    """Prove genuine initialisms survive while semantically related martial arts are rejected.

    Repeated aliases should not inflate rejection counts.
    """

    result = validate_identity_aliases(
        "Brazilian Jiu-Jitsu",
        ["BJJ", "judo", "BJJ"],
        "BJJ developed by adapting judo in Brazil.",
    )

    assert result.aliases == ["BJJ"]
    assert result.rejected_count == 1
    assert identity_surface_match("Brazilian Jiu-Jitsu", "BJJ")
    assert not identity_surface_match("Brazilian Jiu-Jitsu", "judo")


def test_identity_alias_validation_rejects_stopword_initialisms() -> None:
    result = validate_identity_aliases(
        "Information Systems",
        ["IS"],
        "The pipeline is developed by the compliance team.",
    )

    assert result.aliases == []
    assert not identity_surface_match("Information Systems", "IS")

    three_letter = validate_identity_aliases(
        "Advanced Neural Design",
        ["AND"],
        "The model and the data are evaluated together.",
    )
    assert three_letter.aliases == []
    assert identity_surface_match("Advanced Neural Design", "AND")
    assert not identity_surface_match("Advanced Neural Design", "and")


def test_identity_surface_match_accepts_regular_named_family_plurals_only() -> None:
    """Merge grammatical family plurals without broad stemming or semantic matching."""

    assert identity_surface_match("ResNet", "ResNets")
    assert identity_surface_match("residual network", "residual networks")
    assert not identity_surface_match("residual network", "residual learning")
    assert not identity_surface_match("loss", "los")


def test_entity_surface_normalization_folds_unicode_compatibility_forms() -> None:
    """Treat typographic PDF ligatures as the ordinary letters they represent."""

    assert normalize_entity_surface("uni\ufb01ed architecture") == "unified architecture"
    assert identity_surface_match("uni\ufb01ed architecture", "unified architecture")


def test_identity_surface_match_accepts_one_ocr_split_token() -> None:
    """Accept one first-letter OCR split while rejecting arbitrary spacing changes."""

    assert identity_surface_match("Veda Panneershelvam", "V eda Panneershelvam")
    assert not identity_surface_match("therapist model", "the rapist model")


def test_llm_entity_extractor_cannot_ground_entity_through_untrusted_alias() -> None:
    """Prevent a hallucinated alias from laundering an otherwise ungrounded entity candidate.

    Both rejection reasons must remain auditable.
    """

    extractor = LlmEntityExtractor(
        _EntityPayloadLlm(
            _entity_payload(
                name="Brazilian Jiu-Jitsu",
                quote="karate moved from Okinawa into mainland Japan",
                aliases=["karate"],
            )
        )
    )

    outcome = extractor.extract(
        _window("karate moved from Okinawa into mainland Japan"),
        _ontology(),
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )

    assert outcome.entities == []
    assert outcome.trace["record_actions"] == {
        "ungrounded_quote": 1,
        "untrusted_alias": 1,
    }


def test_a_category_definition_is_not_recorded_as_an_entity_description() -> None:
    """Do not store what the type means as what this entity is.

    The extraction request carries every ontology type definition, so a model can
    answer the description field with the definition it was just given. That text
    is identical for every entity of the type, so it inflates resolution
    similarity between distinct entities, and description merging counts distinct
    descriptions and so can never see enough of it to replace it. Left in place it
    reaches the explorer as a statement about this entity.
    """

    ontology = _ontology()
    definition = next(
        item.description for item in ontology.entity_types if item.name == "MARTIAL_ART"
    )
    payload = _entity_payload(
        name="karate",
        quote="karate moved from Okinawa into mainland Japan",
        aliases=[],
    )
    extractor = LlmEntityExtractor(_EntityPayloadLlm({**payload, "description": definition}))

    outcome = extractor.extract(
        _window("karate moved from Okinawa into mainland Japan"),
        ontology,
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )

    assert [entity.name for entity in outcome.entities] == ["karate"]
    assert outcome.entities[0].description == ""
    # Counted rather than silent, so a corpus running on a weaker model shows it.
    assert outcome.trace["record_actions"]["type_definition_as_description"] == 1


def test_any_category_definition_is_rejected_not_only_this_entitys_own() -> None:
    """Recognize a definition the model reached for from the wrong category.

    Every type definition is supplied in the request, so a model can answer with
    one that does not belong to the entity it is describing, such as a CONCEPT
    described with the LOCATION definition. That is still a category rather than
    the entity, and reading only the entity's own definition lets it through.
    """

    ontology = _ontology()
    other = next(item.description for item in ontology.entity_types if item.name != "MARTIAL_ART")
    payload = _entity_payload(
        name="karate",
        quote="karate moved from Okinawa into mainland Japan",
        aliases=[],
    )
    # Repeated back without its full stop, which is how a definition reappears
    # when the model is not copying it exactly.
    extractor = LlmEntityExtractor(_EntityPayloadLlm({**payload, "description": other.rstrip(".")}))

    outcome = extractor.extract(
        _window("karate moved from Okinawa into mainland Japan"),
        ontology,
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )

    assert outcome.entities[0].description == ""
    assert outcome.trace["record_actions"]["type_definition_as_description"] == 1


def test_an_entitys_own_description_survives_extraction() -> None:
    """Keep a description that says something about this entity and not its type."""

    extractor = LlmEntityExtractor(
        _EntityPayloadLlm(
            _entity_payload(
                name="karate",
                quote="karate moved from Okinawa into mainland Japan",
                aliases=[],
            )
        )
    )

    outcome = extractor.extract(
        _window("karate moved from Okinawa into mainland Japan"),
        _ontology(),
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )

    assert outcome.entities[0].description == "karate is a martial art."
    assert "type_definition_as_description" not in outcome.trace["record_actions"]


def test_llm_entity_extractor_keeps_source_grounded_initialism() -> None:
    """Retain a valid source-grounded initialism while dropping an unrelated proposed alias.

    The canonical long-form name should remain unchanged.
    """

    extractor = LlmEntityExtractor(
        _EntityPayloadLlm(
            _entity_payload(
                name="Brazilian Jiu-Jitsu",
                quote="BJJ",
                aliases=["BJJ", "judo"],
            )
        )
    )

    outcome = extractor.extract(
        _window("BJJ developed by adapting judo in Brazil."),
        _ontology(),
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )

    assert len(outcome.entities) == 1
    assert outcome.entities[0].name == "Brazilian Jiu-Jitsu"
    assert outcome.entities[0].aliases == ["BJJ"]
    assert outcome.trace["record_actions"] == {"untrusted_alias": 1}


def test_ordinary_entity_extraction_rejects_document_context_surface() -> None:
    """Keep a discourse pointer from competing with its focal document entity.

    The relation stage may still use ``this paper`` as a grounded local surface
    for a separately extracted PAPER context entity; only creation of a second
    canonical PAPER node is prohibited here.
    """

    extractor = LlmEntityExtractor(
        _EntityPayloadLlm(
            _entity_payload(
                name="This paper",
                quote="This paper",
                aliases=[],
                entity_type="PAPER",
            )
        )
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = extractor.extract(
        _window("This paper introduces Adam for stochastic optimization."),
        ontology,
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )

    assert outcome.entities == []
    assert outcome.trace["record_actions"] == {"contextual_surface_as_entity": 1}


def test_contextual_surface_filter_is_scoped_to_its_entity_type() -> None:
    """Avoid globally blocking a phrase declared only for another ontology type."""

    extractor = LlmEntityExtractor(
        _EntityPayloadLlm(
            _entity_payload(
                name="Our method",
                quote="Our method",
                aliases=[],
                entity_type="PAPER",
            )
        )
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = extractor.extract(
        _window("Our method is the title of the archived paper."),
        ontology,
        model="fake",
        timeout_seconds=30,
        max_entities=10,
    )

    assert [entity.name for entity in outcome.entities] == ["Our method"]


def test_document_context_extraction_marks_grounded_context_without_identity_aliases() -> None:
    """Keep discourse pronouns separate from canonical aliases and entity resolution."""

    title = "Adam: A Method for Stochastic Optimization"
    ocr_title = "ADAM: A M ETHOD FOR STOCHASTIC OPTIMIZATION"
    extractor = LlmEntityExtractor(
        _EntityPayloadLlm(
            {
                "name": title,
                "type": "PAPER",
                "description": "The source document's title.",
                "source_chunk_id": "chunk-1",
                "quote": ocr_title,
                "confidence": 1.0,
                "aliases": [],
            }
        )
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = extractor.extract_document_context_entities(
        _window(f"{ocr_title}\nDiederik P. Kingma and Jimmy Lei Ba"),
        ontology,
        model="fake",
        timeout_seconds=30,
        max_entities=3,
    )

    assert len(outcome.entities) == 1
    subject = outcome.entities[0]
    assert subject.name == title
    assert subject.is_document_context is True
    assert "this paper" in subject.contextual_surfaces
    assert "we" in subject.contextual_surfaces
    assert subject.aliases == []
    assert outcome.trace["stage"] == "document_context_extraction"


def test_document_context_accepts_name_inside_grounded_front_matter_phrase() -> None:
    """Allow a focal method's exact name inside a longer source-grounded phrase."""

    quote = "introducing a deep residual learning framework"
    extractor = LlmEntityExtractor(
        _EntityPayloadLlm(
            _entity_payload(
                name="deep residual learning",
                quote=quote,
                aliases=["residual learning framework"],
                entity_type="METHOD",
            )
        )
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = extractor.extract_document_context_entities(
        _window(f"We present a framework, {quote}, for image recognition."),
        ontology,
        model="fake",
        timeout_seconds=30,
        max_entities=3,
    )

    assert [entity.name for entity in outcome.entities] == ["deep residual learning"]
    assert outcome.entities[0].aliases == []
    assert outcome.trace["record_actions"] == {"untrusted_alias": 1}


def test_document_context_rejects_related_phrase_without_identity_surface() -> None:
    """Keep topical similarity from promoting a model into document-wide context."""

    extractor = LlmEntityExtractor(
        _EntityPayloadLlm(
            _entity_payload(
                name="residual network",
                quote="Deep Residual Learning for Image Recognition",
                aliases=[],
                entity_type="MODEL",
            )
        )
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = extractor.extract_document_context_entities(
        _window("Deep Residual Learning for Image Recognition"),
        ontology,
        model="fake",
        timeout_seconds=30,
        max_entities=3,
    )

    assert outcome.entities == []
    assert outcome.trace["record_actions"] == {"document_context_name_mismatch": 1}


def _ontology() -> OntologyProfile:
    """Load the public martial-arts ontology used by alias-boundary unit scenarios.

    Tests therefore exercise production relation and entity labels.
    """

    return load_ontology(Path("data/martial_arts/ontology.yaml"), [], None).profile


def _window(content: str) -> ExtractionWindow:
    """Wrap one controlled chunk in the window whose id the payload helpers cite."""

    return window(chunk(content, chunk_id="chunk-1"))


def _entity_payload(
    *,
    name: str,
    quote: str,
    aliases: list[str],
    entity_type: str = "MARTIAL_ART",
) -> dict[str, object]:
    """Build a strict entity candidate payload with configurable evidence and aliases.

    All unrelated fields remain production-shaped constants.
    """

    return {
        "name": name,
        "type": entity_type,
        "description": f"{name} is a martial art.",
        "source_chunk_id": "chunk-1",
        "quote": quote,
        "confidence": 1.0,
        "aliases": aliases,
    }
