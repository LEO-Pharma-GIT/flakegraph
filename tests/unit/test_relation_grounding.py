from __future__ import annotations

import pytest

from kg_processor.application.extraction_grounding import ground_evidence, surface_spans
from kg_processor.application.relation_grounding import relation_evidence_supported
from kg_processor.domain.ontology import OntologyProfile, RelationTypeDefinition


def test_relation_evidence_rejects_entity_list_without_predicate_cue() -> None:
    """Reject entity-list co-occurrence when no local predicate cue supports it.

    Mention proximity alone cannot create a graph fact.
    """

    definition = _relation("LOCATED_IN", [r"\b(?:in|at|located|based|situated)\b"])

    assert not relation_evidence_supported(
        "Jigoro Kano, Kodokan, Gichin Funakoshi, Okinawa, Korea",
        ["Kodokan"],
        ["Korea"],
        definition,
    )


def test_relation_evidence_keeps_cue_local_to_the_endpoint_pair() -> None:
    """Require the predicate cue to be local to the specific endpoint pair."""

    definition = _relation(
        "DERIVED_FROM",
        [r"\b(?:adapt(?:ed|ation)|derived|descended|evolved|grew|based|roots?)\b"],
    )
    source = (
        "judo was organized by Jigoro Kano in Tokyo, karate moved from Okinawa into "
        "Japan, and Brazilian jiu-jitsu developed through the adaptation of judo and "
        "ground-fighting methods in Brazil."
    )

    assert not relation_evidence_supported(
        source,
        ["karate"],
        ["judo"],
        definition,
    )
    assert relation_evidence_supported(
        source,
        ["Brazilian jiu-jitsu"],
        ["judo"],
        definition,
    )


def test_relation_evidence_accepts_compact_structured_rows() -> None:
    """Allow bounded table-like rows to express selected relations without prose verbs.

    This exception is limited to configured structured predicates.
    """

    definition = _relation("HAS_PRACTICE", [r"\b(?:includes?|uses?)\b"])

    assert relation_evidence_supported(
        "Judo\nTokyo\nEducation, throws, randori",
        ["Judo"],
        ["randori"],
        definition,
    )


def test_relation_evidence_does_not_treat_wrapped_prose_as_structured_row() -> None:
    """Prevent wrapped prose from bypassing configured predicate-cue grounding rules.

    Line wrapping alone does not make text tabular.
    """

    definition = _relation("PRACTICED_IN", [r"\b(?:in|to|throughout)\b"])
    source = (
        "A village wrestling style, a military sword curriculum, a temple\n"
        "boxing practice, and a modern sport gym can look unrelated, yet each answers similar\n"
        "questions about how people train safely and remember techniques."
    )

    assert not relation_evidence_supported(
        source,
        ["boxing"],
        ["modern sport gym"],
        definition,
    )


def test_ontology_rejects_invalid_relation_evidence_regex() -> None:
    """Reject malformed ontology evidence expressions before extraction begins.

    Runtime grounding should never encounter invalid regular expressions.
    """

    with pytest.raises(ValueError, match="invalid evidence cue"):
        OntologyProfile.model_validate(
            {
                "name": "invalid-cue",
                "description": "Invalid cue profile",
                "entity_types": [
                    {"name": "PERSON", "description": "A person"},
                ],
                "relation_types": [
                    {
                        "name": "KNOWS",
                        "description": "A relationship",
                        "evidence_cues": ["["],
                    }
                ],
            }
        )


def test_evidence_grounding_maps_unicode_ligatures_back_to_source_offsets() -> None:
    """Accept compatibility-equivalent text without rewriting published evidence."""

    source = "Adam is an efﬁcient stochastic optimization method."

    span = ground_evidence(
        source,
        "Adam is an efficient stochastic optimization method.",
        [["Adam"], ["efficient stochastic optimization method"]],
    )

    assert span is not None
    assert span.quote == source
    assert (span.start_offset, span.end_offset) == (0, len(source))
    start = source.index("efﬁcient")
    assert surface_spans(source, ["efficient"]) == [(start, start + len("efﬁcient"))]


def test_evidence_grounding_repairs_bounded_cross_sentence_antecedent() -> None:
    """Ground an explicit adjacent-sentence fact without using remote co-occurrence."""

    source = "The proposed optimizer is Adam. It uses bias correction. Later results follow."

    span = ground_evidence(
        source,
        "Adam uses bias correction.",
        [["Adam"], ["bias correction"]],
    )

    assert span is not None
    assert span.quote == "The proposed optimizer is Adam. It uses bias correction."
    assert span.repair == "supporting_passage"


def _relation(name: str, cues: list[str]) -> RelationTypeDefinition:
    """Create a minimal relation definition for isolated predicate-grounding scenarios.

    Only name and evidence cues vary between tests.
    """

    return RelationTypeDefinition(
        name=name,
        description=f"Evidence test for {name}",
        evidence_cues=cues,
    )
