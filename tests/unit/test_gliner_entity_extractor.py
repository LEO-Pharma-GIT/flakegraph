from __future__ import annotations

from documents import chunk, window

from kg_processor.adapters.extraction.gliner import GlinerEntityExtractor
from kg_processor.domain.ontology import EntityTypeDefinition, OntologyProfile

_PEOPLE = OntologyProfile(
    name="people",
    description="People",
    entity_types=[EntityTypeDefinition(name="PERSON", description="A person")],
)


class _StubGliner:
    """Provide one deterministic GLiNER-style prediction without loading model dependencies.

    Adapter behavior remains isolated from inference quality.
    """

    def predict_entities(  # type: ignore[no-untyped-def]
        self, text, labels, threshold, flat_ner
    ):
        """Validate adapter options and return one exact source-span prediction record.

        The shape matches GLiNER's public output contract.
        """

        assert labels == ["PERSON"]
        assert threshold == 0.5
        assert flat_ner is True
        return [{"text": "Jigoro Kano", "label": "PERSON", "start": 0, "end": 11, "score": 0.97}]


class _WindowRecordingGliner:
    """Return a late entity only from the window where its exact text appears."""

    def __init__(self) -> None:
        self.window_word_counts: list[int] = []

    def predict_entities(  # type: ignore[no-untyped-def]
        self, text, labels, threshold, flat_ner
    ):
        """Record bounded inputs and return one source-relative late prediction."""

        _ = labels, threshold, flat_ner
        self.window_word_counts.append(len(text.split()))
        name = "Late Entity"
        start = text.find(name)
        if start < 0:
            return []
        return [{"label": "PERSON", "start": start, "end": start + len(name), "score": 0.9}]


def test_gliner_adapter_maps_exact_spans_to_entity_port() -> None:
    """Verify GLiNER predictions become grounded provider-neutral entity mentions.

    Exact offsets and provider provenance must survive translation.
    """

    content = "Jigoro Kano founded the Kodokan."
    extractor = GlinerEntityExtractor("stub")
    extractor._model = _StubGliner()

    outcome = extractor.extract(
        window(chunk(content)),
        _PEOPLE,
        model="ignored",
        timeout_seconds=1,
        max_entities=10,
    )

    assert len(outcome.entities) == 1
    assert outcome.entities[0].name == "Jigoro Kano"
    assert outcome.entities[0].quote == content[:11]
    assert outcome.entities[0].start_offset == 0
    assert outcome.trace["provider"] == "gliner"


def test_gliner_adapter_windows_long_chunks_without_losing_source_offsets() -> None:
    """Find entities beyond a model's normal truncation range and restore offsets."""

    prefix = " ".join(f"word{index}" for index in range(700)) + " "
    content = prefix + "Late Entity appears here."
    model = _WindowRecordingGliner()
    extractor = GlinerEntityExtractor("stub")
    extractor._model = model

    outcome = extractor.extract(
        window(chunk(content)),
        _PEOPLE,
        model="ignored",
        timeout_seconds=1,
        max_entities=10,
    )

    assert max(model.window_word_counts) <= 320
    assert outcome.entities[0].name == "Late Entity"
    assert outcome.entities[0].start_offset == content.index("Late Entity")
    assert outcome.trace["inference_windows"] > 1
