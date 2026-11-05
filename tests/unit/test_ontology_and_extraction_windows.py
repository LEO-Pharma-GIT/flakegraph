from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from documents import chunk

from kg_processor.application.extraction_windows import (
    build_document_context_windows,
    build_extraction_windows,
)
from kg_processor.application.llm_extractors import _relation_definition
from kg_processor.application.ontology import load_ontology
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ontology import OntologyProfile


def test_ontology_loader_validates_and_fingerprints_profile() -> None:
    """Verify ontology aliases, signatures, and fingerprints remain deterministic.

    Repeated loads should produce identical cache identity.
    """

    first = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)
    second = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)

    assert first.profile.relation("founded_by") is not None
    assert first.profile.relation("created_by").name == "FOUNDED_BY"  # type: ignore[union-attr]
    assert first.profile.relation("founded_by").source_types == [  # type: ignore[union-attr]
        "SCHOOL",
        "ORGANIZATION",
    ]
    assert first.profile.relation("associated_with") is None
    assert first.checksum == second.checksum
    assert len(first.checksum) == 64


def test_ontology_rejects_unknown_inverse() -> None:
    """Reject relation inverses that do not reference another declared predicate.

    Broken referential integrity must fail during profile loading.
    """

    with pytest.raises(ValueError, match="unknown inverse"):
        OntologyProfile.model_validate(
            {
                "name": "broken",
                "description": "Broken profile",
                "entity_types": [{"name": "PERSON", "description": "A person"}],
                "relation_types": [
                    {
                        "name": "TAUGHT",
                        "description": "Teaching",
                        "inverse": "STUDIED_UNDER",
                    }
                ],
            }
        )


def test_ontology_rejects_normalized_relation_alias_collisions() -> None:
    """Reject predicates that would resolve ambiguously after label normalization."""

    with pytest.raises(ValueError, match="unique after normalization"):
        OntologyProfile.model_validate(
            {
                "name": "ambiguous",
                "description": "Ambiguous profile",
                "entity_types": [{"name": "PERSON", "description": "A person"}],
                "relation_types": [
                    {
                        "name": "WORKS_AT",
                        "description": "Employment",
                        "aliases": ["works-at"],
                    },
                    {"name": "works at", "description": "Duplicate employment"},
                ],
            }
        )


def test_ontology_requires_document_context_for_a_document_subject() -> None:
    """Only a type the document-context pass looks for can be the document's subject."""

    with pytest.raises(ValueError, match="document subject without enabling document_context"):
        OntologyProfile.model_validate(
            {
                "name": "invalid-subject",
                "description": "Invalid subject profile",
                "entity_types": [
                    {"name": "PRODUCT", "description": "A product", "document_subject": True}
                ],
            }
        )


def test_ontology_requires_document_context_for_contextual_surfaces() -> None:
    """Prevent discourse aliases from entering an entity type that is never contextualized."""

    with pytest.raises(ValueError, match="without enabling document_context"):
        OntologyProfile.model_validate(
            {
                "name": "invalid-context",
                "description": "Invalid context profile",
                "entity_types": [
                    {
                        "name": "PAPER",
                        "description": "A paper",
                        "contextual_surfaces": ["this paper"],
                    }
                ],
            }
        )


def test_ontology_rejects_contextual_surface_shared_by_multiple_types() -> None:
    """Prevent one discourse phrase from identifying two document-wide actors."""

    with pytest.raises(ValueError, match="contextual surface 'We' is shared"):
        OntologyProfile.model_validate(
            {
                "name": "ambiguous-context",
                "description": "An invalid profile with an ambiguous pronoun.",
                "entity_types": [
                    {
                        "name": "PAPER",
                        "description": "A source paper.",
                        "document_context": True,
                        "contextual_surfaces": ["we"],
                    },
                    {
                        "name": "METHOD",
                        "description": "A source method.",
                        "document_context": True,
                        "contextual_surfaces": ["We"],
                    },
                ],
            }
        )


def test_open_ontology_accepts_grounded_relation_outside_declared_vocabulary() -> None:
    """Treat open relation definitions as guidance rather than a drop-only allowlist."""

    profile = OntologyProfile.model_validate(
        {
            "name": "open",
            "description": "Open profile",
            "mode": "open",
            "entity_types": [{"name": "PERSON", "description": "A person"}],
            "relation_types": [{"name": "RELATED_TO", "description": "Generic relation"}],
        }
    )
    reasons: Counter[str] = Counter()

    definition, stated = _relation_definition("studied-under", profile, reasons)

    assert definition is not None
    assert stated is None
    assert definition.name == "STUDIED_UNDER"
    assert reasons == {"open_relation_type": 1}


def test_extraction_windows_never_mix_documents() -> None:
    """Guarantee windowing preserves document boundaries despite interleaved input chunks.

    Adjacent chunks from the same document should still regroup.
    """

    chunks = [
        _chunk("a1", "document-a", 0, 40),
        _chunk("b1", "document-b", 0, 40),
        _chunk("a2", "document-a", 1, 40),
    ]

    windows = build_extraction_windows(chunks, target_tokens=100, max_chunks=3)

    assert [[chunk.id for chunk in window.chunks] for window in windows] == [
        ["a1", "a2"],
        ["b1"],
    ]
    assert all(len({chunk.document_id for chunk in window.chunks}) == 1 for window in windows)


def test_extraction_windows_preserve_gaps_in_filtered_chunk_sequences() -> None:
    """Keep removed reference windows from changing a queued batch's boundaries."""

    chunks = [
        _chunk("c17", "document-a", 17, 245),
        _chunk("c24", "document-a", 24, 322),
        _chunk("c25", "document-a", 25, 500),
        _chunk("c26", "document-a", 26, 221),
    ]

    windows = build_extraction_windows(chunks, target_tokens=700, max_chunks=2)

    assert [[chunk.chunk_index for chunk in window.chunks] for window in windows] == [
        [17],
        [24],
        [25],
        [26],
    ]


def test_document_context_windows_take_one_bounded_prefix_per_document() -> None:
    """Reuse front matter without coupling context cost to document length."""

    chunks = [
        _chunk("a1", "document-a", 0, 400),
        _chunk("b1", "document-b", 0, 500),
        _chunk("a2", "document-a", 1, 400),
        _chunk("a3", "document-a", 2, 400),
        _chunk("b2", "document-b", 1, 500),
    ]

    windows = build_document_context_windows(chunks, target_tokens=900, max_chunks=2)

    assert [[chunk.id for chunk in window.chunks] for window in windows] == [
        ["a1", "a2"],
        ["b1"],
    ]
    assert [window.token_count for window in windows] == [800, 500]


def _chunk(chunk_id: str, document_id: str, index: int, tokens: int) -> Chunk:
    """Create a chunk whose identity, order and token count the window tests control."""

    return chunk(
        f"Content for {chunk_id}",
        chunk_id=chunk_id,
        file_id=document_id,
        chunk_index=index,
        token_count=tokens,
    )


def test_inline_ontology_is_preferred_over_a_path() -> None:
    """Runtimes that cannot see the author's filesystem receive the profile by value."""

    on_disk = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)
    inline = load_ontology(
        Path("/nonexistent/ontology.yaml"),
        [],
        None,
        inline=on_disk.profile.model_dump(mode="json"),
    )

    assert inline.source == "inline:configuration"
    assert inline.checksum == on_disk.checksum
    assert [t.name for t in inline.profile.entity_types] == [
        t.name for t in on_disk.profile.entity_types
    ]
