# SPDX-License-Identifier: Apache-2.0
"""Replaceable ports for entity, relation, and semantic verification stages."""

from __future__ import annotations

from typing import Protocol

from kg_processor.domain.extraction import (
    EntityExtractionOutcome,
    EntityMention,
    ExtractionWindow,
    RelationExtractionOutcome,
    RelationObservation,
    VerificationOutcome,
)
from kg_processor.domain.ontology import OntologyProfile


class EntityExtractor(Protocol):
    """Define the replaceable port for grounded entity mention recognition.

    Implementations may use an LLM or local model but must return exact mention
    evidence and provider diagnostics in the shared domain contract.
    """

    def extract(
        self,
        window: ExtractionWindow,
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        max_entities: int,
        previous_entities: list[EntityMention] | None = None,
    ) -> EntityExtractionOutcome:
        """Return exact mentions from one document-scoped extraction window.

        Previous mentions are supplied only for targeted gleaning and deduplication.
        """
        ...


class RelationExtractor(Protocol):
    """Define the replaceable port for relations between accepted mention IDs.

    Implementations must honor ontology constraints and return grounded observation
    records rather than unconstrained endpoint names.
    """

    def extract(
        self,
        window: ExtractionWindow,
        entities: list[EntityMention],
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        max_relations: int,
        previous_relations: list[RelationObservation] | None = None,
        seed: int | None = None,
    ) -> RelationExtractionOutcome:
        """Return grounded observations that reference supplied entity mention IDs.

        Previous observations support targeted relation gleaning without duplication.
        A seed, when given, replaces the extractor's own for this one call.
        """
        ...


class RelationVerifier(Protocol):
    """Define an independent semantic entailment verifier for candidate triples.

    Verification cannot create relations or entities; it may only judge supplied
    observation and mention IDs.
    """

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
        """Return one evidence-based verdict for every supplied relation and entity.

        ``entities`` lets a relation's ends be recognized; ``entities_to_verify``
        are judged. The document's opening and subjects say what the document
        is about, which a row or field far below its header does not repeat.
        Missing or malformed provider decisions are reconciled conservatively by callers.
        """
        ...
