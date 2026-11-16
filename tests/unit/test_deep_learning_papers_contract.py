"""Public contracts for the non-redistributed deep-learning paper benchmark."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from kg_processor.application.graph_evaluation import load_gold_graph
from kg_processor.application.ontology import load_ontology

_ROOT = Path("data/deep_learning_papers")
_GOLD = _ROOT / "gold.json"
_EXPECTED_DOCUMENTS = 49
# The dataset licence promises that nothing quoted from a paper exceeds this.
_MAX_QUOTED_WORDS = 10
# Bibliographic identifiers, not quotations: a paper's title names the work
# the way a citation does, and the dataset description is FlakeGraph's own.
_UNQUOTED_FIELDS = {"title", "description"}


def test_gold_covers_every_indexed_paper_without_redistributing_sources() -> None:
    """Keep one checksum-pinned annotation record per externally downloaded PDF."""

    gold = load_gold_graph(_GOLD)

    assert gold.schema_version == 2
    assert gold.version == "1.0.1"
    assert len(gold.documents) == _EXPECTED_DOCUMENTS
    assert len({document.path for document in gold.documents}) == _EXPECTED_DOCUMENTS
    assert all(
        document.path.startswith("data/deep_learning_papers/files/") for document in gold.documents
    )
    assert all(document.path.endswith(".pdf") for document in gold.documents)
    assert all(document.checksum and len(document.checksum) == 64 for document in gold.documents)
    assert all(
        document.source_uri
        and urlparse(document.source_uri).scheme == "https"
        and document.title
        and document.year
        for document in gold.documents
    )
    assert not (_ROOT / "files").is_dir() or all(
        hashlib.sha256(Path(document.path).read_bytes()).hexdigest() == document.checksum
        for document in gold.documents
    )


def test_gold_uses_only_declared_scientific_ontology_labels() -> None:
    """Prevent annotation labels from drifting beyond the extraction profile."""

    gold = load_gold_graph(_GOLD)
    ontology = load_ontology(_ROOT / "ontology.yaml", [], None).profile
    entity_types = {definition.name for definition in ontology.entity_types}
    relation_types = {definition.name for definition in ontology.relation_types}
    entities_by_id = {entity.id: entity for entity in gold.entities}
    relations_by_name = {definition.name: definition for definition in ontology.relation_types}

    assert {entity.type for entity in gold.entities} <= entity_types
    assert {relation.relation_type for relation in gold.relations} <= relation_types
    assert all(
        entities_by_id[relation.source].type
        in relations_by_name[relation.relation_type].source_types
        and entities_by_id[relation.target].type
        in relations_by_name[relation.relation_type].target_types
        for relation in gold.relations
    )
    assert all(relation.observations for relation in gold.relations)
    assert all(
        observation.page_number is not None
        for relation in gold.relations
        for observation in relation.observations
    )
    assert all(
        len(observation.evidence_contains.split()) <= _MAX_QUOTED_WORDS
        and re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", observation.evidence_contains) is None
        for relation in gold.relations
        for observation in relation.observations
    )


def test_gold_quotes_no_more_than_ten_words_from_any_paper() -> None:
    """Every text field copied from a paper must stay within the licence's limit.

    The papers are copyrighted and not redistributed; the dataset terms say the
    only verbatim text carried is a locator of at most ten words. That has to
    hold for every field an annotator might paste source text into - the
    evidence anchors, the sentence field, entity names and aliases - not only
    the anchors, so the rule is checked over the raw JSON rather than over the
    fields the evaluator happens to read. Paper titles are exempt because a
    title identifies the work rather than quoting it, and the same goes for a
    PAPER entity whose name is that title.
    """

    gold = json.loads(_GOLD.read_text(encoding="utf-8"))
    paper_titles = {entity["name"] for entity in gold["entities"] if entity.get("type") == "PAPER"}
    too_long: list[str] = []

    def walk(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str):
            field = path.rsplit(".", 1)[-1]
            if field in _UNQUOTED_FIELDS or value in paper_titles:
                return
            if len(value.split()) > _MAX_QUOTED_WORDS:
                too_long.append(f"{path}: {value[:40]}...")

    walk(gold, "gold")

    assert too_long == []
    # The sentence field is the anchor itself, so it never widens the quotation.
    assert all(
        observation["sentence"] == observation["evidence_contains"]
        for relation in gold["relations"]
        for observation in relation["observations"]
    )


def test_downloaded_sources_and_generated_manifest_are_git_ignored() -> None:
    """Keep third-party papers and machine-local paths out of public history."""

    ignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "data/deep_learning_papers/files/" in ignore
    assert "data/deep_learning_papers/manifest.jsonl" in ignore
