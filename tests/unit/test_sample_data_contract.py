from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.adapters.files.manifest import ManifestFileSource
from kg_processor.adapters.ocr.builtin_text import BuiltinTextOcrProvider
from kg_processor.application.graph_evaluation import load_gold_graph
from kg_processor.domain.ids import sha256_hex
from kg_processor.ports.ocr import OcrOptions

_DATASET_DIR = Path("data/martial_arts")
_SAMPLE_DIR = _DATASET_DIR / "files"
_MANIFEST = _DATASET_DIR / "manifest.jsonl"
_GOLD = Path("data/martial_arts/gold.json")
_REQUIRED_DATASET_PATHS = {
    "README.md",
    "BENCHMARKS.md",
    "LICENSE.md",
    "manifest.jsonl",
    "gold.json",
    "ontology.yaml",
    "files",
    "configs",
    "results",
}


def _expected_files() -> set[str]:
    """Return every source document referenced by the gold annotation."""

    return {Path(document.path).name for document in load_gold_graph(_GOLD).documents}


def test_dataset_package_keeps_all_benchmark_artifacts_together() -> None:
    """Keep the dataset portable and make missing package sections visible."""

    assert {path.name for path in _DATASET_DIR.iterdir()} == _REQUIRED_DATASET_PATHS
    assert sorted((_DATASET_DIR / "configs").glob("*.yaml"))
    assert sorted((_DATASET_DIR / "results").glob("*.json"))
    assert (_DATASET_DIR / "ontology.yaml").is_file()


def test_sample_manifest_pins_every_reusable_original_fixture() -> None:
    rows = [
        json.loads(line)
        for line in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert {Path(row["path"]).name for row in rows} == _expected_files()
    for row in rows:
        path = _DATASET_DIR / row["path"]
        assert path.is_file()
        assert row["source_uri"] == f"file://data/martial_arts/files/{path.name}"
        assert row["checksum"] == sha256_hex(path.read_bytes())
        assert row["size_bytes"] == path.stat().st_size
        assert isinstance(row["mime_type"], str)
        assert row["mime_type"]


def test_sample_manifest_file_source_resolves_full_corpus() -> None:
    files = ManifestFileSource(_MANIFEST).list_files()

    assert {file.path.name for file in files} == _expected_files()
    assert all(file.checksum for file in files)
    assert all(file.mime_type for file in files)
    assert all(file.size_bytes > 0 for file in files)


def test_sample_readme_documents_every_reusable_fixture() -> None:
    readme = (_DATASET_DIR / "README.md").read_text(encoding="utf-8")

    for filename in _expected_files():
        assert f"`files/{filename}`" in readme
    assert "`manifest.jsonl`" in readme
    assert "Document Matrix" in readme
    assert "gold.json" in readme
    assert "results/" in readme


def test_expanded_gold_graph_is_connected_versioned_and_source_grounded() -> None:
    """Protect the benchmark scale, topology, and exact evidence-source contract."""

    gold = load_gold_graph(_GOLD)
    graph: nx.Graph[str] = nx.Graph()
    graph.add_nodes_from(entity.id for entity in gold.entities)
    graph.add_edges_from((relation.source, relation.target) for relation in gold.relations)

    assert gold.schema_version == 2
    assert gold.version == "2.1.0"
    assert gold.license == "CC0-1.0"
    assert gold.evaluation_scope.entity_coverage == "exhaustive"
    assert gold.evaluation_scope.relation_coverage == "exhaustive"
    assert len(gold.documents) == 10
    assert len(gold.entities) == 119
    assert len(gold.relations) == 148
    assert sum(relation.required for relation in gold.relations) == 135
    assert sum(len(relation.observations) for relation in gold.relations) == 161
    assert all(relation.observations for relation in gold.relations)
    # Exhaustive annotation keeps a named entity the text states no relation
    # for; everything that takes part in a relation is one connected graph. A
    # time period is an entity only when it takes part in a claim.
    isolates = set(nx.isolates(graph))
    assert nx.is_connected(graph.subgraph(set(graph) - isolates))
    types = {entity.id: entity.type for entity in gold.entities}
    assert all(types[entity_id] != "TIME_PERIOD" for entity_id in isolates)

    parsed_by_document: dict[str, str] = {}
    parser = BuiltinTextOcrProvider()
    for document in gold.documents:
        path = Path(document.path)
        assert document.checksum == sha256_hex(path.read_bytes())
        file = LocalFileSource(path).list_files()[0]
        parsed = parser.parse(file, OcrOptions())
        parsed_by_document[document.id] = " ".join(
            " ".join(page.raw_text.casefold().split()) for page in parsed.pages
        )

    for relation in gold.relations:
        for observation in relation.observations:
            normalized_sentence = " ".join(observation.sentence.casefold().split())
            assert normalized_sentence in parsed_by_document[observation.document]
