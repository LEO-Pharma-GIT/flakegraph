"""Contracts for compact, publishable benchmark result records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from kg_processor.application.graph_evaluation import load_gold_graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_STATUSES = {
    "measured",
    "validation-only",
    "partial",
    "failed",
    "pending",
    "superseded",
}
PRIVATE_KEY_FRAGMENTS = {"api_key", "endpoint", "hostname", "username", "password"}


def _result_paths() -> list[Path]:
    """Return every version-controlled compact benchmark result deterministically."""

    return sorted((PROJECT_ROOT / "data").glob("*/results/*.json"))


@pytest.mark.parametrize("result_path", _result_paths(), ids=lambda path: path.stem)
def test_published_benchmark_result_is_reproducible_and_private(
    result_path: Path,
) -> None:
    """Verify artifact digests, comparison dimensions, and privacy boundaries.

    Compact results are intended for a public repository. This contract catches
    stale fixture/config hashes and rejects connection or machine identity fields
    that do not belong in a reusable benchmark record.
    """

    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["schema_version"] == 1
    assert result["status"] in ALLOWED_STATUSES
    assert result["result_id"]
    gold_path = PROJECT_ROOT / result["dataset"]["gold_path"]
    gold = load_gold_graph(gold_path)
    assert result["dataset"]["document_count"] == len(gold.documents)

    digest_fields = {}
    if result["dataset"]["version"] == gold.version:
        digest_fields["gold_path"] = "gold_sha256"
        assert result["dataset"]["expected_entity_count"] == len(gold.entities)
        assert result["dataset"]["expected_relation_count"] == len(gold.relations)
        assert result["dataset"]["evidence_observation_count"] == sum(
            len(relation.observations) for relation in gold.relations
        )
    else:
        # A measurement is a record of the gold set it was scored against, so
        # its digest and counts name that historic version, not the current file.
        # The revision must be a later one that the dataset guide accounts for.
        assert _version_tuple(gold.version) > _version_tuple(result["dataset"]["version"])
        guide = (gold_path.parent / "README.md").read_text(encoding="utf-8")
        assert result["dataset"]["version"] in guide
        assert gold.version in guide
    if "manifest_path" in result["dataset"]:
        digest_fields["manifest_path"] = "manifest_sha256"
    for path_field, digest_field in digest_fields.items():
        artifact = PROJECT_ROOT / result["dataset"][path_field]
        assert _sha256(artifact) == result["dataset"][digest_field]
    config_path = result["configuration"].get("path")
    if config_path is not None:
        assert _sha256(PROJECT_ROOT / config_path) == result["configuration"]["sha256"]
    else:
        assert len(result["configuration"]["sha256"]) == 64
    assert (
        _sha256(PROJECT_ROOT / result["configuration"]["ontology_path"])
        == result["configuration"]["ontology_sha256"]
    )

    serialized = json.dumps(result, sort_keys=True).lower()
    assert "/users/" not in serialized
    assert "/home/private-user/" not in serialized
    assert "private-hostname" not in serialized
    assert not PRIVATE_KEY_FRAGMENTS & _all_keys(result)


def test_published_result_ids_are_unique_and_documented() -> None:
    """Keep result filenames discoverable without duplicating comparison IDs."""

    results = [json.loads(path.read_text(encoding="utf-8")) for path in _result_paths()]
    result_ids = [str(result["result_id"]) for result in results]
    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "data").glob("*/BENCHMARKS.md"))
    )

    assert results, "at least one real benchmark result should be published"
    assert len(result_ids) == len(set(result_ids))
    assert all(result_id in documentation for result_id in result_ids)


def test_single_repeat_result_does_not_claim_stability() -> None:
    """Represent absent run pairs as null instead of a perfect consistency score."""

    for result_path in _result_paths():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["protocol"]["completed_repeats"] != 1:
            continue
        stability = result["measurements"]["stability"]
        assert result["status"] == "partial"
        assert stability["pair_count"] == 0
        assert stability["mean_node_jaccard"] is None
        assert stability["mean_triple_jaccard"] is None


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _sha256(path: Path) -> str:
    """Return the exact-byte SHA-256 used by benchmark provenance records."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_keys(value: Any) -> set[str]:
    """Collect normalized mapping keys recursively for privacy checks."""

    if isinstance(value, dict):
        return {str(key).lower() for key in value} | {
            key for item in value.values() for key in _all_keys(item)
        }
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()
