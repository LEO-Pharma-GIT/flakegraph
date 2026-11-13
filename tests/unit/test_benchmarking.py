"""Unit contracts for versioned extraction benchmark report assembly."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from kg_processor.application.benchmarking import (
    benchmark_stability,
    build_benchmark_report,
)
from kg_processor.config.settings import Settings


def test_report_aggregates_timing_quality_success_and_stability(tmp_path: Path) -> None:
    """Average core quality while retaining timing extrema and pairwise overlap."""

    config, gold = _write_inputs(tmp_path)
    runs = [
        _run(1, elapsed=1.0, ok=True, quality=0.8),
        _run(2, elapsed=3.0, ok=False, quality=0.6),
        _run(3, elapsed=8.0, ok=True, quality=1.0),
    ]
    signatures = [
        ({"alice", "acme"}, {"alice|works_at|acme"}),
        ({"alice", "acme"}, {"alice|works_at|acme"}),
        ({"alice"}, set()),
    ]

    report = build_benchmark_report(
        config_path=config,
        gold_path=gold,
        repeats=3,
        settings=_settings(),
        runs=runs,
        signatures=signatures,
        generated_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
    )

    assert report["schema_version"] == 2
    assert report["generated_at_utc"] == "2026-07-11T12:00:00+00:00"
    assert report["protocol"] == {
        "repeats": 3,
        "cache_disabled": True,
        "extraction_parallelism": 2,
        "finalization_provider_concurrency": 2,
        "embedding_batch_size": 32,
        "deterministic_seed": 17,
    }
    assert report["summary"]["completed_run_count"] == 3
    assert report["summary"]["successful_run_count"] == 3
    assert report["summary"]["accepted_run_count"] == 2
    assert report["summary"]["timing"] == {
        "mean_seconds": 4.0,
        "median_seconds": 3.0,
        "min_seconds": 1.0,
        "max_seconds": 8.0,
    }
    assert set(report["summary"]["quality"].values()) == {0.8}
    assert report["summary"]["throughput"] == {"mean_documents_per_minute": 29.167}
    assert report["summary"]["stability"] == {
        "pair_count": 3,
        "mean_node_jaccard": 0.6667,
        "mean_triple_jaccard": 0.3333,
        "minimum_node_jaccard": 0.5,
        "minimum_triple_jaccard": 0.0,
    }
    assert report["runs"] == runs


def test_provenance_is_allowlisted_and_excludes_private_runtime_values(tmp_path: Path) -> None:
    """Keep comparison identities while excluding transports, secrets, and users."""

    config, gold = _write_inputs(tmp_path)
    settings = _settings()
    settings.llm.endpoint = "https://private-endpoint.example.invalid"
    settings.llm.api_key = "top-secret-llm-key"
    settings.embedding.endpoint = "https://embedding.internal.invalid"
    settings.embedding.api_key = "top-secret-embedding-key"
    settings.generic_http_ocr.api_key = "top-secret-ocr-key"
    settings.job.lease_owner = "private-username"
    settings.distributed.worker_id = "private-hostname"
    settings.files.manifest_path = Path.home() / "fixtures" / "manifest.json"

    report = build_benchmark_report(
        config_path=config,
        gold_path=gold,
        repeats=1,
        settings=settings,
        runs=[_run(1, elapsed=2.0, ok=True, quality=1.0)],
        signatures=[({"alice"}, set())],
    )
    serialized = json.dumps(report, sort_keys=True)

    assert "private-endpoint" not in serialized
    assert "top-secret" not in serialized
    assert "private-username" not in serialized
    assert "private-hostname" not in serialized
    assert str(Path.home()) not in serialized
    assert set(report["provenance"]["platform"]) == {"system", "release", "machine"}
    assert report["provenance"]["providers"]["llm"] == {
        "provider": "fake",
        "model": "benchmark-model",
    }
    assert report["provenance"]["dataset"]["input"] == {
        "source": "manifest",
        "manifest_path": "~/fixtures/manifest.json",
    }


def test_single_run_stability_is_explicitly_not_measured() -> None:
    """Do not claim perfect pairwise consistency when no run pair exists."""

    assert benchmark_stability([({"alice"}, {"alice|knows|bob"})]) == {
        "pair_count": 0,
        "mean_node_jaccard": None,
        "mean_triple_jaccard": None,
        "minimum_node_jaccard": None,
        "minimum_triple_jaccard": None,
    }


def _settings() -> Settings:
    """Build resolved benchmark settings with deterministic comparison identities."""

    return Settings.model_validate(
        {
            "files": {
                "source": "manifest",
                "manifest_path": "fixtures/manifest.json",
            },
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "benchmark-model"},
            "embedding": {"provider": "hash", "model": "hash-v1"},
            "writer": {"provider": "local_artifacts"},
            "cache": {"provider": "none"},
        }
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Write minimal valid config and gold artifacts used by report provenance."""

    config = tmp_path / "benchmark.yaml"
    config.write_text("llm:\n  model: benchmark-model\n", encoding="utf-8")
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "name": "tiny-gold",
                "description": "Tiny benchmark fixture.",
                "entities": [],
                "relations": [],
            }
        ),
        encoding="utf-8",
    )
    return config, gold


def _run(run: int, *, elapsed: float, ok: bool, quality: float) -> dict[str, object]:
    """Create one compact raw run containing every core evaluation metric."""

    return {
        "run": run,
        "output": f"out/run-{run:02d}",
        "elapsed_seconds": elapsed,
        "run_report": {"run_id": f"run-{run}", "files_processed": 1},
        "evaluation": {
            "ok": ok,
            "entities": {"precision": quality, "recall": quality, "f1": quality},
            "triples": {"precision": quality, "recall": quality, "f1": quality},
            "evidence": {"support_rate": quality},
            "structural": {"components": quality, "isolate_ratio": quality},
            "two_hop_recoverability": quality,
            "information_retention": quality,
        },
    }
