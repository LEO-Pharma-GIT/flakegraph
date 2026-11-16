# SPDX-License-Identifier: Apache-2.0
"""Versioned, privacy-conscious assembly of extraction benchmark reports.

The module keeps benchmark orchestration at the CLI boundary while centralizing
the durable report contract.  Schema version 2 makes dataset, model, protocol,
runtime, hardware, timing, quality, and stability dimensions explicit so reports
from different environments can be compared without exposing connection details
or machine/user identities.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from kg_processor import __version__
from kg_processor.application.graph_evaluation import jaccard, load_gold_graph
from kg_processor.config.settings import Settings
from kg_processor.domain.ids import sha256_file

BENCHMARK_SCHEMA_VERSION = 2
_QUALITY_METRICS: dict[str, tuple[str, ...]] = {
    "entity_precision": ("entities", "precision"),
    "entity_recall": ("entities", "recall"),
    "entity_f1": ("entities", "f1"),
    "triple_precision": ("triples", "precision"),
    "triple_recall": ("triples", "recall"),
    "triple_f1": ("triples", "f1"),
    "evidence_support_rate": ("evidence", "support_rate"),
    "component_count": ("structural", "components"),
    "isolate_ratio": ("structural", "isolate_ratio"),
    "two_hop_recoverability": ("two_hop_recoverability",),
    "information_retention": ("information_retention",),
}


def benchmark_stability(
    signatures: Sequence[tuple[set[str], set[str]]],
) -> dict[str, int | float | None]:
    """Summarize pairwise node and triple overlap across completed benchmark runs.

    Jaccard comparisons are computed for every unique run pair.  With fewer than
    two runs no pairwise observation exists, so metric values are ``None``.
    """

    node_scores: list[float] = []
    triple_scores: list[float] = []
    for left_index, left in enumerate(signatures):
        for right in signatures[left_index + 1 :]:
            node_scores.append(jaccard(left[0], right[0]))
            triple_scores.append(jaccard(left[1], right[1]))
    return {
        "pair_count": len(node_scores),
        "mean_node_jaccard": _rounded_mean(node_scores, digits=4),
        "mean_triple_jaccard": _rounded_mean(triple_scores, digits=4),
        "minimum_node_jaccard": round(min(node_scores), 4) if node_scores else None,
        "minimum_triple_jaccard": round(min(triple_scores), 4) if triple_scores else None,
    }


def build_benchmark_report(
    *,
    config_path: Path,
    gold_path: Path,
    repeats: int,
    settings: Settings,
    runs: Sequence[Mapping[str, Any]],
    signatures: Sequence[tuple[set[str], set[str]]],
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Assemble a schema-version-2 extraction benchmark report.

    ``runs`` contains the unabridged CLI per-run payloads. Summary data is
    derived from that list and an explicit allowlist of resolved settings.
    Endpoint URLs, credentials, environment variables,
    hostnames, usernames, and unrelated provider configuration cannot leak into
    the report through broad model serialization.

    Args:
        config_path: YAML configuration artifact used to resolve the run.
        gold_path: Curated JSON graph used to evaluate each extracted graph.
        repeats: Requested uncached repetition count from the CLI protocol.
        settings: Effective settings after benchmark-specific overrides.
        runs: Raw completed run records, including evaluations and run reports.
        signatures: Normalized node/triple sets corresponding to completed runs.
        generated_at: Optional timestamp injection used by deterministic tests.

    Returns:
        A JSON-serializable benchmark report with explicit provenance and summary data.

    Raises:
        ValueError: If run/signature counts differ or no completed runs are given.
    """

    if not runs:
        raise ValueError("benchmark report requires at least one completed run")
    if len(runs) != len(signatures):
        raise ValueError("benchmark runs and signatures must have matching lengths")

    elapsed_seconds = [_number(run, "elapsed_seconds") for run in runs]
    quality = _average_quality_metrics(runs)
    stability = benchmark_stability(signatures)
    documents_per_minute = [
        _documents_per_minute(run, elapsed)
        for run, elapsed in zip(runs, elapsed_seconds, strict=True)
    ]
    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    timing = {
        "mean_seconds": round(mean(elapsed_seconds), 3),
        "median_seconds": round(median(elapsed_seconds), 3),
        "min_seconds": round(min(elapsed_seconds), 3),
        "max_seconds": round(max(elapsed_seconds), 3),
    }
    safe_config_path = _safe_path(config_path)
    safe_gold_path = _safe_path(gold_path)
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "generated_at_utc": timestamp,
        "provenance": {
            "config": {"path": safe_config_path, "sha256": sha256_file(config_path)},
            "dataset": {
                "gold_name": load_gold_graph(gold_path).name,
                "gold_path": safe_gold_path,
                "gold_sha256": sha256_file(gold_path),
                "input": _input_provenance(settings),
            },
            "providers": _provider_provenance(settings),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "runtime": {
                "python_version": platform.python_version(),
                "package_version": __version__,
            },
        },
        "protocol": {
            "repeats": repeats,
            "cache_disabled": settings.cache.provider == "none",
            "extraction_parallelism": settings.graph.extraction_parallelism,
            "finalization_provider_concurrency": settings.graph.finalization_provider_concurrency,
            "embedding_batch_size": settings.embedding.batch_size,
            "deterministic_seed": settings.graph.deterministic_seed,
        },
        "summary": {
            "completed_run_count": len(runs),
            "successful_run_count": len(runs),
            "accepted_run_count": sum(bool(_evaluation(run).get("ok")) for run in runs),
            "timing": timing,
            "throughput": {
                "mean_documents_per_minute": _rounded_mean(
                    documents_per_minute,
                    digits=3,
                )
            },
            "quality": quality,
            "stability": stability,
        },
        "runs": [_safe_run(run) for run in runs],
    }


def _input_provenance(settings: Settings) -> dict[str, str | None]:
    """Describe the selected dataset source using only its comparison-safe path.

    Manifest-backed runs identify the manifest; all other sources retain the
    configured input path.  Provider account, stage, and blob connection fields
    are deliberately outside this allowlist.
    """

    if settings.files.source == "manifest":
        return {
            "source": settings.files.source,
            "manifest_path": (
                _safe_path(settings.files.manifest_path)
                if settings.files.manifest_path is not None
                else None
            ),
        }
    return {
        "source": settings.files.source,
        "input_path": _safe_path(settings.files.input_path),
    }


def _provider_provenance(settings: Settings) -> dict[str, dict[str, str]]:
    """Return provider and model identities required for cross-report comparison.

    The fixed shape is intentionally assembled field by field.  Transport URLs,
    API versions, keys, and all other provider-specific settings are excluded.
    """

    return {
        "ocr": {"provider": settings.ocr.provider},
        "llm": {"provider": settings.llm.provider, "model": settings.llm.model},
        "embedding": {
            "provider": settings.embedding.provider,
            "model": settings.embedding.model,
        },
        "entity_extractor": {
            "provider": settings.extractors.entity_provider,
            "model": (
                settings.extractors.gliner_model
                if settings.extractors.entity_provider == "gliner"
                else settings.llm.model
            ),
        },
        "relation_extractor": {
            "provider": settings.extractors.relation_provider,
            "model": settings.llm.model,
        },
        "verifier": {
            "provider": settings.extractors.verifier_provider,
            "model": settings.llm.model,
        },
        "writer": {"provider": settings.writer.provider},
        "cache": {"provider": settings.cache.provider},
    }


def _average_quality_metrics(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    """Average stable core evaluation metrics across completed runs.

    Explicit metric paths keep verbose diagnostic counts and changing quality
    payload internals out of the comparison summary while raw evaluations remain
    available under ``runs``.
    """

    result: dict[str, float | None] = {}
    for name, path in _QUALITY_METRICS.items():
        values = [
            value
            for run in runs
            if (value := _nested_optional_number(_evaluation(run), path)) is not None
        ]
        result[name] = round(mean(values), 4) if values else None
    return result


def _documents_per_minute(run: Mapping[str, Any], elapsed_seconds: float) -> float:
    """Calculate end-to-end document throughput from one completed run report."""

    run_report = run.get("run_report")
    if not isinstance(run_report, Mapping):
        raise ValueError("benchmark run is missing a run_report mapping")
    documents = _number(run_report, "files_processed")
    if elapsed_seconds <= 0:
        raise ValueError("benchmark elapsed_seconds must be positive")
    return documents * 60.0 / elapsed_seconds


def _evaluation(run: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return one run's evaluation mapping or fail with a useful contract error."""

    value = run.get("evaluation")
    if not isinstance(value, Mapping):
        raise ValueError("benchmark run is missing an evaluation mapping")
    return value


def _number(mapping: Mapping[str, Any], key: str) -> float:
    """Read a finite numeric report field without accepting booleans."""

    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"benchmark field '{key}' must be numeric")
    return float(value)


def _nested_optional_number(
    mapping: Mapping[str, Any],
    path: tuple[str, ...],
) -> float | None:
    """Read a metric while preserving an explicit not-applicable null."""

    current: Any = mapping
    for key in path[:-1]:
        if not isinstance(current, Mapping):
            raise ValueError(f"benchmark metric '{'.'.join(path)}' is missing")
        current = current.get(key)
    if not isinstance(current, Mapping):
        raise ValueError(f"benchmark metric '{'.'.join(path)}' is missing")
    value = current.get(path[-1])
    if value is None:
        return None
    return _number(current, path[-1])


def _rounded_mean(values: Sequence[float], *, digits: int) -> float | None:
    """Return a rounded arithmetic mean, or ``None`` for an unmeasured series."""

    return round(mean(values), digits) if values else None


def _safe_run(run: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a raw run while privacy-normalizing its known filesystem path fields.

    Metrics, reports, and diagnostics are retained unchanged.  Only paths emitted
    by benchmark orchestration and graph evaluation are rewritten, preventing an
    absolute output directory from reintroducing a username into an otherwise
    allowlisted report.
    """

    result = dict(run)
    output = result.get("output")
    if isinstance(output, str):
        result["output"] = _safe_path(Path(output))
    evaluation = result.get("evaluation")
    if isinstance(evaluation, Mapping):
        safe_evaluation = dict(evaluation)
        for key in ("output_path", "gold_path"):
            value = safe_evaluation.get(key)
            if isinstance(value, str):
                safe_evaluation[key] = _safe_path(Path(value))
        result["evaluation"] = safe_evaluation
    return result


def _safe_path(path: Path) -> str:
    """Render a path while eliding the current user's home directory component.

    Relative paths remain unchanged. Paths under the active home directory use
    ``~``; exact username path components elsewhere use a neutral placeholder.
    Substrings inside legitimate artifact names remain unchanged so provenance
    paths retain their reproducible identity.
    """

    try:
        relative = path.expanduser().relative_to(Path.home())
    except ValueError:
        rendered = str(path)
    else:
        rendered = str(Path("~") / relative)
    username = Path.home().name
    if not username or rendered.startswith("~"):
        return rendered
    return str(Path(*("<user>" if part == username else part for part in path.parts)))
