from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from kg_processor.config.settings import Settings, SnowflakeSettings

CONFIG_DIR = Path("configs")
DATA_DIR = Path("data")
ENV_PLACEHOLDER_PREFIX = "${"
ENV_PLACEHOLDER_SUFFIX = "}"
SENSITIVE_KEYS = {
    "api_key",
    "mineru_api_key",
    "connection_string",
    "oauth_token",
    "password",
    "private_key_path",
    "sas_token",
}
TEST_PROVIDER_NAMES = {"fake", "hash"}
TEST_CONFIG_STEM_SUFFIXES = ("smoke", "test")
SHARED_GRAPH_TUNING_FIELDS = {
    "chunk_token_size",
    "chunk_token_overlap",
    "extraction_window_tokens",
    "max_chunks_per_llm_call",
    "max_entities_per_batch",
    "max_relations_per_batch",
    "gleaning_max_passes",
    "verify_relations",
    "resolution_embedding_lexical_floor",
    "resolution_max_candidates_per_mention",
    "resolution_adjudication_batch_size",
}
# Concurrency is deliberately absent above. How many calls may be in flight is a
# property of the provider serving them, not of extraction policy: it changes how
# long a run takes and never what the run extracts. A hosted endpoint and a model
# on a workstation have entirely different budgets, so holding them equal would
# force one of the two to be wrong.
PROVIDER_THROUGHPUT_FIELDS = {
    "extraction_parallelism",
    "finalization_provider_concurrency",
}


def _public_config_paths() -> list[Path]:
    """Return generic and dataset-specific public profiles."""

    return sorted([*CONFIG_DIR.glob("*.yaml"), *DATA_DIR.glob("*/configs/*.yaml")])


def test_example_configs_do_not_inline_secret_values() -> None:
    violations: list[str] = []
    for config_path in _public_config_paths():
        config = _load_config(config_path)
        for dotted_path, value in _flatten(config):
            key = dotted_path.rsplit(".", 1)[-1]
            if key not in SENSITIVE_KEYS:
                continue
            if not isinstance(value, str) or not (
                value.startswith(ENV_PLACEHOLDER_PREFIX) and value.endswith(ENV_PLACEHOLDER_SUFFIX)
            ):
                violations.append(f"{config_path}:{dotted_path}={value!r}")

    assert violations == []


def test_fake_and_hash_providers_are_confined_to_smoke_or_test_configs() -> None:
    violations: list[str] = []
    for config_path in _public_config_paths():
        config = _load_config(config_path)
        test_provider_paths = [
            dotted_path
            for dotted_path, value in _flatten(config)
            if dotted_path.endswith(".provider") and value in TEST_PROVIDER_NAMES
        ]
        if test_provider_paths and not config_path.stem.endswith(TEST_CONFIG_STEM_SUFFIXES):
            violations.extend(f"{config_path}:{path}" for path in test_provider_paths)

    assert violations == []


def test_real_provider_profiles_inherit_shared_graph_tuning_defaults() -> None:
    """Prevent real-provider profiles from silently drifting from shared quality defaults.

    Provider selection should not alter extraction policy.
    """

    violations: list[str] = []
    for config_path in _public_config_paths():
        if config_path.stem.endswith(TEST_CONFIG_STEM_SUFFIXES):
            continue
        graph = _load_config(config_path).get("graph", {})
        if not isinstance(graph, dict):
            violations.append(f"{config_path}:graph must be a mapping")
            continue
        violations.extend(
            f"{config_path}:graph.{field}"
            for field in sorted(SHARED_GRAPH_TUNING_FIELDS.intersection(graph))
        )

    assert violations == []


def test_throughput_and_extraction_policy_stay_separate_concerns() -> None:
    """A profile may tune its provider's concurrency without touching policy.

    The two sets must not overlap: with a concurrency field among the shared
    tuning fields, profiles could not state the budget their provider actually
    serves, which is the setting that governs how long a run takes.
    """

    assert not SHARED_GRAPH_TUNING_FIELDS & PROVIDER_THROUGHPUT_FIELDS


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(item, child_prefix))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_flatten(item, f"{prefix}[{index}]"))
        return rows
    return [(prefix, value)]


def test_settings_validate_back_from_their_own_dump() -> None:
    """Distributed execution rebuilds settings on every Spark executor.

    The driver serializes settings and each executor validates them back. A dump
    emits field names while the configuration spelling of some fields is an
    alias, so a model that accepted only aliases could not read its own output
    and every executor function would fail before doing any work.
    """

    settings = Settings.model_validate(
        {"snowflake": {"schema": "PUBLIC", "database": "DB", "account": "acct"}}
    )

    restored = Settings.model_validate(settings.model_dump(mode="json"))

    assert restored.snowflake.schema_name == "PUBLIC"
    assert restored == settings
    # The configuration spelling stays valid, and a misspelling stays rejected.
    assert SnowflakeSettings.model_validate({"schema": "S"}).schema_name == "S"
    with pytest.raises(ValidationError):
        SnowflakeSettings.model_validate({"schemaa": "S"})
