from __future__ import annotations

import importlib
from pathlib import Path

from kg_processor.adapters.writers.snowflake_direct import TABLE_COLUMNS
from kg_processor.application.inspect import (
    _PARQUET_TABLES,
    _REQUIRED_COLUMNS,
    _STABILITY_KEY_COLUMNS,
)
from kg_processor.application.snowflake_access import (
    REQUIRED_SNOWFLAKE_COLUMNS,
    REQUIRED_SNOWFLAKE_TABLES,
)
from kg_processor.application.snowflake_schema import snowflake_schema_columns
from kg_processor.config.provider_registry import (
    ProviderKind,
    provider_catalog,
    provider_kinds,
    provider_names,
)
from kg_processor.config.settings import Settings

_LOCAL_TO_SNOWFLAKE_TABLES = {
    "documents": "KG_DOCUMENT",
    "pages": "KG_PAGE",
    "blocks": "KG_BLOCK",
    "assets": "KG_ASSET",
    "chunks": "KG_CHUNK",
    "nodes": "KG_NODE",
    "edges": "KG_EDGE",
    "edge_observations": "KG_EDGE_OBSERVATION",
    "evidence": "KG_EVIDENCE",
    "entity_sources": "KG_ENTITY_SOURCE",
    "communities": "KG_COMMUNITY",
    "community_findings": "KG_COMMUNITY_FINDING",
    "discarded_windows": "KG_DISCARDED_WINDOW",
    "rejected_records": "KG_REJECTED_RECORD",
    "failed_documents": "KG_FAILED_DOCUMENT",
}
_EXPECTED_PROVIDER_NAMES: dict[ProviderKind, set[str]] = {
    "file_source": {"local", "manifest", "azure_blob", "s3", "snowflake_stage"},
    "ocr": {
        "fallback",
        "mineru_internal",
        "mineru_api",
        "builtin_text",
        "generic_http",
        "tesseract_internal",
        "snowflake_cortex",
    },
    "llm": {
        "fake",
        "ollama",
        "openai_compatible",
        "azure_openai",
        "vllm_local",
        "snowflake_cortex",
    },
    "embedding": {
        "hash",
        "sentence_transformers",
        "openai_compatible",
        "azure_openai",
        "snowflake_cortex",
    },
    "writer": {"local_artifacts", "snowflake_direct", "snowflake_bulk"},
    "cache": {"none", "local", "snowflake"},
}


def test_provider_catalog_entries_are_unique_supported_and_importable() -> None:
    seen: set[tuple[str, str]] = set()
    providers_by_kind = {kind: provider_names(kind) for kind in provider_kinds()}

    assert set(providers_by_kind) == set(_EXPECTED_PROVIDER_NAMES)
    assert providers_by_kind == {
        kind: frozenset(names) for kind, names in _EXPECTED_PROVIDER_NAMES.items()
    }

    for provider in provider_catalog():
        key = (provider["kind"], provider["name"])
        assert key not in seen
        seen.add(key)
        assert provider["kind"] in providers_by_kind
        assert provider["summary"]
        if provider["implementation"]:
            module_name, attribute_name = provider["implementation"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            assert hasattr(module, attribute_name)


def test_example_configs_select_registered_providers() -> None:
    for config_path in sorted(Path("configs").glob("*.yaml")):
        settings = Settings.load(config_path, env={})
        selected: dict[ProviderKind, str] = {
            "file_source": settings.files.source,
            "ocr": settings.ocr.provider,
            "llm": settings.llm.provider,
            "embedding": settings.embedding.provider,
            "writer": settings.writer.provider,
            "cache": settings.cache.provider,
        }
        for kind, name in selected.items():
            assert name in provider_names(kind), f"{config_path} selected {kind}={name}"


def test_local_artifact_tables_have_snowflake_logical_schema_counterparts() -> None:
    assert set(_PARQUET_TABLES) == set(_LOCAL_TO_SNOWFLAKE_TABLES)
    assert set(_PARQUET_TABLES) == set(_REQUIRED_COLUMNS)

    for local_table, snowflake_table in _LOCAL_TO_SNOWFLAKE_TABLES.items():
        snowflake_columns = {column.name.lower() for column in TABLE_COLUMNS[snowflake_table]}
        assert _REQUIRED_COLUMNS[local_table] <= snowflake_columns
        assert set(_STABILITY_KEY_COLUMNS) <= snowflake_columns


def test_snowflake_access_schema_contract_tracks_rendered_ddl_and_writer_columns() -> None:
    rendered_columns = snowflake_schema_columns()

    assert rendered_columns == REQUIRED_SNOWFLAKE_COLUMNS
    assert tuple(rendered_columns) == REQUIRED_SNOWFLAKE_TABLES
    for table_name, writer_columns in TABLE_COLUMNS.items():
        schema_columns = set(rendered_columns[table_name])
        assert {column.name for column in writer_columns} <= schema_columns
