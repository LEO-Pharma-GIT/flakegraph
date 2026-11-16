"""The gateway hook that keeps request-specific ids out of compiled schemas."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml
from helm import CHART, FULLNAME, one, render

_HOOK = CHART / "gateway" / "stable_schemas.py"
_CONFIG = {
    "model_list": [
        {"model_name": "llm-batch", "litellm_params": {"model": "hosted_vllm/engine"}},
        {
            "model_name": "llm-hosted",
            "litellm_params": {"model": "provider/hosted"},
            "model_info": {"stable_schemas": True},
        },
    ]
}


@pytest.fixture
def hook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> types.ModuleType:
    """Load the chart's hook beside a config naming one stable-schema alias."""

    integrations = types.ModuleType("litellm.integrations")
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")
    custom_logger.CustomLogger = type("CustomLogger", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))
    monkeypatch.setitem(sys.modules, "litellm.integrations", integrations)
    monkeypatch.setitem(sys.modules, "litellm.integrations.custom_logger", custom_logger)
    (tmp_path / "stable_schemas.py").write_text(_HOOK.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("stable_schemas", tmp_path / "stable_schemas.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(model: str) -> dict[str, Any]:
    item = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["PRODUCT", "BATCH"]},
            "source_chunk_id": {"type": "string", "enum": ["chunk_0123456789abcdef0123"]},
            "target_entity_id": {
                "type": "string",
                "enum": ["entity_mention_aaaaaaaaaaaaaaaa", "entity_mention_bbbbbbbbbbbbbbbb"],
            },
        },
    }
    schema = {"type": "object", "properties": {"records": {"type": "array", "items": item}}}
    return {
        "model": model,
        "messages": [{"role": "user", "content": "extract"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": schema}},
    }


def _sent(hook: types.ModuleType, data: dict[str, Any], call_type: str) -> dict[str, Any]:
    result: dict[str, Any] = asyncio.run(
        hook.instance.async_pre_call_hook(None, None, data, call_type)
    )
    return result


def _item(data: dict[str, Any]) -> dict[str, Any]:
    schema = data["response_format"]["json_schema"]["schema"]
    properties: dict[str, Any] = schema["properties"]["records"]["items"]["properties"]
    return properties


def test_only_aliases_that_ask_for_stable_schemas_are_listed(hook: types.ModuleType) -> None:
    assert hook.instance.models == frozenset({"llm-hosted"})


def test_id_enums_are_removed_and_vocabularies_kept(hook: types.ModuleType) -> None:
    """Two requests over different text then send one schema."""

    first = _sent(hook, _request("llm-hosted"), "acompletion")
    other = _request("llm-hosted")
    _item(other)["source_chunk_id"]["enum"] = ["chunk_fedcba9876543210fedc"]
    second = _sent(hook, other, "completion")

    assert _item(first)["type"]["enum"] == ["PRODUCT", "BATCH"]
    assert "enum" not in _item(first)["source_chunk_id"]
    assert "enum" not in _item(first)["target_entity_id"]
    assert first["response_format"] == second["response_format"]


def test_other_models_and_calls_keep_their_schemas(hook: types.ModuleType) -> None:
    """The engines' guided decoding keeps every enum the caller sent."""

    engine = _sent(hook, _request("llm-batch"), "acompletion")
    embedding = _sent(hook, _request("llm-hosted"), "aembedding")

    assert _item(engine)["source_chunk_id"]["enum"] == ["chunk_0123456789abcdef0123"]
    assert _item(embedding)["target_entity_id"]["enum"]


def test_an_enum_mixing_ids_and_words_is_kept(hook: types.ModuleType) -> None:
    schema = {"enum": ["chunk_0123456789abcdef0123", "none"]}

    hook.remove_id_enums(schema)

    assert schema == {"enum": ["chunk_0123456789abcdef0123", "none"]}


def test_the_chart_ships_the_hook_beside_the_proxy_config() -> None:
    config_map = one(render(), "ConfigMap", f"{FULLNAME}-litellm")
    assert config_map["data"]["stable_schemas.py"] == _HOOK.read_text(encoding="utf-8")
    config = yaml.safe_load(config_map["data"]["config.yaml"])
    assert "stable_schemas.instance" in config["litellm_settings"]["callbacks"]
