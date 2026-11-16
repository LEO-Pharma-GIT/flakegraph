from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kg_processor.application.inspect import inspect_local_graph
from kg_processor.application.pipeline import KgProcessorPipeline
from kg_processor.config.preflight import run_preflight
from kg_processor.config.settings import Settings
from kg_processor.factories import (
    build_cache,
    build_embedding_provider,
    build_file_source,
    build_llm_provider,
    build_ocr_provider,
    build_writer,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("KG_RUN_AZURE_OPENAI_LIVE") != "1",
        reason="Set KG_RUN_AZURE_OPENAI_LIVE=1 to run live Azure OpenAI integration checks.",
    ),
]

_CONFIG = Path("tests/fixtures/configs/azure-openai-live.yaml")
_DEFAULT_LLM_MODEL = "gpt-4.1-mini-2025-04-14"
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"
_DEFAULT_API_VERSION = "2025-01-01-preview"


def test_live_azure_openai_pipeline_writes_quality_graph(tmp_path: Path) -> None:
    env = _live_env()
    output_path = tmp_path / "out"
    settings = Settings.load(
        _CONFIG,
        env=env,
        overrides={"writer": {"output_path": output_path}},
    )

    preflight = run_preflight(settings)
    assert preflight.ok, preflight.model_dump(mode="json")

    batch = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
    ).run()
    inspection = inspect_local_graph(output_path)

    assert batch.run_report["files_processed"] == 1
    assert batch.chunks
    assert batch.nodes
    assert batch.evidence
    assert batch.graph_metrics["providers"]["llm"] == "azure_openai"
    assert batch.graph_metrics["providers"]["embedding"] == "azure_openai"
    assert batch.graph_metrics["quality"]["ok"] is True
    assert inspection["schema"]["ok"] is True
    assert inspection["quality"]["ok"] is True


def _live_env() -> dict[str, str]:
    env = dict(os.environ)
    if not _has_provider_env(env):
        endpoint, api_key = _azure_ai_endpoint_and_key(env)
        env["KG_LLM_ENDPOINT"] = endpoint
        env["KG_EMBED_ENDPOINT"] = endpoint
        env["KG_LLM_API_KEY"] = api_key
        env["KG_EMBED_API_KEY"] = api_key
    env.setdefault("KG_LLM_MODEL", _DEFAULT_LLM_MODEL)
    env.setdefault("KG_EMBED_MODEL", _DEFAULT_EMBED_MODEL)
    env.setdefault("KG_LLM_API_VERSION", _DEFAULT_API_VERSION)
    env.setdefault("KG_EMBED_API_VERSION", _DEFAULT_API_VERSION)
    return env


def _has_provider_env(env: dict[str, str]) -> bool:
    return all(
        env.get(name)
        for name in (
            "KG_LLM_ENDPOINT",
            "KG_LLM_API_KEY",
            "KG_EMBED_ENDPOINT",
            "KG_EMBED_API_KEY",
        )
    )


def _azure_ai_endpoint_and_key(env: dict[str, str]) -> tuple[str, str]:
    required_names = (
        "KG_AZURE_OPENAI_SUBSCRIPTION",
        "KG_AZURE_OPENAI_RESOURCE_GROUP",
        "KG_AZURE_OPENAI_ACCOUNT",
    )
    missing = [name for name in required_names if not env.get(name)]
    if missing:
        pytest.skip(
            "Set KG_LLM_ENDPOINT/KG_LLM_API_KEY/KG_EMBED_ENDPOINT/KG_EMBED_API_KEY, "
            "or set the Azure CLI lookup variables: " + ", ".join(missing)
        )
    subscription = env["KG_AZURE_OPENAI_SUBSCRIPTION"]
    resource_group = env["KG_AZURE_OPENAI_RESOURCE_GROUP"]
    account = env["KG_AZURE_OPENAI_ACCOUNT"]
    endpoint = _az_output(
        [
            "az",
            "cognitiveservices",
            "account",
            "show",
            "--subscription",
            subscription,
            "--resource-group",
            resource_group,
            "--name",
            account,
            "--query",
            "properties.endpoint",
            "-o",
            "tsv",
        ]
    )
    api_key = _az_output(
        [
            "az",
            "cognitiveservices",
            "account",
            "keys",
            "list",
            "--subscription",
            subscription,
            "--resource-group",
            resource_group,
            "--name",
            account,
            "--query",
            "key1",
            "-o",
            "tsv",
        ]
    )
    return endpoint, api_key


def _az_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"Azure CLI is not available for live Azure OpenAI test: {exc}")
    if completed.returncode != 0:
        pytest.skip(f"Azure CLI lookup failed for live Azure OpenAI test: {completed.stderr}")
    value = completed.stdout.strip()
    if not value:
        pytest.skip("Azure CLI lookup returned an empty value for live Azure OpenAI test.")
    return value
