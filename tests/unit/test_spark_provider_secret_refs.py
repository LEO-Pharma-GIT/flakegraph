# SPDX-License-Identifier: Apache-2.0
"""Keep executor credentials bound to the driver's actual Kubernetes Secrets."""

from __future__ import annotations

import json
from typing import Any

import pytest
from helm import FULLNAME, container, env, one, pod, render

from kg_processor.application.spark_finalization import _with_provider_environment
from kg_processor.config.settings import DistributedSettings, Settings


class Builder:
    """Capture Spark configuration without starting a JVM or provider."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def config(self, name: str, value: str) -> Builder:
        """Record the configured key and remain chainable."""

        self.values[name] = value
        return self


def test_sensitive_executor_values_require_explicit_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither the default Secret name nor environment spelling can infer a binding."""

    for name in list(__import__("os").environ):
        if name.startswith(("KG_LLM_", "KG_EMBED_", "KG_EMBEDDING_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("KG_LLM_ENDPOINT", "http://gateway.invalid/v1")
    monkeypatch.setenv("KG_LLM_API_KEY", "synthetic-driver-value")
    monkeypatch.setenv("KG_EMBEDDING_API_KEY", "synthetic-embedding-value")
    monkeypatch.setenv("KG_EMBED_API_KEY", "synthetic-unmapped-value")
    builder = Builder()
    _with_provider_environment(
        builder,
        DistributedSettings(
            spark_provider_secret_refs={
                "KG_LLM_API_KEY": "gateway-keys:batch-key",
                "KG_EMBEDDING_API_KEY": "embedding-keys:custom-key",
            }
        ),
    )
    assert builder.values == {
        "spark.executorEnv.KG_LLM_ENDPOINT": "http://gateway.invalid/v1",
        "spark.kubernetes.executor.secretKeyRef.KG_LLM_API_KEY": "gateway-keys:batch-key",
        "spark.kubernetes.executor.secretKeyRef.KG_EMBEDDING_API_KEY": "embedding-keys:custom-key",
    }


def test_secret_reference_json_uses_the_supported_environment_loader() -> None:
    """The Helm-rendered mapping survives the regular Settings environment path."""

    settings = Settings.load(
        env={"KG_DISTRIBUTED_SPARK_PROVIDER_SECRET_REFS": '{"KG_LLM_API_KEY":"keys:batch"}'}
    )
    assert settings.distributed.spark_provider_secret_refs == {"KG_LLM_API_KEY": "keys:batch"}


@pytest.mark.parametrize("gateway", [True, False])
def test_driver_and_executor_secret_references_agree(gateway: bool) -> None:
    """Cover gateway virtual keys and operator-renamed provider Secret keys."""

    rendered = render(
        (
            "spark.enabled=true",
            f"gateway.enabled={str(gateway).lower()}",
            "gateway.litellm.virtualKeySecret.name=custom-gateway",
            "gateway.litellm.virtualKeySecret.batchKey=custom-batch",
            "providerSecret.name=custom-providers",
            "providerSecret.env[0].name=KG_LLM_API_KEY",
            "providerSecret.env[0].key=renamed-llm-key",
            "providerSecret.env[0].optional=true",
        )
    )
    driver = env(container(pod(one(rendered, "Deployment", f"{FULLNAME}-finalize")), "worker"))
    references: dict[str, Any] = json.loads(
        driver["KG_DISTRIBUTED_SPARK_PROVIDER_SECRET_REFS"]["value"]
    )
    reference = driver["KG_LLM_API_KEY"]["valueFrom"]["secretKeyRef"]
    assert references["KG_LLM_API_KEY"] == f"{reference['name']}:{reference['key']}"
    assert references["KG_LLM_API_KEY"] == (
        "custom-gateway:custom-batch" if gateway else "custom-providers:renamed-llm-key"
    )
