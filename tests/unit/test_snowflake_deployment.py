from __future__ import annotations

import json
from typing import Any

import pytest
import yaml

from kg_processor.application.snowflake_deployment import (
    render_execute_job_service_sql,
    render_kubernetes_job_yaml,
    render_snowflake_image_reference,
    render_snowflake_objects_sql,
    render_snowflake_setup_sql,
    render_spcs_service_spec_yaml,
)
from kg_processor.config.settings import Settings, _deep_update

_DIGEST = "sha256:" + "a" * 64


def test_render_spcs_service_spec_uses_configured_image_env_and_resources() -> None:
    settings = _settings()

    rendered = render_spcs_service_spec_yaml(settings, "configs/snowflake-cortex.yaml")

    spec = yaml.safe_load(rendered)
    container = spec["spec"]["containers"][0]
    assert container["name"] == "flakegraph"
    assert container["image"] == "/KG_DB/GRAPH/KG_IMAGES/flakegraph:latest"
    assert container["args"] == ["worker", "--config", "configs/snowflake-cortex.yaml"]
    assert container["env"]["KG_RUNTIME"] == "spcs"
    assert container["env"]["KG_JOB_ID"] == "job-123"
    assert container["env"]["KG_JOB_LEASE_OWNER"] == "worker-1"
    assert container["env"]["KG_JOB_USE_FILE_QUEUE"] == "true"
    assert container["env"]["KG_BATCH_FILES"] == "100"
    assert container["env"]["KG_SNOWFLAKE_BULK_TARGET_FILE_MB"] == "128"
    assert container["env"]["KG_SNOWFLAKE_AUTHENTICATOR"] == "oauth"
    runtime_config = json.loads(container["env"]["KG_CONFIG_JSON"])
    assert runtime_config["graph"]["max_chunks_per_llm_call"] == 2
    assert runtime_config["graph"]["max_relations_per_batch"] == 40
    assert runtime_config["snowflake"]["warehouse"] == "KG_PROCESSOR_WH"
    assert "password" not in runtime_config["snowflake"]
    assert container["resources"]["requests"]["cpu"] == "500m"
    assert container["resources"]["limits"]["memory"] == "5Gi"


def test_render_spcs_service_spec_always_uses_injected_oauth_token() -> None:
    """Prevent local client authentication context from leaking into SPCS specs."""

    settings = _settings({"snowflake": {"authenticator": "externalbrowser"}})

    spec = yaml.safe_load(render_spcs_service_spec_yaml(settings))
    environment = spec["spec"]["containers"][0]["env"]

    assert environment["KG_SNOWFLAKE_AUTHENTICATOR"] == "oauth"
    assert environment["KG_SNOWFLAKE_OAUTH_TOKEN_PATH"] == "/snowflake/session/token"
    assert "KG_SNOWFLAKE_ACCOUNT" not in environment
    assert "KG_SNOWFLAKE_HOST" not in environment


def test_render_spcs_service_spec_preserves_processing_overrides() -> None:
    """Keep app-selected graph capacity settings effective inside SPCS."""

    settings = _settings(
        {
            "graph": {
                "max_chunks_per_llm_call": 1,
                "max_entities_per_batch": 30,
                "max_relations_per_batch": 20,
            }
        }
    )

    spec = yaml.safe_load(render_spcs_service_spec_yaml(settings))
    runtime_config = json.loads(spec["spec"]["containers"][0]["env"]["KG_CONFIG_JSON"])

    assert runtime_config["graph"]["max_chunks_per_llm_call"] == 1
    assert runtime_config["graph"]["max_entities_per_batch"] == 30
    assert runtime_config["graph"]["max_relations_per_batch"] == 20


def test_render_spcs_service_spec_can_pin_image_digest() -> None:
    settings = _settings({"snowflake": {"image_digest": _DIGEST.upper()}})

    rendered = render_spcs_service_spec_yaml(settings, "configs/snowflake-cortex.yaml")

    spec = yaml.safe_load(rendered)
    container = spec["spec"]["containers"][0]
    assert container["image"] == f"/KG_DB/GRAPH/KG_IMAGES/flakegraph:latest@{_DIGEST}"
    assert container["env"]["KG_SNOWFLAKE_IMAGE_DIGEST"] == _DIGEST


def test_render_spcs_service_spec_includes_gpu_resource_when_configured() -> None:
    settings = _settings({"snowflake": {"service_gpu_count": 1}})

    spec = yaml.safe_load(render_spcs_service_spec_yaml(settings))
    resources = spec["spec"]["containers"][0]["resources"]

    assert resources["requests"]["nvidia.com/gpu"] == "1"
    assert resources["limits"]["nvidia.com/gpu"] == "1"


def test_render_execute_job_service_sql_uses_stage_spec_file_and_async() -> None:
    settings = _settings()

    sql = render_execute_job_service_sql(settings, "deploy/flakegraph-job.yaml", True)

    assert "EXECUTE JOB SERVICE" in sql
    assert "IN COMPUTE POOL KG_PROCESSOR_CPU_POOL" in sql
    assert "NAME = KG_DB.GRAPH.KG_PROCESSOR_JOB" in sql
    assert "ASYNC = TRUE" in sql
    assert "FROM @KG_DB.GRAPH.KG_SERVICE_SPECS" in sql
    assert "SPEC = 'deploy/flakegraph-job.yaml'" in sql


def test_render_execute_job_service_sql_accepts_full_stage_spec_path() -> None:
    settings = _settings()

    sql = render_execute_job_service_sql(
        settings,
        "@KG_DB.GRAPH.KG_SERVICE_SPECS/deploy/flakegraph-job.yaml",
    )

    assert "FROM @KG_DB.GRAPH.KG_SERVICE_SPECS" in sql
    assert "SPEC = 'deploy/flakegraph-job.yaml'" in sql


def test_render_kubernetes_job_uses_onprem_env_secret_and_gpu_resources() -> None:
    settings = _settings(
        {
            "runtime": {"runtime": "onprem"},
            "files": {"source": "azure_blob"},
            "azure_blob": {
                "account_url": "https://storage.example",
                "container": "documents",
                "prefix": "incoming",
                "sas_token": "secret-sas",
            },
            "ocr": {"provider": "mineru_internal", "model_cache_dir": "/cache/mineru"},
            "llm": {
                "provider": "vllm_local",
                "endpoint": "http://vllm:8000/v1",
                "model": "Qwen/Qwen2.5-14B-Instruct",
                "api_key": "secret-llm-key",
            },
            "embedding": {
                "provider": "sentence_transformers",
                "model": "Snowflake/snowflake-arctic-embed-l-v2.0",
                "dimension": 1024,
                "device": "cuda",
            },
            "snowflake": {"service_gpu_count": 1},
        }
    )

    rendered = render_kubernetes_job_yaml(
        settings,
        image="registry.example/flakegraph:latest",
        secret_name="flakegraph-secrets",
        parallelism=3,
        gpu_count=1,
    )

    manifest = yaml.safe_load(rendered)
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"]}
    assert manifest["kind"] == "Job"
    assert manifest["spec"]["parallelism"] == 3
    assert manifest["spec"]["completions"] == 3
    assert container["image"] == "registry.example/flakegraph:latest"
    assert container["envFrom"] == [{"secretRef": {"name": "flakegraph-secrets"}}]
    assert env["KG_RUNTIME"] == "onprem"
    assert env["KG_JOB_USE_FILE_QUEUE"] == "true"
    assert env["KG_FILE_SOURCE"] == "azure_blob"
    assert env["KG_AZURE_BLOB_CONTAINER"] == "documents"
    assert env["KG_LLM_PROVIDER"] == "vllm_local"
    assert env["KG_EMBED_MODEL"] == "Snowflake/snowflake-arctic-embed-l-v2.0"
    assert env["KG_SNOWFLAKE_BULK_STAGE"] == "@KG_DB.GRAPH.KG_LOAD_STAGE"
    assert env["KG_SNOWFLAKE_BULK_TARGET_FILE_MB"] == "128"
    assert "KG_LLM_API_KEY" not in env
    assert "KG_AZURE_BLOB_SAS_TOKEN" not in env
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    volumes = manifest["spec"]["template"]["spec"]["volumes"]
    assert {"name": "mineru-cache", "emptyDir": {}} in volumes


def test_render_kubernetes_job_gpu_count_zero_makes_a_cpu_only_job() -> None:
    settings = _settings({"snowflake": {"service_gpu_count": 1}})

    manifest = yaml.safe_load(render_kubernetes_job_yaml(settings, gpu_count=0))

    resources = manifest["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert "nvidia.com/gpu" not in resources["requests"]
    assert "nvidia.com/gpu" not in resources["limits"]


def test_render_kubernetes_job_sanitizes_instance_label() -> None:
    settings = _settings()
    settings.job.job_id = "run/2026-07-09"

    manifest = yaml.safe_load(render_kubernetes_job_yaml(settings))

    assert (
        manifest["spec"]["template"]["metadata"]["labels"]["app.kubernetes.io/instance"]
        == "run-2026-07-09"
    )


def test_render_snowflake_setup_sql_contains_objects_grants_and_schema() -> None:
    settings = _settings(
        {
            "snowflake": {
                "compute_pool_instance_family": "GPU_NV_SM",
                "compute_pool_min_nodes": 1,
                "compute_pool_max_nodes": 2,
            }
        }
    )

    sql = render_snowflake_setup_sql(settings, "KG_PROCESSOR_ROLE")

    assert "CREATE ROLE IF NOT EXISTS KG_PROCESSOR_ROLE" in sql
    assert "CREATE DATABASE IF NOT EXISTS KG_DB" in sql
    assert "CREATE SCHEMA IF NOT EXISTS KG_DB.GRAPH" in sql
    assert "CREATE COMPUTE POOL IF NOT EXISTS KG_PROCESSOR_CPU_POOL" in sql
    assert "INSTANCE_FAMILY = GPU_NV_SM" in sql
    assert "INITIALLY_SUSPENDED = TRUE" in sql
    assert "AUTO_SUSPEND_SECS = 60" in sql
    assert "CREATE STAGE IF NOT EXISTS KG_DB.GRAPH.KG_DOCS" in sql
    assert "GRANT READ ON STAGE KG_DB.GRAPH.KG_DOCS" in sql
    assert "CREATE STAGE IF NOT EXISTS KG_DB.GRAPH.KG_LOAD_STAGE" in sql
    assert "CREATE STAGE IF NOT EXISTS KG_DB.GRAPH.KG_SERVICE_SPECS" in sql
    assert "CREATE IMAGE REPOSITORY IF NOT EXISTS KG_DB.GRAPH.KG_IMAGES" in sql
    assert "GRANT READ ON IMAGE REPOSITORY KG_DB.GRAPH.KG_IMAGES" in sql
    assert "GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE KG_PROCESSOR_ROLE" in sql
    assert "USE DATABASE KG_DB" in sql
    assert "USE SCHEMA GRAPH" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_NODE" in sql
    assert "VECTOR(FLOAT, 1024)" in sql
    assert "ALTER TABLE KG_" not in sql
    assert sql.index("USE DATABASE KG_DB") < sql.index("CREATE TABLE IF NOT EXISTS KG_JOB")
    assert sql.index("USE SCHEMA GRAPH") < sql.index("CREATE TABLE IF NOT EXISTS KG_JOB")


def test_render_snowflake_setup_sql_creates_document_stage_base_when_prefix_configured() -> None:
    settings = _settings({"snowflake": {"stage": "@KG_DB.GRAPH.KG_DOCS/incoming/docs"}})

    sql = render_snowflake_setup_sql(settings, "KG_PROCESSOR_ROLE")

    assert "CREATE STAGE IF NOT EXISTS KG_DB.GRAPH.KG_DOCS" in sql
    assert "KG_DOCS/incoming/docs" not in sql


def test_render_snowflake_objects_sql_omits_admin_statements() -> None:
    settings = _settings(
        {
            "snowflake": {
                "compute_pool_instance_family": "GPU_NV_SM",
                "compute_pool_min_nodes": 1,
                "compute_pool_max_nodes": 2,
            }
        }
    )

    sql = render_snowflake_objects_sql(settings)

    assert "CREATE ROLE" not in sql
    assert "GRANT " not in sql
    assert "CREATE COMPUTE POOL" not in sql
    assert "USE DATABASE KG_DB" in sql
    assert "USE SCHEMA GRAPH" in sql
    assert "CREATE STAGE IF NOT EXISTS KG_DB.GRAPH.KG_DOCS" in sql
    assert "CREATE STAGE IF NOT EXISTS KG_DB.GRAPH.KG_LOAD_STAGE" in sql
    assert "CREATE STAGE IF NOT EXISTS KG_DB.GRAPH.KG_SERVICE_SPECS" in sql
    assert "CREATE IMAGE REPOSITORY IF NOT EXISTS KG_DB.GRAPH.KG_IMAGES" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_JOB" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_NODE" in sql
    assert sql.index("USE SCHEMA GRAPH") < sql.index("CREATE TABLE IF NOT EXISTS KG_JOB")


def test_render_snowflake_image_reference_accepts_slash_repository_path() -> None:
    settings = _settings({"snowflake": {"image_repository": "/KG_DB/GRAPH/KG_IMAGES"}})

    assert render_snowflake_image_reference(settings) == (
        "/KG_DB/GRAPH/KG_IMAGES/flakegraph:latest"
    )


def test_render_snowflake_image_reference_appends_digest_when_configured() -> None:
    settings = _settings({"snowflake": {"image_digest": _DIGEST}})

    assert render_snowflake_image_reference(settings) == (
        f"/KG_DB/GRAPH/KG_IMAGES/flakegraph:latest@{_DIGEST}"
    )


def test_render_snowflake_image_reference_rejects_unsafe_repository() -> None:
    settings = _settings({"snowflake": {"image_repository": "KG_DB.GRAPH.BAD;DROP"}})

    with pytest.raises(ValueError, match="image repository"):
        render_snowflake_image_reference(settings)


def test_render_execute_job_service_sql_rejects_unsafe_spec_file() -> None:
    settings = _settings()

    with pytest.raises(ValueError, match="service spec file"):
        render_execute_job_service_sql(settings, "../job.yaml")

    with pytest.raises(ValueError, match="service spec file"):
        render_execute_job_service_sql(
            settings,
            "@KG_DB.GRAPH.KG_SERVICE_SPECS/../job.yaml",
        )


def test_render_kubernetes_job_rejects_invalid_parallelism() -> None:
    with pytest.raises(ValueError, match="parallelism"):
        render_kubernetes_job_yaml(_settings(), parallelism=0)


def _settings(overrides: dict[str, Any] | None = None) -> Settings:
    base: dict[str, Any] = {
        "runtime": {"runtime": "spcs"},
        "job": {
            "job_id": "job-123",
            "graph_id": "graph-123",
            "use_file_queue": True,
            "lease_owner": "worker-1",
            "lease_seconds": 900,
            "file_batch_size": 100,
        },
        "files": {"source": "snowflake_stage", "stage_prefix": "incoming"},
        "ocr": {"provider": "snowflake_cortex"},
        "llm": {"provider": "snowflake_cortex", "model": "llama3.3-70b"},
        "embedding": {
            "provider": "snowflake_cortex",
            "model": "snowflake-arctic-embed-l-v2.0",
            "dimension": 1024,
        },
        "writer": {"provider": "snowflake_bulk"},
        "cache": {"provider": "snowflake"},
        "snowflake": {
            "account": "EXAMPLE_ACCOUNT",
            "database": "KG_DB",
            "schema": "GRAPH",
            "role": "KG_PROCESSOR_ROLE",
            "warehouse": "KG_PROCESSOR_WH",
            "stage": "@KG_DB.GRAPH.KG_DOCS",
            "bulk_stage": "@KG_DB.GRAPH.KG_LOAD_STAGE",
            "image_repository": "KG_DB.GRAPH.KG_IMAGES",
            "image_name": "flakegraph:latest",
            "compute_pool": "KG_PROCESSOR_CPU_POOL",
            "service_name": "KG_PROCESSOR_JOB",
            "service_spec_stage": "@KG_DB.GRAPH.KG_SERVICE_SPECS",
        },
    }
    return Settings.load(overrides=_deep_update(base, overrides or {}))


def test_spcs_runtime_configuration_reloads_through_settings() -> None:
    """The worker reloads this payload, so it must satisfy Settings on the way back.

    Settings forbids extra keys and declares ``schema`` as the alias for
    ``schema_name``. A payload dumped with field names instead of aliases is
    rejected inside the container before any work is claimed, which surfaces only
    as a failed job service rather than a local test failure.
    """

    settings = _settings({"snowflake": {"store_temporary_credential": True}})

    spec = yaml.safe_load(render_spcs_service_spec_yaml(settings))
    payload = json.loads(spec["spec"]["containers"][0]["env"]["KG_CONFIG_JSON"])

    assert "schema" in payload["snowflake"]
    assert "schema_name" not in payload["snowflake"]
    assert "store_temporary_credential" not in payload["snowflake"]

    reloaded = Settings.model_validate(payload)

    assert reloaded.snowflake.schema_name == settings.snowflake.schema_name


def test_spcs_spec_inlines_the_ontology_profile() -> None:
    """A profile path resolved on the author's machine dangles inside SPCS.

    The container receives only the staged specification and the image, so a
    dataset-relative ontology path makes the worker abort before claiming any
    file. The spec must therefore carry the profile itself.
    """

    settings = _settings({"ontology": {"profile_path": "data/martial_arts/ontology.yaml"}})

    spec = yaml.safe_load(render_spcs_service_spec_yaml(settings))
    payload = json.loads(spec["spec"]["containers"][0]["env"]["KG_CONFIG_JSON"])
    ontology = payload["ontology"]

    assert ontology["profile_path"] is None
    assert ontology["profile"]["name"] == "martial-arts-history"
    assert ontology["profile"]["entity_types"]

    reloaded = Settings.model_validate(payload)

    assert reloaded.ontology.profile is not None
    assert reloaded.ontology.profile_path is None


def test_spcs_spec_leaves_a_pathless_ontology_alone() -> None:
    """Generic profiles carry no ontology file and must stay untouched."""

    spec = yaml.safe_load(render_spcs_service_spec_yaml(_settings()))
    payload = json.loads(spec["spec"]["containers"][0]["env"]["KG_CONFIG_JSON"])

    assert payload["ontology"]["profile_path"] is None
    assert payload["ontology"]["profile"] is None


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        (
            {
                "files": {"source": "azure_blob"},
                "azure_blob": {
                    "account_url": "https://storage.example/?sv=2024&sig=SECRET",
                    "container": "documents",
                },
            },
            "azure_blob.account_url",
        ),
        (
            {"llm": {"provider": "vllm_local", "endpoint": "http://user:pw@vllm:8000/v1"}},
            "llm.endpoint",
        ),
        (
            {
                "distributed": {
                    "artifact_uri": "s3://bucket/prefix",
                    "artifact_endpoint_url": "https://minio.example/?api_key=SECRET",
                }
            },
            "distributed.artifact_endpoint_url",
        ),
    ],
)
def test_spcs_spec_refuses_a_credential_carried_inside_a_url(
    overrides: dict[str, Any], field: str
) -> None:
    """Secret fields are excluded by name, but a URL can carry one anyway.

    A staged spec is readable by every role that can list the stage, so a SAS
    signature, userinfo password or token query parameter inside an endpoint
    is refused with the field named, rather than staged verbatim or redacted
    into an endpoint the worker cannot use.
    """

    with pytest.raises(ValueError, match=field):
        render_spcs_service_spec_yaml(_settings(overrides))


def test_kubernetes_job_refuses_a_credential_carried_inside_a_url() -> None:
    settings = _settings(
        {
            "runtime": {"runtime": "onprem"},
            "files": {"source": "azure_blob"},
            "azure_blob": {
                "account_url": "https://storage.example/?sig=SECRET",
                "container": "documents",
            },
        }
    )

    with pytest.raises(ValueError, match="KG_AZURE_BLOB_ACCOUNT_URL"):
        render_kubernetes_job_yaml(settings)


def test_spcs_spec_keeps_a_plain_url_verbatim() -> None:
    """Refusal is only for credentials; an ordinary endpoint must pass unchanged."""

    settings = _settings(
        {"llm": {"provider": "vllm_local", "endpoint": "http://vllm.internal:8000/v1"}}
    )

    spec = yaml.safe_load(render_spcs_service_spec_yaml(settings))
    payload = json.loads(spec["spec"]["containers"][0]["env"]["KG_CONFIG_JSON"])
    assert payload["llm"]["endpoint"] == "http://vllm.internal:8000/v1"
