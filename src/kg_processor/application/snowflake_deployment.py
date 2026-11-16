# SPDX-License-Identifier: Apache-2.0
"""Render Snowflake/SPCS and on-prem deployment artifacts from typed settings."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from kg_processor.adapters.snowflake import (
    quote_sql_string,
    split_stage_uri,
    validate_stage_location,
)
from kg_processor.application.redaction import credentialed_urls
from kg_processor.application.snowflake_schema import render_snowflake_schema_sql
from kg_processor.config.settings import Settings

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_IMAGE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*(?::[A-Za-z0-9_.-]+)?$")
_IMAGE_DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_STAGE_FILE_RE = re.compile(r"^[A-Za-z0-9_./=$-]+\.ya?ml$")
_MAX_QUALIFIED_IDENTIFIER_PARTS = 3
_IMAGE_REPOSITORY_PARTS = 3
_KUBERNETES_NAME_RE = re.compile(r"[^a-z0-9-]+")


def render_spcs_service_spec_yaml(
    settings: Settings,
    config_path: str = "configs/snowflake-cortex.yaml",
) -> str:
    """Render the YAML spec used by Snowpark Container Services job execution."""

    image = render_snowflake_image_reference(settings)
    env = _spcs_environment(settings, config_path)
    container: dict[str, Any] = {
        "name": "flakegraph",
        "image": image,
        "args": ["worker", "--config", config_path],
        "env": env,
        "resources": _resources(settings, settings.snowflake.service_gpu_count),
    }
    spec = {"spec": {"containers": [container]}}
    return yaml.safe_dump(spec, sort_keys=False)


def render_execute_job_service_sql(
    settings: Settings,
    spec_file: str = "flakegraph-job.yaml",
    async_execution: bool = False,
) -> str:
    """Render SQL that executes an SPCS job service from a staged spec file."""

    compute_pool = _required_identifier(settings.snowflake.compute_pool, "compute pool")
    service_name = _service_identifier(settings)
    spec_stage, safe_spec_file = _stage_spec_location(settings, spec_file)
    async_clause = "\n  ASYNC = TRUE" if async_execution else ""
    return (
        "EXECUTE JOB SERVICE\n"
        f"  IN COMPUTE POOL {compute_pool}\n"
        f"  NAME = {service_name}"
        f"{async_clause}\n"
        f"  FROM {spec_stage}\n"
        f"  SPEC = {quote_sql_string(safe_spec_file)};"
    )


def render_kubernetes_job_yaml(
    settings: Settings,
    *,
    image: str = "flakegraph:latest",
    config_path: str = "configs/onprem-azure-blob-vllm-mineru-oss.yaml",
    secret_name: str | None = None,
    parallelism: int = 1,
    completions: int | None = None,
    gpu_count: int | None = None,
) -> str:
    """Render an on-prem Kubernetes Job manifest for the same worker image."""

    if parallelism <= 0:
        raise ValueError("Kubernetes parallelism must be positive")
    resolved_completions = completions if completions is not None else parallelism
    if resolved_completions <= 0:
        raise ValueError("Kubernetes completions must be positive")
    resolved_gpu_count = (
        gpu_count if gpu_count is not None else settings.snowflake.service_gpu_count
    )
    if resolved_gpu_count < 0:
        raise ValueError("Kubernetes gpu_count must be non-negative")
    if not _IMAGE_NAME_RE.fullmatch(image):
        raise ValueError(f"Kubernetes image must be a safe OCI image reference, got: {image}")

    resources = _resources(settings, resolved_gpu_count)
    container: dict[str, Any] = {
        "name": "flakegraph",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "args": ["worker", "--config", config_path],
        "env": _env_list(_kubernetes_environment(settings, config_path)),
        "resources": resources,
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "runAsNonRoot": True,
            "runAsUser": 1000,
            "runAsGroup": 1000,
            "capabilities": {"drop": ["ALL"]},
        },
        "volumeMounts": [
            {"name": "kg-output", "mountPath": "/app/out"},
            {"name": "kg-tmp", "mountPath": "/tmp"},
            {"name": "mineru-cache", "mountPath": "/home/kgprocessor/.cache/mineru"},
        ],
    }
    if secret_name:
        container["envFrom"] = [{"secretRef": {"name": _kubernetes_name(secret_name)}}]

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": _kubernetes_name(f"flakegraph-{settings.job.job_id}")},
        "spec": {
            "parallelism": parallelism,
            "completions": resolved_completions,
            "backoffLimit": 3,
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "flakegraph",
                        "app.kubernetes.io/instance": _kubernetes_name(settings.job.job_id),
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "securityContext": {
                        "runAsNonRoot": True,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "containers": [container],
                    "volumes": [
                        {"name": "kg-output", "emptyDir": {}},
                        {"name": "kg-tmp", "emptyDir": {}},
                        {"name": "mineru-cache", "emptyDir": {}},
                    ],
                },
            },
        },
    }
    return yaml.safe_dump(manifest, sort_keys=False)


def render_snowflake_setup_sql(
    settings: Settings,
    role_name: str = "KG_PROCESSOR_ROLE",
) -> str:
    """Render admin setup SQL for Snowflake roles, grants, objects, and DDL."""

    role = _required_identifier(role_name, "role")
    database = _required_identifier(settings.snowflake.database, "database")
    schema = _required_identifier(settings.snowflake.schema_name, "schema")
    target_schema = f"{database}.{schema}"
    statements: list[str] = [
        f"CREATE ROLE IF NOT EXISTS {role}",
        f"CREATE DATABASE IF NOT EXISTS {database}",
        f"CREATE SCHEMA IF NOT EXISTS {target_schema}",
        *_target_context_statements(database, schema),
    ]

    if settings.snowflake.warehouse:
        warehouse = _qualified_identifier(settings.snowflake.warehouse, "warehouse")
        statements.append(f"GRANT USAGE ON WAREHOUSE {warehouse} TO ROLE {role}")
    statements.extend(
        [
            f"GRANT USAGE ON DATABASE {database} TO ROLE {role}",
            f"GRANT USAGE ON SCHEMA {target_schema} TO ROLE {role}",
            f"GRANT CREATE TABLE ON SCHEMA {target_schema} TO ROLE {role}",
            f"GRANT CREATE STAGE ON SCHEMA {target_schema} TO ROLE {role}",
            f"GRANT CREATE IMAGE REPOSITORY ON SCHEMA {target_schema} TO ROLE {role}",
            f"GRANT CREATE SERVICE ON SCHEMA {target_schema} TO ROLE {role}",
            "GRANT SELECT, INSERT, UPDATE, DELETE "
            f"ON ALL TABLES IN SCHEMA {target_schema} TO ROLE {role}",
            "GRANT SELECT, INSERT, UPDATE, DELETE "
            f"ON FUTURE TABLES IN SCHEMA {target_schema} TO ROLE {role}",
            f"GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE {role}",
            f"GRANT USE AI FUNCTIONS ON ACCOUNT TO ROLE {role}",
        ]
    )

    if settings.snowflake.compute_pool:
        compute_pool = _qualified_identifier(settings.snowflake.compute_pool, "compute pool")
        if settings.snowflake.compute_pool_instance_family:
            family = _required_identifier(
                settings.snowflake.compute_pool_instance_family,
                "compute pool instance family",
            )
            statements.append(
                "CREATE COMPUTE POOL IF NOT EXISTS "
                f"{compute_pool}\n"
                f"  MIN_NODES = {settings.snowflake.compute_pool_min_nodes}\n"
                f"  MAX_NODES = {settings.snowflake.compute_pool_max_nodes}\n"
                f"  INSTANCE_FAMILY = {family}\n"
                "  AUTO_RESUME = TRUE\n"
                "  INITIALLY_SUSPENDED = TRUE\n"
                "  AUTO_SUSPEND_SECS = 60"
            )
        statements.append(f"GRANT USAGE ON COMPUTE POOL {compute_pool} TO ROLE {role}")

    for stage_object in _stage_objects(settings):
        statements.append(f"CREATE STAGE IF NOT EXISTS {stage_object}")
        statements.append(f"GRANT READ ON STAGE {stage_object} TO ROLE {role}")
        statements.append(f"GRANT WRITE ON STAGE {stage_object} TO ROLE {role}")

    if settings.snowflake.image_repository:
        image_repository = _image_repository_identifier(settings.snowflake.image_repository)
        statements.append(f"CREATE IMAGE REPOSITORY IF NOT EXISTS {image_repository}")
        statements.append(f"GRANT READ ON IMAGE REPOSITORY {image_repository} TO ROLE {role}")
        statements.append(f"GRANT WRITE ON IMAGE REPOSITORY {image_repository} TO ROLE {role}")

    setup_sql = ";\n\n".join(statements) + ";"
    schema_sql = render_snowflake_schema_sql(settings.embedding.dimension).strip()
    return f"{setup_sql}\n\n{schema_sql}"


def render_snowflake_objects_sql(settings: Settings) -> str:
    """Render object-only setup SQL for an already-approved Snowflake schema.

    Production Snowflake setup usually has two actors: an admin role that grants
    database, Cortex, compute-pool, and service privileges, and a schema-owning
    role that creates ordinary schema objects. This renderer is for the second
    path. It deliberately omits role DDL, grants, and compute-pool DDL so it can
    be run by a lower-privilege schema owner without tripping account-admin
    statements before tables and stages are created.
    """

    database = _required_identifier(settings.snowflake.database, "database")
    schema = _required_identifier(settings.snowflake.schema_name, "schema")
    statements = [
        *_target_context_statements(database, schema),
        *_schema_object_statements(settings),
    ]
    object_sql = ";\n\n".join(statements) + ";"
    schema_sql = render_snowflake_schema_sql(settings.embedding.dimension).strip()
    return f"{object_sql}\n\n{schema_sql}"


def render_snowflake_image_reference(settings: Settings) -> str:
    """Render the immutable or tagged image reference used by SPCS specs."""

    repository = _required(settings.snowflake.image_repository, "image repository")
    image_name = _image_name(settings.snowflake.image_name)
    image_digest = _image_digest_suffix(settings.snowflake.image_digest)
    repository_path = _image_repository_path(repository)
    return f"{repository_path}/{image_name}{image_digest}"


def _spcs_environment(settings: Settings, config_path: str) -> dict[str, str]:
    runtime_configuration = _spcs_runtime_configuration(settings)
    _reject_credentialed_urls(runtime_configuration, "SPCS service spec")
    env: dict[str, str] = {
        "KG_CONFIG_FILE": config_path,
        "KG_CONFIG_JSON": json.dumps(
            runtime_configuration,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "KG_RUNTIME": "spcs",
        "KG_JOB_ID": settings.job.job_id,
        "KG_GRAPH_ID": settings.job.graph_id,
        "KG_JOB_USE_LEASE": _bool_env(settings.job.use_lease),
        "KG_JOB_USE_FILE_QUEUE": _bool_env(settings.job.use_file_queue),
        "KG_JOB_LEASE_SECONDS": str(settings.job.lease_seconds),
        "KG_BATCH_FILES": str(settings.job.file_batch_size),
        "KG_FILE_SOURCE": settings.files.source,
        "KG_OCR_PROVIDER": settings.ocr.provider,
        "KG_LLM_PROVIDER": settings.llm.provider,
        "KG_LLM_MODEL": settings.llm.model,
        "KG_EMBED_PROVIDER": settings.embedding.provider,
        "KG_EMBED_MODEL": settings.embedding.model,
        "KG_EMBED_DIM": str(settings.embedding.dimension),
        "KG_WRITER": settings.writer.provider,
        "KG_CACHE_PROVIDER": settings.cache.provider,
        "KG_SNOWFLAKE_BULK_TARGET_FILE_MB": str(settings.snowflake.bulk_target_file_size_mb),
        # SPCS injects a short-lived Snowflake OAuth token at runtime. The
        # service spec should therefore avoid embedding a password or browser
        # authenticator, even when local access checks use externalbrowser.
        "KG_SNOWFLAKE_AUTHENTICATOR": "oauth",
        "KG_SNOWFLAKE_OAUTH_TOKEN_PATH": str(
            settings.snowflake.oauth_token_path or "/snowflake/session/token"
        ),
    }
    optional = {
        "KG_JOB_LEASE_OWNER": settings.job.lease_owner,
        "KG_STAGE_PREFIX": settings.files.stage_prefix,
        # Snowflake injects service-specific SNOWFLAKE_ACCOUNT and
        # SNOWFLAKE_HOST values that are cryptographically paired with the OAuth
        # token in /snowflake/session/token. External client identifiers can be
        # different, so propagating them into SPCS makes an otherwise valid
        # service token fail authentication. Settings loads Snowflake's ambient
        # variables when the KG_* overrides are absent.
        "KG_SNOWFLAKE_DATABASE": settings.snowflake.database,
        "KG_SNOWFLAKE_SCHEMA": settings.snowflake.schema_name,
        "KG_SNOWFLAKE_ROLE": settings.snowflake.role,
        "KG_SNOWFLAKE_WAREHOUSE": settings.snowflake.warehouse,
        "KG_SNOWFLAKE_STAGE": settings.snowflake.stage,
        "KG_SNOWFLAKE_BULK_STAGE": settings.snowflake.bulk_stage,
        "KG_SNOWFLAKE_IMAGE_REPOSITORY": settings.snowflake.image_repository,
        "KG_SNOWFLAKE_IMAGE_NAME": settings.snowflake.image_name,
        "KG_SNOWFLAKE_IMAGE_DIGEST": settings.snowflake.image_digest,
    }
    env.update({key: str(value) for key, value in optional.items() if value not in {None, ""}})
    _reject_credentialed_urls(env, "SPCS service spec")
    return env


def _reject_credentialed_urls(payload: dict[str, Any], artifact: str) -> None:
    """Refuse to render a spec that would stage a credential inside a URL.

    Secret-valued fields are excluded from the payload by name, but a URL field
    can smuggle one anyway: ``https://user:pass@host``, a blob URL with its SAS
    token attached, or an endpoint with ``?api_key=`` on it. A spec is staged
    and read by anyone who can see the stage, so such a URL is refused rather
    than redacted, because a redacted endpoint would only fail later on the
    worker with a less useful error.
    """

    offending = credentialed_urls(payload)
    if offending:
        raise ValueError(
            f"{artifact} would embed a credential carried inside a URL; move it into a "
            f"secret or an environment variable instead: {', '.join(sorted(offending))}"
        )


def _spcs_runtime_configuration(settings: Settings) -> dict[str, Any]:
    """Serialize effective worker behavior without local client credentials.

    App-selected processing settings must override the portable YAML baked into
    the image. Snowflake injects a scoped OAuth token into SPCS, so local login
    material and external-provider secrets are excluded from the staged spec.
    """

    payload = settings.model_dump(
        mode="json",
        # The worker reloads this payload through Settings, which forbids extra
        # keys and declares `schema` as the alias for `schema_name`. Dumping
        # field names instead of aliases produces a payload the worker rejects
        # before it can claim any work.
        by_alias=True,
        exclude={
            "distributed": {
                "database_url",
                "artifact_access_key_id",
                "artifact_secret_access_key",
            },
            "azure_blob": {"connection_string", "sas_token"},
            "ocr": {"mineru_api_key"},
            "generic_http_ocr": {"api_key"},
            "llm": {"api_key"},
            "embedding": {"api_key"},
            "snowflake": {
                "account",
                "host",
                "user",
                "password",
                "authenticator",
                "private_key_path",
                "oauth_token",
                "oauth_token_path",
                # Caching browser-SSO tokens is a workstation concern. SPCS
                # authenticates with its mounted OAuth token and has no
                # credential store to write to.
                "store_temporary_credential",
            },
        },
    )
    _inline_ontology_profile(payload, settings.ontology.profile_path)
    return payload


def _inline_ontology_profile(payload: dict[str, Any], profile_path: Path | None) -> None:
    """Replace an ontology file reference in a run payload with the profile itself.

    ``profile_path`` is resolved against the author's filesystem. The container
    only receives this payload and the image, so a dataset-relative path dangles
    and the worker aborts before claiming any file. Reading the profile here
    keeps the promise that a staged specification is complete on its own. The
    application ships without this package and carries its own copy, pinned to
    this one by a parity test.
    """

    ontology = payload.get("ontology")
    if not isinstance(ontology, dict):
        return
    if ontology.get("profile") is not None:
        ontology["profile_path"] = None
        return
    if profile_path is None:
        return
    with profile_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Ontology file must contain a mapping: {profile_path}")
    ontology["profile"] = raw
    # The path would not resolve inside the container and must not be retried.
    ontology["profile_path"] = None


def _kubernetes_environment(settings: Settings, config_path: str) -> dict[str, str]:
    env: dict[str, str] = {
        "KG_CONFIG_FILE": config_path,
        "KG_RUNTIME": "onprem",
        "KG_JOB_ID": settings.job.job_id,
        "KG_GRAPH_ID": settings.job.graph_id,
        "KG_JOB_USE_LEASE": _bool_env(settings.job.use_lease),
        "KG_JOB_USE_FILE_QUEUE": _bool_env(settings.job.use_file_queue),
        "KG_JOB_LEASE_SECONDS": str(settings.job.lease_seconds),
        "KG_BATCH_FILES": str(settings.job.file_batch_size),
        "KG_FILE_SOURCE": settings.files.source,
        "KG_OCR_PROVIDER": settings.ocr.provider,
        "KG_LLM_PROVIDER": settings.llm.provider,
        "KG_LLM_MODEL": settings.llm.model,
        "KG_EMBED_PROVIDER": settings.embedding.provider,
        "KG_EMBED_MODEL": settings.embedding.model,
        "KG_EMBED_DIM": str(settings.embedding.dimension),
        "KG_EMBED_BATCH_SIZE": str(settings.embedding.batch_size),
        "KG_WRITER": settings.writer.provider,
        "KG_CACHE_PROVIDER": settings.cache.provider,
        "KG_SNOWFLAKE_BULK_TARGET_FILE_MB": str(settings.snowflake.bulk_target_file_size_mb),
    }
    optional = {
        "KG_JOB_LEASE_OWNER": settings.job.lease_owner,
        "KG_INPUT_PATH": settings.files.input_path,
        "KG_AZURE_BLOB_ACCOUNT_URL": settings.azure_blob.account_url,
        "KG_AZURE_BLOB_CONTAINER": settings.azure_blob.container,
        "KG_AZURE_BLOB_PREFIX": settings.azure_blob.prefix,
        "KG_AZURE_BLOB_DOWNLOAD_PATH": settings.azure_blob.download_path,
        "KG_MINERU_COMMAND": settings.ocr.mineru_command,
        "KG_MINERU_METHOD": settings.ocr.mineru_method,
        "KG_MINERU_BACKEND": settings.ocr.mineru_backend,
        "KG_MINERU_EFFORT": settings.ocr.mineru_effort,
        "KG_OCR_MODEL_CACHE_DIR": settings.ocr.model_cache_dir,
        "KG_LLM_ENDPOINT": settings.llm.endpoint,
        "KG_LLM_API_VERSION": settings.llm.api_version,
        "KG_EMBED_ENDPOINT": settings.embedding.endpoint,
        "KG_EMBED_API_VERSION": settings.embedding.api_version,
        "KG_EMBED_DEVICE": settings.embedding.device,
        "KG_OUTPUT_PATH": settings.writer.output_path,
        "KG_CACHE_PATH": settings.cache.path,
        "KG_SNOWFLAKE_ACCOUNT": settings.snowflake.account,
        "KG_SNOWFLAKE_HOST": settings.snowflake.host,
        "KG_SNOWFLAKE_USER": settings.snowflake.user,
        "KG_SNOWFLAKE_AUTHENTICATOR": settings.snowflake.authenticator,
        "KG_SNOWFLAKE_DATABASE": settings.snowflake.database,
        "KG_SNOWFLAKE_SCHEMA": settings.snowflake.schema_name,
        "KG_SNOWFLAKE_ROLE": settings.snowflake.role,
        "KG_SNOWFLAKE_WAREHOUSE": settings.snowflake.warehouse,
        "KG_SNOWFLAKE_BULK_STAGE": settings.snowflake.bulk_stage,
    }
    env.update({key: str(value) for key, value in optional.items() if value not in {None, ""}})
    _reject_credentialed_urls(env, "Kubernetes job manifest")
    return env


def _resources(settings: Settings, gpu_count: int) -> dict[str, dict[str, str]]:
    requests = {
        "cpu": settings.snowflake.service_cpu_request,
        "memory": settings.snowflake.service_memory_request,
    }
    limits = {
        "cpu": settings.snowflake.service_cpu_limit,
        "memory": settings.snowflake.service_memory_limit,
    }
    if gpu_count:
        gpu = str(gpu_count)
        requests["nvidia.com/gpu"] = gpu
        limits["nvidia.com/gpu"] = gpu
    return {"requests": requests, "limits": limits}


def _env_list(values: dict[str, str]) -> list[dict[str, str]]:
    return [{"name": name, "value": value} for name, value in sorted(values.items())]


def _stage_objects(settings: Settings) -> list[str]:
    values = [
        settings.snowflake.stage,
        settings.snowflake.bulk_stage,
        settings.snowflake.service_spec_stage,
    ]
    stage_objects: list[str] = []
    for value in values:
        if not value:
            continue
        stage_object = _stage_object_identifier(value)
        if stage_object not in stage_objects:
            stage_objects.append(stage_object)
    return stage_objects


def _schema_object_statements(settings: Settings) -> list[str]:
    statements = [f"CREATE STAGE IF NOT EXISTS {stage}" for stage in _stage_objects(settings)]
    if settings.snowflake.image_repository:
        statements.append(
            "CREATE IMAGE REPOSITORY IF NOT EXISTS "
            f"{_image_repository_identifier(settings.snowflake.image_repository)}"
        )
    return statements


def _target_context_statements(database: str, schema: str) -> list[str]:
    # The reusable schema renderer intentionally emits unqualified table names
    # because live writers already connect with an explicit database/schema.
    # Standalone Snowflake scripts must pin the same target context before
    # appending those table statements.
    return [f"USE DATABASE {database}", f"USE SCHEMA {schema}"]


def _stage_object_identifier(stage: str) -> str:
    validated = validate_stage_location(stage)
    without_at = validated[1:]
    stage_object = without_at.split("/", 1)[0]
    return _qualified_identifier(stage_object, "stage")


def _required_stage(value: str | None, label: str) -> str:
    return validate_stage_location(_required(value, label))


def _required_identifier(value: str | None, label: str) -> str:
    return _identifier(_required(value, label), label)


def _qualified_identifier(value: str | None, label: str) -> str:
    raw = _required(value, label)
    parts = raw.split(".")
    if not 1 <= len(parts) <= _MAX_QUALIFIED_IDENTIFIER_PARTS:
        raise ValueError(f"{label} must be an unquoted identifier or 2/3-part identifier")
    return ".".join(_identifier(part, label) for part in parts)


def _service_identifier(settings: Settings) -> str:
    """Return a standalone-safe service identifier for generated execution SQL.

    `execute-job-sql` is commonly applied in a fresh CLI session without current
    database or schema context. A concise configured service name is therefore
    qualified with the same target database and schema used by the worker tables.
    Explicit two- or three-part service names remain available for advanced setups.
    """

    raw = _required(settings.snowflake.service_name, "service name")
    if "." in raw:
        return _qualified_identifier(raw, "service name")
    database = _required_identifier(settings.snowflake.database, "database")
    schema = _required_identifier(settings.snowflake.schema_name, "schema")
    name = _required_identifier(raw, "service name")
    return f"{database}.{schema}.{name}"


def _image_repository_identifier(value: str) -> str:
    normalized = value[1:] if value.startswith("/") else value
    normalized = normalized.replace("/", ".")
    parts = normalized.split(".")
    if len(parts) != _IMAGE_REPOSITORY_PARTS:
        raise ValueError("image repository must be a 3-part identifier")
    return ".".join(_identifier(part, "image repository") for part in parts)


def _image_repository_path(value: str) -> str:
    identifier = _image_repository_identifier(value)
    return "/" + "/".join(identifier.split("."))


def _identifier(value: str, label: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{label} must be an unquoted Snowflake identifier, got: {value}")
    return value


def _image_name(value: str) -> str:
    if "/" in value or ".." in value or not _IMAGE_NAME_RE.fullmatch(value):
        raise ValueError(
            f"image name must be a safe OCI image name with optional tag, got: {value}"
        )
    return value


def _image_digest_suffix(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    if not _IMAGE_DIGEST_RE.fullmatch(normalized):
        raise ValueError("image digest must be sha256:<64 hex chars>")
    return f"@{normalized}"


def _stage_spec_file(value: str) -> str:
    parts = value.split("/")
    if (
        not _STAGE_FILE_RE.fullmatch(value)
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"service spec file must be a safe YAML file path, got: {value}")
    return value


def _stage_spec_location(settings: Settings, spec_file: str) -> tuple[str, str]:
    """Resolve the service-spec stage and relative YAML file path.

    The CLI accepts the concise `flakegraph-job.yaml` form used by generated
    docs and the fully staged `@DB.SCHEMA.STAGE/flakegraph-job.yaml` form that
    operators naturally copy from `LIST @stage` output. In both cases Snowflake's
    `EXECUTE JOB SERVICE` syntax still wants `FROM @stage` plus a relative
    `SPEC = 'path.yaml'`, so this helper validates and normalizes that boundary.
    """

    if spec_file.startswith("@"):
        stage, relative_path = split_stage_uri(spec_file)
        return stage, _stage_spec_file(relative_path)
    spec_stage = _required_stage(settings.snowflake.service_spec_stage, "service spec stage")
    return spec_stage, _stage_spec_file(spec_file)


def _required(value: str | None, label: str) -> str:
    if not value:
        raise ValueError(f"Snowflake {label} is required")
    return value


def _bool_env(value: bool) -> str:
    return "true" if value else "false"


def _kubernetes_name(value: str) -> str:
    lowered = value.strip().lower()
    normalized = _KUBERNETES_NAME_RE.sub("-", lowered).strip("-")
    if not normalized:
        raise ValueError("Kubernetes resource name must not be blank")
    return normalized[:63].rstrip("-")
