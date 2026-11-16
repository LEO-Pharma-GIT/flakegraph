from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from types import NoneType, UnionType
from typing import Literal, Union, get_args, get_origin

import pytest

from kg_processor.config.settings import Settings

# Every environment variable a deployment may already set, paired with the
# settings key it fills. Charts, service specs, and operator shells rely on
# these names, so a row leaves only when its consumers have moved off it.
ENVIRONMENT_NAMES: tuple[tuple[str, str], ...] = (
    ("KG_RUNTIME", "runtime.runtime"),
    ("KG_JOB_ID", "job.job_id"),
    ("KG_GRAPH_ID", "job.graph_id"),
    ("KG_JOB_USE_LEASE", "job.use_lease"),
    ("KG_JOB_USE_FILE_QUEUE", "job.use_file_queue"),
    ("KG_JOB_LEASE_OWNER", "job.lease_owner"),
    ("KG_JOB_LEASE_SECONDS", "job.lease_seconds"),
    ("KG_BATCH_FILES", "job.file_batch_size"),
    ("KG_DISTRIBUTED_DATABASE_URL", "distributed.database_url"),
    ("KG_DISTRIBUTED_WORKER_ID", "distributed.worker_id"),
    ("KG_DISTRIBUTED_WORKER_STAGES", "distributed.worker_stages"),
    ("KG_DISTRIBUTED_WORKER_SLOTS", "distributed.worker_slots"),
    ("KG_DISTRIBUTED_LEASE_SECONDS", "distributed.lease_seconds"),
    ("KG_DISTRIBUTED_POLL_INTERVAL_SECONDS", "distributed.poll_interval_seconds"),
    ("KG_DISTRIBUTED_RETRY_DELAY_SECONDS", "distributed.retry_delay_seconds"),
    ("KG_DISTRIBUTED_MAX_ATTEMPTS", "distributed.max_attempts"),
    ("KG_DISTRIBUTED_ARTIFACT_COMPRESSION_LEVEL", "distributed.artifact_compression_level"),
    ("KG_DISTRIBUTED_MAX_ARTIFACT_BYTES", "distributed.max_artifact_bytes"),
    ("KG_DISTRIBUTED_ARTIFACT_URI", "distributed.artifact_uri"),
    ("KG_DISTRIBUTED_ARTIFACT_ENDPOINT_URL", "distributed.artifact_endpoint_url"),
    ("KG_DISTRIBUTED_ARTIFACT_ACCESS_KEY_ID", "distributed.artifact_access_key_id"),
    ("KG_DISTRIBUTED_ARTIFACT_SECRET_ACCESS_KEY", "distributed.artifact_secret_access_key"),
    ("KG_DISTRIBUTED_ARTIFACT_REGION", "distributed.artifact_region"),
    ("KG_DISTRIBUTED_FINALIZATION_ENGINE", "distributed.finalization_engine"),
    ("KG_DISTRIBUTED_SPARK_MASTER", "distributed.spark_master"),
    ("KG_DISTRIBUTED_SPARK_IMAGE", "distributed.spark_image"),
    ("KG_DISTRIBUTED_SPARK_NAMESPACE", "distributed.spark_namespace"),
    ("KG_DISTRIBUTED_SPARK_SERVICE_ACCOUNT", "distributed.spark_service_account"),
    ("KG_DISTRIBUTED_SPARK_EXECUTOR_POD_TEMPLATE", "distributed.spark_executor_pod_template"),
    ("KG_DISTRIBUTED_SPARK_EXECUTOR_INSTANCES", "distributed.spark_executor_instances"),
    ("KG_DISTRIBUTED_SPARK_EXECUTOR_CORES", "distributed.spark_executor_cores"),
    ("KG_DISTRIBUTED_SPARK_EXECUTOR_MEMORY", "distributed.spark_executor_memory"),
    ("KG_DISTRIBUTED_SPARK_EXECUTOR_MEMORY_OVERHEAD", "distributed.spark_executor_memory_overhead"),
    ("KG_DISTRIBUTED_SPARK_SHUFFLE_PARTITIONS", "distributed.spark_shuffle_partitions"),
    ("KG_FILE_SOURCE", "files.source"),
    ("KG_INPUT_PATH", "files.input_path"),
    ("KG_STAGE_PREFIX", "files.stage_prefix"),
    ("KG_STAGE_CONTENT_HASH", "files.stage_content_hash"),
    ("KG_AZURE_BLOB_ACCOUNT_URL", "azure_blob.account_url"),
    ("KG_AZURE_BLOB_CONNECTION_STRING", "azure_blob.connection_string"),
    ("KG_AZURE_BLOB_CONTAINER", "azure_blob.container"),
    ("KG_AZURE_BLOB_PREFIX", "azure_blob.prefix"),
    ("KG_AZURE_BLOB_SAS_TOKEN", "azure_blob.sas_token"),
    ("KG_AZURE_BLOB_DOWNLOAD_PATH", "azure_blob.download_path"),
    ("KG_S3_BUCKET", "s3.bucket"),
    ("KG_S3_PREFIX", "s3.prefix"),
    ("KG_S3_ENDPOINT_URL", "s3.endpoint_url"),
    ("KG_S3_REGION", "s3.region"),
    ("KG_S3_DOWNLOAD_PATH", "s3.download_path"),
    ("KG_OCR_PROVIDER", "ocr.provider"),
    ("KG_OCR_LANGUAGE", "ocr.language"),
    ("KG_OCR_PAGE_RANGE", "ocr.page_range"),
    ("KG_OCR_MODEL_CACHE_DIR", "ocr.model_cache_dir"),
    ("KG_MINERU_COMMAND", "ocr.mineru_command"),
    ("KG_MINERU_METHOD", "ocr.mineru_method"),
    ("KG_MINERU_BACKEND", "ocr.mineru_backend"),
    ("KG_MINERU_EFFORT", "ocr.mineru_effort"),
    ("KG_MINERU_API_URL", "ocr.mineru_api_url"),
    ("KG_MINERU_API_KEY", "ocr.mineru_api_key"),
    ("KG_TESSERACT_COMMAND", "ocr.tesseract_command"),
    ("KG_GENERIC_HTTP_OCR_ENDPOINT", "generic_http_ocr.endpoint"),
    ("KG_GENERIC_HTTP_OCR_MAX_RESPONSE_BYTES", "generic_http_ocr.max_response_bytes"),
    ("KG_GENERIC_HTTP_OCR_API_KEY", "generic_http_ocr.api_key"),
    ("KG_GENERIC_HTTP_OCR_API_KEY_HEADER", "generic_http_ocr.api_key_header"),
    ("KG_GENERIC_HTTP_OCR_API_KEY_PREFIX", "generic_http_ocr.api_key_prefix"),
    ("KG_GENERIC_HTTP_OCR_FILE_FIELD", "generic_http_ocr.file_field"),
    ("KG_GENERIC_HTTP_OCR_RESULT_PATH", "generic_http_ocr.result_path"),
    ("KG_GENERIC_HTTP_OCR_PAGES_PATH", "generic_http_ocr.pages_path"),
    ("KG_GENERIC_HTTP_OCR_PAGE_NUMBER_PATH", "generic_http_ocr.page_number_path"),
    ("KG_GENERIC_HTTP_OCR_MARKDOWN_PATH", "generic_http_ocr.markdown_path"),
    ("KG_GENERIC_HTTP_OCR_RAW_TEXT_PATH", "generic_http_ocr.raw_text_path"),
    ("KG_GENERIC_HTTP_OCR_BLOCKS_PATH", "generic_http_ocr.blocks_path"),
    ("KG_GENERIC_HTTP_OCR_BLOCK_ID_PATH", "generic_http_ocr.block_id_path"),
    ("KG_GENERIC_HTTP_OCR_BLOCK_KIND_PATH", "generic_http_ocr.block_kind_path"),
    ("KG_GENERIC_HTTP_OCR_BLOCK_TEXT_PATH", "generic_http_ocr.block_text_path"),
    ("KG_GENERIC_HTTP_OCR_BLOCK_BBOX_PATH", "generic_http_ocr.block_bbox_path"),
    ("KG_GENERIC_HTTP_OCR_BLOCK_CONFIDENCE_PATH", "generic_http_ocr.block_confidence_path"),
    ("KG_GENERIC_HTTP_OCR_BLOCK_METADATA_PATH", "generic_http_ocr.block_metadata_path"),
    ("KG_GENERIC_HTTP_OCR_ASSETS_PATH", "generic_http_ocr.assets_path"),
    ("KG_GENERIC_HTTP_OCR_ASSET_ID_PATH", "generic_http_ocr.asset_id_path"),
    ("KG_GENERIC_HTTP_OCR_ASSET_KIND_PATH", "generic_http_ocr.asset_kind_path"),
    ("KG_GENERIC_HTTP_OCR_ASSET_URI_PATH", "generic_http_ocr.asset_uri_path"),
    ("KG_GENERIC_HTTP_OCR_ASSET_PAGE_NUMBER_PATH", "generic_http_ocr.asset_page_number_path"),
    ("KG_GENERIC_HTTP_OCR_ASSET_CONFIDENCE_PATH", "generic_http_ocr.asset_confidence_path"),
    ("KG_GENERIC_HTTP_OCR_ASSET_METADATA_PATH", "generic_http_ocr.asset_metadata_path"),
    ("KG_GENERIC_HTTP_OCR_WARNINGS_PATH", "generic_http_ocr.warnings_path"),
    ("KG_GENERIC_HTTP_OCR_ERROR_PATH", "generic_http_ocr.error_path"),
    ("KG_GENERIC_HTTP_OCR_STATUS_PATH", "generic_http_ocr.status_path"),
    ("KG_LLM_PROVIDER", "llm.provider"),
    ("KG_LLM_ENDPOINT", "llm.endpoint"),
    ("KG_LLM_MODEL", "llm.model"),
    ("KG_LLM_API_KEY", "llm.api_key"),
    ("KG_LLM_API_VERSION", "llm.api_version"),
    ("KG_LLM_TIMEOUT_SECONDS", "llm.timeout_seconds"),
    ("KG_LLM_MAX_OUTPUT_TOKENS", "llm.max_output_tokens"),
    ("KG_EMBED_PROVIDER", "embedding.provider"),
    ("KG_EMBED_ENDPOINT", "embedding.endpoint"),
    ("KG_EMBED_MODEL", "embedding.model"),
    ("KG_EMBED_API_KEY", "embedding.api_key"),
    ("KG_EMBED_API_VERSION", "embedding.api_version"),
    ("KG_EMBED_DIM", "embedding.dimension"),
    ("KG_EMBED_BATCH_SIZE", "embedding.batch_size"),
    ("KG_EMBED_DEVICE", "embedding.device"),
    ("KG_ONTOLOGY_PROFILE", "ontology.profile_path"),
    ("KG_GRAPH_EXTRACTION_WINDOW_TOKENS", "graph.extraction_window_tokens"),
    ("KG_GRAPH_MAX_CHUNKS_PER_LLM_CALL", "graph.max_chunks_per_llm_call"),
    ("KG_GRAPH_EXTRACTION_PARALLELISM", "graph.extraction_parallelism"),
    ("KG_GRAPH_FINALIZATION_PROVIDER_CONCURRENCY", "graph.finalization_provider_concurrency"),
    ("KG_GRAPH_MAX_ENTITIES_PER_BATCH", "graph.max_entities_per_batch"),
    ("KG_GRAPH_MAX_RELATIONS_PER_BATCH", "graph.max_relations_per_batch"),
    ("KG_GRAPH_GLEANING_MAX_PASSES", "graph.gleaning_max_passes"),
    ("KG_GRAPH_GLEANING_MIN_UNCOVERED_TOKENS", "graph.gleaning_min_uncovered_tokens"),
    ("KG_GRAPH_GLEANING_SATURATION_THRESHOLD", "graph.gleaning_saturation_threshold"),
    ("KG_GRAPH_MIN_ENTITY_CONFIDENCE", "graph.min_entity_confidence"),
    ("KG_GRAPH_MIN_RELATION_CONFIDENCE", "graph.min_relation_confidence"),
    ("KG_GRAPH_VERIFY_RELATIONS", "graph.verify_relations"),
    ("KG_GRAPH_VERIFY_ENTITIES", "graph.verify_entities"),
    ("KG_GRAPH_VERIFICATION_MIN_CONFIDENCE", "graph.verification_min_confidence"),
    ("KG_GRAPH_ENTITY_RESOLUTION_ENABLED", "graph.entity_resolution_enabled"),
    ("KG_GRAPH_RESOLUTION_LEXICAL_AUTO_MERGE", "graph.resolution_lexical_auto_merge"),
    ("KG_GRAPH_RESOLUTION_EMBEDDING_AUTO_MERGE", "graph.resolution_embedding_auto_merge"),
    ("KG_GRAPH_RESOLUTION_CANDIDATE_THRESHOLD", "graph.resolution_candidate_threshold"),
    ("KG_GRAPH_RESOLUTION_EMBEDDING_LEXICAL_FLOOR", "graph.resolution_embedding_lexical_floor"),
    (
        "KG_GRAPH_RESOLUTION_MAX_CANDIDATES_PER_MENTION",
        "graph.resolution_max_candidates_per_mention",
    ),
    ("KG_GRAPH_RESOLUTION_ADJUDICATION_BATCH_SIZE", "graph.resolution_adjudication_batch_size"),
    ("KG_GRAPH_RESOLUTION_LLM_MERGE_MIN_CONFIDENCE", "graph.resolution_llm_merge_min_confidence"),
    ("KG_GRAPH_DETERMINISTIC_SEED", "graph.deterministic_seed"),
    ("KG_GRAPH_MIN_ENTITY_NAME_LENGTH", "graph.min_entity_name_length"),
    ("KG_GRAPH_REQUIRE_RELATION_ENDPOINT_GROUNDING", "graph.require_relation_endpoint_grounding"),
    ("KG_GRAPH_ENTITY_BLOCKLIST", "graph.entity_blocklist"),
    ("KG_GRAPH_DESCRIPTION_MERGE_MIN_OBSERVATIONS", "graph.description_merge_min_observations"),
    ("KG_GRAPH_DESCRIPTION_MERGE_MAX_DESCRIPTIONS", "graph.description_merge_max_descriptions"),
    ("KG_GRAPH_DESCRIPTION_MERGE_MAX_EVIDENCE", "graph.description_merge_max_evidence"),
    ("KG_GRAPH_MIN_COMMUNITY_SIZE", "graph.min_community_size"),
    ("KG_GRAPH_MAX_COMMUNITY_SIZE", "graph.max_community_size"),
    ("KG_GRAPH_COMMUNITY_RESOLUTION", "graph.community_resolution"),
    ("KG_GRAPH_COMMUNITY_CO_MENTION_WEIGHT", "graph.community_co_mention_weight"),
    ("KG_GRAPH_FAIL_ON_QUALITY_ERROR", "graph.fail_on_quality_error"),
    ("KG_OUTPUT_PATH", "writer.output_path"),
    ("KG_WRITER", "writer.provider"),
    ("KG_CACHE_PROVIDER", "cache.provider"),
    ("KG_CACHE_PATH", "cache.path"),
    ("KG_SNOWFLAKE_ACCOUNT", "snowflake.account"),
    ("KG_SNOWFLAKE_HOST", "snowflake.host"),
    ("KG_SNOWFLAKE_USER", "snowflake.user"),
    ("KG_SNOWFLAKE_PASSWORD", "snowflake.password"),
    ("KG_SNOWFLAKE_AUTHENTICATOR", "snowflake.authenticator"),
    ("KG_SNOWFLAKE_PRIVATE_KEY_PATH", "snowflake.private_key_path"),
    ("KG_SNOWFLAKE_OAUTH_TOKEN", "snowflake.oauth_token"),
    ("KG_SNOWFLAKE_OAUTH_TOKEN_PATH", "snowflake.oauth_token_path"),
    ("KG_SNOWFLAKE_STORE_TEMPORARY_CREDENTIAL", "snowflake.store_temporary_credential"),
    ("KG_SNOWFLAKE_DATABASE", "snowflake.database"),
    ("KG_SNOWFLAKE_SCHEMA", "snowflake.schema"),
    ("KG_SNOWFLAKE_ROLE", "snowflake.role"),
    ("KG_SNOWFLAKE_WAREHOUSE", "snowflake.warehouse"),
    ("KG_SNOWFLAKE_STAGE", "snowflake.stage"),
    ("KG_SNOWFLAKE_BULK_STAGE", "snowflake.bulk_stage"),
    ("KG_SNOWFLAKE_BULK_TARGET_FILE_MB", "snowflake.bulk_target_file_size_mb"),
    ("KG_SNOWFLAKE_IMAGE_REPOSITORY", "snowflake.image_repository"),
    ("KG_SNOWFLAKE_IMAGE_NAME", "snowflake.image_name"),
    ("KG_SNOWFLAKE_IMAGE_DIGEST", "snowflake.image_digest"),
    ("KG_SNOWFLAKE_COMPUTE_POOL", "snowflake.compute_pool"),
    ("KG_SNOWFLAKE_COMPUTE_POOL_INSTANCE_FAMILY", "snowflake.compute_pool_instance_family"),
    ("KG_SNOWFLAKE_COMPUTE_POOL_MIN_NODES", "snowflake.compute_pool_min_nodes"),
    ("KG_SNOWFLAKE_COMPUTE_POOL_MAX_NODES", "snowflake.compute_pool_max_nodes"),
    ("KG_SNOWFLAKE_SERVICE_NAME", "snowflake.service_name"),
    ("KG_SNOWFLAKE_SERVICE_SPEC_STAGE", "snowflake.service_spec_stage"),
)

# Fields whose validators want a specific vocabulary rather than a type-shaped
# sample, and the value the loader should produce for it.
_VALIDATOR_SHAPED_VALUES: dict[str, tuple[str, object]] = {
    "files.source": ("manifest", "manifest"),
    "ocr.provider": ("tesseract_internal", "tesseract_internal"),
    "llm.provider": ("fake", "fake"),
    "embedding.provider": ("hash", "hash"),
    "writer.provider": ("snowflake_bulk", "snowflake_bulk"),
    "snowflake.image_digest": ("sha256:" + "A" * 64, "sha256:" + "a" * 64),
    "snowflake.authenticator": ("SNOWFLAKE_JWT", "SNOWFLAKE_JWT"),
}

# Fields a model validator checks against a sibling that the sample would trip.
_COMPANION_ENVIRONMENT: dict[str, dict[str, str]] = {
    "snowflake.compute_pool_min_nodes": {"KG_SNOWFLAKE_COMPUTE_POOL_MAX_NODES": "7"},
}


def _sample_environment_value(annotation: object) -> tuple[str, object]:
    """Pick an environment string for a field type and the value it should become."""

    if get_origin(annotation) in (UnionType, Union):
        annotation = next(arg for arg in get_args(annotation) if arg is not NoneType)
    if get_origin(annotation) is Literal:
        first = get_args(annotation)[0]
        return first, first
    if get_origin(annotation) is list:
        (item,) = get_args(annotation)
        if get_origin(item) is Literal:
            items = list(get_args(item))[:2]
        elif isinstance(item, type) and issubclass(item, StrEnum):
            items = list(item)[:2]
        else:
            items = ["a", "b"]
        return ",".join(items), items
    samples: dict[object, tuple[str, object]] = {
        bool: ("true", True),
        int: ("7", 7),
        float: ("0.5", 0.5),
        Path: ("/tmp/x", Path("/tmp/x")),
        str: ("x", "x"),
    }
    return samples[annotation]


def test_settings_defaults_match_local_open_source_runtime_profile() -> None:
    """Lock zero-configuration runtime behavior to real providers and shared quality defaults.

    New users should receive the tested path automatically.
    """

    settings = Settings.load(env={})

    assert settings.runtime.runtime == "local"
    assert settings.files.source == "local"
    # Nothing to install: the text layer a document carries is read directly,
    # and MinerU is an explicit selection for scanned corpora.
    assert settings.ocr.provider == "builtin_text"
    assert settings.ocr.fallback_secondary_provider == "mineru_internal"
    assert settings.llm.provider == "openai_compatible"
    assert settings.llm.timeout_seconds == 180
    assert settings.embedding.provider == "sentence_transformers"
    assert settings.embedding.model == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.embedding.dimension == 1024
    assert settings.embedding.batch_size == 32
    assert settings.graph.chunk_token_size == 500
    assert settings.graph.chunk_token_overlap == 60
    assert settings.graph.extraction_window_tokens == 700
    assert settings.graph.max_chunks_per_llm_call == 2
    assert settings.graph.max_entities_per_batch == 40
    assert settings.graph.max_relations_per_batch == 40
    assert settings.graph.gleaning_max_passes == 1
    assert settings.graph.drop_isolated_entities is False
    assert settings.graph.verify_relations is True
    assert settings.graph.verify_entities is True
    assert settings.graph.extraction_parallelism == 2
    assert settings.graph.finalization_provider_concurrency == 2
    assert settings.distributed.lease_seconds == 300
    assert settings.distributed.worker_slots == 8
    assert settings.distributed.worker_stages == [
        "prepare_document",
        "extract_document_context",
        "extract_entity_window",
        "compact_entity_inventory",
        "extract_relation_window",
        "compact_document",
        "revalidate_document_context",
        "revalidate_entity_window",
        "revalidate_relation_window",
        "revalidate_document",
        "finalize_graph",
    ]
    assert settings.writer.provider == "local_artifacts"


def test_settings_env_overrides_yaml(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
job:
  job_id: yaml-job
files:
  input_path: yaml-data
embedding:
  dimension: 32
""",
        encoding="utf-8",
    )

    settings = Settings.load(
        config,
        env={
            "KG_JOB_ID": "env-job",
            "KG_INPUT_PATH": str(tmp_path / "env-data"),
            "KG_EMBED_DIM": "64",
        },
    )

    assert settings.job.job_id == "env-job"
    assert settings.files.input_path == tmp_path / "env-data"
    assert settings.embedding.dimension == 64


def test_settings_cli_overrides_env(tmp_path: Path) -> None:
    settings = Settings.load(
        env={"KG_JOB_ID": "env-job"},
        overrides={"job": {"job_id": "cli-job"}},
    )

    assert settings.job.job_id == "cli-job"


def test_settings_loads_inline_container_configuration_before_explicit_env() -> None:
    """Apply a complete orchestrator contract while retaining env precedence."""

    inline = json.dumps(
        {
            "graph": {"max_chunks_per_llm_call": 1},
            "llm": {"model": "inline-model"},
        }
    )

    settings = Settings.load(
        env={
            "KG_CONFIG_JSON": inline,
            "KG_LLM_MODEL": "explicit-env-model",
        }
    )

    assert settings.graph.max_chunks_per_llm_call == 1
    assert settings.llm.model == "explicit-env-model"


@pytest.mark.parametrize("value", ["not-json", "[]"])
def test_settings_rejects_invalid_inline_container_configuration(value: str) -> None:
    """Reject malformed configuration without reproducing its complete value."""

    with pytest.raises(ValueError, match="KG_CONFIG_JSON"):
        Settings.load(env={"KG_CONFIG_JSON": value})


def test_settings_validation_error_does_not_echo_secret_input() -> None:
    """Report a precise invalid field without serializing sibling credentials."""

    with pytest.raises(ValueError) as captured:
        Settings.load(
            env={},
            overrides={
                "snowflake": {
                    "password": "do-not-print-this-password",
                    "service_gpu_count": -1,
                }
            },
        )

    message = str(captured.value)
    assert "snowflake.service_gpu_count" in message
    assert "do-not-print-this-password" not in message


def test_settings_allows_explicit_llm_timeout_override() -> None:
    """Keep the shared timeout overrideable for exceptionally slow validated models.

    Environment configuration should require no YAML edits.
    """

    settings = Settings.load(env={"KG_LLM_TIMEOUT_SECONDS": "900"})

    assert settings.llm.timeout_seconds == 900


def test_settings_loads_spcs_and_snowflake_spec_aliases_from_env() -> None:
    settings = Settings.load(
        env={
            "SNOWFLAKE_ACCOUNT": "EXAMPLE-ACCOUNT",
            "SNOWFLAKE_HOST": "example-account.snowflakecomputing.com",
            "SNOWFLAKE_USER": "KG_SERVICE_USER",
            "SNOWFLAKE_AUTH": "oauth",
            "SNOWFLAKE_DATABASE": "KG_DB",
            "SNOWFLAKE_SCHEMA": "GRAPH",
            "SNOWFLAKE_ROLE": "KG_WRITER",
            "SNOWFLAKE_WAREHOUSE": "KG_PROCESSOR_WH",
            "SNOWFLAKE_OAUTH_TOKEN_PATH": "/snowflake/session/token",
            "KG_SNOWFLAKE_STAGE": "@KG_DB.GRAPH.KG_DOCS",
            "KG_SNOWFLAKE_BULK_STAGE": "@KG_DB.GRAPH.KG_BULK",
        }
    )

    assert settings.snowflake.account == "EXAMPLE-ACCOUNT"
    assert settings.snowflake.host == "example-account.snowflakecomputing.com"
    assert settings.snowflake.user == "KG_SERVICE_USER"
    assert settings.snowflake.authenticator == "oauth"
    assert settings.snowflake.database == "KG_DB"
    assert settings.snowflake.schema_name == "GRAPH"
    assert settings.snowflake.role == "KG_WRITER"
    assert settings.snowflake.warehouse == "KG_PROCESSOR_WH"
    assert settings.snowflake.oauth_token_path == Path("/snowflake/session/token")
    assert settings.snowflake.stage == "@KG_DB.GRAPH.KG_DOCS"
    assert settings.snowflake.bulk_stage == "@KG_DB.GRAPH.KG_BULK"


def test_settings_kg_snowflake_env_overrides_ambient_snowflake_aliases() -> None:
    settings = Settings.load(
        env={
            "SNOWFLAKE_ACCOUNT": "ambient-account",
            "SNOWFLAKE_AUTH": "oauth",
            "KG_SNOWFLAKE_ACCOUNT": "configured-account",
            "KG_SNOWFLAKE_AUTHENTICATOR": "SNOWFLAKE_JWT",
        }
    )

    assert settings.snowflake.account == "configured-account"
    assert settings.snowflake.authenticator == "SNOWFLAKE_JWT"


def test_settings_rejects_unknown_nested_keys() -> None:
    with pytest.raises(ValueError, match="graph.min_entity_confidnce: Extra inputs"):
        Settings.load(overrides={"graph": {"min_entity_confidnce": 0.9}})


def test_settings_yaml_overrides_ambient_snowflake_env_but_not_kg_env(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
snowflake:
  account: yaml-account
  database: YAML_DB
  schema: YAML_SCHEMA
""",
        encoding="utf-8",
    )

    settings = Settings.load(
        config,
        env={
            "SNOWFLAKE_ACCOUNT": "ambient-account",
            "SNOWFLAKE_DATABASE": "AMBIENT_DB",
            "KG_SNOWFLAKE_SCHEMA": "KG_SCHEMA",
        },
    )

    assert settings.snowflake.account == "yaml-account"
    assert settings.snowflake.database == "YAML_DB"
    assert settings.snowflake.schema_name == "KG_SCHEMA"


def test_settings_expands_ai_backend_profiles_before_specific_provider_overrides() -> None:
    oss_settings = Settings.load(env={"KG_AI_BACKEND": "oss"})

    assert oss_settings.ocr.provider == "mineru_internal"
    assert oss_settings.llm.provider == "openai_compatible"
    assert oss_settings.embedding.provider == "sentence_transformers"

    cortex_settings = Settings.load(env={"KG_AI_BACKEND": "cortex"})

    assert cortex_settings.ocr.provider == "snowflake_cortex"
    assert cortex_settings.llm.provider == "snowflake_cortex"
    assert cortex_settings.embedding.provider == "snowflake_cortex"

    override_settings = Settings.load(
        env={
            "KG_AI_BACKEND": "cortex",
            "KG_OCR_PROVIDER": "tesseract_internal",
            "KG_LLM_PROVIDER": "vllm_local",
            "KG_EMBED_PROVIDER": "sentence_transformers",
        }
    )

    assert override_settings.ocr.provider == "tesseract_internal"
    assert override_settings.llm.provider == "vllm_local"
    assert override_settings.embedding.provider == "sentence_transformers"


def test_local_vllm_profile_defaults_to_a_portable_open_model() -> None:
    settings = Settings.load(Path("configs/local-vllm-mineru-oss.yaml"), env={})

    assert settings.llm.provider == "vllm_local"
    assert settings.llm.endpoint == "http://localhost:8000/v1"
    assert settings.llm.model == "Qwen/Qwen3-8B"
    assert settings.llm.api_key is None


def test_local_vllm_profile_defaults_to_real_local_providers() -> None:
    """Verify the quick-start profile selects vLLM and local embeddings."""

    settings = Settings.load(
        Path("data/martial_arts/configs/local-vllm.yaml"),
        env={},
    )

    assert settings.ocr.provider == "builtin_text"
    assert settings.files.input_path == Path("data/martial_arts/files")
    assert "*.html" in settings.files.include_globs
    assert all(not pattern.startswith("**/") for pattern in settings.files.include_globs)
    assert "*.txt" in settings.files.include_globs
    assert "**/*.png" not in settings.files.include_globs
    assert settings.llm.provider == "vllm_local"
    assert settings.llm.endpoint == "http://localhost:8000/v1"
    assert settings.llm.model == "Qwen/Qwen3-8B"
    assert settings.llm.api_key is None
    assert settings.llm.timeout_seconds == 180
    assert settings.embedding.provider == "sentence_transformers"
    assert settings.embedding.endpoint is None
    assert settings.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding.api_key is None
    assert settings.embedding.dimension == 384
    assert settings.ontology.profile_path == Path("data/martial_arts/ontology.yaml")
    assert settings.graph.chunk_token_size == 500
    assert settings.graph.chunk_token_overlap == 60
    assert settings.graph.extraction_window_tokens == 700
    assert settings.graph.max_chunks_per_llm_call == 2
    assert settings.graph.gleaning_max_passes == 1
    assert settings.graph.extraction_parallelism == 2
    assert settings.graph.resolution_embedding_lexical_floor == 0.45
    assert settings.graph.resolution_max_candidates_per_mention == 3
    assert settings.graph.resolution_adjudication_batch_size == 40
    assert settings.graph.finalization_provider_concurrency == 2
    assert settings.writer.output_path == Path("out/local-vllm")


def test_settings_rejects_unknown_ai_backend_profile() -> None:
    with pytest.raises(ValueError, match="KG_AI_BACKEND must be one of"):
        Settings.load(env={"KG_AI_BACKEND": "surprise"})


def test_settings_rejects_unknown_provider_names_before_preflight(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
files:
  source: file-source-that-does-not-exist
  input_path: data/martial_arts/files
ocr:
  provider: ocr-that-does-not-exist
llm:
  provider: llm-that-does-not-exist
embedding:
  provider: embedding-that-does-not-exist
writer:
  provider: writer-that-does-not-exist
cache:
  provider: cache-that-does-not-exist
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        Settings.load(config, env={})

    message = str(exc_info.value)
    assert "Unsupported file_source provider 'file-source-that-does-not-exist'" in message
    assert "azure_blob" in message
    assert "Unsupported ocr provider 'ocr-that-does-not-exist'" in message
    assert "Supported providers:" in message
    assert "mineru_internal" in message
    assert "Unsupported llm provider 'llm-that-does-not-exist'" in message
    assert "openai_compatible" in message
    assert "Unsupported embedding provider 'embedding-that-does-not-exist'" in message
    assert "sentence_transformers" in message
    assert "Unsupported writer provider 'writer-that-does-not-exist'" in message
    assert "local_artifacts" in message
    assert "Unsupported cache provider 'cache-that-does-not-exist'" in message
    assert "snowflake" in message


def test_settings_interpolates_yaml_environment_placeholders(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
llm:
  endpoint: ${KG_LLM_ENDPOINT}
  api_key: ${KG_LLM_API_KEY}
""",
        encoding="utf-8",
    )

    settings = Settings.load(
        config,
        env={
            "KG_LLM_ENDPOINT": "https://example.test/v1",
            "KG_LLM_API_KEY": "secret",
        },
    )

    assert settings.llm.endpoint == "https://example.test/v1"
    assert settings.llm.api_key == "secret"


def test_settings_omits_missing_whole_value_yaml_environment_placeholders(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
embedding:
  dimension: ${KG_EMBED_DIM}
snowflake:
  account: ${KG_SNOWFLAKE_ACCOUNT}
  host: ${KG_SNOWFLAKE_HOST}
""",
        encoding="utf-8",
    )

    settings = Settings.load(
        config,
        env={
            "KG_EMBED_DIM": "",
        },
    )

    assert settings.embedding.dimension == 1024
    assert settings.snowflake.account is None
    assert settings.snowflake.host is None


def test_settings_rejects_missing_mixed_yaml_environment_placeholder(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
llm:
  endpoint: https://${KG_MISSING_HOST}/v1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="KG_MISSING_HOST"):
        Settings.load(config, env={})


def test_settings_environment_value_errors_name_the_field_they_reach() -> None:
    with pytest.raises(ValueError, match="ocr.tesseract_dpi"):
        Settings.load(env={"KG_OCR_TESSERACT_DPI": "not-an-int"})


def test_settings_accepts_zero_gleaning_passes() -> None:
    settings = Settings.load(overrides={"graph": {"gleaning_max_passes": 0}})

    assert settings.graph.gleaning_max_passes == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"embedding": {"batch_size": 0}}, "embedding.batch_size"),
        ({"llm": {"timeout_seconds": 0}}, "llm.timeout_seconds"),
        ({"generic_http_ocr": {"max_response_bytes": -1}}, "generic_http_ocr.max_response_bytes"),
        ({"graph": {"chunk_token_size": 0}}, "graph.chunk_token_size"),
        ({"graph": {"chunk_token_overlap": -1}}, "graph.chunk_token_overlap"),
        (
            {"graph": {"chunk_token_size": 10, "chunk_token_overlap": 10}},
            "chunk_token_overlap must be smaller than chunk_token_size",
        ),
        ({"graph": {"gleaning_max_passes": -1}}, "graph.gleaning_max_passes"),
        ({"graph": {"relation_weight_max": 0}}, "graph.relation_weight_max"),
        (
            {"graph": {"resolution_max_candidates_per_mention": 0}},
            "graph.resolution_max_candidates_per_mention",
        ),
        (
            {"graph": {"finalization_provider_concurrency": 0}},
            "graph.finalization_provider_concurrency",
        ),
        (
            {"graph": {"resolution_embedding_lexical_floor": 1.1}},
            "graph.resolution_embedding_lexical_floor",
        ),
        (
            {"snowflake": {"compute_pool_min_nodes": 3, "compute_pool_max_nodes": 2}},
            "compute_pool_max_nodes must be greater than or equal to min nodes",
        ),
        (
            {"snowflake": {"image_digest": "sha256:not-a-real-digest"}},
            "image_digest must be a sha256:<64 hex chars> OCI digest",
        ),
        (
            {
                "distributed": {
                    "worker_stages": [
                        "extract_entity_window",
                        "extract_entity_window",
                    ]
                }
            },
            "distributed.worker_stages must not contain duplicates",
        ),
        (
            {"distributed": {"finalization_engine": "spark"}},
            "spark finalization requires distributed.artifact_uri",
        ),
    ],
)
def test_settings_rejects_invalid_runtime_bounds(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Settings.load(overrides=overrides)


def test_settings_expose_both_community_bounds_to_the_environment() -> None:
    """A coupled pair of bounds must be reachable together without a config file.

    ``min_community_size`` may not exceed ``max_community_size``, so a deployment
    that only injects environment variables needs both to reach a small maximum.
    """

    settings = Settings.load(
        env={
            "KG_GRAPH_MIN_COMMUNITY_SIZE": "1",
            "KG_GRAPH_MAX_COMMUNITY_SIZE": "1",
        }
    )

    assert settings.graph.min_community_size == 1
    assert settings.graph.max_community_size == 1


@pytest.mark.parametrize(("env_name", "field_path"), ENVIRONMENT_NAMES)
def test_every_environment_name_still_reaches_its_field(env_name: str, field_path: str) -> None:
    group, key = field_path.split(".")
    group_model = Settings.model_fields[group].annotation
    assert group_model is not None
    field = next(
        name for name, info in group_model.model_fields.items() if key in (name, info.alias)
    )
    raw, expected = _VALIDATOR_SHAPED_VALUES.get(field_path) or _sample_environment_value(
        group_model.model_fields[field].annotation
    )

    settings = Settings.load(env={env_name: raw, **_COMPANION_ENVIRONMENT.get(field_path, {})})

    assert getattr(getattr(settings, group), field) == expected


@pytest.mark.parametrize(
    ("alias", "conventional", "field_path", "alias_value", "conventional_value"),
    [
        (
            "KG_SNOWFLAKE_BULK_TARGET_FILE_MB",
            "KG_SNOWFLAKE_BULK_TARGET_FILE_SIZE_MB",
            "snowflake.bulk_target_file_size_mb",
            "1",
            2,
        ),
        ("KG_BATCH_FILES", "KG_JOB_FILE_BATCH_SIZE", "job.file_batch_size", "1", 2),
        (
            "SNOWFLAKE_AUTH",
            "SNOWFLAKE_AUTHENTICATOR",
            "snowflake.authenticator",
            "oauth",
            "SNOWFLAKE_JWT",
        ),
    ],
)
def test_a_field_set_under_two_names_keeps_its_established_winner(
    alias: str,
    conventional: str,
    field_path: str,
    alias_value: str,
    conventional_value: object,
) -> None:
    """Which spelling wins is part of the contract, whatever order the shell exports them."""

    group, field = field_path.split(".")
    for env in (
        {alias: alias_value, conventional: str(conventional_value)},
        {conventional: str(conventional_value), alias: alias_value},
    ):
        settings = Settings.load(env=env)

        assert getattr(getattr(settings, group), field) == conventional_value


def test_comma_separated_environment_lists_trim_their_items() -> None:
    settings = Settings.load(
        env={
            "KG_GRAPH_ENTITY_BLOCKLIST": "chapter,page, appendix ,",
            "KG_DISTRIBUTED_WORKER_STAGES": "prepare_document, extract_relation_window",
        }
    )

    assert settings.graph.entity_blocklist == ["chapter", "page", "appendix"]
    assert settings.distributed.worker_stages == ["prepare_document", "extract_relation_window"]


def test_settings_reject_a_non_positive_ocr_timeout() -> None:
    """The value becomes a subprocess and HTTP deadline that must be reachable."""

    with pytest.raises(ValueError, match="ocr.timeout_seconds"):
        Settings.load(overrides={"ocr": {"timeout_seconds": 0}})


def test_settings_resolve_snowflake_secrets_from_the_variables_they_name() -> None:
    """A configuration that records where a secret lives must produce that secret."""

    settings = Settings.load(
        env={"DEPLOY_SNOWFLAKE_PASSWORD": "s3cret", "DEPLOY_SNOWFLAKE_TOKEN": "t0ken"},
        overrides={
            "snowflake": {
                "password_environment_variable": "DEPLOY_SNOWFLAKE_PASSWORD",
                "oauth_token_environment_variable": "DEPLOY_SNOWFLAKE_TOKEN",
            }
        },
    )

    assert settings.snowflake.password == "s3cret"
    assert settings.snowflake.oauth_token == "t0ken"


def test_settings_keep_a_directly_configured_snowflake_secret() -> None:
    """A named variable is a fallback and never replaces a configured value."""

    settings = Settings.load(
        env={"DEPLOY_SNOWFLAKE_PASSWORD": "from-environment"},
        overrides={
            "snowflake": {
                "password": "from-configuration",
                "password_environment_variable": "DEPLOY_SNOWFLAKE_PASSWORD",
            }
        },
    )

    assert settings.snowflake.password == "from-configuration"


def test_settings_report_a_malformed_configuration_file_as_a_value_error(
    tmp_path: Path,
) -> None:
    """Callers distinguish bad configuration from defects by catching ValueError."""

    config_file = tmp_path / "broken.yaml"
    config_file.write_text("llm:\n  provider: [unclosed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid YAML"):
        Settings.load(config_file, env={})
