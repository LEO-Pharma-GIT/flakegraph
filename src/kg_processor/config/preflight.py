# SPDX-License-Identifier: Apache-2.0
"""Preflight checks for provider availability and account/object readiness.

The worker should fail before expensive OCR/LLM calls when credentials, paths,
Snowflake objects, dimensions, or provider binaries are not ready. These checks
mirror factory selection so configuration errors are caught early and plainly.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
from collections.abc import Callable
from os import W_OK, access
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field

from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.adapters.files.manifest import manifest_candidate_paths
from kg_processor.adapters.snowflake import validate_stage_location
from kg_processor.application.ontology import load_ontology
from kg_processor.config.settings import Settings, require_location_page_for_folder_entity
from kg_processor.ports.ocr import (
    OCR_SUPPORTED_SUFFIXES,
    PAGE_RANGE_GRAMMARS,
    PageWindow,
    parse_page_range,
)

_TESSERACT_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_SPCS_WRITERS = {"snowflake_direct", "snowflake_bulk"}
_SYSTEM_COMPUTE_POOL_PREFIX = "SYSTEM_COMPUTE_POOL"
_SNOWFLAKE_MAX_VECTOR_DIMENSION = 4096


class PreflightResult(BaseModel):
    """Collected preflight checks and errors for a settings object."""

    ok: bool
    checks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def require(self, condition: bool, ok_message: str, error_message: str) -> None:
        """Record either a passing check message or an error."""

        if condition:
            self.checks.append(ok_message)
        else:
            self.errors.append(error_message)
            self.ok = False


def run_preflight(
    settings: Settings,
    *,
    deployment_mode: bool = False,
    orchestrator_mode: bool = False,
) -> PreflightResult:
    """Validate settings before provider construction or expensive processing.

    Preflight mirrors the same provider choices the worker will make, but it
    limits itself to deterministic local checks: configured names, required
    fields, local binaries, file suffix compatibility, writable paths, and
    Snowflake/SPCS shape. It deliberately avoids network calls and heavy model
    loading so users can run it repeatedly while editing YAML or environment
    variables. ``deployment_mode`` skips paths supplied by pod volumes because
    the Helm bootstrap hook does not mount worker data volumes. ``orchestrator_mode``
    retains coordinator-side path and configuration checks while deferring binaries,
    Python packages, model caches, and accelerator availability to deployed workers.
    """

    result = PreflightResult(ok=True)

    _validate_extraction_configuration(
        settings,
        result,
        check_runtime_dependencies=not orchestrator_mode,
    )
    if settings.files.source == "local" and not deployment_mode:
        result.require(
            settings.files.input_path.exists(),
            f"Input path exists: {settings.files.input_path}",
            f"Input path does not exist: {settings.files.input_path}. "
            "The KG_INPUT_PATH environment variable overrides files.input_path from YAML.",
        )
    if settings.files.source == "manifest" and not deployment_mode:
        result.require(
            bool(settings.files.manifest_path and settings.files.manifest_path.is_file()),
            f"Manifest file exists: {settings.files.manifest_path}",
            f"Manifest file source requires an existing files.manifest_path: "
            f"{settings.files.manifest_path}",
        )
    # File-source checks stay close to input-path checks because failures here
    # mean the worker will not even have documents to hand to OCR.
    _validate_azure_blob_file_source(settings, result)
    _validate_s3_file_source(settings, result)
    _validate_snowflake_provider_selection(settings, result)
    if settings.writer.provider == "local_artifacts" and not deployment_mode:
        parent = _nearest_existing_parent(settings.writer.output_path)
        result.require(
            parent.is_dir() and access(parent, W_OK),
            f"Output path can be created below: {parent}",
            f"Output path cannot be created below: {parent}",
        )
    if settings.cache.provider == "local" and not deployment_mode:
        parent = _nearest_existing_parent(settings.cache.path)
        result.require(
            parent.is_dir() and access(parent, W_OK),
            f"Cache path can be created below: {parent}",
            f"Cache path cannot be created below: {parent}",
        )
    # OCR validation is split between provider prerequisites and document shape:
    # binaries/endpoints are checked separately from page-window and file-type
    # compatibility so error messages point to the exact fix.
    _validate_local_ocr_provider_selection(
        settings,
        result,
        check_runtime_dependencies=not orchestrator_mode,
    )
    _validate_ocr_page_range(settings, result)
    _validate_local_ocr_file_types(settings, result)
    _validate_llm_provider_selection(settings, result)
    _validate_embedding_provider_selection(
        settings,
        result,
        check_runtime_dependencies=not orchestrator_mode,
    )
    if settings.runtime.runtime == "spcs":
        # SPCS jobs need more than Snowflake credentials: they also need image,
        # compute, stage, and service-spec fields before deployment SQL is
        # meaningful.
        required = [
            settings.snowflake.account or settings.snowflake.host,
            settings.snowflake.database,
            settings.snowflake.schema_name,
            settings.snowflake.role,
            settings.snowflake.warehouse,
            settings.snowflake.compute_pool,
            settings.snowflake.image_repository,
            settings.snowflake.service_spec_stage,
        ]
        result.require(
            all(required),
            "Snowflake runtime fields are configured",
            "Snowflake runtime requires account/host, database, schema, role, warehouse, "
            "compute pool, image repository, and service spec stage",
        )
        if settings.snowflake.service_spec_stage:
            _require_valid_stage(settings.snowflake.service_spec_stage, result)
        _validate_spcs_compute_pool(settings, result)
        _validate_spcs_runtime_provider_selection(settings, result)
    if settings.job.use_lease:
        result.require(
            bool(settings.job.lease_owner),
            "Job lease owner is configured",
            "job.use_lease requires job.lease_owner",
        )
        _require_snowflake_target(settings, "job lease", result)
    if settings.job.use_file_queue:
        result.require(
            bool(settings.job.lease_owner),
            "Job file queue worker id is configured",
            "job.use_file_queue requires job.lease_owner",
        )
        _require_snowflake_target(settings, "job file queue", result)
    _validate_no_secret_files(settings, result)
    return result


def _validate_extraction_configuration(
    settings: Settings,
    result: PreflightResult,
    *,
    check_runtime_dependencies: bool = True,
) -> None:
    """Validate ontology content and optional extractor dependencies before processing.

    Expensive OCR and model loading should never begin when the selected extraction
    profile is missing, malformed, or lacks its optional installation extra.
    """

    _validate_ontology_profile(settings, result)
    if settings.extractors.entity_provider == "gliner" and check_runtime_dependencies:
        result.require(
            importlib.util.find_spec("gliner") is not None,
            "GLiNER package is installed",
            "GLiNER entity extraction requires: uv sync --extra extract-gliner",
        )


def _validate_ontology_profile(settings: Settings, result: PreflightResult) -> None:
    """Validate the same ontology the run will extract with.

    ``load_ontology`` gives an inline profile precedence over a profile path, so
    preflight resolves the source the same way. An orchestrator that inlines a
    resolved profile is otherwise told a file it will never read is valid.
    """

    inline_profile = settings.ontology.profile
    ontology_path = settings.ontology.profile_path
    if inline_profile is None:
        if ontology_path is None:
            return
        result.require(
            ontology_path.is_file(),
            f"Ontology profile exists: {ontology_path}",
            f"Ontology profile does not exist: {ontology_path}",
        )
        if not ontology_path.is_file():
            return
    try:
        loaded = load_ontology(
            ontology_path,
            settings.graph.entity_types,
            settings.graph.relation_types,
            inline=inline_profile,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        result.require(False, "", f"Ontology profile is invalid: {exc}")
        return
    result.checks.append(
        f"Ontology profile is valid: {loaded.profile.name} "
        f"({loaded.checksum[:12]}) from {loaded.source}"
    )
    try:
        require_location_page_for_folder_entity(
            loaded.profile.folder_entity is not None, settings.files.location_page
        )
    except ValueError as exc:
        result.require(False, "", str(exc))


def _validate_azure_blob_file_source(settings: Settings, result: PreflightResult) -> None:
    if settings.files.source != "azure_blob":
        return
    result.require(
        bool(settings.azure_blob.container),
        "Azure Blob container is configured",
        "azure_blob file source requires azure_blob.container",
    )
    result.require(
        bool(settings.azure_blob.connection_string or settings.azure_blob.account_url),
        "Azure Blob connection is configured",
        "azure_blob file source requires connection_string or account_url; "
        "account_url uses the Azure default credential chain when sas_token is absent",
    )
    parent = _nearest_existing_parent(settings.azure_blob.download_path)
    result.require(
        parent.is_dir() and access(parent, W_OK),
        f"Azure Blob download path can be created below: {parent}",
        f"Azure Blob download path cannot be created below: {parent}",
    )


def _validate_s3_file_source(settings: Settings, result: PreflightResult) -> None:
    """Validate the namespace and local staging path for an S3-compatible source."""

    if settings.files.source != "s3":
        return
    result.require(
        bool(settings.s3.bucket),
        "S3 bucket is configured",
        "s3 file source requires s3.bucket",
    )
    parent = _nearest_existing_parent(settings.s3.download_path)
    result.require(
        parent.is_dir() and access(parent, W_OK),
        f"S3 download path can be created below: {parent}",
        f"S3 download path cannot be created below: {parent}",
    )


def _validate_local_ocr_provider_selection(
    settings: Settings,
    result: PreflightResult,
    *,
    check_runtime_dependencies: bool = True,
) -> None:
    """Validate prerequisites for every concrete OCR provider that may execute.

    A fallback adapter is only a routing policy; its primary and secondary
    providers still need the same executable, endpoint, and cache checks as a
    directly selected provider. Expanding the composition here prevents a
    misleading successful preflight followed by a late failure on a scanned
    document.
    """

    if settings.ocr.provider == "fallback":
        primary = settings.ocr.fallback_primary_provider
        secondary = settings.ocr.fallback_secondary_provider
        result.require(
            "fallback" not in {primary, secondary},
            "fallback OCR contains only concrete providers",
            "fallback OCR cannot contain another fallback provider",
        )
        result.require(
            primary != secondary,
            "fallback OCR primary and secondary providers differ",
            "fallback OCR primary and secondary providers must differ",
        )
    for provider in _effective_ocr_providers(settings):
        _validate_named_local_ocr_provider(
            settings,
            result,
            provider,
            check_runtime_dependencies=check_runtime_dependencies,
        )


def _validate_named_local_ocr_provider(
    settings: Settings,
    result: PreflightResult,
    provider: str,
    *,
    check_runtime_dependencies: bool = True,
) -> None:
    """Validate local requirements for one concrete OCR adapter name."""

    if provider == "mineru_internal":
        if check_runtime_dependencies:
            result.require(
                shutil.which(settings.ocr.mineru_command) is not None,
                f"MinerU command is available: {settings.ocr.mineru_command}",
                f"MinerU command is not available: {settings.ocr.mineru_command}",
            )
            if settings.ocr.model_cache_dir is not None:
                parent = _nearest_existing_parent(settings.ocr.model_cache_dir)
                result.require(
                    parent.is_dir() and access(parent, W_OK),
                    f"MinerU model/cache path can be created below: {parent}",
                    f"MinerU model/cache path cannot be created below: {parent}",
                )
        if settings.ocr.mineru_api_url:
            result.require(
                _is_http_url(settings.ocr.mineru_api_url),
                "MinerU internal API endpoint is a valid HTTP URL",
                "mineru_internal api_url must be an http(s) URL",
            )
        if settings.ocr.mineru_server_url:
            result.require(
                _is_http_url(settings.ocr.mineru_server_url),
                "MinerU internal server URL is a valid HTTP URL",
                "mineru_internal server_url must be an http(s) URL",
            )
    if provider == "mineru_api":
        result.require(
            bool(settings.ocr.mineru_api_url),
            "MinerU API endpoint is configured",
            "mineru_api OCR requires mineru_api_url",
        )
        if settings.ocr.mineru_api_url:
            result.require(
                _is_http_url(settings.ocr.mineru_api_url),
                "MinerU API endpoint is a valid HTTP URL",
                "mineru_api OCR endpoint must be an http(s) URL",
            )
    if provider == "generic_http":
        result.require(
            bool(settings.generic_http_ocr.endpoint),
            "Generic HTTP OCR endpoint is configured",
            "generic_http OCR requires generic_http_ocr.endpoint",
        )
        if settings.generic_http_ocr.endpoint:
            result.require(
                _is_http_url(settings.generic_http_ocr.endpoint),
                "Generic HTTP OCR endpoint is a valid HTTP URL",
                "generic_http OCR endpoint must be an http(s) URL",
            )
    if provider == "tesseract_internal" and check_runtime_dependencies:
        result.require(
            shutil.which(settings.ocr.tesseract_command) is not None,
            f"Tesseract command is available: {settings.ocr.tesseract_command}",
            f"Tesseract command is not available: {settings.ocr.tesseract_command}",
        )
        result.require(
            shutil.which(settings.ocr.tesseract_pdf_renderer_command) is not None,
            (
                "Tesseract PDF renderer command is available: "
                f"{settings.ocr.tesseract_pdf_renderer_command}"
            ),
            (
                "Tesseract PDF renderer command is not available: "
                f"{settings.ocr.tesseract_pdf_renderer_command}"
            ),
        )


def _validate_ocr_page_range(settings: Settings, result: PreflightResult) -> None:
    """Validate OCR page-window syntax before a provider starts parsing documents.

    Each OCR backend mirrors a different upstream API. Builtin text extraction
    and Tesseract expose one-based page numbers because users think in document
    pages. MinerU and Snowflake Cortex expose zero-based page ids because their
    APIs use index-style page filters. Keeping that distinction visible here
    prevents quiet off-by-one surprises in long container jobs.
    """

    page_range = settings.ocr.page_range
    if page_range is not None and not page_range.strip():
        page_range = None

    if settings.ocr.provider == "fallback":
        has_provider_specific_window = (
            page_range is not None
            or settings.ocr.mineru_start_page_id is not None
            or settings.ocr.mineru_end_page_id is not None
        )
        result.require(
            not has_provider_specific_window,
            "fallback OCR uses complete documents",
            "fallback OCR does not support page ranges because composed providers may use "
            "different page-number conventions",
        )
        return

    provider = settings.ocr.provider
    start = settings.ocr.mineru_start_page_id
    end = settings.ocr.mineru_end_page_id
    if provider.startswith("mineru") and (start is not None or end is not None):
        # MinerU's explicit page ids take precedence over the portable
        # page_range field, so page_range is not reinterpreted beside them.
        result.require(
            start is None or end is None or end >= start,
            f"{provider} OCR explicit page window is valid",
            f"Invalid {provider} page window: mineru_end_page_id is before mineru_start_page_id",
        )
        return

    if not page_range or provider not in PAGE_RANGE_GRAMMARS:
        return
    windows = _require_page_range(
        result,
        lambda: parse_page_range(page_range, provider),
        f"{provider} OCR page_range is valid",
    )
    if provider == "tesseract_internal" and windows is not None:
        _validate_tesseract_image_page_range(settings, windows, result)


def _validate_tesseract_image_page_range(
    settings: Settings,
    windows: list[PageWindow],
    result: PreflightResult,
) -> None:
    # A standalone image represents exactly one OCR page. Tesseract can page
    # slice PDFs, but an image filter that excludes page 1 would produce no
    # document content and should fail while config is still cheap to fix.
    candidates = _candidate_paths_or_error(settings, result)
    if candidates is None:
        return
    if not any(path.suffix.lower() in _TESSERACT_IMAGE_SUFFIXES for path in candidates):
        return
    includes_page_one = any(
        (start is None or start <= 1) and (end is None or end >= 1) for start, end in windows
    )
    result.require(
        includes_page_one,
        "tesseract_internal image OCR page_range includes page 1",
        "tesseract_internal image OCR only supports page 1; adjust ocr.page_range",
    )


def _require_page_range(
    result: PreflightResult,
    parser: Callable[[], list[PageWindow]],
    ok_message: str,
) -> list[PageWindow] | None:
    try:
        windows = parser()
    except ValueError as exc:
        result.errors.append(str(exc))
        result.ok = False
        return None
    result.checks.append(ok_message)
    return windows


def _validate_local_ocr_file_types(settings: Settings, result: PreflightResult) -> None:
    if settings.files.source not in {"local", "manifest"}:
        return
    candidates = _candidate_paths_or_error(settings, result)
    if candidates is None:
        return
    if not candidates:
        return
    providers = _effective_ocr_providers(settings)
    unsupported = sorted(
        {
            path.suffix.lower()
            for path in candidates
            if not any(
                _provider_supports_suffix(provider, path.suffix.lower()) for provider in providers
            )
        }
    )
    label = settings.ocr.provider
    result.require(
        not unsupported,
        f"{label} OCR supports all configured local file types",
        f"{label} OCR does not support configured file suffixes: {', '.join(unsupported)}",
    )


def _provider_supports_suffix(provider: str, suffix: str) -> bool:
    """Treat remote schema-mapped providers as dynamic and declared adapters strictly."""

    supported = OCR_SUPPORTED_SUFFIXES.get(provider)
    return supported is None or suffix in supported


def _effective_ocr_providers(settings: Settings) -> tuple[str, ...]:
    """Return the concrete OCR adapters reachable from the selected provider.

    Keeping this expansion in preflight mirrors factory composition without
    constructing heavy adapters or importing optional OCR runtimes.
    """

    if settings.ocr.provider != "fallback":
        return (settings.ocr.provider,)
    return (
        settings.ocr.fallback_primary_provider,
        settings.ocr.fallback_secondary_provider,
    )


def _local_or_manifest_candidate_paths(settings: Settings) -> list[Path]:
    if settings.files.source == "manifest" and settings.files.manifest_path:
        if not settings.files.manifest_path.is_file():
            return []
        return manifest_candidate_paths(
            settings.files.manifest_path,
            settings.files.include_globs,
            containment_root=settings.files.manifest_root,
        )
    if settings.files.source != "local":
        return []
    source = LocalFileSource(settings.files.input_path, settings.files.include_globs)
    return list(source.candidate_paths())


def _candidate_paths_or_error(
    settings: Settings,
    result: PreflightResult,
) -> list[Path] | None:
    """Turn malformed manifest rows into a structured preflight failure."""

    try:
        return _local_or_manifest_candidate_paths(settings)
    except (OSError, TypeError, ValueError) as exc:
        label = (
            "Input manifest" if settings.files.source == "manifest" else "Local source selection"
        )
        result.require(False, f"{label} is valid", f"{label} is invalid: {exc}")
        return None


def _validate_llm_provider_selection(settings: Settings, result: PreflightResult) -> None:
    if settings.llm.provider in {"ollama", "openai_compatible", "azure_openai", "vllm_local"}:
        result.require(
            bool(settings.llm.model),
            f"{settings.llm.provider} LLM model is configured",
            f"{settings.llm.provider} LLM requires llm.model",
        )
    if settings.llm.provider in {"ollama", "vllm_local"}:
        result.require(
            bool(settings.llm.endpoint),
            f"{settings.llm.provider} LLM endpoint is configured",
            f"{settings.llm.provider} LLM requires endpoint",
        )
    if settings.llm.provider in {"openai_compatible", "azure_openai"}:
        result.require(
            bool(settings.llm.endpoint and settings.llm.api_key),
            f"{settings.llm.provider} LLM endpoint and API key are configured",
            f"{settings.llm.provider} LLM requires endpoint and API key",
        )
    if settings.llm.provider == "azure_openai":
        result.require(
            bool(settings.llm.api_version),
            "azure_openai LLM API version is configured",
            "azure_openai LLM requires llm.api_version",
        )
    if settings.llm.provider == "snowflake_cortex":
        result.require(
            bool(settings.llm.model),
            "Snowflake Cortex LLM model is configured",
            "snowflake_cortex LLM requires llm.model",
        )


def _validate_embedding_provider_selection(
    settings: Settings,
    result: PreflightResult,
    *,
    check_runtime_dependencies: bool = True,
) -> None:
    if settings.embedding.provider in {
        "openai_compatible",
        "azure_openai",
        "sentence_transformers",
        "snowflake_cortex",
    }:
        result.require(
            bool(settings.embedding.model),
            f"{settings.embedding.provider} embedding model is configured",
            f"{settings.embedding.provider} embedding requires embedding.model",
        )
    if settings.embedding.provider in {"openai_compatible", "azure_openai"}:
        result.require(
            bool(settings.embedding.endpoint and settings.embedding.api_key),
            f"{settings.embedding.provider} embedding endpoint and API key are configured",
            f"{settings.embedding.provider} embedding requires endpoint and API key",
        )
    if settings.embedding.provider == "azure_openai":
        result.require(
            bool(settings.embedding.api_version),
            "azure_openai embedding API version is configured",
            "azure_openai embedding requires embedding.api_version",
        )
    if settings.embedding.provider == "sentence_transformers" and check_runtime_dependencies:
        result.require(
            importlib.util.find_spec("sentence_transformers") is not None,
            "sentence_transformers package is installed",
            "sentence_transformers embeddings require the local-embeddings extra",
        )
        if _is_cuda_device(settings.embedding.device):
            result.require(
                _torch_cuda_device_available(settings.embedding.device or "cuda"),
                f"CUDA embedding device is available: {settings.embedding.device}",
                (
                    "sentence_transformers embedding device "
                    f"{settings.embedding.device} requires CUDA but CUDA is not available"
                ),
            )


def _validate_no_secret_files(settings: Settings, result: PreflightResult) -> None:
    paths: list[Path] = [settings.files.input_path, settings.writer.output_path]
    for path in paths:
        if ".env" in path.parts:
            result.errors.append(f"Runtime path must not point inside a secret file path: {path}")
            result.ok = False


def _nearest_existing_parent(path: Path) -> Path:
    current = path if path.exists() else path.parent
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_cuda_device(device: str | None) -> bool:
    if not device:
        return False
    return device.strip().lower() == "cuda" or device.strip().lower().startswith("cuda:")


def _torch_cuda_device_available(device: str) -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return False
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available) or not bool(is_available()):
        return False
    index = _cuda_device_index(device)
    if index is None:
        return True
    device_count = getattr(cuda, "device_count", None)
    return callable(device_count) and int(device_count()) > index


def _cuda_device_index(device: str) -> int | None:
    normalized = device.strip().lower()
    if normalized == "cuda":
        return None
    _prefix, _separator, suffix = normalized.partition(":")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _validate_snowflake_provider_selection(
    settings: Settings,
    result: PreflightResult,
) -> None:
    if settings.files.source == "snowflake_stage":
        result.require(
            bool(settings.snowflake.stage),
            "Snowflake stage is configured",
            "snowflake_stage file source requires snowflake.stage",
        )
        if settings.snowflake.stage:
            _require_valid_stage(settings.snowflake.stage, result)
        _require_snowflake_target(settings, "snowflake_stage file source", result)
        result.require(
            settings.ocr.provider == "snowflake_cortex",
            "Snowflake stage files will be parsed by Snowflake Cortex OCR",
            (
                "snowflake_stage file source requires ocr.provider=snowflake_cortex "
                "because staged files are not downloaded to a local path"
            ),
        )
    if "snowflake_cortex" in _effective_ocr_providers(settings):
        result.require(
            settings.ocr.provider == "snowflake_cortex"
            and settings.files.source == "snowflake_stage",
            "Snowflake Cortex OCR will read staged files",
            "snowflake_cortex OCR must be selected directly and requires "
            "files.source=snowflake_stage",
        )
        _require_snowflake_target(settings, "snowflake_cortex OCR", result)
    if settings.llm.provider == "snowflake_cortex":
        _require_snowflake_target(settings, "snowflake_cortex LLM", result)
    if settings.embedding.provider == "snowflake_cortex":
        _require_snowflake_target(settings, "snowflake_cortex embedding", result)
    if settings.cache.provider == "snowflake":
        _require_snowflake_target(settings, "snowflake cache", result)
    if settings.writer.provider == "snowflake_direct":
        _require_snowflake_target(settings, "snowflake_direct writer", result)
    if settings.writer.provider == "snowflake_bulk":
        _require_snowflake_target(settings, "snowflake_bulk writer", result)
        result.require(
            bool(settings.snowflake.bulk_stage),
            "Snowflake bulk load stage is configured",
            "snowflake_bulk writer requires snowflake.bulk_stage",
        )
        if settings.snowflake.bulk_stage:
            _require_valid_stage(settings.snowflake.bulk_stage, result)
    if settings.writer.provider in {"snowflake_direct", "snowflake_bulk"}:
        result.require(
            settings.embedding.dimension <= _SNOWFLAKE_MAX_VECTOR_DIMENSION,
            "Embedding dimension fits Snowflake VECTOR",
            "Snowflake writers require embedding.dimension <= 4096",
        )


def _validate_spcs_runtime_provider_selection(
    settings: Settings,
    result: PreflightResult,
) -> None:
    """Keep SPCS jobs on the Snowflake-native provider set.

    The local/open-source profile is intentionally complete, but it should not
    leak into Snowpark Container Services. SPCS containers are ephemeral and run
    inside the Snowflake boundary, so staged input, Cortex providers, Snowflake
    cache tables, and Snowflake writers are the reviewable production contract.
    """

    result.require(
        settings.files.source == "snowflake_stage",
        "SPCS runtime uses staged Snowflake input",
        "SPCS runtime requires files.source=snowflake_stage",
    )
    result.require(
        settings.ocr.provider == "snowflake_cortex",
        "SPCS runtime uses Snowflake Cortex OCR",
        "SPCS runtime requires ocr.provider=snowflake_cortex",
    )
    result.require(
        settings.llm.provider == "snowflake_cortex",
        "SPCS runtime uses Snowflake Cortex LLM",
        "SPCS runtime requires llm.provider=snowflake_cortex",
    )
    result.require(
        settings.embedding.provider == "snowflake_cortex",
        "SPCS runtime uses Snowflake Cortex embeddings",
        "SPCS runtime requires embedding.provider=snowflake_cortex",
    )
    result.require(
        settings.writer.provider in _SPCS_WRITERS,
        "SPCS runtime uses a Snowflake graph writer",
        "SPCS runtime requires writer.provider=snowflake_direct or snowflake_bulk",
    )
    result.require(
        settings.cache.provider == "snowflake",
        "SPCS runtime uses Snowflake cache tables",
        "SPCS runtime requires cache.provider=snowflake",
    )


def _validate_spcs_compute_pool(settings: Settings, result: PreflightResult) -> None:
    pool = settings.snowflake.compute_pool or ""
    result.require(
        not pool.upper().startswith(_SYSTEM_COMPUTE_POOL_PREFIX),
        "SPCS runtime uses a dedicated service-compatible compute pool",
        "SPCS runtime requires a dedicated compute pool; Snowflake SYSTEM_COMPUTE_POOL_* "
        "pools reject FlakeGraph EXECUTE JOB SERVICE workloads",
    )


def _require_snowflake_target(
    settings: Settings,
    label: str,
    result: PreflightResult,
) -> None:
    result.require(
        bool(
            (settings.snowflake.account or settings.snowflake.host)
            and settings.snowflake.database
            and settings.snowflake.schema_name
        ),
        f"{label} Snowflake target is configured",
        f"{label} requires account/host, database, and schema",
    )


def _require_valid_stage(stage: str, result: PreflightResult) -> None:
    try:
        validate_stage_location(stage)
    except ValueError as exc:
        result.errors.append(str(exc))
        result.ok = False
    else:
        result.checks.append(f"Snowflake stage location is valid: {stage}")
