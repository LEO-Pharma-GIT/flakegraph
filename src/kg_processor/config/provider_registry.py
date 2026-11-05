# SPDX-License-Identifier: Apache-2.0
"""Single provider catalog used by docs, CLI inspection, factories, and tests."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ProviderKind = Literal["file_source", "ocr", "llm", "embedding", "writer", "cache"]
ProviderProfile = Literal["local", "open_source", "external", "snowflake", "test"]


class ProviderDescriptor(BaseModel):
    """Catalog entry describing one selectable provider implementation."""

    kind: ProviderKind
    name: str
    profile: ProviderProfile
    implementation: str
    summary: str
    requires: list[str] = Field(default_factory=list)


_PROVIDERS: tuple[ProviderDescriptor, ...] = (
    ProviderDescriptor(
        kind="file_source",
        name="local",
        profile="local",
        implementation="kg_processor.adapters.files.local.LocalFileSource",
        summary="Read files from a local or mounted directory.",
        requires=["files.input_path"],
    ),
    ProviderDescriptor(
        kind="file_source",
        name="manifest",
        profile="local",
        implementation="kg_processor.adapters.files.manifest.ManifestFileSource",
        summary="Read an explicit JSON, JSONL, NDJSON, or CSV manifest.",
        requires=["files.manifest_path"],
    ),
    ProviderDescriptor(
        kind="file_source",
        name="azure_blob",
        profile="external",
        implementation="kg_processor.adapters.files.azure_blob.AzureBlobFileSource",
        summary="Download files from Azure Blob Storage into a local work directory.",
        requires=[
            "azure_blob.container",
            "azure_blob.connection_string or account_url with SAS/default Azure identity",
        ],
    ),
    ProviderDescriptor(
        kind="file_source",
        name="s3",
        profile="external",
        implementation="kg_processor.adapters.files.s3.S3FileSource",
        summary="Download files from AWS S3, MinIO, or another S3-compatible bucket.",
        requires=["s3.bucket", "boto3 credential provider chain"],
    ),
    ProviderDescriptor(
        kind="file_source",
        name="snowflake_stage",
        profile="snowflake",
        implementation="kg_processor.adapters.files.snowflake_stage.SnowflakeStageFileSource",
        summary="List files from a Snowflake internal or external stage.",
        requires=["snowflake.stage", "snowflake.database", "snowflake.schema"],
    ),
    ProviderDescriptor(
        kind="ocr",
        name="mineru_internal",
        profile="open_source",
        implementation="kg_processor.adapters.ocr.mineru_internal.MineruInternalOcrProvider",
        summary="Run the MinerU CLI inside the worker container.",
        requires=["ocr.mineru_command"],
    ),
    ProviderDescriptor(
        kind="ocr",
        name="fallback",
        profile="open_source",
        implementation="kg_processor.adapters.ocr.fallback.FallbackOcrProvider",
        summary="Use a primary parser when text is sufficient and otherwise run secondary OCR.",
        requires=[
            "ocr.fallback_primary_provider",
            "ocr.fallback_secondary_provider",
        ],
    ),
    ProviderDescriptor(
        kind="ocr",
        name="mineru_api",
        profile="external",
        implementation="kg_processor.adapters.ocr.mineru_api.MineruApiOcrProvider",
        summary="Call a MinerU HTTP API deployment and normalize its response.",
        requires=["ocr.mineru_api_url"],
    ),
    ProviderDescriptor(
        kind="ocr",
        name="builtin_text",
        profile="test",
        implementation="kg_processor.adapters.ocr.builtin_text.BuiltinTextOcrProvider",
        summary="Deterministic local text/PDF parser for tests and smoke runs.",
    ),
    ProviderDescriptor(
        kind="ocr",
        name="generic_http",
        profile="external",
        implementation="kg_processor.adapters.ocr.generic_http.GenericHttpOcrProvider",
        summary="Call a schema-mapped multipart HTTP OCR endpoint.",
        requires=["generic_http_ocr.endpoint"],
    ),
    ProviderDescriptor(
        kind="ocr",
        name="tesseract_internal",
        profile="open_source",
        implementation="kg_processor.adapters.ocr.tesseract_internal.TesseractInternalOcrProvider",
        summary="Run Tesseract and Poppler inside the worker container.",
        requires=["ocr.tesseract_command", "ocr.tesseract_pdf_renderer_command"],
    ),
    ProviderDescriptor(
        kind="ocr",
        name="snowflake_cortex",
        profile="snowflake",
        implementation="kg_processor.adapters.ocr.snowflake_cortex.SnowflakeCortexOcrProvider",
        summary="Use Snowflake Cortex AI_PARSE_DOCUMENT against staged files.",
        requires=["files.source=snowflake_stage", "snowflake.database", "snowflake.schema"],
    ),
    ProviderDescriptor(
        kind="llm",
        name="fake",
        profile="test",
        implementation="kg_processor.adapters.llm.fake.FakeLlmProvider",
        summary="Deterministic extraction and summary provider for tests and smoke runs.",
    ),
    ProviderDescriptor(
        kind="llm",
        name="ollama",
        profile="open_source",
        implementation="kg_processor.adapters.llm.ollama.OllamaLlmProvider",
        summary="Call a local or remote Ollama native chat endpoint.",
        requires=["llm.endpoint", "llm.model"],
    ),
    ProviderDescriptor(
        kind="llm",
        name="openai_compatible",
        profile="external",
        implementation="kg_processor.adapters.llm.openai_compatible.OpenAICompatibleLlmProvider",
        summary="Call an OpenAI-compatible chat completions API.",
        requires=["llm.endpoint", "llm.api_key", "llm.model"],
    ),
    ProviderDescriptor(
        kind="llm",
        name="azure_openai",
        profile="external",
        implementation="kg_processor.adapters.llm.azure_openai.AzureOpenAILlmProvider",
        summary="Call an Azure OpenAI deployment endpoint.",
        requires=["llm.endpoint", "llm.api_key", "llm.model", "llm.api_version"],
    ),
    ProviderDescriptor(
        kind="llm",
        name="vllm_local",
        profile="open_source",
        implementation="kg_processor.adapters.llm.vllm_local.VllmLocalLlmProvider",
        summary="Call a local vLLM OpenAI-compatible endpoint.",
        requires=["llm.endpoint", "llm.model"],
    ),
    ProviderDescriptor(
        kind="llm",
        name="snowflake_cortex",
        profile="snowflake",
        implementation="kg_processor.adapters.llm.snowflake_cortex.SnowflakeCortexLlmProvider",
        summary="Use the Snowflake Cortex REST completion API with the shared graph schemas.",
        requires=["snowflake.database", "snowflake.schema", "llm.model"],
    ),
    ProviderDescriptor(
        kind="embedding",
        name="hash",
        profile="test",
        implementation="kg_processor.adapters.embeddings.hash.HashEmbeddingProvider",
        summary="Deterministic hash embeddings for tests and smoke runs.",
        requires=["embedding.dimension"],
    ),
    ProviderDescriptor(
        kind="embedding",
        name="sentence_transformers",
        profile="open_source",
        implementation=(
            "kg_processor.adapters.embeddings.sentence_transformers."
            "SentenceTransformersEmbeddingProvider"
        ),
        summary="Run local sentence-transformer embeddings.",
        requires=["local-embeddings extra", "embedding.model", "embedding.dimension"],
    ),
    ProviderDescriptor(
        kind="embedding",
        name="openai_compatible",
        profile="external",
        implementation=(
            "kg_processor.adapters.embeddings.openai_compatible.OpenAICompatibleEmbeddingProvider"
        ),
        summary="Call an OpenAI-compatible embeddings API.",
        requires=["embedding.endpoint", "embedding.api_key", "embedding.model"],
    ),
    ProviderDescriptor(
        kind="embedding",
        name="azure_openai",
        profile="external",
        implementation="kg_processor.adapters.embeddings.azure_openai.AzureOpenAIEmbeddingProvider",
        summary="Call an Azure OpenAI embedding deployment endpoint.",
        requires=[
            "embedding.endpoint",
            "embedding.api_key",
            "embedding.model",
            "embedding.api_version",
        ],
    ),
    ProviderDescriptor(
        kind="embedding",
        name="snowflake_cortex",
        profile="snowflake",
        implementation=(
            "kg_processor.adapters.embeddings.snowflake_cortex.SnowflakeCortexEmbeddingProvider"
        ),
        summary="Use the Snowflake Cortex REST embedding API.",
        requires=["snowflake.database", "snowflake.schema", "embedding.model"],
    ),
    ProviderDescriptor(
        kind="writer",
        name="local_artifacts",
        profile="local",
        implementation="kg_processor.adapters.writers.local_artifacts.LocalArtifactsWriter",
        summary="Write Parquet/JSON artifacts to a local directory.",
        requires=["writer.output_path"],
    ),
    ProviderDescriptor(
        kind="writer",
        name="snowflake_direct",
        profile="snowflake",
        implementation="kg_processor.adapters.writers.snowflake_direct.SnowflakeDirectWriter",
        summary="Write graph batches directly through parameterized Snowflake SQL.",
        requires=["snowflake.database", "snowflake.schema"],
    ),
    ProviderDescriptor(
        kind="writer",
        name="snowflake_bulk",
        profile="snowflake",
        implementation="kg_processor.adapters.writers.snowflake_bulk.SnowflakeBulkWriter",
        summary="Stage Parquet load files and MERGE them into Snowflake tables.",
        requires=["snowflake.database", "snowflake.schema", "snowflake.bulk_stage"],
    ),
    ProviderDescriptor(
        kind="cache",
        name="none",
        profile="local",
        implementation="",
        summary="Disable pipeline caching.",
    ),
    ProviderDescriptor(
        kind="cache",
        name="local",
        profile="local",
        implementation="kg_processor.adapters.cache.local_json.LocalJsonCache",
        summary="Store OCR and extraction cache entries in local JSON files.",
        requires=["cache.path"],
    ),
    ProviderDescriptor(
        kind="cache",
        name="snowflake",
        profile="snowflake",
        implementation="kg_processor.adapters.cache.snowflake.SnowflakeCache",
        summary="Store OCR and extraction cache entries in Snowflake tables.",
        requires=["snowflake.database", "snowflake.schema"],
    ),
)

_PROVIDER_KINDS: tuple[ProviderKind, ...] = (
    "file_source",
    "ocr",
    "llm",
    "embedding",
    "writer",
    "cache",
)


def provider_catalog() -> list[dict[str, Any]]:
    """Return the full provider catalog as JSON-serializable dictionaries."""

    return [provider.model_dump(mode="json") for provider in _PROVIDERS]


def provider_kinds() -> tuple[ProviderKind, ...]:
    """Return the supported provider categories in display order."""

    return _PROVIDER_KINDS


def provider_profile(name: str) -> ProviderProfile | None:
    """Return the profile a provider name is registered under, whatever its kind.

    A name that appears under several kinds (``snowflake_cortex`` parses,
    completes and embeds) carries the same profile under each, so the kind is
    not needed to answer where its work runs.
    """

    return next((provider.profile for provider in _PROVIDERS if provider.name == name), None)


def provider_names(kind: ProviderKind) -> frozenset[str]:
    """Return all registered provider names for one category."""

    return frozenset(provider.name for provider in _PROVIDERS if provider.kind == kind)


def provider_catalog_for_kind(kind: ProviderKind) -> list[dict[str, Any]]:
    """Return catalog entries for one provider category."""

    return [provider.model_dump(mode="json") for provider in _PROVIDERS if provider.kind == kind]
