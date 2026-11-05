# SPDX-License-Identifier: Apache-2.0
"""Factory wiring from typed settings to concrete provider adapters.

This module is the only place that maps provider names to implementations. That
centralization makes unsupported-provider errors deterministic and keeps
application modules dependent only on ports.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import TaskStage
from kg_processor.domain.jobs import JobFileClaim
from kg_processor.ports.blob_store import BlobStore
from kg_processor.ports.cache import PipelineCache
from kg_processor.ports.embeddings import EmbeddingProvider
from kg_processor.ports.extraction import EntityExtractor, RelationExtractor, RelationVerifier
from kg_processor.ports.file_source import FileSource
from kg_processor.ports.graph_manifest_publisher import GraphManifestPublisher
from kg_processor.ports.graph_writer import GraphWriter
from kg_processor.ports.llm import LlmProvider, StructuredCompletionProvider
from kg_processor.ports.ocr import OCR_SUPPORTED_SUFFIXES, OcrProvider

if TYPE_CHECKING:
    from kg_processor.adapters.distributed.postgres import PostgresDistributedStore
    from kg_processor.adapters.jobs.snowflake import SnowflakeJobManager
    from kg_processor.adapters.snowflake import SnowflakeConnectionConfig
    from kg_processor.adapters.writers.local_artifacts import LocalArtifactsWriter
    from kg_processor.application.consumption import ConsumptionCollector
    from kg_processor.application.pipeline import KgProcessorPipeline
    from kg_processor.application.progress import ProgressSink
    from kg_processor.application.revalidation import ReplayExtractors


def worker_pool_connections(slots: int) -> int:
    """The coordination connections a worker process with ``slots`` slots pools.

    A transaction takes milliseconds of a task that takes seconds, so slots
    seldom transact at the same moment. Two connections serve the claiming
    loop, lease renewals and one slot's transition; one more for every four
    slots absorbs the moments they coincide. Every pod of an autoscaled pool
    holds up to this many, so the pool spends the shared database's connection
    budget in proportion to what a pod does rather than to its slot count.
    """

    return 2 + slots // 4


def build_distributed_store(settings: Settings) -> PostgresDistributedStore:
    """Build the durable coordination/artifact adapter for distributed commands."""

    from kg_processor.adapters.distributed.postgres import PostgresDistributedStore

    if not settings.distributed.database_url:
        raise ValueError(
            "distributed execution requires distributed.database_url or KG_DISTRIBUTED_DATABASE_URL"
        )
    blob_store = build_blob_store(settings)
    return PostgresDistributedStore(
        settings.distributed.database_url,
        compression_level=settings.distributed.artifact_compression_level,
        max_artifact_bytes=settings.distributed.max_artifact_bytes,
        blob_store=blob_store,
        max_connections=worker_pool_connections(settings.distributed.worker_slots),
    )


def build_blob_store(settings: Settings) -> BlobStore | None:
    """Build shared immutable object storage when a distributed URI is configured."""

    from kg_processor.adapters.distributed.local_blob import LocalBlobStore
    from kg_processor.adapters.distributed.s3_blob import S3BlobStore, S3BlobStoreConfig

    uri = settings.distributed.artifact_uri
    if not uri:
        return None
    if uri.startswith("file://"):
        return LocalBlobStore(uri)
    if uri.startswith("s3://"):
        return S3BlobStore(
            S3BlobStoreConfig(
                root_uri=uri,
                endpoint_url=settings.distributed.artifact_endpoint_url,
                access_key_id=settings.distributed.artifact_access_key_id,
                secret_access_key=settings.distributed.artifact_secret_access_key,
                region=settings.distributed.artifact_region,
            )
        )
    raise ValueError("distributed.artifact_uri must use file:// or s3://")


def build_graph_manifest_publisher(
    settings: Settings,
    blob_store: BlobStore | None,
) -> GraphManifestPublisher | None:
    """Bind partitioned output publication at the adapter-composition boundary."""

    from kg_processor.adapters.writers.snowflake_manifest import TaskConfiguredManifestPublisher

    return TaskConfiguredManifestPublisher(settings, blob_store) if blob_store else None


def build_local_artifacts_writer(output_path: Path) -> LocalArtifactsWriter:
    """Build the canonical local writer for explicit export operations."""

    from kg_processor.adapters.writers.local_artifacts import LocalArtifactsWriter

    return LocalArtifactsWriter(output_path)


def build_pipeline(
    settings: Settings,
    claimed_files: list[JobFileClaim] | None = None,
    progress_sink: ProgressSink | None = None,
    write_scope_override: Literal["graph_snapshot", "file_batch"] | None = None,
    consumption: ConsumptionCollector | None = None,
) -> KgProcessorPipeline:
    """Assemble the provider-independent pipeline from the effective settings.

    Keeping this composition root outside the CLI guarantees local commands,
    Kubernetes workers, and direct application tests construct the same graph
    behavior. Concrete provider selection remains confined to this module.
    """

    from kg_processor.application.consumption import ConsumptionCollector
    from kg_processor.application.pipeline import KgProcessorPipeline
    from kg_processor.application.progress import JsonLineProgressSink

    collector = consumption or ConsumptionCollector(
        settings.job.graph_id, job_id=settings.job.job_id
    )
    llm = build_llm_provider(settings, collector)
    return KgProcessorPipeline(
        consumption=collector,
        settings=settings,
        # A claimed row already carries the authoritative checksum recorded at
        # submission, and the claim overwrites whatever discovery computed. Hashing
        # every staged file in every worker would therefore download the whole
        # stage repeatedly to produce a value that is immediately discarded.
        file_source=build_file_source(settings, content_hash=False if claimed_files else None),
        ocr=build_ocr_provider(settings),
        llm=llm,
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
        claimed_files=claimed_files,
        write_scope_override=write_scope_override,
        progress_sink=progress_sink or JsonLineProgressSink(),
        entity_extractor=build_entity_extractor(settings, llm),
        relation_extractor=build_relation_extractor(settings, llm),
        relation_verifier=build_relation_verifier(settings, llm),
        replay_extractors=build_replay_extractors(settings),
    )


class _UnusedDistributedDependency:
    """Fail clearly if stage-scoped composition accesses an unneeded provider."""

    def close(self) -> None:
        """Allow uniform pipeline cleanup when this placeholder was never used."""

        return None

    def __getattr__(self, name: str) -> Any:
        """Reject accidental cross-stage dependency use at the first method access."""

        raise RuntimeError(f"distributed stage attempted to use unavailable dependency: {name}")


def build_distributed_pipeline(
    settings: Settings,
    stages: set[TaskStage],
    progress_sink: ProgressSink | None = None,
) -> KgProcessorPipeline:
    """Build only providers required by a distributed worker's eligible stages.

    Preparation needs OCR/cache, extraction needs LLM/extractors, and
    local finalization needs LLM/embeddings/writer. Spark finalization runs through
    its executor composition and therefore does not require those providers in the
    lightweight queue worker process.
    """

    from kg_processor.application.consumption import ConsumptionCollector
    from kg_processor.application.pipeline import KgProcessorPipeline
    from kg_processor.application.progress import JsonLineProgressSink

    unused = _UnusedDistributedDependency()
    needs_prepare = TaskStage.PREPARE_DOCUMENT in stages
    needs_extract = bool(
        {
            TaskStage.EXTRACT_DOCUMENT_CONTEXT,
            TaskStage.EXTRACT_ENTITY_WINDOW,
            TaskStage.EXTRACT_RELATION_WINDOW,
            TaskStage.REVALIDATE_DOCUMENT_CONTEXT,
            TaskStage.REVALIDATE_ENTITY_WINDOW,
            TaskStage.REVALIDATE_RELATION_WINDOW,
            TaskStage.REVALIDATE_DOCUMENT,
        }
        & stages
    )
    spark_finalization = settings.distributed.finalization_engine == "spark" or (
        settings.distributed.finalization_engine == "auto"
        and settings.runtime.runtime == "kubernetes"
        and bool(settings.distributed.artifact_uri)
    )
    needs_local_finalize = TaskStage.FINALIZE_GRAPH in stages and not spark_finalization
    needs_llm = needs_extract or needs_local_finalize
    collector = ConsumptionCollector(settings.job.graph_id, job_id=settings.job.job_id)
    llm = build_llm_provider(settings, collector) if needs_llm else cast(LlmProvider, unused)
    return KgProcessorPipeline(
        consumption=collector,
        settings=settings,
        file_source=cast(FileSource, unused),
        ocr=build_ocr_provider(settings) if needs_prepare else cast(OcrProvider, unused),
        llm=llm,
        embeddings=(
            build_embedding_provider(settings)
            if needs_local_finalize
            else cast(EmbeddingProvider, unused)
        ),
        writer=build_writer(settings) if needs_local_finalize else cast(GraphWriter, unused),
        cache=build_cache(settings) if needs_prepare else None,
        progress_sink=progress_sink or JsonLineProgressSink(),
        entity_extractor=build_entity_extractor(settings, llm) if needs_extract else None,
        relation_extractor=build_relation_extractor(settings, llm) if needs_extract else None,
        relation_verifier=build_relation_verifier(settings, llm) if needs_extract else None,
        replay_extractors=build_replay_extractors(settings) if needs_extract else None,
    )


def build_replay_extractors(settings: Settings) -> ReplayExtractors:
    """Build the configured extractors over any structured provider, for revalidation.

    Revalidation answers extraction requests with a document's stored records,
    so it needs the extractors a new extraction would use - every flag of the
    configuration applied - built over a provider of its own.
    """

    def build(provider: StructuredCompletionProvider) -> tuple[EntityExtractor, RelationExtractor]:
        llm = cast(LlmProvider, provider)
        return build_entity_extractor(settings, llm), build_relation_extractor(settings, llm)

    return build


def build_file_source(settings: Settings, *, content_hash: bool | None = None) -> FileSource:
    """Build the configured file-source adapter."""

    from kg_processor.adapters.files.azure_blob import (
        AzureBlobFileSource,
        AzureBlobFileSourceConfig,
    )
    from kg_processor.adapters.files.local import LocalFileSource
    from kg_processor.adapters.files.manifest import ManifestFileSource
    from kg_processor.adapters.files.s3 import S3FileSource, S3FileSourceConfig
    from kg_processor.adapters.files.snowflake_stage import SnowflakeStageFileSource

    # Provider construction is intentionally explicit. Missing credentials or
    # unknown provider names fail here instead of silently swapping to a test or
    # local implementation.
    if settings.files.source == "local":
        return LocalFileSource(settings.files.input_path, settings.files.include_globs)
    if settings.files.source == "manifest":
        if not settings.files.manifest_path:
            raise ValueError("manifest file source requires files.manifest_path")
        return ManifestFileSource(
            settings.files.manifest_path,
            settings.files.include_globs,
            containment_root=settings.files.manifest_root,
        )
    if settings.files.source == "azure_blob":
        if not settings.azure_blob.container:
            raise ValueError("azure_blob file source requires azure_blob.container")
        return AzureBlobFileSource(
            AzureBlobFileSourceConfig(
                account_url=settings.azure_blob.account_url,
                connection_string=settings.azure_blob.connection_string,
                container=settings.azure_blob.container,
                prefix=settings.azure_blob.prefix,
                sas_token=settings.azure_blob.sas_token,
                download_path=settings.azure_blob.download_path,
            ),
            settings.files.include_globs,
        )
    if settings.files.source == "s3":
        if not settings.s3.bucket:
            raise ValueError("s3 file source requires s3.bucket")
        return S3FileSource(
            S3FileSourceConfig(
                bucket=settings.s3.bucket,
                prefix=settings.s3.prefix,
                endpoint_url=settings.s3.endpoint_url,
                region=settings.s3.region,
                download_path=settings.s3.download_path,
            ),
            settings.files.include_globs,
        )
    if settings.files.source == "snowflake_stage":
        if not settings.snowflake.stage:
            raise ValueError("snowflake_stage file source requires snowflake.stage")
        return SnowflakeStageFileSource(
            _snowflake_config(settings),
            settings.snowflake.stage,
            settings.files.stage_prefix,
            settings.files.include_globs,
            content_hash=(
                settings.files.stage_content_hash if content_hash is None else content_hash
            ),
        )
    raise ValueError(f"Unsupported file source: {settings.files.source}")


def build_ocr_provider(settings: Settings) -> OcrProvider:
    """Build the configured OCR provider adapter."""

    from kg_processor.adapters.ocr.fallback import FallbackOcrProvider

    if settings.ocr.provider == "fallback":
        primary_name = settings.ocr.fallback_primary_provider
        secondary_name = settings.ocr.fallback_secondary_provider
        if "fallback" in {primary_name, secondary_name}:
            raise ValueError("fallback OCR cannot contain another fallback provider")
        if primary_name == secondary_name:
            raise ValueError("fallback OCR primary and secondary providers must differ")
        return FallbackOcrProvider(
            _build_named_ocr_provider(settings, primary_name),
            _build_named_ocr_provider(settings, secondary_name),
            primary_name=primary_name,
            secondary_name=secondary_name,
            min_characters_per_page=settings.ocr.fallback_min_characters_per_page,
            max_sparse_page_ratio=settings.ocr.fallback_max_sparse_page_ratio,
            max_unbroken_text_ratio=settings.ocr.fallback_max_unbroken_text_ratio,
            max_fragmented_text_ratio=settings.ocr.fallback_max_fragmented_text_ratio,
            max_unspaced_page_ratio=settings.ocr.fallback_max_unspaced_page_ratio,
            max_interleaved_column_page_ratio=(
                settings.ocr.fallback_max_interleaved_column_page_ratio
            ),
            secondary_supported_suffixes=OCR_SUPPORTED_SUFFIXES.get(secondary_name),
        )
    return _build_named_ocr_provider(settings, settings.ocr.provider)


def _build_named_ocr_provider(settings: Settings, provider: str) -> OcrProvider:
    """Construct one concrete OCR adapter for direct or composed selection."""

    from kg_processor.adapters.ocr.builtin_text import BuiltinTextOcrProvider
    from kg_processor.adapters.ocr.generic_http import GenericHttpOcrProvider
    from kg_processor.adapters.ocr.mineru_api import MineruApiOcrProvider
    from kg_processor.adapters.ocr.mineru_internal import MineruInternalOcrProvider
    from kg_processor.adapters.ocr.snowflake_cortex import SnowflakeCortexOcrProvider
    from kg_processor.adapters.ocr.tesseract_internal import TesseractInternalOcrProvider

    # OCR providers all normalize into ParsedDocument, so the pipeline never
    # needs provider-specific branches after this factory boundary.
    if provider == "mineru_internal":
        return MineruInternalOcrProvider(settings.ocr.mineru_command)
    if provider == "mineru_api":
        if not settings.ocr.mineru_api_url:
            raise ValueError("mineru_api OCR requires mineru_api_url")
        return MineruApiOcrProvider(settings.ocr.mineru_api_url, settings.ocr.mineru_api_key)
    if provider == "builtin_text":
        return BuiltinTextOcrProvider()
    if provider == "generic_http":
        return GenericHttpOcrProvider(settings.generic_http_ocr)
    if provider == "tesseract_internal":
        return TesseractInternalOcrProvider(
            settings.ocr.tesseract_command,
            settings.ocr.tesseract_pdf_renderer_command,
            settings.ocr.tesseract_dpi,
        )
    if provider == "snowflake_cortex":
        return SnowflakeCortexOcrProvider(_snowflake_config(settings))
    raise ValueError(f"Unsupported OCR provider: {provider}")


def build_llm_provider(
    settings: Settings,
    consumption: ConsumptionCollector | None = None,
) -> LlmProvider:
    """Build the configured LLM adapter behind the shared provider interface.

    Required endpoint and credential checks fail here so application orchestration
    never needs provider-specific branches.
    """

    from kg_processor.application.metered_llm import MeteredLlmProvider

    provider = _build_raw_llm_provider(settings)
    if consumption is None:
        return provider
    # Metered outside any cache wrapper, so a cached response — which called
    # nobody — records nothing.
    return cast(
        LlmProvider,
        MeteredLlmProvider(
            provider,
            consumption,
            provider=settings.llm.provider,
            model=settings.llm.model,
        ),
    )


def _build_raw_llm_provider(settings: Settings) -> LlmProvider:
    """Select the configured adapter before any decoration."""

    from kg_processor.adapters.llm.azure_openai import AzureOpenAILlmProvider
    from kg_processor.adapters.llm.fake import FakeLlmProvider
    from kg_processor.adapters.llm.ollama import OllamaLlmProvider
    from kg_processor.adapters.llm.openai_compatible import OpenAICompatibleLlmProvider
    from kg_processor.adapters.llm.snowflake_cortex import SnowflakeCortexLlmProvider
    from kg_processor.adapters.llm.vllm_local import VllmLocalLlmProvider

    # Fake/vLLM/OpenAI/Azure/Cortex all implement the same structured LLM port;
    # selection stays in configuration rather than prompt/extraction code.
    if settings.llm.provider == "fake":
        return FakeLlmProvider()
    if settings.llm.provider == "openai_compatible":
        if not settings.llm.endpoint or not settings.llm.api_key:
            raise ValueError("openai_compatible LLM requires endpoint and api_key")
        return OpenAICompatibleLlmProvider(
            settings.llm.endpoint,
            settings.llm.api_key,
            settings.llm.model,
            timeout_seconds=settings.llm.timeout_seconds,
            max_output_tokens=settings.llm.max_output_tokens,
        )
    if settings.llm.provider == "ollama":
        if not settings.llm.endpoint:
            raise ValueError("ollama LLM requires endpoint")
        return OllamaLlmProvider(
            settings.llm.endpoint,
            settings.llm.model,
            timeout_seconds=settings.llm.timeout_seconds,
            api_key=settings.llm.api_key,
            context_window_tokens=settings.llm.context_window_tokens,
            max_output_tokens=settings.llm.max_output_tokens,
        )
    if settings.llm.provider == "azure_openai":
        if not settings.llm.endpoint or not settings.llm.api_key:
            raise ValueError("azure_openai LLM requires endpoint and api_key")
        return AzureOpenAILlmProvider(
            settings.llm.endpoint,
            settings.llm.api_key,
            settings.llm.api_version,
            settings.llm.model,
            timeout_seconds=settings.llm.timeout_seconds,
            max_output_tokens=settings.llm.max_output_tokens,
        )
    if settings.llm.provider == "vllm_local":
        if not settings.llm.endpoint:
            raise ValueError("vllm_local LLM requires endpoint")
        return VllmLocalLlmProvider(
            settings.llm.endpoint,
            settings.llm.api_key,
            settings.llm.model,
            timeout_seconds=settings.llm.timeout_seconds,
            max_output_tokens=settings.llm.max_output_tokens,
        )
    if settings.llm.provider == "snowflake_cortex":
        return SnowflakeCortexLlmProvider(
            _snowflake_config(settings),
            settings.llm.model,
            timeout_seconds=settings.llm.timeout_seconds,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")


def build_entity_extractor(settings: Settings, llm: LlmProvider) -> EntityExtractor:
    """Build the independently replaceable entity-recognition stage adapter.

    LLM and GLiNER implementations expose the same grounded mention contract.
    """

    from kg_processor.adapters.extraction.gliner import GlinerEntityExtractor
    from kg_processor.application.llm_extractors import LlmEntityExtractor

    if settings.extractors.entity_provider == "llm":
        return LlmEntityExtractor(
            _structured_llm(llm),
            settings.graph.deterministic_seed,
            grounding_fallback=settings.graph.entity_grounding_fallback,
        )
    if settings.extractors.entity_provider == "gliner":
        return GlinerEntityExtractor(
            settings.extractors.gliner_model,
            settings.extractors.gliner_threshold,
        )
    raise ValueError(f"Unsupported entity extractor: {settings.extractors.entity_provider}")


def build_relation_extractor(settings: Settings, llm: LlmProvider) -> RelationExtractor:
    """Build relation extraction independently from the entity-recognition algorithm.

    This boundary permits other relation engines without changing pipeline orchestration.
    """

    from kg_processor.application.llm_extractors import LlmRelationExtractor

    if settings.extractors.relation_provider == "llm":
        return LlmRelationExtractor(
            _structured_llm(llm),
            settings.graph.deterministic_seed,
            grounding_fallback=settings.graph.llm_grounding_fallback,
            new_endpoints=settings.graph.relation_new_endpoints,
            relabel=settings.graph.relation_relabel,
            hold_type_violations=settings.graph.relation_type_resolution,
        )
    raise ValueError(f"Unsupported relation extractor: {settings.extractors.relation_provider}")


def build_relation_verifier(settings: Settings, llm: LlmProvider) -> RelationVerifier:
    """Build the semantic entailment verifier behind its dedicated stage port.

    Verification stays replaceable and cannot introduce new relation candidates.
    """

    from kg_processor.application.llm_extractors import LlmRelationVerifier

    if settings.extractors.verifier_provider == "llm":
        return LlmRelationVerifier(_structured_llm(llm), settings.graph.deterministic_seed)
    raise ValueError(f"Unsupported relation verifier: {settings.extractors.verifier_provider}")


def _structured_llm(llm: LlmProvider) -> StructuredCompletionProvider:
    """Narrow a configured LLM to the strict capability required by two-pass stages.

    The explicit error avoids a late attribute failure after expensive OCR work.
    """

    if not isinstance(llm, StructuredCompletionProvider):
        raise ValueError("Selected LLM provider does not support strict structured completion")
    return llm


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the configured embedding provider adapter."""

    from kg_processor.adapters.embeddings.azure_openai import AzureOpenAIEmbeddingProvider
    from kg_processor.adapters.embeddings.hash import HashEmbeddingProvider
    from kg_processor.adapters.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
    from kg_processor.adapters.embeddings.sentence_transformers import (
        SentenceTransformersEmbeddingProvider,
    )
    from kg_processor.adapters.embeddings.snowflake_cortex import SnowflakeCortexEmbeddingProvider

    # Embedding providers are dimension-checked later by quality/inspect logic;
    # this factory only enforces the credentials needed to call the provider.
    if settings.embedding.provider == "hash":
        return HashEmbeddingProvider()
    if settings.embedding.provider == "sentence_transformers":
        return SentenceTransformersEmbeddingProvider(
            device=settings.embedding.device,
            torch_dtype=settings.embedding.torch_dtype,
        )
    if settings.embedding.provider == "openai_compatible":
        if not settings.embedding.endpoint or not settings.embedding.api_key:
            raise ValueError("openai_compatible embeddings require endpoint and api_key")
        return OpenAICompatibleEmbeddingProvider(
            settings.embedding.endpoint,
            settings.embedding.api_key,
        )
    if settings.embedding.provider == "azure_openai":
        if not settings.embedding.endpoint or not settings.embedding.api_key:
            raise ValueError("azure_openai embeddings require endpoint and api_key")
        return AzureOpenAIEmbeddingProvider(
            settings.embedding.endpoint,
            settings.embedding.api_key,
            settings.embedding.api_version,
        )
    if settings.embedding.provider == "snowflake_cortex":
        return SnowflakeCortexEmbeddingProvider(_snowflake_config(settings))
    raise ValueError(f"Unsupported embedding provider: {settings.embedding.provider}")


def build_writer(settings: Settings) -> GraphWriter:
    """Build the configured graph writer adapter."""

    from kg_processor.adapters.writers.local_artifacts import LocalArtifactsWriter
    from kg_processor.adapters.writers.snowflake_bulk import SnowflakeBulkWriter
    from kg_processor.adapters.writers.snowflake_direct import SnowflakeDirectWriter

    # Writers receive the same GraphWriteBatch shape. Local artifacts and
    # Snowflake tables should therefore stay schema-parity concerns, not
    # separate pipeline implementations.
    if settings.writer.provider == "local_artifacts":
        return LocalArtifactsWriter(settings.writer.output_path)
    if settings.writer.provider == "snowflake_direct":
        return SnowflakeDirectWriter(
            _snowflake_config(settings),
            settings.embedding.dimension,
        )
    if settings.writer.provider == "snowflake_bulk":
        if not settings.snowflake.bulk_stage:
            raise ValueError("snowflake_bulk writer requires snowflake.bulk_stage")
        return SnowflakeBulkWriter(
            _snowflake_config(settings),
            settings.embedding.dimension,
            settings.snowflake.bulk_stage,
            target_file_size_bytes=settings.snowflake.bulk_target_file_size_mb * 1024 * 1024,
        )
    raise ValueError(f"Unsupported graph writer: {settings.writer.provider}")


def build_cache(settings: Settings) -> PipelineCache | None:
    """Build the optional cache adapter, or None when caching is disabled."""

    from kg_processor.adapters.cache.local_json import LocalJsonCache
    from kg_processor.adapters.cache.snowflake import SnowflakeCache

    if settings.cache.provider == "none":
        return None
    if settings.cache.provider == "local":
        return LocalJsonCache(settings.cache.path)
    if settings.cache.provider == "snowflake":
        return SnowflakeCache(_snowflake_config(settings))
    raise ValueError(f"Unsupported cache provider: {settings.cache.provider}")


def build_job_manager(settings: Settings) -> SnowflakeJobManager | None:
    """Build a Snowflake job manager when leasing or file queues are enabled."""

    from kg_processor.adapters.jobs.snowflake import SnowflakeJobManager

    if not settings.job.use_lease and not settings.job.use_file_queue:
        return None
    return SnowflakeJobManager(_snowflake_config(settings))


def _snowflake_config(settings: Settings) -> SnowflakeConnectionConfig:
    from kg_processor.adapters.snowflake import SnowflakeConnectionConfig

    return SnowflakeConnectionConfig.from_settings(settings.snowflake)
