# SPDX-License-Identifier: Apache-2.0
"""Typed configuration model for the full local/on-prem/SPCS runtime.

Defaults intentionally describe the production local-open-source profile rather
than a test fallback. Smoke tests opt into fake/hash/builtin providers
explicitly, which keeps real runs from silently changing providers.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from kg_processor.config.provider_registry import ProviderKind, provider_names
from kg_processor.domain.consumption import RateCard
from kg_processor.domain.distributed import TaskStage
from kg_processor.ports.llm import DEFAULT_LLM_TIMEOUT_SECONDS

_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ENV_PLACEHOLDER_ONLY_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_MISSING_ENV_PLACEHOLDER = object()


class _SettingsModel(BaseModel):
    """Reject misspelled configuration keys at every nested settings boundary.

    Field names are accepted alongside their configuration aliases so a dump of
    these settings validates back into them. Distributed execution relies on that
    round trip: settings are serialized on the driver and rebuilt on each Spark
    executor, and a dump emits field names while ``extra="forbid"`` rejects
    anything the model does not declare.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RuntimeSettings(_SettingsModel):
    """Selects the deployment runtime profile without changing provider contracts."""

    runtime: Literal["local", "onprem", "spcs", "kubernetes"] = "local"


_OWNER_MAX_LENGTH = 200
_OWNER_PATTERN = re.compile(r"^[A-Za-z0-9._%+'-]+(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})?$")


class JobSettings(_SettingsModel):
    """Controls job identity, leasing, and file queue batching for resumable runs."""

    job_id: str = "local-job"
    graph_id: str = "local-graph"
    # The person a graph belongs to: their sign-in address. A distributed run
    # records it, and the console shows a graph only to its owner and the
    # people they share it with.
    owner: str | None = None
    use_lease: bool = False
    use_file_queue: bool = False
    lease_owner: str | None = None
    lease_seconds: int = Field(default=900, gt=0)
    file_batch_size: int = Field(default=100, gt=0)

    @field_validator("owner", mode="before")
    @classmethod
    def _owner_is_a_principal(cls, value: object) -> object:
        """One spelling per person - upper case, as the console keys principals."""

        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        text = str(value).strip()
        if len(text) > _OWNER_MAX_LENGTH or not _OWNER_PATTERN.match(text):
            raise ValueError("job.owner must be a sign-in address such as name@example.com")
        return text.upper()


class DistributedSettings(_SettingsModel):
    """Configure durable multi-process execution independently of Kubernetes.

    PostgreSQL provides transactional task leases and durable storage with one
    operational dependency.
    The application layer still targets task/artifact ports, so another backend can
    be introduced without changing graph stages or provider implementations.
    """

    database_url: str | None = None
    worker_id: str | None = None
    # A general-purpose worker claims every stage.
    worker_stages: list[TaskStage] = Field(default_factory=lambda: list(TaskStage))
    # How many tasks one worker process runs at once. A pipeline task spends
    # most of its time waiting on a model, a parser or object storage, so one
    # process keeps several in flight; each slot claims, heartbeats and
    # completes its own task.
    worker_slots: int = Field(default=8, gt=0)
    # Live workers renew ownership in the background, so this window controls
    # crash detection rather than maximum task duration. Five minutes tolerates
    # brief coordination outages while returning work from a dead node promptly.
    lease_seconds: int = Field(default=300, gt=0)
    poll_interval_seconds: float = Field(default=1.0, ge=0)
    retry_delay_seconds: float = Field(default=15.0, ge=0)
    max_attempts: int = Field(default=3, gt=0)
    artifact_compression_level: int = Field(default=6, ge=0, le=9)
    max_artifact_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    artifact_uri: str | None = None
    artifact_endpoint_url: str | None = None
    artifact_access_key_id: str | None = None
    artifact_secret_access_key: str | None = None
    artifact_region: str = "us-east-1"
    finalization_engine: Literal["auto", "local", "spark"] = "auto"
    spark_master: str = "local[*]"
    spark_image: str | None = None
    spark_namespace: str = "default"
    spark_service_account: str = "flakegraph-spark"
    spark_executor_pod_template: str = "/etc/flakegraph-spark/executor-pod-template.yaml"
    # The driver's provider credentials as Kubernetes "secret-name:key"
    # references, including gateway virtual keys and renamed provider keys.
    # Executors are given these references rather than the values, which Spark
    # would otherwise write into every executor Pod spec in clear text.
    spark_provider_secret_refs: dict[str, str] = Field(default_factory=dict)

    spark_executor_instances: int = Field(default=4, gt=0)
    spark_executor_cores: int = Field(default=4, gt=0)
    spark_executor_memory: str = "8g"
    spark_executor_memory_overhead: str = "8g"
    spark_shuffle_partitions: int = Field(default=0, ge=0)

    @field_validator("spark_provider_secret_refs", mode="before")
    @classmethod
    def parse_provider_secret_refs(cls, value: Any) -> Any:
        """Read JSON environment input without ever embedding credential values."""

        return json.loads(value) if isinstance(value, str) else value

    @field_validator(
        "database_url",
        "worker_id",
        "artifact_uri",
        "artifact_endpoint_url",
        "artifact_access_key_id",
        "artifact_secret_access_key",
        "spark_image",
    )
    @classmethod
    def optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        """Reject present-but-empty values that would fail later and less clearly."""

        if value is not None and not value.strip():
            raise ValueError("distributed text settings must not be blank")
        return value

    @field_validator("worker_stages", mode="before")
    @classmethod
    def worker_stages_from_comma_list(cls, value: object) -> object:
        """Accept the comma-separated spelling an environment variable carries."""

        return _str_to_list(value) if isinstance(value, str) else value

    @field_validator("worker_stages")
    @classmethod
    def worker_stages_must_not_be_empty(cls, value: list[TaskStage]) -> list[TaskStage]:
        """Ensure a worker can claim at least one kind of useful work."""

        if not value:
            raise ValueError("distributed.worker_stages must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("distributed.worker_stages must not contain duplicates")
        return value

    @field_validator(
        "artifact_region",
        "spark_master",
        "spark_namespace",
        "spark_service_account",
        "spark_executor_pod_template",
        "spark_executor_memory",
        "spark_executor_memory_overhead",
    )
    @classmethod
    def required_distributed_text_must_not_be_blank(cls, value: str) -> str:
        """Reject blank execution values before starting a remote data engine."""

        if not value.strip():
            raise ValueError("required distributed text settings must not be blank")
        return value

    @model_validator(mode="after")
    def scalable_finalization_requires_object_storage(self) -> DistributedSettings:
        """Require shared object storage when Spark is explicitly selected.

        Auto mode retains the zero-configuration local path when no shared store
        is configured. Kubernetes charts configure both together for the scalable
        path, avoiding an arbitrary document-count switch.
        """

        if self.finalization_engine == "spark" and not self.artifact_uri:
            raise ValueError("spark finalization requires distributed.artifact_uri")
        return self


class FileSettings(_SettingsModel):
    """Describes where source documents are discovered before OCR begins."""

    source: Literal["local", "manifest", "stage", "snowflake_stage", "azure_blob", "s3"] = "local"
    # Keep zero-configuration discovery independent from bundled benchmark
    # corpora. User-facing profiles and the app normally replace this root.
    input_path: Path = Path("data")
    manifest_path: Path | None = None
    # A manifest names files by path, relative to itself or absolute, and an
    # absolute path can name anything the worker can read. When the manifest
    # comes from a user rather than the operator, this root confines every row
    # to one directory; rows resolving outside it are rejected before OCR.
    manifest_root: Path | None = None
    stage_prefix: str | None = None
    include_globs: list[str] = Field(default_factory=lambda: ["**/*"])
    # Folder and file names say what a document belongs to - a project, an
    # experiment, a batch - which a file named IMG_0392.jpg cannot. Each
    # document then starts with a page 0 naming its location relative to the
    # source root, which extraction reads and grounds like any other text.
    location_page: bool = True
    # Snowflake reports its own MD5 for staged files, which cannot be compared
    # with the content hashes every other source records. Reading the bytes once
    # keeps document identity consistent across runtimes; disable it only where
    # cross-runtime comparison and gold evaluation are not needed.
    stage_content_hash: bool = True

    @field_validator("source", mode="before")
    @classmethod
    def source_must_be_registered(cls, value: object) -> object:
        """Check a file-source name against the shared provider catalog."""

        return _validate_provider_name(value, "file_source")


class AzureBlobSettings(_SettingsModel):
    """Configures optional Azure Blob input discovery and local download staging."""

    account_url: str | None = None
    connection_string: str | None = None
    container: str | None = None
    prefix: str | None = None
    sas_token: str | None = None
    download_path: Path = Path("out/azure-blob")


class S3Settings(_SettingsModel):
    """Configures S3-compatible input discovery and local download staging."""

    bucket: str | None = None
    prefix: str | None = None
    endpoint_url: str | None = None
    region: str | None = None
    download_path: Path = Path("out/s3")


class OcrSettings(_SettingsModel):
    """Configures OCR provider selection plus provider-specific execution knobs."""

    # The text layer a document already carries needs nothing installed, so it
    # is what a run gets by default. Scanned corpora select `fallback` or
    # `mineru_internal`, both of which need the separately installed MinerU CLI.
    provider: str = "builtin_text"
    fallback_primary_provider: str = "builtin_text"
    fallback_secondary_provider: str = "mineru_internal"
    fallback_min_characters_per_page: int = Field(default=80, ge=1)
    fallback_max_sparse_page_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    fallback_max_unbroken_text_ratio: float = Field(default=0.5, ge=0.0, lt=1.0)
    fallback_max_fragmented_text_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    # Share of measured pages whose text layer lost most of its word spacing,
    # leaving runs of words written together, before the secondary parser takes over.
    fallback_max_unspaced_page_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    # Share of a PDF's pages whose text layer interleaves two columns line by
    # line before the secondary parser, which reads the layout, takes over.
    fallback_max_interleaved_column_page_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    language: str | None = None
    page_range: str | None = None
    timeout_seconds: int = Field(default=900, gt=0)
    model_cache_dir: Path | None = None
    mineru_command: str = "mineru"
    mineru_method: Literal["auto", "txt", "ocr"] = "auto"
    # The parsing pool is built pipeline-only, and MinerU's own default is a
    # VLM backend this image cannot run. Default to what works rather than to
    # what the upstream picks.
    mineru_backend: str | None = "pipeline"
    mineru_effort: str | None = None
    mineru_api_url: str | None = None
    mineru_api_key: str | None = None
    mineru_server_url: str | None = None
    mineru_start_page_id: int | None = Field(default=None, ge=0)
    mineru_end_page_id: int | None = Field(default=None, ge=0)
    mineru_formula: bool | None = None
    mineru_table: bool | None = None
    mineru_image_analysis: bool | None = None
    mineru_client_side_output_generation: bool = False
    tesseract_command: str = "tesseract"
    tesseract_pdf_renderer_command: str = "pdftoppm"
    tesseract_dpi: int = Field(default=300, gt=0)
    snowflake_parse_mode: Literal["OCR", "LAYOUT"] = "OCR"
    snowflake_extract_images: bool = False
    snowflake_page_split: bool = True

    def consumption_model(self, provider: str | None = None) -> str:
        """Return the name document parsing should be billed under.

        Most OCR providers charge one way, so the provider name identifies the
        rate. Snowflake does not: AI_PARSE_DOCUMENT bills Layout at more than
        five times the OCR rate, and a card that cannot tell them apart has to
        pick one and be wrong about the other. The mode therefore travels with
        the model name into the consumption record. A fallback policy passes
        the provider that actually parsed the document.
        """

        provider = provider or self.provider
        if provider == "snowflake_cortex":
            return f"{provider}-{self.snowflake_parse_mode.lower()}"
        return provider

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Check an OCR provider name against the shared provider catalog."""

        return _validate_provider_name(value, "ocr")

    @field_validator("fallback_primary_provider", "fallback_secondary_provider", mode="before")
    @classmethod
    def fallback_providers_must_be_registered(cls, value: object) -> object:
        """Validate composed OCR provider names through the shared catalog."""

        return _validate_provider_name(value, "ocr")

    @field_validator("snowflake_parse_mode", mode="before")
    @classmethod
    def snowflake_parse_mode_uppercase(cls, value: object) -> object:
        """Normalize Snowflake Cortex parse modes to the literals accepted by SQL."""

        return value.upper() if isinstance(value, str) else value


class GenericHttpOcrSettings(_SettingsModel):
    """Maps an arbitrary HTTP OCR response into the normalized document schema."""

    max_response_bytes: int = Field(default=25 * 1024 * 1024, ge=0)
    endpoint: str | None = None
    api_key: str | None = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    file_field: str = "file"
    result_path: str | None = None
    pages_path: str = "pages"
    page_number_path: str = "page_number"
    markdown_path: str = "markdown"
    raw_text_path: str = "raw_text"
    detected_language_path: str = "detected_language"
    blocks_path: str = "blocks"
    block_id_path: str = "id"
    block_kind_path: str = "kind"
    block_text_path: str = "text"
    block_bbox_path: str = "bbox"
    block_confidence_path: str | None = "confidence"
    block_metadata_path: str | None = "metadata"
    assets_path: str = "assets"
    asset_id_path: str = "id"
    asset_kind_path: str = "kind"
    asset_uri_path: str = "uri"
    asset_page_number_path: str = "page_number"
    asset_confidence_path: str | None = "confidence"
    asset_metadata_path: str | None = "metadata"
    warnings_path: str = "warnings"
    error_path: str = "error"
    status_path: str = "status"


class LlmSettings(_SettingsModel):
    """Select the graph-extraction LLM transport, model, credentials, and timeout.

    Provider-specific construction remains in factories; this model only validates
    the portable configuration shared by local, hosted, and Snowflake backends.
    """

    provider: str = "openai_compatible"
    endpoint: str | None = None
    # A portable hosted default; configs/app-defaults.yaml selects a local
    # Ollama model instead so a laptop run needs no account.
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    api_version: str = "2025-01-01-preview"
    timeout_seconds: int = Field(default=DEFAULT_LLM_TIMEOUT_SECONDS, gt=0)
    context_window_tokens: int = Field(default=32768, gt=0)
    # The ceiling on a single completion. On a shared fleet this is the primary
    # control over how long interactive work waits: a queue-jumping request is
    # served after the next running request finishes, so capping how long any one
    # of them can run is what bounds that wait. Left unset, each adapter keeps
    # its own budget.
    max_output_tokens: int | None = Field(default=None, gt=0)

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Validate LLM provider names before adapter construction."""

        return _validate_provider_name(value, "llm")


class EmbeddingSettings(_SettingsModel):
    """Selects the embedding backend used for chunks, nodes, edges, and communities."""

    provider: str = "sentence_transformers"
    endpoint: str | None = None
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    api_key: str | None = None
    api_version: str = "2025-01-01-preview"
    # A Snowflake destination creates its VECTOR columns at this width the
    # first time the graph tables are made, and a later change cannot be
    # written into them; keep it equal to what the embedding model produces.
    dimension: int = Field(default=1024, gt=0)
    batch_size: int = Field(default=32, gt=0)
    device: str | None = None
    # The precision a locally loaded model is held in. Every Spark Python
    # worker keeps its own copy, so the choice is a memory budget as much as a
    # numeric one: a 0.6B model is ~5 GiB per worker in float32 and about half
    # that in float16. "auto" leaves it to the library, which loads float32.
    # It takes part in the compatibility digest because it changes the vectors.
    torch_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Validate embedding provider names before adapter construction."""

        return _validate_provider_name(value, "embedding")


class ConsumptionSettings(_SettingsModel):
    """Prices used to turn recorded usage into cost.

    Rates are configuration rather than code: providers change prices, and a run
    already recorded its usage, so correcting the table reprices history without
    re-running anything. Keys are ``provider:model``, with ``provider:*`` as a
    deliberate fallback so a new model is priced approximately rather than
    silently counted as free.
    """

    rates: dict[str, dict[str, float]] = Field(default_factory=dict)
    # The hosted model local work is priced against. Without one, a saving would
    # have no stated alternative and could not be checked.
    local_reference: str | None = None
    # What one Snowflake AI credit costs this account. Deliberately zero by
    # default: the figure comes from the customer's contract, and a guessed one
    # would report a confident price that is quietly wrong. Until it is set, the
    # Cortex rates below leave their calls counted as unpriced, which the
    # application says out loud rather than showing as free.
    usd_per_credit: float = 0.0

    def rate_card(self) -> RateCard:
        """Build the domain rate card this configuration describes."""

        return RateCard(
            rates=self.rates,
            local_reference=self.local_reference,
            usd_per_credit=self.usd_per_credit,
        )


class OntologySettings(_SettingsModel):
    """Select the optional reviewable ontology profile used across graph stages.

    When omitted, the loader derives a compatible fallback from graph entity and
    relation settings so generic configurations remain operational.

    ``profile`` carries the resolved profile itself. Runtimes that cannot see the
    author's filesystem — a Snowpark container reading a staged specification —
    receive the ontology by value, because a path alone would dangle there.
    """

    profile_path: Path | None = None
    profile: dict[str, Any] | None = None


class ExtractorSettings(_SettingsModel):
    """Select independently replaceable algorithms for two-pass extraction stages.

    Entity recognition may use an LLM or local GLiNER while relation extraction and
    verification retain their own interfaces and provider choices.
    """

    entity_provider: Literal["llm", "gliner"] = "llm"
    relation_provider: Literal["llm"] = "llm"
    verifier_provider: Literal["llm"] = "llm"
    gliner_model: str = "urchade/gliner_multi-v2.1"
    gliner_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class GraphSettings(_SettingsModel):
    """Define provider-neutral graph quality, throughput, and safety policy.

    Centralizing these defaults keeps OCR and LLM provider profiles behaviorally
    consistent. Explicit overrides remain available for measured corpus or backend
    constraints without duplicating policy across YAML files.
    """

    # These defaults are the provider-neutral quality profile. Provider YAML
    # files should normally select transports and credentials only; keeping the
    # extraction behavior here prevents provider profiles from changing graph
    # semantics merely because a transport configuration was copied.
    chunk_token_size: int = Field(default=500, gt=0)
    chunk_token_overlap: int = Field(default=60, ge=0)
    # A compact window keeps the structured extraction task exhaustive. Larger
    # section-sized requests reduce provider calls, but on scientific papers they
    # omit substantial entities and relations even when the response remains
    # below its record limit. Fleet execution supplies throughput by leasing these
    # independent windows dynamically rather than weakening per-window recall.
    extraction_window_tokens: int = Field(default=700, gt=0)
    max_chunks_per_llm_call: int = Field(default=2, gt=0)
    document_context_tokens: int = Field(default=1200, gt=0)
    document_context_max_chunks: int = Field(default=3, gt=0)
    max_document_context_entities: int = Field(default=3, gt=0)
    # How many windows one process extracts at once when a single call covers a
    # whole document: a local run, or a document's context pass. A fleet runs
    # each window as its own task, so there the worker's slots set the pace.
    extraction_parallelism: int = Field(default=2, gt=0)
    # Each window receives a completeness/gleaning pass, so two bounded responses
    # can recover more records without risking one oversized, truncated JSON body.
    max_entities_per_batch: int = Field(default=40, gt=0)
    max_relations_per_batch: int = Field(default=40, gt=0)
    gleaning_max_passes: int = Field(default=1, ge=0)
    gleaning_saturation_threshold: int = Field(default=10, gt=0)
    gleaning_min_uncovered_tokens: int = Field(default=40, gt=0)
    # A relation call that fills its record limit may have stopped early; ask
    # for the rest this many times before moving on.
    relation_continuation_max_passes: int = Field(default=2, ge=0)
    # A relation call over at least this many entities that returns nothing is
    # retried once with another seed; 0 turns the retry off.
    relation_empty_retry_min_entities: int = Field(default=8, ge=0)
    # One extra call per window points at the lines that state relations
    # string grounding could not place; code still confirms both endpoints.
    llm_grounding_fallback: bool = False
    # A relation response may add an endpoint the window's entities lack - a
    # named thing of an ontology type - grounded like any entity. Off by
    # default: on a benchmark corpus it cost more relations than it added.
    relation_new_endpoints: bool = False
    # A relation whose type rules its two entities break is offered the
    # relations the ontology allows between them; the model picks one or none.
    relation_relabel: bool = False
    # A catalogue dataset's rows state relations through its columns - a
    # thing's supplier, the form it is used in - as its profile maps them; each
    # is grounded in its row. Off, a catalogue gets its entities only.
    catalogue_relations: bool = True
    # A relation rejected only for its entity types is still grounded and held
    # to finalization, which admits it when an endpoint's name is typed, in at
    # least this many other documents, as a type the relation's rules allow.
    relation_type_resolution: bool = False
    relation_type_resolution_min_documents: int = Field(default=2, ge=1)
    # One extra call per window points at the lines that name entities string
    # grounding could not place; code checks the lines share the name's stem.
    entity_grounding_fallback: bool = False
    verify_relations: bool = True
    # The verification call for a window also judges the window's entities: a
    # generic class term, heading, placeholder, value, or wrongly typed name is
    # rejected, and relations that used it fall away with it.
    verify_entities: bool = True
    verification_min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    entity_resolution_enabled: bool = True
    resolution_lexical_auto_merge: float = Field(default=0.96, ge=0.0, le=1.0)
    resolution_embedding_auto_merge: float = Field(default=0.94, ge=0.0, le=1.0)
    resolution_candidate_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    resolution_embedding_lexical_floor: float = Field(default=0.45, ge=0.0, le=1.0)
    resolution_max_candidates_per_mention: int = Field(default=3, gt=0)
    # A decision is a compact boolean/confidence/reason tuple. Forty bounded
    # candidates fit comfortably in one strict response and halve provider round
    # trips without changing candidate generation or merge thresholds.
    resolution_adjudication_batch_size: int = Field(default=40, gt=0)
    resolution_llm_merge_min_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    deterministic_seed: int = 17
    min_entity_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    min_relation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # A grounded entity remains useful even when no relation survives extraction
    # or verification. Consumers can derive a connected/core view later; deleting
    # isolates here irreversibly couples entity recall to relation recall.
    drop_isolated_entities: bool = False
    min_entity_name_length: int = Field(default=2, gt=0)
    require_relation_endpoint_grounding: bool = True
    relation_weight_max: float = Field(default=10.0, gt=0)
    min_community_size: int = Field(default=2, gt=0)
    max_community_size: int = Field(default=50, gt=0)
    # Louvain ships with NetworkX under a BSD licence and is the default. Leiden
    # needs the GPL-licensed igraph and leidenalg packages, installed by the
    # `leiden` extra, so selecting it is a licensing decision as much as an
    # algorithmic one.
    community_algorithm: Literal["louvain", "leiden"] = "louvain"
    community_resolution: float = Field(default=1.0, gt=0)
    community_co_mention_weight: float = Field(default=0.05, gt=0)
    # Provider batches finalization keeps in flight: identity adjudication,
    # description merges, community reports and, on Spark, embeddings. The
    # local finalizer runs this many at once. Spark divides it across its
    # execution slots (executor instances x cores), each running at least one,
    # so a multiple of the slot count is the figure to set; see
    # docs/kubernetes-fleet.md for sizing it against serving capacity.
    finalization_provider_concurrency: int = Field(default=2, gt=0)
    description_merge_min_observations: int = Field(default=2, gt=0)
    description_merge_max_descriptions: int = Field(default=8, gt=0)
    description_merge_max_evidence: int = Field(default=5, gt=0)
    fail_on_quality_error: bool = True
    entity_types: list[str] = Field(
        default_factory=lambda: [
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
            "PRODUCT",
            "EVENT",
            "CONCEPT",
            "DATE",
            "QUANTITY",
        ]
    )
    relation_types: list[str] | None = None
    entity_blocklist: list[str] = Field(
        default_factory=lambda: [
            "chapter",
            "section",
            "introduction",
            "conclusion",
            "paragraph",
            "document",
            "page",
        ]
    )

    @field_validator("entity_blocklist", mode="before")
    @classmethod
    def entity_blocklist_from_comma_list(cls, value: object) -> object:
        """Accept the comma-separated spelling an environment variable carries."""

        return _str_to_list(value) if isinstance(value, str) else value

    @field_validator("chunk_token_overlap")
    @classmethod
    def overlap_smaller_than_chunk(cls, value: int, info: Any) -> int:
        """Prevent overlap settings that would duplicate or block chunk advancement."""

        chunk_size = info.data.get("chunk_token_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_token_overlap must be smaller than chunk_token_size")
        return value

    @model_validator(mode="after")
    def community_size_bounds_must_overlap(self) -> GraphSettings:
        """Reject community bounds that make every detected group ineligible."""

        if self.min_community_size > self.max_community_size:
            raise ValueError("min_community_size must not exceed max_community_size")
        return self


class WriterSettings(_SettingsModel):
    """Selects where the assembled graph artifacts are persisted."""

    provider: str = "local_artifacts"
    output_path: Path = Path("out/kg")

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Validate writer provider names before adapter construction."""

        return _validate_provider_name(value, "writer")


class CacheSettings(_SettingsModel):
    """Controls optional OCR and extraction result caching."""

    provider: Literal["none", "local", "snowflake"] = "none"
    path: Path = Path("out/cache")

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Validate cache provider names before adapter construction."""

        return _validate_provider_name(value, "cache")


class SnowflakeSettings(_SettingsModel):
    """Holds Snowflake connection, stage, image, and SPCS deployment settings.

    ``password_environment_variable`` and ``oauth_token_environment_variable``
    name where a secret lives instead of carrying it. Loading resolves the named
    variable into the matching credential when one is not configured directly,
    which lets a distributed plan travel to a finalizer worker as a reference
    rather than as a value.
    """

    account: str | None = None
    host: str | None = None
    user: str | None = None
    password: str | None = None
    password_environment_variable: str | None = None
    authenticator: str | None = None
    private_key_path: Path | None = None
    oauth_token: str | None = None
    oauth_token_environment_variable: str | None = None
    oauth_token_path: Path | None = Path("/snowflake/session/token")
    store_temporary_credential: bool = False
    database: str | None = None
    schema_name: str | None = Field(default=None, alias="schema")
    role: str | None = None
    warehouse: str | None = None
    stage: str | None = None
    bulk_stage: str | None = None
    image_repository: str | None = None
    image_name: str = "flakegraph:latest"
    image_digest: str | None = None
    compute_pool: str | None = None
    compute_pool_instance_family: str | None = None
    compute_pool_min_nodes: int = Field(default=1, gt=0)
    compute_pool_max_nodes: int = Field(default=1, gt=0)
    service_name: str = "KG_PROCESSOR_JOB"
    service_spec_stage: str | None = None
    service_cpu_request: str = "500m"
    service_cpu_limit: str = "1"
    service_memory_request: str = "3Gi"
    service_memory_limit: str = "5Gi"
    service_gpu_count: int = Field(default=0, ge=0)
    bulk_target_file_size_mb: int = Field(default=128, gt=0)

    @field_validator("image_digest")
    @classmethod
    def image_digest_must_be_sha256(cls, value: str | None) -> str | None:
        """Accept only immutable OCI SHA-256 image digests when one is supplied."""

        if value is None:
            return None
        normalized = value.strip().lower()
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", normalized):
            raise ValueError("image_digest must be a sha256:<64 hex chars> OCI digest")
        return normalized

    @model_validator(mode="after")
    def compute_pool_max_must_cover_min(self) -> SnowflakeSettings:
        """Ensure rendered compute pools can satisfy their configured minimum size."""

        if self.compute_pool_max_nodes < self.compute_pool_min_nodes:
            raise ValueError("compute_pool_max_nodes must be greater than or equal to min nodes")
        return self


class Settings(_SettingsModel):
    """Provide one validated configuration consumed by CLI, factories, and workers.

    The object is the single source of effective runtime behavior after defaults,
    YAML, environment aliases, and explicit command overrides have been merged.
    """

    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    job: JobSettings = Field(default_factory=JobSettings)
    distributed: DistributedSettings = Field(default_factory=DistributedSettings)
    files: FileSettings = Field(default_factory=FileSettings)
    azure_blob: AzureBlobSettings = Field(default_factory=AzureBlobSettings)
    s3: S3Settings = Field(default_factory=S3Settings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    generic_http_ocr: GenericHttpOcrSettings = Field(default_factory=GenericHttpOcrSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    ontology: OntologySettings = Field(default_factory=OntologySettings)
    consumption: ConsumptionSettings = Field(default_factory=ConsumptionSettings)
    extractors: ExtractorSettings = Field(default_factory=ExtractorSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    writer: WriterSettings = Field(default_factory=WriterSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    snowflake: SnowflakeSettings = Field(default_factory=SnowflakeSettings)

    @model_validator(mode="after")
    def folder_entity_needs_location_page(self) -> Settings:
        """An inline profile that ties documents to folders needs the location page.

        The folder entity and its relations are grounded in the page's folder
        line; without the page they have no evidence. A profile read from a
        file is checked where it is loaded.
        """

        profile = self.ontology.profile or {}
        require_location_page_for_folder_entity(
            profile.get("folder_entity") is not None, self.files.location_page
        )
        return self

    @classmethod
    def load(
        cls,
        config_file: Path | None = None,
        env: dict[str, str] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Settings:
        """Load YAML, environment variables, and explicit overrides into settings."""

        env = env if env is not None else dict(os.environ)
        selected_file = config_file or (
            Path(env["KG_CONFIG_FILE"]) if env.get("KG_CONFIG_FILE") else None
        )
        data: dict[str, Any] = _ambient_snowflake_from_env(env)
        if selected_file:
            data = _deep_update(data, _read_yaml(selected_file, env))
        inline_config = env.get("KG_CONFIG_JSON")
        if inline_config:
            data = _deep_update(data, _read_inline_json(inline_config))
        data = _deep_update(data, _from_env(env))
        if overrides:
            data = _deep_update(data, overrides)
        _resolve_snowflake_credential_references(data, env)
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(_safe_validation_error(exc)) from None


def _resolve_snowflake_credential_references(data: dict[str, Any], env: dict[str, str]) -> None:
    """Read Snowflake secrets from the environment variables the config names.

    ``password_environment_variable`` and ``oauth_token_environment_variable``
    let a configuration record where a secret lives without embedding it, which
    is what a distributed plan propagates to finalizer workers. Resolving them
    here means every connection path receives the same credential the plan
    describes. A directly configured secret wins, and a named variable that is
    absent leaves the credential unset for preflight to report.
    """

    snowflake = data.get("snowflake")
    if not isinstance(snowflake, dict):
        return
    for reference_key, secret_key in (
        ("password_environment_variable", "password"),
        ("oauth_token_environment_variable", "oauth_token"),
    ):
        variable = snowflake.get(reference_key)
        if not isinstance(variable, str) or not variable or snowflake.get(secret_key):
            continue
        secret = env.get(variable)
        if secret:
            snowflake[secret_key] = secret


def _read_inline_json(value: str) -> dict[str, Any]:
    """Parse an orchestrator-provided configuration without exposing its value.

    Container platforms can inject the complete validated runtime contract through
    ``KG_CONFIG_JSON`` when the image's baked YAML is only a portable baseline.
    Parse failures identify the variable without echoing operator-specific content.
    """

    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"KG_CONFIG_JSON must contain valid JSON: {exc.msg}") from None
    if not isinstance(loaded, dict):
        raise ValueError("KG_CONFIG_JSON must contain a JSON object")
    return loaded


def require_location_page_for_folder_entity(folder_entity: bool, location_page: bool) -> None:
    """Refuse a folder entity without the location page that grounds it."""

    if folder_entity and not location_page:
        raise ValueError(
            "ontology folder_entity ties documents to their folder through each "
            "document's location page; files.location_page must be on"
        )


def _safe_validation_error(exc: ValidationError) -> str:
    """Render configuration failures without echoing rejected input values.

    Pydantic's default string includes ``input_value`` which can contain complete
    configuration sections and plaintext credentials. Locations and validator
    messages provide the useful diagnosis without reproducing those inputs.
    """

    details = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in error.get("loc", ())) or "configuration"
        details.append(f"{location}: {error.get('msg', 'invalid value')}")
    return "Invalid FlakeGraph configuration: " + "; ".join(details)


def _read_yaml(path: Path, env: dict[str, str]) -> dict[str, Any]:
    """Parse a configuration file into a mapping, reporting parse failures uniformly.

    Callers distinguish configuration problems from defects by catching
    ``ValueError``, so a malformed document is reported the same way as a
    well-formed document of the wrong shape.
    """

    with path.open("r", encoding="utf-8") as handle:
        try:
            loaded = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Configuration file is not valid YAML: {path}: {exc}") from None
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    interpolated = _interpolate_env(loaded, env)
    if not isinstance(interpolated, dict):
        raise ValueError(f"Configuration file must contain a mapping after interpolation: {path}")
    return interpolated


def _interpolate_env(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            interpolated = _interpolate_env(item, env)
            if interpolated is _MISSING_ENV_PLACEHOLDER:
                continue
            result[key] = interpolated
        return result
    if isinstance(value, list):
        list_result: list[Any] = []
        for item in value:
            interpolated = _interpolate_env(item, env)
            if interpolated is _MISSING_ENV_PLACEHOLDER:
                continue
            list_result.append(interpolated)
        return list_result
    if isinstance(value, str):
        whole_placeholder = _ENV_PLACEHOLDER_ONLY_RE.fullmatch(value)
        if whole_placeholder and not env.get(whole_placeholder.group(1)):
            return _MISSING_ENV_PLACEHOLDER
        return _ENV_PLACEHOLDER_RE.sub(lambda match: _env_placeholder_value(match, env), value)
    return value


def _env_placeholder_value(match: re.Match[str], env: dict[str, str]) -> str:
    env_name = match.group(1)
    value = env.get(env_name)
    if value:
        return value
    raise ValueError(f"Environment variable {env_name} is required for configuration interpolation")


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


# Environment names outside the ``KG_<GROUP>_<FIELD>`` convention, which charts,
# service specs, and operator shells set.
# When a name here and the conventional name both reach a field, the
# conventional name wins.
_ENV_ALIASES: dict[str, tuple[str, str]] = {
    "KG_RUNTIME": ("runtime", "runtime"),
    "KG_JOB_ID": ("job", "job_id"),
    "KG_GRAPH_ID": ("job", "graph_id"),
    "KG_BATCH_FILES": ("job", "file_batch_size"),
    "KG_FILE_SOURCE": ("files", "source"),
    "KG_INPUT_PATH": ("files", "input_path"),
    "KG_STAGE_PREFIX": ("files", "stage_prefix"),
    "KG_STAGE_CONTENT_HASH": ("files", "stage_content_hash"),
    "KG_MINERU_COMMAND": ("ocr", "mineru_command"),
    "KG_MINERU_METHOD": ("ocr", "mineru_method"),
    "KG_MINERU_BACKEND": ("ocr", "mineru_backend"),
    "KG_MINERU_EFFORT": ("ocr", "mineru_effort"),
    "KG_MINERU_API_URL": ("ocr", "mineru_api_url"),
    "KG_MINERU_API_KEY": ("ocr", "mineru_api_key"),
    "KG_TESSERACT_COMMAND": ("ocr", "tesseract_command"),
    "KG_EMBED_PROVIDER": ("embedding", "provider"),
    "KG_EMBED_ENDPOINT": ("embedding", "endpoint"),
    "KG_EMBED_MODEL": ("embedding", "model"),
    "KG_EMBED_API_KEY": ("embedding", "api_key"),
    "KG_EMBED_API_VERSION": ("embedding", "api_version"),
    "KG_EMBED_DIM": ("embedding", "dimension"),
    "KG_EMBED_BATCH_SIZE": ("embedding", "batch_size"),
    "KG_EMBED_DEVICE": ("embedding", "device"),
    # Shadows the conventional name of ``ontology.profile``, which is an inline
    # document and has no sensible single-variable spelling anyway.
    "KG_ONTOLOGY_PROFILE": ("ontology", "profile_path"),
    "KG_OUTPUT_PATH": ("writer", "output_path"),
    "KG_WRITER": ("writer", "provider"),
    "KG_SNOWFLAKE_BULK_TARGET_FILE_MB": ("snowflake", "bulk_target_file_size_mb"),
}

# Snowpark containers receive the connection through these unprefixed names.
_AMBIENT_SNOWFLAKE_ENV: dict[str, str] = {
    "SNOWFLAKE_ACCOUNT": "account",
    "SNOWFLAKE_HOST": "host",
    "SNOWFLAKE_USER": "user",
    "SNOWFLAKE_PASSWORD": "password",
    "SNOWFLAKE_AUTH": "authenticator",
    "SNOWFLAKE_AUTHENTICATOR": "authenticator",
    "SNOWFLAKE_PRIVATE_KEY_PATH": "private_key_path",
    "SNOWFLAKE_OAUTH_TOKEN": "oauth_token",
    "SNOWFLAKE_OAUTH_TOKEN_PATH": "oauth_token_path",
    "SNOWFLAKE_DATABASE": "database",
    "SNOWFLAKE_SCHEMA": "schema",
    "SNOWFLAKE_ROLE": "role",
    "SNOWFLAKE_WAREHOUSE": "warehouse",
}


def _environment_names() -> dict[str, tuple[str, str]]:
    """Map every supported environment variable to the settings key it fills.

    Every field answers to ``KG_<GROUP>_<FIELD>`` (spelled with the field's
    configuration alias, so ``schema`` reads the same from YAML and from the
    environment), and a conventional name overrides an alias for the same field.
    """

    names = dict(_ENV_ALIASES)
    for group, group_field in Settings.model_fields.items():
        group_model = group_field.annotation
        if isinstance(group_model, type) and issubclass(group_model, _SettingsModel):
            for field, field_info in group_model.model_fields.items():
                key = field_info.alias or field
                names.setdefault(f"KG_{group}_{key}".upper(), (group, key))
    return names


def _from_env(env: dict[str, str]) -> dict[str, Any]:
    """Translate supported environment variables into the nested settings structure.

    Values stay raw strings so the models coerce and validate environment input
    exactly as they do YAML, and an error names the field rather than a variable.
    An empty value means unset, which lets a chart template leave a slot blank.
    """

    data = _ai_backend_profile_from_env(env)
    for env_name, (group, key) in _environment_names().items():
        value = env.get(env_name, "")
        if value != "":
            data.setdefault(group, {})[key] = value
    return data


def _ambient_snowflake_from_env(env: dict[str, str]) -> dict[str, Any]:
    """Load Snowpark-provided Snowflake env vars as low-priority defaults."""

    snowflake = {key: env[name] for name, key in _AMBIENT_SNOWFLAKE_ENV.items() if env.get(name)}
    return {"snowflake": snowflake} if snowflake else {}


def _ai_backend_profile_from_env(env: dict[str, str]) -> dict[str, Any]:
    raw_backend = env.get("KG_AI_BACKEND")
    if not raw_backend:
        return {}
    backend = raw_backend.strip().lower()
    if backend == "oss":
        return {
            "ocr": {"provider": "mineru_internal"},
            "llm": {"provider": "openai_compatible"},
            "embedding": {"provider": "sentence_transformers"},
        }
    if backend == "cortex":
        return {
            "ocr": {"provider": "snowflake_cortex"},
            "llm": {"provider": "snowflake_cortex"},
            "embedding": {"provider": "snowflake_cortex"},
        }
    raise ValueError("KG_AI_BACKEND must be one of: oss, cortex")


def _str_to_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_provider_name(value: object, kind: ProviderKind) -> object:
    if not isinstance(value, str):
        return value
    supported = provider_names(kind)
    if value not in supported:
        # This is deliberately a configuration-load error, not just a preflight
        # error. Unknown provider names should never survive long enough for a
        # factory or runtime path to guess at intent.
        raise ValueError(
            f"Unsupported {kind} provider '{value}'. Supported providers: "
            f"{', '.join(sorted(supported))}"
        )
    return value
