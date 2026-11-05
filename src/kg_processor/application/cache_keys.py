# SPDX-License-Identifier: Apache-2.0
"""Behavioral cache keys for OCR and graph extraction.

Secrets are excluded by construction. Prompt/schema/provider options are included
because changing any of them can change the normalized graph, even when the input
file bytes are identical.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from kg_processor.application.chunking import compute_ordered_chunk_hash
from kg_processor.application.extraction_contracts import extraction_contract_fingerprint
from kg_processor.application.prompt_registry import two_pass_prompt_fingerprints
from kg_processor.application.redaction import is_sensitive_key, redact_sensitive_data
from kg_processor.config.settings import (
    EmbeddingSettings,
    ExtractorSettings,
    GraphSettings,
    LlmSettings,
)
from kg_processor.domain.documents import InputFile
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ids import sha256_hex, stable_id
from kg_processor.ports.cache import ExtractionCacheKey, OcrCacheKey
from kg_processor.ports.ocr import OcrOptions

_COMMON_OCR_CACHE_FIELDS = {"language", "page_range"}
_PROVIDER_OCR_CACHE_FIELDS = {
    "builtin_text": set(),
    "generic_http": {
        "method",
        "backend",
        "effort",
        "formula",
        "table",
        "image_analysis",
    },
    "mineru_api": {
        "api_url",
        "method",
        "backend",
        "effort",
        "formula",
        "table",
        "image_analysis",
        "client_side_output_generation",
        "start_page_id",
        "end_page_id",
    },
    "mineru_internal": {
        "api_url",
        "server_url",
        "method",
        "backend",
        "effort",
        "formula",
        "table",
        "image_analysis",
        "client_side_output_generation",
        "start_page_id",
        "end_page_id",
    },
    "snowflake_cortex": {
        "snowflake_parse_mode",
        "snowflake_extract_images",
        "snowflake_page_split",
    },
    "tesseract_internal": {
        "tesseract_command",
        "tesseract_pdf_renderer_command",
        "tesseract_dpi",
    },
}


def build_ocr_cache_key(
    file: InputFile,
    provider: str,
    options: OcrOptions,
    provider_settings: dict[str, Any] | None = None,
) -> OcrCacheKey:
    """Build a deterministic OCR cache key from transport and parsing behavior.

    Adapter mapping and endpoint settings matter for generic HTTP OCR, while
    credential values are removed recursively before hashing.
    """

    options_hash = _hash_payload(
        {
            "ocr_options": _provider_ocr_options(provider, options),
            "provider_settings": _without_secrets(provider_settings or {}),
        }
    )
    return OcrCacheKey(
        id=stable_id("ocr_cache", file.id, file.checksum, provider, options_hash),
        file_id=file.id,
        checksum=file.checksum,
        ocr_provider=provider,
        options_hash=options_hash,
    )


def build_extraction_cache_key(
    graph_id: str,
    chunks: list[Chunk],
    provider: str,
    model: str,
    graph_settings: GraphSettings,
    timeout_seconds: int,
    ontology_checksum: str = "",
    extractor_settings: ExtractorSettings | None = None,
    embedding_settings: EmbeddingSettings | None = None,
    llm_settings: LlmSettings | None = None,
) -> ExtractionCacheKey:
    """Build an extraction cache key from content and every semantic input.

    Provider credentials are excluded, while graph settings, prompt revisions,
    ontology content, model identity, and ordered chunks participate so cached
    output is invalidated whenever extraction behavior can change.
    """

    chunk_batch_hash = compute_ordered_chunk_hash(chunks)
    # Extraction cache invalidation follows the whole prompt/schema contract,
    # not just the model name. This prevents stale graph output after prompt
    # tuning or schema changes while keeping credential rotation cache-neutral.
    options_hash = _hash_payload(
        {
            "graph_settings": graph_settings.model_dump(mode="json"),
            "extractor_settings": (extractor_settings or ExtractorSettings()).model_dump(
                mode="json"
            ),
            "embedding_settings": (embedding_settings or EmbeddingSettings()).model_dump(
                mode="json",
                exclude={"api_key"},
            ),
            "llm_settings": _effective_llm_options(provider, llm_settings),
            "extraction_contracts": extraction_contract_fingerprint(),
            "two_pass_prompts": two_pass_prompt_fingerprints(),
            "ontology_checksum": ontology_checksum,
            "timeout_seconds": timeout_seconds,
        }
    )
    return ExtractionCacheKey(
        id=stable_id("extraction_cache", graph_id, chunk_batch_hash, provider, model, options_hash),
        graph_id=graph_id,
        chunk_batch_hash=chunk_batch_hash,
        llm_provider=provider,
        model=model,
        options_hash=options_hash,
    )


def _effective_llm_options(provider: str, settings: LlmSettings | None) -> dict[str, Any]:
    """Select the non-secret options consumed by the selected LLM adapter.

    Unused settings stay cache-neutral, as do credentials and endpoint secrets.
    Endpoint normalization follows the adapters' accepted URL forms.
    """

    settings = settings or LlmSettings()
    options: dict[str, Any] = {}
    if provider in {"openai_compatible", "azure_openai", "ollama", "vllm_local"}:
        endpoint = (settings.endpoint or "").rstrip("/")
        if provider == "ollama" and endpoint.endswith("/v1"):
            endpoint = endpoint[:-3]
        parsed = urlsplit(endpoint)
        endpoint = urlunsplit(parsed._replace(netloc=parsed.netloc.rsplit("@", 1)[-1]))
        options["endpoint"] = redact_sensitive_data(endpoint)
        options["max_output_tokens"] = settings.max_output_tokens
    if provider == "azure_openai":
        options["api_version"] = settings.api_version
    if provider == "ollama":
        options["context_window_tokens"] = settings.context_window_tokens
    return options


def _provider_ocr_options(provider: str, options: OcrOptions) -> dict[str, Any]:
    """Select provider-relevant OCR request fields without hashing credentials."""

    raw = options.model_dump(mode="json")
    fields = _COMMON_OCR_CACHE_FIELDS | _PROVIDER_OCR_CACHE_FIELDS.get(
        provider,
        set(raw) - {"api_key", "model_cache_dir", "timeout_seconds"},
    )
    return {field: raw.get(field) for field in sorted(fields)}


def _without_secrets(value: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove credential-bearing fields from cache semantics."""

    return {
        str(key): (_without_secrets(item) if isinstance(item, dict) else item)
        for key, item in value.items()
        if not is_sensitive_key(str(key))
    }


def _hash_payload(value: dict[str, Any]) -> str:
    return sha256_hex(json.dumps(value, sort_keys=True, separators=(",", ":")))
