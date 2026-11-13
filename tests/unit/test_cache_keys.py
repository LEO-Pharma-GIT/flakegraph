from __future__ import annotations

from pathlib import Path

import pytest
from documents import chunk

from kg_processor.application.cache_keys import build_extraction_cache_key, build_ocr_cache_key
from kg_processor.config.settings import ExtractorSettings, GraphSettings, LlmSettings
from kg_processor.domain.documents import InputFile
from kg_processor.ports.ocr import OcrOptions

_CHUNKS = [chunk("Alice Smith works at Acme Corp.")]


def test_ocr_cache_key_excludes_secret_api_key() -> None:
    file = _input_file()

    first = build_ocr_cache_key(
        file,
        "mineru_internal",
        OcrOptions(method="ocr", api_url="https://mineru.example", api_key="first-secret"),
    )
    second = build_ocr_cache_key(
        file,
        "mineru_internal",
        OcrOptions(method="ocr", api_url="https://mineru.example", api_key="rotated-secret"),
    )

    assert first.options_hash == second.options_hash
    assert first.id == second.id


def test_ocr_cache_key_changes_for_behavioral_options() -> None:
    file = _input_file()

    text_key = build_ocr_cache_key(file, "mineru_internal", OcrOptions(method="txt"))
    ocr_key = build_ocr_cache_key(file, "mineru_internal", OcrOptions(method="ocr"))

    assert text_key.options_hash != ocr_key.options_hash
    assert text_key.id != ocr_key.id


def test_fallback_ocr_cache_key_changes_with_routing_policy() -> None:
    """Invalidate normalized text when fallback quality selection changes."""

    file = _input_file()
    strict = build_ocr_cache_key(
        file,
        "fallback",
        OcrOptions(method="ocr"),
        {"max_fragmented_text_ratio": 0.15},
    )
    permissive = build_ocr_cache_key(
        file,
        "fallback",
        OcrOptions(method="ocr"),
        {"max_fragmented_text_ratio": 0.30},
    )

    assert strict.id != permissive.id


def test_ocr_cache_key_ignores_unrelated_provider_options() -> None:
    file = _input_file()

    first = build_ocr_cache_key(file, "mineru_internal", OcrOptions(method="ocr"))
    second = build_ocr_cache_key(
        file,
        "mineru_internal",
        OcrOptions(
            method="ocr",
            model_cache_dir="/tmp/other-cache",
            tesseract_dpi=450,
            snowflake_parse_mode="LAYOUT",
        ),
    )

    assert first.options_hash == second.options_hash
    assert first.id == second.id


def test_ocr_cache_key_keeps_snowflake_cortex_specific_options() -> None:
    file = _input_file()

    ocr = build_ocr_cache_key(file, "snowflake_cortex", OcrOptions(snowflake_parse_mode="OCR"))
    layout = build_ocr_cache_key(
        file,
        "snowflake_cortex",
        OcrOptions(snowflake_parse_mode="LAYOUT"),
    )

    assert ocr.options_hash != layout.options_hash
    assert ocr.id != layout.id


def test_ocr_cache_key_changes_for_page_range() -> None:
    file = _input_file()

    full_document = build_ocr_cache_key(file, "builtin_text", OcrOptions())
    first_page = build_ocr_cache_key(file, "builtin_text", OcrOptions(page_range="1"))

    assert full_document.options_hash != first_page.options_hash
    assert full_document.id != first_page.id


def test_generic_http_ocr_cache_key_includes_endpoint_and_mapping_but_not_secret() -> None:
    """Invalidate generic OCR output when its service contract changes."""

    file = _input_file()
    first = build_ocr_cache_key(
        file,
        "generic_http",
        OcrOptions(),
        {"endpoint": "https://ocr-a.example", "pages_path": "pages", "api_key": "one"},
    )
    rotated = build_ocr_cache_key(
        file,
        "generic_http",
        OcrOptions(),
        {"endpoint": "https://ocr-a.example", "pages_path": "pages", "api_key": "two"},
    )
    changed = build_ocr_cache_key(
        file,
        "generic_http",
        OcrOptions(),
        {"endpoint": "https://ocr-b.example", "pages_path": "result.pages", "api_key": "two"},
    )

    assert first.id == rotated.id
    assert first.id != changed.id


def test_extraction_cache_key_changes_for_prompt_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = GraphSettings()
    first = build_extraction_cache_key(
        "graph",
        _CHUNKS,
        "openai_compatible",
        "model",
        settings,
        120,
    )

    monkeypatch.setattr(
        "kg_processor.application.cache_keys.two_pass_prompt_fingerprints",
        lambda: {
            "entity_extraction": {
                "revision": "entity_extraction@changed",
                "checksum": "0" * 64,
            }
        },
    )
    second = build_extraction_cache_key(
        "graph",
        _CHUNKS,
        "openai_compatible",
        "model",
        settings,
        120,
    )

    assert first.options_hash != second.options_hash
    assert first.id != second.id


def test_extraction_cache_key_follows_the_contract_the_provider_was_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not serve output produced for a schema the model was never shown.

    The request's schema decides what a provider may return, so a change to it
    changes the extraction. Tracked by a label somebody has to remember to bump,
    a contract change silently reuses the previous model's answers.
    """

    settings = GraphSettings()
    key = build_extraction_cache_key("graph", _CHUNKS, "openai_compatible", "model", settings, 120)

    monkeypatch.setattr(
        "kg_processor.application.cache_keys.extraction_contract_fingerprint",
        lambda: "0" * 64,
    )
    after = build_extraction_cache_key(
        "graph", _CHUNKS, "openai_compatible", "model", settings, 120
    )

    assert key.options_hash != after.options_hash
    assert key.id != after.id


def test_extraction_cache_key_includes_extractor_provider_and_model() -> None:
    """Do not serve LLM extraction output after switching to GLiNER semantics."""

    common = ("graph", _CHUNKS, "openai_compatible", "model", GraphSettings(), 120)

    llm = build_extraction_cache_key(
        *common,
        extractor_settings=ExtractorSettings(entity_provider="llm"),
    )
    gliner = build_extraction_cache_key(
        *common,
        extractor_settings=ExtractorSettings(
            entity_provider="gliner",
            gliner_model="urchade/gliner_multi-v2.1",
            gliner_threshold=0.65,
        ),
    )

    assert llm.id != gliner.id


def _input_file() -> InputFile:
    return InputFile(
        id="file_1",
        path=Path("data/martial_arts/files/smoke.txt"),
        source_uri="file:///data/martial_arts/files/smoke.txt",
        checksum="abc123",
        mime_type="text/plain",
        size_bytes=42,
    )


@pytest.mark.parametrize(
    ("provider", "field", "value", "changes_key"),
    [
        ("ollama", "context_window_tokens", 65536, True),
        ("openai_compatible", "context_window_tokens", 65536, False),
        ("openai_compatible", "api_version", "another-version", False),
        ("fake", "endpoint", "https://unused.example", False),
        ("snowflake_cortex", "max_output_tokens", 1000, False),
    ],
)
def test_extraction_cache_tracks_only_provider_effective_options(
    provider: str, field: str, value: object, changes_key: bool
) -> None:
    settings = LlmSettings(provider=provider, endpoint="https://service.example")
    changed = settings.model_copy(update={field: value})
    first = build_extraction_cache_key(
        "graph", _CHUNKS, provider, "model", GraphSettings(), 120, llm_settings=settings
    )
    second = build_extraction_cache_key(
        "graph", _CHUNKS, provider, "model", GraphSettings(), 120, llm_settings=changed
    )
    assert (first.id != second.id) is changes_key


def test_extraction_cache_ignores_endpoint_credentials_and_equivalent_ollama_urls() -> None:
    first = LlmSettings(provider="ollama", endpoint="https://user:password@service.example/v1/")
    changed = first.model_copy(
        update={"api_key": "rotated", "endpoint": "https://other:rotated@service.example"}
    )
    assert (
        build_extraction_cache_key(
            "graph", _CHUNKS, "ollama", "model", GraphSettings(), 120, llm_settings=first
        ).id
        == build_extraction_cache_key(
            "graph", _CHUNKS, "ollama", "model", GraphSettings(), 120, llm_settings=changed
        ).id
    )
