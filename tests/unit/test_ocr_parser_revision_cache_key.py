"""A parser revision invalidates the OCR cache entries that parser wrote."""

from __future__ import annotations

from pathlib import Path

import pytest

from kg_processor.application import cache_keys
from kg_processor.application.cache_keys import build_ocr_cache_key
from kg_processor.domain.documents import InputFile
from kg_processor.ports.ocr import OcrOptions


def _file() -> InputFile:
    return InputFile(
        id="file_1",
        path=Path("invented.pptx"),
        source_uri="file:///invented.pptx",
        checksum="abc123",
        mime_type="application/octet-stream",
        size_bytes=42,
    )


def _fallback_settings(primary: str, secondary: str) -> dict[str, object]:
    return {"primary_provider": primary, "secondary_provider": secondary}


@pytest.mark.parametrize("provider", ["builtin_text", "mineru_internal", "mineru_api"])
def test_a_parser_revision_changes_that_providers_key(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = build_ocr_cache_key(_file(), provider, OcrOptions())
    monkeypatch.setitem(
        cache_keys.OCR_PARSER_REVISIONS,
        provider,
        cache_keys.OCR_PARSER_REVISIONS[provider] + 1,
    )

    after = build_ocr_cache_key(_file(), provider, OcrOptions())

    assert before.id != after.id


@pytest.mark.parametrize(
    ("primary", "secondary"),
    [("builtin_text", "tesseract_internal"), ("snowflake_cortex", "mineru_api")],
)
def test_a_fallback_key_follows_the_revision_of_either_composed_parser(
    primary: str, secondary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _fallback_settings(primary, secondary)
    before = build_ocr_cache_key(_file(), "fallback", OcrOptions(), settings)
    revised = primary if primary in cache_keys.OCR_PARSER_REVISIONS else secondary
    monkeypatch.setitem(
        cache_keys.OCR_PARSER_REVISIONS,
        revised,
        cache_keys.OCR_PARSER_REVISIONS[revised] + 1,
    )

    after = build_ocr_cache_key(_file(), "fallback", OcrOptions(), settings)

    assert before.id != after.id


def test_providers_without_a_revision_keep_their_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    before = build_ocr_cache_key(_file(), "tesseract_internal", OcrOptions())
    unrelated = build_ocr_cache_key(
        _file(), "fallback", OcrOptions(), _fallback_settings("tesseract_internal", "generic_http")
    )
    monkeypatch.setitem(cache_keys.OCR_PARSER_REVISIONS, "builtin_text", 99)

    assert build_ocr_cache_key(_file(), "tesseract_internal", OcrOptions()).id == before.id
    assert (
        build_ocr_cache_key(
            _file(),
            "fallback",
            OcrOptions(),
            _fallback_settings("tesseract_internal", "generic_http"),
        ).id
        == unrelated.id
    )


def test_the_shipped_builtin_revision_is_past_the_original_parser() -> None:
    """Keys written before structured Office parsing must no longer match."""

    assert cache_keys.OCR_PARSER_REVISIONS["builtin_text"] >= 2
    assert cache_keys.OCR_PARSER_REVISIONS["mineru_internal"] >= 2
    assert cache_keys.OCR_PARSER_REVISIONS["mineru_api"] >= 2


def test_the_builtin_key_differs_from_the_key_before_revisions_existed() -> None:
    options = OcrOptions()
    unrevised = cache_keys._hash_payload(
        {
            "ocr_options": cache_keys._provider_ocr_options("builtin_text", options),
            "provider_settings": {},
        }
    )

    assert build_ocr_cache_key(_file(), "builtin_text", options).options_hash != unrevised
