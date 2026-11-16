from __future__ import annotations

from pathlib import Path

from kg_processor.adapters.cache.local_json import LocalJsonCache
from kg_processor.domain.documents import LayoutBlock, ParsedDocument, ParsedPage
from kg_processor.domain.graph import ExtractedEntity, ExtractionResult
from kg_processor.ports.cache import EnrichmentCacheKey, ExtractionCacheKey, OcrCacheKey

_OCR_KEY = OcrCacheKey(
    id="ocr_1",
    file_id="file_1",
    checksum="checksum",
    ocr_provider="builtin_text",
    options_hash="options",
)
_EXTRACTION_KEY = ExtractionCacheKey(
    id="extraction_1",
    graph_id="graph",
    chunk_batch_hash="chunks",
    llm_provider="fake",
    model="fake",
    options_hash="options",
)


def test_local_json_cache_roundtrips_ocr_and_extraction(tmp_path: Path) -> None:
    cache = LocalJsonCache(tmp_path)
    enrichment_key = EnrichmentCacheKey(
        id="enrichment_1",
        graph_id="graph",
        stage="description_merge",
        llm_provider="fake",
        model="fake",
        options_hash="options",
    )
    document = ParsedDocument(
        file_id="file_1",
        checksum="checksum",
        source_uri="file:///sample.txt",
        mime_type="text/plain",
        pages=[
            ParsedPage(
                page_number=1,
                markdown="Alice",
                raw_text="Alice",
                blocks=[LayoutBlock(id="block_1", page_number=1, kind="text", text="Alice")],
            )
        ],
    )
    extraction = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Alice",
                type="PERSON",
                description="Alice is mentioned.",
                source_chunk_id="chunk_1",
            )
        ],
        relations=[],
    )

    assert cache.get_ocr_document(_OCR_KEY) is None
    assert cache.get_extraction_result(_EXTRACTION_KEY) is None
    assert cache.get_enrichment_result(enrichment_key) is None

    cache.put_ocr_document(_OCR_KEY, document)
    cache.put_extraction_result(_EXTRACTION_KEY, extraction)
    cache.put_enrichment_result(enrichment_key, {"description": "Alice"})

    assert cache.get_ocr_document(_OCR_KEY) == document
    assert cache.get_extraction_result(_EXTRACTION_KEY) == extraction
    assert cache.get_enrichment_result(enrichment_key) == {"description": "Alice"}
    assert not list(tmp_path.rglob("*.tmp"))


def test_local_json_cache_treats_corrupt_entries_as_misses(tmp_path: Path) -> None:
    cache = LocalJsonCache(tmp_path)
    (tmp_path / "ocr").mkdir()
    (tmp_path / "extraction").mkdir()
    (tmp_path / "ocr" / "ocr_1.json").write_text("{not-json", encoding="utf-8")
    (tmp_path / "extraction" / "extraction_1.json").write_text(
        '{"entities": "not-a-list"}',
        encoding="utf-8",
    )

    assert cache.get_ocr_document(_OCR_KEY) is None
    assert cache.get_extraction_result(_EXTRACTION_KEY) is None
