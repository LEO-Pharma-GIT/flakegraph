from __future__ import annotations

import json

from snowflake_fakes import CONFIG, FakeConnection

from kg_processor.adapters.cache.snowflake import (
    SnowflakeCache,
    build_extraction_cache_merge_statement,
    build_ocr_cache_merge_statement,
)
from kg_processor.domain.documents import ParsedDocument, ParsedPage
from kg_processor.domain.graph import ExtractedEntity, ExtractionResult
from kg_processor.ports.cache import ExtractionCacheKey, OcrCacheKey


def test_snowflake_cache_reads_and_writes_ocr_document() -> None:
    document = ParsedDocument(
        file_id="file_1",
        checksum="checksum",
        source_uri="file:///sample.txt",
        mime_type="text/plain",
        pages=[ParsedPage(page_number=1, markdown="Alice", raw_text="Alice")],
    )
    connection = FakeConnection([(json.dumps(document.model_dump(mode="json")),)])
    cache = SnowflakeCache(CONFIG, connector_factory=lambda **_: connection)
    key = OcrCacheKey(
        id="ocr_1",
        file_id="file_1",
        checksum="checksum",
        ocr_provider="builtin_text",
        options_hash="options",
    )

    assert cache.get_ocr_document(key) == document
    cache.put_ocr_document(key, document)

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert executed_sql[0] == "SELECT PARSED_DOCUMENT FROM KG_OCR_CACHE WHERE ID = ?"
    assert executed_sql[1] == build_ocr_cache_merge_statement()
    assert connection.committed
    assert not connection.closed
    cache.close()
    assert connection.closed


def test_snowflake_cache_reads_and_writes_extraction_result() -> None:
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
    connection = FakeConnection([(extraction.model_dump(mode="json"),)])
    cache = SnowflakeCache(CONFIG, connector_factory=lambda **_: connection)
    key = ExtractionCacheKey(
        id="extraction_1",
        graph_id="graph",
        chunk_batch_hash="chunks",
        llm_provider="fake",
        model="fake",
        options_hash="options",
    )

    assert cache.get_extraction_result(key) == extraction
    cache.put_extraction_result(key, extraction)

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert executed_sql[0] == "SELECT RESULT FROM KG_EXTRACTION_CACHE WHERE ID = ?"
    assert executed_sql[1] == build_extraction_cache_merge_statement()
    assert connection.committed
