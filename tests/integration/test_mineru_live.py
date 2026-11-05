from __future__ import annotations

import os
from pathlib import Path

import pytest

from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.adapters.ocr.mineru_internal import MineruInternalOcrProvider
from kg_processor.ports.ocr import OcrOptions

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("KG_RUN_MINERU_LIVE") != "1",
        reason="Set KG_RUN_MINERU_LIVE=1 to run live MinerU OCR integration checks.",
    ),
]

_DEFAULT_INPUT = Path("data/martial_arts/files/martial-arts-overview.pdf")


def test_live_mineru_internal_parse_normalizes_sample_document() -> None:
    file_path = Path(os.getenv("KG_MINERU_LIVE_INPUT", str(_DEFAULT_INPUT)))
    if not file_path.exists():
        pytest.skip(f"Live MinerU input does not exist: {file_path}")
    file = LocalFileSource(file_path).list_files()[0]
    parsed = MineruInternalOcrProvider(
        command=os.getenv("KG_MINERU_COMMAND", "mineru"),
    ).parse(
        file,
        OcrOptions(
            page_range=os.getenv("KG_MINERU_LIVE_PAGE_RANGE", "0"),
            timeout_seconds=int(os.getenv("KG_MINERU_LIVE_TIMEOUT_SECONDS", "900")),
            method=os.getenv("KG_MINERU_METHOD", "txt"),
            backend=os.getenv("KG_MINERU_BACKEND", "pipeline"),
            effort=os.getenv("KG_MINERU_EFFORT", "medium"),
            model_cache_dir=os.getenv("KG_MINERU_MODEL_CACHE_DIR"),
        ),
    )

    page_text = "\n".join(page.markdown or page.raw_text for page in parsed.pages).strip()
    assert parsed.file_id == file.id
    assert parsed.checksum == file.checksum
    assert parsed.source_uri == file.source_uri
    assert parsed.provider_metadata["provider"] == "mineru_internal"
    assert parsed.provider_metadata["argv"]
    assert parsed.pages
    assert page_text
    assert all(page.blocks for page in parsed.pages)
