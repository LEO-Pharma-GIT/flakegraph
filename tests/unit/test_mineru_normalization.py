from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.adapters.ocr.mineru_common import redact_blob_metadata
from kg_processor.adapters.ocr.mineru_internal import MineruInternalOcrProvider
from kg_processor.ports.ocr import OcrOptions


def test_blob_redaction_covers_every_provider_spelling_of_an_inline_payload() -> None:
    """One redaction, so no adapter leaks a key another adapter already hides."""

    redacted = redact_blob_metadata(
        {
            "base64": "aa",
            "bytes": "bb",
            "content": "cc",
            "content_bytes": "dd",
            "data": "ee",
            "image_base64": "ff",
            "image_data": "gg",
            "caption": "Figure 1",
        }
    )

    assert redacted["caption"] == "Figure 1"
    blob_keys = (
        "base64",
        "bytes",
        "content",
        "content_bytes",
        "data",
        "image_base64",
        "image_data",
    )
    for key in blob_keys:
        assert key not in redacted
        assert redacted[f"{key}_present"] is True
        assert redacted[f"{key}_length"] == 2


def test_content_list_paragraphs_are_not_mistaken_for_binary_assets(tmp_path: Path) -> None:
    """``content`` names a paragraph's text in MinerU and a blob elsewhere."""

    sample = tmp_path / "sample.txt"
    sample.write_text("Alice Smith", encoding="utf-8")
    file = LocalFileSource(sample).list_files()[0]
    output = tmp_path / "mineru-out"
    output.mkdir()
    (output / "content_list.json").write_text(
        json.dumps([{"page_idx": 0, "type": "text", "content": "Alice works at Acme."}]),
        encoding="utf-8",
    )

    parsed = MineruInternalOcrProvider()._load_output(file, output, stdout="ok")

    assert parsed.pages[0].raw_text == "Alice works at Acme."
    assert parsed.assets == []


def test_mineru_json_output_normalizes_pages_blocks_and_assets(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice Smith", encoding="utf-8")
    file = LocalFileSource(sample).list_files()[0]
    output = tmp_path / "mineru-out"
    output.mkdir()
    (output / "result.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_id": "doc-image",
                        "kind": "image",
                        "page_number": 1,
                        "path": "images/doc.png",
                    }
                ],
                "pages": [
                    {
                        "page_number": 3,
                        "markdown": "Alice Smith works at Acme Corp.",
                        "language": "en",
                        "blocks": [
                            {
                                "type": "paragraph",
                                "text": "Alice Smith works at Acme Corp.",
                                "bbox": [1, 2, 3, 4],
                            }
                        ],
                        "figures": [
                            {
                                "type": "figure",
                                "url": "images/page-3.png",
                                "image_base64": "encoded-image",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    parsed = MineruInternalOcrProvider()._load_output(file, output, stdout="ok")

    assert parsed.pages[0].page_number == 3
    assert parsed.pages[0].blocks[0].kind == "paragraph"
    assert parsed.pages[0].blocks[0].bbox == (1, 2, 3, 4)
    assert [asset.id for asset in parsed.assets] == ["doc-image", parsed.assets[1].id]
    assert parsed.assets[0].uri == "images/doc.png"
    assert parsed.assets[1].kind == "figure"
    assert parsed.assets[1].page_number == 3
    assert parsed.assets[1].metadata["image_base64_present"] is True
    assert parsed.assets[1].metadata["image_base64_length"] == len("encoded-image")
    assert "image_base64" not in parsed.assets[1].metadata
    assert parsed.provider_metadata["provider"] == "mineru_internal"


def test_mineru_command_includes_configured_cli_options(tmp_path: Path) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]

    command = MineruInternalOcrProvider().build_command(
        "/usr/local/bin/mineru",
        file,
        tmp_path / "out",
        OcrOptions(
            language="en",
            page_range="1-3",
            method="ocr",
            backend="pipeline",
            effort="high",
            api_url="http://mineru-api:8000",
            server_url="http://mineru-vlm:30000",
            formula=False,
            table=True,
            image_analysis=False,
            client_side_output_generation=True,
        ),
    )

    assert command == [
        "/usr/local/bin/mineru",
        "--path",
        str(file.path),
        "--output",
        str(tmp_path / "out"),
        "--method",
        "ocr",
        "--backend",
        "pipeline",
        "--effort",
        "high",
        "--lang",
        "en",
        "--api-url",
        "http://mineru-api:8000",
        "--url",
        "http://mineru-vlm:30000",
        "--start",
        "1",
        "--end",
        "3",
        "--formula",
        "false",
        "--table",
        "true",
        "--image-analysis",
        "false",
        "--client-side-output-generation",
        "true",
    ]


def test_mineru_json_output_preserves_zero_page_and_accepts_page_idx(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice Smith", encoding="utf-8")
    file = LocalFileSource(sample).list_files()[0]
    output = tmp_path / "mineru-out"
    output.mkdir()
    (output / "result.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": "zero-page-image",
                        "kind": "image",
                        "page_number": 0,
                        "path": "images/zero.png",
                    },
                    {
                        "id": "page-idx-image",
                        "kind": "image",
                        "page_idx": 1,
                        "path": "images/page-idx.png",
                    },
                ],
                "pages": [
                    {"page_number": 0, "text": "Zero page"},
                    {"page": "not-a-number", "page_idx": "1", "text": "Second page"},
                ],
            }
        ),
        encoding="utf-8",
    )

    parsed = MineruInternalOcrProvider()._load_output(file, output, stdout="ok")

    assert [page.page_number for page in parsed.pages] == [0, 2]
    assert [asset.page_number for asset in parsed.assets] == [0, 2]


def test_mineru_command_explicit_page_ids_override_page_range(tmp_path: Path) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]

    command = MineruInternalOcrProvider().build_command(
        "mineru",
        file,
        tmp_path / "out",
        OcrOptions(page_range="1-3", start_page_id=4, end_page_id=5),
    )

    assert command[-4:] == ["--start", "4", "--end", "5"]


def test_mineru_output_prefers_markdown_when_json_is_not_page_shaped(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice Smith", encoding="utf-8")
    file = LocalFileSource(sample).list_files()[0]
    output = tmp_path / "mineru-out"
    output.mkdir()
    (output / "metadata.json").write_text('{"status": "ok"}', encoding="utf-8")
    (output / "result.md").write_text("# Alice Smith\n\nWorks at Acme.", encoding="utf-8")

    parsed = MineruInternalOcrProvider()._load_output(file, output, stdout="ok")

    assert parsed.pages[0].markdown == "# Alice Smith\n\nWorks at Acme."


def test_mineru_output_rejects_an_unrecognized_json_shape(tmp_path: Path) -> None:
    """A MinerU output change must fail rather than become the document text.

    Serializing the payload into a page would chunk, embed, and extract the
    output structure itself while the run reported success.
    """

    sample = tmp_path / "sample.txt"
    sample.write_text("Alice Smith", encoding="utf-8")
    file = LocalFileSource(sample).list_files()[0]
    output = tmp_path / "mineru-out"
    output.mkdir()
    (output / "result.json").write_text(
        '{"document": {"blocks": ["Alice Smith"]}}', encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="unrecognized JSON output") as error:
        MineruInternalOcrProvider()._load_output(file, output, stdout="ok")

    assert "document" in str(error.value)
    assert "Alice Smith" not in str(error.value)


def test_mineru_content_list_preserves_physical_pages_over_flat_markdown(
    tmp_path: Path,
) -> None:
    """Group MinerU content blocks by page index instead of collapsing Markdown."""

    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]
    output = tmp_path / "mineru-out"
    output.mkdir()
    (output / "sample_content_list.json").write_text(
        json.dumps(
            [
                {"type": "text", "text": "First page", "page_idx": 0},
                {
                    "type": "list",
                    "list_items": ["Second-page item one", "Second-page item two"],
                    "page_idx": 1,
                },
                {"type": "text", "text": "Second-page paragraph", "page_idx": 1},
            ]
        ),
        encoding="utf-8",
    )
    (output / "sample.md").write_text(
        "First page\n\nSecond-page item one\n\nSecond-page paragraph",
        encoding="utf-8",
    )

    parsed = MineruInternalOcrProvider()._load_output(file, output, stdout="ok")

    assert [page.page_number for page in parsed.pages] == [1, 2]
    assert parsed.pages[0].raw_text == "First page"
    assert parsed.pages[1].raw_text == (
        "Second-page item one\nSecond-page item two\n\nSecond-page paragraph"
    )
    assert [block.kind for block in parsed.pages[1].blocks] == ["list", "text"]


def test_mineru_internal_parse_raises_when_command_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]
    monkeypatch.setattr("kg_processor.adapters.ocr.mineru_internal.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="MinerU command 'mineru' is not installed"):
        MineruInternalOcrProvider().parse(file, OcrOptions())


def test_mineru_internal_parse_raises_with_stderr_when_command_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]
    monkeypatch.setattr(
        "kg_processor.adapters.ocr.mineru_internal.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="bad input")

    monkeypatch.setattr("kg_processor.adapters.ocr.mineru_internal.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="MinerU failed with exit code 2: bad input"):
        MineruInternalOcrProvider().parse(file, OcrOptions())


def test_mineru_internal_parse_passes_configured_model_cache_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = tmp_path / "sample.pdf"
    cache_dir = tmp_path / "mineru-cache"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]
    captured_env: dict[str, str] = {}
    monkeypatch.setenv("KG_LLM_API_KEY", "must-not-reach-ocr")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "must-not-reach-ocr")
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-ocr")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "legacy-hf-token")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example")
    monkeypatch.setattr(
        "kg_processor.adapters.ocr.mineru_internal.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        _ = command
        captured_env.update(kwargs["env"])
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="bad input")

    monkeypatch.setattr("kg_processor.adapters.ocr.mineru_internal.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="MinerU failed with exit code 2: bad input"):
        MineruInternalOcrProvider().parse(file, OcrOptions(model_cache_dir=str(cache_dir)))

    assert cache_dir.is_dir()
    assert captured_env["MINERU_MODEL_CACHE_DIR"] == str(cache_dir)
    assert captured_env["HF_HOME"] == str(cache_dir / "huggingface")
    assert captured_env["HF_HUB_DISABLE_XET"] == "1"
    assert captured_env["MODELSCOPE_CACHE"] == str(cache_dir / "modelscope")
    assert captured_env["HTTPS_PROXY"] == "https://proxy.example"
    assert "KG_LLM_API_KEY" not in captured_env
    assert "SNOWFLAKE_PASSWORD" not in captured_env
    assert captured_env["HF_TOKEN"] == "must-not-reach-ocr"
    assert captured_env["HUGGING_FACE_HUB_TOKEN"] == "legacy-hf-token"


def test_mineru_internal_rejects_invalid_page_range(tmp_path: Path) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]

    with pytest.raises(ValueError, match="end page is before start page"):
        MineruInternalOcrProvider().build_command(
            "mineru",
            file,
            tmp_path / "out",
            OcrOptions(page_range="3-1"),
        )
