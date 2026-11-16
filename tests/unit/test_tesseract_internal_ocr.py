from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.adapters.ocr.tesseract_internal import TesseractInternalOcrProvider
from kg_processor.ports.ocr import NoTextReadError, OcrOptions


def test_tesseract_internal_builds_pdf_render_command(tmp_path: Path) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]

    command = TesseractInternalOcrProvider().build_pdf_render_command(
        "/usr/bin/pdftoppm",
        file,
        tmp_path / "page",
        OcrOptions(page_range="2-4", tesseract_dpi=200),
    )

    assert command == [
        "/usr/bin/pdftoppm",
        "-png",
        "-r",
        "200",
        "-f",
        "2",
        "-l",
        "4",
        str(sample),
        str(tmp_path / "page"),
    ]


def test_tesseract_internal_image_parse_normalizes_plain_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = tmp_path / "sample.png"
    sample.write_bytes(b"not-a-real-image")
    file = LocalFileSource(sample).list_files()[0]
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "kg_processor.adapters.ocr.tesseract_internal.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="Alice Smith\n", stderr="")

    monkeypatch.setattr("kg_processor.adapters.ocr.tesseract_internal.subprocess.run", fake_run)

    parsed = TesseractInternalOcrProvider().parse(file, OcrOptions(language="eng"))

    assert calls == [["/usr/bin/tesseract", str(sample), "stdout", "-l", "eng"]]
    assert parsed.pages[0].page_number == 1
    assert parsed.pages[0].markdown == "Alice Smith"
    assert parsed.pages[0].blocks[0].kind == "ocr_text"
    assert parsed.provider_metadata["provider"] == "tesseract_internal"
    assert parsed.provider_metadata["layout_support"] == "plain_text"


def test_tesseract_internal_pdf_parse_renders_then_ocr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "kg_processor.adapters.ocr.tesseract_internal.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[0].endswith("pdftoppm"):
            Path(f"{command[-1]}-1.png").write_bytes(b"rendered")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="Rendered text", stderr="")

    monkeypatch.setattr("kg_processor.adapters.ocr.tesseract_internal.subprocess.run", fake_run)

    parsed = TesseractInternalOcrProvider().parse(
        file,
        OcrOptions(language="eng", page_range="1", tesseract_dpi=150),
    )

    assert calls[0][:6] == ["/usr/bin/pdftoppm", "-png", "-r", "150", "-f", "1"]
    assert calls[1][0] == "/usr/bin/tesseract"
    assert parsed.pages[0].page_number == 1
    assert parsed.pages[0].raw_text == "Rendered text"


def test_tesseract_internal_rejects_a_recognition_that_produced_no_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tesseract exits successfully when it recognizes nothing.

    Accepting that records the file as OCR'd while it contributes nothing to the
    graph, which is indistinguishable from a document that genuinely says
    nothing until the graph is inspected.
    """

    sample = tmp_path / "sample.png"
    sample.write_bytes(b"not-a-real-image")
    file = LocalFileSource(sample).list_files()[0]

    monkeypatch.setattr(
        "kg_processor.adapters.ocr.tesseract_internal.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="  \n\f\n", stderr="")

    monkeypatch.setattr("kg_processor.adapters.ocr.tesseract_internal.subprocess.run", fake_run)

    with pytest.raises(NoTextReadError, match="recognized no text"):
        TesseractInternalOcrProvider().parse(file, OcrOptions())


def test_tesseract_internal_raises_for_missing_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = tmp_path / "sample.png"
    sample.write_bytes(b"image")
    file = LocalFileSource(sample).list_files()[0]
    monkeypatch.setattr(
        "kg_processor.adapters.ocr.tesseract_internal.shutil.which",
        lambda _command: None,
    )

    with pytest.raises(RuntimeError, match="Tesseract command"):
        TesseractInternalOcrProvider().parse(file, OcrOptions())


def test_tesseract_internal_rejects_image_page_ranges_after_page_one(tmp_path: Path) -> None:
    sample = tmp_path / "sample.png"
    sample.write_bytes(b"image")
    file = LocalFileSource(sample).list_files()[0]

    with pytest.raises(ValueError, match="only supports page 1"):
        TesseractInternalOcrProvider()._ocr_image(
            file,
            sample,
            1,
            "/usr/bin/tesseract",
            OcrOptions(page_range="2"),
        )
