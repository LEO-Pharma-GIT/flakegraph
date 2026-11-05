# SPDX-License-Identifier: Apache-2.0
"""Tesseract OCR adapter for explicit lightweight image/simple-PDF runs."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from kg_processor.domain.documents import InputFile, LayoutBlock, ParsedDocument, ParsedPage
from kg_processor.domain.ids import stable_id
from kg_processor.ports.ocr import NoTextReadError, OcrOptions, parse_page_range

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_PDF_SUFFIXES = {".pdf"}
_RENDERED_PAGE_RE = re.compile(r"-(\d+)\.[^.]+$")


class TesseractInternalOcrProvider:
    """Runs local Tesseract OCR for images and simple rendered PDFs."""

    def __init__(
        self,
        command: str = "tesseract",
        pdf_renderer_command: str = "pdftoppm",
        dpi: int = 300,
    ) -> None:
        self.command = command
        self.pdf_renderer_command = pdf_renderer_command
        self.dpi = dpi

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        """Parse images directly or render PDFs before invoking Tesseract."""

        executable = shutil.which(self.command)
        if executable is None:
            raise RuntimeError(
                f"Tesseract command '{self.command}' is not installed or not on PATH. "
                "Install Tesseract in the image or choose another OCR provider explicitly."
            )
        suffix = file.path.suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            pages = [self._ocr_image(file, file.path, 1, executable, options)]
        elif suffix in _PDF_SUFFIXES:
            renderer = shutil.which(self.pdf_renderer_command)
            if renderer is None:
                raise RuntimeError(
                    f"PDF renderer command '{self.pdf_renderer_command}' is not installed "
                    "or not on PATH. Install poppler-utils for tesseract_internal PDF OCR."
                )
            pages = self._ocr_pdf(file, executable, renderer, options)
        else:
            raise ValueError(f"tesseract_internal OCR does not support {suffix}")

        # Tesseract exits successfully when it recognizes nothing. Returning that
        # as a parsed document records the file as successfully OCR'd while it
        # contributes nothing, so the empty result is a failure the caller can
        # route or see.
        if not any(page.raw_text.strip() for page in pages):
            raise NoTextReadError(f"Tesseract recognized no text in {file.source_uri}")

        return ParsedDocument(
            file_id=file.id,
            checksum=file.checksum,
            source_uri=file.source_uri,
            mime_type=file.mime_type,
            pages=pages,
            provider_metadata={
                "provider": "tesseract_internal",
                "command": self.command,
                "pdf_renderer_command": self.pdf_renderer_command,
                "dpi": self.dpi,
                "layout_support": "plain_text",
            },
        )

    def build_tesseract_command(
        self,
        executable: str,
        image_path: Path,
        options: OcrOptions,
    ) -> list[str]:
        """Build the Tesseract command for one rendered image."""

        command = [executable, str(image_path), "stdout"]
        if options.language:
            command.extend(["-l", options.language])
        return command

    def build_pdf_render_command(
        self,
        renderer: str,
        file: InputFile,
        output_prefix: Path,
        options: OcrOptions,
    ) -> list[str]:
        """Build the pdftoppm command used before PDF OCR."""

        command = [
            renderer,
            "-png",
            "-r",
            str(options.tesseract_dpi),
        ]
        start_page, end_page = _parse_page_range(options.page_range)
        if start_page is not None:
            command.extend(["-f", str(start_page)])
        if end_page is not None:
            command.extend(["-l", str(end_page)])
        command.extend([str(file.path), str(output_prefix)])
        return command

    def _ocr_pdf(
        self,
        file: InputFile,
        executable: str,
        renderer: str,
        options: OcrOptions,
    ) -> list[ParsedPage]:
        with tempfile.TemporaryDirectory(prefix="kg-tesseract-") as tmp:
            output_prefix = Path(tmp) / "page"
            render_command = self.build_pdf_render_command(renderer, file, output_prefix, options)
            completed = subprocess.run(
                render_command,
                check=False,
                capture_output=True,
                text=True,
                timeout=options.timeout_seconds,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "pdftoppm failed with exit code "
                    f"{completed.returncode}: {completed.stderr.strip()}"
                )
            rendered_pages = _rendered_pages(output_prefix.parent)
            if not rendered_pages:
                raise RuntimeError("pdftoppm did not render any pages for Tesseract OCR")
            return [
                self._ocr_image(file, image_path, page_number, executable, options)
                for page_number, image_path in rendered_pages
            ]

    def _ocr_image(
        self,
        file: InputFile,
        image_path: Path,
        page_number: int,
        executable: str,
        options: OcrOptions,
    ) -> ParsedPage:
        if file.path.suffix.lower() in _IMAGE_SUFFIXES:
            _validate_image_page_range(options.page_range)
        command = self.build_tesseract_command(executable, image_path, options)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=options.timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Tesseract failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )
        text = completed.stdout.strip()
        return ParsedPage(
            page_number=page_number,
            markdown=text,
            raw_text=text,
            blocks=[
                LayoutBlock(
                    id=stable_id("tesseract_block", file.id, page_number, text[:256]),
                    page_number=page_number,
                    kind="ocr_text",
                    text=text,
                    metadata={"layout_support": "plain_text"},
                )
            ],
            detected_language=options.language,
        )


def _parse_page_range(page_range: str | None) -> tuple[int | None, int | None]:
    windows = parse_page_range(page_range, "tesseract_internal")
    return windows[0] if windows else (None, None)


def _validate_image_page_range(page_range: str | None) -> None:
    start, end = _parse_page_range(page_range)
    if start is None and end is None:
        return
    includes_page_one = (start is None or start <= 1) and (end is None or end >= 1)
    if not includes_page_one:
        raise ValueError("tesseract_internal image OCR only supports page 1")


def _rendered_pages(directory: Path) -> list[tuple[int, Path]]:
    pages: list[tuple[int, Path]] = []
    for path in directory.glob("page-*.png"):
        match = _RENDERED_PAGE_RE.search(path.name)
        if match is None:
            continue
        pages.append((int(match.group(1)), path))
    return sorted(pages)
