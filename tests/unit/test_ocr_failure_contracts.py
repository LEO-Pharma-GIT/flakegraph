# SPDX-License-Identifier: Apache-2.0
"""Resource bounds and pipeline containment for real OCR adapter failures."""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from http_fakes import StreamingClient, input_file
from pypdf import PdfWriter

from kg_processor.adapters.ocr.generic_http import GenericHttpOcrProvider
from kg_processor.adapters.ocr.mineru_api import MineruApiOcrProvider
from kg_processor.application.pipeline import KgProcessorPipeline
from kg_processor.application.window_errors import is_systemic_provider_error
from kg_processor.config.settings import GenericHttpOcrSettings, Settings
from kg_processor.factories import (
    build_embedding_provider,
    build_file_source,
    build_llm_provider,
    build_ocr_provider,
    build_writer,
)
from kg_processor.ports.ocr import NoTextReadError, OcrOptions


class _ErrorBody(httpx.SyncByteStream):
    """Count a finite streamed error body without prebuffering it."""

    def __init__(self) -> None:
        self.chunks_read = 0
        self.closed = False

    def __iter__(self) -> Any:
        for _ in range(40):
            self.chunks_read += 1
            yield b"synthetic upstream reason " * 4

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("provider_name", ["generic_http", "mineru_api"])
@pytest.mark.parametrize("status", [401, 403, 415, 500])
@pytest.mark.parametrize("declared_oversized", [False, True])
def test_http_ocr_status_keeps_a_bounded_reason_and_typed_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    status: int,
    declared_oversized: bool,
) -> None:
    """An error closes the stream after an excerpt and preserves status classification."""

    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF")
    body = _ErrorBody()
    responses: list[httpx.Response] = []
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            status,
            stream=body,
            request=request,
            headers={"content-length": str(101 * 1024 * 1024)} if declared_oversized else {},
        )
        responses.append(response)
        return response

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda: real_client(transport=httpx.MockTransport(handler), trust_env=False),
    )
    provider = (
        GenericHttpOcrProvider(
            GenericHttpOcrSettings(endpoint="https://ocr.invalid/parse", max_response_bytes=512)
        )
        if provider_name == "generic_http"
        else MineruApiOcrProvider("https://ocr.invalid")
    )
    try:
        with pytest.raises(
            httpx.HTTPStatusError, match=f"HTTP {status}: synthetic upstream reason"
        ) as caught:
            provider.parse(input_file(source), OcrOptions())
        assert caught.value.response is responses[0]
        assert is_systemic_provider_error(caught.value) is (status in {401, 403})
        assert len(str(caught.value).split(": ", 1)[1].encode()) <= 300
        assert body.chunks_read == 3
        assert body.closed
        with pytest.raises(httpx.ResponseNotRead):
            _ = responses[0].content
    finally:
        provider._client.close()  # noqa: SLF001 - close the mocked transport on either adapter


def _pipeline(path: Path, ocr: dict[str, Any]) -> KgProcessorPipeline:
    """Build a local pipeline with fake models and a factory-selected OCR provider."""

    settings = Settings.load(
        env={},
        overrides={
            "files": {"input_path": path},
            "ocr": ocr,
            "generic_http_ocr": {"endpoint": "https://ocr.invalid/parse"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "writer": {"provider": "local_artifacts", "output_path": path.parent / "out"},
            "cache": {"provider": "none"},
        },
    )
    return KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
    )


@pytest.mark.parametrize("status", [401, 403, 404, 500])
@pytest.mark.parametrize("provider_name", ["generic_http", "mineru_api", "fallback"])
def test_preparation_treats_ocr_authentication_status_as_systemic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider_name: str, status: int
) -> None:
    """Neutral HTTP auth failures fail preparation even when document failures are recorded."""

    source = tmp_path / "scan.pdf"
    pdf = PdfWriter()
    pdf.add_blank_page(width=100, height=100)
    pdf.write(source)
    client = StreamingClient(content=b"", status_code=status)
    monkeypatch.setattr(httpx, "Client", client.open)
    pipeline = _pipeline(
        source,
        {
            "provider": provider_name,
            "mineru_api_url": "https://ocr.invalid",
            "fallback_primary_provider": "builtin_text",
            "fallback_secondary_provider": "generic_http",
        },
    )
    try:
        files = pipeline.file_source.list_files()
        if status in {401, 403}:
            with pytest.raises(httpx.HTTPStatusError) as caught:
                pipeline.prepare_documents(files)
            assert caught.value.response.status_code == status
        else:
            prepared = pipeline.prepare_documents(files)
            assert prepared.documents_processed == 0
            assert sum(event.get("status") == "failed" for event in prepared.trace) == 1
    finally:
        pipeline.close()


@pytest.mark.parametrize(
    "provider_name,output_format",
    [
        ("tesseract_internal", "text"),
        ("mineru_internal", "markdown"),
        ("mineru_internal", "json"),
    ],
)
@pytest.mark.parametrize("fallback", [False, True])
def test_successful_empty_cli_ocr_preserves_the_image_in_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    output_format: str,
    fallback: bool,
) -> None:
    """Only successful recognized-empty output becomes an image document and asset."""

    source = tmp_path / "photo.png"
    source.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6"
            "FUYAAAAASUVORK5CYII="
        )
    )
    exit_code = 0

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["check"] is False
        if provider_name == "mineru_internal" and exit_code == 0:
            output = Path(command[command.index("--output") + 1])
            (output / ("result.json" if output_format == "json" else "result.md")).write_text(
                '{"pages": []}' if output_format == "json" else " \n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(
            command, exit_code, stdout=" \n", stderr="synthetic failure"
        )

    monkeypatch.setattr("shutil.which", lambda command: f"/synthetic/{command}")
    monkeypatch.setattr("subprocess.run", run)
    config = (
        {"provider": provider_name}
        if not fallback
        else {
            "provider": "fallback",
            "fallback_primary_provider": "builtin_text",
            "fallback_secondary_provider": provider_name,
        }
    )
    pipeline = _pipeline(source, config)
    try:
        [file] = pipeline.file_source.list_files()
        with pytest.raises(NoTextReadError):
            pipeline.ocr.parse(file, OcrOptions())
        prepared = pipeline.prepare_documents([file])
        assert prepared.documents_processed == 1
        assert [row["kind"] for row in prepared.document_rows] == ["image"]
        assert [row["kind"] for row in prepared.asset_rows] == ["image"]
        assert not any(event.get("status") == "failed" for event in prepared.trace)

        exit_code = 2
        with pytest.raises(RuntimeError) as caught:
            pipeline._parse_file_or_image(file, OcrOptions())  # noqa: SLF001
        assert not isinstance(caught.value, NoTextReadError)
    finally:
        pipeline.close()
