# SPDX-License-Identifier: Apache-2.0
"""Internal MinerU OCR adapter for the default local production profile."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kg_processor.adapters.ocr.mineru_common import (
    first_int,
    first_string,
    mineru_assets_from_payloads,
    resolve_page_window,
    shape_description,
)
from kg_processor.domain.documents import (
    InputFile,
    LayoutBlock,
    ParsedAsset,
    ParsedDocument,
    ParsedPage,
)
from kg_processor.domain.ids import stable_id
from kg_processor.ports.ocr import NoTextReadError, OcrOptions

_BBOX_COORDS = 4
_SECRET_ENVIRONMENT_PREFIXES = (
    "AWS_",
    "AZURE_",
    "GOOGLE_",
    "KG_",
    "OPENAI_",
    "SNOWFLAKE_",
)
_SECRET_ENVIRONMENT_MARKERS = ("API_KEY", "CREDENTIAL", "PASSWORD", "SECRET", "TOKEN")


class MineruInternalOcrProvider:
    """Runs the local MinerU CLI and normalizes its generated artifacts."""

    def __init__(self, command: str = "mineru") -> None:
        self.command = command

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        """Execute MinerU locally and read the generated markdown/json outputs."""

        executable = shutil.which(self.command)
        if executable is None:
            raise RuntimeError(
                f"MinerU command '{self.command}' is not installed or not on PATH. "
                "Install MinerU in the image or choose another OCR provider explicitly."
            )

        with tempfile.TemporaryDirectory(prefix="kg-mineru-") as tmp:
            output_dir = Path(tmp)
            command = self.build_command(executable, file, output_dir, options)
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                env=_mineru_environment(options),
                text=True,
                timeout=options.timeout_seconds,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "MinerU failed with exit code "
                    f"{completed.returncode}: {completed.stderr.strip()}"
                )
            return self._load_output(file, output_dir, completed.stdout, command)

    def build_command(
        self,
        executable: str,
        file: InputFile,
        output_dir: Path,
        options: OcrOptions,
    ) -> list[str]:
        """Build the MinerU CLI command from provider-neutral OCR options."""

        command = [
            executable,
            "--path",
            str(file.path),
            "--output",
            str(output_dir),
        ]
        _append_value(command, "--method", options.method)
        _append_value(command, "--backend", options.backend)
        _append_value(command, "--effort", options.effort)
        _append_value(command, "--lang", options.language)
        _append_value(command, "--api-url", options.api_url)
        _append_value(command, "--url", options.server_url)

        start_page_id, end_page_id = resolve_page_window(options, provider="mineru_internal")
        _append_value(command, "--start", start_page_id)
        _append_value(command, "--end", end_page_id)
        _append_bool(command, "--formula", options.formula)
        _append_bool(command, "--table", options.table)
        _append_bool(command, "--image-analysis", options.image_analysis)
        if options.client_side_output_generation:
            command.extend(["--client-side-output-generation", "true"])
        return command

    def _load_output(
        self,
        file: InputFile,
        output_dir: Path,
        stdout: str,
        command: list[str] | None = None,
    ) -> ParsedDocument:
        json_files = sorted(output_dir.rglob("*.json"))
        markdown_files = sorted(output_dir.rglob("*.md"))

        page_json = _first_page_json(json_files)
        if page_json:
            pages, assets = _document_parts_from_mineru_json(file, page_json)
        elif markdown_files:
            text = markdown_files[0].read_text(encoding="utf-8")
            pages = [_page(file, 1, text)]
            assets = []
        elif json_files:
            pages, assets = _document_parts_from_mineru_json(file, json_files[0])
        else:
            raise RuntimeError(f"MinerU did not produce JSON or Markdown output in {output_dir}")
        if not any(page.markdown.strip() or page.raw_text.strip() for page in pages):
            raise NoTextReadError(f"MinerU produced no usable text in {output_dir}")

        return ParsedDocument(
            file_id=file.id,
            checksum=file.checksum,
            source_uri=file.source_uri,
            mime_type=file.mime_type,
            pages=pages,
            assets=assets,
            provider_metadata={
                "provider": "mineru_internal",
                "command": self.command,
                "argv": command or [],
                "stdout_tail": stdout[-2000:],
            },
        )


def _document_parts_from_mineru_json(
    file: InputFile,
    path: Path,
) -> tuple[list[ParsedPage], list[ParsedAsset]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _pages_from_mineru_payload(file, payload), _assets_from_mineru_payload(file, payload)


def _pages_from_mineru_payload(file: InputFile, payload: object) -> list[ParsedPage]:
    """Map a recognized MinerU output shape onto pages, or reject the output.

    An unrecognized payload is a MinerU output change, not document text.
    Serializing it into a page would chunk, embed, and extract the output
    structure itself while the run reported success.
    """

    if isinstance(payload, dict) and isinstance(payload.get("pages"), list):
        raw_pages = payload["pages"]
    elif isinstance(payload, list) and any(
        isinstance(item, dict) and first_int(item, ["page_idx", "page_index"]) is not None
        for item in payload
    ):
        return _pages_from_content_list(file, payload)
    elif isinstance(payload, list):
        raw_pages = payload
    else:
        raise RuntimeError(
            "MinerU produced an unrecognized JSON output for "
            f"{file.source_uri}: expected a page list or an object with pages, "
            f"got {shape_description(payload)}"
        )

    pages: list[ParsedPage] = []
    for index, raw in enumerate(raw_pages, start=1):
        if not isinstance(raw, dict):
            pages.append(_page(file, index, str(raw)))
            continue
        page_number = first_int(raw, ["page_number", "page", "page_id"])
        if page_number is None:
            page_idx = first_int(raw, ["page_idx", "page_index", "index"])
            page_number = page_idx + 1 if page_idx is not None else index
        text = first_string(raw, ["markdown", "md", "text", "raw_text", "content"])
        blocks = _blocks_from_raw(file, page_number, raw.get("blocks"))
        pages.append(
            ParsedPage(
                page_number=page_number,
                markdown=text,
                raw_text=text,
                blocks=blocks or [_block(file, page_number, text)],
                detected_language=first_string(raw, ["language", "detected_language"]) or None,
            )
        )
    return pages


def _pages_from_content_list(file: InputFile, items: list[object]) -> list[ParsedPage]:
    """Group MinerU's legacy content-list blocks by zero-based physical page.

    MinerU emits one flat record per paragraph, title, list, equation, image, or
    table. Every record retains ``page_idx`` even though the generated Markdown
    has no page delimiters. Reconstructing pages from that index preserves the
    provenance required by chunking, evidence, and benchmark evaluation.
    """

    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        page_index = first_int(item, ["page_idx", "page_index"])
        if page_index is None or page_index < 0:
            continue
        grouped.setdefault(page_index, []).append(item)

    pages: list[ParsedPage] = []
    for page_index, raw_blocks in sorted(grouped.items()):
        page_number = page_index + 1
        blocks: list[LayoutBlock] = []
        text_parts: list[str] = []
        for block_index, raw in enumerate(raw_blocks):
            text = _content_list_text(raw)
            if not text:
                continue
            text_parts.append(text)
            bbox = raw.get("bbox")
            parsed_bbox = (
                tuple(float(value) for value in bbox)
                if isinstance(bbox, list) and len(bbox) == _BBOX_COORDS
                else None
            )
            blocks.append(
                LayoutBlock(
                    id=stable_id(
                        "block",
                        file.id,
                        page_number,
                        block_index,
                        str(raw.get("type", "text")),
                        text[:128],
                    ),
                    page_number=page_number,
                    kind=str(raw.get("type") or "text"),
                    text=text,
                    bbox=parsed_bbox,
                    metadata={
                        key: value
                        for key, value in raw.items()
                        if key
                        not in {
                            "text",
                            "content",
                            "list_items",
                            "bbox",
                        }
                    },
                )
            )
        page_text = "\n\n".join(text_parts)
        pages.append(
            ParsedPage(
                page_number=page_number,
                markdown=page_text,
                raw_text=page_text,
                blocks=blocks or [_block(file, page_number, page_text)],
            )
        )
    return pages


def _content_list_text(raw: dict[str, Any]) -> str:
    """Render the textual fields of one MinerU content-list record."""

    parts: list[str] = []
    for key in (
        "text",
        "content",
        "list_items",
        "image_caption",
        "image_footnote",
        "table_caption",
        "table_body",
        "table_footnote",
    ):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, list):
            parts.extend(str(item).strip() for item in value if str(item).strip())
    return "\n".join(parts)


def _assets_from_mineru_payload(file: InputFile, payload: object) -> list[ParsedAsset]:
    if isinstance(payload, dict):
        return mineru_assets_from_payloads(
            file,
            [payload],
            id_prefix="mineru_internal_asset",
        )
    return []


def _first_page_json(json_files: list[Path]) -> Path | None:
    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("pages"), list):
            return path
        if isinstance(payload, list) and any(
            isinstance(item, dict) and first_int(item, ["page_idx", "page_index"]) is not None
            for item in payload
        ):
            return path
    return None


def _blocks_from_raw(file: InputFile, page_number: int, raw_blocks: Any) -> list[LayoutBlock]:
    if not isinstance(raw_blocks, list):
        return []
    blocks: list[LayoutBlock] = []
    for idx, raw in enumerate(raw_blocks):
        if not isinstance(raw, dict):
            continue
        text = first_string(raw, ["text", "content", "markdown"]) or ""
        bbox = raw.get("bbox")
        parsed_bbox = (
            tuple(float(v) for v in bbox)
            if isinstance(bbox, list) and len(bbox) == _BBOX_COORDS
            else None
        )
        blocks.append(
            LayoutBlock(
                id=stable_id("block", file.id, page_number, idx, text[:128]),
                page_number=page_number,
                kind=str(raw.get("type") or raw.get("kind") or "text"),
                text=text,
                bbox=parsed_bbox,
                metadata={k: v for k, v in raw.items() if k not in {"text", "content", "markdown"}},
            )
        )
    return blocks


def _append_value(command: list[str], flag: str, value: str | int | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _append_bool(command: list[str], flag: str, value: bool | None) -> None:
    if value is not None:
        command.extend([flag, str(value).lower()])


def _mineru_environment(options: OcrOptions) -> dict[str, str] | None:
    """Build a least-privilege environment for the isolated MinerU process.

    MinerU is an optional executable boundary rather than an in-process Python
    dependency. The subprocess needs ordinary runtime, proxy, GPU, and cache
    variables, but it never needs FlakeGraph's LLM, database, cloud, or writer
    credentials. Removing those values limits the impact of defects in the OCR
    dependency or any model-loading code it invokes.
    """

    hugging_face_credentials = {
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }
    env = {
        name: value
        for name, value in os.environ.items()
        if name in hugging_face_credentials
        or (
            not name.startswith(_SECRET_ENVIRONMENT_PREFIXES)
            and not any(marker in name.upper() for marker in _SECRET_ENVIRONMENT_MARKERS)
        )
    }
    # Hugging Face Hub may route large files through the optional hf-xet client
    # by default. That path is noticeably brittle on corporate networks and
    # some DNS/security products, so MinerU subprocesses default to the regular
    # HTTP downloader. Keep this as setdefault so a user who explicitly wants
    # Xet can still export HF_HUB_DISABLE_XET=0.
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    if not options.model_cache_dir:
        return env
    cache_dir = Path(options.model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # MinerU itself may delegate downloads to Hugging Face, ModelScope, Torch,
    # or other model libraries. Pointing the common cache variables under the
    # configured root keeps local runs and containers reviewably self-contained.
    env.update(
        {
            "MINERU_MODEL_CACHE_DIR": str(cache_dir),
            "HF_HOME": str(cache_dir / "huggingface"),
            "HUGGINGFACE_HUB_CACHE": str(cache_dir / "huggingface" / "hub"),
            "TRANSFORMERS_CACHE": str(cache_dir / "huggingface" / "transformers"),
            "MODELSCOPE_CACHE": str(cache_dir / "modelscope"),
            "TORCH_HOME": str(cache_dir / "torch"),
            "XDG_CACHE_HOME": str(cache_dir),
        }
    )
    return env


def _page(file: InputFile, page_number: int, text: str) -> ParsedPage:
    return ParsedPage(
        page_number=page_number,
        markdown=text,
        raw_text=text,
        blocks=[_block(file, page_number, text)],
    )


def _block(file: InputFile, page_number: int, text: str) -> LayoutBlock:
    return LayoutBlock(
        id=stable_id("block", file.id, page_number, text[:128]),
        page_number=page_number,
        kind="text",
        text=text,
    )
