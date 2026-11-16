# SPDX-License-Identifier: Apache-2.0
"""Snowflake stage file source.

The adapter lists stage contents but still exposes the same `InputFile` contract
as local/manifest sources so the pipeline does not need Snowflake-specific file
logic.

Snowflake reports its own MD5 for staged files. Every other source records a
SHA-256 of the bytes, so adopting the provider value would give one document two
identities depending on which runtime ingested it, and would silently fail any
comparison that binds a document by checksum. The bytes are therefore read once
to produce the same content hash as everywhere else, while the provider value is
retained separately as provenance.
"""

from __future__ import annotations

import gzip
import mimetypes
import tempfile
from pathlib import Path
from typing import Any

from kg_processor.adapters.files.common import is_supported_file, matches_include_globs
from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    ReusableSnowflakeConnections,
    SnowflakeConnectionConfig,
    stage_path,
    validate_stage_location,
)
from kg_processor.domain.documents import InputFile
from kg_processor.domain.ids import sha256_file, stable_id

_LIST_ROW_MIN_COLUMNS = 2
_LIST_ROW_CHECKSUM_COLUMNS = 3
_LIST_ROW_LAST_MODIFIED_COLUMNS = 4
# A stage may store a document compressed. That is a property of the stage, not
# of the document, so the suffix is stripped before the file is identified.
# A staged document is a document: this ceiling is far above any real corpus
# file and far below what fills a worker's disk.
_MAX_DECOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
_DECOMPRESSION_CHUNK_BYTES = 1024 * 1024
_STAGE_COMPRESSION_SUFFIXES = {".gz"}


class SnowflakeStageFileSource:
    """Lists supported files from a Snowflake stage."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        stage: str,
        prefix: str | None = None,
        include_globs: list[str] | None = None,
        connector_factory: ConnectorFactory | None = None,
        content_hash: bool = True,
    ) -> None:
        self.config = config
        self.stage = stage
        self.prefix = prefix
        self.include_globs = include_globs or ["**/*"]
        self.connector_factory = connector_factory
        self.content_hash = content_hash
        self._connections = ReusableSnowflakeConnections(config, connector_factory)

    def close(self) -> None:
        """Release retained Snowflake sessions."""

        self._connections.close()

    def list_files(self) -> list[InputFile]:
        """Run LIST against the configured stage and normalize the returned rows."""

        location = stage_path(self.stage, self.prefix)
        with self._connections.cursor() as cursor:
            cursor.execute(f"LIST {location}")
            rows = cursor.fetchall()

        files = [_input_file_from_list_row(self.stage, row) for row in rows]
        selected = [
            file
            for file in sorted(files, key=lambda item: item.source_uri)
            if is_supported_file(file.path)
            and matches_include_globs(file.path.as_posix(), self.include_globs)
        ]
        if not self.content_hash:
            return selected
        return [self._with_content_hash(file) for file in selected]

    def _with_content_hash(self, file: InputFile) -> InputFile:
        """Replace the provider checksum with a hash of the file's own bytes.

        Identity derives from the checksum, so it is recomputed here rather than
        patched afterwards: a file id built from Snowflake's MD5 would not match
        the same document ingested through any other source.
        """

        digest = self._download_and_hash(file.source_uri)
        if not digest:
            return file
        return file.model_copy(
            update={
                "id": stable_id("stage_file", file.source_uri, digest),
                "checksum": digest,
                "provider_checksum": file.checksum,
            }
        )

    def _download_and_hash(self, source_uri: str) -> str:
        # The object's name inside the stage decides this statement, and anyone
        # who can write to the stage chooses it. Unvalidated, a name carrying a
        # second `file://` target and a comment redirects the download to a path
        # of the writer's choosing inside the worker.
        validate_stage_location(source_uri)
        with tempfile.TemporaryDirectory(prefix="flakegraph-stage-") as directory:
            with self._connections.cursor() as cursor:
                cursor.execute(f"GET {source_uri} file://{directory}")
            downloaded = sorted(Path(directory).glob("*"))
            if not downloaded:
                raise FileNotFoundError(f"Snowflake GET returned no file for {source_uri}")
            return sha256_file(_restore_original_bytes(downloaded[0]))


def _restore_original_bytes(downloaded: Path) -> Path:
    """Undo stage-side compression so the hash describes the document itself.

    A file uploaded with ``AUTO_COMPRESS`` is stored, and returned, gzipped, and
    the stage path itself carries the ``.gz`` suffix. Hashing those bytes would
    describe how the stage chose to store the document rather than the document,
    and would not match the same file read anywhere else.
    """

    if downloaded.suffix.lower() not in _STAGE_COMPRESSION_SUFFIXES:
        return downloaded
    restored = downloaded.with_suffix("")
    # A gzip member declares nothing about how much it expands to, and a small
    # staged object can hold many gigabytes. Copied without a ceiling it fills
    # the worker's disk and takes the run down with it, so expansion is bounded
    # and an object that exceeds the bound is rejected as unusable input.
    written = 0
    with gzip.open(downloaded, "rb") as source, restored.open("wb") as target:
        while chunk := source.read(_DECOMPRESSION_CHUNK_BYTES):
            written += len(chunk)
            if written > _MAX_DECOMPRESSED_BYTES:
                restored.unlink(missing_ok=True)
                raise ValueError(
                    f"Staged object {downloaded.name} expands beyond "
                    f"{_MAX_DECOMPRESSED_BYTES} bytes and was not read"
                )
            target.write(chunk)
    return restored


def _document_path(relative_path: str) -> Path:
    """Return the staged object's path as the document it holds.

    Suffix-driven decisions downstream — OCR routing, MIME type, page-addressable
    formats, include globs — describe the document, so a stage-side compression
    suffix is not part of that identity. The stage path used by ``GET`` keeps it.
    """

    path = Path(relative_path)
    return path.with_suffix("") if path.suffix.lower() in _STAGE_COMPRESSION_SUFFIXES else path


def _input_file_from_list_row(stage: str, row: object) -> InputFile:
    values = _row_values(row)
    if len(values) < _LIST_ROW_MIN_COLUMNS:
        raise ValueError("Snowflake LIST rows must include at least name and size")
    raw_name = str(values[0])
    size_bytes = int(values[1] or 0)
    listed_checksum = str(values[2] or "") if len(values) >= _LIST_ROW_CHECKSUM_COLUMNS else ""
    last_modified = str(values[3] or "") if len(values) >= _LIST_ROW_LAST_MODIFIED_COLUMNS else ""
    checksum = listed_checksum or stable_id(
        "stage_file_checksum",
        raw_name,
        size_bytes,
        last_modified,
    )
    relative_path = _relative_path_from_list_name(stage, raw_name)
    source_uri = f"{stage.rstrip('/')}/{relative_path}"
    document_path = _document_path(relative_path)
    return InputFile(
        id=stable_id("stage_file", source_uri, checksum),
        path=document_path,
        source_uri=source_uri,
        checksum=checksum,
        mime_type=mimetypes.guess_type(document_path.name)[0] or "application/octet-stream",
        size_bytes=size_bytes,
    )


def _row_values(row: object) -> list[Any]:
    if isinstance(row, dict):
        return [
            row.get("name") or row.get("NAME"),
            row.get("size") or row.get("SIZE"),
            row.get("md5") or row.get("MD5") or row.get("etag") or row.get("ETAG"),
            row.get("last_modified") or row.get("LAST_MODIFIED"),
        ]
    if isinstance(row, tuple | list):
        return list(row)
    raise ValueError(f"Unsupported Snowflake LIST row type: {type(row).__name__}")


def _relative_path_from_list_name(stage: str, raw_name: str) -> str:
    name = raw_name.lstrip("@")
    stage_name = stage.lstrip("@").rstrip("/")
    if name.startswith(stage_name + "/"):
        return name[len(stage_name) + 1 :]
    if "/" in name:
        return name.split("/", 1)[1]
    return name
