# SPDX-License-Identifier: Apache-2.0
"""Manifest-backed file source for pinned, reviewable input corpora."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from kg_processor.adapters.files.common import (
    build_local_input_file,
    is_supported_file,
    matches_include_globs,
)
from kg_processor.domain.documents import InputFile

_JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}


class ManifestFileSource:
    """Reads an explicit CSV/JSON manifest into canonical input files.

    ``containment_root`` confines every row to one directory. A manifest an
    operator wrote may legitimately point anywhere on the host, but one that
    arrived from a user must not be able to name ``/etc/passwd`` or another
    tenant's corpus, so the caller that accepts uploads passes the directory
    those rows are allowed to name.
    """

    def __init__(
        self,
        manifest_path: Path,
        include_globs: list[str] | None = None,
        *,
        containment_root: Path | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.include_globs = include_globs or ["**/*"]
        self.containment_root = containment_root

    def list_files(self) -> list[InputFile]:
        """Validate manifest rows and return supported file entries."""

        return sorted(self.iter_files(), key=lambda file: file.source_uri)

    def iter_files(self) -> Iterator[InputFile]:
        """Yield manifest entries without retaining corpus-sized row lists.

        CSV, JSON Lines, and newline-delimited JSON are streamed directly. Plain
        JSON arrays still use the standard-library parser and should be reserved
        for smaller manifests; JSONL is the scalable public interchange format.
        """

        for row in _iter_manifest_rows(self.manifest_path):
            path_value = _required_string(row, "path")
            path = _resolve_manifest_path(self.manifest_path, path_value, self.containment_root)
            if not matches_include_globs(path_value, self.include_globs):
                continue
            _raise_if_unsupported_file(path)
            yield build_local_input_file(
                path,
                source_uri=_optional_string(row, "source_uri"),
                file_id=_optional_string(row, "file_id"),
                checksum=_optional_string(row, "checksum"),
                mime_type=_optional_string(row, "mime_type"),
                size_bytes=_optional_int(row, "size_bytes"),
                identity_hint=path_value.replace("\\", "/"),
            )


def manifest_candidate_paths(
    manifest_path: Path,
    include_globs: list[str],
    *,
    containment_root: Path | None = None,
) -> list[Path]:
    """Return manifest paths that match the same broad file-source filters.

    Preflight uses this to check OCR/provider compatibility without building
    full `InputFile` objects or hashing every document in a large pinned corpus.
    The worker still verifies existence and optional checksums before OCR.
    """

    candidates: list[Path] = []
    for row in _iter_manifest_rows(manifest_path):
        path_value = _required_string(row, "path")
        path = _resolve_manifest_path(manifest_path, path_value, containment_root)
        if matches_include_globs(path_value, include_globs):
            _raise_if_unsupported_file(path)
            candidates.append(path)
    return sorted(candidates)


def _iter_manifest_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    """Stream row-oriented manifests and validate the selected file format."""

    if not path.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield _object_row(json.loads(line), path, line_number)
        return
    if suffix in _JSON_SUFFIXES:
        yield from _read_json_rows(path)
        return
    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def _read_json_rows(path: Path) -> list[Mapping[str, Any]]:
    loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(loaded, dict) and isinstance(loaded.get("files"), list):
        return [_object_row(row, path, index) for index, row in enumerate(loaded["files"], 1)]
    if isinstance(loaded, list):
        return [_object_row(row, path, index) for index, row in enumerate(loaded, 1)]
    raise ValueError(f"JSON manifest must be a list or object with files list: {path}")


def _object_row(value: object, path: Path, row_number: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Manifest row {row_number} in {path} must be an object")
    return value


def _resolve_manifest_path(
    manifest_path: Path,
    value: str,
    containment_root: Path | None,
) -> Path:
    path = Path(value).expanduser()
    resolved = path if path.is_absolute() else (manifest_path.parent / path).resolve()
    if containment_root is not None:
        # Symlinks are followed on both sides so a link planted inside the root
        # cannot lead out of it, and the root itself is allowed to be a link.
        root = containment_root.expanduser().resolve()
        if not resolved.resolve().is_relative_to(root):
            raise ValueError(f"Manifest row names a file outside the allowed root {root}: {value}")
    return resolved


def _raise_if_unsupported_file(path: Path) -> None:
    if not is_supported_file(path):
        raise ValueError(f"Manifest file has unsupported suffix for OCR input: {path.suffix}")


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = _first_present(row, [key, "local_path"] if key == "path" else [key])
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest row requires non-empty {key}")
    return value


def _optional_string(row: Mapping[str, Any], key: str) -> str | None:
    value = _first_present(row, _aliases(key))
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"Manifest field {key} must be a string")
    return value


def _optional_int(row: Mapping[str, Any], key: str) -> int | None:
    value = _first_present(row, _aliases(key))
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"Manifest field {key} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise ValueError(f"Manifest field {key} must be an integer")


def _aliases(key: str) -> list[str]:
    if key == "file_id":
        return ["file_id", "id"]
    if key == "size_bytes":
        return ["size_bytes", "size", "bytes"]
    return [key]


def _first_present(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None
