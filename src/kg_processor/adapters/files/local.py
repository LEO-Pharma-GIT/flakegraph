# SPDX-License-Identifier: Apache-2.0
"""Local filesystem file source for mounted-folder runs and fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from kg_processor.adapters.files.common import build_local_input_file, is_supported_file
from kg_processor.domain.documents import InputFile, SourceListing


class LocalFileSource:
    """Discovers supported files from a local file or directory."""

    def __init__(self, input_path: Path, include_globs: list[str] | None = None) -> None:
        self.input_path = input_path
        self.include_globs = include_globs or ["**/*"]
        for pattern in self.include_globs:
            normalized = pattern.replace("\\", "/")
            if (
                not normalized
                or PurePosixPath(normalized).is_absolute()
                or PureWindowsPath(normalized).drive
                or ".." in normalized.split("/")
            ):
                raise ValueError("Local include globs must be relative without parent components")

    def list_files(self) -> list[InputFile]:
        """Return supported local files as canonical input-file records."""

        return list(self.iter_files())

    def browse(self, limit: int) -> Iterator[SourceListing]:
        """Name up to ``limit`` supported files without hashing any of them."""

        for index, path in enumerate(self.candidate_paths()):
            if index >= limit:
                return
            stat = path.stat()
            yield SourceListing(
                uri=path.resolve().as_uri(),
                name=path.name,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            )

    def candidate_paths(self) -> Iterator[Path]:
        """Supported files under the input path, in path order."""

        candidates: set[Path] = set()
        if self.input_path.is_file():
            candidates.add(self.input_path)
        else:
            for pattern in self.include_globs:
                for path in self.input_path.glob(pattern):
                    if path.is_file():
                        candidates.add(path)
        root = (self.input_path if self.input_path.is_dir() else self.input_path.parent).resolve()
        for path in sorted(candidates):
            if not is_supported_file(path):
                continue
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Local input resolves outside its source directory: {path}")
            yield path

    def iter_files(self) -> Iterator[InputFile]:
        """Yield sorted local inputs while constructing one rich record at a time.

        A large corpus still needs a lightweight set of candidate paths to merge
        overlapping globs and preserve deterministic order. Checksums and Pydantic
        records are produced lazily, avoiding hundreds of megabytes of resident
        metadata before source staging can begin.
        """

        for path in self.candidate_paths():
            identity_root = self.input_path if self.input_path.is_dir() else self.input_path.parent
            yield build_local_input_file(
                path,
                identity_hint=path.relative_to(identity_root).as_posix(),
            )
