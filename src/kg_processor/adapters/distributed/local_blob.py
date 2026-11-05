# SPDX-License-Identifier: Apache-2.0
"""Filesystem blob storage for local Spark tests and single-host execution."""

from __future__ import annotations

import os
from pathlib import Path
from shutil import copyfileobj, rmtree
from typing import BinaryIO
from urllib.parse import unquote, urlparse
from uuid import uuid4


class LocalBlobStore:
    """Persist immutable artifacts below one local filesystem root.

    This adapter is not a shared Kubernetes storage solution. It provides the
    identical blob contract for local Spark, single-host runs, and installations whose
    filesystem is genuinely shared by an external storage layer.
    """

    def __init__(self, root_uri: str) -> None:
        """Resolve and retain an absolute ``file://`` root directory."""

        parsed = urlparse(root_uri)
        if parsed.scheme != "file" or not parsed.path:
            raise ValueError("local artifact URI must be an absolute file:// URI")
        self.root = Path(unquote(parsed.path)).resolve()

    def initialize(self) -> None:
        """Create the artifact root idempotently."""

        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, payload: bytes, media_type: str) -> str:
        """Atomically publish bytes beneath a validated deterministic key."""

        _ = media_type
        path = self._key_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        return path.as_uri()

    def get(self, uri: str) -> bytes:
        """Read bytes only when the file remains beneath this store's root."""

        return self._uri_path(uri).read_bytes()

    def download_to(self, uri: str, destination: BinaryIO) -> None:
        """Stream a local object into a caller-owned binary destination."""

        with self._uri_path(uri).open("rb") as source:
            copyfileobj(source, destination, length=1024 * 1024)

    def delete_prefix(self, prefix: str) -> int:
        """Remove the directory a relative prefix names, beneath the root only."""

        directory = self._key_path(prefix)
        if directory == self.root or not directory.is_dir():
            return 0
        removed = sum(1 for path in directory.rglob("*") if path.is_file())
        rmtree(directory)
        return removed

    def _key_path(self, key: str) -> Path:
        """Resolve a relative object key and prevent traversal outside the root."""

        normalized = key.strip("/")
        if not normalized:
            raise ValueError("blob object key must not be blank")
        return self._ensure_beneath_root(self.root / normalized)

    def _uri_path(self, uri: str) -> Path:
        """Resolve a file URI and enforce store ownership."""

        parsed = urlparse(uri)
        if parsed.scheme != "file" or not parsed.path:
            raise ValueError("local artifact storage expects a file:// URI")
        return self._ensure_beneath_root(Path(unquote(parsed.path)))

    def _ensure_beneath_root(self, path: Path) -> Path:
        """Return a resolved path only when it is owned by this namespace."""

        resolved = path.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("artifact path escapes the configured local blob root")
        return resolved
