# SPDX-License-Identifier: Apache-2.0
"""S3-compatible object storage file source.

The adapter deliberately targets the small boto3 S3 client surface rather than
AWS-specific orchestration. It therefore works with AWS S3, MinIO, and other
S3-compatible stores while preserving the same local-file contract expected by
OCR providers.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import boto3

from kg_processor.adapters.files.common import (
    DOWNLOAD_PARALLELISM,
    claimed_download_root,
    fetch_input_file,
    is_supported_file,
    matches_include_globs,
    normalized_prefix,
    object_download_path,
)
from kg_processor.application.ordered_map import bounded_map
from kg_processor.domain.documents import InputFile, SourceListing
from kg_processor.domain.ids import stable_id


class S3Body(Protocol):
    """Streaming response body returned by ``get_object``."""

    def iter_chunks(self, chunk_size: int = ...) -> Iterable[bytes]:
        """Yield bounded byte chunks from one object."""
        ...


class S3Client(Protocol):
    """Minimal boto3-compatible client surface used by this adapter."""

    def get_paginator(self, operation_name: str) -> Any:
        """Return a paginator for ``list_objects_v2``."""
        ...

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        """Open one object for streaming download."""
        ...


@dataclass(frozen=True)
class S3FileSourceConfig:
    """Connection, namespace, and local staging settings for an S3 source."""

    bucket: str
    prefix: str | None
    endpoint_url: str | None
    region: str | None
    download_path: Path


class S3FileSource:
    """Download supported objects from an S3-compatible bucket."""

    def __init__(
        self,
        config: S3FileSourceConfig,
        include_globs: list[str] | None = None,
        client: S3Client | None = None,
    ) -> None:
        """Retain source settings and an optional injected client for tests."""

        self.config = config
        self.include_globs = include_globs or ["**/*"]
        self.client = client

    def list_files(self) -> list[InputFile]:
        """Return all downloaded objects in deterministic source-URI order."""

        return sorted(self.iter_files(), key=lambda item: item.source_uri)

    def iter_files(self) -> Iterator[InputFile]:
        """List lazily and download objects through a bounded worker pool.

        The bounded map prevents a bucket with hundreds of thousands of keys
        from allocating one future per object. Checksums are calculated from the
        downloaded bytes because multipart ETags are not content hashes.
        """

        client = self.client or _build_s3_client(self.config)
        candidates = self._candidates(client)
        download_root = _download_root(self.config)
        with ThreadPoolExecutor(max_workers=DOWNLOAD_PARALLELISM) as executor:
            yield from bounded_map(
                executor,
                lambda candidate: _download_input_file(
                    self.config,
                    client,
                    candidate,
                    download_root,
                ),
                candidates,
                buffersize=DOWNLOAD_PARALLELISM,
            )

    def browse(self, limit: int) -> Iterator[SourceListing]:
        """Name up to ``limit`` supported objects from the listing alone."""

        client = self.client or _build_s3_client(self.config)
        for index, (key, relative_path, identity) in enumerate(self._candidates(client)):
            if index >= limit:
                return
            yield SourceListing(
                uri=f"s3://{self.config.bucket}/{key}",
                name=relative_path,
                size_bytes=int(identity["size"]) if identity.get("size") else None,
                modified_at=_iso_timestamp(identity.get("last_modified", "")),
                checksum=identity.get("etag", "").strip('"') or None,
            )

    def _candidates(self, client: S3Client) -> Iterable[tuple[str, str, dict[str, str]]]:
        """Yield supported object keys and metadata without buffering the listing."""

        prefix = normalized_prefix(self.config.prefix)
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item.get("Key", ""))
                if not key or key.endswith("/"):
                    continue
                relative_path = _relative_object_path(key, prefix)
                if not is_supported_file(Path(relative_path)):
                    continue
                if not matches_include_globs(relative_path, self.include_globs):
                    continue
                identity = {
                    name: str(item.get(source) or "")
                    for name, source in (
                        ("etag", "ETag"),
                        ("size", "Size"),
                        ("last_modified", "LastModified"),
                    )
                }
                yield key, relative_path, identity


def _build_s3_client(config: S3FileSourceConfig) -> S3Client:
    """Build a boto3 client using its standard credential provider chain."""

    return cast(
        S3Client,
        boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region,
        ),
    )


def _download_input_file(
    config: S3FileSourceConfig,
    client: S3Client,
    candidate: tuple[str, str, dict[str, str]],
    download_root: Path,
) -> InputFile:
    """Download one object atomically enough for a single worker invocation."""

    key, relative_path, remote_identity = candidate
    local_path = object_download_path(
        download_root,
        relative_path,
        stable_id("s3_download_path", config.bucket, key),
    )

    def open_body() -> Iterable[bytes]:
        body = client.get_object(Bucket=config.bucket, Key=key).get("Body")
        if body is None or not hasattr(body, "iter_chunks"):
            raise ValueError(f"S3 object response for {key!r} did not include a streaming body")
        return cast(S3Body, body).iter_chunks(chunk_size=1024 * 1024)

    return fetch_input_file(
        local_path,
        remote_identity,
        f"s3://{config.bucket}/{key}",
        ("s3_file", config.bucket, key),
        # Judged from the key on every path, so a file's type does not depend on
        # whether it was downloaded now or found in the cache from an earlier run.
        mimetypes.guess_type(key)[0] or "application/octet-stream",
        open_body,
    )


def _iso_timestamp(value: str) -> str | None:
    """Render the listing's ``LastModified`` (stringified by ``_candidates``) as ISO 8601."""

    if not value:
        return None
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return value


def _relative_object_path(key: str, prefix: str) -> str:
    """Return the path below the selected prefix and reject empty results."""

    relative = key.removeprefix(prefix)
    if not relative:
        raise ValueError("S3 object key resolves to an empty relative path")
    return relative


def _download_root(config: S3FileSourceConfig) -> Path:
    """Claim a run cache for one source and namespace any different source."""

    source_id = stable_id(
        "s3_download",
        config.endpoint_url or "aws",
        config.bucket,
        normalized_prefix(config.prefix),
    )
    return claimed_download_root(config.download_path, source_id)
