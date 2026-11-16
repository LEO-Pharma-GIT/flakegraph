# SPDX-License-Identifier: Apache-2.0
"""Azure Blob file source for on-prem/local workers outside Snowflake stages.

Blobs are streamed into a local cache because OCR providers operate on local
paths, while the original Azure URI and checksum remain part of provenance.
"""

from __future__ import annotations

import mimetypes
import re
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit, urlunsplit

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

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


class AzureBlobDownloader(Protocol):
    """Minimal Azure downloader surface used for streaming blob bytes."""

    def chunks(self) -> Iterable[bytes]:
        """Yield byte chunks from a blob download."""
        ...


class AzureBlobContainerClient(Protocol):
    """Minimal container client surface used by the file-source adapter."""

    def list_blobs(self, name_starts_with: str | None = None) -> Iterable[object]:
        """List blobs under an optional prefix."""
        ...

    def download_blob(self, blob: str) -> AzureBlobDownloader:
        """Open a streaming downloader for one blob name."""
        ...


class AzureBlobServiceClient(Protocol):
    """Minimal service client surface for selecting a blob container."""

    def get_container_client(self, container: str) -> AzureBlobContainerClient:
        """Return the client for the configured container."""
        ...


AzureBlobClientFactory = Callable[["AzureBlobFileSourceConfig"], AzureBlobServiceClient]


@dataclass(frozen=True)
class AzureBlobFileSourceConfig:
    """Connection and download settings for Azure Blob file discovery."""

    account_url: str | None
    connection_string: str | None
    container: str
    prefix: str | None
    sas_token: str | None
    download_path: Path


class AzureBlobFileSource:
    """Downloads supported Azure blobs into a local cache as input files."""

    def __init__(
        self,
        config: AzureBlobFileSourceConfig,
        include_globs: list[str] | None = None,
        client_factory: AzureBlobClientFactory | None = None,
    ) -> None:
        self.config = config
        self.include_globs = include_globs or ["**/*"]
        self.client_factory = client_factory or _load_azure_blob_client

    def list_files(self) -> list[InputFile]:
        """Return all normalized Azure inputs, ordered by source URI."""

        return sorted(self.iter_files(), key=lambda item: item.source_uri)

    def iter_files(self) -> Iterator[InputFile]:
        """Stream filtered Azure downloads through a bounded worker pool.

        Azure enumeration remains lazy on the caller thread because cloud SDK
        iterators are not generally safe to consume concurrently. Only independent
        blob downloads enter the bounded pool. ``buffersize`` prevents enumeration
        of a very large container from creating one future per object. Azure's
        listing order and ordered mapping give the distributed planner stable input
        while avoiding a corpus-sized list of rich ``InputFile`` records.
        """

        client = self.client_factory(self.config)
        container_client = client.get_container_client(self.config.container)
        prefix = normalized_prefix(self.config.prefix)
        candidates = self._download_candidates(container_client, prefix)
        download_root = _download_root(self.config)
        with ThreadPoolExecutor(max_workers=DOWNLOAD_PARALLELISM) as executor:
            yield from bounded_map(
                executor,
                lambda candidate: _download_input_file(
                    self.config,
                    container_client,
                    candidate,
                    download_root,
                ),
                candidates,
                buffersize=DOWNLOAD_PARALLELISM,
            )

    def browse(self, limit: int) -> Iterator[SourceListing]:
        """Name up to ``limit`` supported blobs from the listing alone."""

        client = self.client_factory(self.config)
        container_client = client.get_container_client(self.config.container)
        prefix = normalized_prefix(self.config.prefix)
        candidates = self._download_candidates(container_client, prefix)
        for index, (blob, blob_name, relative_path) in enumerate(candidates):
            if index >= limit:
                return
            size = _object_value(blob, "size")
            modified = _object_value(blob, "last_modified")
            etag = _object_value(blob, "etag")
            yield SourceListing(
                uri=_source_uri(self.config, blob_name),
                name=relative_path,
                size_bytes=int(size) if isinstance(size, int | str) and str(size) else None,
                modified_at=(
                    modified.isoformat()
                    if isinstance(modified, datetime)
                    else str(modified)
                    if modified
                    else None
                ),
                checksum=str(etag).strip('"') if etag else None,
            )

    def _download_candidates(
        self,
        container_client: AzureBlobContainerClient,
        prefix: str,
    ) -> Iterable[tuple[object, str, str]]:
        """Yield supported blob descriptors without buffering the full listing."""

        for blob in container_client.list_blobs(name_starts_with=prefix or None):
            blob_name = _blob_name(blob)
            relative_path = _relative_blob_path(blob_name, prefix)
            if not is_supported_file(Path(relative_path)):
                continue
            if not matches_include_globs(relative_path, self.include_globs):
                continue
            yield blob, blob_name, relative_path


def _download_input_file(
    config: AzureBlobFileSourceConfig,
    container_client: AzureBlobContainerClient,
    candidate: tuple[object, str, str],
    download_root: Path,
) -> InputFile:
    """Download one independent candidate and build its immutable source record."""

    blob, blob_name, relative_path = candidate
    local_path = object_download_path(
        download_root,
        relative_path,
        stable_id("azure_blob_download_path", config.container, blob_name),
    )
    remote_identity = {
        name: str(_object_value(blob, source) or "")
        for name, source in (
            ("etag", "etag"),
            ("size", "size"),
            ("last_modified", "last_modified"),
        )
    }
    return fetch_input_file(
        local_path,
        remote_identity,
        _source_uri(config, blob_name),
        ("azure_blob_file", config.container, blob_name),
        _content_type(blob, blob_name),
        lambda: container_client.download_blob(blob_name).chunks(),
    )


def _load_azure_blob_client(config: AzureBlobFileSourceConfig) -> AzureBlobServiceClient:
    if config.connection_string:
        return cast(
            AzureBlobServiceClient,
            BlobServiceClient.from_connection_string(config.connection_string),
        )
    if not config.account_url:
        raise ValueError("azure_blob file source requires connection_string or account_url")
    # A SAS token is used as given; otherwise Azure's standard local, workload
    # and managed-identity credential chain applies.
    credential = config.sas_token or DefaultAzureCredential()
    return cast(
        AzureBlobServiceClient,
        BlobServiceClient(account_url=config.account_url, credential=credential),
    )


def _blob_name(blob: object) -> str:
    value = _object_value(blob, "name")
    if not isinstance(value, str) or not value:
        raise ValueError("Azure Blob list items must include a non-empty name")
    return value


def _content_type(blob: object, blob_name: str) -> str:
    """Prefer what the blob's own properties say, then the name's extension."""

    declared = getattr(getattr(blob, "content_settings", None), "content_type", None)
    return declared or mimetypes.guess_type(blob_name)[0] or "application/octet-stream"


def _object_value(value: object, key: str) -> object:
    return getattr(value, key, None)


def _relative_blob_path(blob_name: str, prefix: str) -> str:
    return blob_name.strip("/").removeprefix(prefix)


def _download_root(config: AzureBlobFileSourceConfig) -> Path:
    """Claim a run cache for one source and namespace any different source."""

    account = _azure_account_identity(config)
    source_id = stable_id(
        "azure_blob_download",
        account,
        config.container,
        normalized_prefix(config.prefix),
    )
    return claimed_download_root(config.download_path, source_id)


def _source_uri(config: AzureBlobFileSourceConfig, blob_name: str) -> str:
    if config.account_url:
        account_url = _strip_url_query(config.account_url).rstrip("/")
        return f"{account_url}/{config.container}/{blob_name}"
    return f"azblob://{_azure_account_identity(config)}/{config.container}/{blob_name}"


def _azure_account_identity(config: AzureBlobFileSourceConfig) -> str:
    """Return a non-secret account namespace for cache and provenance identity."""

    if config.account_url:
        return _strip_url_query(config.account_url).rstrip("/")
    connection_string = config.connection_string or ""
    match = re.search(r"(?:^|;)AccountName=([^;]+)", connection_string, re.IGNORECASE)
    if match:
        return match.group(1)
    return stable_id("azure_blob_account", connection_string)


def _strip_url_query(value: str) -> str:
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
