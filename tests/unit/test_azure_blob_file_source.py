from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest

from kg_processor.adapters.files.azure_blob import (
    AzureBlobFileSource,
    AzureBlobFileSourceConfig,
)


def test_azure_blob_file_source_lists_filters_downloads_and_hashes(tmp_path: Path) -> None:
    container = _FakeContainer(
        {
            "incoming/documents/report.pdf": b"%PDF-data",
            "incoming/documents/readme.txt": b"hello",
            "incoming/images/photo.png": b"png",
            "incoming/ignore.tmp": b"ignored",
        }
    )
    source = AzureBlobFileSource(
        _config(tmp_path),
        include_globs=["documents/*"],
        client_factory=lambda _config: _FakeService(container),
    )

    files = source.list_files()

    assert [file.path.name for file in files] == ["readme.txt", "report.pdf"]
    assert files[0].path.read_bytes() == b"hello"
    assert files[0].path.parent.parent == tmp_path
    assert files[0].path.parent.name.startswith("azure_blob_download_path_")
    assert files[0].source_uri == (
        "https://storage.example/container/incoming/documents/readme.txt"
    )
    # The container's declared type wins over the one the name suggests, and
    # a blob that declares none falls back to the guess.
    assert files[0].mime_type == "text/custom"
    assert files[1].mime_type == "application/pdf"
    assert files[0].size_bytes == 5
    assert files[0].checksum == hashlib.sha256(b"hello").hexdigest()


def test_azure_blob_source_uses_distinct_paths_for_case_colliding_keys(tmp_path: Path) -> None:
    container = _FakeContainer(
        {
            "incoming/Report.pdf": b"upper",
            "incoming/report.pdf": b"lower",
        }
    )
    source = AzureBlobFileSource(
        _config(tmp_path),
        include_globs=["*.pdf"],
        client_factory=lambda _config: _FakeService(container),
    )

    files = source.list_files()

    assert len({item.path for item in files}) == 2
    assert {item.path.read_bytes() for item in files} == {b"upper", b"lower"}


def test_azure_blob_double_star_glob_matches_blob_at_prefix_root(tmp_path: Path) -> None:
    container = _FakeContainer(
        {
            "incoming/smoke.txt": b"hello",
            "incoming/nested/report.txt": b"nested",
            "incoming/ignore.tmp": b"ignored",
        }
    )
    source = AzureBlobFileSource(
        _config(tmp_path),
        include_globs=["**/*.txt"],
        client_factory=lambda _config: _FakeService(container),
    )

    files = source.list_files()

    assert [file.path.name for file in files] == ["report.txt", "smoke.txt"]
    assert len({file.path.parent for file in files}) == 2


def test_azure_blob_file_source_isolates_noncanonical_relative_paths(tmp_path: Path) -> None:
    container = _FakeContainer({"incoming/../secret.pdf": b"secret"})
    source = AzureBlobFileSource(
        _config(tmp_path),
        client_factory=lambda _config: _FakeService(container),
    )

    files = source.list_files()

    assert len(files) == 1
    assert files[0].path.read_bytes() == b"secret"
    assert files[0].path.resolve().is_relative_to(tmp_path.resolve())


def test_azure_blob_file_source_isolates_windows_drive_relative_paths(
    tmp_path: Path,
) -> None:
    container = _FakeContainer({"incoming/C:secret.pdf": b"secret"})
    source = AzureBlobFileSource(
        _config(tmp_path),
        client_factory=lambda _config: _FakeService(container),
    )

    files = source.list_files()

    assert len(files) == 1
    assert files[0].path.read_bytes() == b"secret"
    assert files[0].path.resolve().is_relative_to(tmp_path.resolve())


def test_azure_blob_file_source_uses_connection_string_source_uri(tmp_path: Path) -> None:
    container = _FakeContainer({"file.pdf": b"pdf"})
    source = AzureBlobFileSource(
        AzureBlobFileSourceConfig(
            account_url=None,
            connection_string="UseDevelopmentStorage=true",
            container="container",
            prefix=None,
            sas_token=None,
            download_path=tmp_path,
        ),
        client_factory=lambda _config: _FakeService(container),
    )

    files = source.list_files()

    assert files[0].source_uri.startswith("azblob://azure_blob_account_")
    assert files[0].source_uri.endswith("/container/file.pdf")


def test_azure_blob_file_source_does_not_persist_sas_query_in_source_uri(
    tmp_path: Path,
) -> None:
    container = _FakeContainer({"file.pdf": b"pdf"})
    source = AzureBlobFileSource(
        AzureBlobFileSourceConfig(
            account_url="https://storage.example?sv=secret-version&sig=secret-signature",
            connection_string=None,
            container="container",
            prefix=None,
            sas_token="sas",
            download_path=tmp_path,
        ),
        client_factory=lambda _config: _FakeService(container),
    )

    files = source.list_files()

    assert files[0].source_uri == "https://storage.example/container/file.pdf"
    assert "sig=" not in files[0].source_uri


def test_azure_blob_file_source_requires_blob_names(tmp_path: Path) -> None:
    class BadContainer(_FakeContainer):
        def list_blobs(self, name_starts_with: str | None = None) -> list[object]:
            return [{}]

    source = AzureBlobFileSource(
        _config(tmp_path),
        client_factory=lambda _config: _FakeService(BadContainer({})),
    )

    with pytest.raises(ValueError, match="non-empty name"):
        source.list_files()


def test_azure_blob_file_source_downloads_independent_blobs_concurrently(
    tmp_path: Path,
) -> None:
    """Prevent large remote containers from regressing to serial network I/O."""

    rendezvous = Barrier(2, timeout=2)

    class ConcurrentDownloader(_FakeDownloader):
        def chunks(self) -> list[bytes]:
            rendezvous.wait()
            return super().chunks()

    class ConcurrentContainer(_FakeContainer):
        def download_blob(self, blob: str) -> _FakeDownloader:
            return ConcurrentDownloader(self.blobs[blob])

    source = AzureBlobFileSource(
        _config(tmp_path),
        client_factory=lambda _config: _FakeService(
            ConcurrentContainer(
                {
                    "incoming/first.pdf": b"first",
                    "incoming/second.pdf": b"second",
                }
            )
        ),
    )

    assert len(source.list_files()) == 2


def test_azure_blob_file_source_rejects_a_download_shorter_than_the_listing(
    tmp_path: Path,
) -> None:
    """A truncated stream must not be published as a valid checksum and cache entry."""

    class TruncatingDownloader(_FakeDownloader):
        def chunks(self) -> list[bytes]:
            return [self.payload[:2]]

    class TruncatingContainer(_FakeContainer):
        def download_blob(self, blob: str) -> _FakeDownloader:
            return TruncatingDownloader(self.blobs[blob])

    source = AzureBlobFileSource(
        _config(tmp_path),
        client_factory=lambda _config: _FakeService(
            TruncatingContainer({"incoming/report.pdf": b"complete-object"}, size_in_listing=True)
        ),
    )

    with pytest.raises(ValueError, match="Truncated download"):
        source.list_files()

    assert not list(tmp_path.rglob("*.source.json"))


def test_azure_blob_file_source_skips_macos_appledouble_sidecars(tmp_path: Path) -> None:
    """One corpus must yield the same documents through every file source."""

    source = AzureBlobFileSource(
        _config(tmp_path),
        client_factory=lambda _config: _FakeService(
            _FakeContainer(
                {
                    "incoming/report.pdf": b"document",
                    "incoming/._report.pdf": b"finder-metadata",
                }
            )
        ),
    )

    assert [file.path.name for file in source.list_files()] == ["report.pdf"]


def test_azure_blob_file_source_exposes_streaming_discovery(tmp_path: Path) -> None:
    """Let distributed planning consume cloud files without a corpus-sized list."""

    source = AzureBlobFileSource(
        _config(tmp_path),
        client_factory=lambda _config: _FakeService(
            _FakeContainer(
                {
                    "incoming/first.pdf": b"first",
                    "incoming/second.pdf": b"second",
                }
            )
        ),
    )

    iterator = source.iter_files()
    first = next(iterator)
    remaining = list(iterator)

    assert first.path.name == "first.pdf"
    assert [file.path.name for file in remaining] == ["second.pdf"]


def test_azure_blob_source_browses_the_listing_without_downloading(tmp_path: Path) -> None:
    container = _FakeContainer(
        {
            "incoming/documents/report.pdf": b"pdf-bytes",
            "incoming/scratch.tmp": b"ignored",
            "incoming/documents/readme.txt": b"hello",
        },
        size_in_listing=True,
    )
    source = AzureBlobFileSource(
        _config(tmp_path),
        client_factory=lambda _config: _FakeService(container),
    )

    listing = list(source.browse(limit=10))

    # Listing order is the container's own; only unsupported blobs drop out.
    assert [item.name for item in listing] == ["documents/report.pdf", "documents/readme.txt"]
    assert listing[0].uri == "https://storage.example/container/incoming/documents/report.pdf"
    assert listing[0].size_bytes == 9
    assert listing[0].modified_at == "2026-09-18T10:00:00+00:00"
    assert listing[0].checksum == "etag-incoming/documents/report.pdf"
    assert [item.name for item in source.browse(limit=1)] == ["documents/report.pdf"]
    assert container.downloaded == []
    assert not any(tmp_path.iterdir())


def _config(tmp_path: Path) -> AzureBlobFileSourceConfig:
    return AzureBlobFileSourceConfig(
        account_url="https://storage.example",
        connection_string=None,
        container="container",
        prefix="incoming",
        sas_token="sas",
        download_path=tmp_path,
    )


@dataclass
class _ContentSettings:
    content_type: str | None


@dataclass
class _FakeBlob:
    name: str
    content_type: str | None = None
    size: int | None = None
    last_modified: datetime | None = None
    etag: str | None = None

    @property
    def content_settings(self) -> _ContentSettings:
        return _ContentSettings(self.content_type)


class _FakeDownloader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def chunks(self) -> list[bytes]:
        midpoint = len(self.payload) // 2
        return [self.payload[:midpoint], self.payload[midpoint:]]


class _FakeContainer:
    def __init__(self, blobs: dict[str, bytes], size_in_listing: bool = False) -> None:
        self.blobs = blobs
        self.size_in_listing = size_in_listing
        self.downloaded: list[str] = []

    def list_blobs(self, name_starts_with: str | None = None) -> list[object]:
        return [
            _FakeBlob(
                name,
                "text/custom" if name.endswith(".txt") else None,
                len(payload) if self.size_in_listing else None,
                datetime(2026, 9, 18, 10, 0, tzinfo=UTC) if self.size_in_listing else None,
                f'"etag-{name}"' if self.size_in_listing else None,
            )
            for name, payload in self.blobs.items()
            if name_starts_with is None or name.startswith(name_starts_with)
        ]

    def download_blob(self, blob: str) -> _FakeDownloader:
        self.downloaded.append(blob)
        return _FakeDownloader(self.blobs[blob])


class _FakeService:
    def __init__(self, container: _FakeContainer) -> None:
        self.container = container

    def get_container_client(self, container: str) -> _FakeContainer:
        assert container == "container"
        return self.container
