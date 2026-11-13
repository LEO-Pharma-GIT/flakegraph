"""Contract tests for the S3-compatible document source adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from kg_processor.adapters.files.s3 import (
    S3FileSource,
    S3FileSourceConfig,
)


def test_s3_source_filters_streams_downloads_and_preserves_provenance(tmp_path: Path) -> None:
    client = _FakeS3Client(
        {
            "incoming/report.pdf": b"pdf-content",
            "incoming/notes.txt": b"notes",
            "incoming/image.png": b"ignored",
        }
    )
    source = S3FileSource(
        _config(tmp_path),
        include_globs=["*.pdf", "*.txt"],
        client=client,
    )

    files = source.list_files()

    assert [item.path.name for item in files] == ["notes.txt", "report.pdf"]
    assert files[0].source_uri == "s3://documents/incoming/notes.txt"
    assert files[0].path.read_bytes() == b"notes"
    assert files[0].checksum == hashlib.sha256(b"notes").hexdigest()
    assert files[0].mime_type == "text/plain"
    # Downloads are intentionally concurrent, so request scheduling is not part
    # of the adapter contract. The returned InputFile order above is stable.
    assert sorted(client.requested_keys) == ["incoming/notes.txt", "incoming/report.pdf"]


def test_s3_source_uses_distinct_paths_for_case_colliding_object_keys(tmp_path: Path) -> None:
    source = S3FileSource(
        _config(tmp_path),
        include_globs=["*.pdf"],
        client=_FakeS3Client(
            {
                "incoming/Report.pdf": b"upper",
                "incoming/report.pdf": b"lower",
            }
        ),
    )

    files = source.list_files()

    assert len({item.path for item in files}) == 2
    assert {item.path.read_bytes() for item in files} == {b"upper", b"lower"}


def test_s3_source_exposes_lazy_discovery(tmp_path: Path) -> None:
    source = S3FileSource(
        _config(tmp_path),
        client=_FakeS3Client(
            {
                "incoming/first.pdf": b"first",
                "incoming/second.pdf": b"second",
            }
        ),
    )

    iterator = source.iter_files()
    first = next(iterator)

    assert first.path.name == "first.pdf"
    assert [item.path.name for item in iterator] == ["second.pdf"]


def test_s3_source_browses_the_listing_without_downloading(tmp_path: Path) -> None:
    client = _FakeS3Client(
        {
            "incoming/first.pdf": b"first",
            "incoming/scratch.tmp": b"ignored",
            "incoming/second.txt": b"second",
        },
        listed_at=datetime(2026, 9, 18, 10, 0, tzinfo=UTC),
    )
    source = S3FileSource(_config(tmp_path), client=client)

    listing = list(source.browse(limit=1))
    everything = list(source.browse(limit=10))

    assert [item.name for item in listing] == ["first.pdf"]
    assert [item.uri for item in everything] == [
        "s3://documents/incoming/first.pdf",
        "s3://documents/incoming/second.txt",
    ]
    assert everything[0].size_bytes == 5
    assert everything[0].modified_at == "2026-09-18T10:00:00+00:00"
    assert everything[0].checksum == "etag-incoming/first.pdf"
    assert client.requested_keys == []
    assert not any(tmp_path.iterdir())


def test_s3_source_isolates_noncanonical_object_key(tmp_path: Path) -> None:
    source = S3FileSource(
        _config(tmp_path),
        client=_FakeS3Client({"incoming/../private.pdf": b"private"}),
    )

    files = source.list_files()

    assert len(files) == 1
    assert files[0].path.read_bytes() == b"private"
    assert files[0].path.resolve().is_relative_to(tmp_path.resolve())


def test_s3_source_rejects_missing_streaming_body(tmp_path: Path) -> None:
    client = _FakeS3Client({"incoming/report.pdf": b"content"})
    client.omit_body = True
    source = S3FileSource(_config(tmp_path), client=client)

    with pytest.raises(ValueError, match="streaming body"):
        source.list_files()


def test_s3_source_rejects_a_download_shorter_than_the_listing(tmp_path: Path) -> None:
    """A truncated stream must not be published as a valid checksum and cache entry."""

    client = _FakeS3Client({"incoming/report.pdf": b"complete-object"})
    client.truncate_after_bytes = 4
    source = S3FileSource(_config(tmp_path), client=client)

    with pytest.raises(ValueError, match="Truncated download"):
        source.list_files()

    assert not list(tmp_path.rglob("*.source.json"))


def test_s3_source_skips_macos_appledouble_sidecars(tmp_path: Path) -> None:
    """One corpus must yield the same documents through every file source."""

    source = S3FileSource(
        _config(tmp_path),
        client=_FakeS3Client(
            {
                "incoming/report.pdf": b"document",
                "incoming/._report.pdf": b"finder-metadata",
            }
        ),
    )

    assert [item.path.name for item in source.list_files()] == ["report.pdf"]


def _config(tmp_path: Path) -> S3FileSourceConfig:
    return S3FileSourceConfig(
        bucket="documents",
        prefix="incoming",
        endpoint_url="https://objects.example",
        region="eu-north-1",
        download_path=tmp_path,
    )


class _FakeBody:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def iter_chunks(self, chunk_size: int = 1024) -> Iterable[bytes]:
        return (
            self.payload[index : index + chunk_size]
            for index in range(0, len(self.payload), chunk_size)
        )


class _FakePaginator:
    def __init__(self, objects: Mapping[str, bytes], listed_at: datetime | None) -> None:
        self.objects = objects
        self.listed_at = listed_at

    def paginate(self, *, Bucket: str, Prefix: str) -> Iterable[dict[str, Any]]:
        assert Bucket == "documents"
        yield {
            "Contents": [
                {
                    "Key": key,
                    "Size": len(payload),
                    # boto3 hands back a datetime and a quoted ETag; the fake does too.
                    **(
                        {"LastModified": self.listed_at, "ETag": f'"etag-{key}"'}
                        if self.listed_at
                        else {}
                    ),
                }
                for key, payload in self.objects.items()
                if key.startswith(Prefix)
            ]
        }


class _FakeS3Client:
    def __init__(self, objects: Mapping[str, bytes], listed_at: datetime | None = None) -> None:
        self.objects = objects
        self.listed_at = listed_at
        self.requested_keys: list[str] = []
        self.omit_body = False
        self.truncate_after_bytes: int | None = None

    def get_paginator(self, operation_name: str) -> _FakePaginator:
        assert operation_name == "list_objects_v2"
        return _FakePaginator(self.objects, self.listed_at)

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        assert Bucket == "documents"
        self.requested_keys.append(Key)
        if self.omit_body:
            return {}
        payload = self.objects[Key]
        if self.truncate_after_bytes is not None:
            payload = payload[: self.truncate_after_bytes]
        return {"Body": _FakeBody(payload)}
