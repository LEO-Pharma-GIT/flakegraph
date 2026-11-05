"""Contract tests for the download cache shared by the cloud file sources."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from kg_processor.adapters.files.common import (
    cached_input_file,
    claimed_download_root,
    stream_to_path,
    verify_download_size,
)


def test_concurrent_downloads_publish_only_complete_files(tmp_path: Path) -> None:
    """Two workers streaming the same object leave one whole file and no parts."""

    target = tmp_path / "shared.pdf"
    barrier = Barrier(2)
    payloads = [b"AAAAAA", b"BBBBBB"]

    def chunks(payload: bytes) -> Iterable[bytes]:
        yield payload[:3]
        barrier.wait(timeout=2)
        yield payload[3:]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda payload: stream_to_path(chunks(payload), target), payloads)
        )

    assert target.read_bytes() in payloads
    assert sorted(checksum for checksum, _size in results) == sorted(
        hashlib.sha256(payload).hexdigest() for payload in payloads
    )
    assert not list(tmp_path.rglob("*.part"))


def test_concurrent_cache_claims_resolve_to_one_complete_root(tmp_path: Path) -> None:
    claim_id = "source-claim"
    worker_count = 16
    for round_index in range(20):
        download_path = tmp_path / str(round_index)
        rendezvous = Barrier(worker_count, timeout=2)

        def claim(
            _index: int,
            barrier: Barrier = rendezvous,
            root: Path = download_path,
        ) -> Path:
            barrier.wait()
            return claimed_download_root(root, claim_id)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            roots = list(executor.map(claim, range(worker_count)))

        assert roots == [download_path] * worker_count
        assert (download_path / ".flakegraph-download-source").read_text() == claim_id
        assert not list(download_path.glob("*.tmp"))


def test_cache_claim_reconciles_existing_marker_after_link_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_id = "source-claim"
    marker = tmp_path / ".flakegraph-download-source"
    marker.write_text(claim_id, encoding="utf-8")

    def fail_link(_source: Path, _target: Path) -> None:
        raise OSError("transient NFS link failure")

    monkeypatch.setattr("kg_processor.adapters.files.common.os.link", fail_link)

    assert claimed_download_root(tmp_path, claim_id) == tmp_path


def test_cache_claim_retries_transient_link_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_link = os.link
    attempts = 0

    def flaky_link(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient NFS link failure")
        real_link(source, target)

    monkeypatch.setattr("kg_processor.adapters.files.common.os.link", flaky_link)

    assert claimed_download_root(tmp_path, "source-claim") == tmp_path
    assert attempts == 2


def test_cache_ignores_non_utf8_metadata_sidecar(tmp_path: Path) -> None:
    local_path = tmp_path / "cached.txt"
    local_path.write_bytes(b"data")
    local_path.with_suffix(".txt.source.json").write_bytes(b"\xff\xfe")

    assert (
        cached_input_file(
            local_path,
            {"etag": "etag", "size": "4"},
            "s3://bucket/cached.txt",
            "fallback",
            "text/plain",
        )
        is None
    )


def test_download_size_verification_rejects_a_short_stream() -> None:
    """A stream that ends early must not become a valid checksum."""

    with pytest.raises(ValueError, match="Truncated download"):
        verify_download_size("s3://bucket/report.pdf", {"size": "12"}, 5)


def test_download_size_verification_accepts_listings_without_a_usable_size() -> None:
    """Sources that list no size stay usable rather than failing every download."""

    verify_download_size("s3://bucket/report.pdf", {"size": ""}, 5)
    verify_download_size("s3://bucket/report.pdf", {}, 5)
    verify_download_size("s3://bucket/report.pdf", {"size": "5"}, 5)
