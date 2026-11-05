"""Contract tests for local and S3-compatible distributed payload stores."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from botocore.exceptions import ClientError

from kg_processor.adapters.distributed.local_blob import LocalBlobStore
from kg_processor.adapters.distributed.s3_blob import S3BlobStore, S3BlobStoreConfig


def test_local_blob_store_round_trips_and_rejects_path_escape(tmp_path: Path) -> None:
    """Local development storage must preserve bytes within its owned namespace."""

    store = LocalBlobStore((tmp_path / "objects").as_uri())
    store.initialize()

    uri = store.put("run/entities/part.json", b'{"id": 1}', "application/json")

    assert store.get(uri) == b'{"id": 1}'
    with pytest.raises(ValueError, match="escapes"):
        store.put("../outside", b"bad", "application/octet-stream")


def test_s3_blob_store_uses_deterministic_path_style_objects(monkeypatch: Any) -> None:
    """S3-compatible endpoints receive exact keys, bytes, and media metadata."""

    client = _MemoryS3Client()
    client_kwargs: dict[str, Any] = {}

    def build_client(*_args: object, **kwargs: Any) -> _MemoryS3Client:
        """Capture transport configuration while returning the in-memory client."""

        client_kwargs.update(kwargs)
        return client

    monkeypatch.setattr(
        "kg_processor.adapters.distributed.s3_blob.boto3.client",
        build_client,
    )
    store = S3BlobStore(
        S3BlobStoreConfig(
            root_uri="s3://flakegraph/prefix",
            endpoint_url="http://object-store:8333",
            access_key_id="access",
            secret_access_key="secret",
        )
    )
    store.initialize()

    uri = store.put("run/chunks/part.parquet", b"parquet", "application/x-parquet")

    assert uri == "s3://flakegraph/prefix/run/chunks/part.parquet"
    assert store.get(uri) == b"parquet"
    assert client.objects == {("flakegraph", "prefix/run/chunks/part.parquet"): b"parquet"}
    assert client_kwargs["config"].max_pool_connections == 64


def test_s3_blob_store_percent_encodes_uri_reserved_key_characters(monkeypatch: Any) -> None:
    client = _MemoryS3Client()
    monkeypatch.setattr(
        "kg_processor.adapters.distributed.s3_blob.boto3.client",
        lambda *_args, **_kwargs: client,
    )
    store = S3BlobStore(S3BlobStoreConfig(root_uri="s3://flakegraph/prefix"))
    store.initialize()

    uri = store.put("run#1/query?2.json", b"payload", "application/json")

    assert uri == "s3://flakegraph/prefix/run%231/query%3F2.json"
    assert store.get(uri) == b"payload"


class _MemoryS3Body(BytesIO):
    """Provide the streaming-body close/read interface used by the adapter."""


class _MemoryS3Client:
    """Record the small boto3 subset exercised by the storage adapter."""

    def __init__(self) -> None:
        self.buckets: set[str] = set()
        self.objects: dict[tuple[str, str], bytes] = {}
        self.handlers: dict[str, Any] = {}
        self.meta = SimpleNamespace(events=SimpleNamespace(register=self.handlers.__setitem__))

    def head_bucket(self, *, Bucket: str) -> None:  # noqa: N803 - boto3 API spelling.
        """Return successfully after the test bucket has been initialized."""

        if Bucket not in self.buckets:
            raise ClientError(
                {
                    "Error": {"Code": "NoSuchBucket", "Message": "missing"},
                    "ResponseMetadata": {
                        "HTTPStatusCode": 404,
                        "HTTPHeaders": {},
                        "HostId": "test-host",
                        "RequestId": "test-request",
                        "RetryAttempts": 0,
                    },
                },
                "HeadBucket",
            )

    def create_bucket(self, *, Bucket: str) -> None:  # noqa: N803
        """Create one in-memory bucket."""

        self.buckets.add(Bucket)

    def put_object(  # noqa: N803
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
    ) -> None:
        """Store exact bytes while accepting boto3's content-type argument."""

        assert ContentType
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, _MemoryS3Body]:  # noqa: N803
        """Return a closeable streaming body."""

        return {"Body": _MemoryS3Body(self.objects[(Bucket, Key)])}

    def get_paginator(self, name: str) -> _MemoryS3Paginator:
        """List keys a page at a time, as boto3's paginator does."""

        assert name == "list_objects_v2"
        return _MemoryS3Paginator(self, page_size=2)

    def delete_objects(
        self,
        *,
        Bucket: str,  # noqa: N803
        Delete: dict[str, Any],  # noqa: N803
    ) -> dict[str, Any]:
        """Remove the named keys and report no errors."""

        for item in Delete["Objects"]:
            self.objects.pop((Bucket, item["Key"]), None)
        return {}


class _MemoryS3Paginator:
    """Pages over the in-memory objects under a prefix."""

    def __init__(self, client: _MemoryS3Client, page_size: int) -> None:
        self.client = client
        self.page_size = page_size

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:  # noqa: N803
        """Every matching key, split into pages."""

        keys = sorted(
            key
            for bucket, key in self.client.objects
            if bucket == Bucket and key.startswith(Prefix)
        )
        return [
            {"Contents": [{"Key": key} for key in keys[start : start + self.page_size]]}
            for start in range(0, len(keys), self.page_size)
        ]


def test_blob_stores_delete_one_runs_objects_and_nothing_else(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A run's prefix goes whole; a run whose id only starts the same stays."""

    local = LocalBlobStore((tmp_path / "blobs").as_uri())
    local.initialize()
    for key in ("run_1/a.json", "run_1/nested/b.json", "run_10/c.json"):
        local.put(key, b"x", "application/json")
    assert local.delete_prefix("run_1") == 2
    assert local.delete_prefix("run_1") == 0
    assert (tmp_path / "blobs" / "run_10" / "c.json").exists()
    with pytest.raises(ValueError):
        local.delete_prefix("../outside")

    client = _MemoryS3Client()
    monkeypatch.setattr(
        "kg_processor.adapters.distributed.s3_blob.boto3.client", lambda *_a, **_k: client
    )
    s3 = S3BlobStore(S3BlobStoreConfig(root_uri="s3://flakegraph/prefix"))
    s3.initialize()
    for key in ("run_1/a.json", "run_1/b.json", "run_1/c/d.json", "run_10/e.json"):
        s3.put(key, b"x", "application/json")
    assert s3.delete_prefix("run_1") == 3
    assert sorted(key for _bucket, key in client.objects) == ["prefix/run_10/e.json"]


def test_s3_bulk_deletes_carry_the_content_md5_every_s3_store_accepts(monkeypatch: Any) -> None:
    """MinIO refuses DeleteObjects without Content-MD5, which botocore does not send by default."""

    client = _MemoryS3Client()
    monkeypatch.setattr(
        "kg_processor.adapters.distributed.s3_blob.boto3.client", lambda *_a, **_k: client
    )
    S3BlobStore(S3BlobStoreConfig(root_uri="s3://flakegraph/prefix"))
    handler = client.handlers["before-sign.s3.DeleteObjects"]
    body = b"<Delete><Object><Key>run_1/a.json</Key></Object></Delete>"
    request = SimpleNamespace(body=body, headers={})
    handler(request=request)
    assert request.headers["Content-MD5"] == base64.b64encode(hashlib.md5(body).digest()).decode()
    streamed = SimpleNamespace(body=BytesIO(body), headers={})
    handler(request=streamed)
    assert streamed.body == body
    assert streamed.headers["Content-MD5"] == request.headers["Content-MD5"]
