# SPDX-License-Identifier: Apache-2.0
"""Immutable blob payload storage used by distributed artifact metadata."""

from __future__ import annotations

from typing import BinaryIO, Protocol


class BlobStore(Protocol):
    """Store large immutable bytes outside the transactional task database.

    PostgreSQL remains authoritative for artifact identity, checksums, task
    ownership, and publication state. Implementations provide only the scalable
    byte plane so workers and distributed data engines can read objects directly.
    """

    def initialize(self) -> None:
        """Create whatever the store needs once; safe to call from every replica."""
        ...

    def put(self, key: str, payload: bytes, media_type: str) -> str:
        """Store bytes idempotently and return their canonical object URI."""
        ...

    def get(self, uri: str) -> bytes:
        """Return exact bytes for an object previously written by this store."""
        ...

    def download_to(self, uri: str, destination: BinaryIO) -> None:
        """Stream an object into an open binary destination without collecting it."""
        ...

    def delete_prefix(self, prefix: str) -> int:
        """Remove every object under a relative key prefix; return how many went.

        The prefix names a directory: ``run_1`` removes ``run_1/...`` and never
        ``run_10/...``. Deleting what is already gone removes nothing and is not
        an error, so a delete that was interrupted can simply be run again.
        """
        ...
