# SPDX-License-Identifier: Apache-2.0
"""File-source port.

Every input backend exposes local-openable files plus the original source URI.
That split lets local OCR engines read bytes from disk while Snowflake/Cortex can
still preserve stage/blob provenance for auditing and reindexing.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from kg_processor.domain.documents import InputFile, SourceListing


class FileSource(Protocol):
    """Lists normalized input files for the pipeline to process."""

    def list_files(self) -> list[InputFile]:
        """Discover source documents and normalize them into input-file records."""
        ...


@runtime_checkable
class IterableFileSource(Protocol):
    """Optional source capability for bounded-memory corpus discovery."""

    def iter_files(self) -> Iterator[InputFile]:
        """Yield deterministically ordered input records without retaining them all."""
        ...


@runtime_checkable
class BrowsableFileSource(Protocol):
    """Optional source capability: name what is there without fetching any of it.

    A console browsing a bucket before a run needs counts, names and sizes; a
    listing that downloaded every object to answer would be the run itself.
    """

    def browse(self, limit: int) -> Iterator[SourceListing]:
        """Yield up to ``limit`` supported objects in the source's listing order."""
        ...
