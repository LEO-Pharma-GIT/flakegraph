# SPDX-License-Identifier: Apache-2.0
"""Shared file-source helpers for globs, checksums, MIME types, and safe paths.

Object-storage sources also share the local download cache: the claim marker
that namespaces one source's cache directory, the sidecar metadata that lets a
later run reuse a download, and the path and prefix normalization both depend
on. One implementation keeps a correction to any of that from landing in only
one of the cloud adapters.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import uuid
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from pathlib import Path, PurePosixPath
from time import sleep

from kg_processor.domain.documents import InputFile
from kg_processor.domain.ids import sha256_file, stable_id

SUPPORTED_SUFFIXES = {
    ".pdf",
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".pptx",
    ".xlsx",
}
# Object listings and downloads are independent per object, so the pool only
# bounds how many transfers a single source may run at once.
DOWNLOAD_PARALLELISM = 16
_CLAIM_MARKER_READ_ATTEMPTS = 3
_CLAIM_MARKER_NAME = ".flakegraph-download-source"
_DOWNLOAD_METADATA_SUFFIX = ".source.json"


def build_local_input_file(
    path: Path,
    *,
    source_uri: str | None = None,
    file_id: str | None = None,
    checksum: str | None = None,
    mime_type: str | None = None,
    size_bytes: int | None = None,
    identity_hint: str | None = None,
) -> InputFile:
    """Validate a local file and build the canonical input-file record."""

    if not path.is_file():
        raise FileNotFoundError(f"Manifest file entry does not exist: {path}")
    actual_size = path.stat().st_size
    if size_bytes is not None and size_bytes != actual_size:
        raise ValueError(
            f"Manifest size mismatch for {path}: expected {size_bytes}, got {actual_size}"
        )
    actual_checksum = sha256_file(path)
    if checksum and checksum != actual_checksum:
        raise ValueError(
            f"Manifest checksum mismatch for {path}: expected {checksum}, got {actual_checksum}"
        )
    effective_source_uri = source_uri or path.resolve().as_uri()
    return InputFile(
        # Equal bytes at separate source locations are separate documents with
        # separate provenance. A corpus-relative hint keeps that identity stable
        # when the same corpus is mounted under another checkout/root directory.
        id=file_id or stable_id("file", identity_hint or path.name, actual_checksum),
        path=path,
        source_uri=effective_source_uri,
        checksum=actual_checksum,
        mime_type=mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        size_bytes=actual_size,
    )


def is_supported_file(path: Path) -> bool:
    """Return whether a real document suffix has a corresponding OCR adapter.

    macOS AppleDouble sidecars preserve Finder metadata as ``._<name>`` files.
    Their inherited document suffix must not make them corpus inputs when data is
    copied through tar, SMB, or another metadata-preserving transport.
    """

    return not path.name.startswith("._") and path.suffix.lower() in SUPPORTED_SUFFIXES


def matches_include_globs(label: str, include_globs: Iterable[str]) -> bool:
    """Match a normalized path label against configured include globs."""

    normalized = label.replace("\\", "/").removeprefix("./")
    return any(_matches_glob(normalized, pattern) for pattern in include_globs)


def _matches_glob(normalized: str, pattern: str) -> bool:
    if pattern in {"*", "**", "**/*"}:
        return True
    return _glob_regex(pattern).fullmatch(str(PurePosixPath(normalized))) is not None


@lru_cache(maxsize=256)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Compile a whole-path glob the way ``PurePath.full_match`` reads it.

    ``full_match`` arrived in Python 3.13; this keeps the same contract on 3.12:
    ``**`` spans any number of directories including none, ``*`` and ``?`` never
    cross a separator, and the pattern is anchored at both ends so ``*.pdf``
    only names files at the root while ``**/*.pdf`` names them anywhere.
    """

    parts: list[str] = []
    segments = pattern.split("/")
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == "**":
            parts.append("(?:.*)" if last else "(?:.*/)?")
            continue
        parts.append(_glob_segment_regex(segment) + ("" if last else "/"))
    return re.compile("".join(parts))


def _glob_segment_regex(segment: str) -> str:
    """Translate one path segment of a glob into a regex that stays inside it."""

    out: list[str] = []
    index = 0
    while index < len(segment):
        char = segment[index]
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            close = segment.find("]", index + 1)
            if close == -1:
                out.append(re.escape(char))
            else:
                body = segment[index + 1 : close]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                index = close
        else:
            out.append(re.escape(char))
        index += 1
    return "".join(out)


def stream_to_path(chunks: Iterable[bytes], local_path: Path) -> tuple[str, int]:
    """Write a remote object's chunks to ``local_path`` and return its digest and size.

    The bytes land in a uniquely named temporary file and are published with
    one atomic replace, so a concurrent reader sees either nothing or the
    complete object, and a failed download leaves nothing behind.
    """

    local_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size_bytes = 0
    temporary_path = local_path.with_name(f".{local_path.name}.{uuid.uuid4().hex}.part")
    try:
        with temporary_path.open("wb") as handle:
            for chunk in chunks:
                digest.update(chunk)
                size_bytes += len(chunk)
                handle.write(chunk)
        temporary_path.replace(local_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return digest.hexdigest(), size_bytes


def normalized_prefix(prefix: str | None) -> str:
    """Normalize an optional container prefix to a relative directory prefix."""

    return prefix.strip("/") + "/" if prefix and prefix.strip("/") else ""


def object_download_path(download_root: Path, relative_path: str, identity: str) -> Path:
    """Return a collision-free staging path while retaining the source suffix.

    Object keys are opaque strings rather than paths: a remote store accepts an
    empty segment, a backslash, a leading slash, and ``..`` alike. Only the final
    component is used, and the directory is the caller's identity for the whole
    key, so a key can neither escape the download root nor land on the file
    another key downloads to. Validating keys as paths instead would refuse
    objects the remote store considers perfectly ordinary.
    """

    name = PurePosixPath(relative_path.rstrip("/")).name
    return download_root / identity / (name or identity)


def verify_download_size(
    source_uri: str,
    remote_identity: Mapping[str, str],
    size_bytes: int,
) -> None:
    """Reject a download whose length disagrees with the listing.

    A stream that ends early yields a self-consistent record whose checksum
    describes a fragment of the document, and which is then published to the
    download cache and reused by every later run of the same corpus. Listings
    that carry no usable size are left unverified rather than rejected.
    """

    listed = str(remote_identity.get("size", "")).strip()
    if not listed.isdigit() or int(listed) == size_bytes:
        return
    raise ValueError(
        f"Truncated download for {source_uri}: the listing declares {listed} bytes "
        f"but {size_bytes} bytes arrived"
    )


def fetch_input_file(
    local_path: Path,
    remote_identity: dict[str, str],
    source_uri: str,
    identity: tuple[str, ...],
    mime_type: str,
    open_chunks: Callable[[], Iterable[bytes]],
) -> InputFile:
    """Return the cached download when the listing still matches, else fetch it.

    ``identity`` is the id namespace and the parts that name the object; the
    checksum completes it once the bytes are known. The stream is opened only
    on a cache miss.
    """

    cached = cached_input_file(
        local_path, remote_identity, source_uri, stable_id(*identity), mime_type
    )
    if cached is not None:
        return cached
    checksum, size_bytes = stream_to_path(open_chunks(), local_path)
    verify_download_size(source_uri, remote_identity, size_bytes)
    result = InputFile(
        id=stable_id(*identity, checksum),
        path=local_path,
        source_uri=source_uri,
        checksum=checksum,
        mime_type=mime_type,
        size_bytes=size_bytes,
    )
    write_download_metadata(local_path, remote_identity, result)
    return result


def cached_input_file(
    local_path: Path,
    remote_identity: dict[str, str],
    source_uri: str,
    fallback_id: str,
    mime_type: str,
) -> InputFile | None:
    """Reuse a prior download only when version-bearing metadata still matches."""

    if not (remote_identity.get("etag") or remote_identity.get("last_modified")):
        return None
    metadata_path = _download_metadata_path(local_path)
    if not local_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict) or metadata.get("remote_identity") != remote_identity:
        return None
    checksum = str(metadata.get("checksum") or "")
    size_bytes = local_path.stat().st_size
    if not checksum or str(size_bytes) != remote_identity.get("size", str(size_bytes)):
        return None
    return InputFile(
        id=str(metadata.get("id") or fallback_id),
        path=local_path,
        source_uri=source_uri,
        checksum=checksum,
        mime_type=mime_type,
        size_bytes=size_bytes,
    )


def write_download_metadata(
    local_path: Path,
    remote_identity: dict[str, str],
    file: InputFile,
) -> None:
    """Publish the sidecar that lets a later run reuse this exact download."""

    metadata_path = _download_metadata_path(local_path)
    temporary = metadata_path.with_name(f".{metadata_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {
                    "remote_identity": remote_identity,
                    "id": file.id,
                    "checksum": file.checksum,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(metadata_path)
    finally:
        temporary.unlink(missing_ok=True)


def claimed_download_root(download_path: Path, claim_id: str) -> Path:
    """Retain the shared root for the first source and isolate later ones.

    The marker must never become visible before its claim is complete. Creating
    the destination as a hard link to a fully-written temporary file gives all
    threads and processes one atomic winner without exposing partial contents.
    """

    download_path.mkdir(parents=True, exist_ok=True)
    marker = download_path / _CLAIM_MARKER_NAME
    temporary = download_path / f".{marker.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(claim_id, encoding="utf-8")
        _publish_claim_marker(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)
    claimed_source = _read_claim_marker(marker)
    return download_path if claimed_source == claim_id else download_path / claim_id


def _download_metadata_path(local_path: Path) -> Path:
    return local_path.with_suffix(local_path.suffix + _DOWNLOAD_METADATA_SUFFIX)


def _read_claim_marker(marker: Path) -> str:
    """Reconcile a shared-volume marker after transient metadata I/O failures."""

    for attempt in range(_CLAIM_MARKER_READ_ATTEMPTS):
        try:
            return marker.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""
        except OSError:
            if attempt == _CLAIM_MARKER_READ_ATTEMPTS - 1:
                return ""
            sleep(0.01)
    return ""


def _publish_claim_marker(temporary: Path, marker: Path) -> None:
    """Retry transient hard-link failures while tolerating unsupported filesystems."""

    for attempt in range(_CLAIM_MARKER_READ_ATTEMPTS):
        try:
            os.link(temporary, marker)
            return
        except FileExistsError:
            return
        except OSError:
            if marker.exists() or attempt == _CLAIM_MARKER_READ_ATTEMPTS - 1:
                return
            sleep(0.01)
