from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kg_processor.adapters.files.local import LocalFileSource


def test_local_file_source_hashes_supported_files(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice in Copenhagen", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"binary")

    files = LocalFileSource(tmp_path).list_files()

    assert len(files) == 1
    assert files[0].path == sample
    assert files[0].checksum == hashlib.sha256(b"Alice in Copenhagen").hexdigest()
    assert files[0].mime_type == "text/plain"


def test_local_file_source_ignores_macos_appledouble_sidecars(tmp_path: Path) -> None:
    """Metadata sidecars must not become duplicate document tasks after transfer."""

    (tmp_path / "history.pdf").write_bytes(b"real-pdf")
    (tmp_path / "._history.pdf").write_bytes(b"apple-double-metadata")

    files = LocalFileSource(tmp_path, ["*.pdf"]).list_files()

    assert [item.path.name for item in files] == ["history.pdf"]


def test_local_file_source_browses_without_hashing(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.md").write_text("second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    (tmp_path / "scratch.tmp").write_bytes(b"ignored")
    source = LocalFileSource(tmp_path, ["**/*"])

    listing = list(source.browse(limit=10))

    assert [item.name for item in listing] == ["a.txt", "b.md"]
    assert listing[0].uri == (tmp_path / "a.txt").resolve().as_uri()
    assert listing[0].size_bytes == 5
    assert listing[0].modified_at is not None
    assert listing[0].checksum is None
    assert [item.name for item in source.browse(limit=1)] == ["a.txt"]
    # Browsing and listing agree on the corpus, so a count shown before a run
    # is the count the run will ingest.
    assert [file.path.name for file in source.list_files()] == ["a.txt", "b.md"]


@pytest.mark.parametrize(
    "pattern", ["../private/*.txt", "/tmp/*.txt", r"..\private\*.txt", r"C:\private\*.txt"]
)
def test_local_source_rejects_globs_outside_its_root(tmp_path: Path, pattern: str) -> None:
    with pytest.raises(ValueError, match="relative without parent"):
        LocalFileSource(tmp_path, [pattern])


def test_local_source_rejects_symlinks_outside_its_root(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("Synthetic private document")
    (corpus / "link.txt").symlink_to(outside)
    source = LocalFileSource(corpus)
    with pytest.raises(ValueError, match="outside its source directory"):
        source.list_files()
    with pytest.raises(ValueError, match="outside its source directory"):
        list(source.browse(10))


def test_local_source_deduplicates_overlapping_safe_globs(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Public fixture")
    (tmp_path / "alias.txt").symlink_to(tmp_path / "a.txt")
    assert [
        item.path.name for item in LocalFileSource(tmp_path, ["**/*", "*.txt"]).list_files()
    ] == ["a.txt", "alias.txt"]


def test_local_source_ignores_unsupported_links_outside_its_root(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Public fixture")
    outside = tmp_path / "tool.bin"
    outside.write_bytes(b"\0")
    (corpus / "tool.bin").symlink_to(outside)
    assert [item.path.name for item in LocalFileSource(corpus).list_files()] == ["a.txt"]
