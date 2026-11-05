from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from kg_processor.adapters.files.common import matches_include_globs
from kg_processor.adapters.files.manifest import ManifestFileSource
from kg_processor.domain.ids import sha256_hex


def test_directory_globs_use_anchored_recursive_matching() -> None:
    assert matches_include_globs("corpus/a/b/x.pdf", ["corpus/**/*.pdf"])
    assert not matches_include_globs("archive/incoming/r.pdf", ["incoming/*.pdf"])
    assert matches_include_globs("archive/incoming/r.pdf", ["**/incoming/*.pdf"])


def test_manifest_file_source_reads_jsonl_and_verifies_checksums(tmp_path: Path) -> None:
    document = tmp_path / "sample.txt"
    document.write_text("Alice in Copenhagen", encoding="utf-8")
    checksum = sha256_hex(document.read_bytes())
    size_bytes = document.stat().st_size
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "path": "sample.txt",
                "source_uri": "azblob://documents/sample.txt",
                "file_id": "explicit-file-id",
                "checksum": checksum,
                "mime_type": "text/custom",
                "size_bytes": size_bytes,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    files = ManifestFileSource(manifest).list_files()

    assert len(files) == 1
    assert files[0].id == "explicit-file-id"
    assert files[0].path == document.resolve()
    assert files[0].source_uri == "azblob://documents/sample.txt"
    assert files[0].checksum == checksum
    assert files[0].mime_type == "text/custom"
    assert files[0].size_bytes == size_bytes


def test_manifest_file_source_reads_utf8_sig_jsonl(tmp_path: Path) -> None:
    document = tmp_path / "sample.txt"
    document.write_text("Alice in Copenhagen", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('\ufeff{"path": "sample.txt"}\n', encoding="utf-8")

    files = ManifestFileSource(manifest).list_files()

    assert [file.path for file in files] == [document.resolve()]


def test_manifest_file_source_reads_csv_and_filters_include_globs(tmp_path: Path) -> None:
    keep = tmp_path / "docs" / "keep.pdf"
    drop = tmp_path / "docs" / "drop.txt"
    keep.parent.mkdir()
    keep.write_bytes(b"%PDF sample")
    drop.write_text("drop", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "source_uri"])
        writer.writeheader()
        writer.writerow({"path": "docs/keep.pdf", "source_uri": "file://keep.pdf"})
        writer.writerow({"path": "docs/drop.txt", "source_uri": "file://drop.txt"})

    files = ManifestFileSource(manifest, include_globs=["**/*.pdf"]).list_files()

    assert [file.path for file in files] == [keep.resolve()]
    assert files[0].mime_type == "application/pdf"


def test_manifest_file_source_rejects_selected_unsupported_suffix(tmp_path: Path) -> None:
    document = tmp_path / "sample.bin"
    document.write_bytes(b"not supported")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "sample.bin"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported suffix"):
        ManifestFileSource(manifest).list_files()


def test_manifest_file_source_rejects_checksum_mismatch(tmp_path: Path) -> None:
    document = tmp_path / "sample.txt"
    document.write_text("Alice", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "sample.txt", "checksum": "wrong"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        ManifestFileSource(manifest).list_files()


def test_manifest_file_source_rejects_size_mismatch(tmp_path: Path) -> None:
    document = tmp_path / "sample.txt"
    document.write_text("Alice", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "sample.txt", "size_bytes": document.stat().st_size + 1}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="size mismatch"):
        ManifestFileSource(manifest).list_files()


def test_manifest_file_source_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Manifest file does not exist"):
        ManifestFileSource(tmp_path / "missing.jsonl").list_files()


def test_manifest_containment_root_rejects_rows_that_escape_it(tmp_path: Path) -> None:
    """A user-supplied manifest must not be able to name files outside its corpus.

    Absolute paths, ``..`` segments and symlinks planted inside the root are the
    three ways out, and all three have to be caught after resolution.
    """

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "inside.txt").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (corpus / "link.txt").symlink_to(outside)
    manifest = corpus / "manifest.jsonl"

    manifest.write_text(json.dumps({"path": "inside.txt"}) + "\n", encoding="utf-8")
    files = ManifestFileSource(manifest, containment_root=corpus).list_files()
    assert [file.path.name for file in files] == ["inside.txt"]

    for escape in (str(outside), "../outside.txt", "link.txt"):
        manifest.write_text(json.dumps({"path": escape}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="outside the allowed root"):
            ManifestFileSource(manifest, containment_root=corpus).list_files()


def test_manifest_without_a_containment_root_keeps_accepting_absolute_paths(
    tmp_path: Path,
) -> None:
    """Operator-written manifests may point anywhere on the host."""

    document = tmp_path / "elsewhere" / "doc.txt"
    document.parent.mkdir()
    document.write_text("text", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"path": str(document)}) + "\n", encoding="utf-8")

    files = ManifestFileSource(manifest).list_files()

    assert [file.path for file in files] == [document]
