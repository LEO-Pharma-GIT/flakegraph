"""Focused contracts for the non-redistributed paper-corpus downloader."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path("data/deep_learning_papers/download.py")


def _module() -> ModuleType:
    """Import the standalone dataset script without making data a Python package."""

    spec = importlib.util.spec_from_file_location("deep_learning_papers_download", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_selects_only_primary_links_and_repairs_bottou_row() -> None:
    """Keep note citations out of the corpus and repair the index's duplicate URL."""

    module = _module()
    html = """
    <table class="table reading"><tbody>
      <tr><td>1991</td><td><a href="https://example.test/Krogh-1991.pdf">
      Léon Bottou, <em>Stochastic Gradient Learning in Neural Networks</em></a></td>
      <td><a href="https://example.test/note.pdf">note</a></td></tr>
      <tr><td>2013</td><td><a href="https://example.test/Le-2012.pdf">
      Distributed Representations of Words and Phrases and their Compositionality</a></td>
      <td>notes</td></tr>
    </tbody></table>
    """

    sources = module.parse_index(html)

    assert [source.filename for source in sources] == ["Bottou-1991.pdf", "Mikolov-2013.pdf"]
    assert sources[0].year == 1991
    assert sources[0].title == "Léon Bottou, Stochastic Gradient Learning in Neural Networks"


def test_parser_rejects_duplicate_or_non_pdf_primary_sources() -> None:
    """Fail before downloading when the remote catalog cannot map to unique PDFs."""

    module = _module()
    duplicate = """
    <table class="reading"><tbody>
      <tr><td>2017</td><td><a href="https://x/paper.pdf">First</a></td></tr>
      <tr><td>2018</td><td><a href="https://x/paper.pdf">Second</a></td></tr>
    </tbody></table>
    """
    invalid = """
    <table class="reading"><tbody>
      <tr><td>2017</td><td><a href="https://x/paper.html">Paper</a></td></tr>
    </tbody></table>
    """

    with pytest.raises(ValueError, match="duplicate filenames"):
        module.parse_index(duplicate)
    with pytest.raises(ValueError, match="PDF filename"):
        module.parse_index(invalid)


def test_gold_checksum_validation_detects_replaced_or_missing_papers(tmp_path: Path) -> None:
    """Bind downloaded bytes to the reviewed benchmark version without network calls."""

    module = _module()
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "documents": [
                    {"path": "data/deep_learning_papers/files/paper.pdf", "checksum": "abc"}
                ]
            }
        ),
        encoding="utf-8",
    )

    module._validate_gold_checksums(
        [{"path": str(tmp_path / "paper.pdf"), "checksum": "abc"}],
        gold,
    )

    with pytest.raises(ValueError, match="checksums differ"):
        module._validate_gold_checksums(
            [{"path": str(tmp_path / "paper.pdf"), "checksum": "changed"}],
            gold,
        )
    with pytest.raises(ValueError, match="catalog does not match"):
        module._validate_gold_checksums([], gold)


def test_versioned_gold_catalog_is_the_default_download_source(tmp_path: Path) -> None:
    """Reconstruct benchmark URLs without depending on the mutable HTML index."""

    module = _module()
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "path": "data/deep_learning_papers/files/paper.pdf",
                        "checksum": "abc",
                        "source_uri": "https://papers.example/paper.pdf",
                        "title": "Example Paper",
                        "year": 2024,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    sources = module._sources_from_gold(gold)

    assert [(source.filename, source.title, source.year) for source in sources] == [
        ("paper.pdf", "Example Paper", 2024)
    ]


def test_manifest_paths_are_relative_to_the_manifest_directory(tmp_path: Path) -> None:
    """Keep generated metadata portable and free of checkout-specific directories."""

    module = _module()
    manifest_directory = tmp_path / "dataset"
    record = {
        "path": str(manifest_directory / "files" / "paper.pdf"),
        "checksum": "abc",
    }

    portable = module._portable_manifest_record(record, manifest_directory)

    assert portable == {"path": "files/paper.pdf", "checksum": "abc"}
    assert record["path"] == str(manifest_directory / "files" / "paper.pdf")
