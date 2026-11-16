#!/usr/bin/env python3
"""Download the public index's primary deep-learning papers for local benchmarking.

The papers remain under their publishers' and authors' respective terms. This
script stores them only in the ignored ``files/`` directory and emits an ignored
FlakeGraph manifest containing source URLs, checksums, sizes, and media types.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

INDEX_URL = "https://papers.baulab.info/"
EXPECTED_PAPER_COUNT = 49
USER_AGENT = "FlakeGraph dataset downloader/0.1 (+https://github.com/flakegraph/flakegraph)"
_PDF_SIGNATURE = b"%PDF-"
_YEAR_CELL = 1
_CITATION_CELL = 2
_SOURCE_OVERRIDES = {
    "Stochastic Gradient Learning in Neural Networks": (
        "https://papers.baulab.info/papers/Bottou-1991.pdf"
    ),
    "Distributed Representations of Words and Phrases and their Compositionality": (
        "https://papers.baulab.info/papers/Mikolov-2013.pdf"
    ),
}


@dataclass(frozen=True)
class PaperSource:
    """Describe one primary paper row discovered in the curated HTML table."""

    year: int
    title: str
    url: str

    @property
    def filename(self) -> str:
        """Return the URL's safe leaf filename after rejecting ambiguous paths."""

        name = Path(urlparse(self.url).path).name
        if not name or name in {".", ".."} or not name.lower().endswith(".pdf"):
            raise ValueError(f"paper URL does not identify a PDF filename: {self.url}")
        return name


class _PaperIndexParser(HTMLParser):
    """Extract the first link from the citation cell of each primary table row."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sources: list[PaperSource] = []
        self._in_primary_table = False
        self._in_row = False
        self._cell_index = 0
        self._capture_cell = False
        self._capture_anchor = False
        self._year_text: list[str] = []
        self._title_text: list[str] = []
        self._url: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track primary-table rows while ignoring links embedded in explanatory notes."""

        attributes = dict(attrs)
        if tag == "table" and "reading" in (attributes.get("class") or "").split():
            self._in_primary_table = True
        elif self._in_primary_table and tag == "tr":
            self._in_row = True
            self._cell_index = 0
            self._year_text = []
            self._title_text = []
            self._url = None
        elif self._in_row and tag == "td":
            self._cell_index += 1
            self._capture_cell = self._cell_index in {_YEAR_CELL, _CITATION_CELL}
        elif (
            self._in_row and self._cell_index == _CITATION_CELL and tag == "a" and self._url is None
        ):
            self._url = attributes.get("href")
            self._capture_anchor = True

    def handle_data(self, data: str) -> None:
        """Collect only the year cell and text inside the row's primary citation link."""

        if self._in_row and self._capture_cell and self._cell_index == _YEAR_CELL:
            self._year_text.append(data)
        if self._capture_anchor:
            self._title_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        """Finalize rows into validated source records when their closing tag arrives."""

        if tag == "a":
            self._capture_anchor = False
        elif tag == "td":
            self._capture_cell = False
        elif tag == "tr" and self._in_row:
            self._finish_row()
            self._in_row = False
        elif tag == "table" and self._in_primary_table:
            self._in_primary_table = False

    def _finish_row(self) -> None:
        """Normalize one complete row and apply the index's known duplicate-link repair."""

        if self._url is None:
            return
        year_text = " ".join("".join(self._year_text).split())
        title = " ".join("".join(self._title_text).split())
        if not year_text.isdigit() or not title:
            raise ValueError(f"invalid paper index row: year={year_text!r}, title={title!r}")
        url = next(
            (
                replacement
                for suffix, replacement in _SOURCE_OVERRIDES.items()
                if title.endswith(suffix)
            ),
            self._url,
        )
        self.sources.append(PaperSource(year=int(year_text), title=title, url=url))


def parse_index(html: str) -> list[PaperSource]:
    """Parse and validate the ordered primary-paper catalog from the index HTML."""

    parser = _PaperIndexParser()
    parser.feed(html)
    sources = parser.sources
    if not sources:
        raise ValueError("paper index did not contain any primary paper rows")
    filenames = [source.filename for source in sources]
    duplicates = sorted({name for name in filenames if filenames.count(name) > 1})
    if duplicates:
        raise ValueError(f"paper index resolves to duplicate filenames: {duplicates}")
    return sources


def download_dataset(
    *,
    index_url: str,
    output_dir: Path,
    manifest_path: Path,
    workers: int,
    refresh: bool,
    timeout_seconds: float,
    discover_index: bool = False,
) -> list[dict[str, object]]:
    """Discover, download, validate, and manifest the complete indexed corpus.

    Existing valid PDFs are reused unless ``refresh`` is true. Downloads use a
    sibling temporary file followed by ``os.replace`` so an interrupted request
    can never leave a partial file under its final name.
    """

    if workers < 1:
        raise ValueError("workers must be at least 1")
    gold_path = output_dir.parent / "gold.json"
    if discover_index:
        request = urllib.request.Request(index_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            sources = parse_index(response.read().decode("utf-8"))
    else:
        sources = _sources_from_gold(gold_path)
    if len(sources) != EXPECTED_PAPER_COUNT:
        raise ValueError(
            f"paper index contains {len(sources)} primary papers; expected "
            f"{EXPECTED_PAPER_COUNT} for this benchmark version"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_one,
                source,
                output_dir,
                refresh,
                timeout_seconds,
            ): source
            for source in sources
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                record = future.result()
            except Exception as error:
                message = f"failed to download {source.title} from {source.url}"
                raise RuntimeError(message) from error
            records.append(record)
            print(f"ready  {record['path']}", file=sys.stderr)
    records.sort(key=lambda record: str(record["path"]))
    _validate_gold_checksums(records, gold_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_records = [
        _portable_manifest_record(record, manifest_path.parent) for record in records
    ]
    manifest_text = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in manifest_records
    )
    _atomic_write(manifest_path, manifest_text.encode("utf-8"))
    return records


def _portable_manifest_record(
    record: dict[str, object],
    manifest_directory: Path,
) -> dict[str, object]:
    """Return a manifest row whose file path is portable and non-identifying.

    Download records retain the concrete destination for validation and console
    feedback. Persisted manifests instead resolve files relative to their own
    directory, preventing a generated benchmark artifact from exposing a user's
    home directory or becoming tied to one checkout location.
    """

    path = Path(str(record["path"]))
    relative_path = os.path.relpath(path, start=manifest_directory)
    return {**record, "path": Path(relative_path).as_posix()}


def _sources_from_gold(gold_path: Path) -> list[PaperSource]:
    """Load the immutable reviewed source catalog embedded in the gold fixture."""

    if not gold_path.is_file():
        raise FileNotFoundError(
            f"versioned source catalog is missing: {gold_path}; "
            "use --discover-index only when deliberately updating the benchmark"
        )
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    sources: list[PaperSource] = []
    for document in gold.get("documents", []):
        url = document.get("source_uri")
        title = document.get("title")
        year = document.get("year")
        if not isinstance(url, str) or not isinstance(title, str) or not isinstance(year, int):
            raise ValueError("gold source catalog requires source_uri, title, and year")
        sources.append(PaperSource(year=year, title=title, url=url))
    filenames = [source.filename for source in sources]
    if len(filenames) != len(set(filenames)):
        raise ValueError("gold source catalog contains duplicate filenames")
    return sources


def _validate_gold_checksums(records: list[dict[str, object]], gold_path: Path) -> None:
    """Reject source drift against the versioned gold document checksums.

    The public index identifies locations rather than immutable content. Binding
    downloaded bytes to the reviewed gold fixture ensures benchmark results are
    comparable even when a publisher later replaces a PDF at the same URL.
    """

    if not gold_path.is_file():
        return
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    expected = {
        Path(document["path"]).name: document["checksum"] for document in gold.get("documents", [])
    }
    actual = {Path(str(record["path"])).name: str(record["checksum"]) for record in records}
    if set(actual) != set(expected):
        raise ValueError(
            "downloaded paper catalog does not match gold.json: "
            f"{sorted(set(actual) ^ set(expected))}"
        )
    changed = sorted(name for name, checksum in actual.items() if checksum != expected[name])
    if changed:
        raise ValueError(f"downloaded paper checksums differ from gold.json: {changed}")


def _download_one(
    source: PaperSource,
    output_dir: Path,
    refresh: bool,
    timeout_seconds: float,
) -> dict[str, object]:
    """Materialize one source and return its FlakeGraph manifest record."""

    destination = output_dir / source.filename
    if refresh or not _is_pdf(destination):
        request = urllib.request.Request(source.url, headers={"User-Agent": USER_AGENT})
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                content_type = response.headers.get_content_type()
                with temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            if content_type != "application/pdf" and not _is_pdf(temporary):
                raise ValueError(f"unexpected content type for {source.url}: {content_type}")
            if not _is_pdf(temporary):
                raise ValueError(f"download is not a PDF: {source.url}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    content = destination.read_bytes()
    return {
        "checksum": hashlib.sha256(content).hexdigest(),
        "mime_type": "application/pdf",
        "path": destination.as_posix(),
        "size_bytes": len(content),
        "source_uri": source.url,
        "title": source.title,
        "year": source.year,
    }


def _is_pdf(path: Path) -> bool:
    """Return whether an existing regular file begins with the PDF signature."""

    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(len(_PDF_SIGNATURE)) == _PDF_SIGNATURE


def _atomic_write(path: Path, content: bytes) -> None:
    """Replace a generated text artifact atomically after fully writing it."""

    temporary = path.with_suffix(path.suffix + ".part")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    """Build the command-line contract for reproducible local corpus materialization."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", default=INDEX_URL)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "files")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "manifest.jsonl")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--discover-index",
        action="store_true",
        help="discover the mutable live index when deliberately reviewing a dataset update",
    )
    return parser


def main() -> int:
    """Run the downloader and print a concise corpus summary for operators."""

    args = _parser().parse_args()
    records = download_dataset(
        index_url=args.index_url,
        output_dir=args.output,
        manifest_path=args.manifest,
        workers=args.workers,
        refresh=args.refresh,
        timeout_seconds=args.timeout_seconds,
        discover_index=args.discover_index,
    )
    total_bytes = sum(cast(int, record["size_bytes"]) for record in records)
    print(f"Downloaded {len(records)} papers ({total_bytes / 1024 / 1024:.1f} MiB).")
    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
