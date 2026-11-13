#!/usr/bin/env python3
"""Build the atopic dermatitis corpus from openly licensed Europe PMC articles.

Every article is CC BY or CC0, so it may be adapted and redistributed with
attribution. Each one is fetched as JATS XML and converted to Markdown: the
title, abstract, body sections and tables are kept; the reference list, author
list, acknowledgements, funding and competing-interest statements are left out,
because they add citation noise without adding treatment facts. ``LICENSE.md``
credits every article and states these changes.

    python data/atopic_dermatitis/build.py            # rebuild files/ from papers.json
    python data/atopic_dermatitis/build.py --select   # choose papers.json afresh

``--select`` queries Europe PMC for the topic, keeps CC BY and CC0 articles
with a usable length, and ranks them by citations with a cap per journal. The
committed ``papers.json`` fixes the selection, so a rebuild without it
reproduces the same corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FILES = ROOT / "files"
PAPERS = ROOT / "papers.json"
SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULL_TEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
USER_AGENT = "FlakeGraph dataset builder/0.1 (+https://github.com/flakegraph/flakegraph)"
OPEN_LICENSES = {"cc by", "cc0"}
TREATMENTS = [
    "dupilumab",
    "tralokinumab",
    "lebrikizumab",
    "nemolizumab",
    "abrocitinib",
    "upadacitinib",
    "baricitinib",
    "delgocitinib",
    "ruxolitinib",
    "tapinarof",
    "crisaborole",
    "roflumilast",
    "JAK inhibitor",
    "biologic",
]
QUERY = (
    '(TITLE:"atopic dermatitis" OR TITLE:"atopic eczema") AND ('
    + " OR ".join(f'ABSTRACT:"{name}"' for name in TREATMENTS)
    + ') AND OPEN_ACCESS:y AND IN_EPMC:y AND (LICENSE:"cc by" OR LICENSE:"cc0")'
    " AND PUB_YEAR:[2019 TO 2026] AND NOT SRC:PPR"
)
EXCLUDED = re.compile(r"case report|letter|erratum|correction|protocol|commentary|editorial", re.I)
PAPER_COUNT = 50
PER_JOURNAL = 4
MIN_WORDS = 1_500
MAX_WORDS = 7_000
SKIPPED_SECTIONS = re.compile(
    r"acknowledg|funding|conflicts? of interest|competing interest|disclosure|author contribution"
    r"|data availability|ethics|supplementary|supporting information|abbreviations|references"
    r"|associated data|keywords",
    re.I,
)
CITATION = "\x00"
# A run of citation markers with the separators between them, and the
# brackets or parentheses around the run when nothing else is inside.
CITATION_GROUP = re.compile(
    r"[\[(]\s*(?:\x00[\s,;\u2013\u2014-]*)+[\])]|(?:\x00[\s,;\u2013\u2014-]*)+"
)


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        body: bytes = response.read()
    return body


def _text(element: ET.Element | None) -> str:
    """Flatten an element's text, leaving citation markers out."""

    if element is None:
        return ""
    parts: list[str] = [element.text or ""]
    for child in element:
        parts.append(
            CITATION if child.tag == "xref" and child.get("ref-type") == "bibr" else _text(child)
        )
        parts.append(child.tail or "")
    text = CITATION_GROUP.sub(" ", "".join(parts))
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s+([.,;:)])", r"\1", text).strip()


def _table(wrap: ET.Element) -> list[str]:
    lines = []
    caption = _text(wrap.find("caption"))
    label = _text(wrap.find("label"))
    if label or caption:
        lines.append(f"**{' '.join(part for part in (label, caption) if part)}**")
        lines.append("")
    rows = [
        [_text(cell).replace("|", "/") for cell in row if cell.tag in ("td", "th")]
        for row in wrap.iter("tr")
    ]
    rows = [row for row in rows if any(row)]
    if rows:
        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        lines.append("| " + " | ".join(rows[0]) + " |")
        lines.append("|" + " --- |" * width)
        lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return lines


def _section(section: ET.Element, depth: int) -> list[str]:
    title = _text(section.find("title"))
    if title and SKIPPED_SECTIONS.search(title):
        return []
    return ([f"{'#' * min(depth, 6)} {title}", ""] if title else []) + _children(section, depth)


def _children(section: ET.Element, depth: int) -> list[str]:
    lines: list[str] = []
    for child in section:
        if child.tag == "p":
            paragraph = _text(child)
            if paragraph:
                lines += [paragraph, ""]
        elif child.tag == "sec":
            lines += _section(child, depth + 1)
        elif child.tag == "list":
            lines += [f"- {_text(item)}" for item in child.findall("list-item")] + [""]
        elif child.tag == "table-wrap":
            lines += _table(child) + [""]
        elif child.tag == "fig":
            caption = " ".join(
                part for part in (_text(child.find("label")), _text(child.find("caption"))) if part
            )
            if caption:
                lines += [f"*{caption}*", ""]
    return lines


def to_markdown(xml: bytes, paper: dict[str, str]) -> str:
    """One article as Markdown: a citation header, the abstract and the body."""

    root = ET.fromstring(xml)
    meta = root.find(".//article-meta")
    title = _text(meta.find(".//article-title")) if meta is not None else paper["title"]
    lines = [
        f"# {title}",
        "",
        f"*{paper['journal']}, {paper['year']}. doi:{paper['doi']}. {paper['pmcid']}. "
        f"Licensed {paper['license'].upper()}; "
        "converted to Markdown for FlakeGraph, see LICENSE.md.*",
        "",
    ]
    for abstract in root.findall(".//article-meta/abstract"):
        if abstract.get("abstract-type") in {"graphical", "teaser"}:
            continue
        lines += ["## Abstract", ""] + _children(abstract, 3)
    body = root.find(".//body")
    if body is not None:
        for child in body:
            if child.tag == "sec":
                lines += _section(child, 2)
            elif child.tag == "p":
                lines += [_text(child), ""]
            elif child.tag == "table-wrap":
                lines += _table(child) + [""]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def _slug(title: str) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", title.lower()))[:70].rstrip("-")


def select() -> list[dict[str, str]]:
    """Choose the corpus afresh from Europe PMC."""

    found: list[dict[str, Any]] = []
    cursor = "*"
    while True:
        query = urllib.parse.urlencode(
            {
                "query": QUERY,
                "format": "json",
                "pageSize": 1000,
                "resultType": "core",
                "cursorMark": cursor,
            }
        )
        page = json.loads(_get(f"{SEARCH_URL}?{query}"))
        found += page["resultList"]["result"]
        if not page["resultList"]["result"] or page.get("nextCursorMark") in (None, cursor):
            break
        cursor = page["nextCursorMark"]
    candidates = [
        item
        for item in found
        if item.get("pmcid")
        and item.get("doi")
        and str(item.get("license", "")).lower() in OPEN_LICENSES
        and not EXCLUDED.search(item["title"])
        and not any(
            EXCLUDED.search(kind) for kind in item.get("pubTypeList", {}).get("pubType", [])
        )
    ]
    candidates.sort(key=lambda item: -int(item.get("citedByCount", 0)))
    chosen: list[dict[str, str]] = []
    journals: Counter[str] = Counter()
    for item in candidates:
        if len(chosen) == PAPER_COUNT:
            break
        journal = item["journalInfo"]["journal"]["title"]
        if journals[journal] >= PER_JOURNAL:
            continue
        paper = {
            "pmcid": item["pmcid"],
            "doi": item["doi"],
            "title": re.sub(r"<[^>]+>", "", item["title"]).rstrip("."),
            "authors": item.get("authorString", ""),
            "journal": journal,
            "year": str(item["pubYear"]),
            "license": item["license"].lower(),
            "file": f"files/{_slug(item['title'])}.md",
        }
        try:
            words = len(
                to_markdown(_get(FULL_TEXT_URL.format(pmcid=paper["pmcid"])), paper).split()
            )
        except (OSError, ET.ParseError) as error:
            print(f"skip {paper['pmcid']}: {error}", file=sys.stderr)
            continue
        if MIN_WORDS <= words <= MAX_WORDS:
            chosen.append(paper)
            journals[journal] += 1
    return chosen


def build(papers: list[dict[str, str]]) -> None:
    """Write files/, manifest.jsonl and LICENSE.md from the selection."""

    FILES.mkdir(exist_ok=True)

    def convert(paper: dict[str, str]) -> None:
        (ROOT / paper["file"]).write_text(
            to_markdown(_get(FULL_TEXT_URL.format(pmcid=paper["pmcid"])), paper), encoding="utf-8"
        )

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(convert, papers))
    with (ROOT / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for paper in papers:
            data = (ROOT / paper["file"]).read_bytes()
            record = {
                "path": paper["file"],
                "source_uri": f"https://europepmc.org/article/PMC/{paper['pmcid']}",
                "checksum": hashlib.sha256(data).hexdigest(),
                "mime_type": "text/markdown",
                "size_bytes": len(data),
            }
            manifest.write(json.dumps(record, separators=(",", ":")) + "\n")
    credits = "\n".join(
        f"- `{paper['file']}`: {paper['authors'].rstrip('.')}. “{paper['title']}.” "
        f"*{paper['journal']}*, {paper['year']}. https://doi.org/{paper['doi']}. "
        f"{'CC BY 4.0' if paper['license'] == 'cc by' else 'CC0 1.0'}."
        for paper in papers
    )
    (ROOT / "LICENSE.md").write_text(LICENSE_TEXT.format(credits=credits), encoding="utf-8")


LICENSE_TEXT = """# Atopic Dermatitis Dataset License

## Source articles

Each document under `files/` is adapted from an open-access article published
under the Creative Commons Attribution license (CC BY) or dedicated under CC0,
as listed below. The articles remain the work of their authors and are
redistributed under those licenses:
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

**Changes made.** Each article was converted from its Europe PMC JATS XML to
Markdown by `build.py`. The title, abstract, body sections, figure captions and
tables are kept; the author list, reference list, citation markers,
acknowledgements, funding, competing-interest, ethics, data-availability and
supplementary sections are removed. No sentence was otherwise edited. The
adaptation does not imply endorsement by the authors or publishers.

{credits}

## Benchmark annotations

`gold.json`, `ontology.yaml`, `papers.json`, `build.py` and `README.md` are
original FlakeGraph work, dedicated under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en).
"""


def main() -> None:
    """Select the papers when asked or when none are chosen, then build the corpus."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--select", action="store_true", help="Choose papers.json afresh from Europe PMC."
    )
    arguments = parser.parse_args()
    if arguments.select or not PAPERS.exists():
        papers = select()
        PAPERS.write_text(json.dumps(papers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        papers = json.loads(PAPERS.read_text(encoding="utf-8"))
    build(papers)
    print(f"{len(papers)} articles in {FILES}")


if __name__ == "__main__":
    main()
