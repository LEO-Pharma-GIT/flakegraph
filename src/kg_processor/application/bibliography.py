# SPDX-License-Identifier: Apache-2.0
"""Conservative bibliography detection shared by local and distributed extraction."""

from __future__ import annotations

import re

_MIN_REFERENCE_ENTRIES = 4
_MIN_REFERENCE_PUBLICATION_MARKERS = 2
REFERENCE_HEADING_RE = re.compile(r"(?im)^\s*(?:references|bibliography)\s*$")
_BRACKETED_REFERENCE_RE = re.compile(r"(?m)^\s*\[\d{1,3}\]\s+\S")
_NUMBERED_REFERENCE_RE = re.compile(
    r"(?m)^\s*\d{1,3}\.\s+"
    r"(?:[A-Z][\w'’-]+(?:,\s*[A-Z]\.|\s+[A-Z]{1,3}(?=[,.;]))|[A-Z]\.)"
)
_AUTHOR_REFERENCE_RE = re.compile(
    r"(?m)^\s*(?:(?:[A-Z]\.\s*){1,3}[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+"
    r"|[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+,\s*(?:[A-Z]\.\s*){1,3})"
)
_REFERENCE_YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}[a-z]?\b", re.IGNORECASE)
_REFERENCE_PUBLICATION_RE = re.compile(
    r"\b(?:arxiv|doi|journal|proceedings|conference|vol\.|pp\.|url)\b",
    re.IGNORECASE,
)


def is_reference_text(content: str) -> bool:
    """Return whether text contains enough independent signals to be a bibliography.

    A heading lowers the entry threshold but never suffices alone. Numbered and
    author-year styles are recognized independently, while publication markers
    keep ordinary prose containing several names and years from being suppressed.
    """

    entry_count = len(_BRACKETED_REFERENCE_RE.findall(content)) + len(
        _NUMBERED_REFERENCE_RE.findall(content)
    )
    minimum = 2 if REFERENCE_HEADING_RE.search(content) else _MIN_REFERENCE_ENTRIES
    if entry_count >= minimum:
        return True
    author_entry_count = len(_AUTHOR_REFERENCE_RE.findall(content))
    year_count = len(_REFERENCE_YEAR_RE.findall(content))
    publication_marker_count = len(_REFERENCE_PUBLICATION_RE.findall(content))
    return (
        author_entry_count >= minimum
        and year_count >= minimum
        and publication_marker_count >= _MIN_REFERENCE_PUBLICATION_MARKERS
    )
