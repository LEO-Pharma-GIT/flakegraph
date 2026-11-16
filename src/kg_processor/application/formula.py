# SPDX-License-Identifier: Apache-2.0
"""Recognise text that is mathematics rather than prose."""

from __future__ import annotations

import re

# LaTeX commands, grouping and sub/superscripts, and display delimiters.
_MATH_MARKER_RE = re.compile(r"\\[A-Za-z]+|[{}^_]|\$\$?")
# A word of at least three letters that is neither a LaTeX command nor a
# run of single-letter symbols the recogniser spaces out (``n e t``).
_PROSE_WORD_RE = re.compile(r"(?<![\\\w])[A-Za-z][a-z]{2,}(?![\w{}^_])")
_MIN_MATH_MARKERS = 20
_MARKERS_PER_PROSE_WORD = 8


def is_display_math_text(content: str) -> bool:
    """Return whether text is equations with at most incidental words.

    A formula recogniser renders an appendix of derivations as pages of LaTeX,
    and a window cut from it holds no entity and no quotable evidence: what a
    model returns for it cannot be grounded. A derivation keeps a sentence or
    two of connective prose ("where", "which are stored"), so the test is the
    ratio of markers to words, not the absence of words; ordinary prose with an
    inline formula keeps far more words than markers and is never matched.
    """

    markers = len(_MATH_MARKER_RE.findall(content))
    if markers < _MIN_MATH_MARKERS:
        return False
    words = len(_PROSE_WORD_RE.findall(content))
    return markers >= _MARKERS_PER_PROSE_WORD * words
