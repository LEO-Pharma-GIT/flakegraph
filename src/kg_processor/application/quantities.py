# SPDX-License-Identifier: Apache-2.0
"""Recognize and parse values stated in text: limits, ranges, and results."""

from __future__ import annotations

import re

from kg_processor.domain.graph import Quantity

_UPPER_BOUND_COMPARATORS = {"<", "<=", "≤", "nmt", "not more than", "max", "maximum", "up to"}
_LOWER_BOUND_COMPARATORS = {">", ">=", "≥", "nlt", "not less than", "min", "minimum", "at least"}
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")
_COMPARATOR_RE = re.compile(
    r"(<=|>=|≤|≥|<|>|\bnmt\b|\bnlt\b|\bnot more than\b|\bnot less than\b|\bmax(?:imum)?\b"
    r"|\bmin(?:imum)?\b|\bup to\b|\bat least\b)",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    rf"({_NUMBER_RE.pattern})\s*(?:-|–|—|to|\.\.)\s*({_NUMBER_RE.pattern})",
    re.IGNORECASE,
)
_UNIT_SYMBOL_RE = re.compile(r"[%‰°µμ]")
# A letter written straight before a digit: a model, part or project code.
_CODE_RE = re.compile(r"[^\W\d_]\d")
# What a value can be written with besides its number: units, comparators,
# range and list words. A name with nothing else is a value, not a thing.
_VALUE_WORD_RE = re.compile(
    r"\b(?:[kmµμnpcd]?(?:g|l|m|mol|s|pa|w|j|v|a|hz)|c|f|k|ppm|ppb|ppt|cfu|iu|u|meq|eq"
    r"|cps|cp|mpas|pas|rpm|rh|ph|bar|psi|min|h|hr|hrs|d|days?|weeks?|months?|years?|x"
    r"|max|min|nmt|nlt|not|more|less|than|to|and|or|up|at|least|approx|ca|about)\b",
    re.IGNORECASE,
)
# What a value can open with before its first number: a comparator, an
# approximation, or the measure or time point it is written after.
_VALUE_LEAD_RE = re.compile(
    r"\b(?:max(?:imum)?|min(?:imum)?|nmt|nlt|not|more|less|than|up|to|at|least|approx"
    r"|about|ph|days?|weeks?|months?|years?|hours?|hrs?)\b|\bca(?=[.\s])",
    re.IGNORECASE,
)


def is_value_text(text: str) -> bool:
    """Whether text is only a stated value: numbers with units or comparators.

    "≤ 0.10 %", "154 - 162", "40 mg/kg max" and "0004417720" are values; "Span 60",
    "Ph. Eur. 2.5.12" and "USP <711>" name things that merely contain numbers. A
    unit follows its number rather than leading it, so letters before the first
    number make a name - "S450", "Q7-2219", "PS 80", "Dv (90)", "D [4:3]" - unless
    they are a comparator or the measure a value is written after ("NMT 5.0",
    "pH 5.5", "day 7").
    """

    first = _NUMBER_RE.search(text)
    if first is None or _CODE_RE.search(text):
        return False
    if re.search(r"[^\W\d_]", _VALUE_LEAD_RE.sub(" ", text[: first.start()])):
        return False
    remainder = _VALUE_WORD_RE.sub(" ", _UNIT_SYMBOL_RE.sub(" ", _NUMBER_RE.sub(" ", text)))
    return not re.search(r"[^\W\d_]", remainder)


def parse_quantity(
    text: str,
    *,
    kind: str | None = None,
    unit: str | None = None,
    comparator: str | None = None,
) -> Quantity:
    """Parse the bounds of a stated value while keeping its text as written.

    A range gives both bounds; "≤ 0.5" gives an upper bound; a single number gives
    both. Decimal commas are read as decimal points.
    """

    numbers = [float(match.replace(",", ".")) for match in _NUMBER_RE.findall(text)]
    # A model's comparator counts only when it is one; otherwise read the text.
    given = comparator.strip() if comparator else ""
    found = _COMPARATOR_RE.fullmatch(given) or _COMPARATOR_RE.search(text)
    symbol = found.group(1).lower() if found else None
    low: float | None = None
    high: float | None = None
    interval = _RANGE_RE.search(text)
    if interval:
        low, high = sorted(float(bound.replace(",", ".")) for bound in interval.groups())
    elif numbers:
        if symbol and symbol.strip() in _UPPER_BOUND_COMPARATORS:
            high = numbers[0]
        elif symbol and symbol.strip() in _LOWER_BOUND_COMPARATORS:
            low = numbers[0]
        else:
            low = high = numbers[0]
    return Quantity(
        text=text,
        kind=kind or None,
        comparator=symbol,
        unit=unit or None,
        low=low,
        high=high,
    )
