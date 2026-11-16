# SPDX-License-Identifier: Apache-2.0
"""Deterministic predicate grounding for relation evidence.

Endpoint presence alone proves co-occurrence, not a graph predicate. Curated
ontologies can therefore provide regular-expression cues that must occur in the
local span connecting both endpoints. Compact table rows are handled separately
because their column structure often expresses a relation without prose verbs.
"""

from __future__ import annotations

import re

from kg_processor.application.extraction_grounding import surface_spans
from kg_processor.domain.ontology import RelationTypeDefinition

_MAX_PROSE_ENDPOINT_GAP = 140
_EVIDENCE_CONTEXT_CHARS = 24
_MIN_STRUCTURED_LINES = 2
_MAX_STRUCTURED_LINES = 4
_MAX_STRUCTURED_CHARS = 300
_MAX_STRUCTURED_WORDS = 24
_STRUCTURED_RELATIONS = {
    "HAS_PRACTICE",
    "LOCATED_IN",
    "PRACTICED_IN",
    "USES_TECHNIQUE",
}


def relation_evidence_supported(
    quote: str,
    source_surfaces: list[str],
    target_surfaces: list[str],
    definition: RelationTypeDefinition,
) -> bool:
    """Require local predicate evidence when an ontology configures cue patterns.

    Endpoint co-occurrence alone is insufficient. The closest endpoint pair must
    be near an approved cue, except for bounded structured rows whose columns can
    express selected relations without prose verbs.
    """

    if not definition.evidence_cues:
        return True
    source_spans = surface_spans(quote, source_surfaces)
    target_spans = surface_spans(quote, target_surfaces)
    if not source_spans or not target_spans:
        return False
    if definition.name in _STRUCTURED_RELATIONS and _looks_like_structured_record(quote):
        return True

    source_span, target_span = min(
        ((source, target) for source in source_spans for target in target_spans),
        key=lambda pair: (_span_gap(*pair), _enclosing_width(*pair), pair),
    )
    if _span_gap(source_span, target_span) > _MAX_PROSE_ENDPOINT_GAP:
        return False
    start = max(min(source_span[0], target_span[0]) - _EVIDENCE_CONTEXT_CHARS, 0)
    end = min(
        max(source_span[1], target_span[1]) + _EVIDENCE_CONTEXT_CHARS,
        len(quote),
    )
    local_evidence = quote[start:end]
    return any(
        re.search(cue, local_evidence, flags=re.IGNORECASE) is not None
        for cue in definition.evidence_cues
    )


def _span_gap(left: tuple[int, int], right: tuple[int, int]) -> int:
    """Measure characters separating two spans, returning zero whenever they overlap.

    Adjacent spans therefore have a zero-character gap.
    """

    if left[1] < right[0]:
        return right[0] - left[1]
    if right[1] < left[0]:
        return left[0] - right[1]
    return 0


def _enclosing_width(left: tuple[int, int], right: tuple[int, int]) -> int:
    """Measure the source width enclosing both grounded endpoint spans.

    It provides deterministic tie-breaking for equally separated pairs.
    """

    return max(left[1], right[1]) - min(left[0], right[0])


def _looks_like_structured_record(quote: str) -> bool:
    """Recognize compact table-like evidence without accepting arbitrary prose.

    Tight line, character, word, and punctuation bounds keep this exception from
    weakening normal predicate-cue enforcement.
    """

    lines = [line.strip() for line in quote.splitlines() if line.strip()]
    compact_multiline = (
        _MIN_STRUCTURED_LINES <= len(lines) <= _MAX_STRUCTURED_LINES
        and len(quote.split()) <= _MAX_STRUCTURED_WORDS
        and re.search(r"[.!?]", quote) is None
    )
    pipe_row = "|" in quote and quote.count("\n") <= 1
    return len(quote) <= _MAX_STRUCTURED_CHARS and (compact_multiline or pipe_row)
