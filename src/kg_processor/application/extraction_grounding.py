# SPDX-License-Identifier: Apache-2.0
"""Deterministic, record-level evidence grounding for model observations."""

from __future__ import annotations

import html
import os
import re
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache

# A sentence ends at a line break, at a table row's end, or at terminal
# punctuation followed by whitespace, so a decimal such as "0.10" or a compact
# reference such as "Ph.Eur.2.5.12" stays inside its sentence.
_SENTENCE_BOUNDARY_RE = re.compile(r"\n+|(?<=[.!?])\s+|(?<=</tr>)", re.IGNORECASE)
_ELLIPSIS_RE = re.compile(r"\s*(?:\.{3,}|…)\s*")
_MIN_SPLITTABLE_OCR_TOKEN_LENGTH = 2
# A name's parts: runs of letters or digits. Everything else - spaces, hyphens,
# underscores - separates them, so "DOC_731902" reads as "DOC 731902".
_LETTER_DIGIT_RUN_RE = re.compile(r"\d+|[^\W\d_]+")
# An HTML character reference a parser left in the text: "&amp;", "&#x27;".
_HTML_ENTITY_RE = re.compile(r"&(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
# A name's last word may carry a plural or genitive ending the name does not:
# "Acme Pharmas ventilationsafdeling", "the tablets". Only for a word of letters
# at least this long, so a code such as "GG9" stays exact.
_MIN_INFLECTED_WORD_LENGTH = 3
# Layout the text carries between words: HTML tags and table cell bars. A quote
# may read across table cells as prose; these count as whitespace for it.
_LAYOUT_MARKUP_RE = re.compile(r"<[^<>\n]{1,200}>|\|")
# A delimited row is a line that tildes, cell bars or tabs divide into at
# least three cells: this many of one divider. Prose uses a tilde singly
# ("~5 mg"), and semicolons divide its clauses more often than cells.
_ROW_DIVIDERS = ("~", "|", "\t")
_MIN_ROW_DIVIDERS = 2
# A name's words of one or two letters are left out when a row is read for
# it; a number never is.
_MAX_SKIPPED_ROW_WORD = 2
_MIN_QUOTE_PIECE_CHARACTERS = 3
_MIN_QUOTE_PIECES = 2
# A quote given in pieces grounds on the stretch of source that holds all of
# them; beyond this length the pieces no longer read as one statement.
MAX_PIECED_QUOTE_CHARACTERS = 1500
# The OCR-tolerant view compares letters and digits only. A name may be found
# there across lost or extra spaces once it has this many of them...
_MIN_SPACELESS_CHARACTERS = 6
# ...and with a misread character once it has this many, one misread for every
# this many characters, rounded up.
_MIN_CONFUSABLE_CHARACTERS = 5
_CHARACTERS_PER_CONFUSION = 8
# A misreading that changes the length ("rn" for "m") only in a name this long:
# in a shorter one it turns one word into another, as "modem" and "modern".
_MIN_LENGTH_CHANGING_CONFUSION_CHARACTERS = 8
# Names are looked for in chunk after chunk, so their patterns are kept; a
# quote longer than this is looked for once, and its pattern is not.
_MAX_CACHED_PATTERN_CHARACTERS = 120
# Characters OCR reads for one another, either way round, after case folding:
# vertical strokes (l, 1, I, j), round letters (c, e, o) and zero, 5 and S, 8 and
# B, and letter pairs that print as one letter. A cell bar "|" is table layout
# here, never a letter.
_OCR_CONFUSIONS = (
    ("l", "1"),
    ("l", "i"),
    ("i", "1"),
    ("l", "j"),
    ("i", "j"),
    ("c", "e"),
    ("c", "o"),
    ("o", "0"),
    ("s", "5"),
    ("b", "8"),
    ("rn", "m"),
    ("cl", "d"),
    ("ii", "u"),
)
_OCR_READINGS: dict[str, tuple[str, ...]] = {
    written: tuple(
        sorted(
            {second for first, second in _OCR_CONFUSIONS if first == written}
            | {first for first, second in _OCR_CONFUSIONS if second == written}
        )
    )
    for pair in _OCR_CONFUSIONS
    for written in pair
}


@dataclass(frozen=True)
class GroundedSpan:
    """Represent an exact source substring and chunk-local character offsets.

    ``repair`` records when a provider quote was replaced by a source sentence,
    allowing traces to distinguish direct grounding from conservative recovery.
    ``ocr_tolerant`` records evidence found only through the OCR-tolerant view,
    so it can be labelled and judged apart from evidence that matched as written.
    """

    quote: str
    start_offset: int
    end_offset: int
    repair: str | None = None
    ocr_tolerant: bool = False


@dataclass(frozen=True)
class SurfaceTextIndex:
    """Cache normalized source text and its exact original offset mapping."""

    normalized_source: str
    source_starts: list[int]
    source_ends: list[int]


def ground_evidence(
    source: str,
    proposed_quote: str,
    required_surface_groups: list[list[str]],
    *,
    allow_nested: bool = False,
) -> GroundedSpan | None:
    """Ground a quote or repair it to the shortest supporting sentence.

    Each surface group represents one required endpoint and is satisfied when
    any configured name or alias occurs in the source span.  Invalid records
    are dropped independently; one hallucinated quote never rewrites or loses
    the other valid observations returned by the same provider call.
    """

    return _ground(source, proposed_quote, required_surface_groups, allow_nested=allow_nested)


def ground_evidence_ocr_tolerant(
    source: str,
    proposed_quote: str,
    required_surface_groups: list[list[str]],
    *,
    allow_nested: bool = False,
) -> GroundedSpan | None:
    """Ground what ``ground_evidence`` could not, reading the text OCR-tolerantly.

    OCR text loses the spaces between words ("SmithChemicalsLtd") or misreads
    a character ("Stain1ess steel"), while the model writes the name as it is
    meant. The same steps run again with the quote and every name also looked
    for in a view of letters and digits only, where a few known misreadings
    count as the same character. It is the last tier: callers try every other
    way first, so text that grounds as written keeps its evidence. The span is
    always the source's own text and is marked ``ocr_tolerant``.
    """

    grounded = _ground(
        source, proposed_quote, required_surface_groups, allow_nested=allow_nested, tolerant=True
    )
    return replace(grounded, ocr_tolerant=True) if grounded is not None else None


def _ground(
    source: str,
    proposed_quote: str,
    required_surface_groups: list[list[str]],
    *,
    allow_nested: bool,
    tolerant: bool = False,
) -> GroundedSpan | None:
    """The quote, else the shortest supporting sentences, else a pieced quote."""

    quote_span = _find_quote(source, proposed_quote, tolerant=tolerant)
    if quote_span is not None and _contains_groups(
        quote_span.quote, required_surface_groups, allow_nested=allow_nested, tolerant=tolerant
    ):
        return quote_span
    supporting = _supporting_span(
        source, required_surface_groups, allow_nested=allow_nested, tolerant=tolerant
    )
    if supporting is not None:
        return supporting
    return _pieced_quote_span(
        source,
        proposed_quote,
        required_surface_groups,
        allow_nested=allow_nested,
        tolerant=tolerant,
    )


def _pieced_quote_span(
    source: str,
    proposed_quote: str,
    required_surface_groups: list[list[str]],
    *,
    allow_nested: bool = False,
    tolerant: bool = False,
) -> GroundedSpan | None:
    """Ground a quote the model gave in pieces joined by an ellipsis.

    Forms, certificates, labels and tables state a fact through layout: the
    subject in a header, the value in a field below. A model quoting that fact
    joins the two parts with "...". Every piece must occur exactly in the
    source and the pieces together must name every endpoint; the evidence is
    the exact source stretch from the first piece to the last, bounded so it
    cannot become the whole document.
    """

    pieces = [piece for piece in _ELLIPSIS_RE.split(proposed_quote) if piece.strip()]
    if len(pieces) < _MIN_QUOTE_PIECES or any(
        len(piece.strip()) < _MIN_QUOTE_PIECE_CHARACTERS for piece in pieces
    ):
        return None
    spans = [_find_quote(source, piece, tolerant=tolerant) for piece in pieces]
    found = [span for span in spans if span is not None]
    if len(found) != len(pieces):
        return None
    if not _contains_groups(
        "\n".join(span.quote for span in found),
        required_surface_groups,
        allow_nested=allow_nested,
        tolerant=tolerant,
    ):
        return None
    start = min(span.start_offset for span in found)
    end = max(span.end_offset for span in found)
    if end - start > MAX_PIECED_QUOTE_CHARACTERS:
        return None
    return GroundedSpan(source[start:end], start, end, repair="quote_pieces")


def surface_occurs(source: str, surfaces: list[str]) -> bool:
    """Return whether any complete, token-bounded entity surface occurs in text.

    The helper shares the exact matching policy used for evidence repair so alias
    validation and relation grounding cannot disagree about occurrence.
    """

    return bool(surface_spans(source, surfaces))


def build_surface_text_index(source: str) -> SurfaceTextIndex:
    """Normalize one source chunk once for repeated endpoint grounding checks."""

    normalized_source, source_starts, source_ends = _normalized_text_with_offsets(source)
    return SurfaceTextIndex(normalized_source, source_starts, source_ends)


def indexed_surface_occurs(index: SurfaceTextIndex, surfaces: list[str]) -> bool:
    """Check surfaces against a previously normalized source chunk."""

    return bool(_surface_spans_from_index(index, surfaces))


def indexed_surface_spans(
    index: SurfaceTextIndex,
    surfaces: list[str],
) -> list[tuple[int, int]]:
    """Return exact source offsets for surfaces using a precomputed text index."""

    return _surface_spans_from_index(index, surfaces)


def surface_spans(source: str, surfaces: list[str]) -> list[tuple[int, int]]:
    """Find token-bounded spans while tolerating punctuation between name tokens.

    Substring matching would incorrectly ground ``India`` in ``Indian`` and
    ``Greece`` in ``Greek``. Token boundaries are therefore part of the graph
    evidence contract, while spaces and hyphens remain interchangeable. A
    multi-token surface may also carry an attached numeric footnote marker, as
    PDF extractors commonly render an author and affiliation as ``Jane Doe1*``.
    One token may contain a first-letter OCR split such as ``M ETHOD``; allowing
    exactly one such split preserves identity while avoiding general fuzzy
    matching. Single-token surfaces keep the strict boundary to avoid matching a
    generic name such as ``Model`` inside ``Model2``.
    """

    return _surface_spans_from_index(build_surface_text_index(source), surfaces)


def ocr_tolerant_surface_occurs(source: str, surfaces: list[str]) -> bool:
    """Return whether any surface occurs in the text's OCR-tolerant view."""

    return bool(ocr_tolerant_surface_spans(source, surfaces))


def ocr_tolerant_surface_spans(source: str, surfaces: list[str]) -> list[tuple[int, int]]:
    """Find surfaces the way OCR may have printed them, as exact source offsets.

    The last tier after ``surface_spans``: letters and digits are compared
    without the spaces and punctuation between them, so "Smith Chemicals Ltd"
    is found in "SmithChemicalsLtd", and a few known misreadings count as the
    same character, so "Stainless steel" is found in "Stain1ess steel". See
    ``_ocr_tolerant_spans`` for the guards that keep a short name out of a
    longer word.
    """

    return sorted({span for surface in surfaces for span in _ocr_tolerant_spans(source, surface)})


def _surface_spans_from_index(
    index: SurfaceTextIndex,
    surfaces: list[str],
) -> list[tuple[int, int]]:
    normalized_source = index.normalized_source
    source_starts = index.source_starts
    source_ends = index.source_ends
    spans: set[tuple[int, int]] = set()
    for surface in surfaces:
        normalized_surface = unicodedata.normalize("NFKC", surface).casefold()
        tokens = re.findall(r"\w+", normalized_surface, flags=re.UNICODE)
        if not tokens:
            continue
        trailing_marker = r"(?:\d+[*†‡]*)?" if len(tokens) > 1 and not tokens[-1].isdigit() else ""
        for pattern in _compiled_surface_patterns(tuple(tokens), trailing_marker):
            for match in pattern.finditer(normalized_source):
                spans.add((source_starts[match.start()], source_ends[match.end() - 1]))
    return sorted(spans)


@lru_cache(maxsize=8192)
def _compiled_surface_patterns(
    tokens: tuple[str, ...],
    trailing_marker: str,
) -> tuple[re.Pattern[str], ...]:
    """Compile one surface's patterns once and reuse them across every chunk.

    Grounding matches every accepted entity against every chunk of its window, so
    the same handful of patterns per entity is used hundreds of times. The re
    module's own cache is bounded and a corpus of a few hundred entities evicts
    faster than it reuses, leaving the work to be redone on nearly every call.
    """

    return tuple(
        re.compile(pattern) for pattern in _surface_patterns(list(tokens), trailing_marker)
    )


def _surface_patterns(tokens: list[str], trailing_marker: str) -> list[str]:
    """Build exact and single-prefix-split patterns for one entity surface.

    A separate alternative is emitted for each eligible token, which means one
    match can repair at most one PDF extraction artifact. The rest of the token
    sequence must remain exact and token-bounded.
    """

    # Letters and digits within a name may be written together or apart:
    # "GG918", "GG-918" and "GG 918" name one thing, so each letter or digit run
    # is its own part and a letter-digit change takes an optional separator.
    runs = [run for token in tokens for run in _LETTER_DIGIT_RUN_RE.findall(token)]
    run_patterns = [re.escape(run) for run in runs]
    alternatives = [run_patterns]
    for index, run in enumerate(runs):
        if len(run) < _MIN_SPLITTABLE_OCR_TOKEN_LENGTH or run.isdigit():
            continue
        repaired = list(run_patterns)
        repaired[index] = re.escape(run[0]) + r"[\W_]+" + re.escape(run[1:])
        alternatives.append(repaired)
    ending = (
        r"(?:e?s)?" if runs[-1].isalpha() and len(runs[-1]) >= _MIN_INFLECTED_WORD_LENGTH else ""
    )
    # An underscore joins words in file and folder names, so it ends a name too.
    # A name of several parts may also touch digits OCR glued to it - a part
    # number before "15548-01SmatProtect" - but never another letter; a
    # one-part name keeps the strict edge, so "Model" is not in "Model2".
    lead, trail = r"(?<![^\W_])", r"(?![^\W_])"
    if len(runs) > 1:
        lead = r"(?<![^\W\d_])" if runs[0].isalpha() else r"(?<!\d)"
        trail = r"(?![^\W\d_])" if runs[-1].isalpha() else r"(?!\d)"
    return [
        lead + _join_runs(runs, parts) + ending + trailing_marker + trail for parts in alternatives
    ]


def _join_runs(runs: list[str], parts: list[str]) -> str:
    """Join name parts: required separators, optional ones at a letter-digit change."""

    joined = parts[0]
    for previous, run, part in zip(runs, runs[1:], parts[1:], strict=False):
        optional = previous[-1].isdigit() != run[0].isdigit()
        joined += (r"[\W_]*" if optional else r"[\W_]+") + part
    return joined


def sentence_spans(source: str) -> list[GroundedSpan]:
    """Return trimmed, offset-preserving sentence spans from normalized source text.

    Extraction and deterministic candidate discovery share this boundary so a
    cue-derived relation can retain the same exact evidence contract as an LLM-proposed
    relation. Line-oriented fragments without terminal punctuation remain supported,
    and each row of an HTML table is its own sentence.
    """

    spans: list[GroundedSpan] = []
    start = 0
    for boundary in [*_SENTENCE_BOUNDARY_RE.finditer(source), None]:
        end = boundary.start() if boundary is not None else len(source)
        raw = source[start:end]
        left_trim = len(raw) - len(raw.lstrip())
        sentence = raw.strip()
        if sentence:
            spans.append(
                GroundedSpan(sentence, start + left_trim, start + left_trim + len(sentence))
            )
        if boundary is not None:
            start = boundary.end()
    return spans


def delimited_row_span(source: str, start: int, end: int) -> tuple[int, int] | None:
    """The delimited row of ``source`` that holds ``source[start:end]``, if one does.

    A delimited row is one line that tildes, cell bars or tabs divide into at
    least three cells. A stretch that spans lines, or lies in a line no such
    dividers divide, lies in no row.
    """

    line_start = source.rfind("\n", 0, start) + 1
    line_end = source.find("\n", start)
    line_end = len(source) if line_end < 0 else line_end
    if end > line_end:
        return None
    line = source[line_start:line_end]
    if any(line.count(divider) >= _MIN_ROW_DIVIDERS for divider in _ROW_DIVIDERS):
        return line_start, line_end
    return None


def row_names(row: str, names: list[str]) -> bool:
    """Whether a delimited row names one of ``names``, word by word.

    Every word of the name - a run of letters or of digits, leaving out words
    of one or two letters but never a number - occurs in the row, both read
    case-insensitively without spaces or punctuation: "tri-Zorvane citrate"
    is named in "TRIZORVANE CITRATE~ORAL", ".ALPHA.-VORANOL" in
    "Alpha-voranol". A name of short words only must occur whole that way.
    """

    compact_row = _compact_words(row)
    for name in names:
        words = _LETTER_DIGIT_RUN_RE.findall(unicodedata.normalize("NFKC", name).casefold())
        wanted = [
            word for word in words if word.isdigit() or len(word) > _MAX_SKIPPED_ROW_WORD
        ] or ["".join(words)]
        if all(word in compact_row for word in wanted):
            return True
    return False


def _compact_words(text: str) -> str:
    """The text's letters and digits, case-folded, with nothing between them."""

    return "".join(_LETTER_DIGIT_RUN_RE.findall(unicodedata.normalize("NFKC", text).casefold()))


def find_quote(source: str, quote: str) -> GroundedSpan | None:
    """Locate a quote in source text under the grounding contract's tolerances.

    Whitespace, case, and compatibility characters may differ; words may not.
    The span returned is always the exact source text.
    """

    return _find_quote(source, quote)


def _find_quote(source: str, quote: str, *, tolerant: bool = False) -> GroundedSpan | None:
    """Locate a proposed quote exactly or with whitespace-only normalization.

    Even when whitespace is repaired, the returned text and offsets always refer
    to the original source rather than a model-normalized representation.
    ``tolerant`` adds the OCR-tolerant view as a last reading.
    """

    found = _find_quote_as_written(source, quote)
    if found is not None or not tolerant:
        return found
    spans = _ocr_tolerant_spans(source, quote)
    if not spans:
        return None
    start, end = spans[0]
    return GroundedSpan(source[start:end], start, end)


def _find_quote_as_written(source: str, quote: str) -> GroundedSpan | None:
    """Locate a quote under every view that compares its words as written."""

    stripped = quote.strip()
    if not stripped:
        return None
    exact_start = source.find(stripped)
    if exact_start >= 0:
        return GroundedSpan(stripped, exact_start, exact_start + len(stripped))

    # OCR and model serializers can normalize whitespace while preserving all
    # words. Match that representation but always return the exact source span.
    tokens = stripped.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, source, flags=re.IGNORECASE)
    if match is not None:
        return GroundedSpan(source[match.start() : match.end()], match.start(), match.end())

    # PDF text commonly contains compatibility ligatures such as ``ﬁ`` while a
    # model returns ordinary ``fi``. Match in NFKC/casefold space, then map the
    # span back to the untouched source so evidence offsets and quotes remain
    # exact. No edit-distance matching is used: words still have to agree fully.
    normalized_quote = unicodedata.normalize("NFKC", html.unescape(stripped)).casefold()
    # A quote may drop a line-break hyphen ("hydration") or keep it
    # ("hydra-tion"); the text is read both ways. Last, it is read the way a
    # quote across table cells reads it: tags and cell bars as spaces.
    # Blanking keeps every offset where it was.
    for keep_break_hyphens in (False, True):
        normalized_source, source_starts, source_ends = _normalized_text_with_offsets(
            source, keep_break_hyphens=keep_break_hyphens
        )
        for view, quote_view in (
            (normalized_source, normalized_quote),
            (_blank_layout(normalized_source), _blank_layout(normalized_quote)),
        ):
            normalized_tokens = quote_view.split()
            if not normalized_tokens:
                continue
            normalized_pattern = r"\s+".join(re.escape(token) for token in normalized_tokens)
            normalized_match = re.search(normalized_pattern, view)
            if normalized_match is not None:
                start = source_starts[normalized_match.start()]
                end = source_ends[normalized_match.end() - 1]
                return GroundedSpan(source[start:end], start, end)
    return None


def _blank_layout(text: str) -> str:
    """The text with layout markup replaced by spaces of the same length."""

    return _LAYOUT_MARKUP_RE.sub(lambda match: " " * len(match.group()), text)


def _normalized_text_with_offsets(
    source: str, *, keep_break_hyphens: bool = False
) -> tuple[str, list[int], list[int]]:
    """Build NFKC/casefold text and an exact normalized-to-source offset map.

    Compatibility characters can expand to several normalized code points. Each
    expanded character records the same original span, allowing callers to map a
    successful normalized regex match back without modifying source evidence.
    An HTML character reference a parser left in the text reads as the character
    it stands for. Visual line-break hyphens between word characters are then
    removed from the matching view - or, with ``keep_break_hyphens``, only the
    break after them, for a quote that keeps the hyphen ("hydra-tion") -
    while their source positions remain recoverable through the surrounding
    characters. Ordinary compounds without following whitespace are preserved.
    """

    normalized: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(source):
        entity = _HTML_ENTITY_RE.match(source, index) if source[index] == "&" else None
        decoded = html.unescape(entity.group()) if entity else ""
        if entity and decoded != entity.group():
            for normalized_character in unicodedata.normalize("NFKC", decoded).casefold():
                normalized.append(normalized_character)
                starts.append(index)
                ends.append(entity.end())
            index = entity.end()
            continue
        cluster_end = index + 1
        while cluster_end < len(source) and unicodedata.combining(source[cluster_end]):
            cluster_end += 1
        cluster = source[index:cluster_end]
        for normalized_character in unicodedata.normalize("NFKC", cluster).casefold():
            normalized.append(normalized_character)
            starts.append(index)
            ends.append(cluster_end)
        index = cluster_end
    return _remove_visual_line_break_hyphens(
        "".join(normalized), starts, ends, keep_hyphen=keep_break_hyphens
    )


def _remove_visual_line_break_hyphens(
    normalized: str,
    starts: list[int],
    ends: list[int],
    *,
    keep_hyphen: bool = False,
) -> tuple[str, list[int], list[int]]:
    """Remove PDF line-wrap hyphens - or only the break after them - from a view."""

    removed_indices = {
        index
        # A hyphen and the break after it; a stray space may precede a hyphen
        # that ends a line ("diame -" / "ter"), but a spaced dash inside a line
        # ("Foam - a review") is punctuation and stays.
        for match in re.finditer(r"(?<=\w)(?: -\n\s*|-\s+)(?=\w)", normalized)
        for index in range(match.start() + keep_hyphen * (match.group().find("-") + 1), match.end())
    }
    if not removed_indices:
        return normalized, starts, ends
    kept_indices = [index for index in range(len(normalized)) if index not in removed_indices]
    return (
        "".join(normalized[index] for index in kept_indices),
        [starts[index] for index in kept_indices],
        [ends[index] for index in kept_indices],
    )


def _supporting_span(
    source: str,
    required_surface_groups: list[list[str]],
    *,
    allow_nested: bool = False,
    tolerant: bool = False,
) -> GroundedSpan | None:
    """Select the shortest one-to-three-sentence span containing every surface group.

    Most evidence is sentence-local. A bounded adjacent span also covers explicit
    antecedents such as ``The model is X. It uses Y`` without allowing arbitrary
    document-level co-occurrence. Length and source offset provide deterministic
    tie-breaking when several spans can ground the same observation.
    """

    if tolerant and not _contains_groups(
        source, required_surface_groups, allow_nested=allow_nested, tolerant=True
    ):
        # No sentence can hold what the whole text does not; this spares
        # reading every sentence run OCR-tolerantly for a name that is absent.
        return None
    source_sentences = sentence_spans(source)
    candidates: list[GroundedSpan] = []
    for start_index, first in enumerate(source_sentences):
        for end_index in range(start_index, min(start_index + 3, len(source_sentences))):
            last = source_sentences[end_index]
            quote = source[first.start_offset : last.end_offset]
            if quote and _contains_groups(
                quote, required_surface_groups, allow_nested=allow_nested, tolerant=tolerant
            ):
                candidates.append(
                    GroundedSpan(
                        quote=quote,
                        start_offset=first.start_offset,
                        end_offset=last.end_offset,
                        repair=(
                            "supporting_sentence"
                            if end_index == start_index
                            else "supporting_passage"
                        ),
                    )
                )
                # Longer spans from the same starting sentence cannot be the
                # shortest candidate once all endpoint groups are present.
                break
    return min(candidates, key=lambda item: (len(item.quote), item.start_offset), default=None)


# Words of a rephrased name shorter than this must appear in its quote exactly;
# longer ones may differ in ending, as "measurement" and "measured" do.
_MIN_INFLECTABLE_WORD = 4
# Letters a longer word and its quoted form must share at the start, at least.
_MIN_SHARED_STEM = 4
# Letters the two may differ by after what they share.
_MAX_ENDING_DIFFERENCE = 4


def name_words_in_quote(name: str, quote: str) -> bool:
    """Whether every word of a name is a word of the quote, up to its ending.

    A model may name what the text says in its own words: "pH measurement" for
    "the pH of the syrups was measured". The name is kept when each of its words
    is in the verbatim quote - short words exactly, longer ones sharing their
    stem ("pores" for "pore") - so it rephrases the text and adds nothing to it.
    """

    words = re.findall(r"\w+", unicodedata.normalize("NFKC", name).casefold())
    quoted = set(re.findall(r"\w+", unicodedata.normalize("NFKC", quote).casefold()))
    return bool(words) and all(any(_same_word(word, other) for other in quoted) for word in words)


def name_shares_stem(name: str, text: str) -> bool:
    """Whether a word of the name shares its stem with a word of the text.

    The weakest tie a model-confirmed entity must still have to its evidence:
    "stability" and "stable" share "stab". Short names must appear whole.
    """

    words = re.findall(r"\w+", unicodedata.normalize("NFKC", name).casefold())
    found = set(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))
    long_words = [word for word in words if len(word) >= _MIN_INFLECTABLE_WORD]
    if not long_words:
        return bool(words) and all(word in found for word in words)
    return any(
        len(os.path.commonprefix([word, other])) >= _MIN_SHARED_STEM
        for word in long_words
        for other in found
    )


def _same_word(word: str, other: str) -> bool:
    if word == other:
        return True
    if min(len(word), len(other)) < _MIN_INFLECTABLE_WORD:
        return False
    shared = len(os.path.commonprefix([word, other]))
    return (
        shared >= _MIN_SHARED_STEM and max(len(word), len(other)) - shared <= _MAX_ENDING_DIFFERENCE
    )


def contains_surface_groups(
    source: str,
    groups: list[list[str]],
    *,
    allow_nested: bool = False,
    ocr_tolerant: bool = False,
) -> bool:
    """Return whether every endpoint group has its own, non-overlapping mention.

    The public form of the rule ``ground_evidence`` applies, for callers that
    locate evidence another way and must hold it to the same standard;
    ``ocr_tolerant`` applies the rule of ``ground_evidence_ocr_tolerant``.
    """

    return _contains_groups(source, groups, allow_nested=allow_nested, tolerant=ocr_tolerant)


def _contains_groups(
    source: str,
    groups: list[list[str]],
    *,
    allow_nested: bool = False,
    tolerant: bool = False,
) -> bool:
    """Require a distinct complete source span for every endpoint alias group.

    Merely finding each string is insufficient when one entity name is nested in
    another, such as ``gradient descent`` inside ``stochastic gradient descent``.
    Distinct graph endpoints must be witnessed by non-overlapping text spans so a
    single mention cannot masquerade as both source and target. Backtracking is
    bounded by the small endpoint-group count used by extraction contracts.
    ``tolerant`` adds each group's OCR-tolerant spans when the spans as written
    do not suffice.
    """

    spans_by_group = [surface_spans(source, group) for group in groups]
    if _groups_fit(spans_by_group, allow_nested=allow_nested):
        return True
    if not tolerant:
        return False
    return _groups_fit(
        [
            sorted({*spans, *ocr_tolerant_surface_spans(source, group)})
            for spans, group in zip(spans_by_group, groups, strict=True)
        ],
        allow_nested=allow_nested,
    )


def _groups_fit(spans_by_group: list[list[tuple[int, int]]], *, allow_nested: bool) -> bool:
    """Whether the groups' spans give each group a mention of its own."""

    if any(not spans for spans in spans_by_group):
        return False
    if _has_non_overlapping_group_spans(spans_by_group, 0, []):
        return True
    # With ``allow_nested``, a name that contains the other endpoint's name
    # states their relation itself: "10% urea foam" is a foam.
    return (
        allow_nested
        and len(spans_by_group) == 2  # noqa: PLR2004 - a relation's two endpoints.
        and any(
            _contains_span(outer, inner) or _contains_span(inner, outer)
            for outer in spans_by_group[0]
            for inner in spans_by_group[1]
        )
    )


def _contains_span(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer != inner and outer[0] <= inner[0] and inner[1] <= outer[1]


def _has_non_overlapping_group_spans(
    spans_by_group: list[list[tuple[int, int]]],
    group_index: int,
    selected: list[tuple[int, int]],
) -> bool:
    """Find one mutually non-overlapping span from each endpoint group."""

    if group_index == len(spans_by_group):
        return True
    for candidate in spans_by_group[group_index]:
        if all(
            candidate[1] <= existing[0] or existing[1] <= candidate[0] for existing in selected
        ) and _has_non_overlapping_group_spans(
            spans_by_group,
            group_index + 1,
            [*selected, candidate],
        ):
            return True
    return False


@dataclass(frozen=True)
class _CompactText:
    """A text's letters and digits alone, with the source span of each.

    ``breaks[index]`` says whether anything else - a space, punctuation, table
    markup - stood between that character and the one before it.
    """

    text: str
    breaks: tuple[bool, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]


@lru_cache(maxsize=64)
def _compact_text(text: str) -> _CompactText:
    """Read a text as its letters and digits alone, keeping each one's source span.

    It is read like the other views - compatibility characters, case, HTML
    references and line-break hyphens normalized, table markup blanked - and
    all but letters and digits is then dropped. Cached, since one chunk is read
    for every name looked for in it.
    """

    normalized, starts, ends = _normalized_text_with_offsets(text)
    characters: list[str] = []
    breaks: list[bool] = []
    kept_starts: list[int] = []
    kept_ends: list[int] = []
    separated = False
    for index, character in enumerate(_blank_layout(normalized)):
        if not character.isalnum():
            separated = True
            continue
        breaks.append(separated and bool(characters))
        characters.append(character)
        kept_starts.append(starts[index])
        kept_ends.append(ends[index])
        separated = False
    return _CompactText("".join(characters), tuple(breaks), tuple(kept_starts), tuple(kept_ends))


def _ocr_tolerant_spans(source: str, surface: str) -> list[tuple[int, int]]:
    """Source spans where a name or quote reads OCR-tolerantly, as exact offsets.

    Letters and digits are compared without what stands between them, and one
    character in every eight may be a known misreading of the one written.
    Guards keep this from finding what is not there: spaces may differ only in
    a surface of six letters and digits, a character may be misread only in one
    of five, and a match may not begin or end inside a longer word unless the
    text there is glued (see ``_edges_hold``).
    """

    wanted = _compact_text(surface)
    if len(wanted.text) < _MIN_CONFUSABLE_CHARACTERS:
        return []
    view = _compact_text(source)
    budget = -(-len(wanted.text) // _CHARACTERS_PER_CONFUSION)
    spans: list[tuple[int, int]] = []
    pattern = (
        _cached_ocr_candidate_pattern(wanted.text)
        if len(wanted.text) <= _MAX_CACHED_PATTERN_CHARACTERS
        else _ocr_candidate_pattern(wanted.text)
    )
    for candidate in pattern.finditer(view.text):
        start = candidate.start()
        match = _confusable_match(view, wanted, start, budget)
        if match is not None and _edges_hold(source, view, wanted, start, *match):
            spans.append((view.starts[start], view.ends[match[0] - 1]))
    return spans


@lru_cache(maxsize=8192)
def _cached_ocr_candidate_pattern(wanted: str) -> re.Pattern[str]:
    """A name's candidate pattern, compiled once for every chunk it is looked for in."""

    return _ocr_candidate_pattern(wanted)


def _ocr_candidate_pattern(wanted: str) -> re.Pattern[str]:
    """Match wherever a compact surface could begin, each character as itself or a reading.

    A quick filter only: every place it finds is checked by ``_confusable_match``,
    which counts the misreadings and applies their limits.
    """

    parts: list[str] = []
    index = 0
    while index < len(wanted):
        pair = wanted[index : index + 2]
        if len(pair) > 1 and pair in _OCR_READINGS:
            parts.append(
                "(?:"
                + _readings_pattern(pair[0])
                + _readings_pattern(pair[1])
                + "".join("|" + re.escape(reading) for reading in _OCR_READINGS[pair])
                + ")"
            )
            index += 2
            continue
        parts.append(_readings_pattern(wanted[index]))
        index += 1
    return re.compile("(?=" + "".join(parts) + ")")


def _readings_pattern(character: str) -> str:
    return (
        "(?:"
        + "|".join(re.escape(text) for text in (character, *_OCR_READINGS.get(character, ())))
        + ")"
    )


def _confusable_match(
    view: _CompactText, wanted: _CompactText, start: int, budget: int
) -> tuple[int, frozenset[int]] | None:
    """Where a surface read from ``start`` ends, and the view positions of its own breaks.

    Each character matches itself or, within the budget, a reading OCR gives it.
    In a surface too short to lose its spaces, the text's breaks must fall where
    the surface's do.
    """

    size = len(wanted.text)
    keep_breaks = size < _MIN_SPACELESS_CHARACTERS
    stack: list[tuple[int, int, int, frozenset[int]]] = [(start, 0, 0, frozenset())]
    while stack:
        at, position, used, name_breaks = stack.pop()
        if position == size:
            return at, name_breaks
        if at >= len(view.text):
            continue
        if position:
            if keep_breaks and view.breaks[at] != wanted.breaks[position]:
                continue
            if wanted.breaks[position]:
                name_breaks = name_breaks | {at}
        if used < budget:
            stack.extend(
                (at + len(reading), position + len(written), used + 1, name_breaks)
                for written, readings in _OCR_READINGS.items()
                if wanted.text.startswith(written, position)
                for reading in readings
                if view.text.startswith(reading, at)
                and _reading_allowed(view, wanted, at, position, written, reading)
            )
        # Pushed last so a character that matches as written is tried first.
        if view.text[at] == wanted.text[position]:
            stack.append((at + 1, position + 1, used, name_breaks))
    return None


def _reading_allowed(  # noqa: PLR0913 - one misreading and where it falls.
    view: _CompactText,
    wanted: _CompactText,
    at: int,
    position: int,
    written: str,
    reading: str,
) -> bool:
    """Whether a misreading may stand where it falls.

    A pair of letters read as one ("rn" as "m") only in a long surface and
    within one word. A digit read for a letter only in a word of letters
    ("Stab1lity", "stee1"), and a letter for a digit only inside a number
    ("2O24"): where a code's letters meet its digits ("AB 5123" against
    "AB s123") both readings are codes, so it must match as written.
    """

    if len(written) > 1 or len(reading) > 1:
        return (
            len(wanted.text) >= _MIN_LENGTH_CHANGING_CONFUSION_CHARACTERS
            and not any(wanted.breaks[position + 1 : position + len(written)])
            and not any(view.breaks[at + 1 : at + len(reading)])
        )
    if written.isdigit() == reading.isdigit():
        return True
    first = position
    while first and not wanted.breaks[first]:
        first -= 1
    last = position + 1
    while last < len(wanted.text) and not wanted.breaks[last]:
        last += 1
    word = wanted.text[first:last]
    if written.isdigit():
        return word.isdigit() and first < position < last - 1
    return word.isalpha()


def _case_edge(source: str, position: int) -> bool:
    """Whether a word edge sits at ``position`` only as a change of case, on a table row.

    Values an export ran together in one cell keep their capitals: a lower-case
    letter before a capital ("alcoholCrodacol"), or a capital before a capital
    that starts a word ("ApSColorcon"). Only table rows are read this way; in
    prose a change of case is inside a name ("PowerPoint").
    """

    if position <= 0 or position + 1 > len(source):
        return False
    before, after = source[position - 1], source[position]
    following = source[position + 1] if position + 1 < len(source) else ""
    changes = (before.islower() and after.isupper()) or (
        before.isupper() and after.isupper() and following.islower()
    )
    if not changes:
        return False
    line_start = source.rfind("\n", 0, position) + 1
    line_end = source.find("\n", position)
    line = source[line_start : line_end if line_end >= 0 else len(source)]
    return "|" in line or "\t" in line or "<td" in line


def _edges_hold(
    source: str,
    view: _CompactText,
    wanted: _CompactText,
    start: int,
    end: int,
    name_breaks: frozenset[int],
) -> bool:
    """Whether a match begins and ends at a word's edge, or where the text is glued.

    An edge inside a longer word of letters is allowed only when the surface
    has a space inside that same word of the text, which the text left out:
    "Smith Chemicals Ltd" may end inside "SmithChemicalsLtdHighStreet", but
    "Test Method" never begins inside "the latest method", and a one-word name
    never begins or ends inside another word. On a table row, a change of case
    is an edge too: a cell listing several values with nothing between them
    ("Acme ApSNorthwindGlobex Ltd") names "Northwind". An edge inside a run of
    digits is never allowed: a number cut short is another number.
    """

    one_run = not any(wanted.breaks) and (wanted.text.isalpha() or wanted.text.isdigit())

    def inside_word(index: int) -> bool:
        if index <= 0 or index >= len(view.text) or view.breaks[index]:
            return False
        before, after = view.text[index - 1], view.text[index]
        if before.isalpha() and after.isalpha() and _case_edge(source, view.starts[index]):
            return False
        return one_run or before.isdigit() == after.isdigit()

    # A name break glues an edge when the text has no break from the edge to it.
    glued_start = not view.text[start].isdigit() and any(
        not any(view.breaks[start + 1 : position + 1]) for position in name_breaks
    )
    glued_end = not view.text[end - 1].isdigit() and any(
        not any(view.breaks[position:end]) for position in name_breaks
    )
    return (not inside_word(start) or glued_start) and (not inside_word(end) or glued_end)
