# SPDX-License-Identifier: Apache-2.0
"""Composable OCR fallback selected by normalized document text coverage."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from kg_processor.application.window_errors import is_systemic_provider_error
from kg_processor.domain.documents import InputFile, ParsedDocument
from kg_processor.ports.ocr import NoTextReadError, OcrOptions, OcrProvider

_UNBROKEN_LATIN_TOKEN_LENGTH = 40
_LATIN_LETTER_RUN_RE = re.compile(r"[A-Za-z]+")
_UNSPACED_RUN_LENGTH = 30
_UNSPACED_RUN_MIN_DISTINCT_LETTERS = 8
_UNSPACED_PAGE_LETTER_SHARE = 0.2
_SPACING_MEASURED_PAGE_LETTERS = 400
_MIN_FRAGMENT_LENGTH = 2
_LATEX_MATH_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$", flags=re.DOTALL)
# A word is a run of letters, joined across an apostrophe ("l'eau", "don't").
_WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*")
_APOSTROPHES = str.maketrans("", "", "'\u2019")
# One-letter words of Latin-script languages: English "a" and "I", French "à",
# Danish and Norwegian "å" and "ø", Portuguese "é".
_LEGITIMATE_SINGLE_LETTER_WORDS = frozenset({"a", "i", "à", "å", "ø", "é"})


class FallbackOcrProvider:
    """Use a lightweight primary parser and fall back when its text is insufficient.

    The adapter composes two ordinary ``OcrProvider`` implementations. Selection
    happens only on their shared ``ParsedDocument`` contract, keeping the pipeline
    independent from concrete OCR libraries and making mixed native/scanned
    corpora efficient without silently accepting blank pages.
    """

    def __init__(
        self,
        primary: OcrProvider,
        secondary: OcrProvider,
        *,
        primary_name: str,
        secondary_name: str,
        min_characters_per_page: int,
        max_sparse_page_ratio: float,
        max_unbroken_text_ratio: float,
        max_fragmented_text_ratio: float,
        max_unspaced_page_ratio: float,
        max_interleaved_column_page_ratio: float = 1.0,
        secondary_supported_suffixes: frozenset[str] | None = None,
    ) -> None:
        """Store the providers and the density and text-coherence thresholds."""

        self.primary = primary
        self.secondary = secondary
        self.primary_name = primary_name
        self.secondary_name = secondary_name
        self.min_characters_per_page = min_characters_per_page
        self.max_sparse_page_ratio = max_sparse_page_ratio
        self.max_unbroken_text_ratio = max_unbroken_text_ratio
        self.max_fragmented_text_ratio = max_fragmented_text_ratio
        self.max_unspaced_page_ratio = max_unspaced_page_ratio
        self.max_interleaved_column_page_ratio = max_interleaved_column_page_ratio
        self.secondary_supported_suffixes = secondary_supported_suffixes

    def close(self) -> None:
        """Release the transport resources retained by both wrapped providers.

        A queue worker builds one pipeline per claimed batch. The pipeline closes
        the OCR provider it holds, which is this composition, so without this the
        wrapped providers' connection pools and database sessions are never
        reached and accumulate for the worker's lifetime.
        """

        failures: list[Exception] = []
        for provider in (self.primary, self.secondary):
            close = getattr(provider, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ExceptionGroup("Failed to close fallback OCR providers", failures)

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        """Return primary output when useful, otherwise parse with the secondary.

        Primary parser failures are deliberately eligible for fallback, and a
        secondary failure falls back to whatever text the primary read. Exception
        text is not copied into metadata because provider errors may contain local
        paths or remote service details; the exception class is enough to explain
        why selection changed.
        """

        try:
            primary = self.primary.parse(file, options)
        except Exception as error:
            if not self._secondary_supports(file.path):
                raise
            return self._secondary(
                file,
                options,
                reason="primary_error",
                primary_error_type=type(error).__name__,
            )
        page_character_counts = [len(page.raw_text.strip()) for page in primary.pages]
        character_count = sum(page_character_counts)
        page_count = max(1, len(primary.pages))
        required = self.min_characters_per_page * page_count
        sparse_page_count = sum(
            count < self.min_characters_per_page for count in page_character_counts
        )
        sparse_page_ratio = sparse_page_count / page_count
        unbroken_text_ratio = _unbroken_latin_text_ratio(primary)
        fragmented_text_ratio = _fragmented_latin_text_ratio(primary)
        unspaced_page_ratio = _unspaced_latin_page_ratio(primary)
        # A parser that knows where its text sat on the page reports how much of
        # the document interleaves two columns; one that does not reports nothing.
        interleaved_column_page_ratio = float(
            primary.provider_metadata.get("interleaved_column_page_ratio") or 0.0
        )
        density_sufficient = (
            character_count >= required and sparse_page_ratio <= self.max_sparse_page_ratio
        )
        text_coherent = (
            unbroken_text_ratio <= self.max_unbroken_text_ratio
            and fragmented_text_ratio <= self.max_fragmented_text_ratio
            and unspaced_page_ratio <= self.max_unspaced_page_ratio
            and interleaved_column_page_ratio <= self.max_interleaved_column_page_ratio
        )
        if density_sufficient and text_coherent:
            return self._with_selection(
                primary,
                selected=self.primary_name,
                reason="primary_text_sufficient",
                primary_character_count=character_count,
                required_character_count=required,
                sparse_page_count=sparse_page_count,
                sparse_page_ratio=sparse_page_ratio,
                unbroken_text_ratio=unbroken_text_ratio,
                fragmented_text_ratio=fragmented_text_ratio,
                unspaced_page_ratio=unspaced_page_ratio,
                interleaved_column_page_ratio=interleaved_column_page_ratio,
            )
        if not self._secondary_supports(file.path):
            return self._with_selection(
                primary,
                selected=self.primary_name,
                reason="secondary_file_type_unsupported",
                primary_character_count=character_count,
                required_character_count=required,
                sparse_page_count=sparse_page_count,
                sparse_page_ratio=sparse_page_ratio,
                unbroken_text_ratio=unbroken_text_ratio,
                fragmented_text_ratio=fragmented_text_ratio,
                unspaced_page_ratio=unspaced_page_ratio,
                interleaved_column_page_ratio=interleaved_column_page_ratio,
            )
        details = {
            "primary_character_count": character_count,
            "required_character_count": required,
            "sparse_page_count": sparse_page_count,
            "sparse_page_ratio": sparse_page_ratio,
            "unbroken_text_ratio": unbroken_text_ratio,
            "fragmented_text_ratio": fragmented_text_ratio,
            "unspaced_page_ratio": unspaced_page_ratio,
            "interleaved_column_page_ratio": interleaved_column_page_ratio,
        }
        try:
            return self._secondary(
                file,
                options,
                reason=(
                    "primary_text_insufficient"
                    if not density_sufficient
                    else "primary_text_malformed"
                ),
                **details,
            )
        except NoTextReadError:
            # The primary read the file to the end and found nothing, and so
            # did the parser that reads pixels. An empty spreadsheet or a
            # blank page is an empty document; it is recorded as one rather
            # than stopping a corpus on it.
            return self._with_selection(
                primary,
                selected=self.primary_name,
                reason="no_text_in_either",
                **details,
            )
        except Exception as error:
            # The secondary refused a file the primary did read, if poorly: a
            # parsing service that rejects the file's type or fails on it. The
            # primary's text is kept, so the document is read with what there
            # is instead of being lost; the refusal stays in the provenance.
            # Rejected credentials refuse every file and so fail the run.
            if character_count == 0 or is_systemic_provider_error(error):
                raise
            return self._with_selection(
                primary,
                selected=self.primary_name,
                reason="secondary_error",
                secondary_error_type=type(error).__name__,
                **details,
            )

    def _secondary(
        self,
        file: InputFile,
        options: OcrOptions,
        *,
        reason: str,
        **details: object,
    ) -> ParsedDocument:
        """Run and annotate the configured secondary provider after primary rejection."""

        parsed = self.secondary.parse(file, options)
        return self._with_selection(
            parsed,
            selected=self.secondary_name,
            reason=reason,
            **details,
        )

    def _with_selection(
        self,
        parsed: ParsedDocument,
        *,
        selected: str,
        reason: str,
        **details: object,
    ) -> ParsedDocument:
        """Attach non-secret routing provenance without mutating provider output."""

        metadata = {
            **parsed.provider_metadata,
            "fallback_ocr": {
                "primary": self.primary_name,
                "secondary": self.secondary_name,
                "selected": selected,
                "reason": reason,
                **details,
            },
        }
        return parsed.model_copy(update={"provider_metadata": metadata})

    def _secondary_supports(self, path: Path) -> bool:
        """Avoid routing a valid primary result to an incompatible OCR adapter."""

        return (
            self.secondary_supported_suffixes is None
            or path.suffix.lower() in self.secondary_supported_suffixes
        )


def _unbroken_latin_text_ratio(parsed: ParsedDocument) -> float:
    """Measure Latin text likely concatenated by a malformed PDF text layer.

    Character density alone cannot detect PDFs whose embedded text contains many
    words but omits nearly every separator. The ratio counts ASCII letters inside
    unusually long non-whitespace tokens and divides by all ASCII letters. URLs,
    formulas, and identifiers are too small a share of normal prose to cross the
    conservative default threshold. Non-Latin scripts are excluded so languages
    that do not conventionally separate words with spaces remain eligible for the
    lightweight primary parser.
    """

    text = "\n".join(page.raw_text for page in parsed.pages)
    latin_character_count = sum(character.isascii() and character.isalpha() for character in text)
    if latin_character_count == 0:
        return 0.0
    unbroken_character_count = 0
    for token in re.findall(r"\S+", text):
        token_latin_count = sum(character.isascii() and character.isalpha() for character in token)
        if token_latin_count >= _UNBROKEN_LATIN_TOKEN_LENGTH:
            unbroken_character_count += token_latin_count
    return unbroken_character_count / latin_character_count


def _fragmented_latin_text_ratio(parsed: ParsedDocument) -> float:
    """Measure prose words split into implausible one-letter fragments.

    Some legacy PDF encodings preserve abundant characters while inserting spaces
    inside nearly every word, producing text such as ``multiplicativ e``. Character
    density and long-token checks both accept that output even though exact entity
    and relation surfaces become unrecoverable. Words are Latin-script letter runs,
    accented letters included, so German, Danish, or French prose reads as whole
    words. Inline and display LaTeX are removed before measuring because isolated
    mathematical variables are legitimate. The conservative default threshold
    tolerates occasional prose variables and OCR slips while routing systematically
    fragmented text to the secondary provider.
    """

    text = "\n".join(page.raw_text for page in parsed.pages)
    prose = _LATEX_MATH_RE.sub(" ", text)
    # An accented letter or an apostrophe stays inside its word: "Größe" and
    # "l'eau" are one word each, not a fragment beside a lone letter.
    words = [
        letters
        for letters in (word.translate(_APOSTROPHES) for word in _WORD_RE.findall(prose))
        if all(map(_is_latin_letter, letters))
    ]
    latin_character_count = sum(len(word) for word in words)
    if latin_character_count == 0:
        return 0.0
    fragmented_character_count = 0
    for left, right in zip(words, words[1:], strict=False):
        single, fragment = (left, right) if len(left) == 1 else (right, left)
        if (
            len(single) != 1
            or len(fragment) < _MIN_FRAGMENT_LENGTH
            or single != single.lower()
            or single.casefold() in _LEGITIMATE_SINGLE_LETTER_WORDS
        ):
            continue
        # Count only the multi-letter fragment. A genuine standalone variable or
        # label must not itself inflate the malformed-text score.
        fragmented_character_count += len(fragment)
    return fragmented_character_count / latin_character_count


def _unspaced_latin_page_ratio(parsed: ParsedDocument) -> float:
    """Measure the share of pages whose text layer lost most of its word spacing.

    Some PDF text layers drop the spaces inside many lines while keeping others,
    producing runs such as ``theresultsweremeasuredtwiceinthelab`` that are too
    short for the unbroken-token check yet hide every short name inside a longer
    word. A page counts as unspaced when a fifth of its ASCII letters sit in
    letter runs of at least 30 letters. Runs end at digits, hyphens, brackets,
    and other punctuation, so systematic chemical names, identifiers, URLs, and
    encoded data rarely form runs that long; the rare compound word or chemical
    name that does is a small share of a page. All-capital runs (codes, peptide
    sequences) and runs over a nucleotide-sized alphabet are not prose and are
    ignored. Pages with too few letters to measure are left out of the ratio.
    """

    measured_page_count = 0
    unspaced_page_count = 0
    for page in parsed.pages:
        runs = _LATIN_LETTER_RUN_RE.findall(page.raw_text)
        letter_count = sum(len(run) for run in runs)
        if letter_count < _SPACING_MEASURED_PAGE_LETTERS:
            continue
        measured_page_count += 1
        unspaced_letter_count = sum(len(run) for run in runs if _is_unspaced_prose_run(run))
        if unspaced_letter_count >= _UNSPACED_PAGE_LETTER_SHARE * letter_count:
            unspaced_page_count += 1
    if measured_page_count == 0:
        return 0.0
    return unspaced_page_count / measured_page_count


def _is_unspaced_prose_run(run: str) -> bool:
    """Tell a run of several words written without spaces from a long token."""

    return (
        len(run) >= _UNSPACED_RUN_LENGTH
        and run != run.upper()
        and len(set(run.casefold())) >= _UNSPACED_RUN_MIN_DISTINCT_LETTERS
    )


def _is_latin_letter(character: str) -> bool:
    """Whether a letter belongs to the Latin script, with or without accents."""

    return character.isascii() or unicodedata.name(character, "").startswith("LATIN ")
