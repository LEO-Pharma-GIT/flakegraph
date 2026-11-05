"""Behavioral tests for provider-neutral OCR fallback routing."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from kg_processor.adapters.ocr.fallback import FallbackOcrProvider
from kg_processor.domain.documents import InputFile, ParsedDocument, ParsedPage
from kg_processor.ports.ocr import NoTextReadError, OcrOptions, OcrProvider


class _StubOcr:
    """Return a configured document or raise a configured error for routing tests."""

    def __init__(
        self,
        document: ParsedDocument | None = None,
        error: Exception | None = None,
    ) -> None:
        self.document = document
        self.error = error
        self.calls = 0
        self.closed = 0

    def parse(self, _file: InputFile, _options: OcrOptions) -> ParsedDocument:
        """Record invocation before returning or raising the configured result."""

        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.document is not None
        return self.document

    def close(self) -> None:
        """Record the release of this provider's retained transport resources."""

        self.closed += 1


def test_fallback_releases_both_wrapped_providers() -> None:
    """A worker builds one pipeline per batch; unreleased pools accumulate.

    The pipeline closes the OCR provider it holds, which is this composition, so
    the wrapped providers are only reached through it.
    """

    primary = _StubOcr(_document("primary", provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))

    _provider(primary, secondary, minimum=80).close()

    assert (primary.closed, secondary.closed) == (1, 1)


def test_fallback_close_tolerates_providers_without_transport_state() -> None:
    """Not every OCR adapter retains resources, and none is required to."""

    class _Closeless:
        def parse(self, _file: InputFile, _options: OcrOptions) -> ParsedDocument:
            raise AssertionError("parse is not exercised by this test")

    _provider(_Closeless(), _Closeless(), minimum=80).close()


def test_fallback_close_reports_every_provider_failure() -> None:
    """One provider's failure must not hide the other's, or skip its release."""

    class _FailingClose(_StubOcr):
        def close(self) -> None:
            super().close()
            raise RuntimeError("connection pool already shut down")

    primary = _FailingClose(_document("primary", provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))

    with pytest.raises(ExceptionGroup):
        _provider(primary, secondary, minimum=80).close()

    assert secondary.closed == 1


def test_fallback_keeps_sufficient_primary_text_without_secondary_call(tmp_path: Path) -> None:
    """Avoid expensive OCR when the primary parser returned useful page text."""

    primary = _StubOcr(_document("useful primary text " * 12, provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))
    provider = _provider(primary, secondary, minimum=80)

    result = provider.parse(_file(tmp_path), OcrOptions())

    assert secondary.calls == 0
    assert result.provider_metadata["fallback_ocr"]["selected"] == "builtin_text"
    assert result.provider_metadata["fallback_ocr"]["reason"] == "primary_text_sufficient"


def test_fallback_routes_sparse_primary_document_to_secondary(tmp_path: Path) -> None:
    """Run real OCR when average page text falls below the configured threshold."""

    primary = _StubOcr(_document("scan", "scan", provider="builtin_text"))
    secondary = _StubOcr(_document("recognized text", provider="mineru_internal"))
    provider = _provider(primary, secondary, minimum=80)

    result = provider.parse(_file(tmp_path), OcrOptions())

    assert secondary.calls == 1
    routing = result.provider_metadata["fallback_ocr"]
    assert routing["selected"] == "mineru_internal"
    assert routing["reason"] == "primary_text_insufficient"
    assert routing["primary_character_count"] == 8
    assert routing["required_character_count"] == 160


def test_fallback_keeps_an_empty_document_when_neither_parser_reads_text(
    tmp_path: Path,
) -> None:
    """An empty spreadsheet is an empty document, not a failed one.

    The native parser read the workbook to the end and found no cells; the
    pixel parser, asked next, read nothing either and said so. Failing here
    would stop a corpus of hundreds of files. The empty primary result is kept
    and the routing says why.
    """

    primary = _StubOcr(_document("", provider="builtin_text"))
    secondary = _StubOcr(error=NoTextReadError("MinerU API returned no text for parts.xlsx"))
    provider = _provider(primary, secondary, minimum=80)

    result = provider.parse(_file(tmp_path, suffix=".xlsx"), OcrOptions())

    assert secondary.calls == 1
    routing = result.provider_metadata["fallback_ocr"]
    assert routing["selected"] == "builtin_text"
    assert routing["reason"] == "no_text_in_either"
    assert routing["primary_character_count"] == 0
    assert [page.raw_text for page in result.pages] == [""]


def test_fallback_still_fails_a_scan_that_neither_parser_reads(tmp_path: Path) -> None:
    """With no primary document to keep, reading nothing stays an error."""

    primary = _StubOcr(error=NoTextReadError("builtin_text extracted no text"))
    secondary = _StubOcr(error=NoTextReadError("MinerU API returned no text"))

    with pytest.raises(NoTextReadError):
        _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())


def test_fallback_keeps_the_primary_text_when_the_secondary_refuses_the_file(
    tmp_path: Path,
) -> None:
    """A parsing service that rejects or fails on a file does not lose the document.

    The primary read a little text; the secondary, asked for more, refused the
    file. The primary's text is kept and the routing records the refusal
    without its message, which may carry service details.
    """

    primary = _StubOcr(_document("Certificate of Analysis", provider="builtin_text"))
    secondary = _StubOcr(error=RuntimeError("service returned HTTP 409 at private host"))

    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    routing = result.provider_metadata["fallback_ocr"]
    assert (routing["selected"], routing["reason"]) == ("builtin_text", "secondary_error")
    assert routing["secondary_error_type"] == "RuntimeError"
    assert "private" not in str(routing)
    assert [page.raw_text for page in result.pages] == ["Certificate of Analysis"]


@pytest.mark.parametrize("status", [401, 403])
def test_fallback_raises_a_secondary_authentication_failure_despite_primary_text(
    tmp_path: Path, status: int
) -> None:
    """Rejected credentials fail every file, so primary text cannot mask them."""

    request = httpx.Request("POST", "https://ocr.invalid/parse")
    response = httpx.Response(status, request=request)
    primary = _StubOcr(_document("Certificate of Analysis", provider="builtin_text"))
    secondary = _StubOcr(
        error=httpx.HTTPStatusError("rejected", request=request, response=response)
    )

    with pytest.raises(httpx.HTTPStatusError):
        _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())


def test_fallback_still_fails_when_the_secondary_errs_and_the_primary_read_nothing(
    tmp_path: Path,
) -> None:
    """With no text to keep, a secondary failure stays the document's failure."""

    primary = _StubOcr(_document("", provider="builtin_text"))
    secondary = _StubOcr(error=RuntimeError("service returned HTTP 400"))

    with pytest.raises(RuntimeError):
        _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())


def test_fallback_routes_primary_error_without_leaking_message(tmp_path: Path) -> None:
    """Keep fallback useful on parser errors while excluding sensitive error text."""

    primary = _StubOcr(error=RuntimeError("private /path and endpoint"))
    secondary = _StubOcr(_document("recognized text", provider="mineru_internal"))
    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    routing = result.provider_metadata["fallback_ocr"]
    assert routing["reason"] == "primary_error"
    assert routing["primary_error_type"] == "RuntimeError"
    assert "private" not in str(routing)


def test_fallback_routes_mixed_document_with_too_many_sparse_pages(tmp_path: Path) -> None:
    """Prevent one dense native page from hiding a scanned or blank companion page."""

    primary = _StubOcr(_document("a" * 500, "", provider="builtin_text"))
    secondary = _StubOcr(_document("recognized text", provider="mineru_internal"))
    provider = _provider(primary, secondary, minimum=80)

    result = provider.parse(_file(tmp_path), OcrOptions())

    routing = result.provider_metadata["fallback_ocr"]
    assert routing["selected"] == "mineru_internal"
    assert routing["sparse_page_count"] == 1
    assert routing["sparse_page_ratio"] == 0.5


def test_fallback_routes_dense_but_concatenated_latin_text(tmp_path: Path) -> None:
    """Reject malformed PDF text layers whose word boundaries have disappeared."""

    malformed = "A" * 120 + " " + "B" * 120
    primary = _StubOcr(_document(malformed, provider="builtin_text"))
    secondary = _StubOcr(_document("recognized prose", provider="mineru_internal"))

    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    routing = result.provider_metadata["fallback_ocr"]
    assert routing["selected"] == "mineru_internal"
    assert routing["reason"] == "primary_text_malformed"
    assert routing["unbroken_text_ratio"] == 1.0


def test_fallback_does_not_apply_latin_spacing_heuristic_to_cjk(tmp_path: Path) -> None:
    """Keep naturally unsegmented non-Latin text on the sufficient primary path."""

    primary = _StubOcr(_document("漢字" * 100, provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))

    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    assert secondary.calls == 0
    assert result.provider_metadata["fallback_ocr"]["selected"] == "builtin_text"


def test_fallback_routes_dense_but_fragmented_prose(tmp_path: Path) -> None:
    """Reject legacy text layers that insert spaces inside most prose words."""

    malformed = "The metho d com bines optimiz ation fea tures. " * 20
    primary = _StubOcr(_document(malformed, provider="builtin_text"))
    secondary = _StubOcr(_document("recognized prose", provider="mineru_internal"))

    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    routing = result.provider_metadata["fallback_ocr"]
    assert routing["selected"] == "mineru_internal"
    assert routing["reason"] == "primary_text_malformed"
    assert routing["fragmented_text_ratio"] > 0.15


def test_fallback_ignores_latex_variables_when_measuring_fragmentation(tmp_path: Path) -> None:
    """Keep coherent prose with dense one-letter mathematical variables."""

    text = "Useful primary prose explains the model. " * 8 + "$ x y z = a b c $ " * 30
    primary = _StubOcr(_document(text, provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))

    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    assert secondary.calls == 0
    assert result.provider_metadata["fallback_ocr"]["selected"] == "builtin_text"


@pytest.mark.parametrize(
    "prose",
    [
        pytest.param(
            "Die Größe der Körner wurde nach dem Mahlen geprüft, und die Lösung blieb über "
            "Nacht bei Raumtemperatur stehen, bevor die Prüfer die Messwerte übertrugen. ",
            id="german",
        ),
        pytest.param(
            "Prøven blev opløst i vand og rystet på bordet, før målingen af størrelsen blev "
            "gentaget på hver æske fra lageret ved Ålborg og Ærø. ",
            id="danish",
        ),
        pytest.param(
            "Le mélange a été chauffé à soixante degrés, puis versé à froid dans les flacons "
            "préparés à l'avance et étiquetés à la main. ",
            id="french",
        ),
    ],
)
def test_fallback_keeps_prose_with_accented_letters(tmp_path: Path, prose: str) -> None:
    """Read accented letters as part of their words, not as one-letter fragments."""

    primary = _StubOcr(_document(prose * 12, provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))

    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    routing = result.provider_metadata["fallback_ocr"]
    assert secondary.calls == 0
    assert routing["selected"] == "builtin_text"
    assert routing["fragmented_text_ratio"] < 0.02


def test_fallback_routes_fragmented_prose_with_accented_letters(tmp_path: Path) -> None:
    """Measure fragments inside accented words like any other fragments."""

    malformed = "Die Grö ß e der Kö r ner wur de geprü f t und ü bertra gen. " * 20
    primary = _StubOcr(_document(malformed, provider="builtin_text"))
    secondary = _StubOcr(_document("recognized prose", provider="mineru_internal"))

    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    routing = result.provider_metadata["fallback_ocr"]
    assert routing["selected"] == "mineru_internal"
    assert routing["fragmented_text_ratio"] > 0.15


_SPACED_PROSE = (
    "The filter cake was washed twice with cold water before drying under reduced "
    "pressure. Each batch was sampled at three points so that the operators could "
    "compare the moisture content across the trays and record the final yield.\n"
)
_UNSPACED_PROSE = (
    "Thefiltercakewaswashedtwicewithcold water before\n"
    "dryingunderreducedpressureovernight. The colour\n"
    "changedfromgreentoyellowwithinanhour and the yield\n"
    "wasrecordedafterweighingeachdriedbatch so that\n"
    "theoperatorscouldcomparethetrays later.\n"
)


def test_fallback_routes_pages_whose_text_layer_lost_its_word_spacing(tmp_path: Path) -> None:
    """Words written together hide short names that grounding cannot split out.

    The runs are too short for the unbroken-token check, which therefore does not
    fire; the unspaced pages alone route the document.
    """

    primary = _StubOcr(
        _document(
            _UNSPACED_PROSE * 3, _UNSPACED_PROSE * 3, _SPACED_PROSE * 3, provider="builtin_text"
        )
    )
    secondary = _StubOcr(_document("recognized prose", provider="mineru_internal"))

    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    routing = result.provider_metadata["fallback_ocr"]
    assert routing["selected"] == "mineru_internal"
    assert routing["reason"] == "primary_text_malformed"
    assert routing["unspaced_page_ratio"] == pytest.approx(2 / 3)
    assert routing["unbroken_text_ratio"] == 0.0


def test_fallback_ignores_a_page_too_short_to_measure_its_spacing(tmp_path: Path) -> None:
    """One unspaced caption on an otherwise empty page says nothing about the layer."""

    primary = _StubOcr(
        _document(
            "Thecolourchangedfromgreentoyellow quickly.", _SPACED_PROSE * 3, provider="builtin_text"
        )
    )
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))

    result = _provider(primary, secondary, minimum=10).parse(_file(tmp_path), OcrOptions())

    assert secondary.calls == 0
    assert result.provider_metadata["fallback_ocr"]["unspaced_page_ratio"] == 0.0


@pytest.mark.parametrize(
    "page",
    [
        pytest.param(_SPACED_PROSE * 4, id="prose"),
        pytest.param(
            "The ester of cyclopropanecarboxylic acid was hydrolysed overnight. "
            "A sample of dimethylaminopropylmethacrylamide was stored in the dark. "
            "The salt 2-(4-isobutylphenyl)propanoic acid and "
            "N,N-dimethyl-4-aminopyridine were weighed into separate vials, and "
            "hexamethylphosphoramide was handled under the fume hood as before.\n" * 3,
            id="chemical-names",
        ),
        pytest.param(
            "Die Zulassungsbehörde prüft jeden Antrag sorgfältig, bevor sie entscheidet. "
            "Für die Arzneimittelzulassung werden die Unterlagen zur Herstellung, zur "
            "Qualitätskontrolle und zur Stabilität bewertet, und die "
            "Donaudampfschifffahrtsgesellschaft bleibt das bekannteste lange Wort. "
            "Laboratoriet bestiller udstyr til sikkerhedsgennemgangen hver uge, og "
            "fremstillingslaboratorierne følger de samme procedurer for rengøring.\n" * 3,
            id="compound-words",
        ),
        pytest.param(
            (
                _SPACED_PROSE + "Read the guide at https://docs.example.org/administrationconsole/"
                "gettingstartedwiththeplatform/index.html before installing, and send "
                "questions to support@example.org. The archive "
                "https://files.example.net/releases/latestinstallerpackage.tar.gz holds "
                "every installer, and the changelog lists the fixes in each release.\n"
            )
            * 3,
            id="urls",
        ),
        pytest.param(
            _SPACED_PROSE * 3
            + "acgtacgttgcatgcaacgtgtacgatcgatgcatgcaacgt\n"
            + "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGD\n",
            id="sequences",
        ),
    ],
)
def test_fallback_keeps_spaced_text_with_long_words_on_the_primary(
    tmp_path: Path, page: str
) -> None:
    """Long chemical names, compounds, URLs and sequences are not lost spacing."""

    primary = _StubOcr(_document(page, page, provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))

    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    assert secondary.calls == 0
    routing = result.provider_metadata["fallback_ocr"]
    assert routing["selected"] == "builtin_text"
    assert routing["unspaced_page_ratio"] == 0.0


def test_fallback_keeps_primary_when_secondary_cannot_parse_file_type(tmp_path: Path) -> None:
    """Let mixed corpora retain sparse native text instead of invoking incompatible OCR."""

    primary = _StubOcr(_document("short", provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))
    provider = _provider(
        primary, secondary, minimum=80, secondary_supported_suffixes=frozenset({".pdf"})
    )

    result = provider.parse(_file(tmp_path, suffix=".txt"), OcrOptions())

    assert secondary.calls == 0
    assert result.provider_metadata["fallback_ocr"]["selected"] == "builtin_text"
    assert result.provider_metadata["fallback_ocr"]["reason"] == "secondary_file_type_unsupported"


def _provider(
    primary: OcrProvider,
    secondary: OcrProvider,
    minimum: int,
    secondary_supported_suffixes: frozenset[str] | None = None,
) -> FallbackOcrProvider:
    """Construct the adapter under test with stable provider identities."""

    return FallbackOcrProvider(
        primary,
        secondary,
        primary_name="builtin_text",
        secondary_name="mineru_internal",
        min_characters_per_page=minimum,
        max_sparse_page_ratio=0.2,
        max_unbroken_text_ratio=0.5,
        max_fragmented_text_ratio=0.15,
        max_unspaced_page_ratio=0.25,
        secondary_supported_suffixes=secondary_supported_suffixes,
    )


def _document(*texts: str, provider: str) -> ParsedDocument:
    """Build a minimal normalized document with one page per text."""

    return ParsedDocument(
        file_id="file-1",
        checksum="abc",
        source_uri="file:///fixture.pdf",
        mime_type="application/pdf",
        pages=[
            ParsedPage(page_number=index, markdown=text, raw_text=text, blocks=[])
            for index, text in enumerate(texts, start=1)
        ],
        provider_metadata={"provider": provider},
    )


def _file(tmp_path: Path, suffix: str = ".pdf") -> InputFile:
    """Create the input-file value required by the OCR port."""

    path = tmp_path / f"fixture{suffix}"
    path.write_bytes(b"%PDF-fixture")
    return InputFile(
        id="file-1",
        source_uri=path.as_uri(),
        path=path,
        checksum="abc",
        mime_type="application/pdf",
        size_bytes=12,
    )


def test_fallback_routes_a_pdf_whose_text_layer_interleaves_two_columns(tmp_path: Path) -> None:
    """Send a two-column page drawn row by row to the parser that reads layout."""

    interleaved = _document("readable but interleaved text " * 12, provider="builtin_text")
    interleaved.provider_metadata["interleaved_column_page_ratio"] = 0.8
    primary = _StubOcr(interleaved)
    secondary = _StubOcr(_document("reading-order text", provider="mineru_internal"))
    provider = FallbackOcrProvider(
        primary,
        secondary,
        primary_name="builtin_text",
        secondary_name="mineru_internal",
        min_characters_per_page=80,
        max_sparse_page_ratio=0.2,
        max_unbroken_text_ratio=0.5,
        max_fragmented_text_ratio=0.15,
        max_unspaced_page_ratio=0.25,
        max_interleaved_column_page_ratio=0.25,
    )

    result = provider.parse(_file(tmp_path), OcrOptions())

    assert secondary.calls == 1
    routing = result.provider_metadata["fallback_ocr"]
    assert routing["reason"] == "primary_text_malformed"
    assert routing["interleaved_column_page_ratio"] == 0.8
