from __future__ import annotations

from kg_processor.application.extraction_grounding import (
    contains_surface_groups,
    delimited_row_span,
    find_quote,
    ground_evidence,
    ground_evidence_ocr_tolerant,
    name_words_in_quote,
    ocr_tolerant_surface_occurs,
    ocr_tolerant_surface_spans,
    row_names,
    sentence_spans,
    surface_occurs,
)


def test_surface_occurs_requires_complete_tokens() -> None:
    """Prevent substring matches from grounding place names inside adjectival tokens.

    Complete token boundaries are part of the evidence contract.
    """

    source = "Greek wrestling and Indian kushti influenced regional practice."

    assert not surface_occurs(source, ["Greece"])
    assert not surface_occurs(source, ["India"])
    assert surface_occurs(source, ["Greek"])
    assert surface_occurs(source, ["Indian kushti"])


def test_surface_occurs_tolerates_space_and_hyphen_variants() -> None:
    """Allow harmless spacing and hyphen variation while preserving complete-token matching.

    OCR normalization should not lose genuine mentions.
    """

    source = "Brazilian Jiu-Jitsu developed in Brazil."

    assert surface_occurs(source, ["Brazilian Jiu Jitsu"])


def test_surface_occurs_accepts_attached_author_footnote_markers() -> None:
    """Ground multi-token byline names without weakening single-token boundaries."""

    source = "David Silver1*, Aja Huang1* and the Model2 baseline."

    assert surface_occurs(source, ["David Silver"])
    assert surface_occurs(source, ["Aja Huang"])
    assert not surface_occurs(source, ["Model"])


def test_surface_occurs_repairs_visual_line_break_hyphens() -> None:
    """Ground dehyphenated model surfaces against exact offset-preserving PDF text."""

    source = "stability with respect to small per-\nturbations to the inputs"

    assert surface_occurs(source, ["small perturbations to the inputs"])
    grounded = ground_evidence(
        source,
        "stability with respect to small perturbations to the inputs",
        [["small perturbations to the inputs"]],
    )
    assert grounded is not None
    assert grounded.quote == source
    assert grounded.start_offset == 0
    assert grounded.end_offset == len(source)


def test_surface_occurs_repairs_one_first_letter_ocr_split() -> None:
    """Ground a complete title despite one PDF extraction token split."""

    source = "ADAM: A M ETHOD FOR STOCHASTIC OPTIMIZATION"

    assert surface_occurs(source, ["Adam: A Method for Stochastic Optimization"])
    assert not surface_occurs(source, ["Adam: A Method for Stochastic Objectives"])


def test_ground_evidence_rejects_substring_only_endpoint() -> None:
    """Reject relation evidence when one required endpoint exists only as a substring."""

    source = "Indian wrestling developed through regional schools."

    grounded = ground_evidence(
        source,
        source,
        [["wrestling"], ["India"]],
    )

    assert grounded is None


def test_ground_evidence_requires_distinct_non_overlapping_endpoints() -> None:
    """Do not ground a shorter target inside the source entity's sole mention."""

    source = "Stochastic gradient descent is an optimization method."

    grounded = ground_evidence(
        source,
        source,
        [["stochastic gradient descent"], ["gradient descent"]],
    )

    assert grounded is None


def test_ground_evidence_repairs_nested_endpoint_to_distinct_comparison() -> None:
    """Select the sentence where both nested names have independent mentions."""

    unsupported = "Stochastic gradient descent is an optimization method."
    comparison = "Stochastic gradient descent outperforms gradient descent."
    source = f"{unsupported} {comparison}"

    grounded = ground_evidence(
        source,
        unsupported,
        [["stochastic gradient descent"], ["gradient descent"]],
    )

    assert grounded is not None
    assert grounded.quote == comparison
    assert grounded.repair == "supporting_sentence"


def test_a_quote_given_in_pieces_grounds_on_the_stretch_that_holds_them() -> None:
    """Ground a layout fact quoted as header and field joined by an ellipsis."""

    source = (
        "Certificate of Analysis\nSodium Benzoate Granular\nReagent Grade Chemical\n"
        "Manufactured by a supplier\nMaterial No.: 8812-04\n"
        "Batch No.: 47K2093315\nAppearance: white powder"
    )

    grounded = ground_evidence(
        source,
        "Batch No.: 47K2093315 ... Sodium Benzoate Granular",
        [["Sodium Benzoate Granular"], ["47K2093315"]],
    )

    assert grounded is not None
    assert grounded.repair == "quote_pieces"
    assert grounded.quote == source[grounded.start_offset : grounded.end_offset]
    assert grounded.quote.startswith("Sodium Benzoate Granular")
    assert grounded.quote.endswith("47K2093315")


def test_every_piece_of_a_pieced_quote_must_occur_exactly() -> None:
    """Refuse a pieced quote with a paraphrased part."""

    source = "Sodium Benzoate Granular\n" + "filler line\n" * 5 + "Batch No.: 47K2093315"

    assert (
        ground_evidence(
            source,
            "Batch number 47K2093315 ... Sodium Benzoate Granular",
            [["Sodium Benzoate Granular"], ["47K2093315"]],
        )
        is None
    )


def test_letters_and_digits_of_a_name_match_with_or_without_a_separator() -> None:
    """Treat GG918, GG-918 and GG 918 as one name, but not Model inside Model2."""

    assert surface_occurs("the potent GG-918 was developed", ["GG918"])
    assert surface_occurs("the potent GG918 was developed", ["GG-918"])
    assert surface_occurs("Span 60 and butanol", ["Span60"])
    assert not surface_occurs("the Model2 baseline", ["Model"])


def test_sentences_keep_decimals_together_and_split_table_rows() -> None:
    """A decimal is not a sentence end, and each table row is its own sentence."""

    table = (
        "<table><tr><td>Hardness</td><td>41.5 - 44.5 %</td></tr>"
        "<tr><td>Gloss</td><td>7.3 %</td></tr></table>"
    )

    assert [span.quote for span in sentence_spans("Limit is 0.10 %. Result conforms.")] == [
        "Limit is 0.10 %.",
        "Result conforms.",
    ]
    rows = [span.quote for span in sentence_spans(table)]
    assert rows[0].endswith("41.5 - 44.5 %</td></tr>")
    assert rows[1].startswith("<tr><td>Gloss")


def test_a_quote_reads_across_table_cells_as_prose() -> None:
    """A parser writes a table as HTML; a faithful quote of a row reads its cells as text."""

    source = (
        "<table><tr><td>Item label:</td><td>NORTHWIND BASE QZ-7-(TA)</td>"
        "<td>Checked on:</td><td>02.05.2019</td></tr></table>"
    )

    span = find_quote(source, "Item label: NORTHWIND BASE QZ-7-(TA)")

    assert span is not None
    assert span.quote == source[span.start_offset : span.end_offset]
    assert span.quote == "Item label:</td><td>NORTHWIND BASE QZ-7-(TA)"
    # Cell bars read the same way, and words still have to agree.
    assert find_quote("| Batch No. | 0204517739 |", "Batch No. 0204517739") is not None
    assert find_quote(source, "Item label: NORTHWIND BASE QZ-7-(TB)") is None


def test_a_name_matches_its_plural_and_genitive_but_not_another_word() -> None:
    assert surface_occurs(
        "Denne test kan udføres af Acme Pharmas ventilationsafdeling.", ["Acme Pharma"]
    )
    assert surface_occurs("The tablets were coated.", ["tablet"])
    assert surface_occurs("Two processes ran.", ["process"])
    assert not surface_occurs("The Indian team won.", ["India"])
    assert not surface_occurs("Model GG91 differs.", ["GG9"])


def test_an_underscore_separates_the_words_of_a_file_name() -> None:
    name = "GEA Small-scale-solid-dosage_Gral PMA Fluidbed_2015-06-EN.pdf"

    assert surface_occurs(name, ["Gral PMA Fluidbed"])
    assert surface_occurs("Funded by GRANT_12-3-2020-0117.", ["GRANT"])


def test_an_underscore_name_matches_as_written_or_spaced() -> None:
    assert surface_occurs("Can anything from DOC_731902 be reused?", ["DOC_731902"])
    assert surface_occurs("Can anything from DOC_731902 be reused?", ["DOC 731902"])


def test_a_quote_matches_text_a_parser_left_html_escaped() -> None:
    source = "<td><p>O&#x27;Neill mixer</p></td><td>pump &amp; valve control</td>"

    coater = find_quote(source, "O'Neill mixer")
    control = find_quote(source, "pump & valve control")

    assert coater is not None and coater.quote == "O&#x27;Neill mixer"
    assert control is not None and control.quote == "pump &amp; valve control"


def test_a_quote_may_keep_or_drop_a_line_break_hyphen() -> None:
    source = "Both products significantly improved skin hydra-\ntion, reduced redness."

    kept = find_quote(source, "significantly improved skin hydra-tion")
    dropped = find_quote(source, "significantly improved skin hydration")

    assert kept is not None and kept.quote == "significantly improved skin hydra-\ntion"
    assert dropped is not None and dropped.quote == kept.quote


def test_a_name_containing_the_other_endpoint_grounds_only_when_allowed() -> None:
    source = "The 10% urea foam was applied twice daily."
    groups = [["10% urea foam"], ["foam"]]

    assert ground_evidence(source, source, groups) is None
    nested = ground_evidence(source, source, groups, allow_nested=True)
    assert nested is not None and nested.quote == source


def test_a_rephrased_name_keeps_every_word_of_its_quote() -> None:
    assert name_words_in_quote("pH measurement", "The pH of the syrups was measured")
    assert name_words_in_quote(
        "scintigraphic examinations", "In scintigraphic and colonoscopic examinations"
    )
    assert not name_words_in_quote("pore size", "interlinked pores spanning 20 to 80 µm")
    assert not name_words_in_quote("pH value", "The pH of the syrups was measured")


def test_a_name_of_several_parts_may_touch_digits_ocr_glued_to_it() -> None:
    assert surface_occurs("15548-01SmatProtect splinger shield", ["SmatProtect splinger shield"])
    assert not surface_occurs("ItemsSmart shield", ["Smart shield"])
    assert not surface_occurs("Model2 is new", ["Model"])


def test_a_hyphen_after_a_stray_space_ends_a_line_but_a_spaced_dash_stays() -> None:
    assert surface_occurs("width, and diame -\nter / length", ["diameter"])
    assert not surface_occurs("Foam - a review", ["Foama"])


def _ocr_found(source: str, name: str) -> list[str]:
    return [source[start:end] for start, end in ocr_tolerant_surface_spans(source, [name])]


def test_the_ocr_tolerant_view_finds_names_in_text_that_lost_its_spaces() -> None:
    """Words OCR ran together match the name written with spaces, as the source's text."""

    address = "Supplier: NorthwindChemicalsLtdHarbourStreet12,Springfield"

    assert not surface_occurs(address, ["Northwind Chemicals Ltd"])
    assert _ocr_found(address, "Northwind Chemicals Ltd") == ["NorthwindChemicalsLtd"]
    assert _ocr_found("the moisturecontent of the powder", "moisture content") == [
        "moisturecontent"
    ]
    assert _ocr_found("Jane Q. Doe −ExampleLabsInc.", "Example Labs Inc.") == ["ExampleLabsInc"]
    assert _ocr_found("RulesofEuropeandtheUnitedStates", "United States") == ["UnitedStates"]
    # Spaces the name does not have may be in the text too.
    assert _ocr_found("the Stainless steel housing", "Stainlesssteel") == ["Stainless steel"]


def test_the_ocr_tolerant_view_reads_known_misreadings_as_the_same_character() -> None:
    """One misread character in every eight counts, from a fixed set of OCR confusions."""

    assert _ocr_found("ORION 432O4 Technical Data Sheet", "Orion 43204") == ["ORION 432O4"]
    assert _ocr_found("Type Il housing", "Type II") == ["Type Il"]
    assert _ocr_found("the Turb1Std range", "TurblStd") == ["Turb1Std"]
    assert _ocr_found("Aluminium Oxidc", "Aluminium Oxide") == ["Aluminium Oxidc"]
    assert _ocr_found("Alurninium housing", "Aluminium") == ["Alurninium"]
    assert _ocr_found("Staln1ess steel frame", "Stainless steel") == ["Staln1ess steel"]
    # Three misreadings are one too many for fourteen characters.
    assert _ocr_found("Staln1ess stcel frame", "Stainless steel") == []
    assert _ocr_found("Stainless stee1 frame", "Stainless steel") == ["Stainless stee1"]


def test_the_ocr_tolerant_view_keeps_short_names_out_of_longer_words() -> None:
    """Guards: length minimums, word edges, whole numbers, and codes as written."""

    # A one-word name never begins or ends inside another word.
    assert _ocr_found("Carbonate is used", "Carbon") == []
    # A name's space where the text has one too is no sign of glued text.
    assert _ocr_found("the latest method", "Test Method") == []
    assert _ocr_found("ItemsSmart shield", "Smart shield") == []
    # A number cut short is another number.
    assert _ocr_found("Batch123456 released", "Batch 12345") == []
    # A letter for a digit only inside a number, not where a code's parts meet.
    assert _ocr_found("Lot s123 released", "Lot 5123") == []
    assert _ocr_found("Part AB5123 released", "Part ABS123") == []
    # A pair of letters read as one only in a long name: "modem" is not "modern".
    assert _ocr_found("a modern tool", "modem") == []
    # Too short to be misread, or to lose a space.
    assert _ocr_found("Ce1l culture", "Cell") == []
    assert _ocr_found("BigOx tools", "Big Ox") == []
    assert not ocr_tolerant_surface_occurs("Model2 is new", ["Model"])
    assert not ocr_tolerant_surface_occurs("close doses", ["dose"])


def test_ocr_tolerant_grounding_is_the_last_tier_and_keeps_the_source_text() -> None:
    """Text that grounds as written is left to ``ground_evidence``; the rest is marked."""

    source = "Supplied by NorthwindChemicalsLtdtoAcmeWorkshop on 2 May."
    quote = "Supplied by Northwind Chemicals Ltd to Acme Workshop"
    groups = [["Northwind Chemicals Ltd"], ["Acme Workshop"]]

    assert ground_evidence(source, quote, groups) is None
    grounded = ground_evidence_ocr_tolerant(source, quote, groups)

    assert grounded is not None
    assert grounded.ocr_tolerant is True
    assert grounded.quote == source[grounded.start_offset : grounded.end_offset]
    assert grounded.quote == "Supplied by NorthwindChemicalsLtdtoAcmeWorkshop"
    assert not contains_surface_groups(grounded.quote, groups)
    assert contains_surface_groups(grounded.quote, groups, ocr_tolerant=True)
    # A quote the model wrote as the text reads grounds unmarked.
    written = ground_evidence("Supplied by Northwind Chemicals Ltd.", quote[:35], groups[:1])
    assert written is not None and written.ocr_tolerant is False


def test_an_ocr_tolerant_name_must_still_be_in_the_text() -> None:
    source = "The housing is made of Stainless steel."

    assert ground_evidence_ocr_tolerant(source, "made of aluminium", [["Aluminium"]]) is None
    supporting = ground_evidence_ocr_tolerant(source, "a paraphrase", [["Stainlesssteel"]])
    assert supporting is not None and supporting.quote == source


def test_a_change_of_case_separates_values_run_together_in_a_table_cell() -> None:
    """An export that lists several values in one cell with nothing between them.

    On a table row the capitals still mark where each value begins; in prose a
    change of case is inside a name, so nothing is read there.
    """

    row = "ITM-4471 | Lactose | Acme Nordic ApSNorthwindGlobex SE | 63-42-3"

    def found(source: str, name: str) -> list[str]:
        return [source[start:end] for start, end in ocr_tolerant_surface_spans(source, [name])]

    assert found(row, "Northwind") == ["Northwind"]
    assert found(row, "Acme Nordic ApS") == ["Acme Nordic ApS"]
    assert found(row, "Globex SE") == ["Globex SE"]
    assert found("Cetyl alcoholNorthwindGlobex CA\tx", "Northwind") == ["Northwind"]
    assert found("We drafted it in PowerPoint and ApSNorthwind prose.", "Point") == []
    assert found("We drafted it in PowerPoint and ApSNorthwind prose.", "Northwind") == []
    assert found(row, "wind") == []


def _row(source: str, quote: str) -> str | None:
    start = source.index(quote)
    span = delimited_row_span(source, start, start + len(quote))
    return source[span[0] : span[1]] if span is not None else None


def test_a_delimited_row_is_a_line_of_three_cells_or_more() -> None:
    """Tildes, cell bars and tabs divide cells; HTML rows and semicolons do not count."""

    delimited = "Name~Route; Form~Code\nZorvalide~ORAL; TABLET~ZV-12~5~mg\nNext line"
    assert _row(delimited, "ORAL; TABLET") == "Zorvalide~ORAL; TABLET~ZV-12~5~mg"
    assert _row("| Zorvalide | Oral |\n| Kestrine | Topical |", "Kestrine") == (
        "| Kestrine | Topical |"
    )
    assert _row("Zorvalide\tOral\tTablet", "Oral") == "Zorvalide\tOral\tTablet"
    # Two cells, a stretch across lines, an HTML row, and prose.
    assert _row("Zorvalide | Oral", "Oral") is None
    assert _row(delimited, "Code\nZorvalide") is None
    assert _row("<tr><td>Zorvalide</td><td>Oral</td><td>5</td></tr>", "Oral") is None
    assert _row("Each tablet holds ~5 mg of Zorvalide.", "Zorvalide") is None
    assert _row("A tablet comprises: Zorvalide; a binder; a filler.", "binder") is None


def test_a_row_names_a_thing_by_every_word_of_its_name_without_spaces_or_punctuation() -> None:
    """Hyphens, brackets, dots and case may differ; a near-identical cell is another thing."""

    assert row_names("TRIZORVANE CITRATE DIHYDRATE~ORAL~5", ["tri-Zorvane citrate dihydrate"])
    assert row_names("HYDROGENATED KESTREL OIL~TOPICAL~2", ["(Hydrogenated) Kestrel Oil"])
    assert row_names("Alpha-voranol~Oral~1", [".ALPHA.-VORANOL"])
    # Words of one or two letters are left out; numbers never are.
    assert row_names("Vitamin Q12~Oral~1", ["Vitamin Q 12"])
    assert not row_names("Vitamin Q13~Oral~1", ["Vitamin Q 12"])
    assert not row_names("ZORVANE PHOSPHATE, ANHYDROUS~ORAL~5", ["ZORVANE PHOSPHATE, MONOHYDRATE"])
    assert not row_names("INTRAMUSCULAR; POWDER~ZV-12", ["INTRAVENOUS; POWDER"])
    # An alias names the thing too.
    assert row_names("ZV-12~ORAL~5", ["Zorvalide", "ZV12"])
