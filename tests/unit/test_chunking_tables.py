"""Chunks keep table and chart rows whole and carry a split table's labels."""

from __future__ import annotations

from kg_processor.application.chunking import chunk_document
from kg_processor.domain.documents import LayoutBlock, ParsedDocument, ParsedPage
from kg_processor.domain.tables import ChartData, ChartSeries, TableData, render_chart, render_table


def _document(text: str, blocks: list[LayoutBlock]) -> ParsedDocument:
    return ParsedDocument(
        file_id="file_1",
        checksum="sha",
        source_uri="file:///report.pdf",
        mime_type="application/pdf",
        pages=[ParsedPage(page_number=1, markdown=text, raw_text=text, blocks=blocks)],
    )


def _table_page(rows: int) -> tuple[str, LayoutBlock, TableData]:
    table = TableData(
        caption="Yield by plot",
        header_rows=1,
        rows=[["Plot", "Yield kg", "Moisture"]]
        + [[f"Plot {index}", str(100 + index), f"{index % 7}.5"] for index in range(rows)],
    )
    intro = "# Field trial\nThe trial compared plots across one season.\n\n"
    rendered = render_table(table)
    text = f"{intro}{rendered}\n\nThe drier plots yielded less."
    start = len(intro)
    block = LayoutBlock(
        id="block_table",
        page_number=1,
        kind="table",
        text=rendered,
        metadata={
            "table": table.model_dump(mode="json"),
            "start_offset": start,
            "end_offset": start + len(rendered),
        },
    )
    return text, block, table


def test_no_chunk_boundary_falls_inside_a_table_row() -> None:
    text, block, _table = _table_page(rows=60)
    chunks = chunk_document(_document(text, [block]), "graph", 40, 8)

    region_start = block.metadata["start_offset"]
    region_end = block.metadata["end_offset"]
    for chunk in chunks:
        assert text[chunk.start_offset : chunk.end_offset] == chunk.content
        if region_start < chunk.end_offset < region_end:
            assert text[chunk.end_offset] == "\n", chunk.content[-40:]
        if region_start < chunk.start_offset < region_end:
            assert text[chunk.start_offset - 1] == "\n", chunk.content[:40]


def test_chunks_cover_the_whole_page_in_order() -> None:
    text, block, _table = _table_page(rows=60)
    chunks = chunk_document(_document(text, [block]), "graph", 40, 8)

    covered = 0
    for chunk in chunks:
        assert chunk.start_offset <= covered or covered == 0
        covered = max(covered, chunk.end_offset)
    assert covered == len(text)


def test_a_chunk_starting_inside_a_table_carries_its_caption_and_columns() -> None:
    text, block, _table = _table_page(rows=60)
    chunks = chunk_document(_document(text, [block]), "graph", 40, 8)

    inside = [
        chunk
        for chunk in chunks
        if block.metadata["start_offset"] < chunk.start_offset < block.metadata["end_offset"]
    ]
    assert inside
    for chunk in inside:
        assert chunk.section_path == [
            "Field trial",
            "Table: Yield by plot (columns: Plot | Yield kg | Moisture)",
        ]
    assert chunks[0].section_path == ["Field trial"]


def test_a_chart_continuation_names_the_chart_and_its_series() -> None:
    chart = ChartData(
        chart_type="line",
        title="Monthly output",
        categories=[f"Month {index}" for index in range(80)],
        series=[ChartSeries(name="Line A", values=[float(index) for index in range(80)])],
    )
    rendered = render_chart(chart)
    block = LayoutBlock(
        id="block_chart",
        page_number=1,
        kind="chart",
        text=rendered,
        metadata={
            "chart": chart.model_dump(mode="json"),
            "start_offset": 0,
            "end_offset": len(rendered),
        },
    )
    chunks = chunk_document(_document(rendered, [block]), "graph", 30, 5)

    assert len(chunks) > 1
    assert chunks[1].section_path == ["Chart: Monthly output (columns: Category | Line A)"]


def test_blocks_whose_offsets_do_not_match_their_text_are_not_treated_as_tables() -> None:
    text, block, _table = _table_page(rows=60)
    misplaced = block.model_copy(
        update={"metadata": {**block.metadata, "start_offset": 0, "end_offset": 20}}
    )
    plain = chunk_document(_document(text, []), "graph", 40, 8)
    chunks = chunk_document(_document(text, [misplaced]), "graph", 40, 8)

    assert [chunk.content for chunk in chunks] == [chunk.content for chunk in plain]


def test_pages_without_tables_split_exactly_as_before() -> None:
    words = " ".join(f"word{index}" for index in range(1000))
    chunks = chunk_document(_document(words, []), "graph", 100, 10)

    assert [chunk.token_count for chunk in chunks[:-1]] == [100] * (len(chunks) - 1)
    assert [chunk.start_offset for chunk in chunks[:3]] == [
        0,
        words.index("word90 "),
        words.index("word180 "),
    ]
