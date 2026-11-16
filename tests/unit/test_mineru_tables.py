"""MinerU's HTML tables become structured table blocks with exact offsets."""

from __future__ import annotations

import json
from pathlib import Path

from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.adapters.ocr.mineru_api import _pages_from_result
from kg_processor.adapters.ocr.mineru_common import render_html_tables
from kg_processor.adapters.ocr.mineru_internal import MineruInternalOcrProvider
from kg_processor.domain.documents import InputFile, ParsedPage
from kg_processor.domain.tables import TableData, render_table

TABLE_HTML = (
    "<table><tr><td>Arm</td><td colspan='2'>Score</td></tr>"
    "<tr><td rowspan='2'>Placebo</td><td>1.5</td><td>1.7</td></tr>"
    "<tr><td>1.9</td><td>2.1</td></tr></table>"
)
ROWS = [["Arm", "Score", "Score"], ["Placebo", "1.5", "1.7"], ["Placebo", "1.9", "2.1"]]


def _file(tmp_path: Path) -> InputFile:
    sample = tmp_path / "sample.txt"
    sample.write_text("placeholder", encoding="utf-8")
    return LocalFileSource(sample).list_files()[0]


def _assert_offsets(page: ParsedPage) -> None:
    for block in page.blocks:
        if "start_offset" in block.metadata:
            start, end = block.metadata["start_offset"], block.metadata["end_offset"]
            assert page.markdown[start:end] == block.text


def test_internal_content_list_table_is_rendered_with_its_caption_and_footnote(
    tmp_path: Path,
) -> None:
    file = _file(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "content_list.json").write_text(
        json.dumps(
            [
                {"page_idx": 0, "type": "text", "text": "Results follow."},
                {
                    "page_idx": 0,
                    "type": "table",
                    "table_caption": ["Table 2: Scores"],
                    "table_body": TABLE_HTML,
                    "table_footnote": ["Invented data."],
                },
            ]
        ),
        encoding="utf-8",
    )

    page = MineruInternalOcrProvider()._load_output(file, output, stdout="ok").pages[0]

    table_block = next(block for block in page.blocks if block.kind == "table")
    table = TableData.model_validate(table_block.metadata["table"])
    assert table.rows == ROWS
    assert table.source == "ocr"
    assert table.caption == "Table 2: Scores"
    assert page.markdown == ("Results follow.\n\n" + render_table(table) + "\n\nInvented data.")
    assert [block.kind for block in page.blocks] == ["text", "table", "table_footnote"]
    _assert_offsets(page)


def test_internal_verbatim_markdown_tables_are_rewritten_and_their_blocks_follow(
    tmp_path: Path,
) -> None:
    file = _file(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    markdown = f"# Summary\n\n{TABLE_HTML}\n\nClosing line."
    (output / "pages.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page_number": 1,
                        "markdown": markdown,
                        "blocks": [
                            {"type": "title", "text": "Summary"},
                            {"type": "table", "text": TABLE_HTML},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    page = MineruInternalOcrProvider()._load_output(file, output, stdout="ok").pages[0]

    table = TableData(header_rows=0, rows=ROWS, source="ocr")
    assert page.markdown == f"# Summary\n\n{render_table(table)}\n\nClosing line."
    assert page.raw_text == page.markdown
    block = page.blocks[1]
    assert block.kind == "table"
    assert block.metadata["table"]["rows"] == ROWS
    _assert_offsets(page)
    assert "start_offset" in block.metadata


def test_api_verbatim_markdown_gains_table_blocks(tmp_path: Path) -> None:
    file = _file(tmp_path)

    [page] = _pages_from_result(file, {"md_content": f"Intro\n\n{TABLE_HTML}"})

    assert page.markdown.startswith("Intro\n\n|  |  |  |\n| --- | --- | --- |\n| Arm |")
    assert page.blocks[0].text == page.markdown
    table_block = page.blocks[1]
    assert table_block.kind == "table"
    assert table_block.metadata["table"]["rows"] == ROWS
    _assert_offsets(page)


def test_api_content_list_table_with_header_cells(tmp_path: Path) -> None:
    file = _file(tmp_path)
    html = (
        "<table><thead><tr><th>Site</th><th>n</th></tr></thead>"
        "<tr><td>A</td><td>4</td></tr></table>"
    )
    content_list = [{"type": "table", "table_body": html, "page_idx": 0}]

    [page] = _pages_from_result(file, {"content_list": content_list})

    [block] = page.blocks
    assert block.text == "| Site | n |\n| --- | --- |\n| A | 4 |"
    assert block.metadata["table"]["header_rows"] == 1
    _assert_offsets(page)


def test_nested_and_unreadable_tables_in_markdown() -> None:
    nested = "<table><tr><td>Outer<table><tr><td>Inner</td></tr></table></td></tr></table>"
    text, rendered = render_html_tables(f"a {nested} b <table></table> c")

    assert len(rendered) == 1
    assert rendered[0].table.rows == [["Outer\nInner"]]
    assert text == f"a {render_table(rendered[0].table)} b <table></table> c"
    assert text[rendered[0].start : rendered[0].end] == render_table(rendered[0].table)
