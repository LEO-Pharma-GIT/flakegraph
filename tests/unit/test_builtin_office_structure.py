"""The built-in parser keeps the structure of Office, HTML and PDF documents.

Every fixture here is built in code from invented names and numbers.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter

from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.adapters.ocr import builtin_text
from kg_processor.adapters.ocr.builtin_text import BuiltinTextOcrProvider, read_image_bytes
from kg_processor.domain.documents import InputFile, ParsedAsset, ParsedDocument, ParsedPage
from kg_processor.domain.tables import ChartData, TableData, render_chart, render_table
from kg_processor.ports.ocr import OcrOptions

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
XDR = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = f'xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}" xmlns:c="{C}"'

CHART_XML = (
    f'<c:chartSpace xmlns:c="{C}" xmlns:a="{A}"><c:chart>'
    "<c:title><c:tx><c:rich><a:p><a:r><a:t>Units by region</a:t></a:r></a:p></c:rich></c:tx>"
    '</c:title><c:plotArea><c:barChart><c:barDir val="col"/>'
    "<c:ser><c:tx><c:v>2024</c:v></c:tx>"
    '<c:cat><c:strRef><c:strCache><c:ptCount val="2"/><c:pt idx="0"><c:v>East</c:v></c:pt>'
    '<c:pt idx="1"><c:v>West</c:v></c:pt></c:strCache></c:strRef></c:cat>'
    '<c:val><c:numRef><c:numCache><c:formatCode>General</c:formatCode><c:ptCount val="2"/>'
    '<c:pt idx="0"><c:v>40</c:v></c:pt><c:pt idx="1"><c:v>55</c:v></c:pt>'
    "</c:numCache></c:numRef></c:val></c:ser></c:barChart></c:plotArea></c:chart></c:chartSpace>"
)


def png_bytes(width: int = 3, height: int = 2) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def rels(*entries: tuple[str, str, str]) -> str:
    body = "".join(
        f'<Relationship Id="{rid}" Type="{REL}/{kind}" Target="{target}"/>'
        for rid, kind, target in entries
    )
    return f'<Relationships xmlns="{PKG}">{body}</Relationships>'


def sp(text_paragraphs: str, *, x: int | None, y: int | None, ph: str = "", name: str = "s") -> str:
    xfrm = (
        f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="1000" cy="1000"/></a:xfrm>'
        if x is not None
        else ""
    )
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="2" name="{name}"/><p:cNvSpPr/><p:nvPr>{ph}</p:nvPr>'
        f"</p:nvSpPr><p:spPr>{xfrm}</p:spPr><p:txBody>{text_paragraphs}</p:txBody></p:sp>"
    )


def para(*runs: str) -> str:
    return "<a:p>" + "".join(f"<a:r><a:t>{run}</a:t></a:r>" for run in runs) + "</a:p>"


def write_pptx(
    path: Path,
    slides: list[tuple[str, list[tuple[str, str, str]]]],
    parts: dict[str, str | bytes] | None = None,
) -> None:
    """Write a deck whose slides are (shape tree XML, slide relationships)."""

    presentation_rels = [
        (f"rId{i}", "slide", f"slides/slide{i}.xml") for i in range(1, len(slides) + 1)
    ]
    ids = "".join(f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, len(slides) + 1))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="png" ContentType="image/png"/>'
            '<Default Extension="emf" ContentType="image/x-emf"/></Types>',
        )
        archive.writestr(
            "ppt/presentation.xml",
            f"<p:presentation {NS}><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>",
        )
        archive.writestr("ppt/_rels/presentation.xml.rels", rels(*presentation_rels))
        for index, (tree, slide_rels) in enumerate(slides, start=1):
            archive.writestr(
                f"ppt/slides/slide{index}.xml",
                f"<p:sld {NS}><p:cSld><p:spTree>{tree}</p:spTree></p:cSld></p:sld>",
            )
            if slide_rels:
                archive.writestr(f"ppt/slides/_rels/slide{index}.xml.rels", rels(*slide_rels))
        for name, payload in (parts or {}).items():
            archive.writestr(name, payload)


def parse(path: Path) -> ParsedDocument:
    return BuiltinTextOcrProvider().parse(LocalFileSource(path).list_files()[0], OcrOptions())


def assert_exact_offsets(page: ParsedPage) -> None:
    for block in page.blocks:
        start, end = block.metadata["start_offset"], block.metadata["end_offset"]
        assert page.markdown[start:end] == block.text, block.kind


def test_slide_reads_title_first_then_shapes_by_position_with_groups_as_units(
    tmp_path: Path,
) -> None:
    group = (
        "<p:grpSp><p:nvGrpSpPr><p:cNvPr id='9' name='g'/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
        "<p:grpSpPr><a:xfrm><a:off x='0' y='3000000'/><a:ext cx='2000' cy='2000'/>"
        "<a:chOff x='500000' y='500000'/><a:chExt cx='1000' cy='1000'/></a:xfrm></p:grpSpPr>"
        + sp(para("Group right"), x=500900, y=500000)
        + sp(para("Group left"), x=500000, y=500000)
        + "</p:grpSp>"
    )
    tree = (
        sp(para("Bottom note"), x=0, y=6000000)
        + sp(para("Right column"), x=5000000, y=1000050)
        + sp(para("Left column"), x=100000, y=1000000)
        + group
        + sp(para("Floating"), x=None, y=None)
        + sp(para("Quarterly ", "review"), x=0, y=5000000, ph='<p:ph type="title"/>')
    )
    path = tmp_path / "order.pptx"
    write_pptx(path, [(tree, [])])

    page = parse(path).pages[0]

    assert page.markdown.split("\n") == [
        "# Quarterly review",
        "Left column",
        "Right column",
        "Group left",
        "Group right",
        "Bottom note",
        "Floating",
    ]
    assert page.blocks[0].kind == "heading"
    assert_exact_offsets(page)


def test_slide_paragraphs_join_runs_and_keep_line_breaks(tmp_path: Path) -> None:
    body = (
        "<a:p><a:r><a:t>Net </a:t></a:r><a:r><a:t>growth</a:t></a:r>"
        "<a:br/><a:r><a:t>second line</a:t></a:r></a:p>"
        "<a:p><a:fld id='{1}' type='slidenum'><a:t>7</a:t></a:fld></a:p>"
    )
    path = tmp_path / "runs.pptx"
    write_pptx(path, [(sp(body, x=0, y=0), [])])

    assert parse(path).pages[0].markdown == "Net growth\nsecond line\n7"


def test_placeholders_without_a_position_take_their_layouts(tmp_path: Path) -> None:
    layout = (
        f"<p:sldLayout {NS}><p:cSld><p:spTree>"
        + sp(para("x"), x=0, y=100, ph='<p:ph idx="1"/>')
        + sp(para("y"), x=0, y=900, ph='<p:ph idx="2"/>')
        + "</p:spTree></p:cSld></p:sldLayout>"
    )
    tree = sp(para("Lower body"), x=None, y=None, ph='<p:ph idx="2"/>') + sp(
        para("Upper body"), x=None, y=None, ph='<p:ph idx="1"/>'
    )
    path = tmp_path / "layout.pptx"
    write_pptx(
        path,
        [(tree, [("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml")])],
        {"ppt/slideLayouts/slideLayout1.xml": layout},
    )

    assert parse(path).pages[0].markdown == "Upper body\nLower body"


def test_slide_table_repeats_merged_cells_and_counts_its_header(tmp_path: Path) -> None:
    def cell(text: str, attrs: str = "") -> str:
        return f"<a:tc {attrs}><a:txBody>{para(text) if text else '<a:p/>'}</a:txBody></a:tc>"

    table = (
        '<a:tbl><a:tblPr firstRow="1"/><a:tblGrid><a:gridCol w="1"/><a:gridCol w="1"/>'
        '<a:gridCol w="1"/></a:tblGrid>'
        "<a:tr>"
        + cell("Site")
        + cell("Visits", 'gridSpan="2"')
        + cell("", 'hMerge="1"')
        + "</a:tr>"
        "<a:tr>" + cell("North", 'rowSpan="2"') + cell("10") + cell("12") + "</a:tr>"
        "<a:tr>" + cell("", 'vMerge="1"') + cell("14") + cell("16") + "</a:tr>"
        "</a:tbl>"
    )
    frame = (
        "<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id='4' name='t'/><p:cNvGraphicFramePr/>"
        "<p:nvPr/></p:nvGraphicFramePr><p:xfrm><a:off x='0' y='2000'/><a:ext cx='1' cy='1'/>"
        f"</p:xfrm><a:graphic><a:graphicData uri='{A}/table'>{table}</a:graphicData></a:graphic>"
        "</p:graphicFrame>"
    )
    path = tmp_path / "table.pptx"
    write_pptx(path, [(sp(para("Intro"), x=0, y=0) + frame, [])])

    document = parse(path)
    page = document.pages[0]
    block = next(block for block in page.blocks if block.kind == "table")
    table_data = TableData.model_validate(block.metadata["table"])

    assert table_data.header_rows == 1
    assert table_data.rows == [
        ["Site", "Visits", "Visits"],
        ["North", "10", "12"],
        ["North", "14", "16"],
    ]
    assert block.text == render_table(table_data)
    assert page.markdown == "Intro\n\n" + render_table(table_data)
    assert document.provider_metadata["tables"] == 1
    assert_exact_offsets(page)


def test_slide_chart_notes_and_picture(tmp_path: Path) -> None:
    chart_frame = (
        "<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id='5' name='c'/><p:cNvGraphicFramePr/>"
        "<p:nvPr/></p:nvGraphicFramePr><p:xfrm><a:off x='0' y='1000'/><a:ext cx='1' cy='1'/>"
        f"</p:xfrm><a:graphic><a:graphicData uri='{C}'><c:chart r:id='rId2'/></a:graphicData>"
        "</a:graphic></p:graphicFrame>"
    )
    broken_frame = chart_frame.replace("rId2", "rId5").replace("y='1000'", "y='1500'")
    picture = (
        "<p:pic><p:nvPicPr><p:cNvPr id='6' name='p' descr='Map of sales offices'/><p:cNvPicPr/>"
        "<p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed='rId3'/></p:blipFill>"
        "<p:spPr><a:xfrm><a:off x='0' y='3000'/><a:ext cx='914400' cy='457200'/></a:xfrm>"
        "</p:spPr></p:pic>"
    )
    notes = (
        f"<p:notes {NS}><p:cSld><p:spTree>"
        + sp(para("1"), x=None, y=None, ph='<p:ph type="sldNum"/>')
        + sp(para("Mention the pilot."), x=None, y=None, ph='<p:ph type="body" idx="1"/>')
        + sp(para("Skip the appendix."), x=None, y=None, ph='<p:ph type="body" idx="1"/>')
        + sp(para("slide image"), x=None, y=None, ph='<p:ph type="sldImg"/>')
        + "</p:spTree></p:cSld></p:notes>"
    )
    slide_rels = [
        ("rId2", "chart", "../charts/chart1.xml"),
        ("rId5", "chart", "../charts/chart2.xml"),
        ("rId3", "image", "../media/image1.png"),
        ("rId4", "notesSlide", "../notesSlides/notesSlide1.xml"),
    ]
    image = png_bytes(3, 2)
    path = tmp_path / "chart.pptx"
    write_pptx(
        path,
        [(sp(para("Results"), x=0, y=0) + chart_frame + broken_frame + picture, slide_rels)],
        {
            "ppt/charts/chart1.xml": CHART_XML,
            "ppt/charts/chart2.xml": "<c:chartSpace xmlns:c='x'><broken",
            "ppt/media/image1.png": image,
            "ppt/notesSlides/notesSlide1.xml": notes,
        },
    )

    document = parse(path)
    page = document.pages[0]
    chart_block = next(block for block in page.blocks if block.kind == "chart")
    chart = ChartData.model_validate(chart_block.metadata["chart"])

    assert chart.series[0].values == [40, 55]
    assert page.markdown == (
        "Results\n\n"
        + render_chart(chart)
        + "\n\nFigure: Map of sales offices"
        + "\n\nSpeaker notes:\nMention the pilot.\nSkip the appendix."
    )
    assert document.provider_metadata["charts"] == 1
    assert document.provider_metadata["charts_unreadable"] == 1
    assert document.provider_metadata["notes_pages"] == 1
    notes_block = next(block for block in page.blocks if block.kind == "notes")
    assert notes_block.text.startswith("Speaker notes:\n")
    assert_exact_offsets(page)

    [asset] = document.assets
    assert asset.page_number == 1
    assert asset.metadata["member"] == "ppt/media/image1.png"
    assert asset.metadata["content_type"] == "image/png"
    assert asset.metadata["size_bytes"] == len(image)
    assert (asset.metadata["width_px"], asset.metadata["height_px"]) == (3, 2)
    assert asset.metadata["alt_text"] == "Map of sales offices"
    assert asset.metadata["extent_emu"] == {"cx": 914400, "cy": 457200}
    assert len(asset.metadata["sha256"]) == 64
    figure = next(block for block in page.blocks if block.kind == "figure")
    assert figure.metadata["asset_id"] == asset.id
    assert figure.text == "Map of sales offices"

    assert read_image_bytes(path, asset) == image
    assert (
        BuiltinTextOcrProvider().read_picture(LocalFileSource(path).list_files()[0], asset) == image
    )


def test_read_image_bytes_refuses_unknown_assets_and_keeps_the_entry_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "picture.pptx"
    picture = (
        "<p:pic><p:nvPicPr><p:cNvPr id='6' name='p'/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>"
        "<p:blipFill><a:blip r:embed='rId3'/></p:blipFill><p:spPr/></p:pic>"
    )
    write_pptx(
        path,
        [(picture, [("rId3", "image", "../media/image1.png")])],
        {"ppt/media/image1.png": png_bytes()},
    )
    document = parse(path)
    [asset] = document.assets
    figure = document.pages[0].blocks[0]
    assert figure.kind == "figure"
    assert figure.text == ""
    assert document.pages[0].markdown == ""
    file = LocalFileSource(path).list_files()[0]

    foreign = ParsedAsset(id="x", kind="image", metadata={"member": "ppt/media/image1.png"})
    with pytest.raises(ValueError, match="not reported"):
        BuiltinTextOcrProvider().read_picture(file, foreign)
    missing = asset.model_copy(update={"metadata": {**asset.metadata, "member": "ppt/none.png"}})
    with pytest.raises(ValueError, match="no part"):
        read_image_bytes(path, missing)
    monkeypatch.setattr(builtin_text, "MAX_OFFICE_ENTRY_BYTES", 10)
    with pytest.raises(ValueError, match="byte limit"):
        read_image_bytes(path, asset)


def write_docx(path: Path, body: str, parts: dict[str, str | bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{W}" xmlns:wp="{WP}" xmlns:a="{A}" xmlns:pic="{PIC}" '
            f'xmlns:c="{C}" xmlns:r="{R}"><w:body>{body}</w:body></w:document>',
        )
        archive.writestr(
            "word/_rels/document.xml.rels",
            rels(("rId1", "chart", "charts/chart1.xml"), ("rId2", "image", "media/image1.png")),
        )
        for name, payload in (parts or {}).items():
            archive.writestr(name, payload)


def test_docx_reads_tables_charts_and_pictures(tmp_path: Path) -> None:
    def cell(text: str, properties: str = "") -> str:
        return f"<w:tc><w:tcPr>{properties}</w:tcPr><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:tc>"

    table = (
        '<w:tbl><w:tblPr><w:tblLook w:val="04A0" w:firstRow="1"/></w:tblPr>'
        "<w:tr>" + cell("Plan") + cell("Cost", '<w:gridSpan w:val="2"/>') + "</w:tr>"
        "<w:tr>" + cell("Basic", '<w:vMerge w:val="restart"/>') + cell("5") + cell("6") + "</w:tr>"
        "<w:tr>" + cell("", "<w:vMerge/>") + cell("7") + cell("8") + "</w:tr>"
        "</w:tbl>"
    )
    chart = (
        "<w:p><w:r><w:t>Figure one</w:t></w:r><w:r><w:drawing><wp:inline>"
        "<wp:extent cx='10' cy='20'/><wp:docPr id='1' name='Chart 1'/>"
        f"<a:graphic><a:graphicData uri='{C}'><c:chart r:id='rId1'/></a:graphicData></a:graphic>"
        "</wp:inline></w:drawing></w:r></w:p>"
    )
    picture = (
        "<w:p><w:r><w:drawing><wp:inline><wp:extent cx='30' cy='40'/>"
        "<wp:docPr id='2' name='Picture 2' descr='Team photo'/><a:graphic>"
        f"<a:graphicData uri='{PIC}'><pic:pic><pic:blipFill><a:blip r:embed='rId2'/>"
        "</pic:blipFill></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"
    )
    path = tmp_path / "report.docx"
    write_docx(
        path,
        "<w:p><w:r><w:t>Intro</w:t></w:r></w:p>" + table + chart + picture,
        {"word/charts/chart1.xml": CHART_XML, "word/media/image1.png": png_bytes()},
    )

    document = parse(path)
    page = document.pages[0]

    assert [block.kind for block in page.blocks] == [
        "paragraph",
        "table",
        "paragraph",
        "chart",
        "figure",
    ]
    table_data = TableData.model_validate(page.blocks[1].metadata["table"])
    assert table_data.header_rows == 1
    assert table_data.rows == [["Plan", "Cost", "Cost"], ["Basic", "5", "6"], ["Basic", "7", "8"]]
    assert "| Basic | 7 | 8 |" in page.markdown
    assert "Chart (column): Units by region" in page.markdown
    assert page.markdown.endswith("Figure: Team photo")
    [asset] = document.assets
    assert asset.metadata["member"] == "word/media/image1.png"
    assert asset.metadata["alt_text"] == "Team photo"
    assert asset.metadata["extent_emu"] == {"cx": 30, "cy": 40}
    assert_exact_offsets(page)


def test_xlsx_charts_follow_the_unchanged_cell_text(tmp_path: Path) -> None:
    sheet = (
        f'<worksheet xmlns="{S}" xmlns:r="{R}"><sheetData>'
        '<row r="1"><c r="A1" t="inlineStr"><is><t>Region</t></is></c>'
        '<c r="B1" t="inlineStr"><is><t>Units</t></is></c></row>'
        '<row r="2"><c r="A2" t="inlineStr"><is><t>East</t></is></c><c r="B2"><v>40</v></c></row>'
        '</sheetData><drawing r:id="rId1"/></worksheet>'
    )
    drawing = (
        f'<xdr:wsDr xmlns:xdr="{XDR}" xmlns:a="{A}" xmlns:c="{C}" xmlns:r="{R}">'
        "<xdr:twoCellAnchor><xdr:from><xdr:col>3</xdr:col><xdr:row>1</xdr:row></xdr:from>"
        "<xdr:graphicFrame><xdr:nvGraphicFramePr><xdr:cNvPr id='2' name='Chart 1'/>"
        "</xdr:nvGraphicFramePr><a:graphic>"
        f"<a:graphicData uri='{C}'><c:chart r:id='rId1'/></a:graphicData></a:graphic>"
        "</xdr:graphicFrame></xdr:twoCellAnchor></xdr:wsDr>"
    )
    path = tmp_path / "book.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
        archive.writestr(
            "xl/worksheets/_rels/sheet1.xml.rels",
            rels(("rId1", "drawing", "../drawings/drawing1.xml")),
        )
        archive.writestr("xl/drawings/drawing1.xml", drawing)
        archive.writestr(
            "xl/drawings/_rels/drawing1.xml.rels", rels(("rId1", "chart", "../charts/chart1.xml"))
        )
        archive.writestr("xl/charts/chart1.xml", CHART_XML)

    document = parse(path)
    page = document.pages[0]
    cells = builtin_text._xlsx_sheet_text(sheet.encode(), [])

    assert page.markdown.startswith("Region | Units\nEast | 40\n\nChart (column): Units by region")
    assert page.markdown.startswith(
        builtin_text._page_from_text(document_file(path), 1, cells).markdown
    )
    assert [block.kind for block in page.blocks] == ["table_row", "table_row", "chart"]
    assert document.provider_metadata["charts"] == 1
    assert_exact_offsets(page)


def document_file(path: Path) -> InputFile:
    return LocalFileSource(path).list_files()[0]


def test_html_tables_become_table_blocks_with_spans_and_header_rows(tmp_path: Path) -> None:
    html = tmp_path / "page.html"
    html.write_text(
        "<h1>Prices</h1><p>Before the table.</p>"
        "<table><caption>Plans</caption><thead><tr><th>Plan</th><th colspan='2'>Price</th></tr>"
        "</thead><tbody><tr><td rowspan='2'>Basic</td><td>5</td><td>6</td></tr>"
        "<tr><td>7</td><td>8 <br>per seat</td></tr></tbody></table><p>After.</p>",
        encoding="utf-8",
    )

    page = parse(html).pages[0]
    block = next(block for block in page.blocks if block.kind == "table")
    table = TableData.model_validate(block.metadata["table"])

    assert table.caption == "Plans"
    assert table.header_rows == 1
    assert table.rows == [
        ["Plan", "Price", "Price"],
        ["Basic", "5", "6"],
        ["Basic", "7", "8\nper seat"],
    ]
    assert page.markdown == ("# Prices\nBefore the table.\n\n" + render_table(table) + "\n\nAfter.")
    assert_exact_offsets(page)


def test_pdf_pictures_are_assets_and_small_ones_are_skipped(tmp_path: Path) -> None:
    big = tmp_path / "big.pdf"
    small = tmp_path / "small.pdf"
    Image.new("RGB", (200, 150), (200, 10, 10)).save(big, format="PDF")
    Image.new("RGB", (20, 20), (200, 10, 10)).save(small, format="PDF")
    writer = PdfWriter()
    for source in (big, small):
        writer.add_page(PdfReader(source).pages[0])
    combined = tmp_path / "combined.pdf"
    with combined.open("wb") as handle:
        writer.write(handle)
    file = InputFile(
        id="f",
        path=combined,
        source_uri="file://combined.pdf",
        checksum="x",
        mime_type="application/pdf",
        size_bytes=combined.stat().st_size,
    )
    parts = builtin_text._DocumentParts(file)
    pages, _ratio = BuiltinTextOcrProvider()._parse_pdf(combined, file, parts)

    assert len(pages) == 2
    [asset] = parts.assets
    assert asset.page_number == 1
    assert (asset.metadata["width_px"], asset.metadata["height_px"]) == (200, 150)
    assert asset.metadata["content_type"] == "image/jpeg"
    payload = read_image_bytes(combined, asset)
    with Image.open(io.BytesIO(payload)) as image:
        assert image.size == (200, 150)
    assert pages[0].blocks[-1].kind == "figure"
    assert_exact_offsets(pages[0])


def test_alternate_content_falls_back_and_smartart_text_is_read(tmp_path: Path) -> None:
    mc = "http://schemas.openxmlformats.org/markup-compatibility/2006"
    dgm = "http://schemas.openxmlformats.org/drawingml/2006/diagram"
    alternate = (
        f'<mc:AlternateContent xmlns:mc="{mc}"><mc:Choice Requires="p14">'
        "<p:contentPart r:id='rId9'/></mc:Choice><mc:Fallback>"
        "<p:pic><p:nvPicPr><p:cNvPr id='6' name='ink' descr='Hand-drawn arrow'/><p:cNvPicPr/>"
        "<p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed='rId3'/></p:blipFill>"
        "<p:spPr><a:xfrm><a:off x='0' y='5000'/><a:ext cx='1' cy='1'/></a:xfrm></p:spPr></p:pic>"
        "</mc:Fallback></mc:AlternateContent>"
    )
    diagram = (
        "<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id='7' name='d'/><p:cNvGraphicFramePr/>"
        "<p:nvPr/></p:nvGraphicFramePr><p:xfrm><a:off x='0' y='1000'/><a:ext cx='1' cy='1'/>"
        f"</p:xfrm><a:graphic><a:graphicData uri='{dgm}'>"
        f'<dgm:relIds xmlns:dgm="{dgm}" r:dm="rId4" r:lo="rId5" r:qs="rId6" r:cs="rId7"/>'
        "</a:graphicData></a:graphic></p:graphicFrame>"
    )
    data = (
        f'<dgm:dataModel xmlns:dgm="{dgm}" xmlns:a="{A}"><dgm:ptLst>'
        "<dgm:pt modelId='0' type='doc'><dgm:t><a:p/></dgm:t></dgm:pt>"
        f"<dgm:pt modelId='1'><dgm:t>{para('Plan')}</dgm:t></dgm:pt>"
        f"<dgm:pt modelId='2' type='parTrans'><dgm:t>{para('ignored')}</dgm:t></dgm:pt>"
        f"<dgm:pt modelId='3'><dgm:t>{para('Build')}</dgm:t></dgm:pt>"
        "</dgm:ptLst></dgm:dataModel>"
    )
    path = tmp_path / "alternate.pptx"
    write_pptx(
        path,
        [
            (
                alternate + diagram,
                [
                    ("rId3", "image", "../media/image1.png"),
                    ("rId4", "diagramData", "../diagrams/data1.xml"),
                ],
            )
        ],
        {"ppt/media/image1.png": png_bytes(), "ppt/diagrams/data1.xml": data},
    )

    document = parse(path)

    assert document.pages[0].markdown == "Plan\nBuild\nFigure: Hand-drawn arrow"
    assert len(document.assets) == 1
    assert_exact_offsets(document.pages[0])
