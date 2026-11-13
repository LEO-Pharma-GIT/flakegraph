"""The DrawingML chart reader recovers exactly what a chart's caches show."""

from __future__ import annotations

import pytest

from kg_processor.adapters.ocr.office_charts import ChartReadError, EmptyChartError, read_chart
from kg_processor.domain.tables import render_chart

C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"


def chart_space(plot: str, *, title: str = "", extra_chart: str = "", axes: str = "") -> bytes:
    return (
        f'<c:chartSpace xmlns:c="{C}" xmlns:a="{A}"><c:chart>{title}{extra_chart}'
        f"<c:plotArea><c:layout/>{plot}{axes}</c:plotArea></c:chart></c:chartSpace>"
    ).encode()


def rich_title(*runs: str) -> str:
    text = "".join(f"<a:r><a:t>{run}</a:t></a:r>" for run in runs)
    return f"<c:title><c:tx><c:rich><a:p>{text}</a:p></c:rich></c:tx></c:title>"


def str_cache(*values: str | None, count: int | None = None) -> str:
    points = "".join(
        f'<c:pt idx="{index}"><c:v>{value}</c:v></c:pt>'
        for index, value in enumerate(values)
        if value is not None
    )
    return (
        f'<c:strRef><c:f>Sheet1!$A$2</c:f><c:strCache><c:ptCount val="{count or len(values)}"/>'
        f"{points}</c:strCache></c:strRef>"
    )


def num_cache(*values: float | None, fmt: str = "General", count: int | None = None) -> str:
    points = "".join(
        f'<c:pt idx="{index}"><c:v>{value}</c:v></c:pt>'
        for index, value in enumerate(values)
        if value is not None
    )
    return (
        f"<c:numRef><c:f>Sheet1!$B$2</c:f><c:numCache><c:formatCode>{fmt}</c:formatCode>"
        f'<c:ptCount val="{count or len(values)}"/>{points}</c:numCache></c:numRef>'
    )


def series(name: str, categories: str, values: str) -> str:
    return (
        f'<c:ser><c:idx val="0"/><c:order val="0"/><c:tx>{str_cache(name)}</c:tx>'
        f"<c:cat>{categories}</c:cat><c:val>{values}</c:val></c:ser>"
    )


def test_a_column_chart_reads_series_categories_title_and_axis_titles() -> None:
    plot = (
        '<c:barChart><c:barDir val="col"/><c:grouping val="clustered"/>'
        + series("North", str_cache("Q1", "Q2", "Q3"), num_cache(12, 15.5, 9))
        + series("South", str_cache("Q1", "Q2", "Q3"), num_cache(7, 8, 11))
        + "</c:barChart>"
    )
    axes = (
        f'<c:catAx><c:axId val="1"/><c:axPos val="b"/>{rich_title("Quarter")}</c:catAx>'
        f'<c:valAx><c:axId val="2"/><c:axPos val="l"/>{rich_title("Units ", "sold")}</c:valAx>'
    )

    chart = read_chart(chart_space(plot, title=rich_title("Widget ", "sales"), axes=axes))

    assert chart.chart_type == "column"
    assert chart.title == "Widget sales"
    assert chart.categories == ["Q1", "Q2", "Q3"]
    assert [entry.name for entry in chart.series] == ["North", "South"]
    assert chart.series[0].values == [12, 15.5, 9]
    assert chart.category_axis_title == "Quarter"
    assert chart.value_axis_title == "Units sold"
    assert render_chart(chart) == (
        "Chart (column): Widget sales (values: Units sold)\n"
        "| Quarter | North | South |\n"
        "| --- | --- | --- |\n"
        "| Q1 | 12 | 7 |\n"
        "| Q2 | 15.5 | 8 |\n"
        "| Q3 | 9 | 11 |"
    )


def test_a_horizontal_bar_chart_is_a_bar_chart() -> None:
    plot = (
        '<c:barChart><c:barDir val="bar"/>'
        + series("Share", str_cache("A", "B"), num_cache(1, 2))
        + "</c:barChart>"
    )

    assert read_chart(chart_space(plot)).chart_type == "bar"


def test_a_pie_chart_in_percent_takes_its_single_series_name_as_title() -> None:
    plot = (
        "<c:pieChart>"
        + series("Preferred brand", str_cache("Alpha", "Beta"), num_cache(0.25, 0.75, fmt="0%"))
        + "</c:pieChart>"
    )

    chart = read_chart(chart_space(plot, extra_chart='<c:autoTitleDeleted val="0"/>'))

    assert chart.chart_type == "pie"
    assert chart.title == "Preferred brand"
    assert chart.series[0].number_format == "0%"
    assert "| Alpha | 25% |" in render_chart(chart)
    assert "| Beta | 75% |" in render_chart(chart)


def test_a_deleted_auto_title_stays_deleted() -> None:
    plot = "<c:pieChart>" + series("Only", str_cache("A"), num_cache(1)) + "</c:pieChart>"

    chart = read_chart(chart_space(plot, extra_chart='<c:autoTitleDeleted val="1"/>'))

    assert chart.title is None


def test_a_line_chart_keeps_missing_points_under_their_categories() -> None:
    plot = (
        "<c:lineChart>"
        + series(
            "Index",
            str_cache("Jan", "Feb", "Mar", "Apr"),
            num_cache(1.5, None, 2.25, None, fmt="0.00", count=4),
        )
        + "</c:lineChart>"
    )

    chart = read_chart(chart_space(plot))

    assert chart.chart_type == "line"
    assert chart.series[0].values == [1.5, None, 2.25, None]
    rendered = render_chart(chart)
    assert "| Feb |  |" in rendered
    assert "| Mar | 2.25 |" in rendered
    assert "| Apr |  |" in rendered


def test_a_point_format_applies_when_the_cache_is_general() -> None:
    values = (
        "<c:numRef><c:numCache><c:formatCode>General</c:formatCode><c:ptCount val='1'/>"
        '<c:pt idx="0" formatCode="0.0%"><c:v>0.123</c:v></c:pt></c:numCache></c:numRef>'
    )
    plot = "<c:barChart>" + series("Rate", str_cache("X"), values) + "</c:barChart>"

    chart = read_chart(chart_space(plot))

    assert chart.series[0].number_format == "0.0%"
    assert "| X | 12.3% |" in render_chart(chart)


def test_a_scatter_chart_plots_y_against_numeric_x() -> None:
    plot = (
        "<c:scatterChart>"
        f"<c:ser><c:tx><c:v>Trial</c:v></c:tx><c:xVal>{num_cache(1, 2, 4)}</c:xVal>"
        f"<c:yVal>{num_cache(10, 20, 35)}</c:yVal></c:ser>"
        "</c:scatterChart>"
    )
    axes = (
        f'<c:valAx><c:axId val="1"/><c:axPos val="b"/>{rich_title("Dose")}</c:valAx>'
        f'<c:valAx><c:axId val="2"/><c:axPos val="l"/>{rich_title("Response")}</c:valAx>'
    )

    chart = read_chart(chart_space(plot, axes=axes))

    assert chart.chart_type == "scatter"
    assert chart.series[0].name == "Trial"
    assert chart.series[0].x_values == [1, 2, 4]
    assert chart.series[0].values == [10, 20, 35]
    assert chart.category_axis_title == "Dose"
    assert chart.value_axis_title == "Response"
    assert "| Trial | 4 | 35 |" in render_chart(chart)


def test_a_combo_chart_keeps_every_series_and_names_each_plot_kind() -> None:
    plot = (
        '<c:barChart><c:barDir val="col"/>'
        + series("Revenue", str_cache("2021", "2022"), num_cache(100, 120))
        + "</c:barChart><c:lineChart>"
        + series("Margin", str_cache("2021", "2022"), num_cache(0.1, 0.12, fmt="0%"))
        + "</c:lineChart>"
    )

    chart = read_chart(chart_space(plot))

    assert chart.chart_type == "column+line"
    assert [entry.name for entry in chart.series] == ["Revenue", "Margin"]
    assert "| 2022 | 120 | 12% |" in render_chart(chart)


def test_multi_level_categories_join_outer_and_inner_labels() -> None:
    categories = (
        "<c:multiLvlStrRef><c:f>Sheet1!$A$2:$B$5</c:f><c:multiLvlStrCache>"
        '<c:ptCount val="4"/>'
        '<c:lvl><c:pt idx="0"><c:v>H1</c:v></c:pt><c:pt idx="1"><c:v>H2</c:v></c:pt>'
        '<c:pt idx="2"><c:v>H1</c:v></c:pt><c:pt idx="3"><c:v>H2</c:v></c:pt></c:lvl>'
        '<c:lvl><c:pt idx="0"><c:v>2023</c:v></c:pt><c:pt idx="2"><c:v>2024</c:v></c:pt></c:lvl>'
        "</c:multiLvlStrCache></c:multiLvlStrRef>"
    )
    plot = "<c:barChart>" + series("Orders", categories, num_cache(1, 2, 3, 4)) + "</c:barChart>"

    chart = read_chart(chart_space(plot))

    assert chart.categories == ["2023 / H1", "2023 / H2", "2024 / H1", "2024 / H2"]


def test_numeric_and_date_categories_read_as_their_format_shows_them() -> None:
    dates = (
        "<c:numRef><c:numCache><c:formatCode>mmm\\-yy</c:formatCode><c:ptCount val='2'/>"
        '<c:pt idx="0"><c:v>44927</c:v></c:pt><c:pt idx="1"><c:v>44958</c:v></c:pt>'
        "</c:numCache></c:numRef>"
    )
    plot = "<c:lineChart>" + series("Visits", dates, num_cache(5, 6)) + "</c:lineChart>"

    chart = read_chart(chart_space(plot))

    assert chart.categories == ["2023-01", "2023-02"]


def test_a_chart_without_series_or_plot_area_is_refused() -> None:
    with pytest.raises(EmptyChartError):
        read_chart(chart_space('<c:barChart><c:barDir val="col"/></c:barChart>'))
    with pytest.raises(ChartReadError):
        read_chart(f'<c:chartSpace xmlns:c="{C}"><c:chart/></c:chartSpace>'.encode())
