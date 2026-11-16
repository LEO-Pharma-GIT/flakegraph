"""Keep the checked-in Grafana dashboards equal to their generator, and sane.

The chart ships ``deploy/helm/flakegraph/dashboards/*.json`` verbatim into
ConfigMaps, so what is committed is what Grafana provisions. The generator is
the source; these tests fail when someone edits the JSON by hand or forgets to
rerun the generator after changing it.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_GENERATOR = Path("deploy/grafana/build_dashboards.py")
_DASHBOARDS = Path("deploy/helm/flakegraph/dashboards")

_PROMETHEUS = {"type": "prometheus", "uid": "prometheus"}
_POSTGRES = {"type": "grafana-postgresql-datasource", "uid": "flakegraph-postgres"}
_EXPECTED_UIDS = {
    "flakegraph-fleet": "Fleet Overview",
    "flakegraph-llm-serving": "LLM Serving",
    "flakegraph-gateway": "Gateway & Consumers",
    "flakegraph-documents": "Document Parsing (OCR)",
    "flakegraph-pipeline": "Pipeline Workers",
    "flakegraph-platform": "Database & Storage",
}
_GRID_COLUMNS = 24
_MIN_PANELS = 8


@cache
def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_dashboards", _GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def _generated() -> dict[str, dict[str, Any]]:
    return dict(_generator().build_all())


def _checked_in() -> dict[str, dict[str, Any]]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(_DASHBOARDS.glob("*.json"))
    }


def _panels(built: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every panel, including those nested inside collapsed rows."""

    for panel in built["panels"]:
        yield panel
        yield from panel.get("panels", [])


def _datasource_refs(built: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    for panel in _panels(built):
        if "datasource" in panel:
            yield panel["title"], panel["datasource"]
        for target in panel.get("targets", []):
            yield panel["title"], target.get("datasource")
    for variable in built["templating"]["list"]:
        yield f"variable {variable['name']}", variable.get("datasource")


def test_checked_in_dashboards_match_the_generator() -> None:
    """A hand edit or an unrun generator shows up as a diff here."""

    generator = _generator()
    generated = {name: generator.render(built) for name, built in _generated().items()}
    on_disk = {
        path.name: path.read_text(encoding="utf-8") for path in sorted(_DASHBOARDS.glob("*.json"))
    }

    assert set(on_disk) == set(generated), (
        "run `uv run python deploy/grafana/build_dashboards.py` and commit the result"
    )
    for name, expected in generated.items():
        assert on_disk[name] == expected, (
            f"{name} differs from the generator; rerun deploy/grafana/build_dashboards.py"
        )


def test_generator_check_mode_agrees_with_the_files() -> None:
    assert _generator().main(["--check"]) == 0


@pytest.mark.parametrize("name", sorted(_EXPECTED_UIDS))
def test_dashboard_identity_is_stable(name: str) -> None:
    """uid, title and tags are what links, folders and the sidecar key off."""

    built = _checked_in()[f"{name}.json"]

    assert built["uid"] == name
    assert built["title"] == _EXPECTED_UIDS[name]
    assert built["tags"] == ["flakegraph"]
    assert isinstance(built["schemaVersion"], int)
    assert built["time"]["from"] == "now-6h"
    assert built["refresh"] == "30s"
    assert built["timezone"] == "browser"
    assert any(
        link["type"] == "dashboards" and "flakegraph" in link["tags"] for link in built["links"]
    )


def test_every_expected_dashboard_exists_and_nothing_else() -> None:
    assert set(_checked_in()) == {f"{name}.json" for name in _EXPECTED_UIDS}


@pytest.mark.parametrize("name", sorted(_EXPECTED_UIDS))
def test_panels_have_unique_ids_and_fit_the_grid(name: str) -> None:
    built = _checked_in()[f"{name}.json"]
    panels = list(_panels(built))

    ids = [panel["id"] for panel in panels]
    assert len(ids) == len(set(ids)), "duplicate panel ids"
    assert len([panel for panel in panels if panel["type"] != "row"]) >= _MIN_PANELS
    for panel in panels:
        pos = panel["gridPos"]
        assert pos["x"] >= 0 and pos["w"] >= 1 and pos["h"] >= 1, panel["title"]
        assert pos["x"] + pos["w"] <= _GRID_COLUMNS, f"{panel['title']} overflows the grid"


@pytest.mark.parametrize("name", sorted(_EXPECTED_UIDS))
def test_datasources_are_referenced_by_uid_only(name: str) -> None:
    """A renamed datasource must not break a panel, so no ref may be a name."""

    built = _checked_in()[f"{name}.json"]

    refs = list(_datasource_refs(built))
    assert refs
    for title, ref in refs:
        assert ref in (_PROMETHEUS, _POSTGRES), f"{title} uses datasource {ref!r}"


@pytest.mark.parametrize("name", sorted(_EXPECTED_UIDS))
def test_every_query_panel_has_a_description_and_a_unit(name: str) -> None:
    """Provisioned dashboards are read-only, so the panel has to explain itself."""

    built = _checked_in()[f"{name}.json"]

    for panel in _panels(built):
        if panel["type"] in {"row", "text"}:
            continue
        assert panel["description"], f"{panel['title']} has no description"
        assert panel["targets"], f"{panel['title']} has no query"
        if panel["type"] != "table":
            assert panel["fieldConfig"]["defaults"].get("unit"), f"{panel['title']} has no unit"


def test_no_panel_legend_exposes_a_hashed_api_key() -> None:
    """LiteLLM's hashed_api_key label is a secret-shaped value; alias only."""

    for built in _checked_in().values():
        for panel in _panels(built):
            for target in panel.get("targets", []):
                assert "hashed_api_key" not in target.get("legendFormat", "")
                assert "hashed_api_key" not in target.get("expr", "")
                raw_sql = target.get("rawSql", "")
                for exposed in ("api_key AS", "api_key,", "token AS", "t.token,", "user_api_key'"):
                    assert exposed not in raw_sql, f"{panel['title']} selects a key column"
