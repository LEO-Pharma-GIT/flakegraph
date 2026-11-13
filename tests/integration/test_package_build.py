from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest

from kg_processor.application.prompt_registry import PROMPT_REVISIONS

# The build is driven through uv rather than a build frontend imported here,
# because the lock file is what pins the build backend; a checkout without uv
# on PATH (a plain pip install of the dev extras) skips rather than fails.
pytestmark = pytest.mark.skipif(
    shutil.which("uv") is None, reason="uv is not on PATH; the wheel build contract needs it"
)


def test_wheel_contains_all_runtime_assets_and_typing_marker(tmp_path: Path) -> None:
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(tmp_path.glob("flakegraph-*.whl"))
    assert len(wheels) == 1, result.stdout + result.stderr

    with ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())

    prompt_files = {f"kg_processor/prompts/{prompt_name}.md" for prompt_name in PROMPT_REVISIONS}
    assert prompt_files <= names
    assert {
        "kg_processor/explorer_assets/graph_explorer.html",
        "kg_processor/explorer_assets/graph_explorer.css",
        "kg_processor/explorer_assets/graph_explorer.js",
    } <= names
    assert "kg_processor/py.typed" in names
