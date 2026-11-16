# SPDX-License-Identifier: Apache-2.0
"""The retry CLI carries explicit document exclusions to the task store."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from kg_processor.cli import app


def test_retry_cli_passes_quarantined_file_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    store = Mock()
    store.get_run_summary.return_value = SimpleNamespace(model_dump=lambda **_: {"run_id": "run"})
    monkeypatch.setattr("kg_processor.factories.build_distributed_store", lambda _: store)

    result = CliRunner().invoke(
        app,
        [
            "distributed",
            "retry",
            "--run-id",
            "run",
            "--skip-file-id",
            "poison-a",
            "--skip-file-id",
            "poison-b",
        ],
    )

    assert result.exit_code == 0, result.output
    store.initialize.assert_called_once_with()
    store.retry_run.assert_called_once_with("run", skipped_file_ids=["poison-a", "poison-b"])
