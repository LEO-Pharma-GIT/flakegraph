from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from snowflake_fakes import FakeConnection, FakeCursor
from typer.testing import CliRunner

from kg_processor.adapters.writers.snowflake_direct import build_snowflake_rows
from kg_processor.application.graph_evaluation import evaluate_graph_artifacts
from kg_processor.application.graph_explorer import build_graph_explorer_dataset
from kg_processor.application.snowflake_export import export_snowflake_graph
from kg_processor.application.snowflake_schema import snowflake_schema_columns
from kg_processor.cli import app
from kg_processor.config.settings import Settings
from kg_processor.domain.graph import GraphWriteBatch

_ROWS: dict[str, list[object]] = {
    "KG_NODE": [
        ("node-1", "g1", "aikido", "Aikido", "CONCEPT", None, None, None, None, None, 1, 0.5),
    ],
    "KG_EDGE": [
        (
            "edge-1",
            "g1",
            "node-1",
            "node-2",
            "RELATED_TO",
            None,
            1.0,
            0.9,
            None,
            None,
            None,
            1,
            "reported",
            None,
        ),
    ],
}


class ExportCursor(FakeCursor):
    """Answer each table read with that table's rows, so every export has data to write."""

    def __init__(self, rows: dict[str, list[object]] | None = None) -> None:
        super().__init__()
        self.table_rows = rows if rows is not None else _ROWS

    def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
        *,
        timeout: int | None = None,
    ) -> object:
        super().execute(sql, params, timeout=timeout)
        self.result_sets = [rows for table, rows in self.table_rows.items() if f".{table} " in sql]
        if "SELECT JOB_ID, RUN_ID," in sql and self.result_sets:
            rows = self.result_sets[0]
            if params is not None and "AND JOB_ID = ?" in sql:
                rows = [row for row in rows if row[0] == params[1]]  # type: ignore[index]
            if params is not None and "AND RUN_ID = ?" in sql:
                index = 2 if "AND JOB_ID = ?" in sql else 1
                rows = [row for row in rows if row[1] == params[index]]  # type: ignore[index]
            self.result_sets = [rows]
        return None


def _connection() -> FakeConnection:
    return FakeConnection(cursor=ExportCursor())


def _settings() -> Settings:
    return Settings.model_validate(
        {
            "snowflake": {
                "account": "EXAMPLE_ACCOUNT",
                "user": "someone@example.com",
                "database": "EXAMPLE_DB",
                "schema": "EXAMPLE_SCHEMA",
                "role": "EXAMPLE_ROLE",
                "warehouse": "EXAMPLE_WH",
            }
        }
    )


def test_export_writes_every_inspection_artifact(tmp_path: Path) -> None:
    """Inspection reads a fixed table set, so every one must exist after export.

    A missing parquet file is read as an empty table rather than an error, which
    would silently understate a graph during gold evaluation.
    """

    connection = _connection()
    output = tmp_path / "kg"

    result = export_snowflake_graph(
        _settings(), "g1", output, connector_factory=lambda **_: connection
    )

    expected = {
        "documents",
        "pages",
        "blocks",
        "assets",
        "chunks",
        "nodes",
        "edges",
        "edge_observations",
        "evidence",
        "entity_sources",
        "communities",
        "community_findings",
        "discarded_windows",
        "rejected_records",
        "failed_documents",
    }
    assert {path.stem for path in output.glob("*.parquet")} == expected
    assert result["tables"]["nodes"] == 1
    assert result["tables"]["edges"] == 1
    assert connection.closed


def test_export_uses_lower_case_artifact_columns(tmp_path: Path) -> None:
    """Inspection addresses columns by lower-case name; Snowflake returns upper."""

    output = tmp_path / "kg"

    export_snowflake_graph(_settings(), "g1", output, connector_factory=lambda **_: _connection())

    nodes = pd.read_parquet(output / "nodes.parquet")

    assert "name" in nodes.columns
    assert "NAME" not in nodes.columns
    assert nodes.loc[0, "name"] == "Aikido"
    assert nodes.loc[0, "aliases"] is None
    assert nodes.loc[0, "source_chunk_ids"] is None


def test_export_excludes_bookkeeping_columns(tmp_path: Path) -> None:
    """UPDATED_AT is a storage detail and is not part of the graph contract."""

    output = tmp_path / "kg"

    export_snowflake_graph(_settings(), "g1", output, connector_factory=lambda **_: _connection())

    nodes = pd.read_parquet(output / "nodes.parquet")

    assert "updated_at" not in nodes.columns
    assert "UPDATED_AT" in snowflake_schema_columns()["KG_NODE"]


def test_export_scopes_every_read_to_the_requested_graph(tmp_path: Path) -> None:
    """A shared schema holds many graphs, so an unscoped read would merge them."""

    connection = _connection()

    export_snowflake_graph(
        _settings(), "g1", tmp_path / "kg", connector_factory=lambda **_: connection
    )

    statements = connection.cursor_instance.executed

    assert statements
    for sql, params in statements:
        assert "WHERE GRAPH_ID = ?" in sql
        assert params == ["g1"]


def test_export_reads_the_gap_and_failure_tables(
    tmp_path: Path,
) -> None:
    connection = _connection()

    export_snowflake_graph(
        _settings(), "g1", tmp_path / "kg", connector_factory=lambda **_: connection
    )

    reads = " ".join(sql for sql, _ in connection.cursor_instance.executed)
    assert ".KG_DISCARDED_WINDOW " in reads
    assert ".KG_REJECTED_RECORD " in reads
    assert ".KG_FAILED_DOCUMENT " in reads


def test_export_rejects_empty_graph_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="graph id"):
        export_snowflake_graph(
            _settings(), "", tmp_path / "kg", connector_factory=lambda **_: _connection()
        )


@pytest.mark.parametrize(
    "target",
    [
        {"database": 'EXAMPLE_DB.PUBLIC.KG_NODE WHERE 1=1 OR "'},
        {"schema": "EXAMPLE_SCHEMA; DROP TABLE KG_NODE; --"},
    ],
)
def test_export_rejects_a_name_that_is_not_an_identifier(
    tmp_path: Path,
    target: dict[str, str],
) -> None:
    """Database and schema reach the statement as text and cannot be bound."""

    settings = Settings.model_validate(
        {
            "snowflake": {
                "account": "EXAMPLE_ACCOUNT",
                "user": "someone@example.com",
                "database": "EXAMPLE_DB",
                "schema": "EXAMPLE_SCHEMA",
                **target,
            }
        }
    )

    with pytest.raises(ValueError, match="must be an unquoted identifier"):
        export_snowflake_graph(
            settings, "g1", tmp_path / "kg", connector_factory=lambda **_: _connection()
        )


def test_export_accepts_a_lower_case_configured_name(tmp_path: Path) -> None:
    """Snowflake resolves unquoted identifiers case-insensitively."""

    settings = Settings.model_validate(
        {
            "snowflake": {
                "account": "EXAMPLE_ACCOUNT",
                "user": "someone@example.com",
                "database": "example_db",
                "schema": "example_schema",
            }
        }
    )
    connection = _connection()

    export_snowflake_graph(
        settings, "g1", tmp_path / "kg", connector_factory=lambda **_: connection
    )

    reads = connection.cursor_instance.executed
    assert reads
    assert all("EXAMPLE_DB.EXAMPLE_SCHEMA." in sql for sql, _ in reads)


def test_export_refuses_to_replace_a_directory_it_did_not_write(tmp_path: Path) -> None:
    """A mistyped export path must not delete unrelated data.

    The export replaces the destination wholesale, so it applies the same
    ownership rule as the local artifacts writer: a non-empty directory is only
    replaced when it carries the ownership manifest.
    """

    output = tmp_path / "notes"
    output.mkdir()
    (output / "thesis.docx").write_bytes(b"irreplaceable")

    with pytest.raises(ValueError, match="Refusing to replace"):
        export_snowflake_graph(
            _settings(), "g1", output, connector_factory=lambda **_: _connection()
        )

    assert (output / "thesis.docx").read_bytes() == b"irreplaceable"


def test_export_replaces_its_own_previous_snapshot(tmp_path: Path) -> None:
    output = tmp_path / "kg"

    export_snowflake_graph(_settings(), "g1", output, connector_factory=lambda **_: _connection())
    (output / "nodes.parquet").write_bytes(b"stale")
    export_snowflake_graph(_settings(), "g1", output, connector_factory=lambda **_: _connection())

    assert (output / "nodes.parquet").read_bytes() != b"stale"
    assert (output / ".flakegraph-artifacts.json").is_file()


def _graph_rows(*, encoded: bool) -> dict[str, list[object]]:
    """Represent writer rows with the connector's JSON ARRAY/VARIANT surface."""

    node = {
        "graph_id": "g1",
        "primary_type": "CONCEPT",
        "types": ["CONCEPT"],
        "description": '["a literal description"]',
        "source_chunk_ids": ["chunk-1"],
        "degree": 1,
    }
    batch = GraphWriteBatch.model_validate(
        {
            "graph_id": "g1",
            "documents": [],
            "pages": [],
            "blocks": [
                {
                    "id": "b1",
                    "graph_id": "g1",
                    "file_id": "f1",
                    "page_number": 1,
                    "kind": "paragraph",
                    "text": "Dose",
                    "bbox": [1, 2, 3, 4],
                    "metadata": {"layout": "body"},
                }
            ],
            "chunks": [],
            "evidence": [],
            "entity_sources": [],
            "community_findings": [],
            "nodes": [
                dict(node, id="n1", normalized_name="acme plc", name="Acme plc", aliases=["Acme"]),
                dict(node, id="n2", normalized_name="dose", name="Dose"),
            ],
            "edges": [
                {
                    "id": "e1",
                    "graph_id": "g1",
                    "source_node_id": "n1",
                    "target_node_id": "n2",
                    "relation_type": "HAS_DOSE",
                    "description": "A dose",
                    "weight": 1,
                    "source_file_id": "f1",
                    "source_chunk_ids": ["chunk-1"],
                }
            ],
            "edge_observations": [
                {
                    "id": "o1",
                    "graph_id": "g1",
                    "edge_id": "e1",
                    "file_id": "f1",
                    "chunk_id": "chunk-1",
                    "description": "A dose",
                    "weight": 1,
                    "confidence": 1,
                    "evidence_id": "ev1",
                    "quantities": [{"text": "5 mg"}],
                }
            ],
            "communities": [
                {
                    "id": "c1",
                    "graph_id": "g1",
                    "stable_key": "c1",
                    "level": 0,
                    "title": "Doses",
                    "summary": "Doses",
                    "rating": 1,
                    "member_node_ids": ["n1", "n2"],
                }
            ],
            "run_report": {
                "graph_id": "g1",
                "job_id": "job-1",
                "run_id": "run-1",
                "extraction": {"records": 2},
            },
            "graph_metrics": {
                "consumption": {
                    "totals": {"billed_usd": 0.25, "avoided_usd": 0.0, "total_tokens": 20},
                    "events": [{"provider": "synthetic", "usage": {"total_tokens": 20}}],
                }
            },
        }
    )
    rows = build_snowflake_rows(batch)
    columns = snowflake_schema_columns()
    result: dict[str, list[object]] = {}
    for table, records in rows.items():
        keys: tuple[str, ...]
        if table in {"KG_RUN_REPORT", "KG_GRAPH_METRICS"}:
            payload_column = "REPORT" if table == "KG_RUN_REPORT" else "METRICS"
            keys = ("JOB_ID", "RUN_ID", payload_column)
        else:
            keys = tuple(
                name for name in columns[table] if name not in {"CREATED_AT", "UPDATED_AT"}
            )
        result[table] = [
            tuple(
                json.dumps(record.get(key))
                if encoded and key != "EMBEDDING" and isinstance(record.get(key), (list, dict))
                else record.get(key)
                for key in keys
            )
            for record in records
        ]
    return result


def _diagnostic_connection(rows: dict[str, list[object]]) -> FakeConnection:
    return FakeConnection(cursor=ExportCursor(rows))


@pytest.mark.parametrize("encoded", [True, False])
def test_export_preserves_list_contract_for_evaluation_and_exploration(
    tmp_path: Path, encoded: bool
) -> None:
    """Aliases, quantities and memberships retain the local writer's list shape."""

    output = tmp_path / "kg"
    export_snowflake_graph(
        _settings(),
        "g1",
        output,
        connector_factory=lambda **_: _diagnostic_connection(_graph_rows(encoded=encoded)),
    )
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps(
            {
                "name": "Dose fixture",
                "description": "Alias, quantity and membership fixture.",
                "entities": [
                    {"id": "acme", "name": "Acme", "type": "CONCEPT"},
                    {"id": "dose", "name": "Dose", "type": "CONCEPT"},
                ],
                "relations": [
                    {
                        "source": "acme",
                        "target": "dose",
                        "relation_type": "HAS_DOSE",
                        "quantities": ["5 mg"],
                    }
                ],
            }
        )
    )
    report = evaluate_graph_artifacts(output, gold_path)
    assert report["entities"]["recall"] == 1.0
    assert report["quantities"]["recall"] == 1.0
    dataset = build_graph_explorer_dataset(output).payload
    assert {node["id"]: node["community_ids"] for node in dataset["nodes"]} == {
        "n1": ["c1"],
        "n2": ["c1"],
    }
    nodes = pd.read_parquet(output / "nodes.parquet")
    assert nodes.loc[0, "description"] == '["a literal description"]'
    assert nodes.loc[0, "embedding"] is None
    blocks = pd.read_parquet(output / "blocks.parquet")
    assert list(blocks.loc[0, "bbox"]) == [1, 2, 3, 4]
    metadata = blocks.loc[0, "metadata"]
    assert json.loads(metadata) == {"layout": "body"} if encoded else metadata == {"layout": "body"}


@pytest.mark.parametrize("encoded", [True, False])
def test_export_publishes_matching_report_and_consumption(tmp_path: Path, encoded: bool) -> None:
    """Both diagnostic VARIANT decoding paths preserve run identity and spend."""

    output = tmp_path / "kg"
    result = export_snowflake_graph(
        _settings(),
        "g1",
        output,
        job_id="job-1",
        connector_factory=lambda **_: _diagnostic_connection(_graph_rows(encoded=encoded)),
    )
    report = json.loads((output / "run_report.json").read_text())
    metrics = json.loads((output / "graph_metrics.json").read_text())
    assert report["job_id"] == result["job_id"] == "job-1"
    assert report["run_id"] == result["run_id"] == "run-1"
    assert report["extraction"] == {"records": 2}
    assert metrics["consumption"]["totals"]["billed_usd"] == 0.25
    assert metrics["consumption"]["events"][0]["usage"]["total_tokens"] == 20


def test_export_attaches_diagnostics_only_to_one_selected_job_and_run(
    tmp_path: Path,
) -> None:
    """Several reports are never combined: unselected, the tables export alone."""

    output = tmp_path / "kg"
    rows = _graph_rows(encoded=True)
    export_snowflake_graph(
        _settings(), "g1", output, connector_factory=lambda **_: _diagnostic_connection(rows)
    )
    before = (output / "run_report.json").read_bytes()
    for table in ("KG_RUN_REPORT", "KG_GRAPH_METRICS"):
        payload = {"graph_id": "g1", "job_id": "job-2", "run_id": "run-2"}
        rows[table].append(("job-2", "run-2", json.dumps(payload)))
    unselected = export_snowflake_graph(
        _settings(), "g1", output, connector_factory=lambda **_: _diagnostic_connection(rows)
    )
    assert (unselected["job_id"], unselected["run_id"]) == (None, None)
    assert unselected["tables"]["nodes"] == 2
    assert json.loads((output / "run_report.json").read_text()) == {}
    assert json.loads((output / "graph_metrics.json").read_text()) == {}
    connection = _diagnostic_connection(rows)
    export_snowflake_graph(
        _settings(),
        "g1",
        output,
        job_id="job-1",
        run_id="run-1",
        connector_factory=lambda **_: connection,
    )
    reads = [
        (sql, params)
        for sql, params in connection.cursor_instance.executed
        if "SELECT JOB_ID, RUN_ID," in sql
    ]
    assert len(reads) == 2
    assert all("AND JOB_ID = ? AND RUN_ID = ?" in sql for sql, _ in reads)
    assert all(params == ["g1", "job-1", "run-1"] for _, params in reads)
    assert (output / "run_report.json").read_bytes() == before


def test_export_requires_run_selector_for_several_batches_of_one_job(tmp_path: Path) -> None:
    rows = _graph_rows(encoded=True)
    rows["KG_GRAPH_METRICS"].append(("job-1", "run-2", "{}"))
    with pytest.raises(ValueError, match="--run-id"):
        export_snowflake_graph(
            _settings(),
            "g1",
            tmp_path / "kg",
            job_id="job-1",
            connector_factory=lambda **_: _diagnostic_connection(rows),
        )


def test_export_rejects_missing_requested_diagnostics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No Snowflake diagnostics match"):
        export_snowflake_graph(
            _settings(),
            "g1",
            tmp_path / "kg",
            job_id="another-job",
            connector_factory=lambda **_: _diagnostic_connection(_graph_rows(encoded=True)),
        )
    assert not (tmp_path / "kg").exists()


@pytest.mark.parametrize("payload", ['"not an object"', "invalid-json", []])
def test_export_rejects_invalid_diagnostics(tmp_path: Path, payload: object) -> None:
    rows = _graph_rows(encoded=True)
    rows["KG_RUN_REPORT"] = [("job-1", "run-1", payload)]
    with pytest.raises(ValueError, match="JSON object"):
        export_snowflake_graph(
            _settings(),
            "g1",
            tmp_path / "kg",
            connector_factory=lambda **_: _diagnostic_connection(rows),
        )
    assert not (tmp_path / "kg").exists()


def test_export_rejects_diagnostic_identity_conflict(tmp_path: Path) -> None:
    rows = _graph_rows(encoded=True)
    rows["KG_RUN_REPORT"] = [("job-1", "run-1", '{"job_id": "another-job"}')]
    with pytest.raises(ValueError, match="job_id conflicts"):
        export_snowflake_graph(
            _settings(),
            "g1",
            tmp_path / "kg",
            connector_factory=lambda **_: _diagnostic_connection(rows),
        )


def test_export_cli_forwards_diagnostic_selectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings()
    monkeypatch.setattr(Settings, "load", lambda _: settings)
    calls: list[dict[str, Any]] = []

    def export(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"graph_id": args[1]}

    monkeypatch.setattr("kg_processor.application.snowflake_export.export_snowflake_graph", export)
    result = CliRunner().invoke(
        app,
        [
            "snowflake",
            "export",
            "--graph-id",
            "g1",
            "--output",
            str(tmp_path / "kg"),
            "--job-id",
            "job-1",
            "--run-id",
            "run-1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [{"job_id": "job-1", "run_id": "run-1"}]
