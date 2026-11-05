# SPDX-License-Identifier: Apache-2.0
"""Execute retry exclusion SQL offline while live contracts pin PostgreSQL locking."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, cast

import pytest
from psycopg.types.json import Jsonb

from kg_processor.adapters.distributed.postgres import PostgresDistributedStore


def test_retry_sql_skips_failed_document_tree_and_preserves_other_work() -> None:
    """The real retry method finds failed windows through their document ancestry."""

    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    # PostgreSQL integration contracts exercise advisory-lock exclusion. This
    # offline test executes the retry SQL's data transitions in one connection.
    database.create_function("hashtextextended", 2, lambda _graph, _seed: 0)
    database.create_function("pg_advisory_xact_lock", 1, lambda _key: None)
    database.executescript("""
        CREATE TABLE flakegraph_run (
            id TEXT PRIMARY KEY, graph_id TEXT, status TEXT, config_json TEXT,
            error_json TEXT, updated_at TEXT
        );
        CREATE TABLE flakegraph_graph_deletion (graph_id TEXT PRIMARY KEY);
        CREATE TABLE flakegraph_task (
            id TEXT PRIMARY KEY, run_id TEXT, stage TEXT, scope_id TEXT, status TEXT,
            attempts INTEGER, remaining_dependencies INTEGER DEFAULT 0, last_error_json TEXT,
            completed_at TEXT, updated_at TEXT, available_at TEXT, lease_owner TEXT,
            lease_expires_at TEXT, started_at TEXT, progress_json TEXT, progress_updated_at TEXT
        );
        CREATE TABLE flakegraph_task_dependency (task_id TEXT, depends_on_task_id TEXT);
        CREATE TABLE flakegraph_publication (
            id TEXT, task_id TEXT, run_id TEXT, status TEXT, attempts INTEGER,
            last_error_json TEXT, completed_at TEXT, updated_at TEXT, available_at TEXT,
            lease_owner TEXT, lease_expires_at TEXT
        );
        CREATE TABLE flakegraph_task_failure (
            run_id TEXT, task_id TEXT, stage TEXT, scope_id TEXT, attempts INTEGER,
            error_json TEXT, failed_at TEXT
        );
        INSERT INTO flakegraph_run VALUES ('run', 'graph', 'failed', '{}', 'failure', NULL);
        INSERT INTO flakegraph_task (id, run_id, stage, scope_id, status, attempts)
        VALUES ('prepare', 'run', 'prepare_document', 'poison', 'succeeded', 1),
               ('window', 'run', 'extract_entity_window', 'window-id', 'failed', 3),
               ('compact', 'run', 'compact_document', 'poison', 'queued', 0),
               ('good', 'run', 'prepare_document', 'good', 'succeeded', 1),
               ('final', 'run', 'finalize_graph', 'graph', 'queued', 0);
        INSERT INTO flakegraph_task_dependency VALUES ('window', 'prepare'), ('compact', 'window');
    """)

    class Result:
        def __init__(self, cursor: sqlite3.Cursor) -> None:
            self.cursor = cursor
            self.rowcount = cursor.rowcount

        def fetchone(self) -> dict[str, Any] | None:
            row = self.cursor.fetchone()
            if row is None:
                return None
            converted = dict(row)
            if "config_json" in converted:
                converted["config_json"] = json.loads(converted["config_json"])
            return converted

        def fetchall(self) -> list[dict[str, Any]]:
            return [dict(row) for row in self.cursor.fetchall()]

    class Connection:
        def execute(self, query: str, parameters: Sequence[Any] = ()) -> Result:
            # Only adapt transport syntax and aggregate spelling. Selection,
            # recursive traversal, transitions and rollback execute in SQLite.
            query = query.replace("%s", "?").replace(" FOR UPDATE", "")
            query = query.replace("= ANY(?)", "IN (SELECT value FROM json_each(?))")
            query = query.replace("BOOL_OR(", "MAX(")
            query = query.replace(
                "UPDATE flakegraph_task dependent", "UPDATE flakegraph_task AS dependent"
            )
            values = [
                json.dumps(value.obj)
                if isinstance(value, Jsonb)
                else json.dumps(value)
                if isinstance(value, list)
                else value
                for value in parameters
            ]
            return Result(database.execute(query, values))

    @contextmanager
    def connection() -> Iterator[Connection]:
        with database:
            yield Connection()

    store = object.__new__(PostgresDistributedStore)
    cast(Any, store)._connection = connection
    with pytest.raises(ValueError, match="must name documents of this run"):
        store.retry_run("run", ["unknown"])
    assert database.execute("SELECT status FROM flakegraph_run").fetchone()[0] == "failed"
    # A healthy document has nothing to retry, so naming it excludes nothing.
    store.retry_run("run", ["good", "poison"])
    run = database.execute("SELECT * FROM flakegraph_run").fetchone()
    assert run["status"] == "queued"
    assert json.loads(run["config_json"])["quarantined_file_ids"] == ["poison"]
    tasks = {row["id"]: dict(row) for row in database.execute("SELECT * FROM flakegraph_task")}
    assert tasks["window"]["status"] == "succeeded"
    assert tasks["window"]["attempts"] == 3
    assert tasks["compact"]["status"] == "succeeded"
    assert tasks["compact"]["remaining_dependencies"] == 0
    assert tasks["good"]["status"] == "succeeded"
    assert tasks["good"]["attempts"] == 1
    assert tasks["final"]["status"] == "queued"
    assert database.execute("SELECT task_id FROM flakegraph_task_failure").fetchone()[0] == "window"
    database.close()
