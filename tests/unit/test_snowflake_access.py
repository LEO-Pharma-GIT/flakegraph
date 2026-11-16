from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from snowflake_fakes import (
    FakeConnection,
    FakeCortexServer,
    FakeCursor,
    completion_reply,
    embedding_reply,
)

from kg_processor.adapters.snowflake import ConnectorFactory
from kg_processor.application import snowflake_access
from kg_processor.application.snowflake_access import (
    REQUIRED_SNOWFLAKE_COLUMNS,
    REQUIRED_SNOWFLAKE_TABLES,
    SnowflakeAccessReport,
    run_snowflake_access_check,
)
from kg_processor.config.settings import Settings, _deep_update


class AccessCheckCursor(FakeCursor):
    """Answer every access-check statement the way a correctly provisioned account does."""

    def __init__(
        self,
        visible_tables: set[str] | None = None,
        visible_columns: dict[str, set[str]] | None = None,
        fail_document_stage: bool = False,
        stored_embedding_dimension: int = 1024,
    ) -> None:
        super().__init__()
        self.visible_tables = visible_tables or set(REQUIRED_SNOWFLAKE_TABLES)
        self.visible_columns = visible_columns or {
            table: set(REQUIRED_SNOWFLAKE_COLUMNS[table]) for table in self.visible_tables
        }
        self.fail_document_stage = fail_document_stage
        self.stored_embedding_dimension = stored_embedding_dimension

    def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
        *,
        timeout: int | None = None,
    ) -> object:
        super().execute(sql, params, timeout=timeout)
        self.rows = []
        self.result_sets = []
        if sql.startswith("SELECT CURRENT_ACCOUNT"):
            self.rows = [
                [
                    "EXAMPLE_ACCOUNT",
                    "AWS_EU_CENTRAL_1",
                    "USER",
                    "KG_PROCESSOR_ROLE",
                    "KG_DB",
                    "GRAPH",
                    "WH",
                ]
            ]
        elif "INFORMATION_SCHEMA.TABLES" in sql:
            self.result_sets = [[(table,) for table in sorted(self.visible_tables)]]
        elif "INFORMATION_SCHEMA.COLUMNS" in sql:
            self.result_sets = [
                [
                    (table, column)
                    for table, columns in sorted(self.visible_columns.items())
                    for column in sorted(columns)
                ]
            ]
        elif sql.startswith("SHOW COLUMNS LIKE 'EMBEDDING'"):
            self.result_sets = [
                [
                    (
                        "created",
                        table,
                        "EMBEDDING",
                        '{"type":"VECTOR","nullable":true,'
                        '"vectorElementType":{"type":"REAL","nullable":false},'
                        f'"dimension":{self.stored_embedding_dimension}}}',
                    )
                    for table in sorted(self.visible_tables)
                ]
            ]
        elif sql.startswith("SHOW WAREHOUSES"):
            self.result_sets = [[("created", "WH")]]
        elif sql.startswith("LIST @KG_DB.GRAPH.KG_DOCS"):
            if self.fail_document_stage:
                raise RuntimeError("stage not authorized")
            self.result_sets = [[("kg_docs/incoming/canary.pdf", 123, "checksum-canary")]]
        elif sql.startswith("LIST @KG_DB.GRAPH.KG_LOAD_STAGE") or sql.startswith(
            "LIST @KG_DB.GRAPH.KG_SERVICE_SPECS"
        ):
            self.result_sets = [[]]
        elif sql.startswith("SHOW COMPUTE POOLS"):
            self.result_sets = [[("KG_PROCESSOR_CPU_POOL",)]]
        elif sql.startswith("SHOW IMAGE REPOSITORIES"):
            self.result_sets = [[("KG_IMAGES",)]]
        elif sql.startswith("SELECT AI_"):
            self._configure_ai_result(sql)
        else:
            raise AssertionError(f"Unexpected SQL: {sql}")
        return None

    def _configure_ai_result(self, sql: str) -> None:
        """Return the minimal shape expected from the Cortex OCR canary.

        Model and embedding canaries go over the REST API, so any other AI
        function reaching SQL fails the test.
        """

        if not sql.startswith("SELECT AI_PARSE_DOCUMENT"):
            raise AssertionError(f"Unexpected Cortex SQL: {sql}")
        self.rows = [[{"value": {"pages": [{"content": "canary"}]}}]]


def _cortex_canary_reply(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("inference:embed"):
        return embedding_reply([[0.1, 0.2, 0.3]])
    return completion_reply({"ok": True})


def _run(
    settings: Settings,
    connector_factory: ConnectorFactory,
    *,
    cortex: FakeCortexServer | None = None,
) -> SnowflakeAccessReport:
    server = cortex or FakeCortexServer(_cortex_canary_reply)
    return run_snowflake_access_check(
        settings, connector_factory=connector_factory, cortex_client=server.client()
    )


def test_snowflake_access_check_reports_configured_objects_and_cortex_access() -> None:
    cursor = AccessCheckCursor()
    connection = FakeConnection(cursor=cursor)
    cortex = FakeCortexServer(_cortex_canary_reply)

    report = _run(_settings(), lambda **_: connection, cortex=cortex)

    assert report.ok
    assert {check.name for check in report.checks} == {
        "connection_context",
        "warehouse",
        "target_tables",
        "target_table_columns",
        "embedding_dimension",
        "document_stage",
        "bulk_stage",
        "service_spec_stage",
        "compute_pool",
        "image_repository",
        "cortex_llm",
        "cortex_ocr",
        "cortex_embedding",
    }
    assert any(sql.startswith("LIST @KG_DB.GRAPH.KG_DOCS/incoming") for sql, _ in cursor.executed)
    assert any(sql == "SHOW WAREHOUSES LIKE 'WH'" for sql, _ in cursor.executed)
    assert any(
        sql == "SHOW COMPUTE POOLS LIKE 'KG\\\\_PROCESSOR\\\\_CPU\\\\_POOL'"
        for sql, _ in cursor.executed
    )
    assert any(
        sql == "SHOW IMAGE REPOSITORIES LIKE 'KG\\\\_IMAGES' IN SCHEMA KG_DB.GRAPH"
        for sql, _ in cursor.executed
    )
    assert [request.url.path for request in cortex.requests] == [
        "/api/v2/cortex/inference:complete",
        "/api/v2/cortex/inference:embed",
    ]
    assert cortex.bodies[0]["model"] == "llama3.3-70b"
    assert cortex.bodies[0]["max_tokens"] == 8
    assert cortex.bodies[1] == {
        "model": "snowflake-arctic-embed-l-v2.0",
        "text": ["FlakeGraph access check"],
    }
    parse_call = next(
        (sql, params)
        for sql, params in cursor.executed
        if sql.startswith("SELECT AI_PARSE_DOCUMENT")
    )
    assert parse_call[1] is not None
    assert parse_call[1][0:2] == ["@KG_DB.GRAPH.KG_DOCS", "incoming/canary.pdf"]
    assert '"page_filter": [{"end": 1, "start": 0}]' in str(parse_call[1][2])
    assert connection.closed
    assert cursor.closed


def test_snowflake_access_check_reports_missing_table_and_stage_error() -> None:
    cursor = AccessCheckCursor(
        visible_tables=set(REQUIRED_SNOWFLAKE_TABLES) - {"KG_NODE"},
        fail_document_stage=True,
    )

    report = _run(_settings(), lambda **_: FakeConnection(cursor=cursor))

    assert not report.ok
    target_tables = next(check for check in report.checks if check.name == "target_tables")
    document_stage = next(check for check in report.checks if check.name == "document_stage")
    assert target_tables.details["missing"] == ["KG_NODE"]
    assert not document_stage.ok
    assert document_stage.details["message"] == "stage not authorized"


def test_snowflake_access_check_reports_missing_required_column() -> None:
    visible_columns = {table: set(columns) for table, columns in REQUIRED_SNOWFLAKE_COLUMNS.items()}
    visible_columns["KG_NODE"].remove("NAME")
    cursor = AccessCheckCursor(visible_columns=visible_columns)

    report = _run(_settings(), lambda **_: FakeConnection(cursor=cursor))

    assert not report.ok
    target_columns = next(check for check in report.checks if check.name == "target_table_columns")
    assert target_columns.details["missing"] == {"KG_NODE": ["NAME"]}


def test_snowflake_access_check_rejects_unsafe_bulk_stage_location() -> None:
    cursor = AccessCheckCursor()

    report = _run(
        _settings({"snowflake": {"bulk_stage": "@KG_DB.GRAPH.KG_LOAD_STAGE;DROP"}}),
        lambda **_: FakeConnection(cursor=cursor),
    )

    bulk_stage = next(check for check in report.checks if check.name == "bulk_stage")
    assert not bulk_stage.ok
    assert (
        "Snowflake stage locations must use unquoted identifiers" in bulk_stage.details["message"]
    )
    assert not any("LIST @KG_DB.GRAPH.KG_LOAD_STAGE;DROP" in sql for sql, _ in cursor.executed)


def test_snowflake_access_check_reports_connection_failure() -> None:
    def factory(**_kwargs: object) -> FakeConnection:
        raise RuntimeError("login failed")

    report = _run(_settings(), factory)

    assert not report.ok
    assert report.checks[0].name == "connect"
    assert report.checks[0].details["message"] == "login failed"


def test_snowflake_access_check_rejects_system_compute_pool_for_spcs_job() -> None:
    report = _run(
        _settings({"snowflake": {"compute_pool": "SYSTEM_COMPUTE_POOL_GPU"}}),
        lambda **_: FakeConnection(cursor=AccessCheckCursor()),
    )

    compute_pool = next(check for check in report.checks if check.name == "compute_pool")
    assert not compute_pool.ok
    assert compute_pool.details["requires_dedicated_pool"] is True
    assert "not compatible" in compute_pool.message


def _settings(overrides: dict[str, Any] | None = None) -> Settings:
    base: dict[str, Any] = {
        "runtime": {"runtime": "spcs"},
        "job": {
            "job_id": "job-123",
            "graph_id": "graph-123",
            "use_lease": True,
            "lease_owner": "worker-1",
        },
        "files": {"source": "snowflake_stage", "stage_prefix": "incoming"},
        "ocr": {"provider": "snowflake_cortex"},
        "llm": {"provider": "snowflake_cortex", "model": "llama3.3-70b"},
        "embedding": {
            "provider": "snowflake_cortex",
            "model": "snowflake-arctic-embed-l-v2.0",
            "dimension": 1024,
        },
        "writer": {"provider": "snowflake_bulk"},
        "cache": {"provider": "snowflake"},
        "snowflake": {
            "account": "EXAMPLE_ACCOUNT",
            "database": "KG_DB",
            "schema": "GRAPH",
            "role": "KG_PROCESSOR_ROLE",
            "warehouse": "WH",
            "stage": "@KG_DB.GRAPH.KG_DOCS",
            "bulk_stage": "@KG_DB.GRAPH.KG_LOAD_STAGE",
            "image_repository": "KG_DB.GRAPH.KG_IMAGES",
            "compute_pool": "KG_PROCESSOR_CPU_POOL",
            "service_spec_stage": "@KG_DB.GRAPH.KG_SERVICE_SPECS",
        },
    }
    return Settings.load(overrides=_deep_update(base, overrides or {}))


def test_an_embedding_width_the_target_tables_cannot_store_is_reported_before_the_run() -> None:
    """Compare the configured embedding width with the columns that already exist.

    A Snowflake VECTOR column fixes its width at creation and the writer runs
    last, so a mismatch is discovered only after every document has been parsed,
    extracted and summarised, and the whole run is then rejected at the write.
    """

    cursor = AccessCheckCursor(stored_embedding_dimension=1024)
    settings = _settings({"embedding": {"dimension": 768}})

    report = _run(settings, lambda **_: FakeConnection(cursor=cursor))

    check = next(item for item in report.checks if item.name == "embedding_dimension")
    assert not check.ok
    assert not report.ok
    assert check.details["configured"] == 768
    assert check.details["stored"] == dict.fromkeys(REQUIRED_SNOWFLAKE_TABLES, 1024)
    # The message has to name the remedy, since neither width is wrong on its own.
    assert "recreate the tables" in check.message


def test_a_matching_embedding_width_passes_without_comment() -> None:
    """Say nothing when the configured width is the one the tables already store."""

    cursor = AccessCheckCursor(stored_embedding_dimension=1024)

    report = _run(_settings(), lambda **_: FakeConnection(cursor=cursor))

    check = next(item for item in report.checks if item.name == "embedding_dimension")
    assert check.ok
    assert check.details["stored"] == {}


def test_snowflake_access_check_redacts_credentials_echoed_by_a_connector_error() -> None:
    """The access report is persisted and shown, so a DSN in an error must not survive."""

    def factory(**_kwargs: object) -> FakeConnection:
        raise RuntimeError("could not connect to postgresql://kg:hunter2@db.example/kg")

    report = run_snowflake_access_check(_settings(), connector_factory=factory)

    assert "hunter2" not in report.checks[0].details["message"]
    assert "postgresql://kg:***@db.example/kg" in report.checks[0].details["message"]


def test_cortex_canaries_reuse_the_access_check_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an explicit client the canaries borrow the session already open.

    A second sign-in would mean a second browser prompt under SSO.
    """

    connection = FakeConnection(cursor=AccessCheckCursor())
    cortex = FakeCortexServer(_cortex_canary_reply)
    lent: list[object] = []

    class RecordingClient:
        @staticmethod
        def from_config(*_args: object, connection: object = None) -> object:
            lent.append(connection)
            return cortex.client()

    monkeypatch.setattr(snowflake_access, "CortexRestClient", RecordingClient)

    report = run_snowflake_access_check(_settings(), connector_factory=lambda **_: connection)

    assert report.ok
    assert lent == [connection]
    assert len(cortex.requests) == 2
    assert cortex.credentials.closed
    assert connection.closed


def test_a_cortex_rest_refusal_fails_the_canary_with_snowflakes_reason() -> None:
    cortex = FakeCortexServer(
        httpx.Response(403, json={"message": "Account not enabled for REST API"})
    )

    report = _run(
        _settings(), lambda **_: FakeConnection(cursor=AccessCheckCursor()), cortex=cortex
    )

    llm = next(check for check in report.checks if check.name == "cortex_llm")
    assert not report.ok
    assert not llm.ok
    assert "Account not enabled for REST API" in llm.details["message"]
