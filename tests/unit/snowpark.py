"""One Snowpark session fake for every app control-plane unit test.

The fakes model exactly the Snowpark surface the Snowflake backend drives. A
method the backend calls that is not modelled here fails with AttributeError
instead of quietly succeeding, so a new session call cannot pass a test by
accident.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, NamedTuple


class Row:
    """One Snowpark result row addressed by column name."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self.values = dict(values)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)

    def __iter__(self) -> Iterator[Any]:
        # A Snowpark row is a tuple of its values, which is how the backend
        # reads a row it has no column names for.
        return iter(self.values.values())


class Query:
    """A prepared statement that yields a fixed result."""

    def __init__(self, rows: Sequence[Row]) -> None:
        self.rows = list(rows)

    def limit(self, count: int) -> Query:
        return Query(self.rows[:count])

    def collect(self) -> list[Row]:
        return list(self.rows)

    def to_local_iterator(self) -> Iterator[Row]:
        return iter(self.rows)


Answer = Sequence[tuple[str, Sequence[Row]]] | Callable[[str, Any], Sequence[Row]]


class PutResult(NamedTuple):
    """Snowpark's upload result, a NamedTuple a naive tuple check would split."""

    source: str
    target: str
    status: str


class StageFiles:
    """The ``session.file`` API, recording what was streamed to which location."""

    def __init__(self, put_status: str = "UPLOADED") -> None:
        self.put_status = put_status
        self.uploads: list[tuple[bytes, str]] = []

    def put_stream(
        self, stream: Any, location: str, *, auto_compress: bool, overwrite: bool
    ) -> PutResult:
        # Uncompressed so the staged object is the file itself, and overwritten
        # so a retry replaces what the last attempt left behind.
        assert (auto_compress, overwrite) == (False, True)
        self.uploads.append((stream.read(), location))
        return PutResult(location.rsplit("/", 1)[-1], location, self.put_status)


class RecordingSession:
    """A Snowpark session that records every statement and answers from a table.

    ``answer`` is either ``(substring, rows)`` pairs, where the first substring
    found in a statement decides its rows and no match answers nothing, or a
    callable taking the statement and its parameters. Every upload through
    ``file`` reports ``put_status``.
    """

    def __init__(self, answer: Answer | None = None, *, put_status: str = "UPLOADED") -> None:
        self.answer: Answer = answer if answer is not None else []
        self.statements: list[tuple[str, Any]] = []
        self.file = StageFiles(put_status)

    @property
    def statement_texts(self) -> list[str]:
        return [statement for statement, _params in self.statements]

    def sql(self, statement: str, params: Any = None) -> Query:
        self.statements.append((statement, params))
        if callable(self.answer):
            return Query(self.answer(statement, params))
        return Query(next((rows for needle, rows in self.answer if needle in statement), []))


class _Batch:
    """A dataframe of submission rows, and the writer that saves it."""

    def __init__(self, session: SubmissionSession, rows: Sequence[tuple[str, ...]]) -> None:
        self.session = session
        self.rows = list(rows)
        self.write = self

    def mode(self, mode: str) -> _Batch:
        assert mode == "overwrite"
        return self

    def save_as_table(self, name: str, *, table_type: str) -> None:
        # Stored procedures reject temporary tables; transient works in both contexts.
        assert table_type == "transient"
        self.session.saved_tables.append(name)
        self.session.batch_sizes.append(len(self.rows))


class SubmissionSession(RecordingSession):
    """A session for SPCS submission: LIST answers ``objects``, nothing else answers.

    Every merged batch leaves its table name in ``saved_tables`` and its row
    count in ``batch_sizes``.
    """

    def __init__(self, objects: Sequence[Row], *, put_status: str = "UPLOADED") -> None:
        super().__init__([("LIST ", objects)], put_status=put_status)
        self.saved_tables: list[str] = []
        self.batch_sizes: list[int] = []

    def create_dataframe(self, rows: Sequence[tuple[str, ...]], schema: Sequence[str]) -> _Batch:
        assert schema, "the MERGE addresses the batch by column name"
        return _Batch(self, rows)


class RecordingManager:
    """The job manager a progress sink writes through, keeping every payload."""

    def __init__(self) -> None:
        self.written: list[dict[str, Any]] = []
        self.file_stages: list[tuple[list[str], str]] = []

    def update_job_progress(
        self, job_id: str, graph_id: str, worker_id: str, payload: dict[str, Any]
    ) -> None:
        self.written.append(payload)

    def update_job_file_progress(
        self, job_id: str, graph_id: str, worker_id: str, file_ids: list[str], stage: str
    ) -> None:
        self.file_stages.append((file_ids, stage))


class Upload:
    """One console upload selection entry, counting how often it is read."""

    name = "martial-arts.pdf"
    size = 7
    file_id = "upload-1"

    def __init__(self) -> None:
        self.reads = 0

    def getvalue(self) -> bytes:
        self.reads += 1
        return b"content"
