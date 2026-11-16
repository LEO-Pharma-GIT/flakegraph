from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from snowflake_fakes import CONFIG, FakeConnection, FakeCursor

from kg_processor.adapters.files.common import build_local_input_file
from kg_processor.adapters.files.manifest import ManifestFileSource
from kg_processor.adapters.files.snowflake_stage import SnowflakeStageFileSource
from kg_processor.config.settings import Settings
from kg_processor.domain.ids import stable_id
from kg_processor.factories import build_file_source


def test_snowflake_stage_file_source_lists_and_filters_supported_files() -> None:
    connection = FakeConnection(
        result_sets=[
            [
                {
                    "name": "DB.SCHEMA.DOC_STAGE/input/a.pdf",
                    "size": 42,
                    "md5": "checksum-a",
                },
                ("DB.SCHEMA.DOC_STAGE/input/notes.txt", 12, "checksum-b"),
                ("DB.SCHEMA.DOC_STAGE/input/image.png", 10, "checksum-c"),
            ]
        ]
    )
    source = SnowflakeStageFileSource(
        CONFIG,
        "@DB.SCHEMA.DOC_STAGE",
        prefix="input",
        include_globs=["**/*.pdf", "input/*.txt"],
        connector_factory=lambda **_: connection,
        content_hash=False,
    )

    files = source.list_files()

    assert [file.path for file in files] == [Path("input/a.pdf"), Path("input/notes.txt")]
    assert files[0].source_uri == "@DB.SCHEMA.DOC_STAGE/input/a.pdf"
    assert files[0].mime_type == "application/pdf"
    assert connection.cursor_instance.executed == [("LIST @DB.SCHEMA.DOC_STAGE/input", None)]
    assert connection.cursor_instance.closed
    assert not connection.closed
    source.close()
    assert connection.closed


def test_snowflake_stage_file_source_skips_macos_appledouble_sidecars() -> None:
    """One corpus must yield the same documents through every file source."""

    connection = FakeConnection(
        result_sets=[
            [
                ("DB.SCHEMA.DOC_STAGE/input/a.pdf", 42, "checksum-a"),
                ("DB.SCHEMA.DOC_STAGE/input/._a.pdf", 4, "checksum-b"),
            ]
        ]
    )
    source = SnowflakeStageFileSource(
        CONFIG,
        "@DB.SCHEMA.DOC_STAGE",
        prefix="input",
        connector_factory=lambda **_: connection,
        content_hash=False,
    )

    assert [file.path for file in source.list_files()] == [Path("input/a.pdf")]


def test_snowflake_stage_file_source_keeps_auto_compressed_documents() -> None:
    """An AUTO_COMPRESS'd upload is the same document, not a different file type."""

    connection = FakeConnection(
        result_sets=[[("DB.SCHEMA.DOC_STAGE/input/a.pdf.gz", 42, "checksum-a")]]
    )
    source = SnowflakeStageFileSource(
        CONFIG,
        "@DB.SCHEMA.DOC_STAGE",
        prefix="input",
        include_globs=["**/*.pdf"],
        connector_factory=lambda **_: connection,
        content_hash=False,
    )

    files = source.list_files()

    assert [file.path for file in files] == [Path("input/a.pdf")]
    assert files[0].source_uri == "@DB.SCHEMA.DOC_STAGE/input/a.pdf.gz"
    assert files[0].mime_type == "application/pdf"


def test_snowflake_stage_file_source_matches_root_files_with_recursive_globs() -> None:
    connection = FakeConnection(
        result_sets=[
            [
                ("DB.SCHEMA.DOC_STAGE/root.pdf", 42, "checksum-a"),
                ("DB.SCHEMA.DOC_STAGE/nested/child.pdf", 12, "checksum-b"),
                ("DB.SCHEMA.DOC_STAGE/root.png", 10, "checksum-c"),
            ]
        ]
    )
    source = SnowflakeStageFileSource(
        CONFIG,
        "@DB.SCHEMA.DOC_STAGE",
        include_globs=["**/*.pdf"],
        connector_factory=lambda **_: connection,
        content_hash=False,
    )

    files = source.list_files()

    assert [file.path for file in files] == [Path("nested/child.pdf"), Path("root.pdf")]


class DownloadingCursor(FakeCursor):
    """Emulate Snowflake GET by materializing the staged bytes in the target dir."""

    def __init__(self, rows: list[object], payloads: dict[str, bytes]) -> None:
        super().__init__(result_sets=[rows])
        self.payloads = payloads
        self.get_statements: list[str] = []

    def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
        *,
        timeout: int | None = None,
    ) -> object:
        super().execute(sql, params, timeout=timeout)
        if sql.startswith("GET "):
            self.get_statements.append(sql)
            _, source_uri, destination = sql.split(" ", 2)
            target = Path(destination.removeprefix("file://"))
            target.mkdir(parents=True, exist_ok=True)
            self.materialize(source_uri, target)
        return None

    def materialize(self, source_uri: str, target: Path) -> None:
        (target / Path(source_uri).name).write_bytes(self.payloads.get(source_uri, b""))


class GzipCursor(DownloadingCursor):
    """A stage that AUTO_COMPRESS'd its objects hands GET a gzip member, named ``.gz``."""

    def materialize(self, source_uri: str, target: Path) -> None:
        name = Path(source_uri).name
        with gzip.open(target / (name if name.endswith(".gz") else f"{name}.gz"), "wb") as handle:
            handle.write(self.payloads[source_uri])


def _staged_source(payload: bytes) -> SnowflakeStageFileSource:
    uri = "@DB.SCHEMA.DOC_STAGE/input/a.txt"
    cursor = DownloadingCursor(
        [("DB.SCHEMA.DOC_STAGE/input/a.txt", len(payload), "0123456789abcdef0123456789abcdef")],
        {uri: payload},
    )
    return SnowflakeStageFileSource(
        CONFIG,
        "@DB.SCHEMA.DOC_STAGE",
        prefix="input",
        connector_factory=lambda **_: FakeConnection(cursor=cursor),
    )


def test_staged_files_hash_their_own_bytes_not_the_provider_value() -> None:
    """Snowflake reports MD5; every other source records a SHA-256 of the bytes.

    Adopting the provider value gives one document two identities depending on
    which runtime ingested it, which silently breaks any comparison that binds a
    document by checksum.
    """

    payload = b"martial arts corpus fixture"
    expected = hashlib.sha256(payload).hexdigest()

    file = _staged_source(payload).list_files()[0]

    assert file.checksum == expected
    assert file.provider_checksum == "0123456789abcdef0123456789abcdef"


def test_identity_follows_the_content_hash() -> None:
    """File identity derives from the checksum, so it must be rebuilt with it."""

    payload = b"martial arts corpus fixture"
    file = _staged_source(payload).list_files()[0]

    assert file.id == stable_id("stage_file", file.source_uri, file.checksum)


def test_local_and_staged_sources_agree_on_the_same_bytes(tmp_path: Path) -> None:
    """The reported defect: the same document must not have two identities."""

    payload = b"martial arts corpus fixture"
    local_path = tmp_path / "a.txt"
    local_path.write_bytes(payload)

    local_file = build_local_input_file(local_path)
    staged_file = _staged_source(payload).list_files()[0]

    assert staged_file.checksum == local_file.checksum


def test_manifest_and_staged_sources_agree_on_the_same_bytes(tmp_path: Path) -> None:
    """Manifest ingestion must produce the same identity as the Snowflake stage.

    The gold fixtures are keyed to manifest checksums, so a divergence here is
    what silently zeroes evidence grounding for a Snowflake-built graph.
    """

    payload = b"martial arts corpus fixture"
    corpus = tmp_path / "files"
    corpus.mkdir()
    (corpus / "a.txt").write_bytes(payload)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"path": "files/a.txt", "mime_type": "text/plain"}) + "\n",
        encoding="utf-8",
    )

    manifest_file = ManifestFileSource(manifest).list_files()[0]
    staged_file = _staged_source(payload).list_files()[0]

    assert manifest_file.checksum == staged_file.checksum
    assert manifest_file.checksum == hashlib.sha256(payload).hexdigest()


def test_content_hashing_can_be_disabled_for_cost() -> None:
    """Opting out keeps the provider value, and must not claim to be a content hash."""

    payload = b"martial arts corpus fixture"
    uri = "@DB.SCHEMA.DOC_STAGE/input/a.txt"
    cursor = DownloadingCursor(
        [("DB.SCHEMA.DOC_STAGE/input/a.txt", len(payload), "0123456789abcdef0123456789abcdef")],
        {uri: payload},
    )
    source = SnowflakeStageFileSource(
        CONFIG,
        "@DB.SCHEMA.DOC_STAGE",
        prefix="input",
        connector_factory=lambda **_: FakeConnection(cursor=cursor),
        content_hash=False,
    )

    file = source.list_files()[0]

    assert file.checksum == "0123456789abcdef0123456789abcdef"
    assert file.provider_checksum is None
    assert cursor.get_statements == []


def test_claimed_workers_do_not_rehash_the_whole_stage() -> None:
    """A claim carries the authoritative checksum, so recomputing it is wasted I/O.

    Without this, every worker would download the entire stage on every batch to
    produce a value the claim immediately overwrites.
    """

    settings = Settings.model_validate(
        {
            "files": {"source": "snowflake_stage"},
            "snowflake": {
                "account": "a",
                "database": "DB",
                "schema": "SCHEMA",
                "stage": "@DB.SCHEMA.DOC_STAGE",
            },
        }
    )

    claimed = build_file_source(settings, content_hash=False)
    discovering = build_file_source(settings)

    assert isinstance(claimed, SnowflakeStageFileSource)
    assert isinstance(discovering, SnowflakeStageFileSource)
    assert claimed.content_hash is False
    assert discovering.content_hash is True


def test_provider_checksum_is_not_used_as_identity() -> None:
    """Provenance must never leak back into identity once both are recorded."""

    payload = b"martial arts corpus fixture"
    file = _staged_source(payload).list_files()[0]

    assert file.provider_checksum is not None
    assert file.provider_checksum != file.checksum
    assert file.provider_checksum not in file.id


def test_compressed_stage_files_hash_their_original_bytes() -> None:
    """AUTO_COMPRESS stores a gzipped copy; the hash must describe the document.

    Hashing the stored bytes would describe how the stage chose to keep the file
    rather than the file itself, and would not match it read anywhere else.
    """

    payload = b"martial arts corpus fixture"
    uri = "@DB.SCHEMA.DOC_STAGE/input/a.txt"
    cursor = GzipCursor(
        [("DB.SCHEMA.DOC_STAGE/input/a.txt", len(payload), "0123456789abcdef0123456789abcdef")],
        {uri: payload},
    )
    source = SnowflakeStageFileSource(
        CONFIG,
        "@DB.SCHEMA.DOC_STAGE",
        prefix="input",
        connector_factory=lambda **_: FakeConnection(cursor=cursor),
    )

    assert source.list_files()[0].checksum == hashlib.sha256(payload).hexdigest()


def test_auto_compressed_stage_paths_hash_the_document_they_hold() -> None:
    """A ``.gz`` stage path is the AUTO_COMPRESS case its own handling describes.

    Hashing the gzip container would give the same document one identity through
    a stage and another through every other source.
    """

    payload = b"martial arts corpus fixture"
    uri = "@DB.SCHEMA.DOC_STAGE/input/a.txt.gz"
    cursor = GzipCursor(
        [("DB.SCHEMA.DOC_STAGE/input/a.txt.gz", len(payload), "0123456789abcdef0123456789abcdef")],
        {uri: payload},
    )
    source = SnowflakeStageFileSource(
        CONFIG,
        "@DB.SCHEMA.DOC_STAGE",
        prefix="input",
        connector_factory=lambda **_: FakeConnection(cursor=cursor),
    )

    file = source.list_files()[0]

    assert file.source_uri == uri
    assert file.checksum == hashlib.sha256(payload).hexdigest()
