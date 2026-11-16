"""Check the shim's admission queue against a real PostgreSQL server.

The ordering guarantee is the whole point of putting this queue in a database
rather than in process memory, and it is expressed in SQL — an advisory lock, a
bounded count, and a priority-ordered window. None of that is exercised by a
stub, so it is verified here.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from kg_processor.adapters.distributed.postgres import PostgresDistributedStore
from kg_processor.serving.ocr_shim import OcrQueue

_POSTGRES_DSN = os.getenv("KG_TEST_POSTGRES_DSN")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        not _POSTGRES_DSN,
        reason="Set KG_TEST_POSTGRES_DSN to run live PostgreSQL coordination checks.",
    ),
]


@pytest.fixture
def isolated_postgres_dsn() -> Iterator[str]:
    """Give each test its own schema so parallel runs cannot collide."""

    assert _POSTGRES_DSN is not None
    schema_name = f"ocr_shim_{uuid4().hex}"
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(f'CREATE SCHEMA "{schema_name}"')
    try:
        yield make_conninfo(_POSTGRES_DSN, options=f"-c search_path={schema_name}")
    finally:
        with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
            connection.execute(f'DROP SCHEMA "{schema_name}" CASCADE')


ONE = ("10.0.0.1",)


async def _with_queue(dsn: str, owner: str, body):  # type: ignore[no-untyped-def]
    async with AsyncConnectionPool(
        dsn,
        min_size=1,
        max_size=4,
        open=False,
        kwargs={"row_factory": dict_row, "autocommit": True},
    ) as pool:
        return await body(OcrQueue(pool, owner, stale_after_seconds=60.0))


def _initialize(dsn: str) -> None:
    """Apply the coordination schema the chart's bootstrap Job would apply."""

    PostgresDistributedStore(dsn).initialize()


def _let_every_heartbeat_lapse(dsn: str) -> None:
    """Age every request's heartbeat past the sweep threshold, as a dead shim would."""

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            "UPDATE flakegraph_ocr_request SET heartbeat_at = CURRENT_TIMESTAMP - interval '1 hour'"
        )


def test_a_full_pool_admits_nothing(isolated_postgres_dsn: str) -> None:
    _initialize(isolated_postgres_dsn)

    async def body(queue: OcrQueue) -> list[str | None]:
        await queue.enqueue("first", 100, "batch")
        await queue.enqueue("second", 100, "batch")
        return [
            await queue.try_admit("first", ONE, 1),
            await queue.try_admit("second", ONE, 1),
        ]

    admitted = asyncio.run(_with_queue(isolated_postgres_dsn, "shim-a", body))

    assert admitted == ["10.0.0.1", None]


def test_interactive_work_is_admitted_ahead_of_a_batch_backlog(
    isolated_postgres_dsn: str,
) -> None:
    _initialize(isolated_postgres_dsn)

    async def body(queue: OcrQueue) -> list[str | None]:
        for index in range(5):
            await queue.enqueue(f"batch-{index}", 100, "batch")
        await queue.enqueue("interactive-0", 0, "interactive")
        # One free slot, six waiting. Lower is served first here, matching the
        # inference plane, so the interactive row takes it.
        return [
            await queue.try_admit("batch-0", ONE, 1),
            await queue.try_admit("interactive-0", ONE, 1),
        ]

    admitted = asyncio.run(_with_queue(isolated_postgres_dsn, "shim-a", body))

    assert admitted == [None, "10.0.0.1"]


def test_ordering_holds_across_two_shim_replicas(isolated_postgres_dsn: str) -> None:
    """Two replicas must not each admit into the same free slot."""

    _initialize(isolated_postgres_dsn)

    async def body(queue: OcrQueue) -> None:
        await queue.enqueue("a", 100, "batch")
        await queue.enqueue("b", 100, "batch")

    asyncio.run(_with_queue(isolated_postgres_dsn, "shim-a", body))

    async def admit(name: str) -> str | None:
        async def inner(queue: OcrQueue) -> str | None:
            return await queue.try_admit(name, ONE, 1)

        admitted: str | None = await _with_queue(isolated_postgres_dsn, f"shim-{name}", inner)
        return admitted

    async def race() -> list[str | None]:
        return list(await asyncio.gather(admit("a"), admit("b")))

    results = asyncio.run(race())

    assert results.count("10.0.0.1") == 1 and results.count(None) == 1


def test_two_shims_admitting_together_spread_the_pool(isolated_postgres_dsn: str) -> None:
    """Placement is decided from the fleet's load, so shims do not pile onto one replica.

    Each shim on its own would pick the replica it has sent the least to, and
    with nothing sent yet that is the same replica for both of them.
    """

    _initialize(isolated_postgres_dsn)
    pool = ("10.0.0.1", "10.0.0.2")

    async def body(queue: OcrQueue) -> None:
        for name in ("a", "b", "c", "d", "e"):
            await queue.enqueue(name, 100, "batch")

    asyncio.run(_with_queue(isolated_postgres_dsn, "shim-a", body))

    async def admit(shim: str, name: str) -> str | None:
        async def inner(queue: OcrQueue) -> str | None:
            return await queue.try_admit(name, pool, 2)

        admitted: str | None = await _with_queue(isolated_postgres_dsn, shim, inner)
        return admitted

    async def take_turns() -> list[str | None]:
        return [
            await admit("shim-a", "a"),
            await admit("shim-b", "b"),
            await admit("shim-a", "c"),
            await admit("shim-b", "d"),
            await admit("shim-a", "e"),
        ]

    first, second, third, fourth, fifth = asyncio.run(take_turns())

    # Two replicas at two each: every slot is used once before any replica
    # takes a second request, and the fifth request finds the pool full.
    assert {first, second} == set(pool)
    assert {third, fourth} == set(pool)
    assert fifth is None


def test_a_released_request_frees_its_slot(isolated_postgres_dsn: str) -> None:
    _initialize(isolated_postgres_dsn)

    async def body(queue: OcrQueue) -> list[str | None]:
        await queue.enqueue("first", 100, "batch")
        await queue.enqueue("second", 100, "batch")
        first = await queue.try_admit("first", ONE, 1)
        blocked = await queue.try_admit("second", ONE, 1)
        await queue.release("first")
        return [first, blocked, await queue.try_admit("second", ONE, 1)]

    admitted = asyncio.run(_with_queue(isolated_postgres_dsn, "shim-a", body))

    assert admitted == ["10.0.0.1", None, "10.0.0.1"]


def test_a_dead_replicas_rows_are_reclaimed(isolated_postgres_dsn: str) -> None:
    """A shim that dies mid-request must not hold a slot forever."""

    _initialize(isolated_postgres_dsn)

    async def body(queue: OcrQueue) -> None:
        await queue.enqueue("abandoned", 100, "batch")
        assert await queue.try_admit("abandoned", ONE, 1)

    asyncio.run(_with_queue(isolated_postgres_dsn, "shim-dead", body))

    _let_every_heartbeat_lapse(isolated_postgres_dsn)

    async def survivor(queue: OcrQueue) -> str | None:
        await queue.enqueue("live", 100, "batch")
        return await queue.try_admit("live", ONE, 1)

    assert asyncio.run(_with_queue(isolated_postgres_dsn, "shim-live", survivor))


def test_a_swept_row_is_put_back_by_the_shim_that_still_owns_it(
    isolated_postgres_dsn: str,
) -> None:
    """A heartbeat that merely missed a beat is not a dead shim; its parse is still running."""

    _initialize(isolated_postgres_dsn)

    async def dispatch(queue: OcrQueue) -> None:
        await queue.enqueue("running", 100, "batch")
        assert await queue.try_admit("running", ONE, 1) == "10.0.0.1"

    asyncio.run(_with_queue(isolated_postgres_dsn, "shim-quiet", dispatch))

    _let_every_heartbeat_lapse(isolated_postgres_dsn)

    async def sweep(queue: OcrQueue) -> str | None:
        return await queue.try_admit("nothing-waiting", ONE, 1)

    asyncio.run(_with_queue(isolated_postgres_dsn, "shim-other", sweep))

    async def heartbeat(queue: OcrQueue) -> None:
        await queue.renew("running", 100, "batch", "dispatched", "10.0.0.1")

    asyncio.run(_with_queue(isolated_postgres_dsn, "shim-quiet", heartbeat))

    with psycopg.connect(isolated_postgres_dsn, row_factory=dict_row) as connection:
        rows = connection.execute(
            "SELECT id, status, replica, shim_owner FROM flakegraph_ocr_request"
        ).fetchall()

    assert rows == [
        {"id": "running", "status": "dispatched", "replica": "10.0.0.1", "shim_owner": "shim-quiet"}
    ]
