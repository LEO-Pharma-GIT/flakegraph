# SPDX-License-Identifier: Apache-2.0
"""Hold, order, and admit document-parsing work in front of a MinerU pool.

MinerU answers **409 when it is busy** and FlakeGraph's HTTP transport does not
retry, so without something in between a saturated pool does not slow down — it
drops documents. The shim exists to hold that work instead.

Holding is also the only way to enforce priority on this plane. There is no
scheduler inside MinerU to hand a band to, so ordering has to happen before
dispatch: a waiting request goes into PostgreSQL and is admitted only when the
pool has room. PostgreSQL rather than process memory because ordering has to hold
*across* shim replicas — two replicas each ordering their own callers correctly
still serves them in the wrong order relative to each other — and because a
restart should not lose work that a client is still waiting on.

Bands follow the same convention as the inference plane: **lower is served
first**. The pipeline's own task queue orders the other way; these are different
queues and the shim shares its vocabulary with the sidecar, not with the planner.

Tracking how much of the pool is busy is what admission control requires anyway,
so dispatching to the least-loaded replica costs nothing extra. The replica is
chosen where the count is kept, in the queue: a shim that picked from its own
bookkeeping would agree with every other shim on the emptiest replica and send
it the whole fleet's admissions at once.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field
from starlette.requests import ClientDisconnect

from kg_processor.serving.priority import ConsumerKeyring, load_keyring, presented_key

logger = logging.getLogger(__name__)

# Every admission decision is made under this lock, so counting the busy pool and
# claiming a slot cannot interleave between replicas and overshoot capacity.
ADMISSION_LOCK_KEY = 0x0CD5_11
# ``/health`` and a scrape of ``/metrics`` have their own routes, declared
# before the catch-all; ``/metrics`` stays listed so a probe with another
# method is answered the same way ``/ping`` is, rather than asked for a key.
UNAUTHENTICATED_PATHS = frozenset({"/ping", "/metrics"})
# The two states a queued request can be in, as the queue table's CHECK
# constraint spells them.
QUEUE_STATUSES = ("waiting", "dispatched")

# The parsing pool is built pipeline-only: the image installs mineru[pipeline],
# which does not pull in `accelerate`. MinerU's own server default is
# `hybrid-engine`, a VLM backend, so a caller who simply omits `backend` reaches
# a path this image cannot run - and finds out only after the pool has spent
# several minutes downloading a 2.3 GB model, as an opaque 409. The shim knows
# what the pool was built for, so it settles the question here.
PARSE_ROUTE = "/file_parse"
SUPPORTED_PARSE_BACKENDS = frozenset({"pipeline"})
DEFAULT_PARSE_BACKEND = "pipeline"

# Metric labels stay a small fixed vocabulary. Consumer classes come from the
# band configuration, replicas from DNS, and every other value is one of the
# constants below - never a key, a request id or a file name, each of which
# would mint a series per value.
UNAUTHENTICATED_CLASS = "unauthenticated"
OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_UPSTREAM_ERROR = "upstream_error"
OUTCOME_UPSTREAM_TIMEOUT = "upstream_timeout"
OUTCOME_REJECTED = "rejected"
OUTCOME_CLIENT_GONE = "client_gone"
OUTCOME_QUEUE_ERROR = "queue_error"
# A document can legitimately wait many minutes behind a batch run, and a parse
# of a long scan takes minutes of its own; the default buckets stop at ten
# seconds and would flatten both into one bin.
QUEUE_WAIT_BUCKETS = (0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800)
PARSE_DURATION_BUCKETS = (1, 2, 5, 10, 20, 30, 60, 120, 300, 600)


class OcrShimConfig(BaseModel):
    """Configure the shim from the environment the pod provides."""

    database_url: str
    upstream_host: str
    upstream_port: int = Field(default=8080, gt=0, lt=65536)
    # How many requests one parsing replica accepts before it starts answering
    # 409. The shim never admits more than this times the replicas it can see.
    upstream_capacity: int = Field(default=8, gt=0)
    listen_host: str = "0.0.0.0"
    listen_port: int = Field(default=8080, gt=0, lt=65536)
    keys_file: Path = Path("/etc/flakegraph/ocr/ocr-keys.json")
    bands: dict[str, int] | None = None
    poll_interval_seconds: float = Field(default=0.5, gt=0)
    request_timeout_seconds: float = Field(default=3600.0, gt=0)
    # A waiting or dispatched row whose owner stopped renewing it is reclaimed,
    # so a shim that dies mid-request does not permanently consume a slot.
    stale_after_seconds: float = Field(default=60.0, gt=0)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> OcrShimConfig:
        """Build a configuration from ``FLAKEGRAPH_OCR_SHIM_*`` variables."""

        env = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        for field, variable in (
            ("database_url", "FLAKEGRAPH_OCR_SHIM_DATABASE_URL"),
            ("upstream_host", "FLAKEGRAPH_OCR_SHIM_UPSTREAM_HOST"),
            ("upstream_port", "FLAKEGRAPH_OCR_SHIM_UPSTREAM_PORT"),
            ("upstream_capacity", "FLAKEGRAPH_OCR_SHIM_UPSTREAM_CAPACITY"),
            ("listen_host", "FLAKEGRAPH_OCR_SHIM_LISTEN_HOST"),
            ("listen_port", "FLAKEGRAPH_OCR_SHIM_LISTEN_PORT"),
            ("keys_file", "FLAKEGRAPH_OCR_SHIM_KEYS_FILE"),
            ("poll_interval_seconds", "FLAKEGRAPH_OCR_SHIM_POLL_INTERVAL_SECONDS"),
            ("request_timeout_seconds", "FLAKEGRAPH_OCR_SHIM_REQUEST_TIMEOUT_SECONDS"),
            ("stale_after_seconds", "FLAKEGRAPH_OCR_SHIM_STALE_AFTER_SECONDS"),
        ):
            if variable in env:
                values[field] = env[variable]
        if "FLAKEGRAPH_OCR_SHIM_BANDS" in env:
            values["bands"] = json.loads(env["FLAKEGRAPH_OCR_SHIM_BANDS"])
        return cls.model_validate(values)


class UpstreamPool:
    """Resolve parsing replicas and count what this process sent to each.

    The shim is configured with one name, not a list of addresses, so the pool
    can autoscale underneath it without a configuration change. Resolution is a
    DNS lookup of that name, which is how any client-side balancer finds its
    backends — the shim still learns nothing about the cluster it runs in.

    The counts are this process's share of the fleet's load, kept for its
    metrics; the fleet-wide count that admission and placement decide on lives
    in the queue.
    """

    def __init__(
        self,
        host: str,
        port: int,
        capacity_per_replica: int,
        resolver: Callable[[], Awaitable[tuple[str, ...]]] | None = None,
    ) -> None:
        """Record the pool's name and the load each replica will accept."""

        self._host = host
        self._port = port
        self._capacity_per_replica = capacity_per_replica
        self._resolver = resolver
        self._in_flight: dict[str, int] = {}
        self._endpoints: tuple[str, ...] = ()

    async def _resolve(self) -> tuple[str, ...]:
        """Return the pool's current addresses, or what was last seen on failure."""

        if self._resolver is not None:
            return await self._resolver()
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(self._host, self._port, type=socket.SOCK_STREAM)
        except socket.gaierror:
            logger.warning("could not resolve parsing pool %s", self._host)
            return self._endpoints
        return tuple(sorted({str(info[4][0]) for info in infos}))

    async def refresh(self) -> tuple[str, ...]:
        """Re-resolve the pool, forgetting counters for replicas that are gone."""

        resolved = await self._resolve()
        if resolved:
            self._endpoints = resolved
            self._in_flight = {
                address: self._in_flight.get(address, 0) for address in self._endpoints
            }
        return self._endpoints

    @property
    def endpoints(self) -> tuple[str, ...]:
        """Return the replica addresses the last resolution found."""

        return self._endpoints

    @property
    def capacity_per_replica(self) -> int:
        """Return the load one replica accepts before it starts answering 409."""

        return self._capacity_per_replica

    @property
    def in_flight(self) -> dict[str, int]:
        """Return how many requests this process has dispatched to each replica."""

        return dict(self._in_flight)

    def dispatched(self, address: str) -> None:
        """Count a request the queue admitted to a replica against it."""

        self._in_flight[address] = self._in_flight.get(address, 0) + 1

    def release(self, address: str) -> None:
        """Return a slot after a request completes, however it completed."""

        if address in self._in_flight:
            self._in_flight[address] = max(0, self._in_flight[address] - 1)

    def base_url(self, address: str) -> str:
        """Return the URL for a resolved replica address."""

        formatted = f"[{address}]" if ":" in address else address
        return f"http://{formatted}:{self._port}"


class OcrQueue:
    """Order waiting requests across every shim replica and admit them in turn."""

    def __init__(self, pool: Any, owner: str, stale_after_seconds: float) -> None:
        """Hold the connection pool and the identity this replica renews under."""

        self._pool = pool
        self._owner = owner
        self._stale_after_seconds = stale_after_seconds

    async def enqueue(self, request_id: str, priority: int, consumer_class: str) -> None:
        """Record a request as waiting, in priority then arrival order."""

        async with self._pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO flakegraph_ocr_request
                    (id, priority, consumer_class, status, shim_owner)
                VALUES (%s, %s, %s, 'waiting', %s)
                """,
                (request_id, priority, consumer_class, self._owner),
            )

    async def try_admit(
        self, request_id: str, replicas: Sequence[str], capacity_per_replica: int
    ) -> str | None:
        """Claim the least-loaded replica when there is room and nothing better waits.

        Counting the busy pool and claiming a slot happen inside one transaction
        holding the advisory lock. Without that, two replicas could both read the
        same free-slot count and both dispatch into it. The replica is chosen
        under the same lock and from the same count, so two shims admitting at
        once are spread across the pool rather than both sent to its emptiest
        member.
        """

        async with self._pool.connection() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(%s)", (ADMISSION_LOCK_KEY,))
            await connection.execute(
                """
                DELETE FROM flakegraph_ocr_request
                WHERE heartbeat_at < CURRENT_TIMESTAMP - make_interval(secs => %s)
                """,
                (self._stale_after_seconds,),
            )
            cursor = await connection.execute(
                """
                SELECT replica, count(*) AS busy FROM flakegraph_ocr_request
                WHERE status = 'dispatched'
                GROUP BY replica
                """
            )
            busy = {str(row["replica"]): int(row["busy"]) for row in await cursor.fetchall()}
            free = sum(max(0, capacity_per_replica - busy.get(r, 0)) for r in replicas)
            if free <= 0:
                return None
            cursor = await connection.execute(
                """
                SELECT id FROM flakegraph_ocr_request
                WHERE status = 'waiting'
                ORDER BY priority, created_at, id
                LIMIT %s
                """,
                (free,),
            )
            admissible = {record["id"] for record in await cursor.fetchall()}
            if request_id not in admissible:
                return None
            replica = min(replicas, key=lambda r: (busy.get(r, 0), r))
            await connection.execute(
                """
                UPDATE flakegraph_ocr_request
                SET status = 'dispatched', heartbeat_at = CURRENT_TIMESTAMP, replica = %s
                WHERE id = %s
                """,
                (replica, request_id),
            )
            return replica

    async def renew(
        self,
        request_id: str,
        priority: int,
        consumer_class: str,
        status: str,
        replica: str | None = None,
    ) -> None:
        """Keep a row alive, or put it back if it was swept while still owned.

        The sweep in ``try_admit`` reclaims any row that went quiet, and it
        cannot tell a dead shim from a live one whose heartbeat missed a beat
        — a queue-database failover is enough. The work is still real either
        way: a client is still waiting, or a parse is still running. Restoring
        the row keeps that work counted so the rest of the fleet does not
        admit into a slot that is in fact taken.
        """

        async with self._pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO flakegraph_ocr_request
                    (id, priority, consumer_class, status, shim_owner, replica)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET heartbeat_at = CURRENT_TIMESTAMP
                """,
                (request_id, priority, consumer_class, status, self._owner, replica),
            )

    async def release(self, request_id: str) -> None:
        """Drop a finished request, freeing its slot for whatever waits next."""

        async with self._pool.connection() as connection:
            await connection.execute(
                "DELETE FROM flakegraph_ocr_request WHERE id = %s", (request_id,)
            )

    async def depth(self) -> list[QueueDepth]:
        """Count what every shim replica is holding, by class and status.

        One aggregate over the whole table rather than this replica's rows: the
        queue is shared, so the depth that matters to an operator is the fleet's.
        """

        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT consumer_class, status, count(*) AS depth,
                       EXTRACT(EPOCH FROM CURRENT_TIMESTAMP - min(created_at)) AS oldest_seconds
                FROM flakegraph_ocr_request
                GROUP BY consumer_class, status
                """
            )
            return [
                QueueDepth(
                    str(row["consumer_class"]),
                    str(row["status"]),
                    int(row["depth"]),
                    float(row["oldest_seconds"]),
                )
                for row in await cursor.fetchall()
            ]


class QueueDepth(NamedTuple):
    """One row of the fleet-wide queue aggregate."""

    consumer_class: str
    status: str
    depth: int
    oldest_seconds: float


class _QueueDepthCollector(Collector):
    """Expose the last queue aggregate the scrape handler managed to fetch.

    Collection is synchronous and the database pool is not, so the query runs in
    the async handler before rendering and this collector only reads what it
    left. When that query failed there is nothing to read, and the series are
    absent rather than reporting a stale or invented depth.

    An empty queue is the other case, and it must not look like a failed one:
    the aggregate returns no rows, so every class the keyring knows is written
    out at zero. A panel then draws a flat line rather than "no data", and an
    alert on the oldest wait has a value to compare.
    """

    def __init__(self, consumer_classes: Iterable[str]) -> None:
        """Start with nothing fetched, which exposes no depth series at all."""

        self._consumer_classes = tuple(consumer_classes)
        self.snapshot: list[QueueDepth] | None = None

    def collect(self) -> Iterable[Metric]:
        """Render the snapshot as gauges, one series per class and status."""

        if self.snapshot is None:
            return []
        depth = GaugeMetricFamily(
            "flakegraph_ocr_queue_depth",
            "Requests held in the shared queue across every shim replica.",
            labels=["consumer_class", "status"],
        )
        oldest = GaugeMetricFamily(
            "flakegraph_ocr_oldest_waiting_seconds",
            "Age of the longest-waiting request not yet dispatched.",
            labels=["consumer_class"],
        )
        depths = {(row.consumer_class, row.status): row for row in self.snapshot}
        classes = dict.fromkeys(self._consumer_classes)
        classes.update(dict.fromkeys(row.consumer_class for row in self.snapshot))
        for consumer_class in classes:
            for status in QUEUE_STATUSES:
                row = depths.get((consumer_class, status))
                depth.add_metric([consumer_class, status], row.depth if row else 0)
            waiting = depths.get((consumer_class, "waiting"))
            oldest.add_metric([consumer_class], waiting.oldest_seconds if waiting else 0.0)
        return [depth, oldest]


class _UpstreamPoolCollector(Collector):
    """Read the dispatch bookkeeping the pool already keeps, at scrape time.

    Admission control has to know how loaded each replica is, so the pool holds
    exactly the numbers a saturation panel needs and nothing has to be counted
    twice.
    """

    def __init__(self, upstreams: UpstreamPool) -> None:
        """Hold the pool whose state is reported."""

        self._upstreams = upstreams

    def collect(self) -> Iterable[Metric]:
        """Report per-replica load and capacity alongside the replica count."""

        in_flight = GaugeMetricFamily(
            "flakegraph_ocr_in_flight",
            "Parses this shim replica currently has dispatched to each parsing replica.",
            labels=["replica"],
        )
        capacity = GaugeMetricFamily(
            "flakegraph_ocr_replica_capacity",
            "Concurrent parses the shim allows each parsing replica.",
            labels=["replica"],
        )
        load = self._upstreams.in_flight
        for address, count in load.items():
            in_flight.add_metric([address], count)
            capacity.add_metric([address], self._upstreams.capacity_per_replica)
        replicas = GaugeMetricFamily(
            "flakegraph_ocr_upstream_replicas",
            "Parsing replicas the shim currently resolves.",
            value=len(load),
        )
        return [in_flight, capacity, replicas]


class OcrShimMetrics:
    """Own the series one shim process exposes.

    Each application gets its own registry rather than the process-wide default,
    which would refuse a second application in the same process and would drag
    the Python runtime collectors into a scrape that has no use for them.
    """

    def __init__(self, upstreams: UpstreamPool, consumer_classes: Iterable[str]) -> None:
        """Register every series up front so an idle shim still exposes them."""

        self.registry = CollectorRegistry()
        self.requests = Counter(
            "flakegraph_ocr_requests_total",
            "Requests answered by the shim, by how they ended.",
            ["consumer_class", "outcome"],
            registry=self.registry,
        )
        self.queue_wait = Histogram(
            "flakegraph_ocr_queue_wait_seconds",
            "Time a request was held from being queued to being dispatched.",
            ["consumer_class"],
            buckets=QUEUE_WAIT_BUCKETS,
            registry=self.registry,
        )
        self.parse_duration = Histogram(
            "flakegraph_ocr_parse_duration_seconds",
            "Time from dispatching a parse to receiving the last byte of its answer.",
            ["replica"],
            buckets=PARSE_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._depth = _QueueDepthCollector(consumer_classes)
        self.registry.register(self._depth)
        self.registry.register(_UpstreamPoolCollector(upstreams))

    async def refresh_depth(self, queue: OcrQueue) -> None:
        """Fetch the fleet-wide queue aggregate, or expose none if that fails.

        A scrape must not fail because the queue database blinked: everything
        this process knows on its own is still worth having, and a gap in the
        depth series is itself the signal that the database was unreachable.
        """

        try:
            self._depth.snapshot = await queue.depth()
        except Exception:
            logger.warning("queue depth is unavailable for this scrape", exc_info=True)
            self._depth.snapshot = None

    def render(self) -> Response:
        """Serialise every series in the text exposition format."""

        return Response(generate_latest(self.registry), media_type=CONTENT_TYPE_LATEST)


def create_app(
    config: OcrShimConfig,
    keyring: ConsumerKeyring | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    queue: OcrQueue | None = None,
    upstreams: UpstreamPool | None = None,
) -> FastAPI:
    """Build the shim application in front of one parsing pool.

    The queue and the pool are the two systems outside this process, so both are
    injectable at the composition root rather than reached for internally.
    """

    resolved = keyring if keyring is not None else load_keyring(config.keys_file, config.bands)
    pool_view = upstreams or UpstreamPool(
        config.upstream_host, config.upstream_port, config.upstream_capacity
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Open the database pool and one upstream client for the process."""

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout_seconds),
            transport=transport,
        ) as client:
            app.state.client = client
            if queue is not None:
                app.state.queue = queue
                await pool_view.refresh()
                yield
                return
            async with AsyncConnectionPool(
                config.database_url,
                min_size=1,
                max_size=8,
                open=False,
                # A connection can die while it sits idle in the pool - a
                # server restart, a failover, an operator terminating backends -
                # and the pool does not notice on its own: it hands the dead
                # connection out and the request that receives it fails, once,
                # for a reason that has nothing to do with the request. Checking
                # at checkout costs a round trip and turns that into a reconnect.
                check=AsyncConnectionPool.check_connection,
                kwargs={"row_factory": dict_row, "autocommit": True},
            ) as connections:
                app.state.queue = OcrQueue(connections, _owner_id(), config.stale_after_seconds)
                await pool_view.refresh()
                yield

    app = FastAPI(
        title="FlakeGraph OCR shim",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.keyring = resolved
    app.state.upstreams = pool_view
    metrics = OcrShimMetrics(pool_view, resolved.bands)
    app.state.metrics = metrics

    @app.get("/health")
    async def health() -> JSONResponse:
        """Report readiness without touching the parsing pool."""

        return JSONResponse({"status": "ok", "replicas": len(await pool_view.refresh())})

    @app.get("/metrics")
    async def scrape(request: Request) -> Response:
        """Refresh what has to be fetched from outside the process, then render."""

        await pool_view.refresh()
        await metrics.refresh_depth(request.app.state.queue)
        return metrics.render()

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def parse(path: str, request: Request) -> Response:
        """Authenticate, hold the request until the pool has room, then forward."""

        route = f"/{path}"
        if route in UNAUTHENTICATED_PATHS:
            return JSONResponse({"status": "ok"})

        consumer_class = resolved.classify(presented_key(request.headers.get("authorization", "")))
        if consumer_class is None:
            metrics.requests.labels(UNAUTHENTICATED_CLASS, OUTCOME_REJECTED).inc()
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        priority = resolved.priority_for(consumer_class)
        try:
            # A caller can hang up while still uploading, so the read is inside.
            body = await request.body()
            # Settle the backend before the request costs anything. Refusing
            # here turns a several-minute download ending in an opaque 409 into
            # an immediate answer that names the problem, and supplying the
            # default means a caller who followed MinerU's own documentation
            # still works.
            if route == PARSE_ROUTE and request.method == "POST":
                settled = _settle_backend(request.headers.get("content-type", ""), body)
                if isinstance(settled, JSONResponse):
                    metrics.requests.labels(consumer_class, OUTCOME_REJECTED).inc()
                    return settled
                body = settled
            return await _hold_and_forward(
                request, route, body, priority, consumer_class, pool_view, config, metrics
            )
        except ClientDisconnect:
            # Nobody is listening; answering keeps a hang-up out of the error log.
            return Response(status_code=499)

    return app


def run(config: OcrShimConfig | None = None) -> None:
    """Serve the shim until the process is signalled."""

    resolved = config if config is not None else OcrShimConfig.from_env()
    uvicorn.run(
        create_app(resolved),
        host=resolved.listen_host,
        port=resolved.listen_port,
        log_level="info",
    )


async def _hold_and_forward(
    request: Request,
    route: str,
    body: bytes,
    priority: int,
    consumer_class: str,
    upstreams: UpstreamPool,
    config: OcrShimConfig,
    metrics: OcrShimMetrics,
) -> Response:
    """Queue the request, wait for its turn, relay it, and account for the result."""

    held: OcrQueue = request.app.state.queue
    request_id = str(uuid4())
    queued_at = time.perf_counter()
    address: str | None = None
    try:
        await held.enqueue(request_id, priority, consumer_class)
        address = await _await_slot(
            request, held, upstreams, request_id, priority, consumer_class, config
        )
        metrics.queue_wait.labels(consumer_class).observe(time.perf_counter() - queued_at)
        return await _forward(
            request,
            route,
            body,
            address,
            upstreams,
            held,
            request_id,
            metrics,
            priority,
            consumer_class,
            renew_every=config.stale_after_seconds / 3,
        )
    except BaseException as exc:
        # Nothing was dispatched if no replica was claimed, so whatever
        # failed did so while the request sat in the queue.
        outcome = _failure_outcome(exc) if address is not None else OUTCOME_QUEUE_ERROR
        if isinstance(exc, asyncio.CancelledError | ClientDisconnect):
            outcome = OUTCOME_CLIENT_GONE
        metrics.requests.labels(consumer_class, outcome).inc()
        if address is not None:
            upstreams.release(address)
        await held.release(request_id)
        raise


async def _await_slot(
    request: Request,
    queue: OcrQueue,
    upstreams: UpstreamPool,
    request_id: str,
    priority: int,
    consumer_class: str,
    config: OcrShimConfig,
) -> str:
    """Wait until this request is both next in line and has a replica to go to.

    The client is held here rather than refused. That is the whole point: a 409
    reaching a caller with no retry loses the document. A caller that hung up
    while waiting has no one to parse for, though; its worker will send the
    document again, so admitting the orphan would only run the parse twice.
    """

    while True:
        if await request.is_disconnected():
            raise ClientDisconnect
        replicas = await upstreams.refresh()
        if replicas:
            address = await queue.try_admit(request_id, replicas, upstreams.capacity_per_replica)
            if address is not None:
                upstreams.dispatched(address)
                return address
        await queue.renew(request_id, priority, consumer_class, "waiting")
        await asyncio.sleep(config.poll_interval_seconds)


async def _forward(
    request: Request,
    route: str,
    body: bytes,
    address: str,
    upstreams: UpstreamPool,
    queue: OcrQueue,
    request_id: str,
    metrics: OcrShimMetrics,
    priority: int,
    consumer_class: str,
    *,
    renew_every: float,
) -> Response:
    """Relay the held request to the chosen replica and free its slot after."""

    client: httpx.AsyncClient = request.app.state.client
    upstream = client.build_request(
        request.method,
        f"{upstreams.base_url(address)}{route}",
        params=request.query_params,
        headers=_forwarded_headers(request.headers),
        content=body,
    )
    dispatched_at = time.perf_counter()
    # A parse can run for many minutes, far past the point at which a silent
    # row is presumed abandoned. Renewing while dispatched is what tells the
    # other replicas the slot is still genuinely busy.
    keep_alive = asyncio.create_task(
        _renew_while_dispatched(
            queue, request_id, priority, consumer_class, address, every=renew_every
        )
    )
    try:
        response = await client.send(upstream, stream=True)
    except BaseException:
        await _stop_renewing(keep_alive)
        raise

    async def _settle() -> None:
        await response.aclose()
        await _stop_renewing(keep_alive)
        await queue.release(request_id)

    async def _measured() -> AsyncIterator[bytes]:
        # The parse is only over when its last byte has been relayed, and the
        # way the relay stops is the outcome: a clean end is judged by the
        # status the pool answered with, a transport failure is the pool's,
        # and a closed-early generator means the caller stopped listening.
        # Whichever it is, the generator's own finally is the one place that
        # runs on all three, so the slot is given back here rather than in a
        # response hook that only fires after a clean end.
        outcome = OUTCOME_CLIENT_GONE
        try:
            async for chunk in response.aiter_raw():
                yield chunk
            outcome = OUTCOME_SUCCEEDED if response.is_success else OUTCOME_UPSTREAM_ERROR
        except httpx.HTTPError as exc:
            outcome = _failure_outcome(exc)
            raise
        finally:
            metrics.parse_duration.labels(address).observe(time.perf_counter() - dispatched_at)
            metrics.requests.labels(consumer_class, outcome).inc()
            upstreams.release(address)
            # A caller that hung up cancels this generator, and the row has to
            # go regardless, so its removal is kept out of reach of that cancel.
            await asyncio.shield(_settle())

    return StreamingResponse(
        _measured(), status_code=response.status_code, headers=dict(response.headers)
    )


async def _renew_while_dispatched(
    queue: OcrQueue,
    request_id: str,
    priority: int,
    consumer_class: str,
    address: str,
    *,
    every: float,
) -> None:
    """Keep a dispatched row fresh for as long as the parse it stands for runs.

    A renewal that fails is logged and tried again on the next beat rather than
    ending the heartbeat: the parse is still running whether or not the queue
    database answered, and a row that goes quiet is swept and the slot handed
    out again.
    """

    while True:
        await asyncio.sleep(every)
        try:
            await queue.renew(request_id, priority, consumer_class, "dispatched", address)
        except Exception:
            logger.warning("could not renew dispatched request %s", request_id, exc_info=True)


async def _stop_renewing(keep_alive: asyncio.Task[None]) -> None:
    """End the heartbeat and wait for it, so no renewal lands after the release.

    A renewal restores a swept row; one still in flight when the row is deleted
    would put it back with nobody left to renew it, holding a slot until the
    sweep takes it again.
    """

    keep_alive.cancel()
    with suppress(asyncio.CancelledError):
        await keep_alive


def _failure_outcome(exc: BaseException) -> str:
    """Name what went wrong with a dispatched request, within the fixed vocabulary."""

    if isinstance(exc, httpx.TimeoutException):
        return OUTCOME_UPSTREAM_TIMEOUT
    return OUTCOME_UPSTREAM_ERROR


def _owner_id() -> str:
    """Identify this replica so its rows can be reclaimed if the process dies."""

    return os.environ.get("POD_NAME") or socket.gethostname()


def _settle_backend(content_type: str, body: bytes) -> bytes | JSONResponse:
    """Return the body with a backend the pool can run, or the refusal to send.

    A body that is not multipart, or that already names a supported backend, is
    returned as it was.
    """

    boundary = _multipart_boundary(content_type)
    if boundary is None:
        return body
    declared = _declared_backend(body, boundary)
    if declared is None:
        return _with_default_backend(body, boundary)
    if declared in SUPPORTED_PARSE_BACKENDS:
        return body
    supported = ", ".join(sorted(SUPPORTED_PARSE_BACKENDS))
    return JSONResponse(
        {
            "error": f"unsupported backend '{declared}'",
            "supported_backends": sorted(SUPPORTED_PARSE_BACKENDS),
            "detail": f"this parsing pool is built pipeline-only; supported backends: {supported}",
        },
        status_code=400,
    )


def _multipart_boundary(content_type: str) -> bytes | None:
    """Return the boundary of a multipart body, or ``None`` if it is not one."""

    kind, _, parameters = content_type.partition(";")
    if kind.strip().lower() != "multipart/form-data":
        return None
    for parameter in parameters.split(";"):
        name, _, value = parameter.strip().partition("=")
        if name.strip().lower() == "boundary":
            return value.strip().strip('"').encode()
    return None


def _declared_backend(body: bytes, boundary: bytes) -> str | None:
    """Return the ``backend`` field a multipart body declares, if it declares one.

    The name is read from each part's own Content-Disposition rather than
    searched for across the whole body, so a file that happens to contain the
    word cannot be mistaken for the field.
    """

    for part in body.split(b"--" + boundary):
        head, separator, value = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        disposition = head.lower()
        if b'name="backend"' not in disposition:
            continue
        return value.rstrip(b"\r\n").decode(errors="replace").strip()
    return None


def _with_default_backend(body: bytes, boundary: bytes) -> bytes:
    """Append the pool's only supported backend to a body that omitted it."""

    closing = b"--" + boundary + b"--"
    index = body.rfind(closing)
    if index == -1:
        return body
    field = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="backend"\r\n\r\n'
        + DEFAULT_PARSE_BACKEND.encode()
        + b"\r\n"
    )
    return body[:index] + field + body[index:]


def _forwarded_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Drop the caller's credential and anything scoped to this connection.

    The parsing pool sits behind the shim and has no authentication of its own,
    so relaying the key would only copy it into another process's logs.
    """

    dropped = {"host", "content-length", "authorization", "connection", "transfer-encoding"}
    return {name: value for name, value in headers.items() if name.lower() not in dropped}
