# SPDX-License-Identifier: Apache-2.0
"""The enforcement floor in front of an inference engine.

The engine binds ``127.0.0.1`` and this process owns the only exposed port, so
there is no route to inference that skips authentication or priority stamping.
Everything here speaks the OpenAI wire format and nothing else, which is what
lets the engine behind it be replaced without touching consumers.

Two properties are deliberate and easy to lose:

* Client-supplied priority is **removed** from every JSON body, on every path,
  before anything else happens. Stamping only the generation paths would leave
  the others usable as a forgery channel.
* The server's value is stamped **unconditionally**. vLLM treats a missing
  ``priority`` as ``0``, which is its *highest* band, so a request that slips
  through unstamped is promoted rather than dropped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from kg_processor import __version__
from kg_processor.serving.priority import ConsumerKeyring, load_keyring, presented_key
from kg_processor.serving.sizing import (
    BYTES_PER_GIB,
    DeviceBudget,
    ModelGeometry,
    SizingVerdict,
    compute_sizing,
)

logger = logging.getLogger(__name__)

# Read-only and carrying no inference. The kubelet probes the health routes and
# the endpoint picker scores replicas on the metrics one, and both reach the pod
# before any consumer holds a key.
#
# The render routes are here for the same reason and with the same limit. The
# picker has to turn a prompt into the engine's own token ids to know which
# replica already holds that prefix - an estimate cannot match, because the
# comparison is against real token ids in the cache events. Rendering allocates
# no KV blocks, runs no forward pass and returns no completion: it is the
# tokenizer, reachable only from whatever the network policy admits to this
# port, which is the picker. Guarding it with a key would mean stamping a
# priority band onto a call that never reaches the scheduler.
UNAUTHENTICATED_PATHS = frozenset(
    {
        "/health",
        "/ping",
        "/metrics",
        "/v1/chat/completions/render",
        "/v1/completions/render",
    }
)

# Adapter management mutates what the pod serves. The platform pins one model per
# pod, so these are refused here rather than guarded by a key nobody should hold.
REFUSED_PATHS = frozenset({"/load_lora_adapter", "/unload_lora_adapter"})

# vLLM reads ``priority`` from the request body on its generation routes. Other
# routes have no use for it, and adding an unexpected field to their payloads
# risks a validation error from the engine.
STAMPED_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/invocations",
    }
)

# Every enabled generation route participates in admission. Anthropic messages
# uses vLLM's trusted priority header rather than a JSON priority field.
GENERATION_PATHS = STAMPED_PATHS | {"/v1/messages"}

# The balancer's own route. vLLM serves nothing here, so the sidecar can own
# it without shadowing an engine path.
ADMISSION_ROUTE = "/flakegraph/admission"

PRIORITY_FIELD = "priority"
PRIORITY_HEADERS = frozenset({"x-vllm-priority", "x-flakegraph-priority"})

# Metric labels have to stay a small fixed vocabulary: a label per key, request
# or free-form path would grow a series per value and eventually take the
# Prometheus server down with it. Paths collapse onto this set, everything else
# is ``other``, and a request that presented no usable key is one class.
ROUTE_LABELS: Mapping[str, str] = {
    "/v1/chat/completions": "chat_completions",
    "/v1/completions": "completions",
    "/v1/embeddings": "embeddings",
    "/v1/messages": "messages",
    "/v1/responses": "responses",
    "/v1/models": "models",
}
OTHER_ROUTE_LABEL = "other"
UNAUTHENTICATED_CLASS = "unauthenticated"
# A generation call can legitimately run for minutes, which the default buckets
# (topping out at ten seconds) would flatten into one bin.
REQUEST_DURATION_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300, 600)
# A scrape has its own deadline on the Prometheus side, and an engine that has
# stopped answering must not hold the sidecar's own series hostage to it.
SCRAPE_TIMEOUT_SECONDS = 5.0

# Headers that describe one connection and must not be relayed onto another.
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)


class SidecarConfig(BaseModel):
    """Configure the sidecar entirely from the environment the pod provides."""

    upstream_base_url: str = "http://127.0.0.1:8001"
    listen_host: str = "0.0.0.0"
    listen_port: int = Field(default=8000, gt=0, lt=65536)
    keys_file: Path = Path("/etc/flakegraph/serving-keys.json")
    bands: dict[str, int] | None = None
    request_timeout_seconds: float = Field(default=3600.0, gt=0)
    geometry: ModelGeometry | None = None
    budget: DeviceBudget | None = None
    expected_context_tokens: int = Field(default=32768, gt=0)
    max_num_seqs: int = Field(default=32, gt=0)
    # Interactive-first admission. Generation requests of these classes are
    # held in the sidecar while any request of another class is in flight on
    # this engine, so a person's reply is not scheduled beside seconds of
    # batch prefill. Empty disables the gate.
    held_classes: list[str] = Field(default_factory=lambda: ["batch"])
    # How long one held request waits before the gate lets it through anyway,
    # and how far apart such releases are once the wait has been exhausted:
    # a long agentic session must not starve batch on its engine forever, and
    # a trickle keeps the engine's batch share small while it lasts.
    hold_seconds: float = Field(default=90.0, ge=0)
    hold_trickle_seconds: float = Field(default=5.0, ge=0)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SidecarConfig:
        """Build a configuration from ``FLAKEGRAPH_SIDECAR_*`` variables.

        Sizing inputs are optional as a group: when the model geometry and device
        budget are both present the startup check runs, and when neither is
        present the sidecar serves without it. A partial group is an error, since
        silently skipping the check is the failure this design exists to prevent.
        """

        env = os.environ if environ is None else environ
        geometry = _optional_model(ModelGeometry, env, "FLAKEGRAPH_SIDECAR_GEOMETRY")
        budget = _optional_model(DeviceBudget, env, "FLAKEGRAPH_SIDECAR_DEVICE_BUDGET")
        if (geometry is None) != (budget is None):
            raise ValueError(
                "FLAKEGRAPH_SIDECAR_GEOMETRY and FLAKEGRAPH_SIDECAR_DEVICE_BUDGET "
                "must be set together or not at all"
            )
        values: dict[str, Any] = {"geometry": geometry, "budget": budget}
        for field, variable in (
            ("upstream_base_url", "FLAKEGRAPH_SIDECAR_UPSTREAM"),
            ("listen_host", "FLAKEGRAPH_SIDECAR_LISTEN_HOST"),
            ("listen_port", "FLAKEGRAPH_SIDECAR_LISTEN_PORT"),
            ("keys_file", "FLAKEGRAPH_SIDECAR_KEYS_FILE"),
            ("request_timeout_seconds", "FLAKEGRAPH_SIDECAR_REQUEST_TIMEOUT_SECONDS"),
            ("expected_context_tokens", "FLAKEGRAPH_SIDECAR_EXPECTED_CONTEXT_TOKENS"),
            ("max_num_seqs", "FLAKEGRAPH_SIDECAR_MAX_NUM_SEQS"),
            ("hold_seconds", "FLAKEGRAPH_SIDECAR_HOLD_SECONDS"),
            ("hold_trickle_seconds", "FLAKEGRAPH_SIDECAR_HOLD_TRICKLE_SECONDS"),
        ):
            if variable in env:
                values[field] = env[variable]
        if "FLAKEGRAPH_SIDECAR_BANDS" in env:
            values["bands"] = json.loads(env["FLAKEGRAPH_SIDECAR_BANDS"])
        if "FLAKEGRAPH_SIDECAR_HELD_CLASSES" in env:
            values["held_classes"] = [
                item.strip()
                for item in env["FLAKEGRAPH_SIDECAR_HELD_CLASSES"].split(",")
                if item.strip()
            ]
        return cls.model_validate(values)

    def sizing_verdict(self) -> SizingVerdict | None:
        """Evaluate the configured sequence limit, or ``None`` when not declared."""

        if self.geometry is None or self.budget is None:
            return None
        return compute_sizing(
            self.geometry,
            self.budget,
            self.expected_context_tokens,
            self.max_num_seqs,
        )


class SidecarMetrics:
    """Own the series this process adds on top of the engine's.

    Each application gets its own registry rather than the process-wide default.
    The default one also carries the Python runtime collectors, and the engine
    already exports those under the same names - emitting them twice in one
    scrape body would make Prometheus reject the whole scrape.
    """

    def __init__(self) -> None:
        """Register every series up front so an idle sidecar still exposes them."""

        self.registry = CollectorRegistry()
        self.requests = Counter(
            "flakegraph_sidecar_requests_total",
            "Requests answered by the sidecar, by the status the caller received.",
            ["consumer_class", "route", "status"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "flakegraph_sidecar_request_duration_seconds",
            "Time from receiving a request to sending its last byte.",
            ["consumer_class", "route"],
            buckets=REQUEST_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "flakegraph_sidecar_requests_in_flight",
            "Requests received whose last byte has not yet been sent.",
            ["consumer_class"],
            registry=self.registry,
        )
        self.batch_cap = Gauge(
            "flakegraph_sidecar_batch_cap",
            "Concurrent held-class requests this engine currently admits; -1 when uncapped.",
            registry=self.registry,
        )
        self.held = Gauge(
            "flakegraph_sidecar_requests_held",
            "Generation requests waiting in the sidecar for interactive traffic to finish.",
            ["consumer_class"],
            registry=self.registry,
        )
        self.hold_duration = Histogram(
            "flakegraph_sidecar_hold_seconds",
            "How long a held generation request waited before the engine saw it.",
            ["consumer_class", "released_by"],
            buckets=REQUEST_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.batch_cap.set(-1)
        self.preempted = Counter(
            "flakegraph_sidecar_preempted_total",
            "Held-class requests cancelled on the engine because a person's request arrived.",
            ["consumer_class"],
            registry=self.registry,
        )
        self.priority_stamped = Counter(
            "flakegraph_sidecar_priority_stamped_total",
            "Generation requests stamped with each priority band.",
            ["consumer_class", "priority"],
            registry=self.registry,
        )
        Info(
            "flakegraph_sidecar_build",
            "Version of the FlakeGraph package serving this sidecar.",
            registry=self.registry,
        ).info({"version": __version__})

    def render(self) -> bytes:
        """Serialise the sidecar's own series in the text exposition format."""

        return generate_latest(self.registry)


class AdmissionGate:
    """Keep batch prefill off an engine while a person is being answered on it.

    vLLM's priority policy decides who enters the scheduler first, and that
    part works: a chat request waits a fraction of a second in the queue. It
    does not decide what shares each scheduler step with that request once
    it is running, and a step that also carries several thousand tokens of
    batch prefill takes seconds, during which the person's stream receives
    nothing. On a compute-bound part that is the whole difference between a
    reply at forty tokens a second and one at eight.

    The gate sits in front of the engine: while any request of a class that
    is *not* held is in flight here, new generation requests of the held
    classes wait. Held requests already on the engine are cancelled the
    moment a person's request arrives - one batch sequence beside a reply
    costs it most of its speed - and their callers are told to retry, which
    routing sends to another engine. When the last interactive request
    finishes, the held ones waiting here are released together. A ceiling
    bounds the wait, and past it the gate trickles one held request through
    per interval, so a long agentic session keeps its engine mostly to itself
    without stopping batch there outright.
    """

    def __init__(
        self,
        metrics: SidecarMetrics,
        held_classes: Iterable[str],
        hold_seconds: float,
        trickle_seconds: float,
        cap_ttl_seconds: float = 300.0,
    ) -> None:
        self._metrics = metrics
        self._held_classes = frozenset(held_classes)
        self._hold_seconds = hold_seconds
        self._trickle_seconds = trickle_seconds
        self._protected_in_flight = 0
        self._held_in_flight = 0
        # Held-class requests on the engine and not yet answered, by class:
        # what a person's arrival cancels.
        self._preemptible: dict[asyncio.Task[httpx.Response], str] = {}
        self._condition: asyncio.Condition | None = None
        self._last_trickle = 0.0
        # How many held-class requests may run here at once. ``None`` is the
        # engine's ordinary state: no cap beyond what vLLM itself schedules.
        # The balancer sets 0 to reserve this engine for people, and restores
        # ``None`` to give it back. On a GB10 one batch sequence
        # beside a chat reply costs about 70% of the reply's speed, so the
        # useful settings in practice are 0 and unlimited.
        self._batch_cap: int | None = None
        self._cap_expires_at: float | None = None
        self._cap_ttl_seconds = cap_ttl_seconds

    @property
    def enabled(self) -> bool:
        """Whether any class is held at all."""

        return bool(self._held_classes)

    async def set_batch_cap(self, cap: int | None) -> None:
        """Set how many held-class requests may run here, or ``None`` for no cap.

        The setting carries a time to live. A balancer that stops calling -
        crashed, evicted, disconnected - must not leave an engine reserved
        forever, so the cap lapses back to no cap and the fleet returns to
        ordinary behaviour on its own.
        """

        condition = self._cond()
        async with condition:
            self._batch_cap = cap
            self._cap_expires_at = None if cap is None else time.monotonic() + self._cap_ttl_seconds
            self._metrics.batch_cap.set(-1 if cap is None else cap)
            condition.notify_all()

    @property
    def batch_cap(self) -> int | None:
        """The cap in force now, after any lapse."""

        if self._cap_expires_at is not None and time.monotonic() >= self._cap_expires_at:
            self._batch_cap = None
            self._cap_expires_at = None
            self._metrics.batch_cap.set(-1)
        return self._batch_cap

    def state(self) -> dict[str, object]:
        """What the balancer reads back: the cap, its lifetime, and what is running."""

        cap = self.batch_cap
        return {
            "batch_cap": cap,
            "cap_expires_in_seconds": (
                None
                if self._cap_expires_at is None
                else max(0.0, round(self._cap_expires_at - time.monotonic(), 1))
            ),
            "held_in_flight": self._held_in_flight,
            "protected_in_flight": self._protected_in_flight,
            "held_classes": sorted(self._held_classes),
        }

    def holds(self, consumer_class: str, route: str) -> bool:
        """Whether a request of this class on this route is subject to the gate."""

        return self.enabled and route in GENERATION_PATHS and consumer_class in self._held_classes

    def protects(self, consumer_class: str, route: str) -> bool:
        """Whether a request of this class on this route holds the gate shut."""

        return (
            self.enabled and route in GENERATION_PATHS and consumer_class not in self._held_classes
        )

    def _cond(self) -> asyncio.Condition:
        if self._condition is None:
            self._condition = asyncio.Condition()
        return self._condition

    async def enter_protected(self) -> None:
        """Count one more request whose presence shuts the gate, and clear the engine.

        Every held request still waiting for its answer is cancelled, so the
        person's request runs with nothing beside it. Closing the connection is
        what makes the engine abort the sequence and free its KV blocks.
        """

        async with self._cond():
            self._protected_in_flight += 1
            preempted, self._preemptible = self._preemptible, {}
        for task, consumer_class in preempted.items():
            if task.cancel():
                self._metrics.preempted.labels(consumer_class).inc()

    def track_preemptible(self, task: asyncio.Task[httpx.Response], consumer_class: str) -> None:
        """Register a held request's call to the engine as one a person may cancel."""

        self._preemptible[task] = consumer_class

    def untrack_preemptible(self, task: asyncio.Task[httpx.Response]) -> None:
        """Forget a call that has its answer, or was cancelled."""

        self._preemptible.pop(task, None)

    async def leave_protected(self) -> None:
        """Count one such request out, opening the gate when it was the last."""

        condition = self._cond()
        async with condition:
            self._protected_in_flight = max(0, self._protected_in_flight - 1)
            if self._protected_in_flight == 0:
                condition.notify_all()

    async def admit(self, consumer_class: str) -> None:
        """Return when a held request may go to the engine, recording how long it waited."""

        condition = self._cond()
        started = time.monotonic()
        deadline = started + self._hold_seconds
        released_by = "clear"
        self._metrics.held.labels(consumer_class).inc()
        try:
            async with condition:
                while self._blocked():
                    now = time.monotonic()
                    if now >= deadline:
                        # The wait is spent. Let one through per interval so the
                        # engine stays mostly the interactive caller's while the
                        # session lasts, and batch there still moves.
                        if now - self._last_trickle >= self._trickle_seconds:
                            self._last_trickle = now
                            released_by = "trickle"
                            break
                        wait = self._trickle_seconds - (now - self._last_trickle)
                    else:
                        wait = deadline - now
                    if self._cap_expires_at is not None:
                        wait = min(wait, max(0.0, self._cap_expires_at - now))
                    try:
                        await asyncio.wait_for(condition.wait(), timeout=max(wait, 0.01))
                    except TimeoutError:
                        continue
            async with condition:
                self._held_in_flight += 1
        finally:
            self._metrics.held.labels(consumer_class).dec()
            self._metrics.hold_duration.labels(consumer_class, released_by).observe(
                time.monotonic() - started
            )

    def _blocked(self) -> bool:
        """Whether a held-class request must wait: people here, or the cap is full."""

        if self._protected_in_flight > 0:
            return True
        cap = self.batch_cap
        return cap is not None and self._held_in_flight >= cap

    async def leave_held(self) -> None:
        """Count one held-class request out, so a waiting one can take its place."""

        condition = self._cond()
        async with condition:
            self._held_in_flight = max(0, self._held_in_flight - 1)
            condition.notify_all()


class _Accounting:
    """Close one request's metrics exactly once, when its last byte has left.

    On a streamed completion the headers leave long before the body, so the
    observation is attached to the stream rather than to the handler returning.
    """

    def __init__(
        self,
        metrics: SidecarMetrics,
        consumer_class: str,
        route: str,
        gate: AdmissionGate | None = None,
        holds_slot: bool = False,
    ) -> None:
        """Mark the request as in flight from the moment it was received."""

        self._metrics = metrics
        self.consumer_class = consumer_class
        self._route = ROUTE_LABELS.get(route, OTHER_ROUTE_LABEL)
        self._started = time.perf_counter()
        self._closed = False
        # A protected request holds the gate from its first byte in to its
        # last byte out; the release runs in the same task the stream ends in.
        self._gate = gate
        # A held-class request occupies one of the capped slots until it is
        # done; a protecting one keeps the gate shut for the same span.
        self._holds_slot = holds_slot
        metrics.in_flight.labels(consumer_class).inc()

    @property
    def preemptible(self) -> bool:
        """Whether a person's request arriving on this engine cancels this one."""

        return self._gate is not None and self._holds_slot

    def close(self, response: Response) -> Response:
        """Record a response that is complete the moment it is returned."""

        self._record(response.status_code)
        return response

    def abandon(self) -> None:
        """Record a request the handler gave up on before any response was built.

        Starlette answers an escaped exception with a 500, so that is the status
        the caller saw. Recording it here is what keeps the in-flight gauge from
        drifting upward by one for every engine timeout.
        """

        self._record(500)

    async def stream(self, chunks: AsyncIterator[bytes], status_code: int) -> AsyncIterator[bytes]:
        """Relay a body and record the request once the relay stops, however it stops."""

        try:
            async for chunk in chunks:
                yield chunk
        finally:
            self._record(status_code)
            await self.release()

    async def close_async(self, response: Response) -> Response:
        """``close`` for a caller that can also release the gate."""

        self._record(response.status_code)
        await self.release()
        return response

    async def release(self) -> None:
        """Let go of the gate, once, whichever side of it this request was on."""

        if self._gate is None:
            return
        gate, self._gate = self._gate, None
        if self._holds_slot:
            await gate.leave_held()
        else:
            await gate.leave_protected()

    def _record(self, status_code: int) -> None:
        if self._closed:
            return
        self._closed = True
        self._metrics.in_flight.labels(self.consumer_class).dec()
        self._metrics.requests.labels(self.consumer_class, self._route, str(status_code)).inc()
        self._metrics.duration.labels(self.consumer_class, self._route).observe(
            time.perf_counter() - self._started
        )


def create_app(
    config: SidecarConfig,
    keyring: ConsumerKeyring | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the proxy application for one engine pod.

    The keyring is loaded once here rather than per request: rotating a key is a
    pod restart, which is already how a projected Secret is rolled, and it keeps
    an unreadable file from being discovered halfway through a shift.
    """

    resolved = keyring if keyring is not None else load_keyring(config.keys_file, config.bands)
    verdict = config.sizing_verdict()
    if verdict is not None and not verdict.sequence_limit_binds_first:
        raise ValueError(f"unsafe model-serving configuration: {verdict.detail}")
    if verdict is not None:
        logger.info(
            "kv budget %.1f GiB, %s",
            verdict.kv_budget_bytes / BYTES_PER_GIB,
            verdict.detail,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Hold one upstream client for the process lifetime."""

        async with httpx.AsyncClient(
            base_url=config.upstream_base_url,
            timeout=httpx.Timeout(config.request_timeout_seconds),
            transport=transport,
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(
        title="FlakeGraph inference sidecar",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.keyring = resolved
    metrics = SidecarMetrics()
    app.state.metrics = metrics
    gate = AdmissionGate(
        metrics, config.held_classes, config.hold_seconds, config.hold_trickle_seconds
    )
    app.state.gate = gate

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def proxy(path: str, request: Request) -> Response:
        """Authenticate, stamp priority, and relay the request to the engine."""

        route = f"/{path}"
        early = await _answer_without_inference(request, route, metrics, resolved, gate)
        if early is not None:
            return early
        body = await request.body()
        # Probe and scoring traffic is not measured. The kubelet and the picker
        # call these paths many times a second, which would bury the consumer
        # series in a volume of requests that never reach the scheduler.
        if route in UNAUTHENTICATED_PATHS:
            return await _relay(request, route, body, None)

        consumer_class = resolved.classify(presented_key(request.headers.get("authorization", "")))
        if consumer_class is None:
            rejected = _Accounting(metrics, UNAUTHENTICATED_CLASS, route)
            return rejected.close(JSONResponse({"error": "unauthorized"}, status_code=401))

        # Stamp before admission so invalid bytes cannot retain a protected
        # request or capped batch slot without a relay to release it.
        stamped, priority = _stamped_body(body, route, resolved, consumer_class)
        accounting = await _admit(gate, metrics, consumer_class, route)
        if priority is not None:
            metrics.priority_stamped.labels(consumer_class, str(priority)).inc()
        return await _relay(request, route, stamped, accounting)

    async def _relay(
        request: Request,
        route: str,
        body: bytes,
        accounting: _Accounting | None,
    ) -> Response:
        """Stream the upstream response back without buffering the whole body."""

        client: httpx.AsyncClient = request.app.state.client
        upstream = client.build_request(
            request.method,
            route,
            params=request.query_params,
            headers=_forwarded_headers(
                request.headers,
                priority=(
                    resolved.priority_for(accounting.consumer_class)
                    if route == "/v1/messages" and accounting is not None
                    else None
                ),
            ),
            content=body,
        )
        try:
            response = await _send(client, upstream, request.app.state.gate, accounting)
        except _Preempted as preempted:
            return await preempted.accounting.close_async(_preempted_response())
        except httpx.ConnectError:
            # Normal for the whole of a cold start: the engine spends minutes
            # loading weights before it binds its port, and the kubelet is
            # probing this path throughout. Report it as a bad gateway so the
            # probe reads "not ready" instead of the process logging a stack
            # trace per second until the engine appears.
            unavailable = JSONResponse({"error": "engine unavailable"}, status_code=502)
            return (
                await accounting.close_async(unavailable) if accounting is not None else unavailable
            )
        except BaseException:
            if accounting is not None:
                accounting.abandon()
                await accounting.release()
            raise
        chunks = response.aiter_raw()
        if accounting is not None:
            chunks = accounting.stream(chunks, response.status_code)
        return StreamingResponse(
            chunks,
            status_code=response.status_code,
            headers=_relayed_headers(response.headers),
            background=BackgroundTask(response.aclose),
        )

    return app


class _Preempted(Exception):
    """A held request cancelled on the engine because a person's request arrived."""

    def __init__(self, accounting: _Accounting) -> None:
        super().__init__("preempted by interactive work")
        self.accounting = accounting


def _preempted_response() -> JSONResponse:
    """Tell a cancelled held request to retry at once.

    The engine is a person's now, and routing sends the retry to one that is not.
    """

    return JSONResponse(
        {"error": "preempted by interactive work"},
        status_code=503,
        headers={"Retry-After": "0"},
    )


async def _send(
    client: httpx.AsyncClient,
    upstream: httpx.Request,
    gate: AdmissionGate,
    accounting: _Accounting | None,
) -> httpx.Response:
    """Send to the engine; a held request stays cancellable until its answer starts.

    A cancellation of the sending task by the gate reads as ``_Preempted``; one
    of this handler's own task - the caller hung up - propagates as usual.
    """

    if accounting is None or not accounting.preemptible:
        return await client.send(upstream, stream=True)
    sending = asyncio.ensure_future(client.send(upstream, stream=True))
    gate.track_preemptible(sending, accounting.consumer_class)
    try:
        return await sending
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise
        raise _Preempted(accounting) from None
    finally:
        gate.untrack_preemptible(sending)


async def _scrape(client: httpx.AsyncClient, metrics: SidecarMetrics) -> Response:
    """Answer one scrape with the engine's series followed by the sidecar's.

    The engine's are the ones worth having, so they are relayed rather than
    replaced - but the sidecar's are appended even when the engine cannot be
    reached, because "the sidecar is up" must not read as "down" for the whole
    of a cold start. The scraper's own headers are deliberately not forwarded:
    an ``Accept`` asking for OpenMetrics would make the engine terminate its
    body with ``# EOF``, after which nothing may follow.
    """

    own = metrics.render()
    try:
        upstream = await client.get("/metrics", timeout=SCRAPE_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        return Response(own, media_type=CONTENT_TYPE_LATEST)
    if not upstream.is_success:
        return Response(own, media_type=CONTENT_TYPE_LATEST)
    engine = upstream.content
    if engine and not engine.endswith(b"\n"):
        engine += b"\n"
    return Response(engine + own, media_type=CONTENT_TYPE_LATEST)


def run(config: SidecarConfig | None = None) -> None:
    """Serve the sidecar until the process is signalled."""

    resolved = config if config is not None else SidecarConfig.from_env()
    uvicorn.run(
        create_app(resolved),
        host=resolved.listen_host,
        port=resolved.listen_port,
        log_level="info",
    )


async def _answer_without_inference(
    request: Request,
    route: str,
    metrics: SidecarMetrics,
    keyring: ConsumerKeyring,
    gate: AdmissionGate,
) -> Response | None:
    """Answer the routes the engine must never see, or ``None`` to relay.

    Adapter management is refused outright, the scrape is merged here, and the
    balancer's admission route is the sidecar's own. Everything else is
    inference and goes through authentication and the gate.
    """

    if route in REFUSED_PATHS:
        refused = _Accounting(metrics, UNAUTHENTICATED_CLASS, route)
        return refused.close(
            JSONResponse({"error": "adapter management is disabled"}, status_code=404)
        )
    if route == "/metrics":
        return await _scrape(request.app.state.client, metrics)
    if route == ADMISSION_ROUTE:
        return await _admission(request, keyring, gate)
    return None


async def _admission(
    request: Request,
    keyring: ConsumerKeyring,
    gate: AdmissionGate,
) -> Response:
    """Report or set this engine's batch cap, for the fleet balancer.

    Guarded by the same keyring as inference and restricted to the classes the
    gate protects: a batch key must not be able to lift the cap that is
    holding batch back. GET reports, POST sets ``{"batch_cap": 0}`` to reserve
    the engine or ``{"batch_cap": null}`` to release it.
    """

    consumer_class = keyring.classify(presented_key(request.headers.get("authorization", "")))
    if consumer_class is None or not gate.protects(consumer_class, next(iter(STAMPED_PATHS))):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if request.method == "GET":
        return JSONResponse(gate.state())
    try:
        payload = json.loads(await request.body() or b"{}")
        raw = payload.get("batch_cap", None) if isinstance(payload, dict) else None
        cap = None if raw is None else int(raw)
    except (ValueError, TypeError):
        return JSONResponse({"error": "batch_cap must be an integer or null"}, status_code=400)
    if cap is not None and cap < 0:
        return JSONResponse({"error": "batch_cap must not be negative"}, status_code=400)
    await gate.set_batch_cap(cap)
    return JSONResponse(gate.state())


async def _admit(
    gate: AdmissionGate,
    metrics: SidecarMetrics,
    consumer_class: str,
    route: str,
) -> _Accounting:
    """Pass the request through the gate and start its accounting.

    A held class waits here until the engine is free of interactive work; a
    protecting class shuts the gate for as long as its request lasts; every
    other request is neither.
    """

    if gate.holds(consumer_class, route):
        await gate.admit(consumer_class)
        return _Accounting(metrics, consumer_class, route, gate, holds_slot=True)
    if gate.protects(consumer_class, route):
        await gate.enter_protected()
        return _Accounting(metrics, consumer_class, route, gate)
    return _Accounting(metrics, consumer_class, route)


def _stamped_body(
    body: bytes,
    route: str,
    keyring: ConsumerKeyring,
    consumer_class: str,
) -> tuple[bytes, int | None]:
    """Remove any client priority and stamp the server's band for this class.

    Returns the body to relay and the band stamped into it, or ``None`` when
    none was. A body that is not a JSON object is relayed untouched. It cannot
    carry a priority field, so there is nothing to strip and nothing to forge.
    Invalid JSON encoding is rejected before the caller acquires admission.
    """

    if not body:
        return body, None
    try:
        payload = json.loads(body)
    except UnicodeDecodeError as error:
        raise HTTPException(status_code=400, detail="invalid JSON encoding") from error
    except json.JSONDecodeError:
        return body, None
    if not isinstance(payload, dict):
        return body, None
    payload.pop(PRIORITY_FIELD, None)
    priority: int | None = None
    if route in GENERATION_PATHS:
        priority = keyring.priority_for(consumer_class)
        if route in STAMPED_PATHS:
            payload[PRIORITY_FIELD] = priority
    return json.dumps(payload).encode("utf-8"), priority


def _forwarded_headers(
    headers: Mapping[str, str], *, priority: int | None = None
) -> dict[str, str]:
    """Drop connection-scoped headers, the caller's credential, and priority hints.

    The engine has no authentication of its own, so relaying the client's key
    would only write it into another process's logs. Length is dropped because
    stamping changes it, and httpx recomputes it from the body it is given.
    A trusted server priority is added only for routes using vLLM's header.
    """

    forwarded: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered in HOP_BY_HOP_HEADERS or lowered in PRIORITY_HEADERS:
            continue
        if lowered in {"host", "content-length", "authorization"}:
            continue
        forwarded[name] = value
    if priority is not None:
        forwarded["X-Vllm-Priority"] = str(priority)
    return forwarded


def _relayed_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return upstream response headers minus the connection-scoped ones.

    The raw byte stream is relayed unchanged, so any content encoding and the
    length that describes it both remain accurate.
    """

    return {
        name: value for name, value in headers.items() if name.lower() not in HOP_BY_HOP_HEADERS
    }


def _optional_model[ModelT: BaseModel](
    model: type[ModelT],
    env: Mapping[str, str],
    variable: str,
) -> ModelT | None:
    """Parse a JSON-valued environment variable into a model when it is present."""

    raw = env.get(variable)
    if raw is None or not raw.strip():
        return None
    return model.model_validate_json(raw)
