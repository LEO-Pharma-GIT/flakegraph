from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from serving import KEYRING, RecordingUpstream, samples, stub

import kg_processor
from kg_processor.serving.priority import ConsumerKeyring, load_keyring
from kg_processor.serving.sidecar import STAMPED_PATHS, SidecarConfig, create_app
from kg_processor.serving.sizing import DeviceBudget, ModelGeometry


def _client(upstream: RecordingUpstream, config: SidecarConfig | None = None) -> TestClient:
    resolved = config or SidecarConfig(upstream_base_url="http://engine.invalid")
    app = create_app(resolved, keyring=KEYRING, transport=httpx.MockTransport(upstream.handler))
    return TestClient(app)


def test_batch_key_cannot_forge_the_interactive_band() -> None:
    upstream = RecordingUpstream()
    with _client(upstream) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-pipeline"},
            json={"model": "m", "messages": [], "priority": 0},
        )

    assert response.status_code == 200
    assert upstream.last_body["priority"] == 100


def test_each_class_is_stamped_with_its_own_band() -> None:
    upstream = RecordingUpstream()
    with _client(upstream) as client:
        for key, expected in (("sk-chat", 0), ("sk-tool", 10), ("sk-pipeline", 100)):
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": "m", "messages": []},
            )
            assert upstream.last_body["priority"] == expected


def test_priority_headers_never_reach_the_engine() -> None:
    upstream = RecordingUpstream()
    with _client(upstream) as client:
        client.post(
            "/v1/completions",
            headers={"Authorization": "Bearer sk-pipeline", "X-Vllm-Priority": "0"},
            json={"model": "m", "prompt": "hi"},
        )

    forwarded = upstream.requests[-1].headers
    assert "x-vllm-priority" not in forwarded
    assert "authorization" not in forwarded


def test_priority_is_stripped_even_on_paths_that_are_not_stamped() -> None:
    upstream = RecordingUpstream()
    with _client(upstream) as client:
        client.post(
            "/tokenize",
            headers={"Authorization": "Bearer sk-pipeline"},
            json={"model": "m", "prompt": "hi", "priority": 0},
        )

    assert "priority" not in upstream.last_body


def test_an_unknown_key_is_rejected_before_the_engine_is_touched() -> None:
    upstream = RecordingUpstream()
    with _client(upstream) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-not-issued"},
            json={"model": "m", "messages": []},
        )

    assert response.status_code == 401
    assert upstream.requests == []


def test_a_missing_credential_is_rejected() -> None:
    upstream = RecordingUpstream()
    with _client(upstream) as client:
        assert client.post("/v1/chat/completions", json={}).status_code == 401
        assert client.get("/v1/models").status_code == 401


def test_an_unknown_class_receives_the_band_served_last() -> None:
    upstream = RecordingUpstream()
    keyring = ConsumerKeyring(
        bands={"interactive": 0, "batch": 100},
        keys={"sk-mystery": "reporting"},
    )
    app = create_app(
        SidecarConfig(upstream_base_url="http://engine.invalid"),
        keyring=keyring,
        transport=httpx.MockTransport(upstream.handler),
    )
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-mystery"},
            json={"model": "m", "messages": []},
        )

    assert upstream.last_body["priority"] == 100


def test_probe_and_scoring_paths_stay_reachable_without_a_key() -> None:
    upstream = RecordingUpstream()
    with _client(upstream) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200
        # The picker renders prompts to the engine's own token ids so it can tell
        # which replica holds a prefix. No key, because nothing is generated and
        # a stamped band would be meaningless on a call the scheduler never sees.
        assert client.post("/v1/chat/completions/render", json={}).status_code != 401
        assert client.post("/v1/completions/render", json={}).status_code != 401


def test_adapter_management_is_refused_outright() -> None:
    upstream = RecordingUpstream()
    with _client(upstream) as client:
        response = client.post(
            "/load_lora_adapter",
            headers={"Authorization": "Bearer sk-chat"},
            json={"lora_name": "x", "lora_path": "/tmp/x"},
        )

    assert response.status_code == 404
    assert upstream.requests == []


def test_streamed_completions_pass_through_chunk_by_chunk() -> None:
    chunks = [b'data: {"a":1}\n\n', b'data: {"a":2}\n\n', b"data: [DONE]\n\n"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(b"".join(chunks)),
        )

    app = create_app(
        SidecarConfig(upstream_base_url="http://engine.invalid"),
        keyring=KEYRING,
        transport=httpx.MockTransport(handler),
    )
    with (
        TestClient(app) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-chat"},
            json={"model": "m", "messages": [], "stream": True},
        ) as response,
    ):
        assert response.headers["content-type"] == "text/event-stream"
        assert b"".join(response.iter_bytes()) == b"".join(chunks)


def test_a_cold_engine_reads_as_not_ready_rather_than_crashing() -> None:
    """A probe during weight loading must get a clean 502, not a stack trace."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    app = create_app(
        SidecarConfig(upstream_base_url="http://engine.invalid"),
        keyring=KEYRING,
        transport=httpx.MockTransport(refuse),
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 502
        assert (
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-chat"},
                json={"model": "m", "messages": []},
            ).status_code
            == 502
        )


def test_query_parameters_and_method_survive_the_hop() -> None:
    upstream = RecordingUpstream()
    with _client(upstream) as client:
        client.get("/v1/models", params={"limit": "5"}, headers={"Authorization": "Bearer sk-chat"})

    assert upstream.requests[-1].method == "GET"
    assert upstream.requests[-1].url.params["limit"] == "5"


def test_an_unsafe_sequence_limit_prevents_the_app_from_being_built() -> None:
    config = SidecarConfig(
        upstream_base_url="http://engine.invalid",
        geometry=ModelGeometry(
            kv_heads=4,
            head_dim=256,
            attention_layers=16,
            weights_bytes=23_416_000_000,
        ),
        budget=DeviceBudget(
            device_memory_bytes=127_990_000_000,
            gpu_memory_utilization=0.60,
            overhead_bytes=12_884_000_000,
        ),
        expected_context_tokens=32768,
        max_num_seqs=512,
    )

    with pytest.raises(ValueError, match="does not bind before KV memory"):
        create_app(config, keyring=KEYRING)


def test_the_keyring_is_read_from_the_projected_secret(tmp_path: Path) -> None:
    keys_file = tmp_path / "serving-keys.json"
    keys_file.write_text(json.dumps({"sk-a": "interactive"}), encoding="utf-8")

    keyring = load_keyring(keys_file, {"interactive": 0, "batch": 100})

    assert keyring.classify("sk-a") == "interactive"
    assert keyring.classify("sk-b") is None
    assert keyring.priority_for(None) == 100


def test_sizing_inputs_must_be_configured_as_a_pair() -> None:
    env = {
        "FLAKEGRAPH_SIDECAR_GEOMETRY": json.dumps(
            {"kv_heads": 4, "head_dim": 256, "attention_layers": 16, "weights_bytes": 1}
        )
    }

    with pytest.raises(ValueError, match="must be set together"):
        SidecarConfig.from_env(env)


def test_an_authenticated_completion_is_counted_under_its_class_and_route() -> None:
    upstream = RecordingUpstream()
    with _client(upstream) as client:
        client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-chat"},
            json={"model": "m", "messages": []},
        )
        exposition = client.get("/metrics").text

    assert samples(exposition, "flakegraph_sidecar_requests_total") == {
        (("consumer_class", "interactive"), ("route", "chat_completions"), ("status", "200")): 1
    }
    assert samples(exposition, "flakegraph_sidecar_request_duration_seconds_count") == {
        (("consumer_class", "interactive"), ("route", "chat_completions")): 1
    }
    assert samples(exposition, "flakegraph_sidecar_requests_in_flight") == {
        (("consumer_class", "interactive"),): 0
    }


def test_a_request_without_a_usable_key_is_counted_as_unauthenticated() -> None:
    upstream = RecordingUpstream()
    with _client(upstream) as client:
        client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer sk-not-issued"},
            json={"model": "m", "input": "hi"},
        )
        exposition = client.get("/metrics").text

    assert samples(exposition, "flakegraph_sidecar_requests_total") == {
        (("consumer_class", "unauthenticated"), ("route", "embeddings"), ("status", "401")): 1
    }
    # A key is never a label: the only series minted is the one shared class.
    assert "sk-not-issued" not in exposition


def test_the_stamped_band_is_counted_per_class() -> None:
    upstream = RecordingUpstream()
    with _client(upstream) as client:
        for _ in range(2):
            client.post(
                "/v1/completions",
                headers={"Authorization": "Bearer sk-pipeline"},
                json={"model": "m", "prompt": "hi"},
            )
        client.get("/v1/models", headers={"Authorization": "Bearer sk-pipeline"})
        exposition = client.get("/metrics").text

    assert samples(exposition, "flakegraph_sidecar_priority_stamped_total") == {
        (("consumer_class", "batch"), ("priority", "100")): 2
    }


def test_a_streamed_completion_is_measured_once_when_its_stream_ends() -> None:
    chunks = [b'data: {"a":1}\n\n', b"data: [DONE]\n\n"]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return stub(b"", "text/plain")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(b"".join(chunks)),
        )

    app = create_app(
        SidecarConfig(upstream_base_url="http://engine.invalid"),
        keyring=KEYRING,
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-chat"},
            json={"model": "m", "messages": [], "stream": True},
        ) as response:
            assert b"".join(response.iter_bytes()) == b"".join(chunks)
        exposition = client.get("/metrics").text

    assert samples(exposition, "flakegraph_sidecar_request_duration_seconds_count") == {
        (("consumer_class", "interactive"), ("route", "chat_completions")): 1
    }
    assert samples(exposition, "flakegraph_sidecar_requests_in_flight") == {
        (("consumer_class", "interactive"),): 0
    }


def test_a_scrape_relays_the_engines_series_ahead_of_the_sidecars_own() -> None:
    upstream = RecordingUpstream()
    with _client(upstream) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    families = {family.name for family in text_string_to_metric_families(response.text)}
    assert "vllm:num_preemptions_total" in families
    assert "flakegraph_sidecar_requests" in families
    assert samples(response.text, "flakegraph_sidecar_build_info") == {
        (("version", kg_processor.__version__),): 1
    }
    assert response.text.index("vllm:") < response.text.index("flakegraph_sidecar_")


def test_a_scrape_still_answers_when_the_engine_cannot_be_reached() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    app = create_app(
        SidecarConfig(upstream_base_url="http://engine.invalid"),
        keyring=KEYRING,
        transport=httpx.MockTransport(refuse),
    )
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    families = {family.name for family in text_string_to_metric_families(response.text)}
    assert "flakegraph_sidecar_requests" in families
    assert not any(name.startswith("vllm") for name in families)


def test_an_engine_that_times_out_leaves_nothing_in_flight() -> None:
    def hang(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return stub(b"", "text/plain")
        raise httpx.ReadTimeout("timed out", request=request)

    app = create_app(
        SidecarConfig(upstream_base_url="http://engine.invalid"),
        keyring=KEYRING,
        transport=httpx.MockTransport(hang),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        status = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-chat"},
            json={"model": "m", "messages": []},
        ).status_code
        exposition = client.get("/metrics").text

    assert status == 500
    assert samples(exposition, "flakegraph_sidecar_requests_total") == {
        (("consumer_class", "interactive"), ("route", "chat_completions"), ("status", "500")): 1
    }
    assert samples(exposition, "flakegraph_sidecar_requests_in_flight") == {
        (("consumer_class", "interactive"),): 0
    }


class _SlowUpstream:
    """An engine whose replies end only when the test says so."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.release = asyncio.Event()

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return stub(b"vllm:num_preemptions_total 0", "text/plain")
        self.started.append(request.headers.get("x-test-name", ""))
        if request.url.path in STAMPED_PATHS:
            await self.release.wait()
        return stub(b'{"ok": true}')


def _gated_app(upstream: _SlowUpstream, **overrides: object) -> Any:
    values: dict[str, object] = {
        "upstream_base_url": "http://engine.invalid",
        "hold_seconds": 30.0,
        **overrides,
    }
    ttl = values.pop("cap_ttl_seconds", None)
    ttl_seconds = float(ttl) if isinstance(ttl, (int, float, str)) else None
    config = SidecarConfig.model_validate(values)
    app = create_app(config, keyring=KEYRING, transport=httpx.MockTransport(upstream.handler))
    if ttl_seconds is not None:
        app.state.gate._cap_ttl_seconds = ttl_seconds  # noqa: SLF001 - a clock the test owns
    return app


@asynccontextmanager
async def _serve(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    """An in-process client over the app with its lifespan running (ASGITransport skips it)."""

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
            yield client


async def _post(client: httpx.AsyncClient, key: str, name: str) -> httpx.Response:
    return await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "x-test-name": name},
        json={"model": "m", "messages": []},
    )


def test_batch_waits_while_a_person_is_being_answered_and_follows_when_they_are_done() -> None:
    """The scheduler must never see batch prefill beside a chat reply.

    Priority admission is the engine's; this is the sidecar's part. Batch
    generation requests arriving while an interactive request is in flight
    wait here, and go through together the moment the reply has fully left.
    """

    async def scenario() -> None:
        upstream = _SlowUpstream()
        async with _serve(_gated_app(upstream)) as client:
            chat = asyncio.create_task(_post(client, "sk-chat", "chat"))
            await asyncio.sleep(0.05)
            assert upstream.started == ["chat"]
            batch = [
                asyncio.create_task(_post(client, "sk-pipeline", f"batch-{i}")) for i in range(3)
            ]
            await asyncio.sleep(0.1)
            # Held: the engine has seen none of them, and the gauge says so.
            assert upstream.started == ["chat"]
            scrape = await client.get("/metrics")
            held = samples(scrape.text, "flakegraph_sidecar_requests_held")
            assert held[(("consumer_class", "batch"),)] == 3
            # A second person is not held by the first.
            tool = asyncio.create_task(_post(client, "sk-tool", "tool"))
            await asyncio.sleep(0.05)
            assert upstream.started == ["chat", "tool"]
            upstream.release.set()
            assert (await chat).status_code == 200
            assert (await tool).status_code == 200
            assert [r.status_code for r in await asyncio.gather(*batch)] == [200, 200, 200]
            assert sorted(upstream.started[2:]) == ["batch-0", "batch-1", "batch-2"]
            scrape = await client.get("/metrics")
            assert (
                samples(scrape.text, "flakegraph_sidecar_requests_held")[
                    (("consumer_class", "batch"),)
                ]
                == 0
            )
            durations = samples(scrape.text, "flakegraph_sidecar_hold_seconds_count")
            assert durations[(("consumer_class", "batch"), ("released_by", "clear"))] == 3

    asyncio.run(scenario())


def test_a_persons_request_cancels_the_batch_already_on_the_engine() -> None:
    """Any engine is ready for a person the moment they arrive.

    Batch already running would share every scheduler step with the reply, so
    it is cancelled rather than left to drain, and its caller is told to retry
    at once - routing sends the retry elsewhere.
    """

    async def scenario() -> None:
        upstream = _SlowUpstream()
        app = _gated_app(upstream)
        async with _serve(app) as client:
            batch = [
                asyncio.create_task(_post(client, "sk-pipeline", f"batch-{i}")) for i in range(2)
            ]
            await asyncio.sleep(0.05)
            assert sorted(upstream.started) == ["batch-0", "batch-1"]
            chat = asyncio.create_task(_post(client, "sk-chat", "chat"))
            preempted = await asyncio.gather(*batch)
            assert [r.status_code for r in preempted] == [503, 503]
            assert {r.headers["retry-after"] for r in preempted} == {"0"}
            upstream.release.set()
            assert (await chat).status_code == 200
            scrape = await client.get("/metrics")
            assert samples(scrape.text, "flakegraph_sidecar_preempted_total") == {
                (("consumer_class", "batch"),): 2
            }
            # Nothing is left counted in flight or holding a slot.
            in_flight = samples(scrape.text, "flakegraph_sidecar_requests_in_flight")
            assert in_flight[(("consumer_class", "batch"),)] == 0
            assert app.state.gate.state()["held_in_flight"] == 0

    asyncio.run(scenario())


def test_batch_is_never_held_when_the_engine_is_free() -> None:
    async def scenario() -> None:
        upstream = _SlowUpstream()
        upstream.release.set()
        async with _serve(_gated_app(upstream)) as client:
            response = await _post(client, "sk-pipeline", "batch")
            assert response.status_code == 200
            scrape = await client.get("/metrics")
            durations = samples(scrape.text, "flakegraph_sidecar_hold_seconds_sum")
            assert durations[(("consumer_class", "batch"), ("released_by", "clear"))] < 0.05

    asyncio.run(scenario())


def test_a_long_session_trickles_batch_through_instead_of_starving_it() -> None:
    """Past the hold ceiling one held request per interval reaches the engine."""

    async def scenario() -> None:
        upstream = _SlowUpstream()
        app = _gated_app(upstream, hold_seconds=0.1, hold_trickle_seconds=0.2)
        async with _serve(app) as client:
            chat = asyncio.create_task(_post(client, "sk-chat", "chat"))
            await asyncio.sleep(0.05)
            batch = [
                asyncio.create_task(_post(client, "sk-pipeline", f"batch-{i}")) for i in range(2)
            ]
            await asyncio.sleep(0.2)
            # One through after the ceiling, the other still waiting for its interval.
            assert len(upstream.started) == 2
            await asyncio.sleep(0.25)
            assert len(upstream.started) == 3
            upstream.release.set()
            await chat
            await asyncio.gather(*batch)
            scrape = await client.get("/metrics")
            durations = samples(scrape.text, "flakegraph_sidecar_hold_seconds_count")
            assert durations[(("consumer_class", "batch"), ("released_by", "trickle"))] == 2

    asyncio.run(scenario())


def test_non_generation_routes_and_disabled_gates_are_untouched() -> None:
    async def scenario() -> None:
        upstream = _SlowUpstream()
        app = _gated_app(upstream, held_classes=[])
        async with _serve(app) as client:
            chat = asyncio.create_task(_post(client, "sk-chat", "chat"))
            await asyncio.sleep(0.05)
            batch = asyncio.create_task(_post(client, "sk-pipeline", "batch"))
            await asyncio.sleep(0.05)
            # Gate disabled: batch went straight to the engine.
            assert upstream.started == ["chat", "batch"]
            upstream.release.set()
            await asyncio.gather(chat, batch)
        upstream = _SlowUpstream()
        async with _serve(_gated_app(upstream)) as client:
            chat = asyncio.create_task(_post(client, "sk-chat", "chat"))
            await asyncio.sleep(0.05)
            # A models listing from a batch key is not generation and is not held.
            models = await client.get("/v1/models", headers={"Authorization": "Bearer sk-pipeline"})
            assert models.status_code == 200
            upstream.release.set()
            await chat

    asyncio.run(scenario())


def test_hold_settings_come_from_the_environment() -> None:
    config = SidecarConfig.from_env(
        {
            "FLAKEGRAPH_SIDECAR_HELD_CLASSES": "batch, dev",
            "FLAKEGRAPH_SIDECAR_HOLD_SECONDS": "45",
            "FLAKEGRAPH_SIDECAR_HOLD_TRICKLE_SECONDS": "2.5",
        }
    )
    assert config.held_classes == ["batch", "dev"]
    assert (config.hold_seconds, config.hold_trickle_seconds) == (45.0, 2.5)
    assert SidecarConfig.from_env({"FLAKEGRAPH_SIDECAR_HELD_CLASSES": ""}).held_classes == []


def test_a_reserved_engine_admits_no_batch_until_it_is_released() -> None:
    """The balancer reserves a machine by capping batch at zero.

    Measured on a GB10: one batch sequence beside a chat reply costs about 70%
    of the reply's speed, so a machine kept ready for people has to be fully
    clear rather than lightly loaded. Setting the cap takes effect at once and
    needs no restart, which is what lets the fleet follow demand at all.
    """

    async def scenario() -> None:
        upstream = _SlowUpstream()
        upstream.release.set()
        app = _gated_app(upstream)
        async with _serve(app) as client:
            gate = app.state.gate
            await gate.set_batch_cap(0)
            batch = asyncio.create_task(_post(client, "sk-pipeline", "batch"))
            await asyncio.sleep(0.1)
            assert upstream.started == []
            # A person is served while batch waits.
            assert (await _post(client, "sk-chat", "chat")).status_code == 200
            assert upstream.started == ["chat"]
            scrape = await client.get("/metrics")
            assert samples(scrape.text, "flakegraph_sidecar_batch_cap")[()] == 0
            await gate.set_batch_cap(None)
            assert (await batch).status_code == 200
            assert upstream.started == ["chat", "batch"]
            scrape = await client.get("/metrics")
            assert samples(scrape.text, "flakegraph_sidecar_batch_cap")[()] == -1

    asyncio.run(scenario())


def test_a_cap_lets_a_bounded_number_of_batch_requests_run() -> None:
    async def scenario() -> None:
        upstream = _SlowUpstream()
        app = _gated_app(upstream)
        async with _serve(app) as client:
            await app.state.gate.set_batch_cap(2)
            running = [asyncio.create_task(_post(client, "sk-pipeline", f"b{i}")) for i in range(4)]
            await asyncio.sleep(0.15)
            assert len(upstream.started) == 2
            upstream.release.set()
            assert [r.status_code for r in await asyncio.gather(*running)] == [200] * 4
            assert len(upstream.started) == 4

    asyncio.run(scenario())


def test_a_cap_lapses_when_the_balancer_stops_calling() -> None:
    """An engine must not stay reserved because the balancer died."""

    async def scenario() -> None:
        upstream = _SlowUpstream()
        upstream.release.set()
        app = _gated_app(upstream, cap_ttl_seconds=0.2)
        async with _serve(app) as client:
            await app.state.gate.set_batch_cap(0)
            assert app.state.gate.batch_cap == 0
            await asyncio.sleep(0.3)
            assert app.state.gate.batch_cap is None
            assert (await _post(client, "sk-pipeline", "batch")).status_code == 200

    asyncio.run(scenario())


def test_the_admission_route_is_readable_and_settable_only_by_protected_classes() -> None:
    """A batch key must not be able to lift the cap that is holding batch back."""

    async def scenario() -> None:
        upstream = _SlowUpstream()
        upstream.release.set()
        app = _gated_app(upstream)
        async with _serve(app) as client:
            headers = {"Authorization": "Bearer sk-chat"}
            state = await client.get("/flakegraph/admission", headers=headers)
            assert state.status_code == 200
            assert state.json()["batch_cap"] is None
            assert state.json()["held_classes"] == ["batch"]

            set_zero = await client.post(
                "/flakegraph/admission", headers=headers, json={"batch_cap": 0}
            )
            assert set_zero.status_code == 200
            assert set_zero.json()["batch_cap"] == 0
            assert set_zero.json()["cap_expires_in_seconds"] is not None

            for key in ("sk-pipeline", "sk-unknown"):
                refused = await client.post(
                    "/flakegraph/admission",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"batch_cap": None},
                )
                assert refused.status_code == 401
            assert app.state.gate.batch_cap == 0

            bad = await client.post(
                "/flakegraph/admission", headers=headers, json={"batch_cap": -1}
            )
            assert bad.status_code == 400
            # The engine never sees any of it.
            assert upstream.started == []

    asyncio.run(scenario())


@pytest.mark.parametrize("key,consumer", [("sk-chat", "interactive"), ("sk-pipeline", "batch")])
def test_invalid_utf8_cannot_retain_admission_or_in_flight_metrics(key: str, consumer: str) -> None:
    """Malformed generation bytes fail before admission and leave batch capacity available."""

    async def exercise() -> None:
        upstream = _SlowUpstream()
        upstream.release.set()
        app = _gated_app(upstream)
        await app.state.gate.set_batch_cap(1)
        async with _serve(app) as client:
            malformed = await client.post(
                "/v1/chat/completions",
                content=b"\xff",
                headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
            )
            assert malformed.status_code == 400
            assert not upstream.started
            state = app.state.gate.state()
            assert state["protected_in_flight"] == state["held_in_flight"] == 0
            assert not app.state.metrics.registry.get_sample_value(
                "flakegraph_sidecar_requests_in_flight", {"consumer_class": consumer}
            )
            # With a 30-second gate hold, any orphan would exceed this deadline.
            good = await asyncio.wait_for(
                _post(client, "sk-pipeline", "batch-after-bad"), timeout=1
            )
            assert good.status_code == 200
            state = app.state.gate.state()
            assert state["protected_in_flight"] == state["held_in_flight"] == 0
            assert upstream.started == ["batch-after-bad"]

    asyncio.run(exercise())


@pytest.mark.parametrize("key,band", [("sk-chat", 0), ("sk-tool", 10), ("sk-pipeline", 100)])
def test_messages_uses_trusted_header_priority(key: str, band: int) -> None:
    """The pinned Anthropic adapter reads priority from the vLLM header."""

    upstream = RecordingUpstream()
    with _client(upstream) as client:
        response = client.post(
            "/v1/messages",
            headers={"Authorization": f"Bearer {key}", "X-Vllm-Priority": "-100"},
            json={"model": "m", "messages": [], "priority": -100},
        )
    assert response.status_code == 200
    assert upstream.requests[-1].headers["x-vllm-priority"] == str(band)
    assert "priority" not in upstream.last_body


def test_messages_participates_in_both_admission_classes() -> None:
    """Every enabled generation route shares the batch/protected policy."""

    app = _gated_app(_SlowUpstream())
    gate = app.state.gate
    assert gate.holds("batch", "/v1/messages")
    assert gate.protects("interactive", "/v1/messages")
    assert not gate.holds("batch", "/v1/embeddings")


def test_releasing_idle_cap_wakes_an_existing_admission() -> None:
    """An administrative release wakes queued work without another inference call."""

    async def scenario() -> None:
        upstream = _SlowUpstream()
        upstream.release.set()
        app = _gated_app(upstream)
        async with _serve(app) as client:
            headers = {"Authorization": "Bearer sk-chat"}
            await client.post("/flakegraph/admission", headers=headers, json={"batch_cap": 0})
            queued = asyncio.create_task(_post(client, "sk-pipeline", "queued"))
            await asyncio.sleep(0.05)
            assert upstream.started == []
            released = await client.post(
                "/flakegraph/admission", headers=headers, json={"batch_cap": None}
            )
            assert released.status_code == 200
            assert (await asyncio.wait_for(queued, timeout=1)).status_code == 200
            assert upstream.started == ["queued"]

    asyncio.run(scenario())


def test_expiring_idle_cap_wakes_an_existing_admission() -> None:
    """The cap's own deadline releases a waiter while the engine stays idle."""

    async def scenario() -> None:
        upstream = _SlowUpstream()
        upstream.release.set()
        app = _gated_app(upstream, cap_ttl_seconds=0.15)
        async with _serve(app) as client:
            await app.state.gate.set_batch_cap(0)
            assert (
                await asyncio.wait_for(_post(client, "sk-pipeline", "expiry"), timeout=1)
            ).status_code == 200
            assert upstream.started == ["expiry"]

    asyncio.run(scenario())
