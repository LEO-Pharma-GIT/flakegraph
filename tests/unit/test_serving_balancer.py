"""Behavioural tests for the standby policy: a machine ready for the next person."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from kg_processor.serving.balancer import (
    LANE_BATCH,
    LANE_LABEL,
    LANE_PEOPLE,
    ROLE_BATCH,
    ROLE_LABEL,
    ROLE_SERVING,
    ROLE_STANDBY,
    Balancer,
    BalancerConfig,
    EngineState,
    _Memory,
    lane_for,
    ocr_lanes,
    plan,
    role_for,
    standby_names,
)


def _engine(
    name: str,
    *,
    interactive: int = 0,
    batch: int = 0,
    cap: int | None = None,
    cap_expires_in: float | None = None,
    neighbours: int = 0,
    ready: bool = True,
    reachable: bool = True,
    role: str = ROLE_BATCH,
) -> EngineState:
    return EngineState(
        name=name,
        address=f"10.0.0.{name[-1]}",
        ready=ready,
        role=role,
        interactive_in_flight=interactive,
        batch_in_flight=batch,
        batch_cap=cap,
        cap_expires_in_seconds=cap_expires_in,
        neighbours=neighbours,
        reachable=reachable,
    )


def _config(**overrides: Any) -> BalancerConfig:
    return BalancerConfig.model_validate({"standby": 1, "api_key": "sk-chat", **overrides})


def test_a_fleet_nobody_is_using_keeps_every_machine_on_batch() -> None:
    """The first person's request clears the engine it reaches; none waits idle."""

    engines = [_engine(f"vllm-{i}", batch=i) for i in range(4)]
    decision = plan(engines, _config(), _Memory(), now=0.0)

    assert (decision.reserve, decision.release) == ((), ())
    assert (decision.standby_now, decision.standby_target) == (0, 0)


def test_once_someone_is_being_answered_a_machine_is_cleared_for_the_next() -> None:
    """The invariant is a machine already clear when the next person arrives."""

    engines = [_engine("vllm-9", interactive=1)] + [
        _engine(f"vllm-{i}", batch=i + 1) for i in range(4)
    ]
    decision = plan(engines, _config(), _Memory(), now=0.0)

    # The least loaded machine drains soonest, so it is the one to clear.
    assert decision.reserve == ("vllm-0",)
    assert (decision.standby_now, decision.standby_target) == (0, 1)


def test_a_standby_nobody_needs_any_more_goes_back_to_batch() -> None:
    """Past the last conversation's cooldown the kept machine is released, not renewed."""

    engines = [
        _engine("vllm-0", cap=0, cap_expires_in=30.0),
        _engine("vllm-1", batch=4),
        _engine("vllm-2", batch=9),
    ]
    decision = plan(engines, _config(), _Memory(), now=0.0)

    assert decision.renew == ()
    assert decision.release == ("vllm-0",)


def test_when_someone_takes_the_ready_machine_another_is_cleared_behind_them() -> None:
    """This is the whole point: the next arrival must also find one ready."""

    memory = _Memory()
    engines = [
        _engine("vllm-0", interactive=1, cap=0),  # the person landed here
        _engine("vllm-1", batch=4),
        _engine("vllm-2", batch=1),
    ]
    decision = plan(engines, _config(), memory, now=100.0)

    assert decision.reserve == ("vllm-2",)
    assert decision.standby_now == 0  # the reserved one is busy now, so none is ready
    assert role_for(engines[0], decision, memory, now=100.0) == ROLE_SERVING
    assert role_for(engines[2], decision, memory, now=100.0) == ROLE_STANDBY


def test_a_machine_stays_reserved_through_the_pauses_in_a_conversation() -> None:
    """Agentic sessions are bursty; re-draining on every pause drains constantly."""

    memory = _Memory()
    engines = [_engine("vllm-0", interactive=1, cap=0), _engine("vllm-1", cap=0)]
    plan(engines, _config(cooldown_seconds=600), memory, now=0.0)

    idle = [_engine("vllm-0", cap=0), _engine("vllm-1", cap=0)]
    # Ten seconds later, between turns: still theirs.
    during = plan(idle, _config(cooldown_seconds=600), memory, now=10.0)
    assert "vllm-0" not in during.release
    # Well after the conversation: handed back.
    after = plan(idle, _config(cooldown_seconds=600), memory, now=601.0)
    assert "vllm-0" in after.release or "vllm-1" in after.release


def test_extra_machines_are_kept_ready_when_people_have_to_share() -> None:
    """One standby cannot absorb arrivals that overlap; the floor grows and decays."""

    config = _config(standby=1, max_reserved=3, grow_after_seconds=60, decay_after_seconds=600)
    memory = _Memory()
    shared = [
        _engine("vllm-0", interactive=2, cap=0),  # two people on one machine
        _engine("vllm-1", batch=2),
        _engine("vllm-2", batch=2),
        _engine("vllm-3", batch=2),
    ]
    plan(shared, config, memory, now=0.0)
    assert memory.grown == 0
    plan(shared, config, memory, now=61.0)
    assert memory.grown == 1

    quiet = [_engine("vllm-0", cap=0), _engine("vllm-1"), _engine("vllm-2"), _engine("vllm-3")]
    plan(quiet, config, memory, now=61.0 + 601.0 + 601.0)
    assert memory.grown == 0


def test_the_whole_fleet_is_never_reserved() -> None:
    """Batch work must always have somewhere to run, whatever the demand."""

    config = _config(standby=5, max_reserved=5)
    # Five machines wanted ready, four are free: three are cleared and one is
    # left for batch, rather than the pipeline being stopped outright.
    person = _engine("vllm-9", interactive=1)
    engines = [person] + [_engine(f"vllm-{i}") for i in range(4)]
    decision = plan(engines, config, _Memory(), now=0.0)
    assert len(decision.reserve) == 3

    # And a fleet with one free machine keeps it on batch work.
    assert plan([person, _engine("vllm-0")], config, _Memory(), now=0.0).reserve == ()


def test_unready_and_unreachable_engines_are_not_counted_or_touched() -> None:
    """A machine the balancer cannot see is not a machine it can reserve."""

    engines = [
        _engine("vllm-0", ready=False),
        _engine("vllm-1", reachable=False),
        _engine("vllm-2", batch=5),
        _engine("vllm-3", batch=1),
        _engine("vllm-4", batch=9),
        _engine("vllm-9", interactive=1),
    ]
    decision = plan(engines, _config(), _Memory(), now=0.0)

    # Only the three usable ones are candidates, and the emptiest is chosen.
    assert decision.reserve == ("vllm-3",)
    assert plan([_engine("vllm-0", ready=False)], _config(), _Memory(), 0.0).reserve == ()


def test_a_pass_sets_caps_then_labels_and_reports_what_it_did() -> None:
    """Caps are the enforcement and go first; labels follow so routing agrees."""

    posted: list[tuple[str, Any]] = []
    labels: list[tuple[str, str]] = []
    lanes: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            name = request.url.host
            busy = {"protected_in_flight": 1} if name.endswith("2") else {}
            return httpx.Response(200, json={"batch_cap": None, "held_in_flight": 3, **busy})
        posted.append((request.url.host, request.read()))
        return httpx.Response(200, json={"batch_cap": 0})

    class _View:
        def engines(self, namespace: str, selector: str) -> list[dict[str, Any]]:
            return [
                {"name": f"10.0.0.{i}", "address": f"10.0.0.{i}", "ready": True, "role": ROLE_BATCH}
                for i in (1, 2, 3)
            ]

        def label(self, namespace: str, pod: str, key: str, value: str) -> None:
            assert key in (ROLE_LABEL, LANE_LABEL)
            (labels if key == ROLE_LABEL else lanes).append((pod, value))

        def pods(self, namespace: str, selector: str) -> list[dict[str, Any]]:
            return []

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with client:
            balancer = Balancer(_config(), _View(), client=client)
            decision = await balancer.pass_once()
            # 10.0.0.2 is answering someone, so one of the other two is cleared.
            assert len(decision.reserve) == 1
            assert "10.0.0.2" not in decision.reserve
            assert [host for host, _ in posted] == list(decision.reserve)
            assert b'"batch_cap": 0' in posted[0][1] or b'"batch_cap":0' in posted[0][1]
            assert (decision.reserve[0], ROLE_STANDBY) in labels
            assert ("10.0.0.2", ROLE_SERVING) in labels
            # Routing reads the lane: the cleared machine and the one a person
            # is on are both for people, the rest for batch.
            assert (decision.reserve[0], LANE_PEOPLE) in lanes
            assert ("10.0.0.2", LANE_PEOPLE) in lanes
            others = {"10.0.0.1", "10.0.0.3"} - set(decision.reserve)
            assert {(host, LANE_BATCH) for host in others} <= set(lanes)

    asyncio.run(scenario())


def test_an_engine_that_will_not_answer_is_left_alone_and_the_pass_continues() -> None:
    """One unreachable engine must not stop the fleet from being balanced."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "10.0.0.1":
            raise httpx.ConnectError("refused", request=request)
        busy = {"protected_in_flight": 1} if request.url.host == "10.0.0.4" else {}
        return httpx.Response(200, json={"batch_cap": None, "held_in_flight": 0, **busy})

    class _View:
        def engines(self, namespace: str, selector: str) -> list[dict[str, Any]]:
            return [
                {"name": "10.0.0.1", "address": "10.0.0.1", "ready": True, "role": ROLE_BATCH},
                {"name": "10.0.0.2", "address": "10.0.0.2", "ready": True, "role": ROLE_BATCH},
                {"name": "10.0.0.3", "address": "10.0.0.3", "ready": True, "role": ROLE_BATCH},
                {"name": "10.0.0.4", "address": "10.0.0.4", "ready": True, "role": ROLE_BATCH},
            ]

        def label(self, namespace: str, pod: str, key: str, value: str) -> None:
            return None

        def pods(self, namespace: str, selector: str) -> list[dict[str, Any]]:
            return []

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with client:
            decision = await Balancer(_config(), _View(), client=client).pass_once()
            assert "10.0.0.1" not in decision.reserve
            assert len(decision.reserve) == 1

    asyncio.run(scenario())


def test_standby_names_reports_the_machines_currently_clear() -> None:
    engines = [_engine("vllm-0", cap=0), _engine("vllm-1"), _engine("vllm-2", cap=0)]
    assert standby_names(engines) == ["vllm-0", "vllm-2"]


def test_configuration_comes_from_the_environment() -> None:
    config = BalancerConfig.from_env(
        {
            "FLAKEGRAPH_BALANCER_STANDBY": "2",
            "FLAKEGRAPH_BALANCER_COOLDOWN_SECONDS": "120",
            "FLAKEGRAPH_BALANCER_NAMESPACE": "other",
        }
    )
    assert (config.standby, config.cooldown_seconds, config.namespace) == (2, 120.0, "other")
    with pytest.raises(ValueError, match="standby"):
        BalancerConfig.model_validate({"standby": -1})


def test_the_standby_is_renewed_before_its_cap_lapses() -> None:
    """The machine kept clear has to stay the same machine.

    The cap expires by itself after five minutes, and a pass that only ever
    reserves when it is short would let it lapse, find itself short on the next
    pass, and clear a *different* engine - one still draining batch work. Every
    five minutes there would be a moment with nothing ready, which is exactly
    the moment a person's first message arrives.
    """

    engines = [
        _engine("vllm-0", batch=0, cap=0, cap_expires_in=30.0),
        _engine("vllm-1", batch=4),
        _engine("vllm-2", batch=9),
        _engine("vllm-9", interactive=1),
    ]
    decision = plan(engines, _config(renew_before_seconds=120.0), _Memory(), now=0.0)

    assert decision.renew == ("vllm-0",)
    assert decision.reserve == () and decision.release == ()
    # It is still the standby it was: the count does not dip while renewing.
    assert (decision.standby_now, decision.standby_target) == (1, 1)


def test_a_standby_with_plenty_of_cap_left_is_not_touched() -> None:
    """Renewal is for a cap about to lapse, not chatter on every pass."""

    engines = [
        _engine("vllm-0", batch=0, cap=0, cap_expires_in=280.0),
        _engine("vllm-1", batch=4),
        _engine("vllm-9", interactive=1),
    ]
    decision = plan(engines, _config(renew_before_seconds=120.0), _Memory(), now=0.0)

    assert (decision.renew, decision.reserve, decision.release) == ((), (), ())
    assert decision.reason == "at target"


def test_a_renewal_reaches_the_sidecar_as_a_cap() -> None:
    """A renewal that never leaves the planner is a standby that still lapses."""

    posted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            if request.url.host == "10.0.0.2":
                # Someone is being answered here, so a standby is wanted.
                return httpx.Response(200, json={"batch_cap": None, "protected_in_flight": 1})
            if request.url.host == "10.0.0.3":
                return httpx.Response(200, json={"batch_cap": None, "held_in_flight": 4})
            # Already clear, and nearly out of time.
            return httpx.Response(
                200,
                json={"batch_cap": 0, "held_in_flight": 0, "cap_expires_in_seconds": 20.0},
            )
        posted.append(request.url.host)
        return httpx.Response(200, json={"batch_cap": 0})

    class _View:
        def engines(self, namespace: str, selector: str) -> list[dict[str, Any]]:
            return [
                {
                    "name": f"10.0.0.{i}",
                    "address": f"10.0.0.{i}",
                    "ready": True,
                    "role": ROLE_STANDBY,
                }
                for i in (1, 2, 3)
            ]

        def label(self, namespace: str, pod: str, key: str, value: str) -> None:
            return None

        def pods(self, namespace: str, selector: str) -> list[dict[str, Any]]:
            return []

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with client:
            balancer = Balancer(_config(renew_before_seconds=120.0), _View(), client=client)
            decision = await balancer.pass_once()
            assert decision.renew == ("10.0.0.1",)
            # Every renewed machine is written to, and nothing else is.
            assert sorted(posted) == sorted(decision.renew)

    asyncio.run(scenario())


def test_a_person_keeps_their_lane_between_turns() -> None:
    """The lane follows the cooldown, not the stream.

    A reply ending is not the conversation ending. Handing the engine back to
    batch between turns would put batch beside the person's next reply and
    move them off the engine that holds their prompt.
    """

    config = _config(cooldown_seconds=600)
    memory = _Memory()
    engine = _engine("vllm-1")
    memory.last_interactive_at["vllm-1"] = 1000.0
    # Mid-reply, between turns, and after the conversation has gone quiet.
    assert lane_for(engine, ROLE_SERVING, config, memory, now=1000.0) == LANE_PEOPLE
    assert lane_for(engine, ROLE_BATCH, config, memory, now=1300.0) == LANE_PEOPLE
    assert lane_for(engine, ROLE_BATCH, config, memory, now=1700.0) == LANE_BATCH


def test_the_standby_is_always_for_people_and_nothing_else_is_by_default() -> None:
    config = _config()
    memory = _Memory()
    assert lane_for(_engine("vllm-0"), ROLE_STANDBY, config, memory, now=0.0) == LANE_PEOPLE
    assert lane_for(_engine("vllm-2"), ROLE_BATCH, config, memory, now=0.0) == LANE_BATCH


def test_serving_engines_can_be_left_to_batch_when_throughput_wins() -> None:
    """The other side of the trade is a setting, not a code change."""

    config = _config(serving_lane="batch")
    memory = _Memory()
    memory.last_interactive_at["vllm-1"] = 0.0
    assert lane_for(_engine("vllm-1"), ROLE_SERVING, config, memory, now=1.0) == LANE_BATCH
    # The cleared machine stays for people either way: that is what it is for.
    assert lane_for(_engine("vllm-0"), ROLE_STANDBY, config, memory, now=1.0) == LANE_PEOPLE
    assert BalancerConfig.from_env({"FLAKEGRAPH_BALANCER_SERVING_LANE": "batch"}).serving_lane == (
        "batch"
    )


def test_the_standby_is_reserved_on_the_quietest_node() -> None:
    """A clear engine beside a busy neighbour is not clear.

    Measured: a GPU neighbour cost a clear engine ~29 % of its speed, and a
    GPU neighbour with memory traffic ~65 %. So the node's neighbours come
    before which engine drains soonest.
    """

    engines = [
        _engine("vllm-0", batch=0, neighbours=2),  # least batch, crowded node
        _engine("vllm-1", batch=6, neighbours=0),  # busier engine, quiet node
        _engine("vllm-2", batch=3, neighbours=1),
        _engine("vllm-9", interactive=1),
    ]
    decision = plan(engines, _config(), _Memory(), now=0.0)
    assert decision.reserve == ("vllm-1",)


def test_parsing_replicas_follow_their_nodes_lane() -> None:
    replicas = [
        {"name": "mineru-0", "node": "a", "lane": LANE_BATCH},
        {"name": "mineru-1", "node": "b", "lane": LANE_BATCH},
        {"name": "mineru-2", "node": "c", "lane": LANE_PEOPLE},
    ]
    assert ocr_lanes(replicas, people_nodes={"a"}) == {
        "mineru-0": LANE_PEOPLE,
        "mineru-1": LANE_BATCH,
        "mineru-2": LANE_BATCH,
    }


def test_one_parsing_replica_always_stays_with_batch() -> None:
    """Documents keep moving even when every engine is serving someone."""

    replicas = [
        {"name": "mineru-0", "node": "a", "lane": LANE_PEOPLE},
        {"name": "mineru-1", "node": "b", "lane": LANE_BATCH},
    ]
    lanes = ocr_lanes(replicas, people_nodes={"a", "b"})
    # The one already taking batch work keeps it, rather than swapping.
    assert lanes == {"mineru-0": LANE_PEOPLE, "mineru-1": LANE_BATCH}
    assert ocr_lanes([], people_nodes={"a"}) == {}


def test_a_pass_labels_parsing_replicas_and_counts_neighbours() -> None:
    labels: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            busy = {"protected_in_flight": 1} if request.url.host == "10.0.0.4" else {}
            return httpx.Response(200, json={"batch_cap": None, "held_in_flight": 2, **busy})
        return httpx.Response(200, json={"batch_cap": 0})

    class _View:
        def engines(self, namespace: str, selector: str) -> list[dict[str, Any]]:
            return [
                {
                    "name": f"10.0.0.{i}",
                    "address": f"10.0.0.{i}",
                    "ready": True,
                    "role": ROLE_BATCH,
                    "lane": LANE_BATCH,
                    "node": f"node-{i}",
                }
                for i in (1, 2, 3, 4)
            ]

        def pods(self, namespace: str, selector: str) -> list[dict[str, Any]]:
            if selector == "spark-role=executor":
                return [{"name": f"exec-{i}", "node": f"node-{i}"} for i in (1, 2, 4)]
            return [
                {"name": f"mineru-{i}", "node": f"node-{i}", "lane": LANE_BATCH} for i in (1, 2, 3)
            ]

        def label(self, namespace: str, pod: str, key: str, value: str) -> None:
            labels.append((pod, key, value))

    config = _config(neighbour_selectors=("spark-role=executor",), ocr_selector="component=mineru")

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with client:
            decision = await Balancer(config, _View(), client=client).pass_once()
            # node-3 has no executor, so its engine is the one cleared.
            assert decision.reserve == ("10.0.0.3",)

    asyncio.run(scenario())
    assert ("10.0.0.3", LANE_LABEL, LANE_PEOPLE) in labels
    # Its node's parsing replica leaves the batch lane; the others stay.
    assert ("mineru-3", LANE_LABEL, LANE_PEOPLE) in labels
    assert not [entry for entry in labels if entry[0] in ("mineru-1", "mineru-2")]


def test_neighbour_selectors_come_from_one_variable() -> None:
    config = BalancerConfig.from_env(
        {"FLAKEGRAPH_BALANCER_NEIGHBOUR_SELECTORS": "spark-role=executor; app=embed,tier=x"}
    )
    assert config.neighbour_selectors == ("spark-role=executor", "app=embed,tier=x")
