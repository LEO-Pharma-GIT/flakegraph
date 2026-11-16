# SPDX-License-Identifier: Apache-2.0
"""Keep machines clear and ready for people, and give the rest to batch work.

The problem this solves is not admission order — the engines already serve a
person's request first. It is what shares each scheduler step with that
request once it is running. On a GB10 a chat reply on an engine with nothing
else on it streams about 50 tokens a second, and about 14 with a single batch
sequence beside it. There is no small amount of batch work that
is free to leave on a machine somebody is using.

When nobody is being answered, every machine runs batch work: the sidecar
cancels the batch on an engine the moment a person's request reaches it, so
any machine is ready for the first arrival. While people are using the fleet,
one more machine - or however many ``standby`` says - is kept clear, so the
next person lands on a machine of their own rather than sharing the first
one's. When the conversation ends its machine goes back to batch work after a
cooldown, because agentic sessions are bursty and a machine re-cleared on
every pause is a machine cleared continuously.

Two properties make this safe to run unattended:

* The only thing it changes is an engine's batch admission, which takes effect
  at once and needs no restart. Reserving a machine costs the drain of what is
  already running on it (seconds to about two minutes), not an engine start.
* Every cap it sets carries a time to live in the sidecar. If this process
  stops calling, the fleet returns to ordinary behaviour on its own rather
  than staying reserved forever.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, Field

from kg_processor.fleet.kubectl import ClusterTarget, kubectl_json

logger = logging.getLogger(__name__)

# The label that carries an engine's role. The endpoint pickers select on it,
# so a person's request reaches a cleared machine and batch work does not.
ROLE_LABEL = "flakegraph.io/serving-role"
ROLE_STANDBY = "standby"
ROLE_SERVING = "serving"
ROLE_BATCH = "batch"
# The lane routing reads. The role says what the balancer did to an engine; the
# lane says who should be sent to it, and the router's header-affinity scorer
# matches it against the lane each gateway alias stamps on its requests.
LANE_LABEL = "flakegraph.io/lane"
LANE_PEOPLE = "people"
LANE_BATCH = "batch"

# After a reply finishes, the machine is still labelled as serving for a short
# grace period: a conversation's next turn arrives in seconds, and relabelling
# twice in that window only churns the pickers.
SERVING_GRACE_SECONDS = 60.0


class BalancerConfig(BaseModel):
    """How many machines to keep ready, and how patiently."""

    namespace: str = "flakegraph"
    # The engines this balancer governs, by label selector.
    engine_selector: str = "app.kubernetes.io/component=model-serving"
    engine_port: int = 8000
    # Machines kept clear for the next arrival while people are using the
    # fleet. None at rest: a person's first request clears its own engine.
    standby: int = Field(default=1, ge=0)
    # An upper bound on everything reserved at once, so a burst of people
    # cannot stop the pipeline outright.
    max_reserved: int = Field(default=4, ge=0)
    # How long a machine stays reserved after the last interactive request.
    cooldown_seconds: float = Field(default=600.0, ge=0)
    # How long the balancer waits before concluding that people are arriving
    # faster than one standby can absorb, and keeping an extra one ready.
    grow_after_seconds: float = Field(default=60.0, ge=0)
    # And how long of no sharing before it gives that extra one back.
    decay_after_seconds: float = Field(default=1800.0, ge=0)
    interval_seconds: float = Field(default=5.0, gt=0)
    # Which lane an engine joins once a person is on it. "people" keeps batch
    # off it until cooldown_seconds after their last message: one batch
    # sequence beside a reply costs most of its speed. "batch"
    # lets batch share it, trading that speed for pipeline throughput.
    serving_lane: Literal["people", "batch"] = "people"
    # Label selectors for long-lived or already-running work that slows decode
    # on its node - Spark executors, the embedding server. The standby is
    # reserved on the node with the fewest; new executors avoid people engines
    # by their own anti-affinity, but a running one cannot be moved.
    neighbour_selectors: tuple[str, ...] = ()
    # Label selector for the document-parsing replicas. Each is labelled with
    # the lane of the engine on its node, so the OCR shim - which resolves only
    # batch-lane replicas - sends no parsing to a node a person is using.
    ocr_selector: str | None = None
    # How much of a standby's cap must be left before it is renewed. The cap
    # expires on its own so a dead balancer cannot hold batch back for ever,
    # which means a live one has to keep saying so. Generous against the
    # sidecar's own ttl: a few missed passes must not cost the standby.
    renew_before_seconds: float = Field(default=120.0, ge=0)
    request_timeout_seconds: float = Field(default=5.0, gt=0)
    # The interactive key: the balancer reads and sets caps through the same
    # authenticated surface as inference, and only a protected class may.
    api_key: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> BalancerConfig:
        """Build a configuration from ``FLAKEGRAPH_BALANCER_*`` variables."""

        env = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        for field_name, variable in (
            ("namespace", "FLAKEGRAPH_BALANCER_NAMESPACE"),
            ("engine_selector", "FLAKEGRAPH_BALANCER_ENGINE_SELECTOR"),
            ("engine_port", "FLAKEGRAPH_BALANCER_ENGINE_PORT"),
            ("standby", "FLAKEGRAPH_BALANCER_STANDBY"),
            ("max_reserved", "FLAKEGRAPH_BALANCER_MAX_RESERVED"),
            ("cooldown_seconds", "FLAKEGRAPH_BALANCER_COOLDOWN_SECONDS"),
            ("grow_after_seconds", "FLAKEGRAPH_BALANCER_GROW_AFTER_SECONDS"),
            ("decay_after_seconds", "FLAKEGRAPH_BALANCER_DECAY_AFTER_SECONDS"),
            ("interval_seconds", "FLAKEGRAPH_BALANCER_INTERVAL_SECONDS"),
            ("serving_lane", "FLAKEGRAPH_BALANCER_SERVING_LANE"),
            ("ocr_selector", "FLAKEGRAPH_BALANCER_OCR_SELECTOR"),
            ("renew_before_seconds", "FLAKEGRAPH_BALANCER_RENEW_BEFORE_SECONDS"),
            ("api_key", "FLAKEGRAPH_BALANCER_API_KEY"),
        ):
            if variable in env:
                values[field_name] = env[variable]
        if env.get("FLAKEGRAPH_BALANCER_NEIGHBOUR_SELECTORS"):
            values["neighbour_selectors"] = tuple(
                part.strip()
                for part in env["FLAKEGRAPH_BALANCER_NEIGHBOUR_SELECTORS"].split(";")
                if part.strip()
            )
        if "FLAKEGRAPH_BALANCER_API_KEY_FILE" in env and "api_key" not in values:
            values["api_key"] = _read_secret(env["FLAKEGRAPH_BALANCER_API_KEY_FILE"])
        return cls.model_validate(values)


@dataclass(frozen=True)
class EngineState:
    """One engine as the balancer sees it this pass."""

    name: str
    address: str
    ready: bool
    role: str
    lane: str = ""
    # The node it runs on, and how many heavy neighbours share it: work that
    # contends for the same GPU and memory bandwidth as decode. A GPU neighbour
    # costs a clear engine ~29 % of its speed and a GPU neighbour plus memory
    # traffic ~65 %.
    node: str = ""
    neighbours: int = 0
    # From the sidecar: who is on this engine right now.
    interactive_in_flight: int = 0
    batch_in_flight: int = 0
    batch_cap: int | None = None
    # How long the cap it carries still has to run. The cap expires on its own
    # so that a balancer that dies cannot leave the fleet holding batch back
    # for ever; the consequence is that a standby it means to keep has to be
    # renewed before then, or the machine quietly goes back to batch work.
    cap_expires_in_seconds: float | None = None
    reachable: bool = True


@dataclass
class _Memory:
    """What the balancer remembers between passes: only timing, never truth.

    Roles come from labels and occupancy from the sidecars, so a restarted
    balancer rebuilds its picture from the cluster. These timestamps only make
    it patient — a fresh process is briefly more eager, which is harmless.
    """

    last_interactive_at: dict[str, float] = field(default_factory=dict)
    short_since: float | None = None
    last_shortage_at: float | None = None
    grown: int = 0


@dataclass(frozen=True)
class Decision:
    """What one pass concluded: the caps to set, and why."""

    reserve: tuple[str, ...] = ()
    renew: tuple[str, ...] = ()
    release: tuple[str, ...] = ()
    standby_target: int = 0
    standby_now: int = 0
    reason: str = ""


def _follow_demand(
    memory: _Memory,
    config: BalancerConfig,
    *,
    busy: Sequence[EngineState],
    standby: Sequence[EngineState],
    now: float,
) -> None:
    """Grow the floor while people are sharing machines, and give it back later.

    People sharing an engine with other people is the signal that one standby
    is not keeping up with arrivals; a stretch with nobody sharing is the
    signal that the extra one is no longer earning its batch work.
    """

    shortage = any(engine.interactive_in_flight > 1 for engine in busy) or (
        bool(busy) and not standby
    )
    if shortage:
        memory.last_shortage_at = now
        # Not ``or``: a timestamp of zero is falsy and would restart the clock
        # every pass, so the floor would never grow.
        if memory.short_since is None:
            memory.short_since = now
        if now - memory.short_since >= config.grow_after_seconds:
            memory.grown = min(memory.grown + 1, max(0, config.max_reserved - config.standby))
            memory.short_since = now
        return
    memory.short_since = None
    if (
        memory.grown > 0
        and memory.last_shortage_at is not None
        and now - memory.last_shortage_at >= config.decay_after_seconds
    ):
        memory.grown -= 1


def plan(
    engines: Sequence[EngineState],
    config: BalancerConfig,
    memory: _Memory,
    now: float,
) -> Decision:
    """Decide which engines to clear and which to give back.

    Pure, so the policy can be exercised over a fleet's worth of situations
    without a cluster: everything it needs is in ``engines`` and ``memory``.
    """

    usable = [engine for engine in engines if engine.ready and engine.reachable]
    if not usable:
        return Decision(reason="no reachable engines")

    for engine in usable:
        if engine.interactive_in_flight > 0:
            memory.last_interactive_at[engine.name] = now

    def serving(engine: EngineState) -> bool:
        last = memory.last_interactive_at.get(engine.name)
        return engine.interactive_in_flight > 0 or (
            last is not None and now - last < config.cooldown_seconds
        )

    busy = [engine for engine in usable if serving(engine)]
    # A machine is ready for the next arrival only if it is both reserved and
    # not already answering someone.
    standby = [engine for engine in usable if not serving(engine) and engine.batch_cap == 0]
    # People sharing an engine with other people is the signal that one
    # standby is not keeping up with arrivals.
    _follow_demand(memory, config, busy=busy, standby=standby, now=now)

    # Machines not currently answering anyone are the ones that can be
    # reserved or run batch work. At least one of them stays with batch,
    # whatever the demand, so the pipeline never stops outright. With nobody
    # being answered nothing is reserved: the first person's request clears
    # the engine it reaches.
    free = len(usable) - len(busy)
    wanted = config.standby + memory.grown if busy else 0
    target = min(wanted, config.max_reserved, max(0, free - 1))

    # A standby that is no longer wanted is released below, not renewed.
    expiring = tuple(
        engine.name
        for engine in (standby if len(standby) <= target else ())
        if engine.cap_expires_in_seconds is not None
        and engine.cap_expires_in_seconds <= config.renew_before_seconds
    )
    if expiring:
        # Renewing keeps the *same* machine clear. Letting the cap lapse and
        # reserving again next pass would leave a window with nothing ready,
        # and would usually pick a different machine - one still draining its
        # batch work, and cold where the last one was warm.
        return Decision(
            renew=expiring,
            standby_target=target,
            standby_now=len(standby),
            reason="standby cap about to expire",
        )
    if len(standby) < target:
        # Clear the machines with the least batch work in flight: they drain
        # soonest, so the next person waits least.
        # The quietest node first - the fewest heavy neighbours to share its GPU
        # and memory bandwidth with - then the machine with the least batch in
        # flight, which drains soonest.
        candidates = sorted(
            (e for e in usable if not serving(e) and e.batch_cap != 0),
            key=lambda e: (e.neighbours, e.batch_in_flight, e.name),
        )
        reserve = tuple(engine.name for engine in candidates[: target - len(standby)])
        if reserve:
            return Decision(
                reserve=reserve,
                standby_target=target,
                standby_now=len(standby),
                reason=f"{len(standby)} of {target} machines ready",
            )
    if len(standby) > target:
        # Give back the most recently cleared, so a machine that has been
        # ready longest stays ready.
        surplus = sorted(standby, key=lambda e: e.name, reverse=True)[: len(standby) - target]
        return Decision(
            release=tuple(engine.name for engine in surplus),
            standby_target=target,
            standby_now=len(standby),
            reason=f"{len(standby)} ready, {target} wanted",
        )
    # Machines that finished serving and are past their cooldown are not
    # standby and not capped: hand them back.
    stale = tuple(
        engine.name
        for engine in usable
        if engine.batch_cap == 0 and not serving(engine) and engine not in standby
    )
    if stale:
        return Decision(
            release=stale,
            standby_target=target,
            standby_now=len(standby),
            reason="past cooldown",
        )
    return Decision(standby_target=target, standby_now=len(standby), reason="at target")


def role_for(engine: EngineState, decision: Decision, memory: _Memory, now: float) -> str:
    """The label an engine should carry, so routing follows the caps."""

    if engine.name in decision.release:
        return ROLE_BATCH
    if engine.name in decision.reserve:
        return ROLE_STANDBY
    last = memory.last_interactive_at.get(engine.name)
    if engine.interactive_in_flight > 0:
        return ROLE_SERVING
    if engine.batch_cap == 0:
        recent = last is not None and now - last < SERVING_GRACE_SECONDS
        return ROLE_SERVING if recent else ROLE_STANDBY
    return ROLE_BATCH


def lane_for(
    engine: EngineState,
    role: str,
    config: BalancerConfig,
    memory: _Memory,
    now: float,
) -> str:
    """The lane routing should send traffic to this engine in.

    Follows the cooldown rather than the stream: a role reads ``batch`` again
    the moment a reply ends, but the person is usually mid-conversation, and
    handing the engine back to batch between their turns would put batch
    beside their next reply and move them off the engine holding their prompt.
    """

    if role == ROLE_STANDBY:
        return LANE_PEOPLE
    if config.serving_lane == LANE_BATCH:
        return LANE_BATCH
    last = memory.last_interactive_at.get(engine.name)
    if role == ROLE_SERVING or (last is not None and now - last < config.cooldown_seconds):
        return LANE_PEOPLE
    return LANE_BATCH


def ocr_lanes(replicas: Sequence[Mapping[str, str]], people_nodes: set[str]) -> dict[str, str]:
    """The lane each parsing replica should carry: its node's.

    A replica beside an engine people are using takes no new parsing work,
    since parsing on the same GPU costs that engine a large share of its
    speed. At least one replica always stays in the batch lane,
    so documents keep moving even when every engine is serving someone.
    """

    lanes = {
        str(replica["name"]): LANE_PEOPLE if replica.get("node") in people_nodes else LANE_BATCH
        for replica in replicas
    }
    if lanes and LANE_BATCH not in lanes.values():
        # Keep the one already in the batch lane if there is one, so a replica
        # mid-document is not the one taken away.
        current = [str(r["name"]) for r in replicas if r.get("lane") == LANE_BATCH]
        keep = sorted(current)[0] if current else sorted(lanes)[0]
        lanes[keep] = LANE_BATCH
    return lanes


class Balancer:
    """Reads the fleet, decides, and writes caps and labels. Holds no truth."""

    def __init__(
        self,
        config: BalancerConfig,
        cluster: FleetView,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Retain configuration, a cluster view, and an HTTP client for sidecars."""

        self.config = config
        self.cluster = cluster
        self.client = client or httpx.AsyncClient(timeout=config.request_timeout_seconds)
        self.memory = _Memory()

    async def observe(self) -> list[EngineState]:
        """Read every engine's role from Kubernetes and occupancy from its sidecar."""

        pods = self.cluster.engines(self.config.namespace, self.config.engine_selector)
        crowding: dict[str, int] = {}
        for selector in self.config.neighbour_selectors:
            for neighbour in self.cluster.pods(self.config.namespace, selector):
                node = str(neighbour.get("node") or "")
                if node:
                    crowding[node] = crowding.get(node, 0) + 1
        states = await asyncio.gather(
            *(self._observe_one(pod, crowding) for pod in pods), return_exceptions=False
        )
        return list(states)

    async def _observe_one(
        self, pod: Mapping[str, Any], crowding: Mapping[str, int]
    ) -> EngineState:
        name = str(pod.get("name", ""))
        address = str(pod.get("address", ""))
        role = str(pod.get("role") or ROLE_BATCH)
        lane = str(pod.get("lane") or "")
        node = str(pod.get("node") or "")
        neighbours = crowding.get(node, 0)
        ready = bool(pod.get("ready"))
        if not ready or not address:
            return EngineState(
                name=name,
                address=address,
                ready=ready,
                role=role,
                lane=lane,
                node=node,
                neighbours=neighbours,
            )
        try:
            response = await self.client.get(
                f"http://{address}:{self.config.engine_port}{_ADMISSION_ROUTE}",
                headers=self._headers(),
            )
            response.raise_for_status()
            state = response.json()
        except Exception as error:  # noqa: BLE001 - an unreachable engine is data
            logger.warning("engine %s did not answer: %s", name, type(error).__name__)
            return EngineState(
                name=name,
                address=address,
                ready=ready,
                role=role,
                lane=lane,
                node=node,
                neighbours=neighbours,
                reachable=False,
            )
        return EngineState(
            name=name,
            address=address,
            ready=True,
            role=role,
            lane=lane,
            node=node,
            neighbours=neighbours,
            interactive_in_flight=int(state.get("protected_in_flight") or 0),
            batch_in_flight=int(state.get("held_in_flight") or 0),
            batch_cap=state.get("batch_cap"),
            cap_expires_in_seconds=state.get("cap_expires_in_seconds"),
        )

    async def apply(self, engines: Sequence[EngineState], decision: Decision) -> None:
        """Set the caps this pass decided, then make the labels agree.

        The cap is the enforcement and is written first: a machine must stop
        taking batch work before routing sends anyone to it. Labels follow so
        the pickers can prefer cleared machines. Both are re-asserted every
        pass, so a sidecar restart or a hand-edited label is corrected without
        anybody noticing.
        """

        by_name = {engine.name: engine for engine in engines}
        for name in (*decision.reserve, *decision.renew):
            await self._set_cap(by_name[name], 0)
        for name in decision.release:
            await self._set_cap(by_name[name], None)
        if decision.reserve or decision.release:
            logger.info(
                "standby %s/%s: reserved %s, released %s (%s)",
                decision.standby_now,
                decision.standby_target,
                list(decision.reserve) or "-",
                list(decision.release) or "-",
                decision.reason,
            )
        elif decision.renew:
            # Renewals are routine and frequent; at debug they stay out of the
            # way of the decisions an operator is actually reading for.
            logger.debug("standby renewed: %s", list(decision.renew))
        now = time.monotonic()
        people_nodes: set[str] = set()
        for engine in engines:
            if not engine.ready:
                continue
            wanted = role_for(engine, decision, self.memory, now)
            if wanted != engine.role:
                self.cluster.label(self.config.namespace, engine.name, ROLE_LABEL, wanted)
            lane = lane_for(engine, wanted, self.config, self.memory, now)
            if lane != engine.lane:
                self.cluster.label(self.config.namespace, engine.name, LANE_LABEL, lane)
            if lane == LANE_PEOPLE and engine.node:
                people_nodes.add(engine.node)
        if self.config.ocr_selector:
            replicas = self.cluster.pods(self.config.namespace, self.config.ocr_selector)
            current = {str(r["name"]): str(r.get("lane") or "") for r in replicas}
            for replica, wanted_lane in ocr_lanes(replicas, people_nodes).items():
                if wanted_lane != current.get(replica):
                    self.cluster.label(self.config.namespace, replica, LANE_LABEL, wanted_lane)

    async def _set_cap(self, engine: EngineState, cap: int | None) -> None:
        if not engine.reachable:
            return
        try:
            response = await self.client.post(
                f"http://{engine.address}:{self.config.engine_port}{_ADMISSION_ROUTE}",
                headers=self._headers(),
                json={"batch_cap": cap},
            )
            response.raise_for_status()
        except Exception as error:  # noqa: BLE001 - the cap lapses on its own
            logger.warning(
                "engine %s refused a cap of %s: %s", engine.name, cap, type(error).__name__
            )

    async def pass_once(self) -> Decision:
        """One observe-decide-apply cycle, as the loop and the tests both use it."""

        engines = await self.observe()
        decision = plan(engines, self.config, self.memory, time.monotonic())
        await self.apply(engines, decision)
        return decision

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Balance the fleet until asked to stop.

        A pass that raises is logged and retried on the next interval: this
        process failing closed would leave machines reserved, and the caps it
        already set expire by themselves, so the fleet degrades to ordinary
        behaviour rather than to no batch work at all.
        """

        logger.info(
            "balancing %s: standby=%s cooldown=%ss",
            self.config.engine_selector,
            self.config.standby,
            self.config.cooldown_seconds,
        )
        while stop is None or not stop.is_set():
            try:
                await self.pass_once()
            except Exception:
                logger.exception("balancer pass failed; retrying next interval")
            if stop is None:
                await asyncio.sleep(self.config.interval_seconds)
            else:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.config.interval_seconds)
                except TimeoutError:
                    continue

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}


class FleetView(Protocol):
    """What the balancer needs from the cluster, and nothing more.

    A protocol rather than a base class: the policy is exercised against a
    double in the tests, and the permissions the balancer asks for stay
    legible here - list engine pods, and set one label on them.
    """

    def engines(self, namespace: str, selector: str) -> list[dict[str, Any]]:
        """Return one record per engine pod: name, address, readiness, role."""
        ...

    def label(self, namespace: str, pod: str, key: str, value: str) -> None:
        """Set one label on one pod."""
        ...

    def pods(self, namespace: str, selector: str) -> list[dict[str, Any]]:
        """Return name, node and lane for every pod the selector matches."""
        ...


class KubectlFleetView:
    """The cluster as ``kubectl`` sees it."""

    def engines(self, namespace: str, selector: str) -> list[dict[str, Any]]:
        """Return one record per engine pod: name, address, readiness, role."""

        target = ClusterTarget()
        payload = kubectl_json(
            ["get", "pods", "-n", namespace, "-l", selector, "-o", "json"],
            target=target,
        )
        engines: list[dict[str, Any]] = []
        for item in payload.get("items", []):
            metadata = item.get("metadata", {})
            status = item.get("status", {})
            conditions = {c.get("type"): c.get("status") for c in status.get("conditions", [])}
            engines.append(
                {
                    "name": metadata.get("name", ""),
                    "address": status.get("podIP", ""),
                    "ready": conditions.get("Ready") == "True",
                    "role": (metadata.get("labels") or {}).get(ROLE_LABEL, ROLE_BATCH),
                    # Empty when absent, so a pod the balancer has not labelled
                    # yet is written on the first pass rather than assumed.
                    "lane": (metadata.get("labels") or {}).get(LANE_LABEL, ""),
                    "node": item.get("spec", {}).get("nodeName", ""),
                }
            )
        return engines

    def pods(self, namespace: str, selector: str) -> list[dict[str, Any]]:
        """Return name, node and lane for every running pod the selector matches."""

        payload = kubectl_json(
            ["get", "pods", "-n", namespace, "-l", selector, "-o", "json"],
            target=ClusterTarget(),
        )
        return [
            {
                "name": item.get("metadata", {}).get("name", ""),
                "node": item.get("spec", {}).get("nodeName", ""),
                "lane": (item.get("metadata", {}).get("labels") or {}).get(LANE_LABEL, ""),
            }
            for item in payload.get("items", [])
            if item.get("status", {}).get("phase") in ("Pending", "Running")
        ]

    def label(self, namespace: str, pod: str, key: str, value: str) -> None:
        """Set one label on one pod. Changing it restarts nothing."""

        kubectl_json(
            ["label", "pod", pod, "-n", namespace, f"{key}={value}", "--overwrite", "-o", "json"],
            target=ClusterTarget(),
        )


def run(config: BalancerConfig | None = None) -> None:
    """Run the balancer as a process."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    resolved = config or BalancerConfig.from_env()
    asyncio.run(Balancer(resolved, KubectlFleetView()).run())


def _read_secret(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def standby_names(engines: Iterable[EngineState]) -> list[str]:
    """The engines currently clear and ready, for logs and tests."""

    return sorted(engine.name for engine in engines if engine.batch_cap == 0)


_ADMISSION_ROUTE = "/flakegraph/admission"
