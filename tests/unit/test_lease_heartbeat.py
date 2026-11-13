from __future__ import annotations

import time

import pytest

from kg_processor.application.lease_heartbeat import LeaseHeartbeat, heartbeat_interval_seconds


def test_lease_heartbeat_retries_after_transient_failure() -> None:
    calls = 0

    def heartbeat() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary Snowflake outage")

    with LeaseHeartbeat(heartbeat, interval_seconds=0.01) as heartbeat_runner:
        deadline = time.monotonic() + 1.0
        while calls < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

    assert calls >= 2
    assert heartbeat_runner.failed_heartbeats == 1
    assert isinstance(heartbeat_runner.last_error, RuntimeError)
    assert heartbeat_runner.consecutive_failures == 0


def test_lease_heartbeat_surfaces_repeated_ownership_failures() -> None:
    """Prevent a worker from completing after repeated lease refresh failures."""

    heartbeat_runner = LeaseHeartbeat(lambda: None, interval_seconds=1)
    heartbeat_runner.consecutive_failures = 3
    heartbeat_runner.last_error = RuntimeError("lease lost")

    with pytest.raises(RuntimeError, match="3 consecutive"):
        heartbeat_runner.raise_if_unhealthy()


def test_heartbeat_interval_uses_one_third_of_lease_with_bounds() -> None:
    assert heartbeat_interval_seconds(3) == 1
    assert heartbeat_interval_seconds(90) == 30
    assert heartbeat_interval_seconds(900) == 60


def test_heartbeat_interval_rejects_non_positive_lease() -> None:
    with pytest.raises(ValueError, match="lease_seconds"):
        heartbeat_interval_seconds(0)
