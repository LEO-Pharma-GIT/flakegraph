# SPDX-License-Identifier: Apache-2.0
"""Background lease heartbeats for long-running claimed work.

Snowflake leases make crash recovery possible, but long OCR/LLM batches need a
separate heartbeat so another worker does not reclaim an actively processed
batch just because it exceeded the original lease window.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

# One blip is recoverable; three in a row means the lease is not ours to keep.
_MAX_CONSECUTIVE_FAILURES = 3


class LeaseHeartbeat:
    """Context manager that heartbeats a leased job on a daemon thread."""

    def __init__(
        self,
        heartbeat: Callable[[], None],
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.heartbeat = heartbeat
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: Exception | None = None
        self.failed_heartbeats = 0
        self.consecutive_failures = 0

    def __enter__(self) -> LeaseHeartbeat:
        self._thread = threading.Thread(
            target=self._run,
            name="flakegraph-lease-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.heartbeat()
                self.consecutive_failures = 0
            except Exception as exc:  # pragma: no cover - thread timing guard
                # Lease refresh failures are often transient network or
                # Snowflake availability blips. Keep the worker alive and keep
                # retrying so a short heartbeat issue does not hide the actual
                # pipeline result at context-manager exit.
                self.last_error = exc
                self.failed_heartbeats += 1
                self.consecutive_failures += 1

    def raise_if_unhealthy(self) -> None:
        """Stop completion after repeated lease-refresh failures.

        A single network blip remains recoverable. Repeated failures mean the
        worker can no longer establish ownership and must not publish potentially
        duplicated results as if its lease were healthy.
        """

        if self.consecutive_failures < _MAX_CONSECUTIVE_FAILURES:
            return
        raise RuntimeError(
            f"lease heartbeat failed {self.consecutive_failures} consecutive times"
        ) from self.last_error


def heartbeat_interval_seconds(lease_seconds: int) -> int:
    """Choose a heartbeat interval that is frequent but bounded."""

    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    # Heartbeat at one third of the lease, capped to a minute so long leases
    # still produce useful liveness signals, and floored at one second for tests
    # and intentionally short leases.
    return max(1, min(60, lease_seconds // 3))
