# SPDX-License-Identifier: Apache-2.0
"""Authentication provider port.

Auth providers return connection-layer credentials only. Session context such
as database, schema, role, and warehouse stays with the concrete connector
config so credential strategy and runtime target do not become tangled.
"""

from __future__ import annotations

from typing import Protocol


class AuthProvider(Protocol):
    """Returns connector credentials without owning Snowflake session context."""

    def connection_kwargs(self) -> dict[str, object]:
        """Return keyword arguments suitable for the concrete connection client."""
        ...
