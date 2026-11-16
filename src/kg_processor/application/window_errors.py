# SPDX-License-Identifier: Apache-2.0
"""Classify provider failures for independent extraction-window containment."""

from __future__ import annotations

_SYSTEMIC_ERROR_MARKERS = (
    "api key",
    "authentication",
    "authorization",
    "credential",
    "forbidden",
    "permission denied",
    "unauthorized",
)
_SYSTEMIC_CLASS_MARKERS = ("authentication", "authorization", "credential", "permission")
_SYSTEMIC_HTTP_STATUS_CODES = {401, 403}


def is_systemic_provider_error(exc: Exception) -> bool:
    """Return whether one window failure invalidates the provider configuration.

    Transport, request-shape, content-policy, throttling, and server failures may
    affect one bounded window and can be audited locally. Authentication and
    permission failures apply to every window and must fail the run.
    """

    if isinstance(exc, OSError):
        return False
    class_name = type(exc).__name__.casefold()
    if any(marker in class_name for marker in _SYSTEMIC_CLASS_MARKERS):
        return True
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in _SYSTEMIC_HTTP_STATUS_CODES:
        return True
    message = str(exc).casefold()
    return any(marker in message for marker in _SYSTEMIC_ERROR_MARKERS)
