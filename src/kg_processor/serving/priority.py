# SPDX-License-Identifier: Apache-2.0
"""Map a bearer key to a consumer class, and a class to an engine priority.

Priority is a request field stamped at a trusted point, never a routing concept.
This module owns the two lookups that make that true, and it is shared by the
inference sidecar and the OCR shim so a key means the same thing on both planes.

vLLM serves the *lower* value first, so its default of ``0`` is the *highest*
priority. Anything that fails to stamp therefore promotes a request rather than
demoting it, which is why callers must stamp unconditionally and why an
unrecognised class resolves to the numerically largest declared band.
"""

from __future__ import annotations

import json
from hmac import compare_digest
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

INTERACTIVE_CLASS = "interactive"
DEV_CLASS = "dev"
BATCH_CLASS = "batch"

DEFAULT_PRIORITY_BANDS: dict[str, int] = {
    INTERACTIVE_CLASS: 0,
    DEV_CLASS: 10,
    BATCH_CLASS: 100,
}


def presented_key(authorization: str) -> str:
    """Extract a bearer credential without treating a malformed header as valid."""

    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return credential.strip()


class ConsumerKeyring(BaseModel):
    """Resolve presented credentials to a consumer class and its priority band."""

    bands: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_PRIORITY_BANDS))
    keys: dict[str, str] = Field(default_factory=dict)

    @field_validator("bands")
    @classmethod
    def bands_must_not_be_empty(cls, value: dict[str, int]) -> dict[str, int]:
        """Refuse a configuration that leaves no band to stamp."""

        if not value:
            raise ValueError("at least one priority band must be declared")
        return value

    @field_validator("keys")
    @classmethod
    def keys_must_not_be_blank(cls, value: dict[str, str]) -> dict[str, str]:
        """Reject blank credentials, which would authenticate an empty header."""

        for key, consumer_class in value.items():
            if not key.strip():
                raise ValueError("consumer keys must not be blank")
            if not consumer_class.strip():
                raise ValueError(f"consumer key for '{consumer_class}' must name a class")
        return value

    @property
    def lowest_priority(self) -> int:
        """Return the band that is served last among those declared."""

        return max(self.bands.values())

    def classify(self, presented_key: str) -> str | None:
        """Return the consumer class for a presented key, or ``None`` if unknown.

        Every configured key is compared, and each comparison is constant-time, so
        neither the number of comparisons nor their duration reveals how much of a
        candidate key was correct. The keyring holds a handful of entries, so the
        linear scan costs nothing that matters.
        """

        if not presented_key:
            return None
        matched: str | None = None
        for key, consumer_class in self.keys.items():
            if compare_digest(key, presented_key):
                matched = consumer_class
        return matched

    def priority_for(self, consumer_class: str | None) -> int:
        """Return the band to stamp, defaulting an unknown class to served-last."""

        if consumer_class is None:
            return self.lowest_priority
        return self.bands.get(consumer_class, self.lowest_priority)


def load_keyring(keys_file: Path, bands: dict[str, int] | None = None) -> ConsumerKeyring:
    """Load the key-to-class mapping a Secret projects into the pod.

    The file is read once at startup. Rotating a key is a pod restart, which is
    what an operator already does to roll a Secret, and it keeps an unreadable or
    malformed file from being discovered mid-request.
    """

    if not keys_file.is_file():
        raise ValueError(f"consumer key file not found: {keys_file}")
    try:
        payload = json.loads(keys_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"consumer key file is not valid JSON: {keys_file}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"consumer key file must be a JSON object: {keys_file}")
    keys = {str(key): str(value) for key, value in payload.items()}
    if not keys:
        raise ValueError(f"consumer key file declares no keys: {keys_file}")
    return ConsumerKeyring(bands=dict(bands or DEFAULT_PRIORITY_BANDS), keys=keys)
