# SPDX-License-Identifier: Apache-2.0
"""Decide whether a directory may be replaced by a FlakeGraph artifact snapshot.

Every writer that publishes a snapshot by replacing a directory shares one
question: is what is there now ours to delete? A typo in ``output_path`` must
not remove arbitrary user data. The answer is the same for the local artifacts
writer and for a Snowflake export, so it lives here rather than in either
adapter.
"""

from __future__ import annotations

import json
from pathlib import Path

ARTIFACT_MANIFEST = ".flakegraph-artifacts.json"
ARTIFACT_FORMAT = "flakegraph-local-artifacts-v1"


def write_ownership_manifest(output_path: Path) -> None:
    """Mark a snapshot directory as FlakeGraph-owned so a later run may replace it."""

    (output_path / ARTIFACT_MANIFEST).write_text(
        json.dumps({"format": ARTIFACT_FORMAT}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def assert_replaceable_output(output_path: Path) -> None:
    """Refuse replacement unless an existing destination is empty or app-owned.

    Empty destinations are safe to claim; a non-empty one must carry the
    ownership manifest every snapshot writes.
    """

    if not output_path.exists():
        return
    if not output_path.is_dir():
        raise ValueError(f"Refusing to replace non-directory output path: {output_path}")
    if not any(output_path.iterdir()):
        return
    manifest_path = output_path / ARTIFACT_MANIFEST
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, dict) and manifest.get("format") == ARTIFACT_FORMAT:
            return
    raise ValueError(f"Refusing to replace unrelated non-empty output directory: {output_path}")
