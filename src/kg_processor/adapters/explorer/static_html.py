# SPDX-License-Identifier: Apache-2.0
"""Render a graph explorer dataset as one self-contained offline HTML file."""

from __future__ import annotations

import html
import json
import re
import uuid
from base64 import b64encode
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Any

from plotly.offline import get_plotlyjs

from kg_processor.application.graph_explorer import GraphExplorerDataset

_ASSET_DIRECTORY = "explorer_assets"
_UNRESOLVED_TOKEN = re.compile(r"__FLAKEGRAPH_[A-Z_]+__")
_MAX_INLINE_PAYLOAD_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class GraphExplorerExportResult:
    """Metadata returned after a static graph explorer is written."""

    html_path: Path
    size_bytes: int
    graph_id: str
    counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready CLI result without embedding graph data."""

        return {
            "html_path": str(self.html_path),
            "size_bytes": self.size_bytes,
            "self_contained": True,
            "graph_id": self.graph_id,
            "counts": self.counts,
        }


class StaticHtmlGraphExplorer:
    """Inline packaged assets, Plotly.js, and graph data into an offline file."""

    def write(
        self,
        dataset: GraphExplorerDataset,
        html_path: Path,
        *,
        title: str = "FlakeGraph Explorer",
        max_payload_bytes: int = _MAX_INLINE_PAYLOAD_BYTES,
    ) -> GraphExplorerExportResult:
        """Write the explorer atomically and return its concise export metadata.

        Static HTML necessarily embeds the complete review dataset in browser
        memory. A configurable upper bound fails clearly before constructing an
        unusable multi-hundred-megabyte document.
        """

        template = _read_asset("graph_explorer.html")
        _validate_template(template)
        encoded_payload = _safe_json_script(dataset.payload)
        payload_size = len(encoded_payload.encode("utf-8"))
        if max_payload_bytes <= 0:
            raise ValueError("graph explorer max payload bytes must be positive")
        if payload_size > max_payload_bytes:
            raise ValueError(
                "Graph explorer payload exceeds the configured inline limit "
                f"({payload_size} > {max_payload_bytes} bytes)"
            )
        plotly_javascript = get_plotlyjs()
        app_javascript = _read_asset("graph_explorer.js")
        script_hashes = " ".join(
            f"'sha256-{b64encode(sha256(script.encode()).digest()).decode()}'"
            for script in (plotly_javascript, app_javascript)
        )
        replacements = {
            "__FLAKEGRAPH_TITLE__": html.escape(title),
            "__FLAKEGRAPH_CSS__": _read_asset("graph_explorer.css"),
            "__FLAKEGRAPH_PLOTLY_JS__": plotly_javascript,
            "__FLAKEGRAPH_DATA__": encoded_payload,
            "__FLAKEGRAPH_APP_JS__": app_javascript,
            "__FLAKEGRAPH_SCRIPT_HASHES__": script_hashes,
        }
        # Substitute against the original template in one pass. A sequence of
        # ``str.replace`` calls would interpret template-like text inside an
        # already-inserted document or title as another real placeholder.
        token_pattern = re.compile("|".join(re.escape(token) for token in replacements))
        rendered = token_pattern.sub(lambda match: replacements[match.group(0)], template)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = html_path.with_name(f".{html_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary_path.write_text(rendered, encoding="utf-8")
            temporary_path.replace(html_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return GraphExplorerExportResult(
            html_path=html_path,
            size_bytes=html_path.stat().st_size,
            graph_id=dataset.graph_id,
            counts=dataset.counts,
        )


def _read_asset(name: str) -> str:
    resource = files("kg_processor").joinpath(_ASSET_DIRECTORY, name)
    return resource.read_text(encoding="utf-8")


def _validate_template(template: str) -> None:
    """Validate packaged placeholders before any untrusted graph text is inserted."""

    required = {
        "__FLAKEGRAPH_TITLE__",
        "__FLAKEGRAPH_CSS__",
        "__FLAKEGRAPH_PLOTLY_JS__",
        "__FLAKEGRAPH_DATA__",
        "__FLAKEGRAPH_APP_JS__",
        "__FLAKEGRAPH_SCRIPT_HASHES__",
    }
    discovered = set(_UNRESOLVED_TOKEN.findall(template))
    if discovered != required:
        raise RuntimeError(
            "Graph explorer template placeholders do not match the packaged contract"
        )
    if any(template.count(token) != 1 for token in required):
        raise RuntimeError("Graph explorer template placeholders must occur exactly once")


def _safe_json_script(payload: dict[str, Any]) -> str:
    # Script tags can be terminated by source text containing </script>. Encode
    # HTML-significant characters while retaining valid JSON and exact text once
    # the browser parses the application/json element.
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
