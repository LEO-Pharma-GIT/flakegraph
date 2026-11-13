# SPDX-License-Identifier: Apache-2.0
"""Name the text windows whose every extracted record was rejected.

Both engines derive these from the extraction trace so one corpus reports
the same gaps wherever it was processed. The local pipeline calls this on
its in-process trace; the Spark finalizer expresses the same rule over its
trace column and joins the chunk table the same way.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from kg_processor.domain.graph import Chunk, DiscardedWindow
from kg_processor.domain.ids import stable_id

# The passes whose windows are worth naming: a context or verification call
# that keeps nothing removes no text from the graph.
DISCARDED_WINDOW_STAGES: dict[str, str] = {
    "entity_extraction": "entities",
    "relation_extraction": "relations",
}
# Characters of the window's text a row carries, enough to recognise a page.
PREVIEW_CHARACTERS = 240


def is_discarded_window_event(event: Mapping[str, Any]) -> bool:
    """A window the model returned records for and validation kept none of."""

    if str(event.get("stage") or "") not in DISCARDED_WINDOW_STAGES:
        return False
    extracted = int(event.get("input_records") or 0)
    accepted = int(event.get("accepted_records") or 0)
    return extracted > 0 and accepted == 0


def discarded_windows_from_trace(
    trace: Iterable[Mapping[str, Any]],
    chunks: Iterable[Chunk],
    graph_id: str,
) -> list[DiscardedWindow]:
    """Turn the trace's fully rejected windows into rows a reader can act on."""

    by_id = {chunk.id: chunk for chunk in chunks}
    rows: list[DiscardedWindow] = []
    for event in trace:
        if not is_discarded_window_event(event):
            continue
        stage = DISCARDED_WINDOW_STAGES[str(event["stage"])]
        window_id = str(event.get("window_id") or "")
        chunk_ids = [str(item) for item in event["chunk_ids"]]
        members = sorted(
            (by_id[item] for item in chunk_ids if item in by_id),
            key=lambda chunk: (chunk.page_number, chunk.chunk_index),
        )
        pages = [chunk.page_number for chunk in members]
        document_id = str(event["document_id"])
        actions = event.get("record_actions") or {}
        rows.append(
            DiscardedWindow(
                id=stable_id("discarded_window", graph_id, stage, window_id),
                graph_id=graph_id,
                document_id=document_id,
                file_id=members[0].file_id if members else document_id,
                stage=stage,
                window_id=window_id,
                chunk_ids=chunk_ids,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                extracted_records=int(event.get("input_records") or 0),
                record_actions={
                    str(name): int(count)
                    for name, count in (actions.items() if isinstance(actions, Mapping) else [])
                },
                preview=window_preview(chunk.content for chunk in members),
            )
        )
    # One row per window pass identity, as the Spark finalizer keeps, however
    # many times the trace carries the event.
    rows = list({row.id: row for row in reversed(rows)}.values())
    rows.sort(key=lambda row: (row.document_id, row.page_start or 0, row.stage, row.window_id))
    return rows


def window_preview(texts: Iterable[str]) -> str:
    """The first characters of a window's text, whitespace folded."""

    joined = " ".join(" ".join(text.split()) for text in texts if text and text.strip())
    if len(joined) <= PREVIEW_CHARACTERS:
        return joined
    return joined[: PREVIEW_CHARACTERS - 1].rstrip() + "…"


def discarded_window_metrics(rows: Iterable[DiscardedWindow]) -> dict[str, Any]:
    """The summary a run report carries: how many windows, documents and why."""

    windows = 0
    documents: set[str] = set()
    by_stage: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for row in rows:
        windows += 1
        documents.add(row.document_id)
        by_stage[row.stage] = by_stage.get(row.stage, 0) + 1
        for name, count in row.record_actions.items():
            reasons[name] = reasons.get(name, 0) + count
    return {
        "windows": windows,
        "documents": len(documents),
        "by_stage": dict(sorted(by_stage.items())),
        "record_actions": dict(sorted(reasons.items())),
    }
