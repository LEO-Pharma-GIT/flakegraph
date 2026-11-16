"""What the serving front doors share: a keyring, a fake engine and a scrape reader."""

from __future__ import annotations

import json

import httpx
from prometheus_client.parser import text_string_to_metric_families

from kg_processor.serving.priority import ConsumerKeyring

KEYRING = ConsumerKeyring(
    bands={"interactive": 0, "dev": 10, "batch": 100},
    keys={"sk-chat": "interactive", "sk-tool": "dev", "sk-pipeline": "batch"},
)


def stub(body: bytes, content_type: str = "application/json") -> httpx.Response:
    """Build an unread response so a front door can relay it as a raw stream.

    ``httpx.Response`` reads eager content during construction, which would leave
    nothing for ``aiter_raw`` to iterate. A real transport always hands back an
    unconsumed stream, so the double has to as well.
    """

    return httpx.Response(
        200,
        headers={"content-type": content_type, "content-length": str(len(body))},
        stream=httpx.ByteStream(body),
    )


class RecordingUpstream:
    """Records what the engine would have received and replies with a stub."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/metrics":
            return stub(b"vllm:num_preemptions_total 0", "text/plain")
        return stub(b'{"ok": true}')

    @property
    def last_body(self) -> dict[str, object]:
        payload: dict[str, object] = json.loads(self.requests[-1].content)
        return payload


def samples(exposition: str, name: str) -> dict[tuple[tuple[str, str], ...], float]:
    """Return every sample of one series, keyed by its sorted label pairs."""

    return {
        tuple(sorted(sample.labels.items())): float(sample.value)
        for family in text_string_to_metric_families(exposition)
        for sample in family.samples
        if sample.name == name
    }
