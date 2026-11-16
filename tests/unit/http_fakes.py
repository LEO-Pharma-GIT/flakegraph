"""Stand-ins for the persistent ``httpx.Client`` every HTTP adapter constructs.

An adapter builds one ``httpx.Client()`` in its constructor and keeps it for
its whole life, so a test hands the adapter one of these through
``monkeypatch.setattr(httpx, "Client", client.open)``. State lives on the
instance, so nothing leaks from one test into the next.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from kg_processor.domain.documents import InputFile

Handler = Callable[[str, dict[str, str], dict[str, Any]], httpx.Response]


class _Client:
    opened = 0

    def open(self) -> _Client:
        """Stand in for ``httpx.Client()``: hand the adapter this one client."""

        self.opened += 1
        return self

    def close(self) -> None:
        """Release nothing; the adapter's ``close`` still has to reach here."""


class ScriptedClient(_Client):
    """Answer ``post`` with a handler's response and record every request.

    The handler receives the URL, headers and JSON body; the response it
    returns is bound to a request so ``raise_for_status`` can name the URL.
    """

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    @property
    def payloads(self) -> list[dict[str, Any]]:
        """Return the JSON body of every request, in order."""

        return [payload for _, _, payload in self.requests]

    def post(
        self,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float | None = None,
    ) -> httpx.Response:
        self.requests.append((url, headers, json))
        response = self.handler(url, headers, json)
        response.request = httpx.Request("POST", url, headers=headers)
        return response


class StreamingClient(_Client):
    """Serve one scripted body through ``stream`` and record the upload.

    ``content`` wins over ``payload`` so a test can send bytes that are not
    JSON; ``response_headers`` lets it declare a size the body does not have.
    """

    def __init__(
        self,
        payload: dict[str, object] | None = None,
        *,
        status_code: int = 200,
        content: bytes | None = None,
        response_headers: dict[str, str] | None = None,
        payloads: list[dict[str, object]] | None = None,
    ) -> None:
        self.payload = payload or {}
        # ``payloads`` answers one request each, in order, for an adapter that
        # asks again; the last one repeats once the script runs out.
        self.payloads = list(payloads or [])
        self.status_code = status_code
        self.content = content
        self.response_headers = response_headers or {}
        self.requests: list[dict[str, Any]] = []

    @contextmanager
    def stream(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: dict[str, str],
        files: dict[str, Any],
        timeout: float | None = None,
    ) -> Iterator[httpx.Response]:
        (upload,) = files.values()
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "data": data,
                "files": files,
                # The uploaded handle must still be open while the body streams.
                "file_handle_open": not upload[1].closed,
            }
        )
        payload = self.payload
        if self.payloads:
            payload = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        body = json.dumps(payload).encode("utf-8") if self.content is None else self.content
        response = httpx.Response(
            self.status_code,
            content=body,
            headers=self.response_headers,
            request=httpx.Request(method, url, headers=headers),
        )
        try:
            yield response
        finally:
            response.close()


def input_file(path: Path) -> InputFile:
    """Describe one PDF on disk the way the file source would."""

    return InputFile(
        id="file_1",
        path=path,
        source_uri=str(path),
        checksum="checksum",
        mime_type="application/pdf",
        size_bytes=path.stat().st_size,
    )
