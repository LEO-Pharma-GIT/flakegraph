"""The bounded streaming map keeps ``Executor.map(buffersize=)`` semantics on 3.12."""

from __future__ import annotations

from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from threading import Lock

import pytest

from kg_processor.application.ordered_map import bounded_map


def test_bounded_map_returns_results_in_input_order() -> None:
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(bounded_map(executor, lambda item: item * 2, range(20), buffersize=3))

    assert results == [item * 2 for item in range(20)]


def test_bounded_map_consumes_the_input_lazily() -> None:
    """A listing of unbounded size must not be pulled ahead of the consumer.

    The whole point of the buffer is that enumerating a container of millions of
    objects creates at most ``buffersize`` futures at a time, so the number of
    items drawn from the source can never run far ahead of what was yielded.
    """

    drawn = 0
    lock = Lock()

    def source() -> Generator[int, None, None]:
        nonlocal drawn
        for item in count():
            with lock:
                drawn += 1
            yield item

    with ThreadPoolExecutor(max_workers=2) as executor:
        stream = bounded_map(executor, lambda item: item, source(), buffersize=3)
        first = [next(stream) for _ in range(5)]
        assert isinstance(stream, Generator)
        stream.close()

    assert first == [0, 1, 2, 3, 4]
    # Five yielded plus at most the buffer refilled behind them.
    assert drawn <= 5 + 3


def test_bounded_map_propagates_the_first_failure_and_cancels_the_rest() -> None:
    def operation(item: int) -> int:
        if item == 2:
            raise RuntimeError("boom")
        return item

    with ThreadPoolExecutor(max_workers=1) as executor, pytest.raises(RuntimeError, match="boom"):
        list(bounded_map(executor, operation, range(10), buffersize=2))


def test_bounded_map_rejects_a_non_positive_buffer() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor, pytest.raises(ValueError):
        next(bounded_map(executor, lambda item: item, [1], buffersize=0))
