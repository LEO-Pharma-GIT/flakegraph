# SPDX-License-Identifier: Apache-2.0
"""Run independent provider calls concurrently and hand the results back in order."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor, as_completed
from itertools import islice

from kg_processor.application.window_errors import is_systemic_provider_error


def bounded_map[TItem, TResult](
    executor: Executor,
    operation: Callable[[TItem], TResult],
    items: Iterable[TItem],
    *,
    buffersize: int,
) -> Iterator[TResult]:
    """Stream ``operation`` over ``items`` with at most ``buffersize`` calls in flight.

    This is ``Executor.map(..., buffersize=)`` for interpreters older than 3.14,
    which is where that keyword appeared. The input is consumed lazily, so a
    listing of millions of objects never turns into a future per object, and
    results are yielded in input order because the planner relies on it. The
    next item is submitted before a result is handed back, so the pool stays
    busy while the consumer works.
    """

    if buffersize <= 0:
        raise ValueError("buffersize must be positive")
    iterator = iter(items)
    pending: deque[Future[TResult]] = deque(
        executor.submit(operation, item) for item in islice(iterator, buffersize)
    )
    try:
        while pending:
            result = pending.popleft().result()
            for item in islice(iterator, 1):
                pending.append(executor.submit(operation, item))
            yield result
    finally:
        # Closing the generator early, or a failed call, must not leave queued
        # work running against a source nobody is reading any more.
        for future in pending:
            future.cancel()


def ordered_map[TItem, TResult](
    items: Sequence[TItem],
    operation: Callable[[int, TItem], TResult],
    *,
    parallelism: int,
    on_complete: Callable[[int, TResult], None] | None = None,
    contain: Callable[[int, TItem, Exception], TResult] | None = None,
) -> list[TResult]:
    """Apply ``operation`` to every item and return the results in input order.

    Provider latency decides completion order, so ``on_complete`` fires as calls
    finish (with the running count) while the returned list follows the input;
    persisted traces and merge order therefore never depend on scheduling.

    ``contain`` turns one failed call into a substitute result so the rest of
    the batch is kept. It is never offered a systemic failure (authentication,
    configuration): that ends the whole map, and the calls not yet started are
    cancelled so no more provider capacity is spent on a run that cannot
    recover. Running calls keep their own provider timeout.

    A pool of one worker runs submissions in order, so there is no separate
    serial path.
    """

    if parallelism <= 0:
        raise ValueError("parallelism must be positive")
    if not items:
        return []
    results: dict[int, TResult] = {}
    executor = ThreadPoolExecutor(max_workers=min(parallelism, len(items)))
    try:
        futures = {
            executor.submit(operation, index, item): index for index, item in enumerate(items)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                if contain is None or is_systemic_provider_error(exc):
                    raise
                result = contain(index, items[index], exc)
            results[index] = result
            if on_complete is not None:
                on_complete(completed, result)
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return [results[index] for index in range(len(items))]
