"""One wall-clock deadline for a whole turn (the server's, not the pipeline's: the CLI and the eval runners have none).

``run(awaitable, seconds)`` runs the awaitable as a task. If it is not done in ``seconds`` it is cancelled, the
cancellation is awaited (so the turn's own clean-up runs: lock release, save) and ``DeadlineExceeded`` is raised.
While the task runs ``expired()`` is False; once the deadline fires it is True inside the cancelled task, which is how
the pipeline tells a deadline from any other cancellation. ``sleep`` is injectable so tests need no real waiting.

What a cancel can and cannot stop: awaits are cancelled at once; work already handed to a thread keeps running, which is
why the pipeline gives every state-touching thread a private copy of the session state (see ``pipeline._threaded_on_copy``).
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


class DeadlineExceeded(Exception):
    """The turn did not finish inside its deadline and was cancelled."""


@dataclass
class _Deadline:
    expired: bool = False


_current: ContextVar[_Deadline | None] = ContextVar("llm_approach_turn_deadline", default=None)


def expired() -> bool:
    """Whether the deadline of the turn this code runs in has fired (False outside a deadline)."""
    deadline = _current.get()
    return bool(deadline and deadline.expired)


async def run(awaitable: Awaitable[Any], seconds: float, *, sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> Any:
    deadline = _Deadline()
    token = _current.set(deadline)
    try:
        task = asyncio.ensure_future(awaitable)  # copies the context: the task sees ``deadline``
    finally:
        _current.reset(token)
    timer = asyncio.ensure_future(sleep(seconds))
    try:
        await asyncio.wait({task, timer}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            return task.result()
        deadline.expired = True
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)  # the turn's clean-up finishes before we answer
        if not task.cancelled() and task.exception() is None:  # it finished in the instant before the cancel landed
            return task.result()
        raise DeadlineExceeded
    finally:
        timer.cancel()
        if not task.done():  # the caller itself was cancelled
            task.cancel()
