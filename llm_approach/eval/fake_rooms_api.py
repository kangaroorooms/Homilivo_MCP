"""A fake failing Rooms API for the offline runners, for both transports (no network).

The direct client calls ``rooms_client._httpx_get``; the copied MCP tool calls ``urllib.request.urlopen``. Both are
patched, so a scenario behaves the same whichever ``LLM_APPROACH_SEARCH_TRANSPORT`` is set. The tool's retry backoff is
set to 0 so a failing scenario costs no real time (its three attempts still happen).
"""

from __future__ import annotations

import contextlib
import urllib.error
from typing import Any, Callable, Iterator
from unittest import mock

import httpx

from llm_approach import rooms_client


class FakeResponse:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code, self._body = status, body

    def json(self) -> dict:
        return self._body


@contextlib.contextmanager
def failing_rooms_api(status: int | None = None, body: dict | None = None) -> Iterator[None]:
    """``status`` set: every call answers that HTTP status. ``status`` None: every call times out."""
    from llm_approach.tools import mcp_search  # lazy: only the offline runners need the mcp package

    def httpx_get(*args: Any, **kwargs: Any) -> Any:
        if status is None:
            raise httpx.ReadTimeout("read timed out")
        return FakeResponse(status, body if body is not None else {"success": False, "message": "internal error"})

    def urlopen(request: Any, timeout: Any = None) -> Any:
        if status is None:
            raise TimeoutError("timed out")
        raise urllib.error.HTTPError(request.full_url, status, "error", {}, None)  # type: ignore[arg-type]

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(rooms_client, "_httpx_get", httpx_get))
        stack.enter_context(mock.patch("urllib.request.urlopen", urlopen))
        stack.enter_context(mock.patch.object(mcp_search, "HOMILIVO_API_RETRY_BACKOFF_SECONDS", 0.0))
        yield
