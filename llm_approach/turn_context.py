"""Who a log line belongs to: the current turn's request id and hashed session id.

ContextVars, so the values also reach ``asyncio.to_thread`` workers. Only a request id and a short
one-way hash of the session id are kept: never user text, never the raw session id.

The server calls ``set_requested`` before a turn so the pipeline's record and the reply carry the
same request id and mode; ``pipeline.handle_turn`` calls ``begin`` / ``end`` around the turn.
"""

from __future__ import annotations

import hashlib
from contextvars import ContextVar, Token
from typing import Any

_requested_id: ContextVar[str | None] = ContextVar("llm_approach_requested_id", default=None)
_requested_mode: ContextVar[str | None] = ContextVar("llm_approach_requested_mode", default=None)
_request_id: ContextVar[str | None] = ContextVar("llm_approach_request_id", default=None)
_session_hash: ContextVar[str | None] = ContextVar("llm_approach_session_hash", default=None)


def hash_session_id(session_id: str) -> str:
    """12 hex characters of SHA-256: stable for correlating one session's log lines, not reversible."""
    return hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:12]


def set_requested(request_id: str, mode: str) -> None:
    """Server side: the id and mode the next ``handle_turn`` in this task should use."""
    _requested_id.set(request_id)
    _requested_mode.set(mode)


def requested_request_id() -> str | None:
    return _requested_id.get()


def requested_mode() -> str | None:
    return _requested_mode.get()


def begin(request_id: str, session_id: str) -> list[Token[Any]]:
    return [_request_id.set(request_id), _session_hash.set(hash_session_id(session_id))]


def end(tokens: list[Token[Any]]) -> None:
    _request_id.reset(tokens[0])
    _session_hash.reset(tokens[1])


def request_id() -> str | None:
    return _request_id.get()


def session_hash() -> str | None:
    return _session_hash.get()
