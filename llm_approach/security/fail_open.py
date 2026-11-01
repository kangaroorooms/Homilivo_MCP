"""Fail-open wrappers for the Redis-backed rate limiter and demo-token store.

A slow or down Redis must not take the chat down with it. The rate limiter allows the request when Redis
times out or raises; the demo-token store falls back to this process's memory. Either way one fixed-word
counter is bumped and one log line names the exception class: never a key, a token or user text.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_approach import config, observability

logger = logging.getLogger(__name__)

RATE_LIMIT_EVENT = "rate_limit_redis_failopen"
DEMO_TOKEN_EVENT = "demo_token_redis_failopen"


def redis_timeout_s() -> float:
    """Connect and read timeout for the two Redis clients (same setting as the session store)."""
    return config.redis_timeout_s()


def _note(event: str, exc: BaseException) -> None:
    logger.warning("%s: %s", event, type(exc).__name__)
    try:
        observability.record_session_store_event(event)
    except Exception:  # noqa: BLE001
        pass


class FailOpenRateLimiter:
    """``hit`` that allows the request when the wrapped limiter raises (Redis timeout, connection lost, ...)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def hit(self, key: str, limit: int, window_seconds: float, now: float | None = None):
        try:
            return self._inner.hit(key, limit, window_seconds, now)
        except Exception as exc:  # noqa: BLE001
            _note(RATE_LIMIT_EVENT, exc)
            from llm_approach.security.rate_limit import RateLimitResult  # lazy: rate_limit imports this module

            return RateLimitResult(allowed=True, remaining=-1, retry_after_seconds=0.0)


class FailOpenDemoAuthBackend:
    """Demo-token backend that uses ``fallback`` (the in-memory backend) for a call Redis could not answer."""

    def __init__(self, inner: Any, fallback: Any) -> None:
        self._inner, self._fallback = inner, fallback

    def issue(self, token: str, now: float) -> None:
        try:
            self._inner.issue(token, now)
        except Exception as exc:  # noqa: BLE001
            _note(DEMO_TOKEN_EVENT, exc)
            self._fallback.issue(token, now)

    def get_issued_at(self, token: str) -> float | None:
        try:
            found = self._inner.get_issued_at(token)
        except Exception as exc:  # noqa: BLE001
            _note(DEMO_TOKEN_EVENT, exc)
            return self._fallback.get_issued_at(token)
        return found if found is not None else self._fallback.get_issued_at(token)

    def delete(self, token: str) -> None:
        self._fallback.delete(token)
        try:
            self._inner.delete(token)
        except Exception as exc:  # noqa: BLE001
            _note(DEMO_TOKEN_EVENT, exc)

