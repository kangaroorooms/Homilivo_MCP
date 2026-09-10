"""
Lightweight request rate limiting.

RateLimiterBackend is the storage abstraction: InMemoryRateLimiter is the
only implementation today (fine for a single-process demo deployment),
but any backend that implements `hit()` the same way - e.g. a Redis
INCR+EXPIRE based one - can replace it later without touching the
FastAPI wiring in app/api.py.

Algorithm: fixed window counter per key. Simple, cheap, and sufficient
for abuse protection on a demo service; it does not need the precision
of a sliding-window log.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after_seconds: float


class RateLimiterBackend(Protocol):
    def hit(
        self, key: str, limit: int, window_seconds: float, now: float
    ) -> RateLimitResult:
        ...


class InMemoryRateLimiter:
    """
    Per-process fixed-window counter, keyed by an arbitrary string
    (IP address, session id, etc.). Thread-safe.

    Stale windows are purged opportunistically on each call so the
    dict does not grow unbounded across the lifetime of a long-running
    server process.
    """

    def __init__(self, stale_after_seconds: float = 3600.0) -> None:
        self._windows: dict[str, list[float]] = {}  # key -> [window_start, count]
        self._lock = threading.Lock()
        self._stale_after_seconds = stale_after_seconds

    def _purge_stale(self, now: float) -> None:
        stale_keys = [
            key
            for key, (window_start, _count) in self._windows.items()
            if now - window_start > self._stale_after_seconds
        ]
        for key in stale_keys:
            del self._windows[key]

    def hit(
        self, key: str, limit: int, window_seconds: float, now: float | None = None
    ) -> RateLimitResult:
        now = time.time() if now is None else now

        with self._lock:
            if len(self._windows) > 10_000:
                self._purge_stale(now)

            window_start, count = self._windows.get(key, (now, 0))

            if now - window_start >= window_seconds:
                window_start, count = now, 0

            if count >= limit:
                retry_after = max(0.0, window_seconds - (now - window_start))
                return RateLimitResult(
                    allowed=False, remaining=0, retry_after_seconds=retry_after
                )

            count += 1
            self._windows[key] = [window_start, count]

            return RateLimitResult(
                allowed=True,
                remaining=max(0, limit - count),
                retry_after_seconds=0.0,
            )


# ============================================================
# CONFIGURATION
# ============================================================

RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() not in (
    "0",
    "false",
    "no",
)

_backend: RateLimiterBackend = InMemoryRateLimiter()


def get_rate_limiter() -> RateLimiterBackend:
    """Overridden in tests to inject a fresh/fake backend and clock."""
    return _backend


def _client_identifier(request: Request) -> str:
    """
    Per-IP identity for the demo deployment. Proxy headers
    (X-Forwarded-For) are intentionally NOT trusted here, since without
    a configured trusted-proxy list they let a client spoof its own
    rate-limit key. request.client.host is the actual TCP peer FastAPI
    (via Starlette/uvicorn) observed, which cannot be spoofed by the
    client.
    """
    return client_host_identifier(request.client.host if request.client else None)


def client_host_identifier(host: str | None) -> str:
    """
    Shared key derivation for both HTTP (Request.client.host) and
    WebSocket (WebSocket.client.host) connections, so both transports
    are limited by the same per-IP budget via check_rate_limit() below.
    """
    return host or "unknown"


def check_rate_limit(key: str) -> RateLimitResult:
    """
    Transport-agnostic rate-limit check. Returns a RateLimitResult
    whether or not the limit is enabled/exceeded - callers decide how
    to react (HTTP 429 in enforce_rate_limit() below, a WebSocket close
    in app/api.py's /ws/chat).
    """
    if not RATE_LIMIT_ENABLED:
        return RateLimitResult(allowed=True, remaining=-1, retry_after_seconds=0.0)

    result = get_rate_limiter().hit(key, RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)

    if not result.allowed:
        from app.observability import record_rate_limit_event
        record_rate_limit_event()

    return result


def enforce_rate_limit(request: Request) -> None:
    """
    FastAPI dependency. Raises HTTP 429 with a Retry-After header when
    the caller's per-IP request rate exceeds the configured limit.
    """
    result = check_rate_limit(_client_identifier(request))

    if not result.allowed:
        retry_after = max(1, int(result.retry_after_seconds) + 1)
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down and try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )
