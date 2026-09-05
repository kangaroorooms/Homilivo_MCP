"""
Lightweight request rate limiting.

RateLimiterBackend is the storage abstraction. InMemoryRateLimiter is
the single-process backend (default); RedisRateLimiter shares one
budget per key across every process pointed at the same Redis,
selected via STATE_BACKEND=memory|redis (default memory) in
_create_default_backend() below - app/api.py's check_rate_limit() /
enforce_rate_limit() call sites are unchanged either way.

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

try:
    import redis as redis_lib
except ImportError:  # pragma: no cover - redis is always in requirements.txt
    redis_lib = None  # STATE_BACKEND=redis will raise a clear error if reached


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


class RedisRateLimiter:
    """
    Redis-backed fixed-window counter sharing one budget per key across
    every process pointed at the same Redis, instead of each process
    keeping its own independent InMemoryRateLimiter counters.

    Same rolling-window semantics as InMemoryRateLimiter (a key's
    window starts on its first hit and resets once `window_seconds`
    have elapsed since then - NOT an epoch-aligned bucket, which would
    let a window reset early or late depending on when the caller
    happened to start it). Read-modify-write is made atomic with
    WATCH/MULTI/EXEC optimistic locking rather than a Lua script, so it
    behaves the same against both a real Redis server and fakeredis.
    """

    def __init__(
        self,
        redis_client=None,
        url: str | None = None,
        key_prefix: str = "homilivo:ratelimit:",
    ) -> None:
        if redis_client is None and redis_lib is None:
            raise RuntimeError(
                "STATE_BACKEND=redis requires the 'redis' package "
                "(pip install -r requirements.txt)."
            )
        self._redis = redis_client or redis_lib.Redis.from_url(
            url or os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
        self._prefix = key_prefix

    def hit(
        self, key: str, limit: int, window_seconds: float, now: float | None = None
    ) -> RateLimitResult:
        now = time.time() if now is None else now
        redis_key = f"{self._prefix}{key}"

        with self._redis.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(redis_key)
                    raw = pipe.hgetall(redis_key)

                    window_start = float(raw["window_start"]) if raw else now
                    count = int(raw["count"]) if raw else 0

                    if now - window_start >= window_seconds:
                        window_start, count = now, 0

                    if count >= limit:
                        pipe.unwatch()
                        retry_after = max(0.0, window_seconds - (now - window_start))
                        return RateLimitResult(
                            allowed=False, remaining=0, retry_after_seconds=retry_after
                        )

                    count += 1
                    pipe.multi()
                    pipe.hset(
                        redis_key,
                        mapping={"window_start": window_start, "count": count},
                    )
                    pipe.expire(redis_key, int(window_seconds) + 1)
                    pipe.execute()

                    return RateLimitResult(
                        allowed=True,
                        remaining=max(0, int(limit) - count),
                        retry_after_seconds=0.0,
                    )
                except redis_lib.WatchError:
                    continue


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


def _create_default_backend() -> RateLimiterBackend:
    """STATE_BACKEND=redis|memory, default memory - read once at import,
    same as every other env-driven constant in this module."""
    if os.getenv("STATE_BACKEND", "memory").strip().lower() == "redis":
        return RedisRateLimiter()
    return InMemoryRateLimiter()


_backend: RateLimiterBackend = _create_default_backend()


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
