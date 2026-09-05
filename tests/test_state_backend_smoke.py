"""
Real-Redis integration smoke test for Phase 2 (Redis-backed state).

Every other Redis-backend test in this suite (test_session_store.py,
test_rate_limit.py, test_demo_auth.py) runs against fakeredis - a
pure-Python, in-memory Redis-protocol server - so it stays fast,
deterministic, and needs no live service. Those tests prove the
RedisSessionStore/RedisRateLimiter code is CORRECT. They cannot prove
the thing Phase 2 actually exists for: that two independent app
instances, each with their own Redis client connection, observe the
SAME state through a real Redis server - fakeredis's default
FakeStrictRedis() talks to an in-process FakeServer object, so "two
clients" sharing one still never leaves the current Python process.

This file provides that missing proof against a REAL Redis reachable
at REDIS_URL (or redis://localhost:6379/0), simulating "two app
instances" as two separate SessionStore / RedisRateLimiter objects each
holding its own redis.Redis connection - the same relationship two
separate `uvicorn` worker processes would have to that Redis. It is
skipped cleanly (not failed) when no real Redis is reachable, so CI
stays green offline without requiring a live Redis service - per the
project's Phase-2 instructions, no Redis service was added to
.github/workflows/ci.yml.
"""

from __future__ import annotations

import socket
import time
import uuid
from urllib.parse import urlparse

import pytest

from app.rate_limit import RedisRateLimiter
from app.session_store import SessionStore

REDIS_URL = __import__("os").getenv("REDIS_URL", "redis://localhost:6379/0")


def _real_redis_is_reachable(url: str, timeout: float = 0.5) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


requires_real_redis = pytest.mark.skipif(
    not _real_redis_is_reachable(REDIS_URL),
    reason=(
        f"No real Redis reachable at {REDIS_URL} - this smoke test only "
        "runs when a live Redis is available (see module docstring); "
        "the Redis-backend LOGIC is still fully covered by the fakeredis "
        "tests in test_session_store.py / test_rate_limit.py / "
        "test_demo_auth.py."
    ),
)


@requires_real_redis
def test_two_session_store_instances_share_a_session_via_real_redis():
    """Simulates two app instances: instance_a writes a session,
    instance_b (a separate SessionStore with its own Redis connection)
    must see it - proving sessions are shared THROUGH Redis, not just
    correctly implemented against one shared fake client."""
    session_id = f"smoke-{uuid.uuid4()}"

    instance_a = SessionStore(backend="redis", url=REDIS_URL)
    instance_b = SessionStore(backend="redis", url=REDIS_URL)

    try:
        state = instance_a.get(session_id)
        state.selected_room = {"id": "shared-room"}
        instance_a.save(session_id, state)

        seen_by_b = instance_b.get(session_id)
        assert seen_by_b.selected_room == {"id": "shared-room"}
    finally:
        instance_a.reset(session_id)


@requires_real_redis
def test_two_rate_limiter_instances_share_one_counter_via_real_redis():
    """Two separate RedisRateLimiter instances (own Redis connection
    each) hitting the SAME key must draw from one shared budget, not
    two independent ones - proving the counter lives in Redis, not in
    either process."""
    key = f"smoke-{uuid.uuid4()}"

    instance_a = RedisRateLimiter(url=REDIS_URL)
    instance_b = RedisRateLimiter(url=REDIS_URL)
    now = time.time()

    try:
        for _ in range(3):
            result = instance_a.hit(key, limit=5, window_seconds=60, now=now)
            assert result.allowed is True

        # instance_b did not make any of the first 3 requests, but must
        # see them: only 2 of the remaining budget of 5 are left.
        allowed_1 = instance_b.hit(key, limit=5, window_seconds=60, now=now)
        allowed_2 = instance_b.hit(key, limit=5, window_seconds=60, now=now)
        denied = instance_b.hit(key, limit=5, window_seconds=60, now=now)

        assert allowed_1.allowed is True
        assert allowed_2.allowed is True
        assert denied.allowed is False
    finally:
        instance_a._redis.delete(f"{instance_a._prefix}{key}")
