# Copied from tests/test_rate_limit.py at commit 7dbcba3 (copied, not imported).
# Changes: imports now point to llm_approach.security.rate_limit; removed the TestClient/app.api import and the FastAPI-level 429 cases. Every changed line is marked '# v2:'.
# Not copied from this file: cases that need app.api, TestClient, agent_modes or the MCP server.

"""
Tests for app/rate_limit.py: both backends directly, and the
FastAPI-level 429/Retry-After behavior on /chat.
"""

from __future__ import annotations

import fakeredis
import pytest

import llm_approach.security.rate_limit as rate_limit_module  # v2: was: import app.rate_limit as rate_limit_module
from llm_approach.security.rate_limit import InMemoryRateLimiter, RedisRateLimiter  # v2: was: from app.rate_limit import InMemoryRateLimiter, RedisRateLimiter


# ============================================================
# BACKEND UNIT TESTS - parametrized over both backends
# ============================================================
#
# RedisRateLimiter is exercised here against fakeredis (a pure-Python,
# in-memory Redis-protocol server), so its actual code path - Redis
# commands and all - gets the same coverage as InMemoryRateLimiter
# without requiring a live Redis server. See
# tests/test_state_backend_smoke.py for a real-Redis check that two
# separate limiter instances share one counter, skipped cleanly when
# no real Redis is reachable.

@pytest.fixture(params=["memory", "redis"])
def limiter_factory(request):
    if request.param == "redis":
        client = fakeredis.FakeStrictRedis(decode_responses=True)
        return lambda: RedisRateLimiter(redis_client=client)
    return lambda: InMemoryRateLimiter()


def test_allows_requests_under_the_limit(limiter_factory):
    limiter = limiter_factory()

    for _ in range(5):
        result = limiter.hit("client-a", limit=5, window_seconds=60, now=1000.0)
        assert result.allowed is True

    denied = limiter.hit("client-a", limit=5, window_seconds=60, now=1000.0)
    assert denied.allowed is False
    assert denied.retry_after_seconds > 0


def test_window_resets_after_expiry(limiter_factory):
    limiter = limiter_factory()

    for _ in range(3):
        limiter.hit("client-b", limit=3, window_seconds=10, now=1000.0)

    denied = limiter.hit("client-b", limit=3, window_seconds=10, now=1005.0)
    assert denied.allowed is False

    allowed_again = limiter.hit("client-b", limit=3, window_seconds=10, now=1011.0)
    assert allowed_again.allowed is True


def test_keys_are_isolated(limiter_factory):
    limiter = limiter_factory()

    for _ in range(3):
        limiter.hit("client-x", limit=3, window_seconds=60, now=1000.0)

    # A different key must have its own independent budget.
    result = limiter.hit("client-y", limit=3, window_seconds=60, now=1000.0)
    assert result.allowed is True


# ============================================================
# IN-MEMORY-BACKEND-SPECIFIC UNIT TESTS
# ============================================================
# Stale-window purging is an InMemoryRateLimiter memory-hygiene detail
# (bounding the size of its in-process dict); Redis's own EXPIRE
# already reclaims stale keys, so RedisRateLimiter has nothing
# equivalent to test here.

def test_stale_windows_are_purged(monkeypatch):
    limiter = InMemoryRateLimiter(stale_after_seconds=100.0)
    monkeypatch.setattr(rate_limit_module, "_backend", limiter, raising=False)

    for i in range(10_001):
        limiter.hit(f"client-{i}", limit=5, window_seconds=60, now=1000.0)

    before_purge = len(limiter._windows)

    # A hit far enough in the future should trigger the size-based purge
    # check and drop everything older than stale_after_seconds.
    limiter.hit("client-fresh", limit=5, window_seconds=60, now=2000.0)

    assert len(limiter._windows) < before_purge
