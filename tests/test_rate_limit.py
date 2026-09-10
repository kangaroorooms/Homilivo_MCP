"""
Tests for app/rate_limit.py: the in-memory backend directly, and the
FastAPI-level 429/Retry-After behavior on /chat.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import app
import app.rate_limit as rate_limit_module
from app.rate_limit import InMemoryRateLimiter


# ============================================================
# BACKEND UNIT TESTS
# ============================================================

def test_allows_requests_under_the_limit():
    limiter = InMemoryRateLimiter()

    for _ in range(5):
        result = limiter.hit("client-a", limit=5, window_seconds=60, now=1000.0)
        assert result.allowed is True

    denied = limiter.hit("client-a", limit=5, window_seconds=60, now=1000.0)
    assert denied.allowed is False
    assert denied.retry_after_seconds > 0


def test_window_resets_after_expiry():
    limiter = InMemoryRateLimiter()

    for _ in range(3):
        limiter.hit("client-b", limit=3, window_seconds=10, now=1000.0)

    denied = limiter.hit("client-b", limit=3, window_seconds=10, now=1005.0)
    assert denied.allowed is False

    allowed_again = limiter.hit("client-b", limit=3, window_seconds=10, now=1011.0)
    assert allowed_again.allowed is True


def test_keys_are_isolated():
    limiter = InMemoryRateLimiter()

    for _ in range(3):
        limiter.hit("client-x", limit=3, window_seconds=60, now=1000.0)

    # A different key must have its own independent budget.
    result = limiter.hit("client-y", limit=3, window_seconds=60, now=1000.0)
    assert result.allowed is True


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


# ============================================================
# FASTAPI-LEVEL 429 BEHAVIOR
# ============================================================

@pytest.fixture()
def strict_limit(monkeypatch):
    """A fresh backend with a tiny limit so a couple of requests trip it."""
    fresh = InMemoryRateLimiter()
    monkeypatch.setattr(rate_limit_module, "_backend", fresh)
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_MAX_REQUESTS", 2)
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_WINDOW_SECONDS", 60.0)
    return fresh


def test_chat_returns_429_after_exceeding_limit(strict_limit, auth_headers, mock_homilivo_success):
    client = TestClient(app)

    for _ in range(2):
        response = client.post(
            "/chat", json={"message": "hello"}, headers=auth_headers
        )
        assert response.status_code == 200

    limited = client.post("/chat", json={"message": "hello"}, headers=auth_headers)

    assert limited.status_code == 429
    assert "Retry-After" in limited.headers
    assert int(limited.headers["Retry-After"]) >= 1


def test_rate_limit_can_be_disabled(monkeypatch, auth_headers, mock_homilivo_success):
    fresh = InMemoryRateLimiter()
    monkeypatch.setattr(rate_limit_module, "_backend", fresh)
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_MAX_REQUESTS", 1)
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_ENABLED", False)

    client = TestClient(app)

    for _ in range(3):
        response = client.post(
            "/chat", json={"message": "hello"}, headers=auth_headers
        )
        assert response.status_code == 200
