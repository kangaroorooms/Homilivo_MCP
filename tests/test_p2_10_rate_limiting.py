# P2-10: User-Aware Rate Limiting Tests
# Verifies session-based rate limiting works
# alongside existing IP-based rate limiting.

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import app.rate_limit as rate_limit_module
from app.api import app
from app.rate_limit import (
    RATE_LIMIT_MAX_REQUESTS,
    SESSION_RATE_LIMIT_MAX_REQUESTS,
    SESSION_RATE_LIMIT_WINDOW_SECONDS,
    InMemoryRateLimiter,
    check_session_rate_limit,
)

WS_URL = "/ws/chat?token=test-homilivo-token"


@pytest.fixture()
def fresh_backend(monkeypatch):
    """Each test gets its own backend so no state leaks between tests."""
    fresh = InMemoryRateLimiter()
    monkeypatch.setattr(rate_limit_module, "_backend", fresh)
    return fresh


@pytest.fixture()
def api_client(fresh_backend):
    return TestClient(app)


def _chat(client, headers, session_id=None):
    body = {"message": "hello"}
    if session_id is not None:
        body["session_id"] = session_id
    return client.post("/chat", json=body, headers=headers)


# ------------------------------------------------------------
# TEST 1
# ------------------------------------------------------------
def test_session_rate_limit_blocks_after_limit(
    api_client, auth_headers, mock_homilivo_success
):
    session_id = str(uuid4())

    for _ in range(SESSION_RATE_LIMIT_MAX_REQUESTS):
        assert _chat(api_client, auth_headers, session_id).status_code == 200

    limited = _chat(api_client, auth_headers, session_id)

    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1
    assert "Too many requests" in limited.json()["detail"]


# ------------------------------------------------------------
# TEST 2
# ------------------------------------------------------------
def test_session_rate_limit_different_sessions(
    api_client, auth_headers, mock_homilivo_success, monkeypatch
):
    session_a, session_b = str(uuid4()), str(uuid4())

    # 2 * SESSION limit (40) exceeds the default IP limit (30), so raise
    # the IP limit for this test to isolate the session behavior.
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_MAX_REQUESTS", 1000)

    for _ in range(SESSION_RATE_LIMIT_MAX_REQUESTS):
        assert _chat(api_client, auth_headers, session_a).status_code == 200
        assert _chat(api_client, auth_headers, session_b).status_code == 200


# ------------------------------------------------------------
# TEST 3
# ------------------------------------------------------------
def test_ip_limit_still_applies(api_client, auth_headers, mock_homilivo_success):
    for _ in range(RATE_LIMIT_MAX_REQUESTS):
        # A new session each time: the session limit never trips.
        assert _chat(api_client, auth_headers, str(uuid4())).status_code == 200

    limited = _chat(api_client, auth_headers, str(uuid4()))

    assert limited.status_code == 429
    assert "Retry-After" in limited.headers


# ------------------------------------------------------------
# TEST 4
# ------------------------------------------------------------
def test_session_rate_limit_constants_exist():
    assert SESSION_RATE_LIMIT_MAX_REQUESTS == 20
    assert SESSION_RATE_LIMIT_WINDOW_SECONDS == 60


# ------------------------------------------------------------
# TEST 5
# ------------------------------------------------------------
def test_no_session_id_skips_session_limit(
    api_client, auth_headers, mock_homilivo_success, monkeypatch
):
    # No session_id: the session limit is skipped (server-generated
    # UUIDs are never limited), so more than the session limit is fine...
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_MAX_REQUESTS", 1000)
    for _ in range(SESSION_RATE_LIMIT_MAX_REQUESTS + 5):
        assert _chat(api_client, auth_headers).status_code == 200

    # ...but the IP limit still applies.
    monkeypatch.setattr(
        rate_limit_module, "_backend", InMemoryRateLimiter()
    )
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_MAX_REQUESTS", 3)
    for _ in range(3):
        assert _chat(api_client, auth_headers).status_code == 200
    assert _chat(api_client, auth_headers).status_code == 429


# ------------------------------------------------------------
# TEST 6
# ------------------------------------------------------------
class _ClockedBackend:
    """Wraps a real limiter with a controllable clock."""

    def __init__(self) -> None:
        self.inner = InMemoryRateLimiter()
        self.now = 1000.0

    def hit(self, key, limit, window_seconds, now=None):
        return self.inner.hit(key, limit, window_seconds, self.now)


def test_session_limit_resets_after_window(monkeypatch):
    clock = _ClockedBackend()
    monkeypatch.setattr(rate_limit_module, "_backend", clock)
    session_id = str(uuid4())

    for _ in range(SESSION_RATE_LIMIT_MAX_REQUESTS):
        assert check_session_rate_limit(session_id).allowed is True

    assert check_session_rate_limit(session_id).allowed is False

    clock.now += SESSION_RATE_LIMIT_WINDOW_SECONDS + 1

    assert check_session_rate_limit(session_id).allowed is True


# ------------------------------------------------------------
# EXTRA COVERAGE
# ------------------------------------------------------------
def test_one_session_flooding_does_not_block_another_on_same_ip(
    api_client, auth_headers, mock_homilivo_success
):
    """The NAT scenario: an abusive session must not lock out its neighbour."""
    abusive, neighbour = str(uuid4()), str(uuid4())

    for _ in range(SESSION_RATE_LIMIT_MAX_REQUESTS + 1):
        _chat(api_client, auth_headers, abusive)

    assert _chat(api_client, auth_headers, abusive).status_code == 429
    # Same IP has used 21-22 of 30, so the neighbour still fits.
    assert _chat(api_client, auth_headers, neighbour).status_code == 200


def test_session_keys_are_prefixed_and_do_not_collide_with_ip_keys(fresh_backend):
    check_session_rate_limit("testclient")

    assert "session:testclient" in fresh_backend._windows
    assert "testclient" not in fresh_backend._windows


def test_session_limit_skipped_when_rate_limiting_disabled(monkeypatch, fresh_backend):
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_ENABLED", False)

    result = check_session_rate_limit(str(uuid4()))

    assert result.allowed is True
    assert fresh_backend._windows == {}


def test_ws_per_turn_session_limit_sends_error_without_closing(
    fresh_backend, mock_homilivo_success, monkeypatch
):
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_MAX_REQUESTS", 1000)
    monkeypatch.setattr(rate_limit_module, "SESSION_RATE_LIMIT_MAX_REQUESTS", 1)
    client = TestClient(app)
    session_id = str(uuid4())

    with client.websocket_connect(WS_URL) as ws:
        ws.send_json({"message": "hello", "session_id": session_id})
        while ws.receive_json()["event"] != "done":
            pass

        ws.send_json({"message": "hello again", "session_id": session_id})
        event = ws.receive_json()
        assert event["event"] == "error"
        assert "Too many requests" in event["detail"]

        # A different session on the same connection still works.
        ws.send_json({"message": "hello", "session_id": str(uuid4())})
        while (event := ws.receive_json())["event"] not in ("done", "error"):
            pass
        assert event["event"] == "done"
