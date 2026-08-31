"""
Tests for outbound Homilivo API reliability: timeout, retry, and
controlled failure (app/server.py's search_rooms()).

These call app.server.search_rooms.fn directly (the underlying Python
function the MCP tool wraps) so retry/backoff/timeout behaviour can be
exercised without going through the full FastAPI/MCP stack. The
FastAPI-level "API failure degrades gracefully" behaviour is already
covered end-to-end in tests/test_api.py.
"""

from __future__ import annotations

import socket
import urllib.error
from urllib.parse import parse_qs, urlparse

import pytest

import app.server as server_module
from app.server import search_rooms


def _call_search_rooms(**kwargs):
    return search_rooms(**kwargs)


@pytest.fixture(autouse=True)
def _bind_token():
    token = server_module.set_kangaroo_token("test-token")
    yield
    server_module._kangaroo_token_var.reset(token)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Retries must not slow the suite down with real backoff waits."""
    monkeypatch.setattr(server_module.time, "sleep", lambda seconds: None)


def test_search_rooms_calls_correct_endpoint_and_params(mock_homilivo_success):
    """
    Tool/API call correctness: asserts the exact outbound path and
    query parameters search_rooms() sends to the Homilivo API, using
    conftest.py's mock_homilivo_success fixture (captures every
    urllib.request.Request passed to the patched urlopen).
    """
    result = _call_search_rooms(
        city="Jaipur",
        max_price=15000,
        room_types=["2BHK"],
        ideal_for=["Students"],
    )

    assert result["success"] is True

    request = mock_homilivo_success[-1]
    parsed = urlparse(request.full_url)

    assert parsed.scheme == "https"
    assert parsed.netloc == "homilivo.onrender.com"
    assert parsed.path == "/api/rooms"

    params = parse_qs(parsed.query)
    assert params["city"] == ["Jaipur"]
    assert params["maxPrice"] == ["15000"]
    assert params["roomTypes"] == ["2BHK"]
    assert params["idealFor"] == ["Students"]

    assert request.headers.get("Authorization") == "Bearer test-token"


def test_search_rooms_passes_a_timeout_to_urlopen(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["timeout"] = timeout
        raise AssertionError("stop before actually reading a response")

    monkeypatch.setattr(server_module.urllib.request, "urlopen", fake_urlopen)

    result = _call_search_rooms(city="Jaipur")

    assert result["success"] is False
    assert captured["timeout"] == server_module.HOMILIVO_API_TIMEOUT_SECONDS


def test_search_rooms_retries_on_timeout_then_succeeds(monkeypatch):
    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return None

        def read(self):
            return b'{"data": []}'

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        if calls["count"] < 2:
            raise socket.timeout("simulated timeout")
        return FakeResponse()

    monkeypatch.setattr(server_module.urllib.request, "urlopen", fake_urlopen)

    result = _call_search_rooms(city="Jaipur")

    assert result["success"] is True
    assert calls["count"] == 2


def test_search_rooms_retries_on_http_500_then_succeeds(monkeypatch):
    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return None

        def read(self):
            return b'{"data": []}'

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        if calls["count"] < 2:
            raise urllib.error.HTTPError(
                request.full_url, 500, "Internal Server Error", {}, None
            )
        return FakeResponse()

    monkeypatch.setattr(server_module.urllib.request, "urlopen", fake_urlopen)

    result = _call_search_rooms(city="Jaipur")

    assert result["success"] is True
    assert calls["count"] == 2


def test_search_rooms_does_not_retry_on_http_400(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        raise urllib.error.HTTPError(
            request.full_url, 400, "Bad Request", {}, None
        )

    monkeypatch.setattr(server_module.urllib.request, "urlopen", fake_urlopen)

    result = _call_search_rooms(city="Jaipur")

    assert result["success"] is False
    assert calls["count"] == 1


def test_search_rooms_exhausts_retries_and_returns_controlled_failure(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        raise socket.timeout("simulated persistent timeout")

    monkeypatch.setattr(server_module.urllib.request, "urlopen", fake_urlopen)

    result = _call_search_rooms(city="Jaipur")

    assert result["success"] is False
    assert "error" in result
    assert calls["count"] == server_module.HOMILIVO_API_MAX_ATTEMPTS


def test_search_rooms_does_not_retry_plain_connection_error(monkeypatch):
    """
    A generic ConnectionError (not HTTPError/TimeoutError/URLError) is
    not classified as retryable, matching tests/conftest.py's
    mock_homilivo_failure fixture used by the FastAPI-level tests.
    """
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        raise ConnectionError("simulated outage")

    monkeypatch.setattr(server_module.urllib.request, "urlopen", fake_urlopen)

    result = _call_search_rooms(city="Jaipur")

    assert result["success"] is False
    assert calls["count"] == 1
