"""
Shared fixtures for the FastAPI-layer test suite (tests/test_api.py).

These mock the outbound Homilivo HTTP call made inside app/server.py's
search_rooms() tool, so the FastAPI/session/auth/mode/guardrail tests
are fast, deterministic, and do not depend on the live third-party API.
Genuine end-to-end verification against the real API is kept separately
in tests/test_mcp.py and tests/real_api_test.py - this file must not
replace those.
"""

from __future__ import annotations

import json
from typing import Any, Callable
from urllib.request import Request

import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """
    Every test shares one FastAPI TestClient app (and therefore one
    process-wide rate limiter backend). Without a reset, an unrelated
    test earlier in the same run could push a later test's /chat calls
    over the per-IP limit purely from request-count accumulation, not
    anything the later test is doing. Rate-limiting behavior itself is
    exercised in tests/test_rate_limit.py against a fresh backend.
    """
    from app.rate_limit import get_rate_limiter

    limiter = get_rate_limiter()
    if hasattr(limiter, "_windows"):
        limiter._windows.clear()

    yield


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-homilivo-token"}


SAMPLE_ROOMS = [
    {
        "id": "room-1",
        "title": "Sunrise 2BHK",
        "city": "Jaipur",
        "location": "Jagatpura",
        "pricePerMonth": 18000,
        "roomType": "2BHK",
        "idealFor": ["Family", "Working Professionals"],
        "amenities": ["AC", "Parking"],
    },
    {
        "id": "room-2",
        "title": "Budget PG",
        "city": "Jaipur",
        "location": "Mansarovar",
        "pricePerMonth": 8500,
        "roomType": "PG",
        "idealFor": ["Students"],
    },
    {
        "id": "room-3",
        "title": "Mid-range Flat",
        "city": "Jaipur",
        "location": "Vaishali Nagar",
        "pricePerMonth": 12000,
        "roomType": "2BHK",
        "idealFor": ["Students", "Working Professionals"],
    },
]


class FakeHomilivoResponse:
    """Minimal stand-in for the object urllib.request.urlopen() returns."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode()
        self.status = 200

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeHomilivoResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


@pytest.fixture()
def mock_homilivo_success(monkeypatch) -> list[Request]:
    """
    Patches the Homilivo HTTP call to return SAMPLE_ROOMS and records
    every outbound urllib.request.Request so tests can assert on the
    Authorization header actually sent (used by the token-isolation
    regression tests).
    """
    captured: list[Request] = []

    def fake_urlopen(request: Request, timeout: float | None = None):
        captured.append(request)
        return FakeHomilivoResponse({"data": SAMPLE_ROOMS})

    monkeypatch.setattr("app.server.urllib.request.urlopen", fake_urlopen)
    return captured


@pytest.fixture()
def mock_homilivo_empty(monkeypatch) -> None:
    def fake_urlopen(request: Request, timeout: float | None = None):
        return FakeHomilivoResponse({"data": []})

    monkeypatch.setattr("app.server.urllib.request.urlopen", fake_urlopen)


@pytest.fixture()
def mock_homilivo_failure(monkeypatch) -> None:
    def fake_urlopen(request: Request, timeout: float | None = None):
        raise ConnectionError("simulated Homilivo API outage")

    monkeypatch.setattr("app.server.urllib.request.urlopen", fake_urlopen)


@pytest.fixture()
def mock_homilivo_token_recorder(monkeypatch) -> Callable[[], list[str | None]]:
    """
    Like mock_homilivo_success, but returns a callable that yields the
    list of Authorization headers seen so far - used by the concurrent
    token-isolation test to prove each request's own token reached the
    outbound call.
    """
    tokens: list[str | None] = []

    def fake_urlopen(request: Request, timeout: float | None = None):
        tokens.append(request.headers.get("Authorization"))
        return FakeHomilivoResponse({"data": SAMPLE_ROOMS})

    monkeypatch.setattr("app.server.urllib.request.urlopen", fake_urlopen)
    return lambda: tokens
