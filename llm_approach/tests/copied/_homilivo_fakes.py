# Copied from tests/conftest.py at commit 2365311 (copied, not imported).
# Changes: only SAMPLE_ROOMS, FakeHomilivoResponse and the mock_homilivo_success fixture; the patched name points at the copied tool. Every changed line is marked '# v2:'.
# Source under test is the copied tool llm_approach/tools/mcp_search.py (from app/server.py); app/ is not imported.

import json
from typing import Any
from urllib.request import Request

import pytest


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

    monkeypatch.setattr("llm_approach.tools.mcp_search.urllib.request.urlopen", fake_urlopen)
    return captured
