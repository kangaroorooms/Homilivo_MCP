"""
Pytest coverage for the PROPERTY_INTEREST -> subscription pricing
follow-up added to app/experiments/agent_modes.py.

Covers the four isolated helpers (detect_property_interest,
extract_verified_city, fetch_subscription_plans,
format_subscription_plans) in the same style as
tests/test_operations.py. No live HTTP calls - fetch_subscription_plans
is exercised against a stubbed httpx.Client.
"""

from __future__ import annotations

import httpx
import pytest

from app.experiments.agent_modes import (
    detect_property_interest,
    extract_verified_city,
    fetch_subscription_plans,
    format_subscription_plans,
)


# ============================================================
# detect_property_interest
# ============================================================

POSITIVE_PHRASES = [
    "contact the owner",
    "owner's number",
    "owner number",
    "get the number",
    "contact number",
    "phone number",
    "i want this",
    "i want this room",
    "i want this property",
    "book this",
    "book the room",
    "i want to book",
    "schedule a visit",
    "visit this",
    "i am interested in this",
    "i'm interested in this",
    "want to proceed",
    "how do i proceed",
    # Tier B semantic fallback - natural variations not literally in
    # _PROPERTY_INTEREST_PHRASES, caught via embedding similarity
    # against _PROPERTY_INTEREST_ANCHORS (gated by a margin over
    # _PROPERTY_INTEREST_DECOY_ANCHORS - see
    # _detect_property_interest_semantically()).
    "i want to contact owner",
    "how do i reach the owner",
    "get me the contact details",
    "connect me with the landlord",
    "can i get their number",
    "i want to contact the owner",
    "i want to book this room",
]

NEGATIVE_PHRASES = [
    "what is the rent",
    "does it have ac",
    "show amenities",
    "compare rooms",
    "show in table",
    "under 8000",
    "which is cheapest",
    # Same room-follow-up vocabulary as the decoy anchors - must not
    # trip the Tier B semantic fallback via shared vocabulary alone
    # ("room", "rent") with no real property-interest intent.
    "show me rooms in bangalore",
    "compare these rooms",
]


@pytest.mark.parametrize("phrase", POSITIVE_PHRASES)
def test_detect_property_interest_positive(phrase: str) -> None:

    assert detect_property_interest(phrase) is True

    assert detect_property_interest(phrase.upper()) is True


@pytest.mark.parametrize("phrase", NEGATIVE_PHRASES)
def test_detect_property_interest_negative(phrase: str) -> None:

    assert detect_property_interest(phrase) is False


# ============================================================
# extract_verified_city
# ============================================================

def test_extract_verified_city_with_selected_room() -> None:

    selected_room = {"city": "  Bangalore "}

    assert extract_verified_city(selected_room, []) == "bangalore"


def test_extract_verified_city_falls_back_to_rooms() -> None:

    rooms = [{"city": "Pune"}, {"city": "Mumbai"}]

    assert extract_verified_city(None, rooms) == "pune"


def test_extract_verified_city_no_city_field() -> None:

    room = {"locality": "Koramangala", "area": "HSR Layout"}

    assert extract_verified_city(room, [room]) is None


def test_extract_verified_city_no_rooms() -> None:

    assert extract_verified_city(None, []) is None


def test_extract_verified_city_nested_location_dict() -> None:

    selected_room = {"location": {"name": "Some Street", "city": "Chennai"}}

    assert extract_verified_city(selected_room, []) == "chennai"


# ============================================================
# format_subscription_plans
# ============================================================

def test_format_subscription_plans() -> None:

    plans = [
        {"plan": "free", "price": 0, "features": ["Browse listings"]},
        {"plan": "gold", "price": 149, "features": ["Contact 5 owners"]},
        {"plan": "platinum", "price": 299, "features": ["Unlimited contacts"]},
    ]

    output = format_subscription_plans(plans, "bangalore")

    assert "FREE" in output
    assert "GOLD" in output
    assert "PLATINUM" in output
    assert "Bangalore" in output
    assert "homilivo.com" in output


def test_format_subscription_plans_empty() -> None:

    output = format_subscription_plans([], "pune")

    assert "Pune" in output
    assert "homilivo.com" in output


# ============================================================
# fetch_subscription_plans
# ============================================================

class _FakeResponse:

    def __init__(self, json_data, status_code: int = 200) -> None:

        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:

        if self.status_code >= 400:

            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", "https://example.com"),
                response=httpx.Response(self.status_code),
            )

    def json(self):

        return self._json_data


class _FakeClient:

    def __init__(self, *args, **kwargs) -> None:

        pass

    def __enter__(self):

        return self

    def __exit__(self, *args) -> None:

        return None

    def get(self, url, params=None):

        return _FakeResponse(
            {
                "success": True,
                "data": [{"plan": "free", "price": 0, "features": []}],
            }
        )


class _FakeFailingClient:

    def __init__(self, *args, **kwargs) -> None:

        pass

    def __enter__(self):

        return self

    def __exit__(self, *args) -> None:

        return None

    def get(self, url, params=None):

        raise httpx.ConnectTimeout("timed out")


def test_fetch_subscription_plans_success(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(httpx, "Client", _FakeClient)

    result = fetch_subscription_plans("bangalore")

    assert isinstance(result, list)
    assert result[0]["plan"] == "free"


def test_fetch_subscription_plans_failure(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(httpx, "Client", _FakeFailingClient)

    result = fetch_subscription_plans("bangalore")

    assert result is None
