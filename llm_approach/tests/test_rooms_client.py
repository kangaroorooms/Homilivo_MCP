"""rooms_client tests. No network, no sleeping."""

from __future__ import annotations

import logging

import httpx
import pytest

from llm_approach import rooms_client
from llm_approach.actions import ToolBudgetExceeded, TurnBudget

TOKEN = "SECRET-TOKEN-VALUE-123"


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class Getter:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def __call__(self, url, params, headers, timeout):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


@pytest.fixture(autouse=True)
def live_env(monkeypatch):
    monkeypatch.delenv("LLM_APPROACH_OFFLINE_ROOMS", raising=False)
    monkeypatch.setenv("KANGAROO_TOKEN", TOKEN)


def go(getter, search=None, budget=None):
    slept = []
    result = rooms_client.search_rooms(
        search or {"city": "Bangalore"},
        budget or TurnBudget(),
        http_get=getter,
        sleep=slept.append,
        jitter=lambda a, b: 3.0,
    )
    return result, slept


OK_BODY = {"success": True, "data": [{"id": 1, "title": "A", "pricePerMonth": 5000, "ownerPhone": "999", "owner": {"x": 1}}]}


def test_success_filters_to_allowlist_and_sends_params():
    g = Getter(Resp(200, OK_BODY))
    result, slept = go(g, {"city": "Bangalore", "bhk": 2, "min_price": 1000, "max_price": 9000, "ideal_for": "Students", "amenity": "wifi"})
    assert result.ok and result.rooms == []  # wifi filtered client-side (room has no amenities)
    call = g.calls[0]
    assert call["params"] == {
        "city": "Bangalore", "minPrice": 1000, "maxPrice": 9000, "roomTypes": "2BHK",
        "idealFor": "Students", "sort": "latest", "limit": 10,
    }
    assert "amenity" not in call["params"]
    assert call["timeout"] == 15.0 and call["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert slept == []


def test_owner_fields_never_returned():
    result, _ = go(Getter(Resp(200, OK_BODY)))
    assert result.rooms == [{"id": 1, "title": "A", "pricePerMonth": 5000}]


def test_retry_once_on_timeout_then_succeed():
    g = Getter(httpx.ReadTimeout("slow"), Resp(200, OK_BODY))
    result, slept = go(g)
    assert result.ok and len(g.calls) == 2 and slept == [3.0]


def test_retry_once_on_connection_error_then_give_up():
    g = Getter(httpx.ConnectError("down"), httpx.ConnectError("down"))
    result, slept = go(g)
    assert not result.ok and result.error_class == "connection_error"
    assert len(g.calls) == 2 and len(slept) == 1


def test_two_timeouts_report_timeout():
    result, _ = go(Getter(httpx.ReadTimeout("a"), httpx.ReadTimeout("b")))
    assert result.error_class == "timeout"


@pytest.mark.parametrize("status", [500, 502, 503, 400, 401, 404, 429])
def test_no_retry_on_http_errors(status):
    g = Getter(Resp(status, {"message": "db busy"}), Resp(200, OK_BODY))
    result, slept = go(g)
    assert not result.ok and result.status == status and result.error_class == f"http_{status}"
    assert len(g.calls) == 1 and slept == []


def test_bad_payload():
    result, _ = go(Getter(Resp(200, {"success": True})))
    assert not result.ok and result.error_class == "bad_payload"


def test_second_tool_call_in_a_turn_is_blocked():
    budget = TurnBudget()
    g = Getter(Resp(200, OK_BODY), Resp(200, OK_BODY))
    go(g, budget=budget)
    with pytest.raises(ToolBudgetExceeded):
        go(g, budget=budget)
    assert len(g.calls) == 1


def test_city_is_required():
    with pytest.raises(ValueError):
        go(Getter(), {"city": None})
    with pytest.raises(ValueError):
        go(Getter(), {"city": ""})


def test_missing_token_makes_no_call(monkeypatch):
    monkeypatch.delenv("KANGAROO_TOKEN")
    g = Getter()
    result, _ = go(g)
    assert result.error_class == "no_token" and g.calls == []


def test_logs_status_and_message_but_no_token_or_headers(caplog):
    caplog.set_level(logging.DEBUG)
    go(Getter(Resp(500, {"message": "Too many connections"})))
    text = caplog.text
    assert "500" in text and "Too many connections" in text
    assert TOKEN not in text and "Bearer" not in text and "Authorization" not in text


def test_timeout_log_has_no_token(caplog):
    caplog.set_level(logging.DEBUG)
    go(Getter(httpx.ReadTimeout("a"), httpx.ReadTimeout("b")))
    assert "timeout" in caplog.text and TOKEN not in caplog.text


# ---- offline mode ---------------------------------------------------------


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)


def off(search):
    result = rooms_client.search_rooms(search, TurnBudget(), http_get=Getter())  # would fail if used
    assert result.ok
    return result.rooms


def test_offline_city_case_insensitive_and_other_city_empty(offline):
    assert len(off({"city": "BANGALORE"})) == 10
    assert off({"city": "Pune"}) == []


def test_offline_filters(offline):
    assert {r["id"] for r in off({"city": "bangalore", "bhk": 1})} == {"fixture-6", "fixture-9", "fixture-10"}
    assert {r["pricePerMonth"] for r in off({"city": "bangalore", "min_price": 17500, "max_price": 20000})} == {17500, 20000}
    assert [r["id"] for r in off({"city": "bangalore", "amenity": "WiFi"})] == ["fixture-10"]
    assert {r["id"] for r in off({"city": "bangalore", "amenity": "Hot Water", "max_price": 8000})} == {"fixture-2", "fixture-3"}
    assert all("Family" in r["idealFor"] for r in off({"city": "bangalore", "ideal_for": "family"}))
    assert len(off({"city": "bangalore", "bhk": 2, "max_price": 20000})) == 2


def test_offline_counts_against_budget(offline):
    budget = TurnBudget()
    rooms_client.search_rooms({"city": "bangalore"}, budget)
    with pytest.raises(ToolBudgetExceeded):
        rooms_client.search_rooms({"city": "bangalore"}, budget)


def test_offline_explicit_path(monkeypatch, tmp_path):
    f = tmp_path / "r.json"
    f.write_text('{"rooms": [{"id": 1, "city": "Pune", "pricePerMonth": 100}]}')
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", str(f))
    assert len(off({"city": "pune"})) == 1
