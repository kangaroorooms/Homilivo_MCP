"""Phase 3B part 2: the whole-search timeout and the per-request rooms token. Fakes only: no network, no keys."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from llm_approach import config, llm_client, pipeline, rooms_client, server_app
from llm_approach.actions import TurnBudget, search
from llm_approach.state import LLMConversationState
from llm_approach.tests.conftest import dec

TOKEN = "rooms-token-SECRET-777"


def good_room(n=1, **over):
    room = {"id": f"room-{n}", "title": f"Villa {n}", "city": "Bangalore", "pricePerMonth": 9000 + n,
            "roomType": "1BHK", "idealFor": ["Students"], "location": "Koramangala"}
    room.update(over)
    return room


class Api:
    """A fake rooms API. ``block`` makes calls wait until released (a slow API)."""

    def __init__(self) -> None:
        self.calls = 0
        self.headers: list[dict] = []
        self.block: threading.Event | None = None
        self.rooms = [good_room(1), good_room(2)]
        self.finished = threading.Event()

    def get(self, url, params, headers, timeout):
        self.calls += 1
        self.headers.append(headers)
        if self.block is not None:
            self.block.wait(10)
        api = self

        class Resp:
            status_code = 200

            def json(self):
                return {"data": api.rooms, "meta": {"total": len(api.rooms), "hasNextPage": False}}

        self.finished.set()
        return Resp()


@pytest.fixture
def api(monkeypatch):
    monkeypatch.delenv(config.OFFLINE_ROOMS_ENV, raising=False)
    monkeypatch.setenv("KANGAROO_TOKEN", TOKEN)
    fake = Api()
    monkeypatch.setattr(rooms_client, "_httpx_get", fake.get)
    yield fake
    if fake.block is not None:
        fake.block.set()


def slow(api, monkeypatch, seconds="0.2"):
    monkeypatch.setenv(config.SEARCH_TIMEOUT_ENV, seconds)
    api.block = threading.Event()


# ---- the timeout ----------------------------------------------------------------------------------


def test_default_timeout_is_15_seconds_and_a_bad_value_falls_back(monkeypatch):
    monkeypatch.delenv(config.SEARCH_TIMEOUT_ENV, raising=False)
    assert config.search_timeout_s() == 15.0
    for bad in ("abc", "0", "-3", "nan", "inf", ""):
        monkeypatch.setenv(config.SEARCH_TIMEOUT_ENV, bad)
        assert config.search_timeout_s() == 15.0
    monkeypatch.setenv(config.SEARCH_TIMEOUT_ENV, "7.5")
    assert config.search_timeout_s() == 7.5


def test_a_slow_rooms_api_times_out_cleanly(api, monkeypatch, caplog):
    slow(api, monkeypatch)
    started = time.monotonic()
    with caplog.at_level(logging.INFO):
        result = rooms_client.search_rooms({"city": "Bangalore"}, TurnBudget())
    assert time.monotonic() - started < 2
    assert not result.ok and result.error_class == "search_timeout" and result.rooms == []
    assert TOKEN not in caplog.text and "Bangalore" not in caplog.text


def test_timeout_gives_the_unavailable_reply_and_keeps_the_previous_state(api, monkeypatch):
    state = LLMConversationState()
    state.apply_search_result({"city": "Pune", "bhk": 2}, [good_room(7, city="Pune")])
    before = state.to_dict()
    slow(api, monkeypatch)
    result = search._run_search({"city": "Bangalore"}, state, TurnBudget(), "new_search")
    assert result.reply == rooms_client.UNAVAILABLE_MESSAGE and result.rooms is None
    assert result.error_class == "search_timeout"
    assert state.to_dict() == before  # nothing half-saved
    api.block.set()  # the slow call now finishes late: it must not touch the session
    assert api.finished.wait(5)
    time.sleep(0.1)
    assert state.to_dict() == before


def test_the_session_stays_usable_after_a_timeout(pipe, api, monkeypatch):
    slow(api, monkeypatch)
    first = pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert first.reply == rooms_client.UNAVAILABLE_MESSAGE and first.rooms is None
    assert pipe.state.rooms_on_screen == []
    api.block.set()
    api.finished.wait(5)
    monkeypatch.delenv(config.SEARCH_TIMEOUT_ENV)
    second = pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert second.success and len(second.rooms) == 2
    follow = pipe.turn("show cheaper ones", dec("REFINEMENT", city="Bangalore", max_price=9001))
    assert follow.success
    assert pipe.state.query_count == 3


def test_the_timeout_is_counted_as_a_failed_api_call(api, monkeypatch):
    from llm_approach import observability

    before = observability.get_metrics_snapshot()["tool_failures"]
    slow(api, monkeypatch)
    rooms_client.search_rooms({"city": "Bangalore"}, TurnBudget())
    assert observability.get_metrics_snapshot()["tool_failures"] == before + 1


def test_a_retry_waiting_when_the_timeout_hits_sends_no_second_request(api, monkeypatch):
    import httpx

    def failing(url, params, headers, timeout):
        api.calls += 1
        raise httpx.ConnectError("down")

    monkeypatch.setattr(rooms_client, "_httpx_get", failing)
    monkeypatch.setenv(config.SEARCH_TIMEOUT_ENV, "0.2")
    gate = threading.Event()
    result = rooms_client.search_rooms({"city": "Bangalore"}, TurnBudget(), sleep=lambda s: gate.wait(1))
    assert result.error_class == "search_timeout"
    gate.set()
    time.sleep(0.2)
    assert api.calls == 1


def test_a_normal_search_is_unchanged(api, monkeypatch):
    monkeypatch.setenv(config.SEARCH_TIMEOUT_ENV, "5")
    wrapped = rooms_client.search_rooms({"city": "Bangalore"}, TurnBudget())
    direct = rooms_client._search_rooms({"city": "Bangalore"}, TurnBudget())
    assert wrapped.ok and wrapped == direct and len(wrapped.rooms) == 2


def test_the_existing_retry_is_untouched(api, monkeypatch):
    import httpx

    calls = []

    def flaky(url, params, headers, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow")
        return api.get(url, params, headers, timeout)

    monkeypatch.setattr(rooms_client, "_httpx_get", flaky)
    slept = []
    result = rooms_client.search_rooms({"city": "Bangalore"}, TurnBudget(), sleep=slept.append, jitter=lambda a, b: 2.5)
    assert result.ok and len(calls) == 2 and slept == [2.5]


def test_a_validation_error_in_the_worker_still_reaches_the_caller(api):
    with pytest.raises(ValueError):
        rooms_client.search_rooms({}, TurnBudget())


def test_offline_search_does_not_use_the_timeout_thread(monkeypatch):
    monkeypatch.setenv(config.OFFLINE_ROOMS_ENV, "1")
    before = threading.active_count()
    assert rooms_client.search_rooms({"city": "Bangalore"}, TurnBudget()).ok
    assert threading.active_count() == before


# ---- the token ------------------------------------------------------------------------------------


def test_the_env_token_is_used_when_none_is_bound(api):
    rooms_client.search_rooms({"city": "Bangalore"}, TurnBudget())
    assert api.headers[-1]["Authorization"] == f"Bearer {TOKEN}"


def test_a_bound_token_wins_and_is_reset(api):
    handle = rooms_client.set_rooms_token("caller-own-token")
    try:
        rooms_client.search_rooms({"city": "Bangalore"}, TurnBudget())
    finally:
        rooms_client.reset_rooms_token(handle)
    assert api.headers[-1]["Authorization"] == "Bearer caller-own-token"
    rooms_client.search_rooms({"city": "Bangalore"}, TurnBudget())
    assert api.headers[-1]["Authorization"] == f"Bearer {TOKEN}"


def test_concurrent_requests_keep_their_own_token(api):
    seen = {}

    async def one(name):
        handle = rooms_client.set_rooms_token(f"token-{name}")
        try:
            await asyncio.sleep(0.01 * (3 - len(name)))
            await asyncio.to_thread(rooms_client.search_rooms, {"city": "Bangalore"}, TurnBudget())
            seen[name] = api.headers[-1]["Authorization"]
        finally:
            rooms_client.reset_rooms_token(handle)

    async def main():
        await asyncio.gather(one("a"), one("bb"))

    asyncio.run(main())
    assert api.calls == 2
    assert sorted(h["Authorization"] for h in api.headers) == ["Bearer token-a", "Bearer token-bb"]


def test_no_token_anywhere_still_gives_no_token_failure(monkeypatch):
    monkeypatch.delenv(config.OFFLINE_ROOMS_ENV, raising=False)
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)
    result = rooms_client.search_rooms({"city": "Bangalore"}, TurnBudget())
    assert not result.ok and result.error_class == "no_token"


def _server_chat(monkeypatch, api, bearer):
    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")

    async def fake_decide(user_message, state_summary, budget):
        from llm_approach.schema import parse_decision

        return parse_decision(dec("NEW_SEARCH", city="Bangalore"))

    monkeypatch.setattr(llm_client, "decide", fake_decide)
    monkeypatch.setattr(pipeline, "_input_blocked", lambda m: False)
    monkeypatch.setattr(pipeline, "_output_ok", lambda r: True)
    with TestClient(server_app.create_app()) as client:
        return client.post("/chat", json={"message": "rooms in bangalore", "session_id": str(uuid.uuid4())},
                           headers={"Authorization": f"Bearer {bearer}"})


def test_server_uses_the_callers_own_token_for_rooms(monkeypatch, api):
    r = _server_chat(monkeypatch, api, "byot-caller-token")
    assert r.status_code == 200 and len(r.json()["rooms"]) == 2
    assert api.headers[-1]["Authorization"] == "Bearer byot-caller-token"


def test_server_swaps_a_demo_token_for_the_server_credential(monkeypatch, api):
    monkeypatch.setattr(server_app._demo, "redeem_demo_token", lambda t: TOKEN if t == "demo-abc" else None)
    r = _server_chat(monkeypatch, api, "demo-abc")
    assert r.status_code == 200
    assert api.headers[-1]["Authorization"] == f"Bearer {TOKEN}"


def test_the_search_timeout_through_the_server_gives_the_polite_reply(monkeypatch, api):
    slow(api, monkeypatch)
    r = _server_chat(monkeypatch, api, "byot")
    assert r.status_code == 200 and r.json()["message"] == rooms_client.UNAVAILABLE_MESSAGE
    assert r.json()["rooms"] == []
