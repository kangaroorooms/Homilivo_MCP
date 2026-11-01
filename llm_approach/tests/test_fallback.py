"""Fallback path: Groq down, timed out, or invalid JSON twice. Offline."""

from __future__ import annotations

import asyncio
import json
import uuid

from llm_approach import fallback, llm_client, pipeline, reuse
from llm_approach.state import LLMConversationState, SessionStore
from llm_approach.tests.conftest import dec


def test_timeout_with_a_city_still_searches(pipe):
    r = pipe.turn("find me a room in bangalore", None)  # None = Groq timeout
    assert r.metrics["fallback_used"] is True and r.metrics["tool_called"] is True
    assert r.metrics["action"] == "fallback_new_search"
    assert r.reply.startswith("I found 10 rooms in Bangalore") and len(r.rooms) == 10
    assert pipe.searches[-1]["city"] == "Bangalore"
    assert len(pipe.state.history) == 2


def test_fallback_keeps_bhk_and_budget(pipe):
    pipe.turn("2bhk in bangalore under 30k", None)
    sent = pipe.searches[-1]
    assert sent["city"] == "Bangalore" and sent["bhk"] == 2 and sent["max_price"] == 30000


def test_fallback_without_city_asks(pipe):
    r = pipe.turn("find me a room", None)
    assert r.reply == fallback.ASK_CITY_AND_BUDGET == "Could you tell me the city and budget you're looking for?"
    assert r.metrics["fallback_used"] is True and r.metrics["tool_called"] is False
    assert pipe.searches == [] and pipe.state.pending_question["type"] == "city"


def test_fallback_rejects_an_invalid_city(pipe):
    r = pipe.turn("rooms in pls urgent", None)  # the old extractor returns "Pls Urgent"
    assert r.reply == fallback.ASK_CITY_AND_BUDGET and pipe.searches == []


def test_fallback_answers_a_pending_city_question(pipe):
    pipe.turn("find me a room", None)
    r = pipe.turn("bangalore", None)  # bare city: the old extractor finds nothing
    assert pipe.searches[-1]["city"] == "Bangalore" and len(r.rooms) == 10


def test_fallback_bare_city_only_when_a_city_was_asked(pipe):
    r = pipe.turn("bangalore", None)  # nothing pending
    assert pipe.searches == [] and r.reply == fallback.ASK_CITY_AND_BUDGET


def test_fallback_pending_filler_is_not_a_city(pipe):
    pipe.turn("find me a room", None)
    pipe.turn("i dont know", None)
    assert pipe.searches == []


def test_fallback_reply_is_friendly_when_rooms_api_fails(pipe, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "")
    monkeypatch.setenv("KANGAROO_TOKEN", "fake-token-for-test")
    from llm_approach import rooms_client

    class Resp:
        status_code = 500

        def json(self):
            return {"message": "pool exhausted"}

    monkeypatch.setattr(rooms_client, "_httpx_get", lambda *a, **k: Resp())
    monkeypatch.setenv("LLM_APPROACH_SEARCH_TRANSPORT", "direct")  # v2 mcp: this fixture fakes the direct httpx client
    r = pipe.turn("rooms in bangalore", None)
    assert "unavailable" in r.reply and r.metrics["fallback_used"] is True
    assert "fake-token" not in json.dumps(r.metrics) + r.reply


def test_run_fallback_directly_never_searches_without_a_valid_city():
    from llm_approach.actions import TurnBudget

    budget = TurnBudget()
    state = LLMConversationState()
    result = fallback.run_fallback("asap pls", state, budget)
    assert result.tool_called is False and budget.tool_calls == 0


def test_fallback_slots_use_the_reused_extractor(monkeypatch):
    seen = []
    monkeypatch.setattr(
        reuse, "get_extract_search_parameters",
        lambda: (lambda q: seen.append(q) or {"city": "pune", "room_types": ["3BHK"], "min_price": 5000}),
    )
    slots = fallback.fallback_slots("anything", LLMConversationState())
    assert seen == ["anything"] and slots["city"] == "Pune" and slots["bhk"] == 3 and slots["min_price"] == 5000


# ---- through the real llm_client with a mocked Groq -------------------------------------


def _run_real_client(monkeypatch, fake_groq, replies, message):
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    groq = fake_groq(replies)
    store, sid = SessionStore(), str(uuid.uuid4())
    result = asyncio.run(pipeline.handle_turn(sid, message, store=store))
    return groq, result


def test_groq_timeout_falls_back_without_crashing(monkeypatch, fake_groq):
    groq, r = _run_real_client(monkeypatch, fake_groq, [asyncio.TimeoutError()], "rooms in bangalore")
    assert r.metrics["fallback_used"] is True and len(groq.calls) == 1  # a timeout is not retried
    assert r.reply.startswith("I found") and r.success is True


def test_groq_error_falls_back(monkeypatch, fake_groq):
    _, r = _run_real_client(monkeypatch, fake_groq, [ConnectionError("down")], "find me a room")
    assert r.metrics["fallback_used"] is True and r.reply == fallback.ASK_CITY_AND_BUDGET


def test_two_invalid_replies_fall_back_after_exactly_two_calls(monkeypatch, fake_groq):
    groq, r = _run_real_client(monkeypatch, fake_groq, ["not json", "{still not"], "rooms in bangalore")
    assert len(groq.calls) == 2 and r.metrics["fallback_used"] is True
    assert r.reply.startswith("I found")


def test_one_invalid_reply_is_retried_and_no_fallback(monkeypatch, fake_groq):
    groq, r = _run_real_client(
        monkeypatch, fake_groq,
        ["oops", json.dumps(dec("NEW_SEARCH", city="Bangalore"))], "rooms in bangalore",
    )
    assert len(groq.calls) == 2 and r.metrics["fallback_used"] is False
    assert r.metrics["intent"] == "NEW_SEARCH" and r.metrics["tokens_in"] == 200
