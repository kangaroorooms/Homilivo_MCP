"""Phase 7, offline: sessions interleaved through the per-session lock, and a clean start after
a simulated restart (a fresh ``SessionStore`` in a new process). No network, no torch."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from llm_approach import llm_client, pipeline, reuse, server_app
from llm_approach.actions import followup
from llm_approach.schema import parse_decision
from llm_approach.state import SessionStore

from .conftest import dec


@pytest.fixture
def interleave(monkeypatch):
    """Pipeline whose LLM call takes a moment, and a log of when each call started and ended."""
    events: list[tuple[str, str]] = []

    async def slow_decide(user_message, state_summary, budget):
        events.append(("start", user_message))
        await asyncio.sleep(0.03)
        events.append(("end", user_message))
        return parse_decision(dec("GENERAL_CHAT", reply="Happy to chat!"))

    monkeypatch.setattr(llm_client, "decide", slow_decide)
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    pipeline.configure(full_guardrails=True)
    yield events
    pipeline.configure(full_guardrails=True)


def test_two_sessions_interleave_while_one_session_stays_in_order(interleave):
    events = interleave
    store = SessionStore()
    a, b = str(uuid.uuid4()), str(uuid.uuid4())

    async def run():
        tasks = []
        for sid, text in ((a, "hello A1"), (b, "hello B1"), (a, "hello A2"), (b, "hello B2")):
            tasks.append(asyncio.create_task(pipeline.handle_turn(sid, text, store=store)))
            await asyncio.sleep(0)  # let each turn reach the lock in this order
        return await asyncio.gather(*tasks)

    results = asyncio.run(run())
    assert all(r.success for r in results)

    def pos(kind, text):
        return events.index((kind, text))

    # Same session: the second turn starts only after the first has finished.
    assert pos("end", "hello A1") < pos("start", "hello A2")
    assert pos("end", "hello B1") < pos("start", "hello B2")
    # Different sessions overlap: B1 starts before A1 ends.
    assert pos("start", "hello B1") < pos("end", "hello A1")

    # No session sees the other's messages, and each history is in turn order.
    for sid, mine, other in ((a, "A", "B"), (b, "B", "A")):
        history = store.peek(sid).history
        users = [m["content"] for m in history if m["role"] == "user"]
        assert users == [f"hello {mine}1", f"hello {mine}2"]
        assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
        assert all(other not in m["content"] for m in history)
        assert store.peek(sid).turn_count == 2


def test_cancelled_turn_releases_the_session_lock(interleave):
    store = SessionStore()
    sid = str(uuid.uuid4())

    async def run():
        first = asyncio.create_task(pipeline.handle_turn(sid, "hello first", store=store))
        await asyncio.sleep(0.01)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        return await asyncio.wait_for(pipeline.handle_turn(sid, "hello second", store=store), timeout=2)

    assert asyncio.run(run()).success


def test_fresh_store_after_restart_starts_clean(monkeypatch):
    sid = str(uuid.uuid4())
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    script = [dec("NEW_SEARCH", city="Bangalore"), dec("ROOM_FOLLOWUP", "LOWEST_PRICE")]

    async def decide(user_message, state_summary, budget):
        return parse_decision(script.pop(0))

    monkeypatch.setattr(llm_client, "decide", decide)

    before = SessionStore()
    first = asyncio.run(pipeline.handle_turn(sid, "rooms in bangalore", store=before))
    assert first.rooms and before.peek(sid).rooms_on_screen

    after = SessionStore()  # a new process: nothing carried over
    assert after.peek(sid) is None and after.count() == 0
    second = asyncio.run(pipeline.handle_turn(sid, "which is cheapest", store=after))
    assert second.success and second.reply == followup.SEARCH_FIRST  # no stale rooms, no crash
    state = after.peek(sid)
    assert state.turn_count == 1 and not state.rooms_on_screen and state.selected_room_index is None
    assert [m["role"] for m in state.history] == ["user", "assistant"]


def test_server_keeps_working_after_a_simulated_restart(monkeypatch):
    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    monkeypatch.setenv("KANGAROO_TOKEN", "fake-rooms-credential")
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    from app.rate_limit import InMemoryRateLimiter

    limiter = InMemoryRateLimiter()
    monkeypatch.setattr(server_app._rl, "get_rate_limiter", lambda: limiter)

    async def decide(user_message, state_summary, budget):
        return parse_decision(dec("GENERAL_CHAT", reply="Happy to chat!"))

    monkeypatch.setattr(llm_client, "decide", decide)
    sid = str(uuid.uuid4())
    headers = {"Authorization": "Bearer anything"}

    monkeypatch.setattr(pipeline, "_store", SessionStore())
    with TestClient(server_app.create_app()) as client:
        assert client.post("/chat", json={"message": "hello", "session_id": sid}, headers=headers).json()["success"]

    monkeypatch.setattr(pipeline, "_store", None)  # process restarted: the module builds a new store lazily
    with TestClient(server_app.create_app()) as client:
        body = client.post("/chat", json={"message": "hello again", "session_id": sid}, headers=headers).json()
    assert body["success"] and body["session_id"] == sid
    assert pipeline.get_store().peek(sid).turn_count == 1
