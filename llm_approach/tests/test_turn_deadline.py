"""B6, option A: one deadline for the whole turn on the server, a shorter session-lock wait, and a keepalive on the WebSocket.

Fakes only; no real waiting. The deadline's timer is replaced by one that fires once the fake slow stage has started
(``server_app._deadline_sleep``), so "40 seconds" costs nothing. No Gemini, Groq, rooms or Redis-server call.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import uuid

import fakeredis
import pytest
from fastapi.testclient import TestClient

from llm_approach import config, gemini_limits, llm_resilience, observability, pipeline, providers, reuse, server_app, turn_deadline
from llm_approach.actions import ActionResult
from llm_approach.schema import Intent
from llm_approach.session_store import RedisSessionStore
from llm_approach.state import SessionBusyError, SessionStore
from llm_approach.tests.conftest import dec

from .test_llm_failures import counters, delta, env  # noqa: F401  (env is a fixture)
from .test_server import ws_turn

DEADLINE_REPLY = "That took too long, please try again"
SEARCH_FIELDS = ("current_search", "rooms_on_screen", "selected_room_index", "pending_question", "rejected_room_ids",
                 "api_room_count", "rooms_truncated", "search_page", "has_next", "total_rooms", "user_facts")


def turn_events() -> dict:
    return dict(observability.get_metrics_snapshot()["turn_events"])


def search_state(state) -> dict:
    return {name: copy.deepcopy(getattr(state, name)) for name in SEARCH_FIELDS}


def fire_when(monkeypatch, started: asyncio.Event) -> None:
    """The deadline fires as soon as the slow stage has started: no clock, no waiting."""

    async def fake_sleep(seconds):
        await started.wait()

    monkeypatch.setattr(server_app, "_deadline_sleep", fake_sleep)


# ---- the helper -----------------------------------------------------------------------------------------


def test_the_deadline_defaults_to_40_seconds_and_can_be_set(monkeypatch):
    monkeypatch.delenv("LLM_APPROACH_TURN_DEADLINE_S", raising=False)
    assert config.turn_deadline_s() == 40.0
    monkeypatch.setenv("LLM_APPROACH_TURN_DEADLINE_S", "25")
    assert config.turn_deadline_s() == 25.0
    for bad in ("0", "-3", "soon", "inf"):
        monkeypatch.setenv("LLM_APPROACH_TURN_DEADLINE_S", bad)
        assert config.turn_deadline_s() == 40.0


def test_a_turn_inside_its_deadline_is_unchanged():
    async def quick():
        assert turn_deadline.expired() is False
        return "answer"

    async def main():
        never = asyncio.Event()

        async def sleep(seconds):
            await never.wait()

        return await turn_deadline.run(quick(), 40, sleep=sleep)

    assert asyncio.run(main()) == "answer"
    assert turn_deadline.expired() is False


def test_a_turn_over_its_deadline_is_cancelled_after_its_cleanup_ran():
    log = []

    async def slow():
        try:
            log.append("started")
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            log.append(f"cancelled expired={turn_deadline.expired()}")  # the turn can tell a deadline from other cancels
            await asyncio.sleep(0)  # clean-up that awaits
            log.append("cleaned up")
            raise

    async def main():
        async def sleep(seconds):
            while not log:
                await asyncio.sleep(0)

        with pytest.raises(turn_deadline.DeadlineExceeded):
            await turn_deadline.run(slow(), 40, sleep=sleep)
        log.append("raised")

    asyncio.run(main())
    assert log == ["started", "cancelled expired=True", "cleaned up", "raised"]  # the clean-up finishes before the caller answers


def test_cancelling_the_caller_cancels_the_turn():
    state = {}

    async def slow():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            state["cancelled"] = turn_deadline.expired()
            raise

    async def main():
        never = asyncio.Event()

        async def sleep(seconds):
            await never.wait()

        outer = asyncio.ensure_future(turn_deadline.run(slow(), 40, sleep=sleep))
        await asyncio.sleep(0.01)
        outer.cancel()
        await asyncio.gather(outer, return_exceptions=True)
        await asyncio.sleep(0)

    asyncio.run(main())
    assert state == {"cancelled": False}  # a plain cancel is not reported as a deadline


# ---- the server ----------------------------------------------------------------------------------------------


@pytest.fixture
def served(pipe, monkeypatch):
    """The scripted pipeline (fake LLM, offline rooms) behind ``server_app.run_turn`` with its in-memory store."""
    monkeypatch.setattr(pipeline, "_store", pipe.store)
    return pipe


def test_a_turn_under_the_deadline_is_unchanged(served, monkeypatch):
    async def main():
        never = asyncio.Event()

        async def sleep(seconds):
            await never.wait()

        monkeypatch.setattr(server_app, "_deadline_sleep", sleep)
        served.script.append(dec("NEW_SEARCH", city="Bangalore"))
        return await server_app.run_turn(served.sid, "rooms in bangalore")

    before = turn_events().get("deadline_exceeded", 0)
    turn = asyncio.run(main())
    assert turn.success and turn.reply != DEADLINE_REPLY and served.state.rooms_on_screen
    assert turn_events().get("deadline_exceeded", 0) == before


def test_a_turn_over_the_deadline_gets_the_fixed_reply_and_is_counted(served, monkeypatch):
    before, status_before = turn_events().get("deadline_exceeded", 0), dict(observability.get_metrics_snapshot()["status_counts"])

    async def main():
        started, never = asyncio.Event(), asyncio.Event()

        async def block():
            started.set()
            await never.wait()

        fire_when(monkeypatch, started)
        served.script.append(block)
        return await server_app.run_turn(served.sid, "rooms in bangalore")

    turn = asyncio.run(main())
    assert turn.reply == DEADLINE_REPLY == pipeline.DEADLINE_REPLY and turn.success is False
    done = server_app.build_done(turn, served.sid, "fast")
    assert done["message"] == DEADLINE_REPLY and done["success"] is False and done["room_count"] == 0
    assert turn_events().get("deadline_exceeded", 0) == before + 1  # a fixed word, nothing else
    status_after = observability.get_metrics_snapshot()["status_counts"]
    assert status_after.get("error", 0) == status_before.get("error", 0) + 1  # the turn is recorded as an error, not a success


def test_a_timed_out_turn_keeps_the_message_and_the_error_reply_and_nothing_else(served, monkeypatch):
    served.turn("rooms in bangalore", llm=dec("NEW_SEARCH", city="Bangalore"))  # a finished turn: rooms on screen
    before_state = copy.deepcopy(served.state)
    assert before_state.rooms_on_screen

    async def main():
        started = asyncio.Event()
        loop = asyncio.get_running_loop()
        gate = threading.Event()

        def blocking_output(text):  # runs in a thread, after the search already changed the state
            loop.call_soon_threadsafe(started.set)
            gate.wait(5)
            return True, text

        monkeypatch.setattr(reuse, "get_output_guardrail", lambda: blocking_output)
        fire_when(monkeypatch, started)
        served.script.append(dec("REFINEMENT", op="CHANGE_PRICE", max_price=9999))
        try:
            return await server_app.run_turn(served.sid, "cheaper please")
        finally:
            gate.set()

    turn = asyncio.run(main())
    assert turn.reply == DEADLINE_REPLY
    state = served.state
    assert search_state(state) == search_state(before_state)  # the search the turn made is gone
    assert state.turn_count == before_state.turn_count + 1
    assert state.history[-2:] == [{"role": "user", "content": "cheaper please"}, {"role": "assistant", "content": DEADLINE_REPLY}]
    assert state.last_bot_message == DEADLINE_REPLY
    assert state.user_queries[-1]["text"] != "cheaper please"  # the query log holds the placeholder, never the text


def test_a_thread_still_running_after_the_cancel_cannot_touch_the_session(served, monkeypatch):
    served.turn("rooms in bangalore", llm=dec("NEW_SEARCH", city="Bangalore"))
    before_state = copy.deepcopy(served.state)
    finished = threading.Event()

    async def main():
        started = asyncio.Event()
        loop = asyncio.get_running_loop()
        gate = threading.Event()

        def late_handler(decision, validated, state, budget, **kwargs):
            loop.call_soon_threadsafe(started.set)
            gate.wait(5)
            state.rooms_on_screen = [{"id": "intruder"}]  # a search that finishes after the deadline
            state.current_search = {"city": "Intruder"}
            finished.set()
            return ActionResult("late reply", None, "late")

        monkeypatch.setitem(pipeline._HANDLERS, Intent.NEW_SEARCH, late_handler)
        fire_when(monkeypatch, started)
        served.script.append(dec("NEW_SEARCH", city="Pune"))
        turn = await server_app.run_turn(served.sid, "pune instead")
        after_cancel = search_state(served.state)
        gate.set()  # now the thread runs to its end
        assert await asyncio.to_thread(finished.wait, 5)
        return turn, after_cancel

    turn, after_cancel = asyncio.run(main())
    assert turn.reply == DEADLINE_REPLY
    assert after_cancel == search_state(before_state)
    assert search_state(served.state) == search_state(before_state)  # still, after the thread wrote to its copy


def test_a_redis_session_holds_the_restored_state_plus_the_error_reply(pipe, monkeypatch):
    server = fakeredis.FakeServer()
    store = RedisSessionStore(redis_client=fakeredis.FakeRedis(server=server, decode_responses=True))
    monkeypatch.setattr(pipeline, "_store", store)
    session = str(uuid.uuid4())
    pipe.script.append(dec("NEW_SEARCH", city="Bangalore"))
    asyncio.run(pipeline.handle_turn(session, "rooms in bangalore"))
    before_state = copy.deepcopy(store.peek(session))

    async def main():
        started, never = asyncio.Event(), asyncio.Event()

        async def block():
            started.set()
            await never.wait()

        fire_when(monkeypatch, started)
        pipe.script.append(block)
        return await server_app.run_turn(session, "something slow")

    turn = asyncio.run(main())
    assert turn.reply == DEADLINE_REPLY
    saved = store.peek(session)
    assert search_state(saved) == search_state(before_state)
    assert saved.history[-1]["content"] == DEADLINE_REPLY and saved.turn_count == before_state.turn_count + 1
    assert store._redis.get(store._lock_key(session)) is None  # the lock was released


def test_the_gemini_request_is_counted_once_and_not_sent_again(env, monkeypatch):
    sends, holder = [], {}
    monkeypatch.setattr(pipeline, "_store", env.store)

    async def via_limiter(messages, max_tokens=None, provider=None):
        async def send():
            sends.append(1)
            holder["started"].set()
            await asyncio.Event().wait()

        await gemini_limits.get_limiter().call(send, budget=450, prompt_estimate=10)

    async def main():
        holder["started"] = asyncio.Event()
        fire_when(monkeypatch, holder["started"])
        monkeypatch.setattr(providers, "complete", via_limiter)
        return await server_app.run_turn(env.sid, "rooms in bangalore")

    before = counters()
    turn = asyncio.run(main())
    assert turn.reply == DEADLINE_REPLY
    assert sends == [1]  # no retry after the cancel
    assert gemini_limits.get_limiter().counter.used_requests() == 1  # counted when it was sent, once
    assert llm_resilience.breaker("gemini").consecutive == 0 and delta(before[0], counters()[0]) == {}  # not a provider failure


# ---- the session lock wait ------------------------------------------------------------------------------------


def test_the_lock_wait_defaults_to_10_seconds_for_both_stores(monkeypatch):
    monkeypatch.delenv("LLM_APPROACH_SESSION_LOCK_WAIT_S", raising=False)
    assert config.DEFAULT_LOCK_WAIT_S == 10.0 and config.session_lock_wait_s() == 10.0
    assert SessionStore().lock_wait_s == 10.0
    assert RedisSessionStore(redis_client=fakeredis.FakeRedis(decode_responses=True))._lock_wait_s == 10.0
    monkeypatch.setenv("LLM_APPROACH_SESSION_LOCK_WAIT_S", "4")
    assert SessionStore().lock_wait_s == 4.0


def test_the_in_memory_store_stops_waiting_for_a_busy_session():
    store = SessionStore(lock_wait_s=0.001)
    session = str(uuid.uuid4())
    before = dict(observability.get_metrics_snapshot()["session_store_events"]).get("lock_timeout", 0)

    async def main():
        ran = []
        async with store.turn(session) as state:
            with pytest.raises(SessionBusyError):
                async with store.turn(session):
                    ran.append("second turn body")  # must not run
            assert ran == []
        async with store.turn(session):  # the lock is free again and the active count did not leak
            pass
        assert store._entries[session].active == 0

    asyncio.run(main())
    assert dict(observability.get_metrics_snapshot()["session_store_events"]).get("lock_timeout", 0) == before + 1


def test_a_busy_session_answers_at_once_without_calling_the_llm(pipe, monkeypatch):
    pipe.store = SessionStore(lock_wait_s=0.001)

    async def main():
        async with pipe.store.turn(pipe.sid):
            return await pipeline.handle_turn(pipe.sid, "rooms in bangalore", store=pipe.store)

    result = asyncio.run(main())
    assert result.reply == pipeline.RATE_LIMIT_REPLY and result.success is False and pipe.llm_calls == 0


def test_the_redis_store_stops_waiting_after_its_bound():
    store = RedisSessionStore(redis_client=fakeredis.FakeRedis(decode_responses=True), lock_wait_s=0.001)
    session = str(uuid.uuid4())

    async def main():
        async with store.turn(session):
            with pytest.raises(SessionBusyError):
                async with store.turn(session):
                    pass

    asyncio.run(main())


def test_the_stage_limits_all_fit_inside_the_turn_deadline():
    # lock wait, then the LLM step (its limiter waits end 20 s after it starts), then one rooms search or one pricing call
    assert config.DEFAULT_LOCK_WAIT_S < gemini_limits.DEFAULT_TURN_DEADLINE_S < config.DEFAULT_TURN_DEADLINE_S
    assert config.DEFAULT_SEARCH_TIMEOUT_S < config.DEFAULT_TURN_DEADLINE_S


# ---- the WebSocket ------------------------------------------------------------------------------------------------


@pytest.fixture
def ws_server(monkeypatch):
    from llm_approach.security.rate_limit import InMemoryRateLimiter

    for name in ("LLM_APPROACH_ENV", "LLM_APPROACH_OFFLINE_ROOMS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")
    monkeypatch.setenv("STREAM_CHUNK_DELAY_SECONDS", "0")
    monkeypatch.setenv("KANGAROO_TOKEN", "fake-rooms-credential")
    limiter = InMemoryRateLimiter()
    monkeypatch.setattr(server_app._rl, "get_rate_limiter", lambda: limiter)
    monkeypatch.setattr(pipeline, "_store", SessionStore())
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    yield
    pipeline.configure(full_guardrails=True)


def test_the_websocket_gets_ping_events_while_a_turn_runs(ws_server, monkeypatch):
    monkeypatch.setenv("WS_HEARTBEAT_INTERVAL_SECONDS", "0.01")
    release = threading.Event()

    async def slow_turn(session_id, message, *, store=None):
        await asyncio.to_thread(release.wait, 5)  # a slow turn that does not block the loop
        return pipeline.TurnResult(reply="finally here", rooms=None, metrics={}, success=True)

    monkeypatch.setattr(pipeline, "handle_turn", slow_turn)
    with TestClient(server_app.create_app()) as client, client.websocket_connect("/ws/chat?token=abc") as ws:
        ws.send_json({"message": "hi", "session_id": str(uuid.uuid4())})
        while (first := ws.receive_json()) != {"event": "typing"}:
            assert first == {"event": "ping"}  # the idle heartbeat may fire before the message arrives (10 ms interval here)
        pings = [ws.receive_json() for _ in range(3)]
        assert pings == [{"event": "ping"}] * 3  # the existing ping event, exactly as the idle heartbeat sends it
        release.set()
        events = []
        while True:
            events.append(ws.receive_json())
            if events[-1]["event"] == "done":
                break
        assert events[-1]["message"] == "finally here"
        assert {e["event"] for e in events} <= {"ping", "token", "done"}  # no new event type, nothing after done


def test_the_websocket_answers_a_timed_out_turn_with_the_fixed_reply(ws_server, monkeypatch):
    async def stuck(session_id, message, *, store=None):
        started.set()
        await asyncio.Event().wait()

    started = asyncio.Event()
    monkeypatch.setattr(pipeline, "handle_turn", stuck)
    fire_when(monkeypatch, started)
    with TestClient(server_app.create_app()) as client, client.websocket_connect("/ws/chat?token=abc") as ws:
        events = ws_turn(ws, message="hi", session_id=str(uuid.uuid4()))
        assert events[0] == {"event": "typing"}
        assert "".join(e["text"] for e in events if e["event"] == "token") == DEADLINE_REPLY
        assert events[-1]["event"] == "done" and events[-1]["success"] is False
        again = ws_turn(ws, message="again", session_id=events[-1]["session_id"])  # the connection is still usable
        assert again[-1]["event"] == "done" and again[-1]["message"] == DEADLINE_REPLY
