"""Phase 4A: sessions in Redis. fakeredis only: no network, no real Redis, no LLM."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
import uuid

import fakeredis
import pytest
from fastapi.testclient import TestClient

from llm_approach import config, observability, pipeline, reuse, server_app
from llm_approach import session_store as ss
from llm_approach.state import (
    PENDING_TYPES,
    STATE_SCHEMA_VERSION,
    ERROR_REPLY,
    LLMConversationState,
    SessionBusyError,
    SessionStore,
    StateDecodeError,
)
from llm_approach.tests.conftest import dec

WARN_MARKER = "SECRET-MARKER-9f3a"  # planted in stored blobs: must never reach a log line


@pytest.fixture
def server():
    return fakeredis.FakeServer()


def client_for(server):
    return fakeredis.FakeRedis(server=server, decode_responses=True)


def make(server, **kw):
    kw.setdefault("lock_wait_s", 2.0)
    kw.setdefault("lock_ttl_ms", 5000)
    kw.setdefault("lock_max_hold_s", 30.0)
    return ss.RedisSessionStore(redis_client=client_for(server), **kw)


def sid() -> str:
    return str(uuid.uuid4())


def events() -> dict[str, int]:
    return dict(observability.get_metrics_snapshot()["session_store_events"])


def delta(before: dict[str, int], name: str) -> int:
    return events().get(name, 0) - before.get(name, 0)


def populated() -> LLMConversationState:
    """A state with every field set to a non-default value."""
    st = LLMConversationState()
    st.apply_search_result(
        {"city": "Bangalore", "area": "Koramangala", "bhk": 2, "min_price": 5000, "max_price": 20000,
         "room_types": ["PG"], "ideal_for": ["Students"], "amenity": "wifi"},
        [{"id": "r1", "title": "Room One", "pricePerMonth": 9000, "city": "Bangalore"},
         {"id": 2, "title": "Room Two", "pricePerMonth": 12000, "city": "Bangalore"}],
        api_count=7, truncated=True, page=2, has_next=True, total=19,
    )
    st.select_room(2)
    st.set_pending("clarify", {"slot": "budget"})
    st.rejected_room_ids = ["r1", 5]
    st.record_query("2bhk in bangalore", "NEW_SEARCH")
    st.set_fact("name", "Faizal")
    st.record_bot_reply("Here are 2 rooms", [{"title": "Room One"}])
    st.record_turn("2bhk in bangalore", "Here are 2 rooms", now=1234.5)
    return st


# ---- to_dict / from_dict --------------------------------------------------------------------


def test_every_field_round_trips():
    st = populated()
    for f in dataclasses.fields(st):  # a new field must be set in populated(), or this fails
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        assert getattr(st, f.name) != default, f"populated() leaves {f.name} at its default"
    data = st.to_dict()
    assert set(data) == {f.name for f in dataclasses.fields(st)} | {"schema_version"}
    assert data["schema_version"] == STATE_SCHEMA_VERSION
    back = LLMConversationState.from_dict(json.loads(json.dumps(data)))  # through real JSON text
    assert back == st
    for f in dataclasses.fields(st):
        assert getattr(back, f.name) == getattr(st, f.name), f.name
    back.rooms_on_screen[0]["title"] = "changed"
    assert st.rooms_on_screen[0]["title"] == "Room One"  # no shared lists


def test_default_state_round_trips_and_pending_without_context():
    st = LLMConversationState()
    assert LLMConversationState.from_dict(json.loads(json.dumps(st.to_dict()))) == st
    st.set_pending("city")
    assert LLMConversationState.from_dict(st.to_dict()).pending_question == {"type": "city", "context": None}


def test_to_dict_refuses_a_value_that_is_not_json():
    st = LLMConversationState()
    st.rejected_room_ids = [object()]
    with pytest.raises(TypeError):
        st.to_dict()


def test_field_checks_cover_every_field():
    from llm_approach import state

    assert set(state._FIELD_CHECKS) == {f.name for f in dataclasses.fields(LLMConversationState)}


@pytest.mark.parametrize("mutate,kind", [
    (lambda d: d.update(schema_version=2), "wrong_schema_version"),
    (lambda d: d.update(schema_version=True), "wrong_schema_version"),
    (lambda d: d.pop("schema_version"), "wrong_schema_version"),
    (lambda d: d.update(history="oops"), "bad_field:history"),
    (lambda d: d.update(history=[{"role": "user", "content": 5}]), "bad_field:history"),
    (lambda d: d.update(rooms_on_screen=[1, 2]), "bad_field:rooms_on_screen"),
    (lambda d: d.update(selected_room_index=True), "bad_field:selected_room_index"),
    (lambda d: d.update(selected_room_index=9), "bad_field:selected_room_index"),
    (lambda d: d.update(pending_question={"type": "nope", "context": None}), "bad_field:pending_question"),
    (lambda d: d.update(turn_count=-1), "bad_field:turn_count"),
    (lambda d: d.update(search_page=0), "bad_field:search_page"),
    (lambda d: d.update(user_facts={"name": "Faizal"}), "bad_field:user_facts"),
    (lambda d: d.update(transcript=[{"role": "robot", "text": "x"}]), "bad_field:transcript"),
    (lambda d: d.update(rooms_truncated="yes"), "bad_field:rooms_truncated"),
])
def test_from_dict_rejects_bad_blobs_with_a_content_free_message(mutate, kind):
    data = populated().to_dict()
    data["last_bot_message"] = WARN_MARKER
    mutate(data)
    with pytest.raises(StateDecodeError) as err:
        LLMConversationState.from_dict(data)
    assert str(err.value) == kind and WARN_MARKER not in str(err.value)


@pytest.mark.parametrize("blob", [None, [], "text", 5])
def test_from_dict_rejects_non_objects(blob):
    with pytest.raises(StateDecodeError):
        LLMConversationState.from_dict(blob)


def test_pending_types_still_match_the_check():
    st = LLMConversationState()
    for kind in PENDING_TYPES:
        st.set_pending(kind)
        LLMConversationState.from_dict(st.to_dict())


# ---- the Redis store ---------------------------------------------------------------------------


def test_turn_saves_and_a_new_store_object_sees_it_after_a_restart(server):
    s1, session = make(server), sid()

    async def first():
        async with s1.turn(session, "hello") as st:
            st.record_query("hello")
            st.set_fact("name", "Faizal")
            st.record_turn("hello", "hi")

    asyncio.run(first())
    s2 = make(server)  # a new process: new object, same Redis
    seen = s2.peek(session)
    assert seen is not None and seen.turn_count == 1 and seen.get_fact("name") == "Faizal"
    assert s2.count() == 1

    async def second():
        async with s2.turn(session, "again") as st:
            assert st.get_fact("name") == "Faizal" and st.turn_count == 1
            st.record_turn("again", "ok")

    asyncio.run(second())
    assert make(server).peek(session).turn_count == 2


def test_pipeline_session_survives_a_restart(pipe, server):
    pipe.store = make(server)
    first = pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert first.rooms
    on_screen = len(pipe.state.rooms_on_screen)
    pipe.store = make(server)  # restart: the in-process state is gone
    assert pipe.store._locals == {}
    r = pipe.turn("which is cheapest", dec("ROOM_FOLLOWUP", op="LOWEST_PRICE"))
    assert r.success and len(pipe.state.rooms_on_screen) == on_screen and pipe.state.turn_count == 2


def test_two_servers_do_not_overwrite_each_other(server):
    """Turns on one session through two store objects, one after the other and at the same time."""
    a, b, session = make(server), make(server), sid()

    async def add(store, text):
        async with store.turn(session, text) as st:
            await asyncio.sleep(0.05)  # a window in which an unlocked second turn would read the same old state
            st.record_query(text)
            st.record_turn(text, "ok")

    async def main():
        await add(a, "one")
        await add(b, "two")
        await asyncio.gather(add(a, "three"), add(b, "four"), add(a, "five"))

    asyncio.run(main())
    st = make(server).peek(session)
    assert st.turn_count == 5 and sorted(q["text"] for q in st.user_queries) == sorted(["one", "two", "three", "four", "five"])


def test_turns_on_one_session_do_not_overlap_but_other_sessions_run_in_parallel(server):
    a, b = make(server), make(server)
    s1, s2 = sid(), sid()
    spans: dict[str, list[tuple[float, float]]] = {"x": [], "y": []}

    async def run(store, session, name):
        async with store.turn(session):
            t0 = time.monotonic()
            await asyncio.sleep(0.15)
            spans[name].append((t0, time.monotonic()))

    async def main():
        await asyncio.gather(run(a, s1, "x"), run(b, s1, "x"), run(a, s2, "y"))

    asyncio.run(main())
    (first, second) = sorted(spans["x"])
    assert second[0] >= first[1]  # one after the other
    assert spans["y"][0][0] < first[1]  # the other session did not wait


def test_exception_in_the_body_records_the_error_reply_and_saves(server):
    store, session = make(server), sid()

    async def boom():
        with pytest.raises(RuntimeError):
            async with store.turn(session, "hi"):
                raise RuntimeError("boom")

    asyncio.run(boom())
    st = make(server).peek(session)
    assert st.turn_count == 1 and st.history[-1] == {"role": "assistant", "content": ERROR_REPLY}
    assert asyncio.run(_lock_free(store, session))


async def _lock_free(store, session) -> bool:
    return store._redis.get(store._lock_key(session)) is None


def test_invalid_session_id_raises_like_the_memory_store(server):
    async def go():
        async with make(server).turn("not-a-uuid"):
            pass

    with pytest.raises(ValueError):
        asyncio.run(go())
    assert make(server).peek("not-a-uuid") is None


def test_get_or_create_does_not_save_and_save_does(server):
    store, session = make(server), sid()
    st = store.get_or_create(session)
    st.record_query("hello")
    assert store.peek(session) is None and store.count() == 0
    store.save(session, st)
    assert store.peek(session).query_count == 1 and make(server).count() == 1


def test_reset(server):
    store, session = make(server), sid()
    store.save(session, populated())
    assert store.reset(session) is True and store.peek(session) is None and store.count() == 0
    assert store.reset(session) is False


def test_reset_during_a_turn_is_not_undone_by_that_turns_save(server):
    store, session = make(server), sid()
    store.save(session, populated())

    async def main():
        async with store.turn(session, "hi") as st:
            st.record_turn("hi", "ok")
            assert store.reset(session) is True  # the user presses reset while the turn runs
        assert store.peek(session) is None  # the finished turn did not bring the session back
        async with store.turn(session, "again") as st:
            assert st.turn_count == 0  # the next turn starts empty
            st.record_turn("again", "ok")

    asyncio.run(main())
    assert store.peek(session).turn_count == 1
    assert store._redis.get(store._reset_key(session)) is None


# ---- corrupt, wrong-version and unreadable blobs -----------------------------------------------------


@pytest.mark.parametrize("blob,event", [
    ("{not json " + WARN_MARKER, "state_corrupt"),
    (json.dumps({"schema_version": 99, "note": WARN_MARKER}), "state_wrong_version"),
    (json.dumps({"schema_version": 1, "history": WARN_MARKER}), "state_corrupt"),
    (json.dumps([WARN_MARKER]), "state_corrupt"),
    (b"\xff\xfe\x00 " + WARN_MARKER.encode(), "state_corrupt"),  # not valid UTF-8
])
def test_bad_blob_gives_a_fresh_session_and_a_warning_without_contents(server, caplog, blob, event):
    store, session = make(server), sid()
    store._redis.set(store._key(session), blob)
    before = events()
    caplog.set_level(logging.DEBUG)

    async def main():
        async with store.turn(session, "hi") as st:
            assert st == LLMConversationState(updated_at=st.updated_at)
            st.record_turn("hi", "ok")

    asyncio.run(main())
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings and WARN_MARKER not in caplog.text
    assert delta(before, event) == 1
    assert make(server).peek(session).turn_count == 1  # the bad blob was replaced
    assert store.peek("not-a-uuid") is None


def test_peek_of_a_bad_blob_is_none_with_a_warning(server, caplog):
    store, session = make(server), sid()
    store._redis.set(store._key(session), "nonsense " + WARN_MARKER)
    with caplog.at_level(logging.WARNING):
        assert store.peek(session) is None
    assert any(r.levelno == logging.WARNING for r in caplog.records) and WARN_MARKER not in caplog.text


# ---- Redis down ------------------------------------------------------------------------------------


def test_turn_answers_from_a_fresh_state_when_redis_is_down(pipe, server, caplog):
    pipe.store = make(server)
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    saved_turns = pipe.state.turn_count
    server.connected = False
    before = events()
    caplog.set_level(logging.ERROR)
    r = pipe.turn("hello", dec("GENERAL_CHAT", reply="Hello!"))
    assert r.success is True and r.reply and r.metrics["error_class"] is None
    assert delta(before, "redis_unavailable") >= 1
    assert any(rec.levelno == logging.ERROR for rec in caplog.records)
    assert "redis://" not in caplog.text
    server.connected = True
    assert pipe.state.turn_count == saved_turns  # the fresh-state turn saved nothing


def test_redis_back_mid_turn_does_not_overwrite_the_real_session(server):
    store, session = make(server), sid()
    store.save(session, populated())
    server.connected = False

    async def main():
        async with store.turn(session, "hi") as st:
            assert st.turn_count == 0  # fresh state for this turn
            server.connected = True  # Redis returns while the turn runs
            st.record_turn("hi", "ok")

    asyncio.run(main())
    assert store.peek(session) == populated().__class__.from_dict(populated().to_dict())  # unchanged


def test_redis_down_during_save_still_returns_the_reply(server):
    store, session = make(server), sid()
    before = events()

    async def main():
        async with store.turn(session, "hi") as st:
            st.record_turn("hi", "ok")
            server.connected = False  # drops before the save

    asyncio.run(main())
    server.connected = True
    assert delta(before, "redis_unavailable") >= 1 and store.peek(session) is None


def test_reads_and_resets_with_redis_down_do_not_raise(server):
    store, session = make(server), sid()
    server.connected = False
    assert store.peek(session) is None and store.count() == 0 and store.reset(session) is False
    assert isinstance(store.get_or_create(session), LLMConversationState) and store.ping() is False
    server.connected = True
    assert store.ping() is True


def test_unreachable_redis_with_the_real_client_does_not_hang_or_leak_the_url(caplog):
    caplog.set_level(logging.ERROR)
    store = ss.RedisSessionStore(url="redis://:pw-SECRET@127.0.0.1:1/0", lock_wait_s=1.0)

    async def main():
        async with store.turn(sid()) as st:
            st.record_turn("hi", "ok")

    started = time.monotonic()
    asyncio.run(main())
    assert time.monotonic() - started < 5
    assert "pw-SECRET" not in caplog.text and "127.0.0.1" not in caplog.text


# ---- TTL and cap ------------------------------------------------------------------------------------


def test_idle_session_expires_by_the_ttl(server):
    now = [1000.0]
    store = make(server, ttl_seconds=60, clock=lambda: now[0])
    a, b = sid(), sid()
    store.save(a, populated())
    store.save(b, populated())
    now[0] += 30
    store.get_or_create(b)  # refreshes b only
    now[0] += 45
    assert store.count() == 1 and store.peek(a) is None and store.peek(b) is not None

    async def main():
        async with store.turn(a, "hi") as st:
            assert st.turn_count == 0  # expired: a new session

    asyncio.run(main())


def test_the_redis_key_itself_expires(server):
    store, session = make(server, ttl_seconds=3600), sid()
    store.save(session, populated())
    assert 0 < store._redis.ttl(store._key(session)) <= 3600
    store._redis.pexpire(store._key(session), 30)
    time.sleep(0.12)
    assert store.peek(session) is None


def test_lru_cap_drops_the_least_recently_used(server):
    now = [1000.0]
    store = make(server, max_sessions=2, clock=lambda: now[0])
    a, b, c = sid(), sid(), sid()
    for s in (a, b):
        store.save(s, populated())
        now[0] += 1
    store.get_or_create(a)  # a is now newer than b
    now[0] += 1
    store.save(c, populated())
    assert store.count() == 2 and store.peek(b) is None and store.peek(a) and store.peek(c)


def test_v2_never_touches_app_sessions(server):
    raw = client_for(server)
    app_sid = sid()
    raw.set(f"homilivo:session:{app_sid}", '{"app": "state"}')
    raw.zadd("homilivo:sessions:lru", {app_sid: 1.0})  # ancient: v2 would evict it if it shared the sorted set
    store = make(server, max_sessions=1, ttl_seconds=60)
    store.save(sid(), populated())
    store.save(sid(), populated())
    store.reset(app_sid)
    assert store.count() == 1
    assert raw.get(f"homilivo:session:{app_sid}") == '{"app": "state"}'
    assert raw.zscore("homilivo:sessions:lru", app_sid) == 1.0
    assert all(k.startswith("homilivo:v2:") or k.startswith("homilivo:session")
               or k.startswith("homilivo:sessions") for k in raw.keys("*"))
    assert store._KEY_PREFIX.startswith("homilivo:v2:") and store._LRU_KEY.startswith("homilivo:v2:")


# ---- the lock ------------------------------------------------------------------------------------------


def test_lock_timeout_raises_busy_and_counts(server):
    holder, other, session = make(server), make(server, lock_wait_s=0.2), sid()
    before = events()

    async def main():
        async with holder.turn(session):
            with pytest.raises(SessionBusyError):
                async with other.turn(session):
                    raise AssertionError("must not run")
        async with other.turn(session):  # free again
            pass

    asyncio.run(main())
    assert delta(before, "lock_timeout") == 1


def test_in_process_lock_wait_is_bounded_too(server):
    store, session = make(server, lock_wait_s=0.2), sid()
    before = events()

    async def main():
        async with store.turn(session):
            with pytest.raises(SessionBusyError):
                async with store.turn(session):
                    raise AssertionError("must not run")

    asyncio.run(main())
    assert delta(before, "lock_timeout") == 1 and store._locals == {}


def test_pipeline_answers_busy_without_calling_the_llm(pipe, server):
    holder = make(server)
    pipe.store = make(server, lock_wait_s=0.2)

    async def main():
        async with holder.turn(pipe.sid):
            return await pipe.turn_async("rooms in bangalore", "never")

    r = asyncio.run(main())
    assert r.reply == pipeline.RATE_LIMIT_REPLY and r.success is False
    assert r.metrics["action"] == "session_busy" and r.metrics["error_class"] == "SessionBusy"
    assert pipe.llm_calls == 0 and pipe.searches == []
    assert pipe.store.peek(pipe.sid).query_count == 0  # the busy turn wrote nothing


def test_release_is_by_token(server):
    store, session = make(server), sid()
    store._redis.set(store._lock_key(session), "someone-elses-token", px=5000)
    assert store._release_blocking(session, "my-token") is False
    assert store._redis.get(store._lock_key(session)) == "someone-elses-token"
    assert store._release_blocking(session, "someone-elses-token") is True
    assert store._redis.get(store._lock_key(session)) is None


def test_a_turn_does_not_release_a_lock_that_was_taken_over(server):
    store, session = make(server, lock_ttl_ms=5000), sid()

    async def main():
        async with store.turn(session):
            store._redis.set(store._lock_key(session), "taken-over", px=5000)  # e.g. Redis was flushed and another server won

    asyncio.run(main())
    assert store._redis.get(store._lock_key(session)) == "taken-over"


def test_a_crashed_servers_lock_lapses_after_its_ttl(server):
    crashed = make(server, lock_ttl_ms=150)
    session = sid()
    assert crashed._try_lock_blocking(session, "dead-server-token")  # never released
    waiter = make(server, lock_ttl_ms=150, lock_wait_s=2.0)

    async def main():
        started = time.monotonic()
        async with waiter.turn(session):
            return time.monotonic() - started

    waited = asyncio.run(main())
    assert 0.05 < waited < 1.5


def test_the_lock_is_refreshed_while_a_long_turn_runs(server):
    """The lock lives 150 ms; the turn takes 600 ms (a stand-in for a 429 wait). Nobody gets in meanwhile."""
    long_turn = make(server, lock_ttl_ms=150, lock_max_hold_s=30)
    rival = make(server, lock_ttl_ms=150, lock_wait_s=0.1)
    session = sid()
    before = events()

    async def main():
        async with long_turn.turn(session):
            for _ in range(4):
                await asyncio.sleep(0.15)
                with pytest.raises(SessionBusyError):  # well past the first 150 ms: still locked
                    async with rival.turn(session):
                        raise AssertionError("must not run")
        async with rival.turn(session):  # released at the end
            pass

    asyncio.run(main())
    assert delta(before, "lock_lost") == 0 and delta(before, "lock_hold_cap") == 0


def test_the_refresh_stops_at_the_max_hold(server):
    store = make(server, lock_ttl_ms=90, lock_max_hold_s=0.15)
    rival = make(server, lock_ttl_ms=90, lock_wait_s=0.1)
    session = sid()
    before = events()

    async def main():
        async with store.turn(session):
            await asyncio.sleep(0.5)  # a hung turn: the lock lapses instead of being held for ever
            async with rival.turn(session):
                pass

    asyncio.run(main())
    assert delta(before, "lock_hold_cap") == 1


def test_a_lost_lock_is_counted_and_the_turn_still_saves(server, caplog):
    store, session = make(server, lock_ttl_ms=90), sid()
    before = events()
    caplog.set_level(logging.WARNING)

    async def main():
        async with store.turn(session, "hi") as st:
            store._redis.delete(store._lock_key(session))  # Redis lost the lock (flush, failover)
            await asyncio.sleep(0.2)
            st.record_turn("hi", "ok")

    asyncio.run(main())
    assert delta(before, "lock_lost") == 1 and make(server).peek(session).turn_count == 1


def test_slow_429_wait_in_the_pipeline_keeps_the_session_locked(pipe, server):
    """A decision that takes longer than the lock TTL, through the real pipeline."""
    pipe.store = make(server, lock_ttl_ms=150, lock_wait_s=0.1)
    rival = make(server, lock_ttl_ms=150, lock_wait_s=0.1)
    outcome = {}

    async def slow():
        await asyncio.sleep(0.5)  # a limiter wait
        return dec("GENERAL_CHAT", reply="Hello!")

    async def main():
        task = asyncio.create_task(pipe.turn_async("hello there", slow))
        await asyncio.sleep(0.35)
        try:
            async with rival.turn(pipe.sid):
                outcome["rival"] = "got in"
        except SessionBusyError:
            outcome["rival"] = "busy"
        return await task

    r = asyncio.run(main())
    assert outcome["rival"] == "busy" and r.success


def test_too_long_message_is_logged_through_the_lock(pipe, server):
    pipe.store = make(server)
    r = pipe.turn("x" * 600, "never")
    assert r.reply == pipeline.TOO_LONG_REPLY
    st = pipe.store.peek(pipe.sid)
    assert st is not None and st.query_count == 1 and st.user_queries[0]["intent"] == "BLOCKED"


# ---- factory and config --------------------------------------------------------------------------------


def test_default_backend_is_the_unchanged_in_memory_store(monkeypatch):
    monkeypatch.delenv("STATE_BACKEND", raising=False)
    monkeypatch.setattr(pipeline, "_store", None)
    store = pipeline.get_store()
    assert type(store) is SessionStore and store.is_shared is False
    monkeypatch.setenv("STATE_BACKEND", "memory")
    assert type(ss.make_store()) is SessionStore


def test_state_backend_redis_selects_the_redis_store(monkeypatch, server):
    monkeypatch.setenv("STATE_BACKEND", " Redis ")
    store = ss.make_store(redis_client=client_for(server))
    assert isinstance(store, ss.RedisSessionStore) and store.is_shared is True
    assert store._ttl == config.SESSION_TTL_SECONDS == 3600 and store._max == config.SESSION_MAX_COUNT


def test_unknown_backend_is_an_error_that_does_not_echo_the_value(monkeypatch):
    monkeypatch.setenv("STATE_BACKEND", "rediss-typo-VALUE")
    with pytest.raises(ValueError) as err:
        ss.make_store()
    assert "rediss-typo-VALUE" not in str(err.value)


def test_lock_settings_come_from_the_environment_with_safe_fallbacks(monkeypatch):
    assert (config.session_lock_ttl_ms(), config.session_lock_wait_s(), config.session_lock_max_hold_s()) == (
        30000, 30.0, 120.0)
    monkeypatch.setenv(config.LOCK_TTL_ENV, "5000")
    monkeypatch.setenv(config.LOCK_WAIT_ENV, "2.5")
    monkeypatch.setenv(config.LOCK_MAX_HOLD_ENV, "-1")
    assert (config.session_lock_ttl_ms(), config.session_lock_wait_s(), config.session_lock_max_hold_s()) == (
        5000, 2.5, 120.0)
    monkeypatch.setenv(config.LOCK_TTL_ENV, "junk")
    assert config.session_lock_ttl_ms() == 30000
    monkeypatch.setenv("REDIS_URL", "redis://:pw@host:6380/2")
    assert config.redis_url() == "redis://:pw@host:6380/2"
    monkeypatch.delenv("REDIS_URL")
    assert config.redis_url() == "redis://localhost:6379/0"


# ---- the server -----------------------------------------------------------------------------------------


@pytest.fixture
def redis_client(monkeypatch, server):
    from llm_approach.security.rate_limit import InMemoryRateLimiter

    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")
    monkeypatch.setenv("STREAM_CHUNK_DELAY_SECONDS", "0")
    monkeypatch.setenv("KANGAROO_TOKEN", "fake-rooms-credential")
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    limiter = InMemoryRateLimiter()
    monkeypatch.setattr(server_app._rl, "get_rate_limiter", lambda: limiter)
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    monkeypatch.setattr(pipeline, "_store", make(server))
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": True, "knowledge_base": True})
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_value_must_not_appear")
    with TestClient(server_app.create_app()) as c:
        yield c
    pipeline.configure(full_guardrails=True)


def test_ready_reports_the_redis_state(redis_client, server):
    body = redis_client.get("/ready")
    assert body.status_code == 200
    assert body.json()["session_store"] == {"backend": "redis", "redis_reachable": True, "sessions_survive_restart": True}
    server.connected = False
    down = redis_client.get("/ready")
    assert down.status_code == 200  # reported, not required: turns still answer
    assert down.json()["session_store"]["redis_reachable"] is False
    assert "gsk_value_must_not_appear" not in down.text and "redis://" not in down.text


def test_ready_for_the_memory_store_is_unchanged(monkeypatch):
    monkeypatch.setattr(pipeline, "_store", SessionStore())
    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": True, "knowledge_base": True})
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    with TestClient(server_app.create_app()) as c:
        body = c.get("/ready").json()
    assert set(body) == {"status", "checks", "provider", "model"}


def test_chat_endpoint_with_the_redis_store_reads_the_state_from_the_turn(redis_client, server, monkeypatch):
    session = sid()

    async def fake_decide(user_message, state_summary, budget):
        from llm_approach.schema import parse_decision

        return parse_decision(dec("NEW_SEARCH", city="Bangalore"))

    from llm_approach import llm_client

    monkeypatch.setattr(llm_client, "decide", fake_decide)
    reads = []
    store = pipeline.get_store()
    real_peek = store.peek
    monkeypatch.setattr(store, "peek", lambda s: reads.append(s) or real_peek(s))
    body = redis_client.post("/chat", json={"message": "rooms in bangalore", "session_id": session},
                             headers={"Authorization": "Bearer x"}).json()
    assert body["success"] and body["room_count"] > 0 and reads == []  # no second store read for the reply
    assert real_peek(session).rooms_on_screen
    assert redis_client.post(f"/chat/reset?session_id={session}").json()["existed"] is True
    assert real_peek(session) is None


def test_startup_refuses_a_bad_backend(monkeypatch):
    monkeypatch.setattr(pipeline, "_store", None)
    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")
    monkeypatch.setenv("STATE_BACKEND", "mongo")
    with pytest.raises(ValueError):
        with TestClient(server_app.create_app()):
            pass
