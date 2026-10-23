"""Phase 2 state and store tests. Offline, no sleeping, no torch."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from llm_approach import config, prompts
from llm_approach.state import (
    ERROR_REPLY,
    ROOM_ALLOWED_FIELDS,
    LLMConversationState,
    SessionStore,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def sid() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(clock) -> SessionStore:
    return SessionStore(ttl_seconds=100, max_sessions=3, clock=clock)


def search(city="Bangalore", **kw):
    return {"city": city, "bhk": None, "min_price": None, "max_price": kw.get("max_price"),
            "room_types": None, "ideal_for": None}


# ---- store: TTL, LRU, validation ---------------------------------------


def test_ttl_expiry(store, clock):
    a = sid()
    state = store.get_or_create(a)
    state.record_turn("hi", "hello")
    clock.now += 99
    assert store.get_or_create(a) is state  # access refreshes TTL
    clock.now += 101
    assert store.peek(a) is None
    assert store.count() == 0
    assert store.get_or_create(a) is not state


def test_lru_eviction(store, clock):
    a, b, c, d = sid(), sid(), sid(), sid()
    for s in (a, b, c):
        store.get_or_create(s)
        clock.now += 1
    store.get_or_create(a)  # a becomes most recent, b is now oldest
    store.get_or_create(d)
    assert store.count() == 3
    assert store.peek(b) is None
    assert store.peek(a) is not None and store.peek(c) is not None and store.peek(d) is not None


def test_defaults_come_from_config():
    s = SessionStore()
    assert s.ttl_seconds == config.SESSION_TTL_SECONDS
    assert s.max_sessions == config.SESSION_MAX_COUNT


def test_peek_does_not_create_or_refresh(store, clock):
    a = sid()
    assert store.peek(a) is None
    assert store.count() == 0
    store.get_or_create(a)
    clock.now += 60
    assert store.peek(a) is not None
    clock.now += 60  # 120s since last real access; peek at 60s must not have refreshed
    assert store.peek(a) is None


def test_peek_does_not_change_lru_order(store, clock):
    a, b, c, d = sid(), sid(), sid(), sid()
    for s in (a, b, c):
        store.get_or_create(s)
    store.peek(a)
    store.get_or_create(d)
    assert store.peek(a) is None  # a was still the oldest


@pytest.mark.parametrize("bad", ["", "abc", "1234", None, 123, "../etc/passwd",
                                 "12345678123456781234567812345678",
                                 "{12345678-1234-5678-1234-567812345678}",
                                 "gggggggg-gggg-gggg-gggg-gggggggggggg"])
def test_invalid_session_id_rejected(store, bad):
    with pytest.raises(ValueError, match="UUID"):
        store.get_or_create(bad)
    with pytest.raises(ValueError):
        store.reset(bad)
    assert store.peek(bad) is None
    assert store.count() == 0


def test_uuid_case_insensitive_and_normalised(store):
    a = sid()
    assert store.get_or_create(a.upper()) is store.get_or_create(a)
    assert store.count() == 1


def test_reset_and_count(store):
    a = sid()
    store.get_or_create(a)
    assert store.count() == 1
    assert store.reset(a) is True
    assert store.reset(a) is False
    assert store.count() == 0


def test_expired_sessions_and_locks_are_cleaned(store, clock):
    ids = [sid() for _ in range(3)]

    async def run():
        for s in ids:
            async with store.turn(s) as st:
                st.record_turn("q", "a")

    asyncio.run(run())
    clock.now += 1000
    assert store.count() == 0
    assert store._entries == {}  # entries (and their locks) are gone


# ---- state --------------------------------------------------------------


def test_history_trim_to_cap(monkeypatch):
    monkeypatch.setattr(config, "SESSION_MAX_HISTORY_MESSAGES", 6)
    st = LLMConversationState()
    for i in range(10):
        st.record_turn(f"u{i}", f"b{i}")
    assert len(st.history) == 6
    assert [m["content"] for m in st.history] == ["u7", "b7", "u8", "b8", "u9", "b9"]
    assert st.history[0]["role"] == "user"
    assert st.last_bot_message == "b9"
    assert st.turn_count == 10


def test_default_history_cap_is_config_value():
    st = LLMConversationState()
    for i in range(config.SESSION_MAX_HISTORY_MESSAGES):
        st.record_turn(f"u{i}", f"b{i}")
    assert len(st.history) == config.SESSION_MAX_HISTORY_MESSAGES


def test_selection_survives_non_selecting_turn_and_clears_on_new_search(rooms):
    st = LLMConversationState()
    st.apply_search_result(search(), rooms[:3])
    st.select_room(2)
    st.record_turn("how much is it", "It is 15000")
    st.record_turn("thanks", "welcome")
    assert st.selected_room_index == 2
    st.select_room(3)
    assert st.selected_room_index == 3
    st.apply_search_result(search("Pune"), rooms[3:6])
    assert st.selected_room_index is None


def test_select_room_validates_range(rooms):
    st = LLMConversationState()
    st.apply_search_result(search(), rooms[:2])
    for bad in (0, 3, -1, True, "1", None):
        with pytest.raises(ValueError):
            st.select_room(bad)
    st.select_room(1)
    st.clear_selection()
    assert st.selected_room_index is None


def test_new_search_clears_selection_pending_and_rejected(rooms):
    st = LLMConversationState()
    st.apply_search_result(search(), rooms[:3])
    st.select_room(1)
    st.set_pending("reject", {"x": 1})
    st.reject_rooms()
    assert len(st.rejected_room_ids) == 3
    assert st.apply_search_result(search("Pune"), rooms[3:5]) is True
    assert st.selected_room_index is None
    assert st.pending_question is None
    assert st.rejected_room_ids == []
    assert st.current_search["city"] == "Pune"


def test_failed_search_keeps_previous_state(rooms):
    st = LLMConversationState()
    st.apply_search_result(search(), rooms[:3])
    st.select_room(2)
    st.set_pending("clarify", "which area")
    st.reject_rooms()
    before = (list(st.rooms_on_screen), dict(st.current_search), list(st.rejected_room_ids))
    assert st.apply_search_result(search("Pune"), None) is False
    assert st.rooms_on_screen == before[0]
    assert st.current_search == before[1]
    assert st.rejected_room_ids == before[2]
    assert st.selected_room_index == 2
    assert st.pending_question == {"type": "clarify", "context": "which area"}
    assert st.selected_room() == st.rooms_on_screen[1]


def test_empty_search_result_replaces_rooms(rooms):
    st = LLMConversationState()
    st.apply_search_result(search(), rooms[:3])
    st.apply_search_result(search("Udupi"), [])
    assert st.rooms_on_screen == []
    assert st.selected_room() is None


def test_pending_question_set_and_clear():
    st = LLMConversationState()
    st.set_pending("city", {"partial": 1})
    assert st.pending_question == {"type": "city", "context": {"partial": 1}}
    st.clear_pending()
    assert st.pending_question is None
    with pytest.raises(ValueError):
        st.set_pending("bogus")


def test_rooms_hold_only_allowlisted_fields(rooms):
    dirty = dict(rooms[0])
    dirty.update({
        "owner": {"name": "X", "phone": "9999999999"},
        "ownerPhone": "9999999999",
        "contactNumber": "1",
        "email": "a@b.c",
        "latitude": 12.9,
        "longitude": 77.6,
        "coordinates": [1, 2],
        "whatsapp": "1",
    })
    st = LLMConversationState()
    st.apply_search_result(search(), [dirty, "not a dict"])
    assert len(st.rooms_on_screen) == 1
    stored = st.rooms_on_screen[0]
    assert set(stored) <= ROOM_ALLOWED_FIELDS
    assert stored["title"] == dirty["title"]
    for banned in ("owner", "ownerPhone", "contactNumber", "email", "latitude",
                   "longitude", "coordinates", "whatsapp"):
        assert banned not in stored
    assert "9999999999" not in str(st.to_prompt_state())
    assert "owner" in dirty  # input was not mutated


def test_rooms_are_sanitised(rooms):
    dirty = dict(rooms[0], title="Nice\x00 room\x07    here")
    st = LLMConversationState()
    st.apply_search_result(search(), [dirty])
    assert "\x00" not in st.rooms_on_screen[0]["title"]
    assert "\x07" not in st.rooms_on_screen[0]["title"]


# ---- prompt state ---------------------------------------------------------

PROMPT_KEYS = {"current_search", "pending_question", "last_bot_message",
               "rooms_on_screen", "selected_room_index", "recent_turns", "user_facts"}


def test_to_prompt_state_keys_match_build_state_summary(rooms):
    st = LLMConversationState()
    st.apply_search_result(search(max_price=15000), rooms[:3])
    st.select_room(2)
    st.set_pending("city")
    st.record_turn("find rooms", "Here are the rooms I found.")
    state = st.to_prompt_state()
    assert set(state) == PROMPT_KEYS
    summary = prompts.build_state_summary(state)
    assert set(summary) == PROMPT_KEYS
    assert summary["current_search"] == {"city": "Bangalore", "max_price": 15000}
    assert summary["pending_question"] == "city"
    assert summary["selected_room_index"] == 2
    assert summary["last_bot_message"] == "Here are the rooms I found."
    assert [r["index"] for r in summary["rooms_on_screen"]] == [1, 2, 3]
    assert summary["rooms_on_screen"][0]["title"] == rooms[0]["title"]
    assert summary["recent_turns"] == st.history
    # and the messages can be built from it
    assert prompts.build_messages("which is cheapest", summary)[1]["role"] == "user"


def test_to_prompt_state_empty_state():
    state = LLMConversationState().to_prompt_state()
    assert set(state) == PROMPT_KEYS
    summary = prompts.build_state_summary(state)
    assert summary["current_search"] is None
    assert summary["rooms_on_screen"] == []
    assert summary["selected_room_index"] is None


def test_to_prompt_state_pending_with_context():
    st = LLMConversationState()
    st.set_pending("clarify", "which area")
    assert st.to_prompt_state()["pending_question"] == {"type": "clarify", "context": "which area"}


def test_to_prompt_state_is_a_copy(rooms):
    st = LLMConversationState()
    st.apply_search_result(search(), rooms[:2])
    out = st.to_prompt_state()
    out["rooms_on_screen"][0]["title"] = "changed"
    out["recent_turns"].append({"role": "user", "content": "x"})
    assert st.rooms_on_screen[0]["title"] != "changed"
    assert st.history == []


# ---- concurrency ----------------------------------------------------------


def test_concurrent_turns_on_one_session_are_serialised(store):
    a = sid()
    events: list[str] = []

    async def turn(name: str, gate: asyncio.Event | None, release: asyncio.Event | None):
        async with store.turn(a, name) as st:
            events.append(f"{name}:start")
            if release:
                release.set()
            if gate:
                await gate.wait()
            await asyncio.sleep(0)
            events.append(f"{name}:end")
            st.record_turn(name, f"reply-{name}")

    async def main():
        gate, started = asyncio.Event(), asyncio.Event()
        t1 = asyncio.create_task(turn("one", gate, started))
        await started.wait()  # turn one holds the lock
        t2 = asyncio.create_task(turn("two", None, None))
        for _ in range(5):
            await asyncio.sleep(0)
        assert events == ["one:start"]  # two is waiting, not running
        gate.set()
        await asyncio.gather(t1, t2)

    asyncio.run(main())
    assert events == ["one:start", "one:end", "two:start", "two:end"]
    history = store.peek(a).history
    assert [(m["role"], m["content"]) for m in history] == [
        ("user", "one"), ("assistant", "reply-one"),
        ("user", "two"), ("assistant", "reply-two"),
    ]


def test_many_overlapping_turns_keep_history_pairs(store):
    a = sid()

    async def turn(i: int):
        async with store.turn(a, f"q{i}") as st:
            await asyncio.sleep(0)
            st.record_turn(f"q{i}", f"a{i}")

    async def main():
        await asyncio.gather(*(turn(i) for i in range(10)))

    asyncio.run(main())
    history = store.peek(a).history
    assert len(history) == 20
    for k in range(0, 20, 2):
        assert history[k]["role"] == "user" and history[k + 1]["role"] == "assistant"
        assert history[k]["content"][1:] == history[k + 1]["content"][1:]
    assert [m["content"] for m in history[::2]] == [f"q{i}" for i in range(10)]  # arrival order


def test_different_sessions_run_concurrently():
    store = SessionStore(max_sessions=10)
    a, b = sid(), sid()
    both_inside = asyncio.Event()
    inside: list[str] = []

    async def turn(s: str):
        async with store.turn(s) as st:
            inside.append(s)
            if len(inside) == 2:
                both_inside.set()
            # would deadlock (timeout) if sessions were serialised
            await asyncio.wait_for(both_inside.wait(), timeout=2)
            st.record_turn("q", "a")

    async def main():
        await asyncio.gather(turn(a), turn(b))

    asyncio.run(main())
    assert sorted(inside) == sorted([a, b])


def test_failed_turn_records_user_message_and_error_reply(store):
    a = sid()

    async def main():
        async with store.turn(a, "find rooms in pune"):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        asyncio.run(main())
    st = store.peek(a)
    assert st.history == [
        {"role": "user", "content": "find rooms in pune"},
        {"role": "assistant", "content": ERROR_REPLY},
    ]
    assert st.last_bot_message == ERROR_REPLY
    assert st.turn_count == 1


def test_failed_turn_after_record_does_not_double_record(store):
    a = sid()

    async def main():
        async with store.turn(a, "hi") as st:
            st.record_turn("hi", "hello")
            raise RuntimeError("late failure")

    with pytest.raises(RuntimeError):
        asyncio.run(main())
    assert len(store.peek(a).history) == 2


def test_failed_turn_without_message_leaves_history_untouched(store):
    a = sid()

    async def main():
        async with store.turn(a):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        asyncio.run(main())
    assert store.peek(a).history == []


def test_lock_released_on_exception_and_next_turn_runs(store):
    a = sid()

    async def failing():
        async with store.turn(a, "bad"):
            raise ValueError("x")

    async def ok():
        async with store.turn(a, "good") as st:
            st.record_turn("good", "fine")

    async def main():
        with pytest.raises(ValueError):
            await failing()
        await asyncio.wait_for(ok(), timeout=2)  # hangs if the lock leaked
        assert not store._entries[a].lock.locked()
        assert store._entries[a].active == 0

    asyncio.run(main())
    assert [m["content"] for m in store.peek(a).history] == ["bad", ERROR_REPLY, "good", "fine"]


def test_lock_released_when_waiting_turn_is_cancelled(store):
    a = sid()

    async def main():
        gate, started = asyncio.Event(), asyncio.Event()

        async def holder():
            async with store.turn(a, "h") as st:
                started.set()
                await gate.wait()
                st.record_turn("h", "r")

        async def waiter():
            async with store.turn(a, "w"):
                pass

        t1 = asyncio.create_task(holder())
        await started.wait()
        t2 = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        t2.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t2
        gate.set()
        await t1
        assert store._entries[a].active == 0
        assert not store._entries[a].lock.locked()

    asyncio.run(main())
    assert [m["content"] for m in store.peek(a).history] == ["h", "r"]


def test_busy_session_is_not_evicted_or_expired(clock):
    store = SessionStore(ttl_seconds=10, max_sessions=1, clock=clock)
    a, b = sid(), sid()

    async def main():
        async with store.turn(a) as st:
            clock.now += 1000  # long turn, well past the TTL
            store.get_or_create(b)  # over the cap; a is busy so it stays
            assert store.peek(a) is st
            st.record_turn("q", "a")

    asyncio.run(main())
    assert store.peek(a).turn_count == 1


def test_turn_rejects_invalid_session_id(store):
    async def main():
        async with store.turn("not-a-uuid"):
            pass

    with pytest.raises(ValueError, match="UUID"):
        asyncio.run(main())
    assert store.count() == 0
