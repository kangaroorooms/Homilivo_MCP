"""
Tests for app/session_store.py: TTL expiry, session-count cap eviction,
conversation-history trimming, and concurrent-session isolation.

Session isolation through the real /chat endpoint (a brand new session
never sees another session's rooms/selection) is already covered end-
to-end in tests/test_api.py::test_chat_session_isolation - these tests
exercise SessionStore directly so TTL/cap/history behavior can be
tested with a controlled clock instead of real sleeps.
"""

from __future__ import annotations

import threading

from app.experiments.agent_modes import ConversationState
from app.session_store import SessionStore


def test_get_creates_a_new_state_for_an_unseen_session_id():
    store = SessionStore(ttl_seconds=60, max_sessions=10)

    state = store.get("session-a")

    assert isinstance(state, ConversationState)
    assert state.conversation_history == []


def test_save_then_get_returns_the_same_state():
    store = SessionStore(ttl_seconds=60, max_sessions=10)

    state = store.get("session-a")
    state.conversation_history.append({"role": "user", "content": "hi"})
    store.save("session-a", state)

    fetched = store.get("session-a")
    assert fetched is state
    assert fetched.conversation_history == [{"role": "user", "content": "hi"}]


def test_sessions_are_isolated_from_each_other():
    store = SessionStore(ttl_seconds=60, max_sessions=10)

    state_a = store.get("session-a")
    state_a.selected_room = {"id": "room-1"}
    store.save("session-a", state_a)

    state_b = store.get("session-b")

    assert state_b.selected_room is None
    assert state_b is not state_a


def test_expired_session_is_evicted_on_access(monkeypatch):
    store = SessionStore(ttl_seconds=10, max_sessions=10)

    fake_now = [1000.0]
    monkeypatch.setattr("app.session_store.time.time", lambda: fake_now[0])

    original = store.get("session-a")
    original.selected_room = {"id": "room-1"}
    store.save("session-a", original)

    fake_now[0] += 11  # past the 10s TTL

    refreshed = store.get("session-a")

    assert refreshed is not original
    assert refreshed.selected_room is None


def test_session_within_ttl_is_not_evicted(monkeypatch):
    store = SessionStore(ttl_seconds=100, max_sessions=10)

    fake_now = [1000.0]
    monkeypatch.setattr("app.session_store.time.time", lambda: fake_now[0])

    original = store.get("session-a")
    store.save("session-a", original)

    fake_now[0] += 50  # within TTL

    assert store.get("session-a") is original


def test_max_session_count_evicts_least_recently_used(monkeypatch):
    store = SessionStore(ttl_seconds=3600, max_sessions=3)

    fake_now = [1000.0]
    monkeypatch.setattr("app.session_store.time.time", lambda: fake_now[0])

    for sid in ("s1", "s2", "s3"):
        store.save(sid, store.get(sid))
        fake_now[0] += 1

    # Touch s1 so it is no longer the least-recently-used.
    fake_now[0] += 1
    store.get("s1")

    # A 4th session pushes the store over its cap; s2 (never touched
    # again after creation) must be the one evicted, not s1 or s3.
    fake_now[0] += 1
    store.save("s4", store.get("s4"))

    assert len(store) == 3
    assert store.reset("s2") is False  # already evicted
    assert store.reset("s1") is True
    assert store.reset("s3") is True
    assert store.reset("s4") is True


def test_conversation_history_is_trimmed_to_the_configured_limit():
    store = SessionStore(ttl_seconds=3600, max_sessions=10, max_history_messages=5)

    state = store.get("session-a")
    for i in range(20):
        state.conversation_history.append({"role": "user", "content": str(i)})
    store.save("session-a", state)

    fetched = store.get("session-a")
    assert len(fetched.conversation_history) == 5
    # Trimming keeps the MOST RECENT messages, not the oldest.
    assert fetched.conversation_history[-1]["content"] == "19"
    assert fetched.conversation_history[0]["content"] == "15"


def test_reset_removes_a_session():
    store = SessionStore(ttl_seconds=3600, max_sessions=10)
    store.save("session-a", store.get("session-a"))

    assert store.reset("session-a") is True
    assert store.reset("session-a") is False  # already gone


def test_concurrent_gets_and_saves_do_not_corrupt_the_store():
    store = SessionStore(ttl_seconds=3600, max_sessions=200)
    errors: list[Exception] = []

    def worker(index: int) -> None:
        try:
            sid = f"session-{index}"
            state = store.get(sid)
            state.conversation_history.append({"role": "user", "content": "hi"})
            store.save(sid, state)
            assert store.get(sid).conversation_history
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(store) == 50
