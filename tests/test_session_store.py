"""
Tests for app/session_store.py: TTL expiry, session-count cap eviction,
conversation-history trimming, and concurrent-session isolation.

Session isolation through the real /chat endpoint (a brand new session
never sees another session's rooms/selection) is already covered end-
to-end in tests/test_api.py::test_chat_session_isolation - these tests
exercise SessionStore directly so TTL/cap/history behavior can be
tested with a controlled clock instead of real sleeps.

Every behavioral test below is parametrized over BOTH backends
(memory, and RedisSessionStore backed by fakeredis - a pure-Python,
in-memory Redis-protocol server) via the `store_factory` fixture, so
the Redis code path gets the exact same coverage as the in-memory one
without requiring a live Redis server (see
tests/test_state_backend_smoke.py for a real-Redis integration check
that skips cleanly when Redis is unavailable).

The Redis backend always hands back a freshly deserialized
ConversationState from `get()`, never the same Python object a caller
last mutated - a real, correct difference from the in-memory backend's
object-cache behavior, not a bug. So the handful of tests below that
used to assert Python object IDENTITY (`fetched is state`) instead
assert content equality, which is the actual cross-backend contract;
test_memory_backend_reuses_the_same_object at the bottom keeps the
identity behavior covered for the backend that actually provides it.
"""

from __future__ import annotations

import threading

import fakeredis
import pytest

from app.experiments.agent_modes import ConversationState
from app.session_store import SessionStore


@pytest.fixture(params=["memory", "redis"])
def store_factory(request):
    """
    Returns a callable that builds a fresh SessionStore for the
    parametrized backend. All stores built by one call to this fixture
    within a single test share one fakeredis FakeServer, so they behave
    like multiple app instances pointed at the same Redis - this
    catches the concurrent-safety test below.
    """
    backend = request.param
    fake_server = fakeredis.FakeServer()

    def _make(**kwargs):
        if backend == "redis":
            client = fakeredis.FakeStrictRedis(server=fake_server, decode_responses=True)
            return SessionStore(backend="redis", redis_client=client, **kwargs)
        return SessionStore(backend="memory", **kwargs)

    return _make


def test_get_creates_a_new_state_for_an_unseen_session_id(store_factory):
    store = store_factory(ttl_seconds=60, max_sessions=10)

    state = store.get("session-a")

    assert isinstance(state, ConversationState)
    assert state.conversation_history == []


def test_save_then_get_returns_the_same_state(store_factory):
    store = store_factory(ttl_seconds=60, max_sessions=10)

    state = store.get("session-a")
    state.conversation_history.append({"role": "user", "content": "hi"})
    store.save("session-a", state)

    fetched = store.get("session-a")
    assert fetched.conversation_history == [{"role": "user", "content": "hi"}]


def test_sessions_are_isolated_from_each_other(store_factory):
    store = store_factory(ttl_seconds=60, max_sessions=10)

    state_a = store.get("session-a")
    state_a.selected_room = {"id": "room-1"}
    store.save("session-a", state_a)

    state_b = store.get("session-b")

    assert state_b.selected_room is None
    assert state_b is not state_a


def test_expired_session_is_evicted_on_access(store_factory, monkeypatch):
    store = store_factory(ttl_seconds=10, max_sessions=10)

    fake_now = [1000.0]
    monkeypatch.setattr("app.session_store.time.time", lambda: fake_now[0])

    original = store.get("session-a")
    original.selected_room = {"id": "room-1"}
    store.save("session-a", original)

    fake_now[0] += 11  # past the 10s TTL

    refreshed = store.get("session-a")

    assert refreshed.selected_room is None


def test_session_within_ttl_is_not_evicted(store_factory, monkeypatch):
    store = store_factory(ttl_seconds=100, max_sessions=10)

    fake_now = [1000.0]
    monkeypatch.setattr("app.session_store.time.time", lambda: fake_now[0])

    original = store.get("session-a")
    original.selected_room = {"id": "room-1"}
    store.save("session-a", original)

    fake_now[0] += 50  # within TTL

    refreshed = store.get("session-a")
    assert refreshed.selected_room == {"id": "room-1"}


def test_max_session_count_evicts_least_recently_used(store_factory, monkeypatch):
    store = store_factory(ttl_seconds=3600, max_sessions=3)

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


def test_conversation_history_is_trimmed_to_the_configured_limit(store_factory):
    store = store_factory(ttl_seconds=3600, max_sessions=10, max_history_messages=5)

    state = store.get("session-a")
    for i in range(20):
        state.conversation_history.append({"role": "user", "content": str(i)})
    store.save("session-a", state)

    fetched = store.get("session-a")
    assert len(fetched.conversation_history) == 5
    # Trimming keeps the MOST RECENT messages, not the oldest.
    assert fetched.conversation_history[-1]["content"] == "19"
    assert fetched.conversation_history[0]["content"] == "15"


def test_reset_removes_a_session(store_factory):
    store = store_factory(ttl_seconds=3600, max_sessions=10)
    store.save("session-a", store.get("session-a"))

    assert store.reset("session-a") is True
    assert store.reset("session-a") is False  # already gone


def test_concurrent_gets_and_saves_do_not_corrupt_the_store(store_factory):
    store = store_factory(ttl_seconds=3600, max_sessions=200)
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


# ============================================================
# MEMORY-BACKEND-SPECIFIC BEHAVIOR
# ============================================================

def test_memory_backend_reuses_the_same_object():
    """
    Unlike the Redis backend (which round-trips through JSON and always
    hands back a freshly deserialized object, see the module docstring
    above), the in-memory backend caches the live ConversationState
    object itself, so a caller's in-place mutations are visible to the
    next get() even without an intervening save(). This is a real
    implementation difference, not part of the interface's cross-
    backend contract (both backends satisfy the same content-equality
    guarantee tested above).
    """
    store = SessionStore(ttl_seconds=60, max_sessions=10, backend="memory")

    state = store.get("session-a")
    fetched = store.get("session-a")

    assert fetched is state
