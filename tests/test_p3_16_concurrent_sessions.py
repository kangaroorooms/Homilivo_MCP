# P3-16: Concurrent Session Behavior Tests
# Verifies session state never crosses between
# users/sessions under concurrent load.
# Tests cover simultaneous reads, writes, and
# mixed operations on same and different sessions.
#
# TEST-ONLY: no production code is changed by this file.
#
# BACKENDS: Tests 1, 2, 4, 5, 8 run against both InMemorySessionStore
# and RedisSessionStore (fakeredis). Tests 3 and 6 are memory-only
# (reads / new-session creation are not Redis-specific risks).
#
# KNOWN RISK (documented, not fixed here): the Redis backend has no
# locking on read-modify-write, so concurrent turns on the SAME session
# can lose updates (last write wins). See the xfail on Test 1 (redis)
# and the redis-specific relaxed assertion in Test 4.

from __future__ import annotations

import threading
import uuid

import fakeredis
import pytest
from fastapi.testclient import TestClient

import app.api as api_module
from app.experiments.agent_modes import ConversationState
from app.session_store import SessionStore


@pytest.fixture(params=["memory", "redis"])
def backend_store(request):
    if request.param == "redis":
        client = fakeredis.FakeStrictRedis(
            server=fakeredis.FakeServer(), decode_responses=True
        )
        store = SessionStore(
            backend="redis", redis_client=client, max_sessions=10_000
        )
    else:
        store = SessionStore(backend="memory", max_sessions=10_000)
    store.backend_name = request.param
    return store


@pytest.fixture()
def memory_store():
    return SessionStore(backend="memory", max_sessions=10_000)


def _run_threads(target, args_list) -> list[BaseException]:
    """Start one thread per args tuple behind a barrier (so they really
    overlap) and return any exceptions raised inside the threads."""
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(args_list))

    def wrapper(*args):
        try:
            barrier.wait(timeout=10)
            target(*args)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=wrapper, args=a) for a in args_list]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return errors


def _append_turn(store, sid: str, content: str) -> None:
    """Same pattern as app.api._run_chat_turn(): get, mutate, save."""
    state = store.get(sid)
    state.conversation_history.append({"role": "user", "content": content})
    store.save(sid, state)


def _contents(store, sid: str) -> list[str]:
    return [m["content"] for m in store.get(sid).conversation_history]


# ---------------------------------------------------------------- Test 1
@pytest.mark.parametrize(
    "backend",
    [
        "memory",
        pytest.param(
            "redis",
            marks=pytest.mark.xfail(
                strict=False,
                reason=(
                    "KNOWN RISK: Redis backend has no locking on "
                    "read-modify-write. Concurrent turns on the same "
                    "session may lose updates (last write wins). This is "
                    "a documented remaining risk requiring a Redis "
                    "pipeline or Lua script to fix properly."
                ),
            ),
        ),
    ],
)
def test_concurrent_writes_same_session_safe(backend):
    # KNOWN RISK: Redis backend has no locking on
    # read-modify-write. Concurrent turns on the
    # same session may lose updates (last write wins).
    # This is a documented remaining risk requiring
    # a Redis pipeline or Lua script to fix properly.
    if backend == "redis":
        store = SessionStore(
            backend="redis",
            redis_client=fakeredis.FakeStrictRedis(
                server=fakeredis.FakeServer(), decode_responses=True
            ),
        )
    else:
        store = SessionStore(backend="memory")

    sid = str(uuid.uuid4())
    messages = [f"msg-{i}" for i in range(10)]

    errors = _run_threads(_append_turn, [(store, sid, m) for m in messages])

    assert errors == []
    stored = _contents(store, sid)
    assert len(stored) == 10
    assert sorted(stored) == sorted(messages)  # none lost, none duplicated


# ---------------------------------------------------------------- Test 2
def test_concurrent_writes_different_sessions(backend_store):
    store = backend_store
    sids = [str(uuid.uuid4()) for _ in range(10)]

    def write(i: int, sid: str) -> None:
        _append_turn(store, sid, f"own-{i}-a")
        _append_turn(store, sid, f"own-{i}-b")

    errors = _run_threads(write, list(enumerate(sids)))

    assert errors == []
    for i, sid in enumerate(sids):
        assert _contents(store, sid) == [f"own-{i}-a", f"own-{i}-b"]


# ---------------------------------------------------------------- Test 3
def test_concurrent_reads_do_not_corrupt(memory_store):
    store = memory_store
    sid = str(uuid.uuid4())
    known_history = [{"role": "user", "content": f"known-{i}"} for i in range(5)]
    known_rooms = [{"title": "Room X"}]
    store.save(
        sid,
        ConversationState(
            conversation_history=list(known_history),
            previous_rooms=list(known_rooms),
        ),
    )

    results: list[tuple[list, list]] = []
    results_lock = threading.Lock()

    def read() -> None:
        state = store.get(sid)
        snapshot = (list(state.conversation_history), list(state.previous_rooms))
        with results_lock:
            results.append(snapshot)

    errors = _run_threads(read, [() for _ in range(20)])

    assert errors == []
    assert len(results) == 20
    assert all(r == (known_history, known_rooms) for r in results)
    final = store.get(sid)
    assert final.conversation_history == known_history
    assert final.previous_rooms == known_rooms
    assert len(store) == 1


# ---------------------------------------------------------------- Test 4
def test_session_isolation_under_load(backend_store):
    """Core isolation proof: 5 users x 10 requests = 50 threads."""
    store = backend_store
    users = {f"user{u}": str(uuid.uuid4()) for u in range(5)}
    args = [
        (store, sid, f"{user}-req{r}")
        for user, sid in users.items()
        for r in range(10)
    ]

    errors = _run_threads(_append_turn, args)

    assert errors == []
    for user, sid in users.items():
        stored = _contents(store, sid)
        # Isolation (all backends): only this user's messages, no dupes.
        assert all(c.startswith(f"{user}-req") for c in stored), (
            f"{user}'s session contains a foreign message: {stored}"
        )
        assert len(stored) == len(set(stored))
        if store.backend_name == "memory":
            assert sorted(stored) == sorted(f"{user}-req{r}" for r in range(10))
        else:
            # Redis: same-session lost updates are the documented known
            # risk (see Test 1), so only require that something landed.
            assert 1 <= len(stored) <= 10


# ---------------------------------------------------------------- Test 5
def test_concurrent_reset_and_write(backend_store):
    store = backend_store
    sid = str(uuid.uuid4())
    written = {f"w-{i}" for i in range(10)}

    def write(i: int) -> None:
        _append_turn(store, sid, f"w-{i}")

    def reset() -> None:
        store.reset(sid)

    args = [(write, (i,)) for i in range(10)] + [(reset, ()) for _ in range(5)]
    errors = _run_threads(lambda fn, a: fn(*a), args)

    assert errors == []
    stored = _contents(store, sid)
    # Empty (reset won) or a duplicate-free subset of the written messages
    # (writes after the last reset survived) - never foreign/corrupt data.
    assert len(stored) == len(set(stored))
    assert set(stored) <= written
    state = store.get(sid)
    assert isinstance(state, ConversationState)


# ---------------------------------------------------------------- Test 6
def test_new_session_created_safely_concurrent(memory_store):
    store = memory_store
    sids = [str(uuid.uuid4()) for _ in range(20)]
    assert len(set(sids)) == 20
    seen: dict[str, ConversationState] = {}
    seen_lock = threading.Lock()

    def create(sid: str) -> None:
        state = store.get(sid)
        with seen_lock:
            seen[sid] = state

    errors = _run_threads(create, [(s,) for s in sids])

    assert errors == []
    assert len(store) == 20
    assert set(seen) == set(sids)
    # No collision: every session got its own distinct, empty state object.
    assert len({id(s) for s in seen.values()}) == 20
    assert all(s.conversation_history == [] for s in seen.values())


# ---------------------------------------------------------------- Test 7
def test_concurrent_via_rest_api(
    client: TestClient, auth_headers, monkeypatch
):
    async def fake_run_graph(*args, user_query="", **kwargs):
        return {
            "response": f"ECHO[{user_query}]",
            "rooms": [{"title": f"room-for-{user_query}"}],
        }

    monkeypatch.setattr(api_module, "run_graph", fake_run_graph)

    sids = [str(uuid.uuid4()) for _ in range(10)]
    responses: dict[str, object] = {}
    lock = threading.Lock()

    def call(i: int, sid: str) -> None:
        resp = client.post(
            "/chat",
            json={"message": f"hello-from-{i}", "session_id": sid, "mode": "fast"},
            headers=auth_headers,
        )
        with lock:
            responses[sid] = (i, resp)

    errors = _run_threads(call, list(enumerate(sids)))

    assert errors == []
    assert len(responses) == 10
    for sid, (i, resp) in responses.items():
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"] == sid
        assert body["message"] == f"ECHO[hello-from-{i}]"
        for j in range(10):
            if j != i:
                assert f"hello-from-{j}]" not in body["message"]
        # Stored state contains only this session's own data.
        state = api_module.SESSIONS.get(sid)
        assert [m["content"] for m in state.conversation_history] == [
            f"hello-from-{i}",
            f"ECHO[hello-from-{i}]",
        ]
        assert state.previous_rooms == [{"title": f"room-for-hello-from-{i}"}]


# ---------------------------------------------------------------- Test 8
def test_memory_backend_thread_safety(backend_store):
    """50 threads doing mixed get/save; store stays consistent."""
    store = backend_store
    shared_sid = str(uuid.uuid4())
    own = {i: str(uuid.uuid4()) for i in range(50)}

    def mixed(i: int) -> None:
        sid = own[i]
        if i % 2 == 0:
            store.save(
                sid,
                ConversationState(previous_rooms=[{"title": f"room-{i}"}]),
            )
            state = store.get(sid)
        else:
            state = store.get(sid)
            state.previous_rooms = [{"title": f"room-{i}"}]
            store.save(sid, state)
        store.get(shared_sid)  # contended read/create on one key
        assert store.get(sid).previous_rooms == [{"title": f"room-{i}"}]

    errors = _run_threads(mixed, [(i,) for i in range(50)])

    assert errors == []  # includes no KeyError / assertion failures
    assert len(store) == 51  # 50 own + 1 shared, none missing
    for i, sid in own.items():
        assert store.get(sid).previous_rooms == [{"title": f"room-{i}"}]
    assert store.get(shared_sid).conversation_history == []
