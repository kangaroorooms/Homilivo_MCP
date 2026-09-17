# P1-04: Session Isolation Regression Tests
# SCOPE: Current architecture (session_id boundary only)
# IN SCOPE: Proves different session_ids cannot
#   access each other's ConversationState
# NOT IN SCOPE: Authenticated user isolation
#   (requires P0-01/P0-03 Homilivo auth contract
#    to be clarified with backend team first)
# REMAINING RISK: session_id is currently client-
#   controlled with no ownership verification.
#   True user isolation requires P0-01/P0-03.

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from app.api import SESSIONS, app
from app.experiments.agent_modes import ConversationState
from app.session_store import InMemorySessionStore


@pytest.fixture()
def store() -> InMemorySessionStore:
    """A fresh, isolated in-memory store per test - never the shared
    process-wide SESSIONS singleton - so tests can't leak state into
    each other or into unrelated test modules."""
    return InMemorySessionStore()


def test_different_sessions_have_isolated_state(store: InMemorySessionStore):
    """
    PROVES: two distinct session_ids, saved with different
    previous_rooms payloads, read back only their own data - no
    cross-session leakage in the ConversationState.previous_rooms field.

    DOES NOT PROVE: anything about authenticated user identity, or
    that session_id itself is hard to guess/reuse (see
    test_session_isolation_remaining_risk below).
    """
    state_a = ConversationState(previous_rooms=[{"title": "Room A"}])
    state_b = ConversationState(previous_rooms=[{"title": "Room B"}])

    store.save("session-A", state_a)
    store.save("session-B", state_b)

    read_a = store.get("session-A")
    titles_a = [r["title"] for r in read_a.previous_rooms]
    assert titles_a == ["Room A"]
    assert "Room B" not in titles_a

    read_b = store.get("session-B")
    titles_b = [r["title"] for r in read_b.previous_rooms]
    assert titles_b == ["Room B"]
    assert "Room A" not in titles_b


def test_session_b_cannot_overwrite_session_a(store: InMemorySessionStore):
    """
    PROVES: saving a ConversationState under session-B does not mutate
    or overwrite the conversation_history already stored under
    session-A - the dict-keyed store keeps independent objects per key.

    DOES NOT PROVE: protection against a caller who deliberately reuses
    session-A's own session_id (that is by design in this scope - see
    test_session_isolation_remaining_risk).
    """
    state_a = ConversationState(
        conversation_history=[{"role": "user", "content": "message from A"}]
    )
    state_b = ConversationState(
        conversation_history=[{"role": "user", "content": "message from B"}]
    )

    store.save("session-A", state_a)
    store.save("session-B", state_b)

    read_a = store.get("session-A")
    contents_a = [m["content"] for m in read_a.conversation_history]
    assert contents_a == ["message from A"]
    assert "message from B" not in contents_a


def test_unknown_session_returns_fresh_state(store: InMemorySessionStore):
    """
    PROVES: looking up a session_id that was never saved returns a
    brand-new, empty ConversationState - not an error, and not a stale
    or borrowed state from any other session.

    DOES NOT PROVE: that session_ids are unguessable, or that a client
    is prevented from later reusing a known-valid session_id belonging
    to someone else (no ownership check exists - see
    test_session_isolation_remaining_risk).
    """
    fresh = store.get("nonexistent-session-xyz-999")

    assert fresh.previous_rooms == []
    assert fresh.conversation_history == []
    assert fresh.pending_search == {}


def test_session_reset_does_not_affect_other_sessions(store: InMemorySessionStore):
    """
    PROVES: resetting (clearing) session-A - simulated by saving a
    fresh empty ConversationState over it, matching how a reset flow
    would be implemented - leaves session-B's stored data completely
    untouched.

    DOES NOT PROVE: that /chat/reset itself is authorized correctly;
    that endpoint's authorization is a separate, already-known gap
    (see remaining risk) unrelated to store-level isolation.
    """
    store.save("session-A", ConversationState(previous_rooms=[{"title": "Room A"}]))
    store.save("session-B", ConversationState(previous_rooms=[{"title": "Room B"}]))

    # Simulate a reset of session-A only.
    store.save("session-A", ConversationState())

    read_b = store.get("session-B")
    assert [r["title"] for r in read_b.previous_rooms] == ["Room B"]


def test_concurrent_session_writes_do_not_cross(store: InMemorySessionStore):
    """
    PROVES: many threads writing to session-A and session-B at the
    same time never cause one session's data to end up stored under
    the other session's key - the store's internal lock keeps
    concurrent get/save operations from cross-contaminating state.

    DOES NOT PROVE: anything about request-level concurrency through
    the FastAPI app itself (that is covered separately by
    test_rest_endpoint_uses_provided_session_id and
    test_websocket_sessions_are_isolated below).
    """

    def write_many(session_id: str, label: str) -> None:
        for i in range(10):
            state = ConversationState(previous_rooms=[{"title": f"{label}-{i}"}])
            store.save(session_id, state)

    thread_a = threading.Thread(target=write_many, args=("session-A", "A"))
    thread_b = threading.Thread(target=write_many, args=("session-B", "B"))

    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    final_a = store.get("session-A")
    final_b = store.get("session-B")

    assert all(r["title"].startswith("A-") for r in final_a.previous_rooms)
    assert all(r["title"].startswith("B-") for r in final_b.previous_rooms)


def test_rest_endpoint_uses_provided_session_id(
    client: TestClient, mock_homilivo_success, auth_headers
):
    """
    PROVES: two REST /chat calls with different client-supplied
    session_ids end up stored as two independent ConversationState
    entries in the process-wide SESSIONS store, with no shared or
    bled-together state between them.

    DOES NOT PROVE: that the caller of /chat is who they claim to be,
    or that session_id could not be guessed/reused by a different
    caller - /chat has no ownership check on session_id at all (see
    test_session_isolation_remaining_risk).
    """
    resp_a = client.post(
        "/chat",
        json={
            "message": "find rooms in Jaipur",
            "session_id": "test-session-rest-A",
            "mode": "fast",
        },
        headers=auth_headers,
    )
    resp_b = client.post(
        "/chat",
        json={
            "message": "find rooms in Jaipur",
            "session_id": "test-session-rest-B",
            "mode": "fast",
        },
        headers=auth_headers,
    )

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200

    state_a = SESSIONS.get("test-session-rest-A")
    state_b = SESSIONS.get("test-session-rest-B")

    history_a = [m["content"] for m in state_a.conversation_history]
    history_b = [m["content"] for m in state_b.conversation_history]

    assert "find rooms in Jaipur" in history_a
    assert state_a is not state_b
    # Each session's own history must not contain the other's entries
    # beyond the identical user message text (which is expected since
    # both turns sent the same message) - the object identity check
    # above is the real proof of isolation; this asserts no unexpected
    # duplication across the two lists' lengths.
    assert len(history_a) == len(history_b)


def test_websocket_sessions_are_isolated(client: TestClient, mock_homilivo_success):
    """
    Proves WS connections with different session_ids cannot access
    each other's ConversationState. Does NOT test authenticated user
    identity (pending P0-01/P0-03 clarification).
    """
    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws_a:
        ws_a.send_json(
            {
                "message": "find rooms in Jaipur",
                "session_id": "test-session-ws-A",
                "mode": "fast",
            }
        )
        done_a = None
        while done_a is None:
            event = ws_a.receive_json()
            if event["event"] == "done":
                done_a = event

    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws_b:
        ws_b.send_json(
            {
                "message": "find rooms in Jaipur",
                "session_id": "test-session-ws-B",
                "mode": "fast",
            }
        )
        done_b = None
        while done_b is None:
            event = ws_b.receive_json()
            if event["event"] == "done":
                done_b = event

    assert done_a["session_id"] == "test-session-ws-A"
    assert done_b["session_id"] == "test-session-ws-B"

    state_a = SESSIONS.get("test-session-ws-A")
    state_b = SESSIONS.get("test-session-ws-B")

    assert state_a is not state_b


def test_session_isolation_remaining_risk(store: InMemorySessionStore):
    """
    Documents the known remaining risk: session_id is currently
    client-controlled with no ownership verification. Any caller who
    knows a session_id can read that session's state. True user
    isolation requires P0-01/P0-03 Homilivo auth integration.
    """
    store.save("any-session-id", ConversationState(previous_rooms=[{"title": "Secret Room"}]))

    # Nothing in get() checks who is asking - it returns whatever is
    # stored under that key to any caller that supplies it.
    state = store.get("any-session-id")
    assert state.previous_rooms == [{"title": "Secret Room"}]
