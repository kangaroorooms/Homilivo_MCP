"""
Tests for the /ws/chat WebSocket endpoint (app/api.py).

/chat (REST) is unchanged and remains the reliable fallback - these
tests only exercise the additive streaming path: connection auth,
typing/token/done event sequencing, session continuity across
messages on one connection, per-turn error handling, and rate
limiting.
"""

from __future__ import annotations

import app.rate_limit as rate_limit_module
from app.rate_limit import InMemoryRateLimiter
from app.api import app
from fastapi.testclient import TestClient


def test_ws_connect_without_token_is_closed(client: TestClient):
    import pytest
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/chat"):
            pass

    assert exc_info.value.code == 4401


def test_ws_search_streams_typing_tokens_then_done(client: TestClient, mock_homilivo_success):
    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        ws.send_json({"message": "find rooms in Jaipur", "mode": "fast"})

        first = ws.receive_json()
        assert first["event"] == "typing"

        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["event"] == "done":
                break

        token_events = [e for e in events if e["event"] == "token"]
        done_event = events[-1]

        assert token_events, "expected at least one token event"
        assert done_event["event"] == "done"
        assert done_event["room_count"] == 3
        assert done_event["session_id"]

        # Concatenating every token chunk must reconstruct the final message.
        reconstructed = "".join(e["text"] for e in token_events)
        assert reconstructed == done_event["message"]


def test_ws_session_continuity_across_messages(client: TestClient, mock_homilivo_success):
    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        ws.send_json({"message": "find rooms in Jaipur", "mode": "fast"})

        done = None
        while done is None:
            event = ws.receive_json()
            if event["event"] == "done":
                done = event

        session_id = done["session_id"]

        ws.send_json(
            {"message": "which one is cheapest?", "session_id": session_id, "mode": "fast"}
        )

        done2 = None
        while done2 is None:
            event = ws.receive_json()
            if event["event"] == "done":
                done2 = event

        assert done2["session_id"] == session_id
        assert done2["selected_room"] is not None
        assert done2["selected_room"]["id"] == "room-2"


def test_ws_invalid_mode_sends_error_without_closing(client: TestClient, mock_homilivo_success):
    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        ws.send_json({"message": "hello", "mode": "not-a-real-mode"})

        error_event = ws.receive_json()
        assert error_event["event"] == "error"

        # Connection must still be usable after an error.
        ws.send_json({"message": "find rooms in Jaipur", "mode": "fast"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["event"] == "done":
                break
        assert events[-1]["room_count"] == 3


def test_ws_empty_message_sends_error(client: TestClient, mock_homilivo_success):
    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        ws.send_json({"message": "", "mode": "fast"})

        error_event = ws.receive_json()
        assert error_event["event"] == "error"


def test_ws_demo_token_is_redeemed_for_the_server_credential(
    client: TestClient, monkeypatch, mock_homilivo_token_recorder,
):
    monkeypatch.setenv("KANGAROO_TOKEN", "real-server-secret")
    get_recorded_tokens = mock_homilivo_token_recorder

    demo_token = client.get("/chat-ui/demo-token").json()["token"]

    with client.websocket_connect(f"/ws/chat?token={demo_token}") as ws:
        ws.send_json({"message": "find rooms in Jaipur", "mode": "fast"})

        done = None
        while done is None:
            event = ws.receive_json()
            if event["event"] == "done":
                done = event

    assert get_recorded_tokens() == ["Bearer real-server-secret"]


def _finish_turn(ws) -> dict:
    done = None
    while done is None:
        event = ws.receive_json()
        if event["event"] == "done":
            done = event
    return done


def test_ws_context_survives_a_dropped_connection_via_rest_fallback(
    client: TestClient, mock_homilivo_success, auth_headers,
):
    """
    There is no socket-level auto-reconnect in this design (see
    app/static/chat_ui.html: on a drop, the client falls back to REST
    and reconnects lazily on the next message) - what must actually
    hold is that session state (keyed by session_id, stored server-side
    in SESSIONS) survives the drop regardless of which transport
    resumes it. Simulates a drop by simply exiting the `with` block
    (closing the socket) and continuing on REST /chat with the same
    session_id.
    """
    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        ws.send_json({"message": "find rooms in Jaipur", "mode": "fast"})
        done = _finish_turn(ws)
        session_id = done["session_id"]

    # Connection is now closed. Resume the SAME session over REST.
    response = client.post(
        "/chat",
        json={"message": "which one is cheapest?", "session_id": session_id, "mode": "fast"},
        headers=auth_headers,
    )
    body = response.json()

    assert response.status_code == 200
    assert body["session_id"] == session_id
    assert body["selected_room"] is not None
    assert body["selected_room"]["id"] == "room-2"


def test_ws_context_survives_reconnect_on_a_new_websocket_connection(
    client: TestClient, mock_homilivo_success,
):
    """Same as above, but resuming on a brand-new WebSocket connection."""
    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        ws.send_json({"message": "find rooms in Jaipur", "mode": "fast"})
        session_id = _finish_turn(ws)["session_id"]

    # First connection closed; open a second one and resume by session_id.
    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws2:
        ws2.send_json(
            {"message": "which one is cheapest?", "session_id": session_id, "mode": "fast"}
        )
        done2 = _finish_turn(ws2)

    assert done2["session_id"] == session_id
    assert done2["selected_room"] is not None
    assert done2["selected_room"]["id"] == "room-2"


def test_ws_connection_rejected_when_rate_limited(client: TestClient, monkeypatch):
    import pytest
    from starlette.websockets import WebSocketDisconnect

    fresh = InMemoryRateLimiter()
    monkeypatch.setattr(rate_limit_module, "_backend", fresh)
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_MAX_REQUESTS", 0)
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_WINDOW_SECONDS", 60.0)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/chat?token=test-homilivo-token"):
            pass

    assert exc_info.value.code == 4429


def test_ws_mid_turn_rate_limit_sends_error_without_closing(
    client: TestClient, monkeypatch, mock_homilivo_success,
):
    """
    Unlike a limit exceeded AT CONNECT TIME (closes with 4429 - see
    test_ws_connection_rejected_when_rate_limited above), a limit hit
    on a later message over an already-open connection sends an error
    event and keeps the connection open, so a client is not forced to
    reconnect just because it briefly exceeded its budget.
    """
    fresh = InMemoryRateLimiter()
    monkeypatch.setattr(rate_limit_module, "_backend", fresh)
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_MAX_REQUESTS", 1)
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_WINDOW_SECONDS", 60.0)

    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        # The connect-time check already spent the budget of 1.
        ws.send_json({"message": "find rooms in Jaipur", "mode": "fast"})

        error_event = ws.receive_json()
        assert error_event["event"] == "error"


def test_ws_malformed_json_sends_error_without_closing(
    client: TestClient, mock_homilivo_success,
):
    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        ws.send_text("not valid json{{{")

        error_event = ws.receive_json()
        assert error_event["event"] == "error"

        # Connection must still be usable after a malformed message.
        ws.send_json({"message": "find rooms in Jaipur", "mode": "fast"})
        done = _finish_turn(ws)
        assert done["room_count"] == 3


def test_ws_heartbeat_ping_is_sent_during_a_quiet_period(
    client: TestClient, monkeypatch,
):
    """
    With no turn in flight, a quiet connection must still hear from
    the server at least once per WS_HEARTBEAT_INTERVAL_SECONDS -
    proof the server-side loop (and the connection) is alive, and the
    frontend's signal for detecting a truly dead socket. The interval
    is patched down so this test doesn't need to wait 20 real seconds.
    """
    import app.api as api_module

    monkeypatch.setattr(api_module, "WS_HEARTBEAT_INTERVAL_SECONDS", 0.05)

    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        ping = ws.receive_json()
        assert ping == {"event": "ping"}

        # The connection must still be fully usable after a heartbeat.
        second_ping = ws.receive_json()
        assert second_ping == {"event": "ping"}


def test_ws_abrupt_client_disconnect_mid_turn_does_not_break_the_server(
    client: TestClient, mock_homilivo_success,
):
    """
    Closing the socket before a turn's event sequence finishes (here,
    right after "typing", before any token/done arrives) must not
    crash the server or leak state that breaks later requests - the
    real-world case of a user closing the browser tab mid-reply.
    """
    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        ws.send_json({"message": "find rooms in Jaipur", "mode": "fast"})
        first = ws.receive_json()
        assert first["event"] == "typing"
        # Exit the `with` block now, dropping the connection mid-turn
        # instead of draining the remaining token/done events.

    # The server must still be healthy: an unrelated fresh request...
    assert client.get("/health").status_code == 200

    # ...and a brand new WebSocket connection must work normally.
    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws2:
        ws2.send_json({"message": "find rooms in Jaipur", "mode": "fast"})
        done = _finish_turn(ws2)
        assert done["room_count"] == 3


def test_ws_done_event_matches_rest_chat_payload_shape(
    client: TestClient, mock_homilivo_success, auth_headers,
):
    """
    /chat (REST) and /ws/chat's "done" event are built from the exact
    same turn dict (app.api._run_chat_turn - see the shared-helper
    docstring there). This proves that holds at the wire level: the
    same key set, and - since both hit the same mocked Homilivo
    fixture with the same query - the same deterministic content,
    aside from the necessarily-unique session_id/request_id.
    """
    rest_body = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        ws.send_json({"message": "find rooms in Jaipur", "mode": "fast"})
        done_event = _finish_turn(ws)

    ws_body = {k: v for k, v in done_event.items() if k != "event"}

    assert set(ws_body.keys()) == set(rest_body.keys())

    deterministic_fields = [
        "success",
        "mode",
        "message",
        "room_count",
        "rooms",
        "selected_room",
        "pending_clarification",
        "sources",
        "search_performed",
    ]
    for field in deterministic_fields:
        assert ws_body[field] == rest_body[field], field
