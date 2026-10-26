"""Phase 5: the FastAPI server and the four carry-over fixes. Offline: TestClient, mocks, no torch."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from llm_approach import fallback, llm_client, pipeline, reuse, server_app
from llm_approach.actions import TurnBudget, platform
from llm_approach.schema import parse_decision
from llm_approach.state import LLMConversationState, SessionStore
from llm_approach.tests.conftest import ROOT, dec

SECRET = "SECRETTOKENVALUE123"
DONE_KEYS = {
    "success", "session_id", "request_id", "mode", "message", "room_count", "rooms",
    "selected_room", "pending_clarification", "sources", "search_performed",
}
CARD_KEYS = {"id", "title", "city", "location", "price_per_month", "room_type", "ideal_for", "amenities"}


@pytest.fixture(autouse=True)
def server_env(monkeypatch):
    from llm_approach.security.rate_limit import InMemoryRateLimiter  # v2: was: from app.rate_limit import InMemoryRateLimiter

    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")
    monkeypatch.setenv("STREAM_CHUNK_DELAY_SECONDS", "0")
    monkeypatch.setenv("KANGAROO_TOKEN", "fake-rooms-credential")
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    limiter = InMemoryRateLimiter()  # fresh budgets for every test
    monkeypatch.setattr(server_app._rl, "get_rate_limiter", lambda: limiter)
    monkeypatch.setattr(pipeline, "_store", SessionStore())
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    pipeline.configure(full_guardrails=True)
    yield
    pipeline.configure(full_guardrails=True)


@pytest.fixture
def client():
    with TestClient(server_app.create_app()) as c:
        yield c


def stub_turn(monkeypatch, reply="Hello there from the stub", success=True, rooms=None, tool=False):
    calls = []

    async def fake(session_id, message, *, store=None):
        calls.append((session_id, message))
        return pipeline.TurnResult(reply=reply, rooms=rooms, metrics={"tool_called": tool}, success=success)

    monkeypatch.setattr(pipeline, "handle_turn", fake)
    return calls


def ws_turn(ws, **payload):
    """Send one message, return the events up to and including done/error."""
    ws.send_json(payload)
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["event"] in ("done", "error"):
            return events


# ---- REST --------------------------------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ready_not_ready_until_preloaded(client, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_value_must_not_appear")
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": False, "knowledge_base": False})
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["checks"] == {
        "llm_key_present": True, "kangaroo_token_present": True, "knowledge_base": False, "guardrails": False,
    }


def test_ready_ok_reports_booleans_only(client, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_value_must_not_appear")
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": True, "knowledge_base": True})
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["status"] == "ready"
    assert "gsk_value_must_not_appear" not in r.text
    monkeypatch.delenv("GROQ_API_KEY")
    assert client.get("/ready").json()["checks"]["llm_key_present"] is False


def test_demo_token_issued_and_distinct_from_credential(client):
    r = client.get("/chat-ui/demo-token")
    token = r.json()["token"]
    assert r.status_code == 200 and token and token != "fake-rooms-credential"
    assert server_app._demo.redeem_demo_token(token) == "fake-rooms-credential"


def test_demo_token_404_without_credential(client, monkeypatch):
    monkeypatch.delenv("KANGAROO_TOKEN")
    assert client.get("/chat-ui/demo-token").status_code == 404


def test_chat_happy_path_with_stub_pipeline(client, monkeypatch):
    calls = stub_turn(monkeypatch)
    sid = str(uuid.uuid4())
    r = client.post("/chat", json={"message": "hi", "session_id": sid, "mode": "groq"},
                    headers={"Authorization": f"Bearer {SECRET}"})
    body = r.json()
    assert r.status_code == 200 and set(body) == DONE_KEYS
    assert body["success"] is True and body["message"] == "Hello there from the stub"
    assert body["session_id"] == sid and body["mode"] == "groq" and body["rooms"] == []
    assert calls == [(sid, "hi")]


def test_chat_requires_authorization_and_valid_body(client, monkeypatch):
    stub_turn(monkeypatch)
    assert client.post("/chat", json={"message": "hi"}).status_code in (401, 403)
    auth = {"Authorization": "Bearer x"}
    assert client.post("/chat", json={"message": "hi", "mode": "turbo"}, headers=auth).status_code == 422
    assert client.post("/chat", json={"message": "a" * 501}, headers=auth).status_code == 422
    assert client.post("/chat", json={"message": ""}, headers=auth).status_code == 422


def test_chat_rate_limit_gives_429_with_retry_after(client, monkeypatch):
    stub_turn(monkeypatch)
    from llm_approach.security.rate_limit import RateLimitResult  # v2: was: from app.rate_limit import RateLimitResult

    monkeypatch.setattr(server_app._rl, "check_rate_limit", lambda key: RateLimitResult(False, 0, 4.0))
    r = client.post("/chat", json={"message": "hi"}, headers={"Authorization": "Bearer x"})
    assert r.status_code == 429 and r.headers["retry-after"] == "5"


def test_chat_non_uuid_session_gets_fresh_uuid(client, monkeypatch):
    stub_turn(monkeypatch)
    r = client.post("/chat", json={"message": "hi", "session_id": "me@example.com"},
                    headers={"Authorization": "Bearer x"})
    sid = r.json()["session_id"]
    assert sid != "me@example.com" and str(uuid.UUID(sid)) == sid


def test_chat_uppercase_uuid_is_lowercased(client, monkeypatch):
    stub_turn(monkeypatch)
    sid = str(uuid.uuid4())
    r = client.post("/chat", json={"message": "hi", "session_id": sid.upper()},
                    headers={"Authorization": "Bearer x"})
    assert r.json()["session_id"] == sid


@pytest.mark.parametrize("bad", [
    "me@example.com", "a" * 5000, "ABCDEF12-3456-7890-ABCD-EF1234567890\n", " 3b1f0c52-8a7e-4c1d-9e2f-0a1b2c3d4e5f",
    "not-a-uuid", 12345, ["x"],
])
def test_normalize_session_id_replaces_invalid_values(bad):
    sid = server_app.normalize_session_id(bad)
    assert sid != bad and str(uuid.UUID(sid)) == sid


def test_normalize_session_id_lowercases_and_keeps_valid():
    sid = str(uuid.uuid4())
    assert server_app.normalize_session_id(sid.upper()) == sid
    assert server_app.normalize_session_id(None) != server_app.normalize_session_id("")


def test_raw_session_id_reaches_no_limiter_store_or_log(client, monkeypatch, caplog):
    seen = []
    real = server_app._rl.check_session_rate_limit
    monkeypatch.setattr(server_app._rl, "check_session_rate_limit", lambda s: (seen.append(s), real(s))[1])
    stub = stub_turn(monkeypatch)
    raw = "victim@example.com"
    with caplog.at_level(logging.DEBUG):
        r = client.post("/chat", json={"message": "hi", "session_id": raw}, headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    assert seen and all(str(uuid.UUID(s)) == s for s in seen)
    assert all(str(uuid.UUID(c[0])) == c[0] for c in stub)
    assert raw not in caplog.text and "victim" not in caplog.text


def test_chat_reset_with_uppercase_id_resets_the_lowercase_session(client):
    sid = str(uuid.uuid4())
    pipeline.get_store().get_or_create(sid)
    body = client.post("/chat/reset", params={"session_id": sid.upper()}).json()
    assert body["existed"] is True
    assert client.post("/chat/reset", params={"session_id": "bad\n"}).json()["existed"] is False


def test_chat_reset(client):
    sid = str(uuid.uuid4())
    assert client.post("/chat/reset", params={"session_id": sid}).json() == {
        "status": "success", "session_id": sid, "existed": False,
    }


def test_chat_with_real_pipeline_returns_cards_in_frontend_format(client, monkeypatch, fake_groq):
    fake_groq([json.dumps(dec("NEW_SEARCH", city="Bangalore"))])
    r = client.post("/chat", json={"message": "find me room in bangalore"}, headers={"Authorization": "Bearer x"})
    body = r.json()
    assert body["success"] is True and body["room_count"] == len(body["rooms"]) == 10
    assert body["search_performed"] is True
    assert set(body["rooms"][0]) == CARD_KEYS and isinstance(body["rooms"][0]["id"], str)


# ---- WebSocket ---------------------------------------------------------------------------


def test_ws_event_order_typing_tokens_done(client, monkeypatch):
    stub_turn(monkeypatch, reply="Here are some words to stream out")
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        events = ws_turn(ws, message="hi", session_id=None, mode="groq")
    names = [e["event"] for e in events]
    assert names[0] == "typing" and names[-1] == "done"
    assert set(names[1:-1]) == {"token"} and len(names) > 3
    assert "".join(e["text"] for e in events if e["event"] == "token") == "Here are some words to stream out"
    done = events[-1]
    assert set(done) == DONE_KEYS | {"event"} and done["message"] == "Here are some words to stream out"
    uuid.UUID(done["session_id"])


def test_ws_done_matches_frontend_types_with_real_pipeline(client, fake_groq):
    fake_groq([json.dumps(dec("NEW_SEARCH", city="Bangalore"))])
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        done = ws_turn(ws, message="rooms in bangalore", session_id=None, mode="fast")[-1]
    assert done["success"] is True and done["room_count"] == 10 and done["search_performed"] is True
    for card in done["rooms"]:
        assert set(card) == CARD_KEYS and isinstance(card["ideal_for"], list) and isinstance(card["amenities"], list)
    assert done["selected_room"] is None and done["sources"] == [] and done["pending_clarification"] is False


def test_ws_selected_room_and_unchanged_cards_on_followup(client, fake_groq):
    fake_groq([
        json.dumps(dec("NEW_SEARCH", city="Bangalore")),
        json.dumps(dec("ROOM_FOLLOWUP", "LOWEST_PRICE")),
    ])
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        first = ws_turn(ws, message="rooms in bangalore", session_id=None, mode="fast")[-1]
        second = ws_turn(ws, message="which is cheapest", session_id=first["session_id"], mode="fast")[-1]
    assert second["session_id"] == first["session_id"] and second["room_count"] == 10
    assert second["selected_room"] is not None and set(second["selected_room"]) == CARD_KEYS
    assert second["search_performed"] is False


def test_ws_missing_token_closes_4401(client):
    for url in ("/ws/chat", "/ws/chat?token="):
        with pytest.raises(WebSocketDisconnect) as info:
            with client.websocket_connect(url) as ws:
                ws.receive_json()
        assert info.value.code == 4401


def test_ws_rate_limit_closes_4429(client, monkeypatch):
    from llm_approach.security.rate_limit import RateLimitResult  # v2: was: from app.rate_limit import RateLimitResult

    monkeypatch.setattr(server_app._rl, "check_rate_limit", lambda key: RateLimitResult(False, 0, 4.0))
    with pytest.raises(WebSocketDisconnect) as info:
        with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
            ws.receive_json()
    assert info.value.code == 4429


def test_ws_pipeline_exception_gives_bot_message_and_connection_stays_open(client, monkeypatch):
    async def boom(session_id, message, *, store=None):
        raise RuntimeError("internal detail that must not leak")

    monkeypatch.setattr(pipeline, "handle_turn", boom)
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        events = ws_turn(ws, message="hi", session_id=None, mode="fast")
        done = events[-1]
        assert done["event"] == "done" and done["success"] is False
        assert done["message"] == "Something went wrong on our side, please try again."
        assert "internal detail" not in json.dumps(events)
        stub_turn(monkeypatch)  # the connection is still usable
        assert ws_turn(ws, message="again", session_id=done["session_id"], mode="fast")[-1]["success"] is True


def test_ws_501_char_message_is_an_error_event_and_connection_stays_open(client, monkeypatch):
    stub_turn(monkeypatch)
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        events = ws_turn(ws, message="a" * 501, session_id=None, mode="fast")
        assert events == [{"event": "error", "detail": "message must be a non-empty string of at most 500 characters."}]
        assert ws_turn(ws, message="a" * 500, session_id=None, mode="fast")[-1]["event"] == "done"
        assert ws_turn(ws, message="   ", session_id=None, mode="fast")[-1]["event"] == "error"


def test_ws_non_uuid_session_id_is_replaced(client, monkeypatch):
    calls = stub_turn(monkeypatch)
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        done = ws_turn(ws, message="hi", session_id="not-a-uuid", mode="fast")[-1]
        sid = done["session_id"]
        assert sid != "not-a-uuid" and str(uuid.UUID(sid)) == sid
        again = ws_turn(ws, message="hi", session_id=None, mode="fast")[-1]  # socket reuses its own id
        assert again["session_id"] == sid
    assert all(call[0] == sid for call in calls)


@pytest.mark.parametrize("mode", ["fast", "qwen", "gemma", "groq"])
def test_ws_all_four_modes_are_accepted_and_ignored(client, monkeypatch, mode):
    stub_turn(monkeypatch)
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        done = ws_turn(ws, message="hi", session_id=None, mode=mode)[-1]
    assert done["event"] == "done" and done["mode"] == mode


def test_ws_missing_mode_defaults_and_bad_mode_is_an_error(client, monkeypatch):
    stub_turn(monkeypatch)
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        assert ws_turn(ws, message="hi", session_id=None)[-1]["mode"] == "fast"
        assert ws_turn(ws, message="hi", session_id=None, mode="turbo")[-1] == {
            "event": "error", "detail": "Invalid mode: 'turbo'",
        }


def test_ws_bad_json_and_non_object_keep_connection_open(client, monkeypatch):
    stub_turn(monkeypatch)
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        ws.send_text("{not json")
        assert ws.receive_json() == {"event": "error", "detail": "Message must be valid JSON."}
        ws.send_json([1, 2])
        assert ws.receive_json() == {"event": "error", "detail": "Message must be a JSON object."}
        assert ws_turn(ws, message="hi", session_id=None, mode="fast")[-1]["event"] == "done"


def test_ws_per_turn_rate_limit_is_an_error_event(client, monkeypatch):
    stub_turn(monkeypatch)
    from llm_approach.security.rate_limit import RateLimitResult  # v2: was: from app.rate_limit import RateLimitResult

    ok = RateLimitResult(True, 1, 0.0)
    answers = iter([ok, RateLimitResult(False, 0, 3.0)])  # connect passes, first turn is limited
    monkeypatch.setattr(server_app._rl, "check_rate_limit", lambda key: next(answers))
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        assert ws_turn(ws, message="hi", session_id=None, mode="fast")[-1] == {
            "event": "error", "detail": "Too many requests. Please slow down.",
        }


def test_ws_ping_heartbeat_when_idle(client, monkeypatch):
    monkeypatch.setenv("WS_HEARTBEAT_INTERVAL_SECONDS", "0.05")
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        assert ws.receive_json() == {"event": "ping"}


def test_ws_demo_token_is_accepted(client, monkeypatch):
    stub_turn(monkeypatch)
    token = client.get("/chat-ui/demo-token").json()["token"]
    with client.websocket_connect(f"/ws/chat?token={token}") as ws:
        assert ws_turn(ws, message="hi", session_id=None, mode="fast")[-1]["event"] == "done"


# ---- Groq failure, logs, CORS, startup ---------------------------------------------------


def test_groq_failure_gives_a_bot_message_not_a_500(client, fake_groq):
    fake_groq([ConnectionError("groq down")])
    r = client.post("/chat", json={"message": "find me a room"}, headers={"Authorization": "Bearer x"})
    body = r.json()
    assert r.status_code == 200 and body["success"] is True
    assert body["message"] == "Could you tell me the city and budget you're looking for?"
    assert "groq down" not in r.text


def test_logs_never_contain_the_token_or_a_token_url(client, monkeypatch, caplog, fake_groq):
    caplog.set_level(logging.DEBUG)
    fake_groq([ConnectionError("down"), ConnectionError("down")])
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        ws_turn(ws, message="hi there", session_id="bad id", mode="fast")
    client.post("/chat", json={"message": "hi"}, headers={"Authorization": f"Bearer {SECRET}"})
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()
    assert SECRET not in caplog.text and "token=" not in caplog.text and "fake-rooms-credential" not in caplog.text


def test_access_log_filter_redacts_the_query_token():
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/1.1" %d',
        ("127.0.0.1:1", "GET", f"/ws/chat?token={SECRET}&x=1", 101), None,
    )
    server_app._RedactTokenFilter().filter(record)
    assert SECRET not in record.getMessage() and "token=REDACTED&x=1" in record.getMessage()


def test_cors_is_off_by_default_and_opt_in(monkeypatch):
    monkeypatch.delenv(server_app.CORS_ENV, raising=False)
    assert not any(m.cls.__name__ == "CORSMiddleware" for m in server_app.create_app().user_middleware)
    monkeypatch.setenv(server_app.CORS_ENV, "http://localhost:5173, http://127.0.0.1:5173")
    app = server_app.create_app()
    cors = [m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"]
    assert cors and cors[0].kwargs["allow_origins"] == ["http://localhost:5173", "http://127.0.0.1:5173"]


def test_startup_preload_can_be_skipped(monkeypatch):
    called = []
    monkeypatch.setattr(pipeline, "preload", lambda: called.append("guardrails"))
    monkeypatch.setattr(reuse, "get_knowledge_preload", lambda: (lambda: called.append("kb")))
    with TestClient(server_app.create_app()):
        pass
    assert called == []  # the autouse fixture sets LLM_APPROACH_SKIP_PRELOAD=1

    monkeypatch.delenv(server_app.SKIP_PRELOAD_ENV)
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": False, "knowledge_base": False})
    with TestClient(server_app.create_app()):
        pass
    assert called == ["guardrails", "kb"] and server_app._readiness == {"guardrails": True, "knowledge_base": True}


def test_preload_failure_is_survivable(monkeypatch):
    def broken():
        raise RuntimeError("no model")

    monkeypatch.delenv(server_app.SKIP_PRELOAD_ENV)
    monkeypatch.setattr(pipeline, "preload", broken)
    monkeypatch.setattr(reuse, "get_knowledge_preload", lambda: broken)
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": False, "knowledge_base": False})
    with TestClient(server_app.create_app()) as c:
        assert c.get("/health").status_code == 200 and c.get("/ready").status_code == 503


def test_server_module_imports_without_torch():
    code = (
        "import sys, llm_approach.server_app as s\n"
        "bad = [m for m in ('torch','sentence_transformers','ollama','mcp','app.experiments.agent_modes','app.guardrails') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, r'{ROOT}')\n{code}"],
                         capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-500:]


# ---- carry-over a: Groq rate limit ------------------------------------------------------


class RateLimitError(Exception):  # same class name the groq SDK uses
    pass


class Err429(Exception):
    status_code = 429


def _real_client_turn(monkeypatch, fake_groq, replies, message="find me a room in bangalore"):
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    fake_groq(replies)
    return asyncio.run(pipeline.handle_turn(str(uuid.uuid4()), message, store=SessionStore()))


@pytest.mark.parametrize("exc", [RateLimitError("429 too many"), Err429("throttled")])
def test_groq_rate_limit_runs_the_rule_based_fallback(monkeypatch, fake_groq, exc):
    r = _real_client_turn(monkeypatch, fake_groq, [exc])
    assert r.reply != "I'm getting a lot of requests right now, please try again in a minute."
    assert r.metrics["error_class"] == "RateLimitError" and r.metrics["fallback_used"] is True
    assert r.metrics["tool_called"] is True and r.success is True


def test_other_groq_errors_still_use_the_fallback(monkeypatch, fake_groq):
    r = _real_client_turn(monkeypatch, fake_groq, [ConnectionError("down")])
    assert r.metrics["fallback_used"] is True and r.reply.startswith("I found 10 rooms in Bangalore")


def test_is_rate_limit_error():
    assert llm_client.is_rate_limit_error(RateLimitError()) and llm_client.is_rate_limit_error(Err429())
    assert not llm_client.is_rate_limit_error(ConnectionError())


# ---- carry-over b: fallback with rooms on screen ------------------------------------------


def test_fallback_with_rooms_on_screen_asks_the_state_aware_question(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    r = pipe.turn("hmm what about it", None)  # no LLM decision
    assert r.reply == "Would you like to search again, or ask about these rooms?"
    assert r.metrics["fallback_used"] is True and pipe.state.pending_question["type"] == "clarify"
    assert len(pipe.state.rooms_on_screen) == 10


def test_fallback_without_rooms_keeps_the_city_question(pipe):
    r = pipe.turn("hmm", None)
    assert r.reply == fallback.ASK_CITY_AND_BUDGET


# ---- carry-over c: placeholder text in platform answers ----------------------------------------

REAL_PLACEHOLDER = "[PLACEHOLDER — NOT VERIFIED] Homilivo charges a fee. confirm wording with the team"


def _platform(monkeypatch, message):
    monkeypatch.setattr(reuse, "get_answer_platform_question",
                        lambda: (lambda q: {"grounded": True, "message": message, "sources": ["doc"]}))
    return platform.handle_platform_info(
        parse_decision({"intent": "PLATFORM_INFO", "confidence": 0.9, "standalone_query": "fees?"}),
        {}, LLMConversationState(), TurnBudget(),
    )


@pytest.mark.parametrize("text", [
    REAL_PLACEHOLDER,
    "this is a placeholder answer",
    "Fees: NOT VERIFIED yet",
    "Please Confirm Wording first",
    "TODO: write this",
    "price is TBD",
    "Lorem ipsum dolor",
])
def test_platform_refuses_internal_markers(monkeypatch, text):
    r = _platform(monkeypatch, text)
    assert r.reply == "I don't have that information, please contact Homilivo support."
    assert r.error_class == "placeholder_text"


def test_platform_still_returns_a_clean_grounded_answer(monkeypatch):
    r = _platform(monkeypatch, "Homilivo earns from owner subscriptions. Tutorials and todos are unrelated.")
    assert r.reply.startswith("Homilivo earns") and r.error_class is None


def test_platform_placeholder_through_the_pipeline(pipe, monkeypatch):
    monkeypatch.setattr(reuse, "get_answer_platform_question",
                        lambda: (lambda q: {"grounded": True, "message": REAL_PLACEHOLDER, "sources": []}))
    r = pipe.turn("how does homilivo make money", dec("PLATFORM_INFO", sq="how does homilivo make money"))
    assert r.reply == "I don't have that information, please contact Homilivo support."


# ---- carry-over d: area question with examples ---------------------------------------------------


def test_area_question_uses_locations_of_rooms_on_screen(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    r = pipe.turn("something in a different area", dec("REFINEMENT", city="A Different Area", nc=True, conf=0.8))
    assert r.reply == "Which area would you like? For example Koramangala or Adugodi."


def test_area_examples_max_three_distinct_never_invented():
    from llm_approach.actions import search

    s = LLMConversationState()
    assert search.area_question(s) == "Which area would you like?"  # no rooms: no examples
    s.rooms_on_screen = [{"location": loc} for loc in ["Indiranagar", "indiranagar", None, "HSR", "Adugodi", "Koramangala"]]
    assert search.area_examples(s) == ["Indiranagar", "HSR", "Adugodi"]
    assert search.area_question(s) == "Which area would you like? For example Indiranagar, HSR or Adugodi."
    s.rooms_on_screen = [{"location": "HSR"}]
    assert search.area_question(s) == "Which area would you like? For example HSR."


def test_repeating_the_current_city_says_so_and_asks_for_an_area(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    searches = len(pipe.searches)
    r = pipe.turn("in bangalore", dec("REFINEMENT", city="Bangalore"))
    assert r.reply == (
        "You're already searching in Bangalore. Which area would you like? "
        "For example Koramangala or Adugodi."
    )
    assert len(pipe.searches) == searches and len(pipe.state.rooms_on_screen) == 10


# ---- .env loading -------------------------------------------------------------------------------

DOTENV_SECRET = "dotenv-dummy-secret-9f3a"


def _clean_env(monkeypatch, *names):
    for name in names:  # registers the original state so the test cannot leak loaded values
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)


def test_load_env_reads_a_temp_dotenv(monkeypatch, tmp_path):
    _clean_env(monkeypatch, "GROQ_API_KEY", "KANGAROO_TOKEN")
    env = tmp_path / ".env"
    env.write_text(f"GROQ_API_KEY={DOTENV_SECRET}-groq\nKANGAROO_TOKEN={DOTENV_SECRET}-rooms\n", encoding="utf-8")
    assert server_app.load_env(env) is True
    assert os.environ["GROQ_API_KEY"] == f"{DOTENV_SECRET}-groq"
    assert server_app._demo.issue_demo_token()  # the demo-token flow now has its credential


def test_load_env_does_not_override_existing_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "already-set")
    _clean_env(monkeypatch, "KANGAROO_TOKEN")
    env = tmp_path / ".env"
    env.write_text(f"GROQ_API_KEY={DOTENV_SECRET}\nKANGAROO_TOKEN={DOTENV_SECRET}\n", encoding="utf-8")
    server_app.load_env(env)
    assert os.environ["GROQ_API_KEY"] == "already-set" and os.environ["KANGAROO_TOKEN"] == DOTENV_SECRET


def test_load_env_missing_file_is_harmless(tmp_path):
    assert server_app.load_env(tmp_path / "nope.env") is False


def test_default_env_path_is_the_repo_root_like_cli():
    from llm_approach import cli

    assert server_app.REPO_ROOT == cli.REPO_ROOT


def test_ready_booleans_only_and_no_secret_in_logs_or_responses(client, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    _clean_env(monkeypatch, "GROQ_API_KEY", "KANGAROO_TOKEN")
    env = tmp_path / ".env"
    env.write_text(f"GROQ_API_KEY={DOTENV_SECRET}-groq\nKANGAROO_TOKEN={DOTENV_SECRET}-rooms\n", encoding="utf-8")
    server_app.load_env(env)
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": True, "knowledge_base": True})
    stub_turn(monkeypatch)

    ready = client.get("/ready")
    assert ready.json()["checks"] == {
        "llm_key_present": True, "kangaroo_token_present": True, "knowledge_base": True, "guardrails": True,
    }
    token = client.get("/chat-ui/demo-token")
    chat = client.post("/chat", json={"message": "hi"}, headers={"Authorization": f"Bearer {token.json()['token']}"})
    with client.websocket_connect(f"/ws/chat?token={token.json()['token']}") as ws:
        events = ws_turn(ws, message="hi", session_id=None, mode="fast")
    for body in (ready.text, token.text, chat.text, json.dumps(events), caplog.text):
        assert DOTENV_SECRET not in body

    monkeypatch.delenv("KANGAROO_TOKEN")
    assert client.get("/ready").json()["checks"]["kangaroo_token_present"] is False
    assert client.get("/ready").status_code == 200  # reported, not required
