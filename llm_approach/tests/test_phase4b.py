"""Phase 4B: background startup, /ready while loading, chat before ready, and the /metrics route.
Fakes only: no Gemini, Groq or rooms API call, no embedding model."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from llm_approach import llm_client, llm_resilience, observability, pipeline, providers, reuse, rooms_client, server_app, switches
from llm_approach.actions import TurnBudget, search
from llm_approach.providers import Completion
from llm_approach.state import LLMConversationState
from llm_approach.tests.test_search_timeout import api, slow  # noqa: F401 - the fake rooms API and its fixture
from llm_approach.tests.test_server import server_env  # noqa: F401 - the autouse server fixture of the suite

SERVER_TOKEN = "fake-rooms-credential"  # set by the server_env fixture
USER_TEXT = "SENTINEL_USER_TEXT_Q7"
ROOM_TEXT = "SENTINEL_ROOM_TITLE_K3"
KEY = "SENTINEL_API_KEY_Z9"
METRICS_TOKEN_ENV_NAME = "LLM_APPROACH_METRICS_TOKEN"
METRICS_TOKEN = "metrics-token-SENTINEL-55"
SESSION = "11111111-2222-4333-8444-555555555555"


def stub_turn(monkeypatch, reply="Hello there from the stub"):
    async def fake(session_id, message, *, store=None):
        return pipeline.TurnResult(reply=reply, rooms=None, metrics={}, success=True)

    monkeypatch.setattr(pipeline, "handle_turn", fake)


def wait_until(check, seconds=5.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.02)
    return False


# ---- startup and readiness --------------------------------------------------------------------


@pytest.fixture
def slow_preload(monkeypatch):
    """The preload (guardrails, then the knowledge base) waits on two events the test controls."""
    gate_guardrails, gate_kb = threading.Event(), threading.Event()
    monkeypatch.delenv(server_app.SKIP_PRELOAD_ENV)
    monkeypatch.setattr(pipeline, "preload", lambda: gate_guardrails.wait(10))
    monkeypatch.setattr(reuse, "get_knowledge_preload", lambda: (lambda: gate_kb.wait(10)))
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": False, "knowledge_base": False})
    yield gate_guardrails, gate_kb
    gate_guardrails.set()
    gate_kb.set()


def test_not_ready_then_ready_and_chat_follows(slow_preload, monkeypatch):
    gate_guardrails, gate_kb = slow_preload
    stub_turn(monkeypatch)
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    with TestClient(server_app.create_app()) as client:  # startup returned although the preload is blocked
        body = client.get("/ready")
        assert body.status_code == 503 and body.json()["status"] == "not_ready"
        assert body.json()["loading"] == ["guardrails", "knowledge_base"] and "still loading" in body.json()["message"]
        gate_guardrails.set()
        assert wait_until(lambda: server_app._readiness["guardrails"])
        half = client.get("/ready")
        assert half.status_code == 503 and half.json()["loading"] == ["knowledge_base"]
        gate_kb.set()
        assert wait_until(lambda: client.get("/ready").status_code == 200)
        done = client.get("/ready").json()
        assert done["status"] == "ready" and "loading" not in done
        chat = client.post("/chat", json={"message": "hi", "session_id": SESSION}, headers={"Authorization": "Bearer x"})
        assert chat.status_code == 200 and chat.json()["message"] == "Hello there from the stub"


def test_chat_before_ready_gets_a_polite_503_without_a_turn(slow_preload, monkeypatch):
    turns = []

    async def counting(session_id, message, *, store=None):
        turns.append(message)
        return pipeline.TurnResult(reply="x", rooms=None, success=True)

    monkeypatch.setattr(pipeline, "handle_turn", counting)
    started = time.monotonic()
    with TestClient(server_app.create_app()) as client:
        r = client.post("/chat", json={"message": USER_TEXT, "session_id": SESSION}, headers={"Authorization": "Bearer x"})
        assert time.monotonic() - started < 3  # no long hang
        assert r.status_code == 503 and r.headers["retry-after"] == str(server_app.STARTING_RETRY_AFTER_S)
        body = r.json()
        assert body["success"] is False and body["message"] == server_app.STARTING_REPLY
        assert "Traceback" not in r.text and USER_TEXT not in r.text
        with client.websocket_connect("/ws/chat?token=abc") as ws:  # the connection stays; the message gets the notice
            ws.send_json({"message": "hello", "session_id": SESSION})
            assert ws.receive_json() == {"event": "error", "detail": server_app.STARTING_REPLY}
    assert turns == []


def test_liveness_is_200_before_during_and_after_loading(slow_preload):
    gate_guardrails, gate_kb = slow_preload
    with TestClient(server_app.create_app()) as client:
        assert client.get("/health").status_code == 200 and client.get("/").status_code == 200
        gate_guardrails.set()
        gate_kb.set()
        assert wait_until(lambda: not server_app._startup["loading"])
        assert client.get("/health").status_code == 200


def test_a_failed_preload_ends_loading_and_chat_is_served_as_before(monkeypatch):
    def broken():
        raise RuntimeError("no model")

    monkeypatch.delenv(server_app.SKIP_PRELOAD_ENV)
    monkeypatch.setattr(pipeline, "preload", broken)
    monkeypatch.setattr(reuse, "get_knowledge_preload", lambda: broken)
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": False, "knowledge_base": False})
    stub_turn(monkeypatch)
    with TestClient(server_app.create_app()) as client:
        assert wait_until(lambda: not server_app._startup["loading"])
        assert client.get("/ready").status_code == 503 and "loading" not in client.get("/ready").json()
        chat = client.post("/chat", json={"message": "hi", "session_id": SESSION}, headers={"Authorization": "Bearer x"})
        assert chat.status_code == 200


def test_the_loading_flag_is_cleared_after_shutdown(slow_preload):
    gate_guardrails, gate_kb = slow_preload
    gate_guardrails.set()
    gate_kb.set()
    with TestClient(server_app.create_app()):
        pass
    assert server_app._startup["loading"] is False and server_app._readiness == {"guardrails": True, "knowledge_base": True}


def test_skipped_preload_leaves_chat_open(monkeypatch):
    stub_turn(monkeypatch)
    with TestClient(server_app.create_app()) as client:
        assert server_app._startup["loading"] is False
        r = client.post("/chat", json={"message": "hi", "session_id": SESSION}, headers={"Authorization": "Bearer x"})
        assert r.status_code == 200


# ---- /metrics: authentication and content -------------------------------------------------------


@pytest.fixture(autouse=True)
def _metrics_token(monkeypatch):
    monkeypatch.setenv(server_app.METRICS_TOKEN_ENV, METRICS_TOKEN)


@pytest.fixture
def client():
    with TestClient(server_app.create_app()) as c:
        yield c


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_metrics_refuses_unauthenticated_calls(client):
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers=bearer("wrong-token")).status_code == 401
    assert client.get("/metrics", headers=bearer("")).status_code == 401
    assert client.get("/metrics", headers={"Authorization": f"Basic {SERVER_TOKEN}"}).status_code == 401
    assert client.get("/metrics", headers={"Authorization": SERVER_TOKEN}).status_code == 401
    refused = client.get("/metrics")
    assert refused.headers["www-authenticate"] == "Bearer" and "request_count" not in refused.text


def test_metrics_accepts_only_the_dedicated_token(client):
    r = client.get("/metrics", headers=bearer(METRICS_TOKEN))
    assert r.status_code == 200 and "request_count" in r.json() and "active_sessions" in r.json()
    assert client.get("/metrics", headers=bearer(SERVER_TOKEN)).status_code == 401  # the server credential
    demo = client.get("/chat-ui/demo-token").json()["token"]
    assert client.get("/metrics", headers=bearer(demo)).status_code == 401  # a valid demo token
    assert client.get("/metrics", headers=bearer(METRICS_TOKEN + "x")).status_code == 401


@pytest.mark.parametrize("value", [None, "", "   "])
def test_metrics_is_off_without_a_token_and_warns_once_at_startup(monkeypatch, caplog, value):
    if value is None:
        monkeypatch.delenv(server_app.METRICS_TOKEN_ENV)
    else:
        monkeypatch.setenv(server_app.METRICS_TOKEN_ENV, value)
    with caplog.at_level("WARNING", logger="llm_approach.server_app"):
        with TestClient(server_app.create_app()) as c:
            for headers in ({}, bearer(""), bearer(value or "x"), bearer(SERVER_TOKEN)):
                assert c.get("/metrics", headers=headers).status_code == 404
    warnings = [r for r in caplog.records if "/metrics is disabled" in r.getMessage()]
    assert len(warnings) == 1 and METRICS_TOKEN_ENV_NAME in warnings[0].getMessage()


def test_the_secret_check_does_not_require_the_metrics_token(monkeypatch):
    from llm_approach.security import secrets_check

    monkeypatch.delenv(server_app.METRICS_TOKEN_ENV, raising=False)
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    secrets_check.validate_secrets()  # raises when a required secret is missing


def test_metrics_holds_no_user_text_room_text_keys_or_session_ids(client, monkeypatch):
    """Real turns carry sentinel strings; none of them may come back, and the credentials stay out too."""
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    calls = []

    async def fake_complete(messages, max_tokens=None, provider=None):  # never a real provider call
        calls.append(1)
        return Completion(text=json.dumps({"intent": "CHITCHAT", "confidence": 0.95, "reply": "Hello!"}),
                          tokens_in=1, tokens_out=1, cached=None)

    monkeypatch.setattr(providers, "complete", fake_complete)
    session = str(uuid.uuid4())
    observability.record_turn(request_id=USER_TEXT, session_id=session, intent=None, operation=None, tool=None,
                              mode="fast", status="success", error_category=None, latency_ms=3.0)
    client.post("/chat", json={"message": f"hello {USER_TEXT}", "session_id": session},
                headers=bearer(f"{ROOM_TEXT}-{KEY}"))
    body = client.get("/metrics", headers=bearer(METRICS_TOKEN)).text
    for secret in (USER_TEXT, ROOM_TEXT, KEY, SERVER_TOKEN, METRICS_TOKEN, session):
        assert secret not in body, f"{secret} leaked into /metrics"


# ---- /metrics: every counter shows up after its event --------------------------------------------


def snap(client):
    r = client.get("/metrics", headers=bearer(METRICS_TOKEN))
    assert r.status_code == 200
    return r.json()


def test_every_counter_appears_after_the_matching_event(client, api, monkeypatch):  # noqa: F811
    monkeypatch.setenv("KANGAROO_TOKEN", SERVER_TOKEN)  # the api fixture set its own
    before = snap(client)
    assert {"llm_failures", "llm_events", "room_data_rejections", "parameter_rejections", "search_timeouts",
            "semantic_gate", "session_store_events"} <= set(before)

    # LLM failure kind and breaker open/close (real breaker, limit 1)
    monkeypatch.setenv("LLM_APPROACH_BREAKER_FAILURES", "1")
    llm_resilience.note_failure("timeout", "gemini", primary=True)
    llm_resilience.breaker("gemini").record_success()
    llm_resilience.note_switch("gemini", "groq", "timeout")  # fallback switch

    # breaker_skip and the 5xx recovery: the real client with a fake provider
    monkeypatch.setenv("LLM_APPROACH_5XX_RETRY_DELAY_SECONDS", "0")
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "gemini")
    script = [type("Err", (Exception,), {"code": 503, "status": "UNAVAILABLE"})("boom")]
    text = json.dumps({"intent": "NEW_SEARCH", "confidence": 0.95, "slots": {"city": "Bangalore"}})

    async def fake_complete(messages, max_tokens=None, provider=None):
        item = script.pop(0) if script else text
        if isinstance(item, BaseException):
            raise item
        return Completion(text=item, tokens_in=1, tokens_out=1, cached=None)

    monkeypatch.setattr(providers, "complete", fake_complete)
    asyncio.run(llm_client.decide("rooms in bangalore", {}, llm_client.LLMBudget()))
    llm_resilience.breaker("gemini").record_failure()  # open it again so the next call is skipped
    asyncio.run(llm_client.decide("rooms in bangalore", {}, llm_client.LLMBudget()))

    # room data rejection, redis error, gate counters
    observability.record_room_data_rejection()
    observability.record_session_store_event("redis_unavailable")
    observability.record_semantic_gate("OUT_OF_SCOPE", True, False, 4.0)
    observability.record_semantic_gate_error()

    # parameter rejection (the real search action) and search timeout (the real rooms client)
    before_switches = switches.snapshot()
    switches.configure(search_arg_check=True)
    try:
        search._run_search({"city": "Bangalore", "bhk": 9}, LLMConversationState(), TurnBudget(), "new_search")
    finally:
        switches.configure(**before_switches)
    slow(api, monkeypatch)
    assert rooms_client.search_rooms({"city": "Bangalore"}, TurnBudget()).error_class == "search_timeout"

    after = snap(client)
    assert after["llm_failures"].get("timeout", 0) == before["llm_failures"].get("timeout", 0) + 1
    for event in ("breaker_open", "breaker_close", "fallback_switch", "recovered_5xx_retry", "breaker_skip"):
        assert after["llm_events"].get(event, 0) > before["llm_events"].get(event, 0), event
    assert after["room_data_rejections"] == before["room_data_rejections"] + 1
    assert after["session_store_events"].get("redis_unavailable", 0) == before["session_store_events"].get("redis_unavailable", 0) + 1
    for name in ("gate_checked", "gate_would_block", "gate_error"):
        assert after["semantic_gate"][name] == before["semantic_gate"][name] + 1, name
    assert sum(after["parameter_rejections"].values()) == sum(before["parameter_rejections"].values()) + 1
    assert after["search_timeouts"] == before["search_timeouts"] + 1
    api.block.set()


def test_a_search_timeout_is_counted_once_even_with_a_retry(api, monkeypatch):  # noqa: F811
    slow(api, monkeypatch)
    before = observability.get_metrics_snapshot()["search_timeouts"]
    rooms_client.search_rooms({"city": "Bangalore"}, TurnBudget())
    api.block.set()
    time.sleep(0.3)
    assert observability.get_metrics_snapshot()["search_timeouts"] == before + 1


# ---- scripts/gate_report.py reads /metrics with a token ---------------------------------------------


def test_gate_report_sends_the_token_in_a_header_and_never_prints_it(monkeypatch, capsys):
    import io
    import sys
    import urllib.error

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    import gate_report

    seen = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout):
        seen.append(request.get_header("Authorization"))
        if request.get_header("Authorization") != "Bearer tok-123":
            raise urllib.error.HTTPError(request.full_url, 401, "no", {}, None)
        return Resp(json.dumps({"semantic_gate": {"gate_checked": 4}}).encode())

    monkeypatch.setattr(gate_report.urllib.request, "urlopen", fake_urlopen)
    assert gate_report.main(["--url", "http://x:1"]) == 1 and "--token" in capsys.readouterr().out
    assert gate_report.main(["--url", "http://x:1", "--token", "tok-123"]) == 0
    out = capsys.readouterr().out
    assert '"gate_checked": 4' in out and "tok-123" not in out
