"""Independent-audit fixes: eviction of the live session, env parsing, WebSocket robustness,
per-call latency with p50/p95, and scrubbing of rooms-API error text. Offline."""

from __future__ import annotations

import asyncio
import importlib
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from llm_approach import config, llm_client, metrics, pipeline, reuse, rooms_client, server_app
from llm_approach.state import SessionStore

from .conftest import dec
from .test_server import SECRET, server_env, stub_turn, ws_turn  # noqa: F401 - server_env is an autouse fixture


# ---- SessionStore: the session in use is never the one evicted ----------------------------


def test_new_session_is_not_evicted_when_every_older_one_is_busy():
    store = SessionStore(ttl_seconds=100, max_sessions=1, clock=lambda: 0.0)
    busy, fresh = str(uuid.uuid4()), str(uuid.uuid4())

    async def run():
        async with store.turn(busy):
            state = store.get_or_create(fresh)  # `busy` cannot be evicted; `fresh` must survive too
            assert store.peek(fresh) is state
            assert store.count() == 2  # briefly over the cap, as documented

    asyncio.run(run())
    store.get_or_create(fresh)  # idle again: trimmed back to the cap, oldest first
    assert store.count() == 1 and store.peek(fresh) is not None and store.peek(busy) is None


def test_turns_on_a_new_session_keep_their_history_when_the_cap_is_full_of_busy_sessions():
    store = SessionStore(ttl_seconds=100, max_sessions=1, clock=lambda: 0.0)
    busy, sid = str(uuid.uuid4()), str(uuid.uuid4())

    async def run():
        async with store.turn(busy):
            async with store.turn(sid, "hello") as state:
                state.record_turn("hello", "hi")
            async with store.turn(sid, "again") as state:
                return [m["content"] for m in state.history]

    assert asyncio.run(run()) == ["hello", "hi"]


# ---- config: a bad environment value is ignored, never crashes or echoes ---------------------


@pytest.mark.parametrize("bad", ["abc", "", "-5", "0", "12.5"])
def test_session_limits_fall_back_to_defaults_on_bad_env(monkeypatch, bad):
    for name in ("SESSION_TTL_SECONDS", "SESSION_MAX_COUNT", "SESSION_MAX_HISTORY_MESSAGES"):
        monkeypatch.setenv(name, bad)
    try:
        reloaded = importlib.reload(config)
        assert (reloaded.SESSION_TTL_SECONDS, reloaded.SESSION_MAX_COUNT, reloaded.SESSION_MAX_HISTORY_MESSAGES) == (
            3600, 1000, 40,
        )
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_session_limits_read_valid_env(monkeypatch):
    monkeypatch.setenv("SESSION_MAX_COUNT", "7")
    try:
        assert importlib.reload(config).SESSION_MAX_COUNT == 7
    finally:
        monkeypatch.undo()
        importlib.reload(config)


# ---- WebSocket robustness --------------------------------------------------------------------


@pytest.fixture
def client():
    with TestClient(server_app.create_app()) as c:
        yield c


def test_ws_binary_frame_is_an_error_event_and_connection_stays_open(client, monkeypatch):
    stub_turn(monkeypatch)
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        ws.send_bytes(b"\x00\x01\x02")
        assert ws.receive_json() == {"event": "error", "detail": "Message must be valid JSON."}
        assert ws_turn(ws, message="hi", session_id=None, mode="fast")[-1]["event"] == "done"


@pytest.mark.parametrize("value", ["0", "-3", "nan", "inf", "junk"])
def test_ws_heartbeat_interval_never_spins(monkeypatch, value):
    monkeypatch.setenv("WS_HEARTBEAT_INTERVAL_SECONDS", value)
    assert server_app._float_env("WS_HEARTBEAT_INTERVAL_SECONDS", 20.0, minimum=0.01) == 20.0


def test_ws_heartbeat_accepts_a_small_positive_interval(monkeypatch):
    monkeypatch.setenv("WS_HEARTBEAT_INTERVAL_SECONDS", "0.05")
    assert server_app._float_env("WS_HEARTBEAT_INTERVAL_SECONDS", 20.0, minimum=0.01) == 0.05


def test_ws_zero_heartbeat_env_does_not_break_a_normal_turn(client, monkeypatch):
    monkeypatch.setenv("WS_HEARTBEAT_INTERVAL_SECONDS", "0")  # invalid: falls back to 20 s, no ping storm
    stub_turn(monkeypatch)
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        events = ws_turn(ws, message="hi", session_id=None, mode="fast")
    assert [e["event"] for e in events if e["event"] == "ping"] == []
    assert events[-1]["event"] == "done"


def test_ws_long_invalid_mode_is_not_echoed_in_full(client, monkeypatch):
    stub_turn(monkeypatch)
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        event = ws_turn(ws, message="hi", session_id=None, mode="x" * 400)[-1]
    assert event["event"] == "error" and event["detail"].startswith("Invalid mode: 'xxx")
    assert len(event["detail"]) < 60


def test_ws_failure_building_the_reply_closes_cleanly_and_logs_class_only(client, monkeypatch, caplog):
    stub_turn(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("internal detail " + SECRET)

    monkeypatch.setattr(server_app, "build_done", boom)
    with client.websocket_connect(f"/ws/chat?token={SECRET}") as ws:
        ws.send_json({"message": "hi", "session_id": None, "mode": "fast"})
        assert ws.receive_json() == {"event": "typing"}
        with pytest.raises(Exception) as info:  # the server closes the socket (1011)
            ws.receive_json()
        assert getattr(info.value, "code", 1011) == 1011
    assert "RuntimeError" in caplog.text
    assert "internal detail" not in caplog.text and SECRET not in caplog.text


# ---- metrics: per-call latency, call count, p50 / p95 ----------------------------------------


def test_llm_budget_records_each_call_latency(fake_groq):
    fake_groq(["not json", json.dumps(dec("GENERAL_CHAT", reply="Hello!"))])
    budget = llm_client.LLMBudget()
    decision = asyncio.run(llm_client.decide("hi", {}, budget))
    assert decision is not None and budget.calls == 2
    assert len(budget.call_ms) == 2 and all(v >= 0 for v in budget.call_ms)
    assert budget.latency_ms == pytest.approx(sum(budget.call_ms))


def test_error_calls_are_recorded_too(fake_groq):
    fake_groq([ConnectionError("down")])
    budget = llm_client.LLMBudget()
    assert asyncio.run(llm_client.decide("hi", {}, budget)) is None
    assert len(budget.call_ms) == 1 and budget.latency_ms == budget.call_ms[0]


def test_turn_metrics_carry_call_count_and_latencies_without_text(fake_groq, monkeypatch, tmp_path):
    path = tmp_path / "m.jsonl"
    monkeypatch.setenv("LLM_APPROACH_METRICS", "1")
    monkeypatch.setenv("LLM_APPROACH_METRICS_PATH", str(path))
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    pipeline.configure(full_guardrails=True)
    fake_groq(["garbage", json.dumps(dec("GENERAL_CHAT", reply="Happy to chat!"))])
    result = asyncio.run(pipeline.handle_turn(str(uuid.uuid4()), "zebrafish-quartz hello", store=SessionStore()))
    assert result.metrics["llm_calls"] == 2 and len(result.metrics["llm_call_ms"]) == 2
    text = path.read_text(encoding="utf-8")
    row = json.loads(text)
    assert row["llm_calls"] == 2 and all(isinstance(v, float) for v in row["llm_call_ms"])
    assert "zebrafish" not in text


def test_percentile_nearest_rank():
    assert metrics.percentile([], 50) is None
    assert metrics.percentile([5], 95) == 5
    values = list(range(1, 101))
    assert (metrics.percentile(values, 50), metrics.percentile(values, 95), metrics.percentile(values, 100)) == (50, 95, 100)


def test_summarize_reads_p50_p95_from_the_file(tmp_path):
    path = tmp_path / "m.jsonl"
    rows = [
        {"total_ms": 100.0 * i, "llm_ms": 50.0 * i, "llm_call_ms": [50.0 * i], "fallback_used": False, "error_class": None}
        for i in range(1, 11)
    ]
    rows.append({"total_ms": 10.0, "llm_ms": 0.0, "llm_call_ms": [], "fallback_used": True, "error_class": "llm_unavailable"})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n[1]\n", encoding="utf-8")
    s = metrics.summarize(path)
    assert s["turns"] == 11 and s["fallback_turns"] == 1 and s["error_turns"] == 1
    assert s["total_ms"]["n"] == 11 and s["total_ms"]["p95"] == 1000.0
    assert s["llm_ms"] == {"n": 10, "p50": 250.0, "p95": 500.0}  # the turn with no Groq call is excluded
    assert s["per_groq_call_ms"] == {"n": 10, "p50": 250.0, "p95": 500.0}


def test_summarize_missing_file_is_empty_not_an_error(tmp_path):
    s = metrics.summarize(tmp_path / "nope.jsonl")
    assert s["turns"] == 0 and s["total_ms"] == {"n": 0, "p50": None, "p95": None}


# ---- rooms-API error text --------------------------------------------------------------------


def test_scrub_removes_list_values_such_as_room_types():
    out = rooms_client.scrub_message("bad roomTypes 2BHK,Single here", {"city": "Pune", "room_types": ["2BHK", "Single"]})
    assert "2BHK" not in out and "Single" not in out and "Pune" not in out


def test_error_message_is_scrubbed_before_it_is_cut():
    value = "Zorbaville"
    body = {"message": "x" * 195 + value + " tail"}  # the value straddles the 200-character cut
    msg = rooms_client._error_message(body, {"city": value})
    assert len(msg) <= 200 and "Zorb" not in msg and "rbaville" not in msg
