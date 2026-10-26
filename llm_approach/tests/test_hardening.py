"""Five hardening fixes: SHOW_MORE off the event loop, rule-based fallback on 429 / daily cap / open breaker,
fail-open Redis rate limiter and demo-token store, blocked replies drop their cards, honest /ready.

Fakes only: no Gemini, Groq, rooms API or real Redis call is made.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_approach import config, llm_client, observability, pipeline, reuse, rooms_client, server_app
from llm_approach.schema import Intent
from llm_approach.security import demo_auth, fail_open, rate_limit
from llm_approach.state import SessionStore
from llm_approach.tests.conftest import dec

HERE = Path(__file__).resolve().parent


# ---- 1. SHOW_MORE runs off the event loop ------------------------------------------------------------------


def test_show_more_is_a_threaded_intent():
    assert Intent.SHOW_MORE in pipeline._THREADED


def test_a_slow_rooms_call_in_show_more_does_not_block_other_coroutines(pipe, monkeypatch):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    pipe.state.has_next = True  # the last result said there is a next page
    inner = rooms_client.search_rooms

    def slow_rooms(search, budget, **kw):
        time.sleep(0.4)  # a blocking call, as the real rooms client is
        return inner(search, budget, **kw)

    monkeypatch.setattr(rooms_client, "search_rooms", slow_rooms)
    ticks = []

    async def ticker():
        while True:
            await asyncio.sleep(0.02)
            ticks.append(1)

    async def main():
        task = asyncio.create_task(ticker())
        try:
            return await pipe.turn_async("show more", dec("SHOW_MORE"))
        finally:
            task.cancel()

    result = asyncio.run(main())
    assert result.metrics["intent"] == "SHOW_MORE" and result.metrics["action"].startswith("show_more")
    assert len(ticks) >= 10, "the event loop was blocked while the rooms call ran"  # about 20 when it is free


# ---- 2. the rule-based fallback runs on 429, the daily cap and an open breaker -----------------------------


def _failing_decide(monkeypatch, *, error_class, breaker=False, kind=None):
    async def decide(user_message, state_summary, budget):
        budget.error_class = error_class
        budget.breaker_open = breaker
        budget.error_kind = kind
        return None

    monkeypatch.setattr(llm_client, "decide", decide)


def _fallback_turn(pipe, message="rooms in pune"):
    pipe.sid = str(uuid.uuid4())
    return asyncio.run(pipe.turn_async(message, "never"))


@pytest.mark.parametrize("case,kwargs", [
    ("429", dict(error_class="RateLimitError")),
    ("daily_cap", dict(error_class="RateLimitError", kind="request_cap")),
    ("open_breaker", dict(error_class="BreakerOpen", breaker=True)),
])
def test_fallback_runs_with_the_same_wording_as_any_other_llm_failure(pipe, monkeypatch, case, kwargs):
    _failing_decide(monkeypatch, error_class="TimeoutError")
    reference = _fallback_turn(pipe)  # the LLM is "otherwise unavailable"
    assert reference.metrics["fallback_used"] is True and reference.metrics["tool_called"] is True

    _failing_decide(monkeypatch, **kwargs)
    result = _fallback_turn(pipe)
    assert result.reply == reference.reply
    assert result.reply not in (pipeline.RATE_LIMIT_REPLY, "Sorry, I'm unavailable right now. Please try again shortly.")
    assert result.metrics["fallback_used"] is True and result.metrics["tool_called"] is True
    assert result.metrics["action"] == reference.metrics["action"] and result.metrics["action"].startswith("fallback_")
    assert result.metrics["error_class"] == kwargs["error_class"]  # the cause stays countable


def test_fallback_without_a_city_asks_for_it_when_the_llm_is_throttled(pipe, monkeypatch):
    _failing_decide(monkeypatch, error_class="RateLimitError")
    result = _fallback_turn(pipe, "hello there")
    assert result.metrics["action"] == "fallback_ask" and result.metrics["tool_called"] is False


def test_the_300_turn_stub_baseline_only_moved_for_the_no_decision_rate_limit():
    baseline = json.loads((HERE / "stub_scenarios_baseline.json").read_text(encoding="utf-8"))
    assert sum(len(v) for v in baseline.values()) == 300
    limited = [(sid, n + 1) for sid, turns in baseline.items() for n, (_i, action) in enumerate(turns)
               if action in ("rate_limited", "llm_unavailable")]
    assert limited == [("S65", 2)]  # the answerer's 429 (a decision exists, so no fallback applies): unchanged
    assert baseline["S24"][1] == [None, "fallback_ask"]  # the one entry fix 2 changes; run_scenarios compares the rest


# ---- 3. Redis rate limiter and demo-token store fail open --------------------------------------------------


class DownRedis:
    """A Redis whose every command times out (or raises ``exc``)."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    def _boom(self, *a, **k):
        self.calls += 1
        raise self.exc

    pipeline = set = get = delete = hgetall = _boom


def _events():
    return dict(observability.get_metrics_snapshot()["session_store_events"])


@pytest.mark.parametrize("exc", [TimeoutError("redis timed out"), ConnectionError("down"), RuntimeError("anything")])
def test_rate_limiter_allows_the_request_when_redis_fails(exc, caplog):
    before = _events().get(fail_open.RATE_LIMIT_EVENT, 0)
    limiter = fail_open.FailOpenRateLimiter(rate_limit.RedisRateLimiter(redis_client=DownRedis(exc)))
    with caplog.at_level("DEBUG"):
        result = limiter.hit("session:SECRETSESSION-123", 5, 60.0)
    assert result.allowed is True and result.retry_after_seconds == 0.0
    assert _events().get(fail_open.RATE_LIMIT_EVENT, 0) == before + 1
    assert "SECRETSESSION-123" not in caplog.text and "redis timed out" not in caplog.text  # class name only
    assert type(exc).__name__ in caplog.text


def test_check_rate_limit_and_the_session_check_allow_when_the_backend_is_down(monkeypatch):
    down = fail_open.FailOpenRateLimiter(rate_limit.RedisRateLimiter(redis_client=DownRedis(TimeoutError())))
    monkeypatch.setattr(rate_limit, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(rate_limit, "get_rate_limiter", lambda: down)
    assert rate_limit.check_rate_limit("1.2.3.4").allowed is True
    assert rate_limit.check_session_rate_limit(str(uuid.uuid4())).allowed is True


def test_a_working_redis_limiter_still_limits():
    import fakeredis

    limiter = fail_open.FailOpenRateLimiter(rate_limit.RedisRateLimiter(redis_client=fakeredis.FakeStrictRedis(decode_responses=True)))
    assert [limiter.hit("k", 2, 60.0, now=1000.0).allowed for _ in range(3)] == [True, True, False]


def test_redis_clients_get_a_short_timeout(monkeypatch):
    seen = []
    monkeypatch.setattr(rate_limit.redis_lib.Redis, "from_url", staticmethod(lambda url, **kw: seen.append(kw) or object()))
    monkeypatch.setenv(config.REDIS_TIMEOUT_ENV, "0.25")
    rate_limit.RedisRateLimiter(url="redis://example:6379/0")
    demo_auth.RedisDemoAuthBackend(url="redis://example:6379/0")
    assert len(seen) == 2
    for kw in seen:
        assert kw["socket_timeout"] == 0.25 and kw["socket_connect_timeout"] == 0.25


@pytest.mark.parametrize("exc", [TimeoutError("redis timed out"), ConnectionError("down")])
def test_demo_tokens_fall_back_to_memory_when_redis_fails(monkeypatch, exc, caplog):
    demo_auth._tokens.clear()
    monkeypatch.setenv("KANGAROO_TOKEN", "server-side-credential")
    backend = fail_open.FailOpenDemoAuthBackend(demo_auth.RedisDemoAuthBackend(redis_client=DownRedis(exc)),
                                                demo_auth.InMemoryDemoAuthBackend())
    monkeypatch.setattr(demo_auth, "get_demo_auth_backend", lambda: backend)
    before = _events().get(fail_open.DEMO_TOKEN_EVENT, 0)
    try:
        with caplog.at_level("DEBUG"):
            token = demo_auth.issue_demo_token()
            assert token and demo_auth.redeem_demo_token(token) == "server-side-credential"
            assert demo_auth.redeem_demo_token("not-a-demo-token") is None
        assert _events().get(fail_open.DEMO_TOKEN_EVENT, 0) > before
        assert token not in caplog.text and "server-side-credential" not in caplog.text
    finally:
        demo_auth._tokens.clear()


# ---- 4. a blocked reply drops its cards --------------------------------------------------------------------


def test_blocked_search_reply_sends_no_cards_and_keeps_none_on_screen(pipe):
    pipe.output_ok = False
    result = pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert result.reply == pipeline.OUTPUT_BLOCKED_REPLY and result.rooms == []
    assert pipe.state.rooms_on_screen == []  # the user never saw them: "the second one" must not point at them
    done = server_app.build_done(result, pipe.sid, "fast")
    assert done["rooms"] == [] and done["room_count"] == 0 and done["selected_room"] is None


def test_an_unblocked_search_still_sends_its_cards(pipe):
    result = pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    done = server_app.build_done(result, pipe.sid, "fast")
    assert done["room_count"] == len(pipe.state.rooms_on_screen) > 0


# ---- 5. /ready reports the real knowledge-base state -------------------------------------------------------


def _ready(monkeypatch, stats):
    monkeypatch.setattr(pipeline, "preload", lambda: None)
    monkeypatch.setattr(reuse, "get_knowledge_preload", lambda: (lambda: None))
    monkeypatch.setattr(reuse, "get_knowledge_index_stats", lambda: (lambda: stats))
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": False, "knowledge_base": False})
    monkeypatch.setattr(server_app, "_knowledge", {"embedding_model": False, "chunks_indexed": 0})
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_value_must_not_appear")
    server_app._preload_blocking()
    with TestClient(server_app.create_app()) as client:
        return client.get("/ready")


def test_ready_when_model_and_collection_are_there(monkeypatch):
    r = _ready(monkeypatch, {"documents_loaded": 2, "chunks_indexed": 7, "embedding_model_available": True})
    assert r.status_code == 200 and {k: v for k, v in r.json()["knowledge"].items() if k != "placeholder"} == {"embedding_model": True, "chunks_indexed": 7}  # v2: placeholder has its own test
    assert r.json()["checks"]["knowledge_base"] is True


def test_not_ready_when_the_embedding_model_did_not_load(monkeypatch):
    r = _ready(monkeypatch, {"documents_loaded": 2, "chunks_indexed": 0, "embedding_model_available": False})
    assert r.status_code == 503 and r.json()["checks"]["knowledge_base"] is False
    assert {k: v for k, v in r.json()["knowledge"].items() if k != "placeholder"} == {"embedding_model": False, "chunks_indexed": 0}  # v2: placeholder has its own test


def test_not_ready_when_the_collection_is_empty(monkeypatch):
    r = _ready(monkeypatch, {"documents_loaded": 0, "chunks_indexed": 0, "embedding_model_available": True})
    assert r.status_code == 503 and r.json()["checks"]["knowledge_base"] is False
    assert {k: v for k, v in r.json()["knowledge"].items() if k != "placeholder"} == {"embedding_model": True, "chunks_indexed": 0}  # v2: placeholder has its own test
    assert "gsk_value_must_not_appear" not in r.text


def test_not_ready_with_the_real_store_when_its_embedding_model_fails(monkeypatch, tmp_path):
    """The real rag.store (Chroma in a temp folder) with an embedding model that cannot load."""
    from llm_approach.rag import embeddings, store

    def no_model():
        raise OSError("model files missing")

    monkeypatch.setenv(store.PERSIST_DIR_ENV, str(tmp_path / "chroma"))
    monkeypatch.setattr(embeddings, "get_shared_model", no_model)
    store._reset_caches()
    monkeypatch.setattr(pipeline, "preload", lambda: None)
    monkeypatch.setattr(reuse, "get_knowledge_index_stats", lambda: store.index_stats)
    monkeypatch.setattr(reuse, "get_knowledge_preload", lambda: store.warmup)
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": True, "knowledge_base": False})
    monkeypatch.setattr(server_app, "_knowledge", {"embedding_model": False, "chunks_indexed": 0})
    try:
        server_app._preload_blocking()
        with TestClient(server_app.create_app()) as client:
            r = client.get("/ready")
        assert r.status_code == 503 and r.json()["knowledge"]["embedding_model"] is False
    finally:
        store._reset_caches()
