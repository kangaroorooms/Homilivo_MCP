"""Startup warm-up: one tiny Gemini call and one light rooms request before the first user message.

Stubs only: no Gemini, Groq, rooms or pricing call is made.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_approach import config, gemini_limits, llm_resilience, pipeline, providers, reuse, rooms_client, server_app

KEY = "AIza-warmup-key-must-not-appear"


class _FakeModels:
    def __init__(self, owner):
        self.owner = owner

    async def generate_content(self, model, contents, config):
        self.owner.calls.append((model, contents, config))
        if self.owner.error:
            raise self.owner.error
        return object()


class _FakeClient:
    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self.aio = type("Aio", (), {})()
        self.aio.models = _FakeModels(self)


@pytest.fixture
def warm(monkeypatch, tmp_path):
    """Gemini active, a fake client, a stub rooms request, a temp request counter, an empty preload."""
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.delenv("LLM_APPROACH_WARMUP", raising=False)
    monkeypatch.delenv(config.OFFLINE_ROOMS_ENV, raising=False)
    client = _FakeClient()
    monkeypatch.setattr(providers, "_get_gemini_client", lambda: client)
    rooms_calls = []
    monkeypatch.setattr(rooms_client, "request_once", lambda params, **kw: rooms_calls.append((params, kw)))
    limiter = gemini_limits.GeminiLimiter(counter=gemini_limits.UsageCounter(path=tmp_path / "count.json"))
    gemini_limits.set_limiter(limiter)
    monkeypatch.setattr(pipeline, "preload", lambda: None)
    monkeypatch.setattr(reuse, "get_knowledge_preload", lambda: (lambda: None))
    monkeypatch.setattr(reuse, "get_knowledge_index_stats",
                        lambda: (lambda: {"chunks_indexed": 3, "embedding_model_available": True}))
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": False, "knowledge_base": False})
    monkeypatch.setattr(server_app, "_knowledge", {"embedding_model": False, "chunks_indexed": 0})
    yield type("W", (), {"client": client, "rooms": rooms_calls, "limiter": limiter})
    gemini_limits.set_limiter(None)


def test_warmup_runs_once_each(warm):
    server_app._warmup_blocking()
    assert len(warm.client.calls) == 1 and len(warm.rooms) == 1
    _model, contents, cfg = warm.client.calls[0]
    assert cfg["max_output_tokens"] <= 16 and "tools" not in cfg
    assert warm.rooms[0][0] == {"limit": 1} and warm.rooms[0][1]["timeout"] == server_app.WARMUP_ROOMS_TIMEOUT_S


def test_env_zero_skips_it(warm, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_WARMUP", "0")
    server_app._warmup_blocking()
    assert warm.client.calls == [] and warm.rooms == []
    assert warm.limiter.counter.used_requests() == 0


def test_failures_do_not_raise_and_log_the_class_only(warm, monkeypatch, caplog):
    warm.client.error = RuntimeError(f"boom {KEY}")
    monkeypatch.setattr(rooms_client, "request_once", lambda *a, **k: (_ for _ in ()).throw(ConnectionError(f"x {KEY}")))
    with caplog.at_level(logging.INFO):
        server_app._warmup_blocking()
    text = caplog.text
    assert "step=warmup_gemini failed: RuntimeError" in text and "step=warmup_rooms failed: ConnectionError" in text
    assert KEY not in text


def test_success_logs_one_info_line_per_step_without_secrets(warm, caplog):
    with caplog.at_level(logging.INFO):
        server_app._warmup_blocking()
    assert "startup step=warmup_gemini seconds=" in caplog.text and "startup step=warmup_rooms seconds=" in caplog.text
    assert KEY not in caplog.text and "ok" not in [r.getMessage() for r in caplog.records]


def test_breaker_untouched_and_one_counted_request(warm):
    warm.client.error = RuntimeError("down")  # even a failing warm-up must not feed the breaker
    server_app._warmup_blocking()
    breaker = llm_resilience.breaker("gemini")
    assert breaker.allow() and breaker.opened_at is None and breaker.consecutive == 0
    assert warm.limiter.counter.used_requests() == 1
    assert len(warm.limiter._window) == 0  # the 60 s window of real traffic is not used


def test_skipped_when_the_daily_cap_is_reached(warm, monkeypatch):
    monkeypatch.setenv(config.GEMINI_REQUEST_BUDGET_ENV, "1")
    warm.limiter.counter.add(1)
    server_app._warmup_blocking()
    assert warm.client.calls == [] and warm.limiter.counter.used_requests() == 1


def test_no_rooms_request_for_offline_rooms(warm, monkeypatch):
    monkeypatch.setenv(config.OFFLINE_ROOMS_ENV, "1")
    server_app._warmup_blocking()
    assert warm.rooms == [] and len(warm.client.calls) == 1


def test_groq_provider_makes_no_gemini_call(warm, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    server_app._warmup_blocking()
    assert warm.client.calls == []


def test_startup_not_ready_until_warmup_ends_and_a_failure_still_ends_it(warm, monkeypatch):
    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")  # the apps below must not start their own preload
    monkeypatch.setenv("KANGAROO_TOKEN", "t")
    seen = {}

    def gemini():
        with TestClient(server_app.create_app()) as inner:  # a second app: only reads /ready
            seen["ready"] = inner.get("/ready").status_code
            seen["loading"] = inner.get("/ready").json().get("loading")
        raise RuntimeError("warm-up failed")

    monkeypatch.setattr(server_app, "_warmup_gemini", gemini)
    monkeypatch.setattr(server_app, "_startup", {"loading": True, "warmup": False})
    asyncio.run(server_app._preload_background())
    assert seen["ready"] == 503 and "warmup" in seen["loading"]
    assert server_app._startup == {"loading": False, "warmup": False}
    with TestClient(server_app.create_app()) as client:
        assert client.get("/ready").status_code == 200


def test_the_real_gemini_client_is_the_cached_one(monkeypatch):
    """The warm-up asks providers._get_gemini_client, not a client of its own."""
    src = Path(server_app.__file__).read_text(encoding="utf-8")
    assert "providers._get_gemini_client()" in src


# ---- pricing warm-up ----------------------------------------------------------------


@pytest.fixture
def pricing(monkeypatch):
    from llm_approach import pricing_client

    calls = []
    monkeypatch.setattr(reuse, "get_known_cities", lambda: {"mumbai", "bangalore", "bad<city>"})
    monkeypatch.setattr(pricing_client, "warm_request", lambda city, timeout, **kw: calls.append((city, timeout)))
    return calls


def test_warmup_calls_pricing_once_with_a_known_city(warm, pricing):
    server_app._warmup_blocking()
    assert pricing == [("bangalore", server_app.WARMUP_PRICING_TIMEOUT_S)]  # sorted known list; invalid names skipped


def test_env_zero_skips_pricing_too(warm, pricing, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_WARMUP", "0")
    server_app._warmup_blocking()
    assert pricing == []


def test_offline_skips_pricing(warm, pricing, monkeypatch):
    monkeypatch.setenv(config.OFFLINE_ROOMS_ENV, "1")
    server_app._warmup_blocking()
    assert pricing == []


def test_failing_pricing_warmup_does_not_break_startup(warm, monkeypatch, caplog):
    from llm_approach import pricing_client

    monkeypatch.setattr(reuse, "get_known_cities", lambda: {"mumbai"})
    monkeypatch.setattr(pricing_client, "warm_request",
                        lambda *a, **k: (_ for _ in ()).throw(TimeoutError(f"slow {KEY}")))
    with caplog.at_level(logging.INFO):
        server_app._warmup_blocking()
    assert "step=warmup_pricing failed: TimeoutError" in caplog.text and KEY not in caplog.text
    assert len(warm.client.calls) == 1 and len(warm.rooms) == 1  # the other steps still ran


def test_slow_pricing_warmup_is_bounded_by_its_own_timeout(warm, monkeypatch):
    """The step passes the 5 s bound to the request; /ready waits only that long for it."""
    from llm_approach import pricing_client

    seen = {}
    monkeypatch.setattr(reuse, "get_known_cities", lambda: {"mumbai"})
    monkeypatch.setattr(pricing_client, "warm_request", lambda city, timeout, **kw: seen.setdefault("timeout", timeout))
    server_app._warmup_blocking()
    assert seen["timeout"] == server_app.WARMUP_PRICING_TIMEOUT_S == 5.0


def test_a_failing_pricing_step_still_clears_the_warmup_flag(warm, monkeypatch):
    from llm_approach import pricing_client

    monkeypatch.setattr(reuse, "get_known_cities", lambda: {"mumbai"})
    monkeypatch.setattr(pricing_client, "warm_request", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(server_app, "_startup", {"loading": True, "warmup": False})
    asyncio.run(server_app._preload_background())
    assert server_app._startup == {"loading": False, "warmup": False}


def test_warm_request_is_one_get_without_retry(monkeypatch):
    from llm_approach import pricing_client

    monkeypatch.delenv(config.OFFLINE_ROOMS_ENV, raising=False)
    calls = []

    def boom(url, params, headers, timeout):
        calls.append((params, timeout))
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        pricing_client.warm_request("mumbai", 5.0, http_get=boom)
    assert calls == [({"city": "mumbai"}, 5.0)]
