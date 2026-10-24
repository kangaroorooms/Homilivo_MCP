"""LLM failure handling (Phase 3A): defined outcomes, optional fallback provider, circuit breaker.

Fakes only: ``providers.complete`` is replaced by a scripted router, so no Gemini, Groq or rooms API call
is made. The real ``llm_client`` and ``pipeline`` run on top of it.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from llm_approach import config, gemini_limits, llm_client, llm_resilience, observability, pipeline, providers
from llm_approach.providers import Completion
from llm_approach.state import ERROR_REPLY, SessionStore

SEARCH = json.dumps({"intent": "NEW_SEARCH", "confidence": 0.95, "slots": {"city": "Bangalore"}})
CHAT = json.dumps({"intent": "GENERAL_CHAT", "confidence": 0.95})


class HttpError(Exception):
    def __init__(self, code: int, status: str | None = None, text: str = "boom") -> None:
        super().__init__(text)
        self.code, self.status = code, status


class Router:
    """Scripted ``providers.complete``: replies per provider name (a str, or an exception to raise)."""

    def __init__(self) -> None:
        self.script: dict[str, list] = {"gemini": [], "groq": []}
        self.calls: list[str] = []

    async def __call__(self, messages, max_tokens=None, provider=None):
        name = provider or config.provider()
        self.calls.append(name)
        item = self.script[name].pop(0) if self.script[name] else SEARCH
        if isinstance(item, BaseException):
            raise item
        return Completion(text=item, tokens_in=10, tokens_out=5, cached=None)

    def count(self, name: str) -> int:
        return self.calls.count(name)


@pytest.fixture
def env(monkeypatch):
    router = Router()
    monkeypatch.setattr(providers, "complete", router)
    monkeypatch.setattr(pipeline, "_full_guardrails", False)
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)
    monkeypatch.delenv("LLM_APPROACH_METRICS", raising=False)
    router.store, router.sid = SessionStore(), str(uuid.uuid4())

    def turn(message="rooms in bangalore"):
        return asyncio.run(pipeline.handle_turn(router.sid, message, store=router.store))

    router.turn = turn
    return router


def counters():
    snap = observability.get_metrics_snapshot()
    return dict(snap["llm_failures"]), dict(snap["llm_events"])


def delta(before, after):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) != before.get(k, 0)}


FAILURES = {
    "timeout": (asyncio.TimeoutError(), "timeout"),
    "rate_limit_429": (HttpError(429, "RESOURCE_EXHAUSTED", "slow down"), "rate_limit"),
    "daily_quota": (gemini_limits.DailyQuotaExhausted(), "daily_quota"),
    "server_error_503": (HttpError(503, "UNAVAILABLE"), "server_error"),
    "key_invalid": (HttpError(400, "INVALID_ARGUMENT", "API key not valid"), "auth"),
    "key_missing": (KeyError("GEMINI_API_KEY"), "auth"),
    "provider_down": (ConnectionError("no route"), "server_error"),
}


@pytest.mark.parametrize("case", FAILURES)
def test_every_failure_ends_in_a_safe_reply_and_the_session_stays_usable(env, case):
    exc, kind = FAILURES[case]
    env.script["gemini"] = [exc]
    before = counters()
    result = env.turn()
    assert result.reply and result.reply != ERROR_REPLY
    assert "Traceback" not in result.reply and "Error" not in result.reply and "boom" not in result.reply
    assert delta(before[0], counters()[0]) == {kind: 1}
    state = env.store.peek(env.sid)
    assert len(state.history) == 2 and state.turn_count == 1  # one finished pair, nothing half-saved
    env.script["gemini"] = [CHAT]
    again = env.turn("hello")  # the next message works
    assert again.success and again.reply != ERROR_REPLY
    assert len(env.store.peek(env.sid).history) == 4


def test_bad_json_twice_is_counted_and_the_session_stays_usable(env):
    env.script["gemini"] = ["not json", "still not json"]
    before = counters()
    result = env.turn()
    assert result.reply and result.reply != ERROR_REPLY
    assert env.count("gemini") == 2
    assert delta(before[0], counters()[0]) == {"bad_json": 1}
    env.script["gemini"] = [CHAT]
    assert env.turn("hello").success


def test_daily_quota_and_rate_limit_keep_their_existing_polite_reply(env):
    env.script["gemini"] = [gemini_limits.DailyQuotaExhausted()]
    assert env.turn().reply == pipeline.RATE_LIMIT_REPLY
    env.script["gemini"] = [HttpError(429, "RESOURCE_EXHAUSTED")]
    assert env.turn().reply == pipeline.RATE_LIMIT_REPLY


def test_failure_logs_are_warnings_without_user_text(env, caplog):
    secret_text = "rooms in bangalore under 12345 xyzzy"
    env.script["gemini"] = [HttpError(503, "UNAVAILABLE", "upstream said xyzzy")]
    with caplog.at_level("WARNING", logger="llm_approach"):
        env.turn(secret_text)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("llm failure kind=server_error" in r.getMessage() for r in warnings)
    assert all("xyzzy" not in r.getMessage() and "12345" not in r.getMessage() for r in caplog.records)


# ---- fallback provider ----------------------------------------------------------------------------------


def test_fallback_is_off_by_default(env):
    assert config.fallback_provider() is None
    env.script["gemini"] = [asyncio.TimeoutError()]
    env.turn()
    assert env.count("groq") == 0


@pytest.mark.parametrize("case", ["timeout", "rate_limit_429", "daily_quota", "server_error_503", "provider_down"])
def test_fallback_answers_the_same_turn_once(env, monkeypatch, case):
    monkeypatch.setenv("LLM_APPROACH_FALLBACK_PROVIDER", "groq")
    env.script["gemini"] = [FAILURES[case][0]]
    env.script["groq"] = [SEARCH]
    before = counters()
    result = env.turn()
    assert result.success and result.metrics["intent"] == "NEW_SEARCH" and not result.metrics["fallback_used"]
    assert env.calls == ["gemini", "groq"]
    assert delta(before[1], counters()[1]) == {"fallback_switch": 1}


def test_fallback_after_bad_json_twice(env, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_FALLBACK_PROVIDER", "groq")
    env.script["gemini"] = ["nope", "nope"]
    env.script["groq"] = [SEARCH]
    result = env.turn()
    assert result.metrics["intent"] == "NEW_SEARCH"
    assert env.calls == ["gemini", "gemini", "groq"]


def test_fallback_is_not_used_for_an_invalid_key(env, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_FALLBACK_PROVIDER", "groq")
    env.script["gemini"] = [HttpError(403, "PERMISSION_DENIED")]
    env.turn()
    assert env.count("groq") == 0


def test_fallback_switches_once_and_does_not_loop(env, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_FALLBACK_PROVIDER", "groq")
    env.script["gemini"] = [asyncio.TimeoutError()]
    env.script["groq"] = [asyncio.TimeoutError()]
    before = counters()
    result = env.turn()
    assert env.calls == ["gemini", "groq"]  # no third call, no switching back
    assert result.reply and result.reply != ERROR_REPLY
    assert delta(before[1], counters()[1]) == {"fallback_switch": 1}
    assert delta(before[0], counters()[0]) == {"timeout": 2}  # one per provider


def test_fallback_equal_to_primary_is_ignored(env, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_FALLBACK_PROVIDER", "gemini")
    assert config.fallback_provider() is None


def test_unknown_fallback_provider_stays_off_without_echoing_the_value(env, monkeypatch, caplog):
    monkeypatch.setenv("LLM_APPROACH_FALLBACK_PROVIDER", "secretvalue")
    with caplog.at_level("WARNING"):
        assert config.fallback_provider() is None
    assert "secretvalue" not in caplog.text


def test_fallback_keeps_its_own_timeout_and_provider_call(monkeypatch):
    """The fallback call goes through providers.complete(provider=...), which uses that provider's own
    client, model and timeout (Groq: config.LLM_TIMEOUT_S, Gemini: config.gemini_timeout_s())."""
    seen = {}

    async def fake_groq(messages, max_tokens):
        seen["groq"] = True
        return Completion("{}", 1, 1, None)

    monkeypatch.setattr(providers, "_complete_groq", fake_groq)
    asyncio.run(providers.complete([{"role": "user", "content": "x"}], None, "groq"))
    assert seen == {"groq": True}


def test_answer_call_stays_on_the_fallback_after_a_switch(env, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_FALLBACK_PROVIDER", "groq")
    budget = llm_client.LLMBudget()
    env.script["gemini"] = [asyncio.TimeoutError()]
    env.script["groq"] = [CHAT, "{}"]
    assert asyncio.run(llm_client.decide("hi", {}, budget)) is not None
    assert asyncio.run(llm_client.generate_json([{"role": "user", "content": "x"}], budget)) == "{}"
    assert env.calls == ["gemini", "groq", "groq"]


# ---- circuit breaker ------------------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    llm_resilience.reset()
    monkeypatch.setattr(llm_resilience, "_breakers", {"gemini": llm_resilience.CircuitBreaker(clock=c)})
    return c


def test_breaker_defaults(monkeypatch):
    assert config.breaker_failures() == 5 and config.breaker_cooldown_s() == 60
    monkeypatch.setenv("LLM_APPROACH_BREAKER_FAILURES", "2")
    monkeypatch.setenv("LLM_APPROACH_BREAKER_COOLDOWN_SECONDS", "7")
    assert config.breaker_failures() == 2 and config.breaker_cooldown_s() == 7


def test_breaker_opens_after_n_failures_and_answers_fast(env, clock, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_BREAKER_FAILURES", "3")
    for _ in range(3):
        env.script["gemini"].append(asyncio.TimeoutError())
        env.turn()
    assert env.count("gemini") == 3
    before = counters()
    result = env.turn()
    assert env.count("gemini") == 3  # skipped: no provider call, so no 20 s wait
    assert result.reply == llm_resilience.LLM_UNAVAILABLE_REPLY
    assert delta(before[1], counters()[1]) == {"breaker_skip": 1}
    assert result.metrics["llm_calls"] == 0 and result.metrics["fallback_used"] is False
    state = env.store.peek(env.sid)
    assert len(state.history) == 8 and state.history[-1]["content"] == llm_resilience.LLM_UNAVAILABLE_REPLY


def test_a_success_resets_the_failure_count(env, clock, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_BREAKER_FAILURES", "3")
    for item in (asyncio.TimeoutError(), asyncio.TimeoutError(), SEARCH, asyncio.TimeoutError(), asyncio.TimeoutError()):
        env.script["gemini"].append(item)
        env.turn()
    env.script["gemini"].append(SEARCH)
    env.turn()
    assert env.count("gemini") == 6  # never opened


def test_breaker_closes_after_a_successful_trial(env, clock, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_BREAKER_FAILURES", "2")
    monkeypatch.setenv("LLM_APPROACH_BREAKER_COOLDOWN_SECONDS", "60")
    env.script["gemini"] = [asyncio.TimeoutError(), asyncio.TimeoutError()]
    env.turn(), env.turn()
    assert env.turn().reply == llm_resilience.LLM_UNAVAILABLE_REPLY
    clock.now += 61
    before = counters()
    env.script["gemini"] = [SEARCH]  # the trial succeeds
    assert env.turn().metrics["intent"] == "NEW_SEARCH"
    assert delta(before[1], counters()[1]) == {"breaker_close": 1}
    env.script["gemini"] = [SEARCH]
    assert env.turn().metrics["intent"] == "NEW_SEARCH"  # closed: normal calls again


def test_a_failed_trial_reopens_the_breaker(env, clock, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_BREAKER_FAILURES", "2")
    env.script["gemini"] = [asyncio.TimeoutError(), asyncio.TimeoutError()]
    env.turn(), env.turn()
    clock.now += 61
    env.script["gemini"] = [asyncio.TimeoutError()]  # the trial fails
    env.turn()
    calls = env.count("gemini")
    assert env.turn().reply == llm_resilience.LLM_UNAVAILABLE_REPLY
    assert env.count("gemini") == calls  # open again, nothing sent


def test_only_one_trial_goes_out_after_the_cooldown(clock):
    b = llm_resilience.breaker("gemini")
    for _ in range(5):
        b.record_failure()
    assert not b.allow()
    clock.now += 61
    assert b.allow() is True and b.allow() is False  # the second caller waits for the trial's result


def test_open_breaker_uses_the_fallback_provider_when_configured(env, clock, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_BREAKER_FAILURES", "2")
    env.script["gemini"] = [asyncio.TimeoutError(), asyncio.TimeoutError()]
    env.turn(), env.turn()
    monkeypatch.setenv("LLM_APPROACH_FALLBACK_PROVIDER", "groq")
    env.script["groq"] = [SEARCH]
    gemini_calls = env.count("gemini")
    result = env.turn()
    assert result.metrics["intent"] == "NEW_SEARCH"
    assert env.count("gemini") == gemini_calls and env.count("groq") == 1


def test_breaker_events_are_logged_at_warning(env, clock, monkeypatch, caplog):
    monkeypatch.setenv("LLM_APPROACH_BREAKER_FAILURES", "1")
    env.script["gemini"] = [asyncio.TimeoutError()]
    with caplog.at_level("WARNING", logger="llm_approach"):
        env.turn()
        env.turn()
    text = " ".join(r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    assert "circuit breaker opened" in text and "breaker open" in text


# ---- secret check in production -------------------------------------------------------------------------


def test_skip_secret_check_is_ignored_in_production_mode(monkeypatch, caplog):
    from fastapi.testclient import TestClient

    from llm_approach import server_app

    monkeypatch.setattr(server_app, "ALLOW_SECRET_CHECK_SKIP", False)  # what a real server runs with
    monkeypatch.setenv(server_app.SKIP_SECRET_CHECK_ENV, "1")
    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")
    called = []
    monkeypatch.setattr(server_app, "validate_secrets", lambda: called.append(1))
    with caplog.at_level("ERROR"), TestClient(server_app.create_app()):
        pass
    assert called == [1]
    assert "is ignored in production mode" in caplog.text
