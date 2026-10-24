"""One retry of the primary provider after a 5xx (500, 502, 503, 504). Fakes only: no Gemini, Groq or rooms call."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from llm_approach import config, gemini_limits, llm_client, llm_resilience, observability, providers
from llm_approach.providers import Completion

SEARCH = json.dumps({"intent": "NEW_SEARCH", "confidence": 0.95, "slots": {"city": "Bangalore"}})


class HttpError(Exception):
    def __init__(self, code: int, status: str | None = None) -> None:
        super().__init__("boom")
        self.code, self.status = code, status


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_5XX_RETRY_DELAY_SECONDS", "0")
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "gemini")
    script, calls, sleeps = [], [], []

    async def fake_complete(messages, max_tokens=None, provider=None):
        calls.append(provider or "primary")
        item = script.pop(0) if script else SEARCH
        if isinstance(item, BaseException):
            raise item
        return Completion(text=item, tokens_in=10, tokens_out=5, cached=None)

    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(providers, "complete", fake_complete)
    monkeypatch.setattr(llm_client.asyncio, "sleep", fake_sleep)
    return script, calls, sleeps


def decide(budget=None):
    budget = budget or llm_client.LLMBudget()
    return asyncio.run(llm_client.decide("rooms in bangalore", {}, budget)), budget


def events():
    return dict(observability.get_metrics_snapshot()["llm_events"])


def test_503_then_success_recovers(env, caplog):
    script, calls, sleeps = env
    script[:] = [HttpError(503, "UNAVAILABLE")]
    before = events().get("recovered_5xx_retry", 0)
    with caplog.at_level(logging.INFO, logger="llm_approach.llm_client"):
        decision, budget = decide()
    assert decision is not None and len(calls) == 2 and budget.calls == 2
    assert "recovered after 5xx retry" in caplog.text
    assert events().get("recovered_5xx_retry", 0) == before + 1
    assert budget.failure_kind is None and not llm_resilience.breaker("gemini").consecutive


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_every_listed_5xx_is_retried(env, code):
    script, calls, _ = env
    script[:] = [HttpError(code)]
    assert decide()[0] is not None and len(calls) == 2


def test_503_twice_falls_through_and_counts_one_breaker_failure(env):
    script, calls, _ = env
    script[:] = [HttpError(503, "UNAVAILABLE"), HttpError(503, "UNAVAILABLE")]
    decision, budget = decide()
    assert decision is None and len(calls) == 2  # no loop
    assert budget.failure_kind == "server_error"
    assert llm_resilience.breaker("gemini").consecutive == 1


def test_503_twice_then_fallback_provider_answers(env, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_FALLBACK_PROVIDER", "groq")
    script, calls, _ = env
    script[:] = [HttpError(503), HttpError(503)]
    decision, _ = decide()
    assert decision is not None and calls == ["primary", "primary", "groq"]


@pytest.mark.parametrize("exc", [HttpError(400, "INVALID_ARGUMENT"), HttpError(401), HttpError(403),
                                 HttpError(404), HttpError(429, "RESOURCE_EXHAUSTED"), asyncio.TimeoutError(),
                                 ConnectionError("down")])
def test_other_errors_are_not_retried(env, exc):
    script, calls, _ = env
    script[:] = [exc]
    assert decide()[0] is None and len(calls) == 1


def test_delay_comes_from_the_env_and_defaults_to_one_second(env, monkeypatch):
    script, _, sleeps = env
    monkeypatch.setenv("LLM_APPROACH_5XX_RETRY_DELAY_SECONDS", "0.25")
    script[:] = [HttpError(503)]
    decide()
    assert sleeps == [0.25]
    monkeypatch.delenv("LLM_APPROACH_5XX_RETRY_DELAY_SECONDS")
    assert config.retry_5xx_delay_s() == 1.0
    monkeypatch.setenv("LLM_APPROACH_5XX_RETRY_DELAY_SECONDS", "-1")
    assert config.retry_5xx_delay_s() is None


def test_retry_skipped_when_the_turn_time_budget_would_be_exceeded(env, monkeypatch):
    script, calls, _ = env
    monkeypatch.setenv("LLM_APPROACH_5XX_RETRY_DELAY_SECONDS", "1")
    script[:] = [HttpError(503)]
    budget = llm_client.LLMBudget()
    budget.latency_ms = config.GEMINI_TURN_LLM_BUDGET_S * 1000 - 500  # the 1 s delay would cross the 25 s line
    asyncio.run(llm_client.decide("hi", {}, budget))
    assert len(calls) == 1


def test_retry_skipped_when_the_delay_alone_crosses_the_budget(env, monkeypatch):
    script, calls, _ = env
    monkeypatch.setenv("LLM_APPROACH_5XX_RETRY_DELAY_SECONDS", "30")
    script[:] = [HttpError(503)]
    assert decide()[0] is None and len(calls) == 1


@pytest.mark.parametrize("cap, retry_sent", [(10, True), (1, False)])
def test_retry_counts_against_the_gemini_limiter_and_the_cap(env, monkeypatch, tmp_path, cap, retry_sent):
    """Through the real limiter: the retry is a second counted request, and the daily cap can refuse it."""
    script, calls, _ = env
    sent = []

    async def via_limiter(messages, max_tokens=None, provider=None):
        async def send():
            sent.append(1)
            if len(sent) == 1:
                raise HttpError(503, "UNAVAILABLE")
            return object()

        await limiter.call(send, budget=cap, prompt_estimate=10)
        return Completion(text=SEARCH, tokens_in=10, tokens_out=5, cached=None)

    limiter = gemini_limits.GeminiLimiter(counter=gemini_limits.UsageCounter(path=tmp_path / "counter.json"))
    monkeypatch.setattr(providers, "complete", via_limiter)
    decision, budget = decide()
    assert len(sent) == (2 if retry_sent else 1)
    assert limiter.counter.used_requests() == (2 if retry_sent else 1)
    assert (decision is not None) is retry_sent
    if not retry_sent:
        assert budget.failure_kind == "daily_quota" and llm_resilience.breaker("gemini").consecutive == 1


def test_answerer_call_is_retried_too(env):
    script, calls, _ = env
    script[:] = [HttpError(502), '{"reply": "ok"}']
    budget = llm_client.LLMBudget()
    assert asyncio.run(llm_client.generate_json([{"role": "user", "content": "hi"}], budget)) == '{"reply": "ok"}'
    assert len(calls) == 2 and llm_resilience.breaker("gemini").consecutive == 0
