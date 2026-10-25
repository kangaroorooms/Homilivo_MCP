"""Audit items A4 and A5: our own Gemini limiter errors do not open the circuit breaker; the counter file is written
atomically, can be moved by an environment variable and is checked at startup.

Fakes only: no Gemini call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading

import pytest

from llm_approach import config, gemini_limits, llm_resilience, observability, pipeline, server_app

from .test_llm_failures import CHAT, HttpError, counters, delta, env  # noqa: F401  (env is a fixture)

LOCAL_ERRORS = {
    "wait_too_long": (lambda: gemini_limits.WaitTooLong(30.0, 2.0), "rate_limit"),
    "request_cap": (gemini_limits.RequestCapReached, "daily_quota"),
}
GOOGLE_ERRORS = {
    "google_429_after_retry": (gemini_limits.RateLimitRetried, "rate_limit"),
    "google_daily_quota": (gemini_limits.DailyQuotaExhausted, "daily_quota"),
    "raw_429": (lambda: HttpError(429, "RESOURCE_EXHAUSTED", "slow down"), "rate_limit"),
    "http_503": (lambda: HttpError(503, "UNAVAILABLE"), "server_error"),
    "timeout": (asyncio.TimeoutError, "timeout"),
    "connection": (lambda: ConnectionError("no route"), "server_error"),
}


def test_only_the_limiters_own_errors_are_local():
    assert gemini_limits.is_local_limit(gemini_limits.WaitTooLong(1.0, 0.0))
    assert gemini_limits.is_local_limit(gemini_limits.RequestCapReached())
    assert not gemini_limits.is_local_limit(gemini_limits.RateLimitRetried())  # Google answered 429 twice
    assert not gemini_limits.is_local_limit(gemini_limits.DailyQuotaExhausted())  # Google said the quota is gone
    assert not gemini_limits.is_local_limit(HttpError(429, "RESOURCE_EXHAUSTED"))
    assert not gemini_limits.is_local_limit(asyncio.TimeoutError())


@pytest.mark.parametrize("case", LOCAL_ERRORS)
def test_local_limiter_errors_never_open_the_breaker_and_the_user_gets_the_fallback_reply(env, monkeypatch, case):
    make, kind = LOCAL_ERRORS[case]
    monkeypatch.setenv("LLM_APPROACH_BREAKER_FAILURES", "2")
    before = counters()
    for _ in range(5):  # far more than the threshold
        env.script["gemini"] = [make()]
        result = env.turn()
        assert result.metrics["fallback_used"] is True and result.reply  # the rule-based fallback answered
    breaker = llm_resilience.breaker("gemini")
    assert breaker.consecutive == 0 and not breaker.is_open()
    assert delta(before[0], counters()[0]) == {kind: 5}  # still counted in /metrics
    assert delta(before[1], counters()[1]) == {}  # no breaker_open event


@pytest.mark.parametrize("case", GOOGLE_ERRORS)
def test_real_google_failures_still_open_the_breaker(env, monkeypatch, case):
    make, kind = GOOGLE_ERRORS[case]
    monkeypatch.setenv("LLM_APPROACH_BREAKER_FAILURES", "2")
    before = counters()
    for _ in range(2):
        env.script["gemini"] = [make()]
        result = env.turn()
        assert result.reply and result.metrics["fallback_used"] is True
    assert llm_resilience.breaker("gemini").is_open()
    assert delta(before[0], counters()[0]) == {kind: 2}
    assert delta(before[1], counters()[1]).get("breaker_open") == 1


def test_a_local_error_between_two_real_failures_neither_counts_nor_resets(env, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_BREAKER_FAILURES", "2")
    for exc in (HttpError(503, "UNAVAILABLE"), gemini_limits.WaitTooLong(30.0, 2.0)):
        env.script["gemini"] = [exc]
        env.turn()
    breaker = llm_resilience.breaker("gemini")
    assert breaker.consecutive == 1 and not breaker.is_open()
    env.script["gemini"] = [HttpError(503, "UNAVAILABLE")]
    env.turn()
    assert breaker.is_open()


def test_the_real_limiter_raises_a_local_error_when_the_daily_cap_is_reached(tmp_path):
    limiter = gemini_limits.GeminiLimiter(counter=gemini_limits.UsageCounter(path=tmp_path / "c.json"))

    async def send():
        return object()

    async def run():
        await limiter.call(send, budget=1, prompt_estimate=1)
        with pytest.raises(gemini_limits.RequestCapReached) as caught:
            await limiter.call(send, budget=1, prompt_estimate=1)
        return caught.value

    assert gemini_limits.is_local_limit(asyncio.run(run()))


# ---- A5: the counter file ---------------------------------------------------------------------------------


def test_default_counter_path_is_kept_and_the_environment_can_move_it(monkeypatch, tmp_path):
    monkeypatch.delenv(config.COUNTER_PATH_ENV, raising=False)
    assert gemini_limits.counter_path() == gemini_limits.COUNTER_PATH
    assert gemini_limits.UsageCounter().path == gemini_limits.COUNTER_PATH
    moved = tmp_path / "volume" / "gemini.json"
    monkeypatch.setenv(config.COUNTER_PATH_ENV, str(moved))
    counter = gemini_limits.UsageCounter()
    assert counter.path == moved
    counter.add(2, tokens=10)
    assert moved.is_file() and counter.used_requests() == 2
    assert gemini_limits.UsageCounter(path=tmp_path / "explicit.json").path == tmp_path / "explicit.json"  # explicit wins


def test_the_counter_is_written_through_a_unique_temp_file_and_replaced_atomically(monkeypatch, tmp_path):
    path = tmp_path / "gemini.json"
    counter = gemini_limits.UsageCounter(path=path)
    counter.add(1)
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: (replaced.append((os.fspath(src), os.fspath(dst))), real_replace(src, dst))[1])
    counter.add(1)
    counter.add(1)
    assert len({src for src, _ in replaced}) == 2 and all(dst == str(path) for _, dst in replaced)  # a new temp name each time
    assert all(src != str(path.with_suffix(".tmp")) for src, _ in replaced)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["gemini.json"]  # no temp file left behind
    assert counter.used_requests() == 3


def test_a_failed_replace_keeps_the_old_file_cleans_up_and_warns_once_without_the_path(monkeypatch, tmp_path, caplog):
    path = tmp_path / "secretdir" / "gemini.json"
    counter = gemini_limits.UsageCounter(path=path)
    counter.add(1)
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr(gemini_limits, "_write_warned", False)

    def boom(src, dst):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "replace", boom)
    with caplog.at_level(logging.WARNING, logger="llm_approach.gemini_limits"):
        counter.add(1)
        counter.add(1)
    assert path.read_text(encoding="utf-8") == before  # the real file is never half-written
    assert [p.name for p in path.parent.iterdir()] == ["gemini.json"]
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == [gemini_limits.WRITE_FAILED_WARNING]  # once, fixed wording
    assert "secretdir" not in caplog.text


def test_concurrent_writers_in_one_process_leave_a_valid_file(tmp_path):
    path = tmp_path / "gemini.json"
    counter = gemini_limits.UsageCounter(path=path)
    threads = [threading.Thread(target=lambda: [counter.add(1) for _ in range(20)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 2 and counter.used_requests() == 80
    assert [p.name for p in tmp_path.iterdir()] == ["gemini.json"]


def test_writable_is_true_for_a_normal_folder_and_false_for_read_only_ones(monkeypatch, tmp_path):
    assert gemini_limits.UsageCounter(path=tmp_path / "ok" / "c.json").writable()
    assert [p.name for p in (tmp_path / "ok").iterdir()] == []  # the probe file is removed
    blocker = tmp_path / "afile"
    blocker.write_text("x", encoding="utf-8")
    assert not gemini_limits.UsageCounter(path=blocker / "c.json").writable()  # the folder cannot be created
    readonly = tmp_path / "ro"
    readonly.mkdir()

    def denied(self, *args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(type(readonly), "write_text", denied)  # a read-only directory refuses new files
    assert not gemini_limits.UsageCounter(path=readonly / "c.json").writable()


def test_startup_warns_with_fixed_wording_when_the_counter_cannot_be_written(monkeypatch, tmp_path, caplog):
    from fastapi.testclient import TestClient

    from llm_approach.state import SessionStore

    blocker = tmp_path / "hidden-layout"
    blocker.write_text("x", encoding="utf-8")
    gemini_limits.set_limiter(gemini_limits.GeminiLimiter(gemini_limits.UsageCounter(path=blocker / "c.json")))
    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")
    monkeypatch.setattr(pipeline, "_store", SessionStore())
    with caplog.at_level(logging.WARNING, logger="llm_approach.server_app"), TestClient(server_app.create_app()):
        pass
    assert server_app.COUNTER_PATH_WARNING in caplog.text and "hidden-layout" not in caplog.text


def test_startup_is_quiet_when_the_counter_is_writable(monkeypatch, caplog):
    from fastapi.testclient import TestClient

    from llm_approach.state import SessionStore

    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")
    monkeypatch.setattr(pipeline, "_store", SessionStore())
    with caplog.at_level(logging.WARNING, logger="llm_approach.server_app"), TestClient(server_app.create_app()):
        pass
    assert server_app.COUNTER_PATH_WARNING not in caplog.text
