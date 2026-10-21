"""Gemini free-tier limits: spacing, daily request counter (Pacific day), cap and fallback, sliding window,
429 handling, run planner, resume, STOPPED EARLY, failure streak, report refusal, plan and llm_info output.
Fake clock, fake client, fake decide only: no network, no key, no SDK."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import uuid

import pytest

from llm_approach import config, gemini_limits, llm_client, llm_info, pipeline, providers, reuse
from llm_approach.eval import plan, report, run_eval, run_scenarios, runs
from llm_approach.llm_client import LLMBudget, decide
from llm_approach.state import SessionStore

from .test_eval_run import GOOD as GOOD_DECISION, Sleeps, make_decide, records
from .test_gemini_provider import GOOD, ClientError, run

KEY = "AIza" + "FAKEKEY-not-a-real-key-1234567"  # fake; split so secret scanners stay quiet


def utc(*parts) -> dt.datetime:
    return dt.datetime(*parts, tzinfo=dt.timezone.utc)


# ---- spacing and budget settings ------------------------------------------------------------------


def test_spacing_default_is_5_seconds_and_env_overrides(monkeypatch):
    assert config.spacing_s("gemini") == 5.0
    monkeypatch.setenv("LLM_APPROACH_GEMINI_SPACING_S", "7.5")
    assert config.spacing_s("gemini") == 7.5
    assert config.spacing_s("groq") == 30.0  # Groq unchanged


def test_spacing_below_4_seconds_is_raised_with_one_warning(monkeypatch, caplog):
    monkeypatch.setattr(config, "_spacing_warned", False)
    monkeypatch.setenv("LLM_APPROACH_GEMINI_SPACING_S", "1.5")
    with caplog.at_level(logging.WARNING, logger="llm_approach.config"):
        assert config.spacing_s("gemini") == 4.0
        assert config.spacing_s("gemini") == 4.0
    warnings = [r for r in caplog.records if "SPACING" in r.getMessage()]
    assert len(warnings) == 1 and "1.5" not in warnings[0].getMessage()
    monkeypatch.setenv("LLM_APPROACH_GEMINI_SPACING_S", "4")
    assert config.spacing_s("gemini") == 4.0


def test_request_budget_default_450_and_env(monkeypatch):
    assert config.daily_request_budget() == 450
    monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET", "100")
    assert config.daily_request_budget() == 100
    monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET", "oops")
    assert config.daily_request_budget() == 450


# ---- daily counter: increments, Pacific day --------------------------------------------------------


def test_pacific_midnight_boundary_summer_and_winter():
    # PDT (UTC-7): midnight Pacific is 07:00 UTC
    assert gemini_limits.pacific_date(utc(2026, 10, 20, 6, 59)) == "2026-10-19"
    assert gemini_limits.pacific_date(utc(2026, 10, 20, 7, 0)) == "2026-10-20"
    # PST (UTC-8): midnight Pacific is 08:00 UTC
    assert gemini_limits.pacific_date(utc(2026, 12, 20, 7, 59)) == "2026-12-19"
    assert gemini_limits.pacific_date(utc(2026, 12, 20, 8, 0)) == "2026-12-20"
    # DST switches in 2026: starts Sunday March 8 (10:00 UTC), ends Sunday November 1 (09:00 UTC)
    assert gemini_limits.pacific_date(utc(2026, 3, 8, 7, 59)) == "2026-03-07"  # PST
    assert gemini_limits.pacific_date(utc(2026, 3, 8, 8, 0)) == "2026-03-08"
    assert gemini_limits.pacific_date(utc(2026, 3, 9, 6, 59)) == "2026-03-08"  # PDT from here on
    assert gemini_limits.pacific_date(utc(2026, 3, 9, 7, 0)) == "2026-03-09"
    assert gemini_limits.pacific_date(utc(2026, 11, 1, 6, 59)) == "2026-10-31"  # still PDT
    assert gemini_limits.pacific_date(utc(2026, 11, 1, 7, 0)) == "2026-11-01"
    assert gemini_limits.pacific_date(utc(2026, 11, 2, 7, 59)) == "2026-11-01"  # PST afterwards
    assert gemini_limits.pacific_date(utc(2026, 11, 2, 8, 0)) == "2026-11-02"


def test_counter_resets_at_pacific_midnight_and_stores_only_date_and_count(tmp_path):
    now = [utc(2026, 10, 20, 6, 0)]
    counter = gemini_limits.DailyCounter(tmp_path / "c.json", wall=lambda: now[0])
    assert counter.add() == 1 and counter.add() == 2 and counter.used_today() == 2
    assert json.loads((tmp_path / "c.json").read_text()) == {"count": 2, "date": "2026-10-19"}
    now[0] = utc(2026, 10, 20, 7, 0)  # Pacific midnight passed
    assert counter.used_today() == 0
    assert counter.add() == 1
    assert json.loads((tmp_path / "c.json").read_text()) == {"count": 1, "date": "2026-10-20"}


def test_counter_survives_a_damaged_file(tmp_path):
    (tmp_path / "c.json").write_text("not json")
    counter = gemini_limits.DailyCounter(tmp_path / "c.json")
    assert counter.used_today() == 0 and counter.add() == 1


def test_counter_file_is_git_ignored():
    ignore = (gemini_limits.COUNTER_PATH.parent / ".gitignore").read_text()
    assert gemini_limits.COUNTER_PATH.name in ignore and gemini_limits.COUNTER_PATH.parent.name == "results"


# ---- the cap: counting, blocking, fallback, sharing --------------------------------------------------


def used(limiter) -> int:
    return limiter.counter.used_today()


def test_every_request_counts_success_failure_and_retry(fake_gemini, _gemini_limiter):
    err = ClientError(500, "INTERNAL", "boom")
    fake_gemini([GOOD, err, ClientError(429, "RESOURCE_EXHAUSTED", "q"), GOOD])
    assert run(decide("a", {}, LLMBudget())) is not None
    assert used(_gemini_limiter) == 1
    assert run(decide("b", {}, LLMBudget())) is None  # failed call still counts
    assert used(_gemini_limiter) == 2
    assert run(decide("c", {}, LLMBudget())) is not None  # 429 then the retry: two requests
    assert used(_gemini_limiter) == 4


def test_groq_never_touches_the_counter(fake_groq, _gemini_limiter):
    fake_groq(['{"intent": "NEW_SEARCH", "slots": {"city": "Pune"}, "confidence": 0.9}'])
    assert run(decide("a", {}, LLMBudget())) is not None
    assert used(_gemini_limiter) == 0


def test_cap_blocks_the_call_and_reports_a_rate_limit(fake_gemini, monkeypatch, _gemini_limiter):
    monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET", "2")
    client = fake_gemini([GOOD, GOOD, GOOD])
    assert run(decide("a", {}, LLMBudget())) is not None and run(decide("b", {}, LLMBudget())) is not None
    budget = LLMBudget()
    assert run(decide("c", {}, budget)) is None
    assert budget.error_class == "RateLimitError" and budget.error_kind == "request_cap"
    assert len(client.calls) == 2 and used(_gemini_limiter) == 2  # nothing was sent, nothing counted


def test_cap_uses_the_pipelines_normal_rate_limit_fallback(fake_gemini, monkeypatch, _gemini_limiter):
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    monkeypatch.setenv(config.OFFLINE_ROOMS_ENV, "1")
    monkeypatch.delenv("LLM_APPROACH_METRICS", raising=False)
    pipeline.configure(full_guardrails=True)
    monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET", "1")
    client = fake_gemini([GOOD, GOOD])
    first = run(pipeline.handle_turn(str(uuid.uuid4()), "rooms in pune", store=SessionStore()))
    assert first.reply != pipeline.RATE_LIMIT_REPLY
    second = run(pipeline.handle_turn(str(uuid.uuid4()), "rooms in pune", store=SessionStore()))
    assert second.reply == pipeline.RATE_LIMIT_REPLY and second.metrics["action"] == "rate_limited"
    assert second.metrics["fallback_used"] is False and len(client.calls) == 1


def test_counter_is_shared_by_the_pipeline_and_the_eval_runner(fake_gemini, monkeypatch, _gemini_limiter):
    fake_gemini([GOOD])
    run(decide("a", {}, LLMBudget()))  # a "live chat" request
    assert runs.request_status() == (1, 450, 449)  # the runner's planner sees it
    monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET", "1")
    calls = make_decide([])
    summary = asyncio.run(run_eval.run_eval(["golden"], sample=2, spacing=0, daily_budget=10**9, results_dir=_gemini_limiter.counter.path.parent,
                                            decide=calls, sleep=Sleeps(), out=lambda *_: None))
    assert calls.calls == [] and summary["completed"] == 0
    assert "daily request cap" in summary["stop_reason"]


# ---- sliding window ------------------------------------------------------------------------------------


def make_limiter(tmp_path):
    t = {"now": 1000.0}
    sleeps = []

    async def sleep(s):
        sleeps.append(s)
        t["now"] += s

    lim = gemini_limits.GeminiLimiter(gemini_limits.DailyCounter(tmp_path / "w.json"), clock=lambda: t["now"], sleep=sleep)
    return lim, t, sleeps


async def ok(result="r"):
    return result


def go(lim, tokens=None, estimate=100, budget=10_000):
    return asyncio.run(lim.call(ok, budget=budget, prompt_estimate=estimate, tokens_of=lambda r: tokens))


def test_window_allows_12_requests_then_waits_for_the_oldest_to_leave(tmp_path):
    lim, t, sleeps = make_limiter(tmp_path)
    for _ in range(12):
        go(lim)
        t["now"] += 1.0  # 1 s apart
    assert sleeps == []
    go(lim)  # the 13th would exceed 12 in 60 s: wait until the first request is 60 s old
    assert sleeps == [pytest.approx(60.0 - 12.0)]
    assert lim.requests_in_window() <= 12


def test_window_waits_when_the_token_margin_would_be_exceeded(tmp_path):
    lim, t, sleeps = make_limiter(tmp_path)
    go(lim, tokens=90_000)
    t["now"] += 10
    go(lim, tokens=90_000)  # estimate for this call = the previous count (90K): 180K fits under 200K
    assert sleeps == [] and lim.tokens_in_window() == 180_000
    t["now"] += 10
    go(lim, tokens=90_000)  # 180K + 90K > 200K: wait until the first call (t=1000) is 60 s old
    assert sleeps == [pytest.approx(60.0 - 20.0)]
    assert lim.tokens_in_window() <= 200_000


def test_first_estimate_is_the_prompt_size_then_the_previous_call(tmp_path):
    lim, t, sleeps = make_limiter(tmp_path)
    go(lim, tokens=None, estimate=190_000)  # unknown real count: the estimate stays in the window
    go(lim, tokens=None, estimate=20_000)  # 190K + 20K > 200K -> waits
    assert len(sleeps) == 1 and sleeps[0] == pytest.approx(60.0)


def test_window_forgets_requests_after_60_seconds(tmp_path):
    lim, t, sleeps = make_limiter(tmp_path)
    for _ in range(12):
        go(lim)
    t["now"] += 61
    go(lim)
    assert sleeps == [] and lim.requests_in_window() == 1


# ---- 429 handling ---------------------------------------------------------------------------------------


def flaky(*outcomes):
    queue = list(outcomes)
    calls = []

    async def send():
        calls.append(1)
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    send.calls = calls
    return send


def call(lim, send):
    return asyncio.run(lim.call(send, budget=450, prompt_estimate=100))


def test_429_with_retry_delay_waits_that_long_and_retries_once(tmp_path):
    lim, t, sleeps = make_limiter(tmp_path)
    send = flaky(ClientError(429, "RESOURCE_EXHAUSTED", "Quota exceeded. Please retry in 23.4s."), "fine")
    assert call(lim, send) == "fine"
    assert sleeps == [pytest.approx(23.4)] and len(send.calls) == 2 and lim.counter.used_today() == 2


def test_429_retry_delay_is_read_from_retryDelay_and_capped_at_60(tmp_path):
    lim, t, sleeps = make_limiter(tmp_path)
    call(lim, flaky(ClientError(429, "RESOURCE_EXHAUSTED", "{'details': [{'retryDelay': '17s'}]}"), "x"))
    call(lim, flaky(ClientError(429, "RESOURCE_EXHAUSTED", "retry in 300s"), "x"))
    call(lim, flaky(ClientError(429, "RESOURCE_EXHAUSTED", "no delay named"), "x"))
    assert [round(s) for s in sleeps if s > 10] == [17, 60, 60]


def test_a_second_429_is_not_retried_again(tmp_path):
    lim, t, sleeps = make_limiter(tmp_path)
    send = flaky(ClientError(429, "RESOURCE_EXHAUSTED", "q"), ClientError(429, "RESOURCE_EXHAUSTED", "q"), "never")
    with pytest.raises(gemini_limits.RateLimitRetried):
        call(lim, send)
    assert len(send.calls) == 2


def test_daily_quota_429_is_never_retried(tmp_path):
    lim, t, sleeps = make_limiter(tmp_path)
    err = ClientError(429, "RESOURCE_EXHAUSTED",
                      "Quota exceeded for metric GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 500")
    send = flaky(err, "never")
    with pytest.raises(gemini_limits.DailyQuotaExhausted):
        call(lim, send)
    assert len(send.calls) == 1 and sleeps == [] and lim.quota_exhausted


def test_other_errors_are_not_retried_by_the_limiter(tmp_path):
    lim, t, sleeps = make_limiter(tmp_path)
    send = flaky(ClientError(500, "INTERNAL", "x"), "never")
    with pytest.raises(ClientError):
        call(lim, send)
    assert len(send.calls) == 1


def test_daily_quota_stops_the_eval_run_cleanly_with_the_resume_command(fake_gemini, monkeypatch, capsys):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET", "5000000")
    quota = ClientError(429, "RESOURCE_EXHAUSTED", "limit GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    client = fake_gemini([GOOD, GOOD, quota, GOOD])
    code = run_eval.main(["--dataset", "golden", "--sample", "5", "--spacing", "0"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert code == 3 and len(client.calls) == 3  # the quota call was not retried
    assert len(records(run_eval.RESULTS_DIR, "golden")) == 2  # the quota item is "not run": nothing recorded
    last = lines[-1]
    assert last.startswith("STOPPED EARLY: 2 of 5 done. Reason: Gemini says the daily request quota is exhausted")
    assert last.endswith("Resume with: python -m llm_approach.eval.run_eval --dataset golden --runs 1 --sample 5 "
                         "--spacing 0 --daily-token-budget 5000000 --resume")
    assert KEY not in "\n".join(lines)


# ---- planner -----------------------------------------------------------------------------------------------


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET", "5000000")

    def install(replies):
        decide_fn = make_decide(replies)
        monkeypatch.setattr(llm_client, "decide", decide_fn)
        return decide_fn

    return install


def test_planner_allows_a_fitting_run_and_prints_the_numbers(live, capsys, _gemini_limiter):
    fn = live([])
    _gemini_limiter.counter.add(10)
    assert run_eval.main(["--dataset", "golden", "--sample", "4", "--spacing", "0"]) == 0
    out = capsys.readouterr().out
    assert "Plan: 4 prompts x 3 worst-case calls = 12 requests" in out
    assert "Requests used today (Pacific day): 10 of 450; remaining: 440." in out
    assert len(fn.calls) == 4
    allowed, lines = runs.preflight(4, 3, 5.0)
    assert allowed and "Estimated time at 5 s spacing: about 0.3 min typical, 1.0 min worst case." in lines[2]


def test_planner_refuses_an_oversized_run_without_calling(live, capsys, _gemini_limiter):
    fn = live([])
    _gemini_limiter.counter.add(440)
    assert run_eval.main(["--dataset", "golden", "--spacing", "5"]) == 2  # 20 x 3 = 60 > 10 left
    out = capsys.readouterr().out
    assert "REFUSED" in out and "60 requests but only 10 remain" in out and fn.calls == []
    with pytest.raises(SystemExit):  # there is no override flag
        run_eval.parse_args(["--force"])


def test_planner_refuses_oversized_scenarios_too(monkeypatch, capsys, _gemini_limiter):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET", "5000000")
    monkeypatch.setattr(run_scenarios.pipeline, "preload", lambda: pytest.fail("must refuse before loading"))
    assert run_scenarios.main(["--live", "--spacing", "5"]) == 2  # 170 turns x 5 > 450
    assert "REFUSED" in capsys.readouterr().out


# ---- checkpoint, resume, STOPPED EARLY, failures --------------------------------------------------------------


def fail_decide(script):
    """Fake decide: each entry is 'ok' or 'timeout' (a transport failure)."""
    from llm_approach.schema import parse_decision

    calls = []

    async def fake(user_message, state_summary, budget):
        calls.append(user_message)
        item = script.pop(0) if script else "ok"
        budget.calls += 1
        budget.tokens_in += 1000
        budget.tokens_out += 100
        if item == "timeout":
            budget.error_class = "TimeoutError"
            return None
        return parse_decision(GOOD_DECISION)

    fake.calls = calls
    return fake


def run_fn(tmp_path, fake, **kw):
    kw.setdefault("spacing", 0)
    kw.setdefault("daily_budget", 10**9)
    out = []
    summary = asyncio.run(run_eval.run_eval(["golden"], results_dir=tmp_path, decide=fake, sleep=Sleeps(),
                                            out=out.append, **kw))
    return summary, out


def test_five_failures_in_a_row_stop_the_run(tmp_path):
    fake = fail_decide(["ok"] + ["timeout"] * 5 + ["ok"] * 10)
    summary, out = run_fn(tmp_path, fake, sample=10)
    assert len(fake.calls) == 6 and summary["completed"] == 6
    assert "5 failures in a row" in summary["stop_reason"] and "TimeoutError" in summary["stop_reason"]
    assert "prompts not run" in out[-1]


def test_a_success_resets_the_failure_streak(tmp_path):
    fake = fail_decide(["ok"] + ["timeout"] * 4 + ["ok"] + ["timeout"] * 4 + ["ok"] * 10)
    summary, _ = run_fn(tmp_path, fake, sample=10)
    assert summary["stop_reason"] is None and summary["completed"] == 10


def test_a_failed_first_call_is_a_clear_connectivity_stop(tmp_path):
    fake = fail_decide(["timeout"])
    summary, _ = run_fn(tmp_path, fake, sample=10)
    assert len(fake.calls) == 1
    assert "first request failed" in summary["stop_reason"] and "key, the model name and the quota" in summary["stop_reason"]


def test_every_prompt_is_checkpointed_with_provider_and_model_and_no_key(tmp_path):
    fake = fail_decide(["ok", "ok", "timeout"] + ["ok"] * 5)
    run_fn(tmp_path, fake, sample=2)
    recs = records(tmp_path, "golden")
    assert [r["status"] for r in recs] == ["ok", "ok"]
    assert all(r["provider"] == "gemini" and r["model"] == "gemini-3.5-flash-lite" for r in recs)
    assert KEY not in (tmp_path / "golden.jsonl").read_text()


def test_stopped_early_line_resume_and_mismatch_refusals(live, capsys, monkeypatch):
    seq = ["ok"] * 2 + ["timeout"] * 5 + ["ok"] * 50
    fake = fail_decide(seq)
    monkeypatch.setattr(llm_client, "decide", fake)
    args = ["--dataset", "golden", "--runs", "1", "--sample", "10", "--spacing", "0"]
    # 1. starts, stops after 5 failures in a row; the last printed line is the STOPPED EARLY line
    assert run_eval.main(args) == 3
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1].startswith("STOPPED EARLY: 7 of 10 done. Reason: 5 failures in a row")
    assert lines[-1].endswith("Resume with: python -m llm_approach.eval.run_eval --dataset golden --runs 1 --sample 10 "
                              "--spacing 0 --daily-token-budget 5000000 --resume")
    # 2. the same command without --resume is refused (an unfinished run exists)
    assert run_eval.main(args) == 2
    assert "Continue it with --resume" in capsys.readouterr().out
    # 3. mismatching dataset / sample / model / provider are refused
    assert run_eval.main(["--dataset", "fresh", "--sample", "10", "--spacing", "0", "--resume"]) == 2
    assert "differs in dataset" in capsys.readouterr().out
    assert run_eval.main(["--dataset", "golden", "--sample", "9", "--spacing", "0", "--resume"]) == 2
    assert "differs in sample" in capsys.readouterr().out
    monkeypatch.setenv("LLM_APPROACH_GEMINI_MODEL", "gemini-other")
    assert run_eval.main(args + ["--resume"]) == 2
    assert "differs in model" in capsys.readouterr().out
    monkeypatch.delenv("LLM_APPROACH_GEMINI_MODEL")
    # 4. the exact command resumes: finished prompts are skipped, the five failed ones are retried
    calls_before = len(fake.calls)
    assert run_eval.main(args + ["--resume"]) == 0
    out = capsys.readouterr().out
    assert len(fake.calls) - calls_before == 8  # 3 never run + 5 failed; the 2 finished are not repeated
    assert "STOPPED EARLY" not in out
    ok = {r["item"] for r in records(run_eval.RESULTS_DIR, "golden") if r["status"] == "ok"}
    assert len(ok) == 10
    # 5. once finished there is nothing to resume
    assert run_eval.main(args + ["--resume"]) == 2
    assert "Nothing to resume" in capsys.readouterr().out


def test_resume_without_any_earlier_run_is_refused(live, capsys):
    live([])
    assert run_eval.main(["--dataset", "golden", "--resume"]) == 2
    assert "Nothing to resume" in capsys.readouterr().out


def test_local_request_cap_stops_a_run_with_the_stopped_early_line(live, capsys, monkeypatch, _gemini_limiter):
    live([])
    monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET", "20")
    _gemini_limiter.counter.add(20)  # nothing left... but the planner refuses first
    assert run_eval.main(["--dataset", "golden", "--sample", "2", "--spacing", "0"]) == 2

    summary = asyncio.run(run_eval.run_eval(["golden"], sample=3, spacing=0, daily_budget=10**9,
                                            results_dir=run_eval.RESULTS_DIR, decide=make_decide([]),
                                            sleep=Sleeps(), out=lambda *_: None))
    assert summary["stop_reason"].startswith("the local daily request cap (20)") and summary["completed"] == 0


# ---- report refuses an incomplete run ------------------------------------------------------------------------


def test_report_refuses_an_incomplete_run_and_names_the_missing_prompts(tmp_path, capsys):
    fake = fail_decide([])
    run_fn(tmp_path, fake, sample=3)  # 3 of 5 sampled prompts finished...
    sig = runs.signature("run_eval", "golden", 1, 5)
    runs.save_entry(tmp_path, sig, "stopped", 3, 5, "test")
    with pytest.raises(runs.IncompleteRunError) as err:
        report.build_report(tmp_path)
    text = str(err.value)
    assert "incomplete run" in text and "golden" in text and "prompts are missing" in text and "golden:g" in text
    assert report.main(["--results-dir", str(tmp_path)]) == 2
    assert "incomplete run" in capsys.readouterr().out
    assert not (tmp_path / "comparison.md").exists()


def test_report_works_once_the_run_is_finished_or_when_no_run_is_recorded(tmp_path):
    fake = fail_decide([])
    run_fn(tmp_path, fake, sample=5)
    md, _ = report.build_report(tmp_path)  # no run state at all: partial results still report, as before
    assert "golden" in md
    runs.save_entry(tmp_path, runs.signature("run_eval", "golden", 1, 5), "stopped", 5, 5, "crashed at the very end")
    report.build_report(tmp_path)  # nothing is missing, so nothing is refused


def test_report_refuses_unfinished_scenarios(tmp_path):
    runs.save_entry(tmp_path, runs.signature("run_scenarios", "scenarios:S1,S2"), "stopped", 0, 2, "test")
    with pytest.raises(runs.IncompleteRunError, match="S1, S2"):
        report.build_report(tmp_path)


# ---- scenarios: stop rules -----------------------------------------------------------------------------------------


def report_with(tokens, calls=1):
    return run_scenarios.TurnReport("S1", 1, "u", tokens=tokens, llm_calls=calls)


def test_scenario_guard_stops_on_quota_cap_and_five_failed_turns(_gemini_limiter):
    guard = run_scenarios.LiveGuard(_gemini_limiter)
    assert guard.before_turn() is None and guard.hard_stop(report_with(100)) is None
    for _ in range(4):
        assert guard.soft_stop(report_with(0)) is None
    assert "5 failed LLM turns in a row" in guard.soft_stop(report_with(0))
    guard = run_scenarios.LiveGuard(_gemini_limiter)
    guard.soft_stop(report_with(0))
    guard.soft_stop(report_with(120))  # a good turn resets
    assert guard.streak.count == 0
    guard.soft_stop(report_with(0, calls=0))  # a turn without an LLM call is neither
    _gemini_limiter.quota_exhausted = True
    assert "daily request quota" in guard.hard_stop(report_with(0))
    _gemini_limiter.counter.add(450)
    assert "daily request cap (450)" in guard.before_turn()


# ---- plan command and llm_info -----------------------------------------------------------------------------------------


def test_plan_rows_for_the_known_dataset_sizes():
    rows = {r["name"]: r for r in plan.rows(5.0)}
    sizes = {n: (r["prompts"], r["worst"]) for n, r in rows.items()}
    assert sizes["ctxY"] == (33, 99) and sizes["golden"] == (20, 60) and sizes["golden_slots"] == (20, 60)
    assert sizes["matrix55"] == (54, 162) and sizes["fresh"] == (59, 177)
    assert rows["ctxY"]["minutes"] == pytest.approx(99 * 5 / 60)
    assert rows["scenarios"]["worst"] == rows["scenarios"]["prompts"] * runs.WORST_CASE_CALLS_PER_SCENARIO_TURN


def test_plan_output_suggests_days_under_the_budget():
    text = "\n".join(plan.build_plan(5.0, 0, 450))
    assert "ctxY" in text and "99" in text and "177" in text
    assert "today: ctxY + golden + golden_slots + matrix55" in text and "next day: fresh + scenarios (part 1)" in text
    days = plan.schedule(plan.rows(5.0), 450, 450)
    assert all(sum(n for _, n in d) <= 450 for d in days)


def test_plan_says_when_everything_fits_in_one_day():
    text = "\n".join(plan.build_plan(5.0, 0, 5000))
    assert "Everything fits in one day" in text
    assert "Suggested order: ctxY + golden + golden_slots + matrix55 + fresh + scenarios" in text


def test_plan_counts_what_is_left_today():
    text = "\n".join(plan.build_plan(5.0, 440, 450))
    assert "remaining today: 10" in text and "today: nothing fits in what is left" in text


def test_plan_main_prints_without_a_key(capsys, monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    assert plan.main() == 0
    assert "Suggested order" in capsys.readouterr().out


def test_llm_info_prints_the_limits_and_never_the_key(monkeypatch, capsys, _gemini_limiter):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET", "2000000")
    _gemini_limiter.counter.add(7)
    assert llm_info.main() == 0
    out = capsys.readouterr().out
    assert "Gemini requests used today (Pacific day): 7 of 450" in out
    assert "Gemini spacing: 5 s" in out and "12 requests and 200000 tokens per 60 s" in out
    assert "Gemini daily token budget (our own cap): 2000000" in out
    assert KEY not in out
    monkeypatch.delenv("LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET")
    assert "not set" in "\n".join(llm_info.limits_lines())
