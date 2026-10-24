"""run_eval: resume, budget stop, rate-limit misses, dry run, cached tokens, no secrets. Offline."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from llm_approach import llm_client, providers
from llm_approach.eval import datasets, metrics, run_eval

SECRET = "gsk_TEST_SECRET_VALUE_123"
GOOD = {"intent": "NEW_SEARCH", "slots": {"city": "Pune"}, "confidence": 0.9}


def make_decide(script, tokens=(1000, 100)):
    """Fake llm_client.decide. Each script entry is a decision dict, None, or 'rate_limit'."""
    calls = []

    async def decide(user_message, state_summary, budget):
        from llm_approach.schema import parse_decision

        calls.append(user_message)
        item = script.pop(0) if script else GOOD
        budget.calls += 1
        budget.tokens_in += tokens[0]
        budget.tokens_out += tokens[1]
        budget.latency_ms += 50.0
        if item == "rate_limit":
            budget.error_class = "RateLimitError"
            return None
        return parse_decision(item) if isinstance(item, dict) else None

    decide.calls = calls
    return decide


class Sleeps:
    def __init__(self):
        self.durations = []

    async def __call__(self, seconds):
        self.durations.append(seconds)


def run(names, tmp_path, decide, **kw):
    kw.setdefault("spacing", 30.0)
    kw.setdefault("today", "2026-01-01")
    kw.setdefault("daily_budget", 150_000)  # the budget is per provider now; Gemini's has no default
    sleeps = kw.pop("sleep", None) or Sleeps()
    summary = asyncio.run(run_eval.run_eval(names, results_dir=tmp_path, decide=decide, sleep=sleeps,
                                            out=lambda *_: None, **kw))
    return summary, sleeps


def records(tmp_path, name):
    return run_eval.read_records(tmp_path / f"{name}.jsonl")


# ---- datasets ---------------------------------------------------------------------


def test_dataset_sizes_and_ids():
    sizes = {n: len(datasets.load_dataset(n)) for n in datasets.DATASETS}
    assert sizes == {"ctxY": 33, "matrix55": 54, "fresh": 59, "golden": 20, "golden_slots": 20}
    ids = [it["item"] for n in datasets.DATASETS for it in datasets.load_dataset(n)]
    assert len(ids) == len(set(ids))
    assert datasets.load_dataset("ctxY")[0]["item"] == "twins:m1-ctxY"
    assert datasets.DATASETS == ["ctxY", "matrix55", "fresh", "golden", "golden_slots"]


def test_golden_slots_gold_is_taken_from_expected_args():
    items = {it["text"]: it for it in datasets.load_dataset("golden_slots")}
    assert items["PG in Mumbai under 10000"]["gold_slots"] == {"city": "Mumbai", "bhk": None, "min_price": None,
                                                              "max_price": 10000, "ideal_for": None}
    assert items["2bhk in Bangalore"]["gold_slots"]["bhk"] == 2
    assert items["hi"]["gold_slots"] is None and items["hi"]["gold_intent"] is None


def test_build_state_context():
    empty = datasets.build_state(False)
    assert not empty["rooms_on_screen"] and empty["selected_room_index"] is None
    ctx = datasets.build_state(True)
    assert len(ctx["rooms_on_screen"]) == 10 and ctx["selected_room_index"] is None and ctx["pending_question"] is None


def test_sample_is_fixed_and_in_file_order():
    a = [i["item"] for i in run_eval.select_items("fresh", 5)]
    b = [i["item"] for i in run_eval.select_items("fresh", 5)]
    assert a == b and len(a) == 5
    full = [i["item"] for i in datasets.load_dataset("fresh")]
    assert a == sorted(a, key=full.index)


# ---- resume ------------------------------------------------------------------------


def test_second_start_skips_completed_items(tmp_path):
    decide = make_decide([])
    s1, _ = run(["golden"], tmp_path, decide, sample=3)
    assert s1["completed"] == 3 and len(decide.calls) == 3
    s2, _ = run(["golden"], tmp_path, decide, sample=3)
    assert s2["completed"] == 0 and len(decide.calls) == 3  # nothing new called
    assert len(records(tmp_path, "golden")) == 3


def test_more_runs_only_add_the_missing_run(tmp_path):
    decide = make_decide([])
    run(["golden"], tmp_path, decide, sample=2, runs=1)
    s, _ = run(["golden"], tmp_path, decide, sample=2, runs=2)
    assert s["completed"] == 2
    assert sorted(r["run"] for r in records(tmp_path, "golden")) == [1, 1, 2, 2]


def test_resume_survives_a_half_written_last_line(tmp_path):
    decide = make_decide([])
    run(["golden"], tmp_path, decide, sample=2)
    with (tmp_path / "golden.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"dataset": "golden", "item": "gol')
    assert len(records(tmp_path, "golden")) == 2
    s, _ = run(["golden"], tmp_path, decide, sample=2)
    assert s["completed"] == 0


def test_spacing_between_calls_only(tmp_path):
    _, sleeps = run(["golden"], tmp_path, make_decide([]), sample=3, spacing=30.0)
    assert sleeps.durations == [30.0, 30.0]  # not before the first call


# ---- budget ------------------------------------------------------------------------


def test_budget_stop_and_next_day_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")  # the per-day ledger is Groq's; Gemini counts a rolling 24 h
    decide = make_decide([], tokens=(4000, 100))
    s1, _ = run(["golden"], tmp_path, decide, sample=5, daily_budget=6000)
    assert s1["stopped_by_budget"] and s1["completed"] == 1
    assert run_eval.tokens_used_today(tmp_path, "2026-01-01") == 4100
    # same day again: still over budget, nothing runs
    s_same, _ = run(["golden"], tmp_path, decide, sample=5, daily_budget=6000)
    assert s_same["completed"] == 0 and s_same["stopped_by_budget"]
    # next day: continues with the remaining items
    s2, _ = run(["golden"], tmp_path, decide, sample=5, daily_budget=6000, today="2026-01-02")
    assert s2["completed"] == 1
    assert len(records(tmp_path, "golden")) == 2


def test_resume_command_is_exact(monkeypatch):
    args = run_eval.parse_args(["--dataset", "fresh", "--runs", "3", "--sample", "30", "--spacing", "30",
                                "--daily-token-budget", "200000"])
    assert run_eval.resume_command(args) == (
        "python -m llm_approach.eval.run_eval --dataset fresh --runs 3 --sample 30 --spacing 30 "
        "--daily-token-budget 200000 --resume")
    defaults = run_eval.parse_args([])
    assert defaults.runs == 1 and defaults.spacing is None and defaults.daily_token_budget is None
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")  # Groq's values are unchanged
    assert run_eval.resolve_limits(defaults) == (30.0, 150_000)


# ---- rate limits ---------------------------------------------------------------------


def test_rate_limit_waits_60s_once_then_succeeds(tmp_path):
    decide = make_decide(["rate_limit", GOOD])
    s, sleeps = run(["golden"], tmp_path, decide, sample=1)
    (rec,) = records(tmp_path, "golden")
    assert rec["status"] == "ok" and 60.0 in sleeps.durations and sleeps.durations.count(60.0) == 1
    assert rec["tokens_in"] == 2000  # both attempts are counted
    assert s["rate_limit_misses"] == 0


def test_second_rate_limit_is_a_miss_not_an_error(tmp_path):
    decide = make_decide(["rate_limit", "rate_limit"])
    s, sleeps = run(["golden"], tmp_path, decide, sample=1)
    (rec,) = records(tmp_path, "golden")
    assert rec["status"] == "rate_limit_miss" and rec["intent"] is None and s["rate_limit_misses"] == 1
    assert sleeps.durations.count(60.0) == 1
    assert metrics.intent_accuracy(records(tmp_path, "golden"))["n"] == 0  # not in any denominator
    assert metrics.invalid_json_rate(records(tmp_path, "golden"))["n"] == 0


def test_miss_is_skipped_by_default_and_retried_with_only_missing(tmp_path):
    run(["golden"], tmp_path, make_decide(["rate_limit", "rate_limit"]), sample=1)
    decide = make_decide([])
    s, _ = run(["golden"], tmp_path, decide, sample=1)
    assert s["completed"] == 0
    s, _ = run(["golden"], tmp_path, decide, sample=1, only_missing=True)
    assert s["completed"] == 1
    statuses = [r["status"] for r in records(tmp_path, "golden")]
    assert statuses == ["rate_limit_miss", "ok"]


def test_no_decision_without_rate_limit_is_invalid(tmp_path):
    run(["golden"], tmp_path, make_decide([None]), sample=1)
    (rec,) = records(tmp_path, "golden")
    assert rec["status"] == "invalid"
    assert metrics.invalid_json_rate([rec])["k"] == 1


# ---- record content ------------------------------------------------------------------


def test_record_has_the_required_fields_and_no_text(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    bad = {"intent": "NEW_SEARCH", "slots": {"city": "Pls Urgent", "bhk": 9}, "confidence": 0.8}
    run(["golden"], tmp_path, make_decide([bad]), sample=1)
    (rec,) = records(tmp_path, "golden")
    for key in ("dataset", "item", "run", "status", "intent", "operation", "slots", "validated_slots", "confidence",
                "slots_valid", "latency_ms", "tokens_in", "tokens_out", "cached_tokens", "would_search", "gold_intent"):
        assert key in rec
    assert rec["slots_valid"] is False and set(rec["rejected"]) >= {"city", "bhk"} and rec["would_search"] is False
    raw = (tmp_path / "golden.jsonl").read_text(encoding="utf-8")
    assert SECRET not in raw
    for item in datasets.load_dataset("golden"):
        assert item["text"] not in raw
    assert "reasoning" not in raw


def test_valid_search_would_search(tmp_path):
    run(["golden"], tmp_path, make_decide([GOOD]), sample=1)
    (rec,) = records(tmp_path, "golden")
    assert rec["would_search"] is True and rec["validated_slots"]["city"] == "Pune" and rec["slots_valid"] is True


# ---- dry run -------------------------------------------------------------------------


def test_dry_run_estimates_without_calls(tmp_path, monkeypatch, capsys):
    async def boom(*a, **k):
        raise AssertionError("dry run must not call the model")

    monkeypatch.setattr(llm_client, "decide", boom)
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    assert run_eval.main(["--dry-run", "--dataset", "all"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "total" in out and "golden_slots" in out
    todo = run_eval.plan_items(datasets.DATASETS, 1, None, tmp_path)
    est = run_eval.estimate(todo, 30.0, 150_000)
    assert est["calls"] == 186 and est["per_dataset"]["ctxY"]["calls"] == 33
    assert 3000 < est["tokens"] / est["calls"] < 6000  # about 4,000 per call
    assert est["minutes"] == pytest.approx(186 * 30 / 60)


def test_dry_run_counts_only_whats_left(tmp_path):
    run(["golden"], tmp_path, make_decide([]), sample=4)
    todo = run_eval.plan_items(["golden"], 1, None, tmp_path)
    assert len(todo) == 16


def test_unknown_dataset_is_rejected(capsys):
    assert run_eval.main(["--dataset", "nope", "--dry-run"]) == 2


# ---- cached tokens in llm_client -------------------------------------------------------


class _Usage:
    def __init__(self, details):
        self.prompt_tokens, self.completion_tokens = 100, 20
        if details is not False:
            self.prompt_tokens_details = details


def _client(usage):
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(GOOD)))], usage=usage)

    async def create(**kwargs):
        return reply

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.mark.parametrize("details, expected", [
    (SimpleNamespace(cached_tokens=64), 64),
    (SimpleNamespace(cached_tokens=0), 0),
    (SimpleNamespace(), None),
    (None, None),
    (False, None),
    (SimpleNamespace(cached_tokens="many"), None),
])
def test_cached_tokens_from_usage(monkeypatch, details, expected):
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")  # the usage shape here is Groq's
    monkeypatch.setattr(providers, "_get_groq_client", lambda: _client(_Usage(details)))
    budget = llm_client.LLMBudget()
    decision = asyncio.run(llm_client.decide("hi", {}, budget))
    assert decision is not None and budget.calls == 1
    assert budget.cached_tokens == expected
    assert (budget.tokens_in, budget.tokens_out) == (100, 20)


def test_cached_tokens_default_is_none():
    assert llm_client.LLMBudget().cached_tokens is None


# ---- --fresh ---------------------------------------------------------------------------


@pytest.fixture
def live(tmp_path, monkeypatch):
    """main() wired to a temp results dir and a fake decide: no network, no .env, no key."""
    import dotenv

    from llm_approach import config

    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(config, "setup_problem", lambda: None)
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    decide = make_decide([], tokens=(4000, 100))
    monkeypatch.setattr(llm_client, "decide", decide)
    return decide


def fresh_dirs(tmp_path):
    return sorted(d for d in tmp_path.glob(run_eval.FRESH_PREFIX + "*") if d.is_dir())


def cli(*argv):
    return run_eval.main(["--dataset", "golden", "--sample", "3", "--spacing", "0", "--daily-token-budget", "150000", *argv])


def test_fresh_runs_everything_again_and_leaves_old_results_untouched(tmp_path, live):
    assert cli() == 0 and len(live.calls) == 3
    old = (tmp_path / "golden.jsonl").read_bytes()
    assert cli() == 0 and len(live.calls) == 3  # without --fresh: nothing to do
    assert cli("--fresh") == 0 and len(live.calls) == 6  # all three again
    (d,) = fresh_dirs(tmp_path)
    assert len(run_eval.read_records(d / "golden.jsonl")) == 3
    assert (tmp_path / "golden.jsonl").read_bytes() == old
    assert not (d / "golden.jsonl").samefile(tmp_path / "golden.jsonl")
    # a second fresh run is another directory, not a merge into the first
    assert cli("--fresh") == 0 and len(live.calls) == 9 and len(fresh_dirs(tmp_path)) == 2


def test_fresh_resume_continues_the_interrupted_fresh_run(tmp_path, live, capsys):
    cli()
    old = (tmp_path / "golden.jsonl").read_bytes()
    assert cli("--fresh", "--daily-token-budget", "18000") == 3  # 12,300 used by the first run: room for one more call
    (d,) = fresh_dirs(tmp_path)
    assert len(run_eval.read_records(d / "golden.jsonl")) == 1
    assert "--fresh --resume" in capsys.readouterr().out
    assert cli("--fresh", "--resume", "--daily-token-budget", "150000") == 0
    assert fresh_dirs(tmp_path) == [d]  # same run, no new directory
    assert len(run_eval.read_records(d / "golden.jsonl")) == 3
    assert (tmp_path / "golden.jsonl").read_bytes() == old


def test_fresh_resume_without_a_fresh_run_is_refused(tmp_path, live):
    cli()
    assert cli("--fresh", "--resume") == 2 and not fresh_dirs(tmp_path)


def test_plain_resume_ignores_fresh_runs(tmp_path, live):
    assert cli("--fresh", "--daily-token-budget", "5000") == 3
    # the unfinished run is the fresh one; plain --resume looks at the normal results only
    assert cli("--resume") == 2


def test_fresh_dry_run_shows_the_full_call_count(tmp_path, live, capsys):
    run_eval.main(["--dataset", "golden", "--sample", "5", "--spacing", "0", "--daily-token-budget", "150000"])
    capsys.readouterr()
    assert run_eval.main(["--dry-run", "--dataset", "golden"]) == 0
    assert "calls   15" in capsys.readouterr().out  # 5 finished
    assert run_eval.main(["--dry-run", "--dataset", "golden", "--fresh"]) == 0
    assert "calls   20" in capsys.readouterr().out  # a fresh run does all 20
    assert not fresh_dirs(tmp_path)  # a dry run creates nothing


def test_fresh_run_shares_the_token_ledger_and_copies_the_baseline(tmp_path, live):
    (tmp_path / "baseline_current.json").write_text('{"x": 1}', encoding="utf-8")
    assert cli("--fresh") == 0
    (d,) = fresh_dirs(tmp_path)
    assert (d / "baseline_current.json").read_text(encoding="utf-8") == '{"x": 1}'
    assert run_eval.tokens_used_today(tmp_path, __import__("datetime").date.today().isoformat()) == 3 * 4100
    assert not (d / run_eval.USAGE_FILE).exists()


def test_report_reads_a_fresh_run_on_its_own(tmp_path, live):
    from llm_approach.eval import report

    cli()
    cli("--fresh")
    (d,) = fresh_dirs(tmp_path)
    for rd in (tmp_path, d):
        _, data = report.build_report(rd)
        assert data["coverage"]["golden"]["done"] == 3  # not 6: runs are never merged
