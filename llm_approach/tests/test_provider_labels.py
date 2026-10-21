"""Provider and model labels on every result, per-provider budgets, the report refusing to mix pairs,
the live runners refusing a Gemini run without a budget, and the small scripts. Offline."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_approach import config, llm_client, llm_info, metrics as metrics_mod, providers, server_app
from llm_approach.eval import report, run_eval, run_scenarios
from llm_approach.tests.test_eval_run import GOOD, make_decide, run
from llm_approach.tests.test_startup_config import server_without_keys  # noqa: F401  (fixture)

GROQ = ("groq", "openai/gpt-oss-20b")
GEMINI = ("gemini", "gemini-3.5-flash-lite")
KEY = "AIza" + "FAKEKEY-label-test-9876543210"
BASELINE = Path(report.RESULTS_DIR) / "baseline_current.json"


@pytest.fixture
def results(tmp_path):
    import shutil

    if BASELINE.exists():
        shutil.copy(BASELINE, tmp_path / "baseline_current.json")
    return tmp_path


def write_old_records(results, n=3):
    """Records exactly as written before the provider switch: no provider or model field."""
    run(["golden"], results, make_decide([]), sample=n)
    path = results / "golden.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r.pop("provider"), r.pop("model")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ---- labels on records ---------------------------------------------------------------------------


def test_eval_records_carry_provider_and_model(results, monkeypatch):
    run(["golden"], results, make_decide([]), sample=2)
    assert {config.record_label(r) for r in run_eval.read_records(results / "golden.jsonl")} == {GEMINI}
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    run(["fresh"], results, make_decide([]), sample=2)
    assert {config.record_label(r) for r in run_eval.read_records(results / "fresh.jsonl")} == {GROQ}


def test_old_records_count_as_groq():
    assert config.record_label({"item": "g1"}) == GROQ
    assert config.record_label({"provider": "gemini", "model": "m"}) == ("gemini", "m")


def test_a_gemini_run_does_not_skip_items_finished_with_groq(results, monkeypatch):
    write_old_records(results, 3)  # old results = groq
    assert len(run_eval.plan_items(["golden"], 1, 3, results)) == 3  # gemini: nothing done yet
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    assert run_eval.plan_items(["golden"], 1, 3, results) == []  # groq: all three finished


def test_the_daily_ledger_is_per_provider(results, monkeypatch):
    run_eval.add_usage(results, "2026-01-01", 5000)  # gemini (default)
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    run_eval.add_usage(results, "2026-01-01", 700)
    assert run_eval.tokens_used_today(results, "2026-01-01") == 700
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "gemini")
    assert run_eval.tokens_used_today(results, "2026-01-01") == 5000
    legacy = results / run_eval.USAGE_FILE
    legacy.write_text(json.dumps({"date": "2026-01-02", "tokens": 90}) + "\n", encoding="utf-8")
    assert run_eval.tokens_used_today(results, "2026-01-02") == 0  # old ledger lines are Groq's
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    assert run_eval.tokens_used_today(results, "2026-01-02") == 90


def test_metrics_lines_carry_provider_and_model_numbers_only(tmp_path, monkeypatch):
    path = tmp_path / "m.jsonl"
    monkeypatch.setenv("LLM_APPROACH_METRICS", "1")
    monkeypatch.setenv("LLM_APPROACH_METRICS_PATH", str(path))
    metrics_mod.record(metrics_mod.TurnMetrics(llm_ms=12.0, llm_calls=1, llm_call_ms=[12.0]))
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    metrics_mod.record(metrics_mod.TurnMetrics())
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [(r["provider"], r["model"]) for r in rows] == [GEMINI, GROQ]
    summary = metrics_mod.summarize(path)
    assert summary["providers"] == [{"provider": p, "model": m} for p, m in sorted([GEMINI, GROQ])]
    assert summary["per_llm_call_ms"] == summary["per_groq_call_ms"]


def test_old_metrics_rows_summarise_as_groq(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps({"ts": 1, "llm_ms": 5.0, "total_ms": 9.0}) + "\n", encoding="utf-8")
    assert metrics_mod.summarize(path)["providers"] == [{"provider": GROQ[0], "model": GROQ[1]}]


def test_live_scenario_record_has_the_labels(monkeypatch):
    rec = run_scenarios.live_record(run_scenarios.TurnReport("S1", 1, "text", intent="X", action="y"))
    assert (rec["provider"], rec["model"]) == GEMINI and "text" not in json.dumps(rec)


# ---- report refuses to mix --------------------------------------------------------------------------


def add_gemini_records(results, n=2):
    run(["fresh"], results, make_decide([]), sample=n)  # active provider: gemini


def test_report_refuses_groq_and_gemini_in_one_report(results):
    write_old_records(results)  # groq
    add_gemini_records(results)  # gemini
    with pytest.raises(report.MixedProvidersError) as err:
        report.build_report(results)
    assert "groq" in str(err.value) and "gemini" in str(err.value) and "--provider" in str(err.value)


def test_report_main_refuses_with_a_clear_message_and_writes_nothing(results, capsys):
    write_old_records(results)
    add_gemini_records(results)
    assert report.main(["--results-dir", str(results)]) == 2
    assert "mix several provider/model pairs" in capsys.readouterr().out
    assert not (results / "comparison.md").exists()


def test_report_refuses_two_models_of_the_same_provider(results, monkeypatch):
    add_gemini_records(results)
    monkeypatch.setenv("LLM_APPROACH_GEMINI_MODEL", "gemini-other")
    run(["golden"], results, make_decide([]), sample=2)
    with pytest.raises(report.MixedProvidersError):
        report.build_report(results)


def test_report_refuses_a_mix_that_only_shows_in_the_live_scenario_log(results):
    add_gemini_records(results)
    live = results / "scenarios_live.jsonl"
    live.write_text(json.dumps({"scenario": "S1", "turn": 1, "status": "pass"}) + "\n", encoding="utf-8")  # old = groq
    with pytest.raises(report.MixedProvidersError):
        report.build_report(results)


def test_report_with_a_choice_covers_only_that_pair_and_labels_every_table(results):
    write_old_records(results, 3)
    add_gemini_records(results, 2)
    md, data = report.build_report(results, "gemini", "gemini-3.5-flash-lite")
    assert (data["provider"], data["model"]) == GEMINI
    assert data["coverage"]["golden"]["done"] == 0 and data["coverage"]["fresh"]["done"] == 2
    md_groq, data_groq = report.build_report(results, "groq")
    assert data_groq["coverage"]["golden"]["done"] == 3 and data_groq["coverage"]["fresh"]["done"] == 0
    assert (data_groq["provider"], data_groq["model"]) == GROQ
    for text, pair in ((md, GEMINI), (md_groq, GROQ)):
        assert f"provider {pair[0]}, model {pair[1]}" in text
        tables = [b for b in re.split(r"\n\n", text) if b.startswith("|")]
        assert tables
        for block in tables:  # every table is followed by its caption
            assert text[text.index(block) + len(block):].startswith(f"\n\n_provider {pair[0]}, model {pair[1]}, prompt ")


def test_report_labels_old_results_as_groq(results):
    write_old_records(results)
    md, data = report.build_report(results)
    assert (data["provider"], data["model"]) == GROQ and "provider groq, model openai/gpt-oss-20b" in md


def test_report_unknown_choice_is_an_error_not_an_empty_report(results):
    add_gemini_records(results)
    with pytest.raises(report.MixedProvidersError) as err:
        report.build_report(results, "groq")
    assert "No results for the requested" in str(err.value)


def test_report_offline_tables_say_so(results):
    (results / "robustness.json").write_text(json.dumps({
        "provider": "offline-mock", "model": "none", "n": 1, "clean": 1,
        "rows": [{"case": "c", "reply_kind": "k", "history_consistent": True, "state_unchanged_where_required": True,
                  "raw_error_leaked": False}]}), encoding="utf-8")
    add_gemini_records(results)
    md, _ = report.build_report(results)
    assert "_offline mocks, no LLM call_" in md


# ---- per-provider limits and the live runners ------------------------------------------------------------


def test_groq_limits_are_unchanged_and_gemini_limits_come_from_the_environment(monkeypatch):
    assert (config.spacing_s("groq"), config.daily_token_budget("groq")) == (30.0, 150_000)
    assert (run_eval.DEFAULT_SPACING, run_eval.DEFAULT_DAILY_BUDGET) == (30.0, 150_000)
    assert (config.spacing_s("gemini"), config.daily_token_budget("gemini")) == (5.0, None)  # 5 s default; no token budget
    monkeypatch.setenv("LLM_APPROACH_GEMINI_SPACING_S", "4.5")
    monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET", "123456")
    assert (config.spacing_s("gemini"), config.daily_token_budget("gemini")) == (4.5, 123456)
    for bad in ("", "abc", "-5", "0"):
        monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET", bad)
        assert config.daily_token_budget("gemini") is None


@pytest.fixture
def live_env(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)

    async def forbidden(*a, **k):
        raise AssertionError("the live runner must not have started")

    monkeypatch.setattr(llm_client, "decide", forbidden)


def test_live_eval_refuses_gemini_without_a_budget(live_env, capsys, monkeypatch):
    started = []
    monkeypatch.setattr(run_eval, "run_eval", lambda *a, **k: started.append(1))
    assert run_eval.main(["--dataset", "golden", "--sample", "1"]) == 2
    out = capsys.readouterr().out
    assert "LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET" in out and KEY not in out and started == []


def test_live_eval_starts_with_the_gemini_budget_and_spacing_from_the_environment(live_env, capsys, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET", "777000")
    monkeypatch.setenv("LLM_APPROACH_GEMINI_SPACING_S", "6")
    seen = {}

    async def fake_run_eval(names, **kw):
        seen.update(kw)
        return {"completed": 0, "planned": 0, "tokens": 0, "rate_limit_misses": 0, "stopped_by_budget": False}

    monkeypatch.setattr(run_eval, "run_eval", fake_run_eval)
    assert run_eval.main(["--dataset", "golden", "--sample", "1"]) == 0
    assert seen["daily_budget"] == 777000 and seen["spacing"] == 6.0
    out = capsys.readouterr().out
    assert "gemini-3.5-flash-lite" in out and KEY not in out


def test_live_eval_flags_override_the_environment(live_env, monkeypatch):
    seen = {}

    async def fake_run_eval(names, **kw):
        seen.update(kw)
        return {"completed": 0, "planned": 0, "tokens": 0, "rate_limit_misses": 0, "stopped_by_budget": False}

    monkeypatch.setattr(run_eval, "run_eval", fake_run_eval)
    assert run_eval.main(["--dataset", "golden", "--daily-token-budget", "1000", "--spacing", "9"]) == 0
    assert (seen["daily_budget"], seen["spacing"]) == (1000, 9.0)


def test_live_eval_checks_the_active_providers_key(monkeypatch, capsys):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-present-but-not-the-active-one")
    assert run_eval.main(["--dataset", "golden"]) == 2
    out = capsys.readouterr().out
    assert "GEMINI_API_KEY is not set" in out and "gsk-present" not in out


def test_live_run_eval_function_refuses_an_unset_gemini_budget(results):
    with pytest.raises(ValueError, match="LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET"):
        asyncio.run(run_eval.run_eval(["golden"], results_dir=results, decide=make_decide([]),
                                      sleep=lambda s: asyncio.sleep(0), out=lambda *_: None))


def test_live_scenarios_refuse_gemini_without_a_budget(live_env, capsys):
    assert run_scenarios.main(["--live", "--only", "S1"]) == 2
    out = capsys.readouterr().out
    assert "LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET" in out and "Loading guardrails" not in out


def test_stub_scenarios_need_no_provider_setup(monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "nonsense")
    assert run_scenarios.main(["--stub", "--only", "S1"]) == 0


def test_dry_run_is_provider_neutral_and_notes_that_tokenizers_differ(monkeypatch, capsys):
    outputs = []
    for provider in (None, "groq", "nonsense"):  # works with any provider setting, and needs no key
        if provider:
            monkeypatch.setenv("LLM_APPROACH_PROVIDER", provider)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert run_eval.main(["--dry-run", "--dataset", "all"]) == 0
        outputs.append(capsys.readouterr().out)
    assert "tokenizers differ" in outputs[0] and "Provider-neutral" in outputs[0]
    first_lines = [o.splitlines()[2:7] for o in outputs]  # the per-dataset numbers do not depend on the provider
    assert first_lines[0] == first_lines[1]


# ---- scripts and the server -------------------------------------------------------------------------------


def test_llm_info_reports_set_or_missing_and_never_the_value(monkeypatch, capsys):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert "gemini-3.5-flash-lite" in llm_info.describe() and "GEMINI_API_KEY: missing" in llm_info.describe()
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    assert llm_info.main() == 0
    out = capsys.readouterr().out
    assert "provider: gemini" in out and "GEMINI_API_KEY: set" in out and KEY not in out
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert "provider: groq" in llm_info.describe() and "GROQ_API_KEY: missing" in llm_info.describe()
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "nonsense")
    assert "nonsense" not in llm_info.describe()


def test_start_script_prints_provider_info_before_starting_and_never_reads_the_key():
    text = (Path(config.__file__).parent / "scripts" / "start-server.ps1").read_text(encoding="utf-8")
    assert text.index("llm_approach.llm_info") < text.index("uvicorn llm_approach.server_app:app")
    assert "--port 8001 --no-access-log" in text
    assert "API_KEY" not in text  # the key is only looked at (not printed) inside llm_info


def test_ready_reports_the_gemini_key_and_labels_without_the_value(server_without_keys, monkeypatch):  # noqa: F811
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)  # never read the developer's real .env
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "gemini")
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": True, "knowledge_base": True})
    with TestClient(server_app.create_app()) as client:
        assert client.get("/ready").json()["checks"]["llm_key_present"] is False
        monkeypatch.setenv("GEMINI_API_KEY", KEY)
        r = client.get("/ready")
    assert r.status_code == 200 and r.json()["checks"]["llm_key_present"] is True
    assert (r.json()["provider"], r.json()["model"]) == GEMINI and KEY not in r.text


@pytest.mark.parametrize("empty", [False, True])
def test_chat_without_gemini_key_gives_a_friendly_reply_not_a_crash(server_without_keys, monkeypatch, empty):  # noqa: F811
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(providers, "_gemini_client", None)
    if empty:
        monkeypatch.setenv("GEMINI_API_KEY", "  ")
    with TestClient(server_app.create_app()) as client:
        r = client.post("/chat", json={"message": "hello", "session_id": str(uuid.uuid4())},
                        headers={"Authorization": "Bearer anything"})
    assert r.status_code == 200 and r.json()["message"]
    for bad in ("Traceback", "KeyError", "GEMINI_API_KEY", "api_key", "google"):
        assert bad not in r.text
