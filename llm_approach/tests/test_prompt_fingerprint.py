"""Prompt fingerprint: on every eval record, and the report refusing to mix prompt versions. Offline."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from llm_approach import prompt_fingerprint, prompts
from llm_approach.eval import report, run_eval, run_scenarios, runs
from llm_approach.tests.test_eval_run import make_decide, records, run


@pytest.fixture
def results(tmp_path):
    baseline = Path(report.RESULTS_DIR) / "baseline_current.json"
    if baseline.exists():
        shutil.copy(baseline, tmp_path / "baseline_current.json")
    return tmp_path


def rewrite_fingerprint(path, value):
    """Mark every record of a results file as made with another prompt (None = written before fingerprints)."""
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r.pop("prompt_fp", None)
        if value is not None:
            r["prompt_fp"] = value
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ---- the fingerprint itself ---------------------------------------------------------------------


def test_fingerprint_is_a_short_stable_hash_of_the_system_prompt():
    fp = prompt_fingerprint.current()
    assert re.fullmatch(r"[0-9a-f]{12}", fp)
    assert fp == prompt_fingerprint.current() == prompt_fingerprint.of_text(prompts.system_prompt())


def test_any_prompt_edit_changes_the_fingerprint():
    text = prompts.system_prompt()
    assert prompt_fingerprint.of_text(text) != prompt_fingerprint.of_text(text + " ")
    assert prompt_fingerprint.of_text(text) != prompt_fingerprint.of_text(text.replace("REFINEMENT", "REFINEMENT2", 1))


def test_a_record_without_a_fingerprint_is_the_pre_change_prompt():
    assert prompt_fingerprint.of_record({"item": "x"}) == prompt_fingerprint.LEGACY == "pre-change"
    assert prompt_fingerprint.of_record({"prompt_fp": "abc"}) == "abc"


def test_importing_the_module_does_not_load_the_prompt():
    code = "import sys; from llm_approach import prompt_fingerprint; assert 'llm_approach.prompts' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2]).returncode == 0


# ---- on the records ---------------------------------------------------------------------------------


def test_eval_records_carry_the_fingerprint(results):
    run(["golden"], results, make_decide([]), sample=2)
    assert {r["prompt_fp"] for r in records(results, "golden")} == {prompt_fingerprint.current()}


def test_live_scenario_record_carries_the_fingerprint():
    rec = run_scenarios.live_record(run_scenarios.TurnReport("S1", 1, "text", intent="X", action="y"))
    assert rec["prompt_fp"] == prompt_fingerprint.current() and "text" not in json.dumps(rec)


def test_results_from_another_prompt_do_not_count_as_done(results):
    run(["golden"], results, make_decide([]), sample=3)
    assert run_eval.plan_items(["golden"], 1, 3, results) == []
    rewrite_fingerprint(results / "golden.jsonl", None)  # an old file: no fingerprint
    assert len(run_eval.plan_items(["golden"], 1, 3, results)) == 3  # the new prompt must run them again
    assert run_eval.plan_items(["golden"], 1, 3, results, fp=prompt_fingerprint.LEGACY) == []
    rewrite_fingerprint(results / "golden.jsonl", "0123456789ab")
    assert len(run_eval.plan_items(["golden"], 1, 3, results)) == 3


def test_the_run_state_signature_has_the_fingerprint():
    sig = runs.signature("run_eval", "ctxY")
    assert sig["prompt_fp"] == prompt_fingerprint.current() and "prompt_fp" in runs.SIGNATURE_KEYS


def test_resume_refuses_an_unfinished_run_made_with_another_prompt(results):
    old = dict(runs.signature("run_eval", "ctxY"), prompt_fp="0123456789ab")
    runs.save_entry(results, old, "stopped", 5, 33, "daily token budget reached")
    reason = runs.begin(results, runs.signature("run_eval", "ctxY"), resume=True)
    assert reason and "prompt_fp" in reason


# ---- the report refuses to mix ------------------------------------------------------------------------


def test_report_refuses_results_of_two_prompt_versions(results):
    run(["golden"], results, make_decide([]), sample=3)
    run(["fresh"], results, make_decide([]), sample=3)
    rewrite_fingerprint(results / "fresh.jsonl", "0123456789ab")
    with pytest.raises(report.MixedPromptsError) as err:
        report.build_report(results)
    assert "mix several prompt versions" in str(err.value) and "--prompt-fp" in str(err.value)
    assert isinstance(err.value, report.MixedProvidersError)  # the CLI already handles that class


def test_a_file_with_old_and_new_records_is_also_refused(results):
    run(["golden"], results, make_decide([]), sample=2)
    rewrite_fingerprint(results / "golden.jsonl", None)
    run(["golden"], results, make_decide([]), sample=2)  # appends records of the current prompt
    with pytest.raises(report.MixedPromptsError):
        report.build_data(results)


def test_a_scenario_log_with_another_prompt_is_refused(results):
    run(["golden"], results, make_decide([]), sample=2)
    (results / "scenarios_live.jsonl").write_text(
        json.dumps({"provider": "gemini", "model": "gemini-3.5-flash-lite", "scenario": "S1", "turn": 1,
                    "status": "pass", "prompt_fp": "0123456789ab"}) + "\n", encoding="utf-8")
    with pytest.raises(report.MixedPromptsError):
        report.build_data(results)


def test_prompt_fp_chooses_one_version_and_labels_every_table(results):
    run(["golden"], results, make_decide([]), sample=3)
    run(["fresh"], results, make_decide([]), sample=2)
    rewrite_fingerprint(results / "fresh.jsonl", None)  # fresh: pre-change
    md_new, new = report.build_report(results, prompt_fp=prompt_fingerprint.current())
    assert new["prompt_fp"] == prompt_fingerprint.current()
    assert new["coverage"]["golden"]["done"] == 3 and new["coverage"]["fresh"]["done"] == 0
    md_old, old = report.build_report(results, prompt_fp="pre-change")
    assert old["prompt_fp"] == "pre-change"
    assert old["coverage"]["golden"]["done"] == 0 and old["coverage"]["fresh"]["done"] == 2
    for text, fp in ((md_new, prompt_fingerprint.current()), (md_old, "pre-change")):
        assert f"prompt {fp}" in text
        tables = [b for b in re.split(r"\n\n", text) if b.startswith("|")]
        assert tables
        for block in tables:  # every table is followed by its caption
            assert text[text.index(block) + len(block):].startswith(f"\n\n_provider gemini, model gemini-3.5-flash-lite, prompt {fp}_")


def test_unknown_prompt_fp_is_an_error_not_an_empty_report(results):
    run(["golden"], results, make_decide([]), sample=2)
    with pytest.raises(report.MixedPromptsError) as err:
        report.build_report(results, prompt_fp="ffffffffffff")
    assert "No results for the requested prompt version" in str(err.value)


def test_old_results_without_a_fingerprint_report_as_pre_change(results):
    run(["golden"], results, make_decide([]), sample=3)
    rewrite_fingerprint(results / "golden.jsonl", None)
    md, data = report.build_report(results)  # all records are pre-change: no mixing
    assert data["prompt_fp"] == "pre-change" and "prompt pre-change" in md


def test_cli_exits_2_with_the_message_on_mixed_prompts(results, capsys):
    run(["golden"], results, make_decide([]), sample=2)
    rewrite_fingerprint(results / "golden.jsonl", None)
    run(["golden"], results, make_decide([]), sample=2)
    assert report.main(["--results-dir", str(results)]) == 2
    assert "mix several prompt versions" in capsys.readouterr().out
    assert not (results / "comparison.md").exists()
    assert report.main(["--results-dir", str(results), "--prompt-fp", "pre-change"]) == 0
