"""Evaluation measurement fixes: invalid-search from the real pipeline code, the unusable-decision rate, few-shot overlap,
slot accuracy with invalid records, the provider that answered, and the scenario assertion audit. Offline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from llm_approach import config, prompt_fingerprint
from llm_approach.actions import search as search_mod
from llm_approach.eval import datasets, metrics, report, run_eval, run_scenarios, scenario_audit
from llm_approach.schema import parse_decision

README = Path(__file__).resolve().parents[1] / "README.md"


def item(ctx=False, **kw):
    base = {"dataset": "golden", "item": "golden:gX", "text": "x", "category": "ctxN", "gold_label": None, "gold_intent": "NEW_SEARCH",
            "gold_operation": None, "gold_slots": None, "acceptable_intents": [], "acceptable_ops": [], "ctx": ctx, "caveat": None}
    return {**base, **kw}


def budget(**kw):
    return SimpleNamespace(**{"error_class": None, "error_code": None, "error_status": None, "latency_ms": 5.0, "tokens_in": 10,
                              "tokens_out": 5, "cached_tokens": None, "calls": 1, "use_provider": None, **kw})


def decision(**d):
    return parse_decision({"intent": "NEW_SEARCH", "confidence": 0.9, **d})


# ---- a. invalid-search from the pipeline's own code ---------------------------------------------------------------


def test_a_good_search_would_search_and_passes_the_guardrail():
    rec = run_eval.build_record(item(), 1, decision(slots={"city": "Pune", "max_price": 15000}), budget(), "ok", 0)
    assert rec["would_search"] is True and rec["search_args_valid"] is True


def test_a_rejected_slot_makes_the_real_rule_ask_instead_of_search():
    rec = run_eval.build_record(item(), 1, decision(slots={"city": "Pune", "bhk": 9}), budget(), "ok", 0)
    assert "bhk" in rec["rejected"] and rec["would_search"] is False and rec["search_args_valid"] is None


def test_a_search_without_a_city_is_not_a_search():
    rec = run_eval.build_record(item(), 1, decision(slots={"max_price": 15000}), budget(), "ok", 0)
    assert rec["would_search"] is False


def test_a_refinement_continues_the_current_search():
    rec = run_eval.build_record(item(ctx=True), 1, decision(intent="REFINEMENT", slots={"max_price": 15000}), budget(), "ok", 10)
    assert rec["would_search"] is True and rec["search_args_valid"] is True


def test_the_invalid_search_rate_can_be_above_zero(monkeypatch):
    real = search_mod.guardrail_arguments
    monkeypatch.setattr(search_mod, "guardrail_arguments", lambda s: {**real(s), "min_price": -5})  # a bug the validators missed
    rec = run_eval.build_record(item(), 1, decision(slots={"city": "Pune"}), budget(), "ok", 0)
    assert rec["would_search"] is True and rec["search_args_valid"] is False
    assert metrics.invalid_search_rate([rec])["k"] == 1


def test_old_records_keep_their_meaning():
    old = {"status": "ok", "intent": "NEW_SEARCH", "would_search": True, "rejected": ["city"]}
    clean = {"status": "ok", "intent": "NEW_SEARCH", "would_search": True, "rejected": []}
    assert metrics.invalid_search_rate([old, clean])["k"] == 1


# ---- b. no usable decision --------------------------------------------------------------------------------------


def test_timeouts_and_api_errors_count_as_no_usable_decision_and_the_old_line_stays():
    rows = [{"status": "ok"}, {"status": "ok"}, {"status": "invalid"}, {"status": "error"}, {"status": "error"},
            {"status": "rate_limit_miss"}]
    new, old = metrics.no_usable_decision_rate(rows), metrics.invalid_json_rate(rows)
    assert (new["k"], new["n"]) == (3, 5)  # invalid + 2 errors, among ok + invalid + error; the throttled call is left out
    assert (old["k"], old["n"]) == (1, 3)  # the earlier definition, unchanged


# ---- c. few-shot overlap ------------------------------------------------------------------------------------------


def test_the_ten_verbatim_few_shot_items_are_found_and_nothing_is_edited():
    before = prompt_fingerprint.current()
    ids = datasets.few_shot_item_ids()
    assert len(ids) == 10 and {"golden:g10", "golden:g11", "matrix55:24", "fresh:f47"} <= ids
    assert prompt_fingerprint.current() == before == "20f328e351bb"
    assert sum(len(datasets.load_dataset(n)) for n in datasets.DATASETS) == 186  # the sets are read, never changed


def rec(item_id, intent, gold, run=1):
    return {"dataset": "golden", "item": item_id, "run": run, "status": "ok", "intent": intent, "gold_intent": gold, "category": "ctxN",
            "gold_label": gold, "ctx": False, "acceptable_intents": [], "confidence": 0.9}


def test_the_report_gives_accuracy_with_and_without_the_few_shot_items(tmp_path):
    rows = [rec("golden:g10", "GENERAL_CHAT", "GENERAL_CHAT"),  # "hi": a few-shot example, right
            rec("golden:g11", "GENERAL_CHAT", "GENERAL_CHAT"),  # "thanks": a few-shot example, right
            rec("golden:g1", "NEW_SEARCH", "REFINEMENT"), rec("golden:g2", "REFINEMENT", "REFINEMENT")]
    (tmp_path / "golden.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    data = report.build_data(tmp_path, prompt_fp="pre-change")
    acc = data["intent_accuracy"]["golden"]
    assert (acc["strict"]["k"], acc["strict"]["n"]) == (3, 4)
    assert (acc["strict_without_few_shot"]["k"], acc["strict_without_few_shot"]["n"]) == (1, 2)
    text = report.render(data)
    assert "Without the items that are word for word few-shot examples" in text and "golden:g10" in text


# ---- d. slot accuracy ----------------------------------------------------------------------------------------


def test_invalid_json_counts_as_wrong_in_slot_accuracy():
    gold = {"city": "Pune", "bhk": None, "min_price": None, "max_price": None, "ideal_for": None}
    ok = {"status": "ok", "gold_slots": gold, "validated_slots": {"city": "Pune"}}
    invalid = {"status": "invalid", "gold_slots": gold, "validated_slots": None}
    miss = {"status": "rate_limit_miss", "gold_slots": gold, "validated_slots": None}
    out = metrics.slot_accuracy([ok, invalid, miss])
    assert (out["city"]["k"], out["city"]["n"]) == (1, 2) and (out["exact"]["k"], out["exact"]["n"]) == (1, 2)


# ---- e. which provider answered -----------------------------------------------------------------------------


def test_a_fallback_answer_is_labelled_with_the_fallback_provider(monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "gemini")
    primary = run_eval.build_record(item(), 1, decision(slots={"city": "Pune"}), budget(), "ok", 0)
    fallback = run_eval.build_record(item(), 1, decision(slots={"city": "Pune"}), budget(use_provider="groq"), "ok", 0)
    assert (primary["provider"], primary["answered_by_fallback"]) == ("gemini", False)
    assert (fallback["provider"], fallback["answered_by_fallback"]) == ("groq", True)
    assert fallback["model"] == config.model("groq") and fallback["model"] != primary["model"]


def test_evaluate_item_records_the_provider_that_answered(monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "gemini")

    async def decide(text, summary, b):
        b.use_provider = "groq"
        return decision(slots={"city": "Pune"})

    async def sleep(_):
        return None

    record = asyncio.run(run_eval.evaluate_item(item(), 1, decide, sleep))
    assert record["provider"] == "groq" and record["answered_by_fallback"] is True


def test_the_report_counts_fallback_answers_and_keeps_them_out_of_the_primary_pair(tmp_path):
    rows = [{**rec("golden:g1", "NEW_SEARCH", "NEW_SEARCH"), "provider": "gemini", "model": "m1"},
            {**rec("golden:g2", "NEW_SEARCH", "NEW_SEARCH"), "provider": "groq", "model": "m2", "answered_by_fallback": True}]
    (tmp_path / "golden.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    data = report.build_data(tmp_path, provider="gemini", model="m1", prompt_fp="pre-change")
    assert data["fallback_answers"] == {"all_records": 1, "in_this_report": 0}
    assert data["intent_accuracy"]["golden"]["strict"]["n"] == 1 and "answered by a fallback provider" in report.render(data)


# ---- f. scenario turns that assert nothing -----------------------------------------------------------------------


def test_the_audit_lists_the_turns_that_assert_nothing_by_scenario():
    audit = scenario_audit.audit()
    assert audit["total_turns"] == 300 and audit["assert_nothing"] == 91 and audit["assert_nothing_in_live_scenarios"] == 1
    assert sum(len(e["turns"]) for e in audit["by_scenario"].values()) == 91
    first = audit["by_scenario"]["S36"]["turns"][0]
    baseline = json.loads((Path(scenario_audit.BASELINE)).read_text(encoding="utf-8"))
    assert [first["proposed"]["intent"], first["proposed"]["action"]] == baseline["S36"][first["turn"] - 1]
    assert scenario_audit.turn_ranges([1, 2, 3, 7, 9, 10]) == "1-3, 7, 9-10"


def test_the_audit_adds_no_expectation_to_a_scenario():
    before = {p.name: p.read_bytes() for p in scenario_audit.SCENARIOS.glob("S*.json")}
    scenario_audit.audit()
    assert before == {p.name: p.read_bytes() for p in scenario_audit.SCENARIOS.glob("S*.json")}


def test_the_report_lists_them(tmp_path):
    text = report.render(report.build_data(tmp_path, prompt_fp="pre-change"))
    assert "91 of 300 scenario turns assert nothing" in text and "S36" in text


# ---- g. stub numbers are circular ---------------------------------------------------------------------------------


def test_the_report_the_readme_and_a_stub_run_say_that_stub_numbers_are_circular(tmp_path, capsys):
    text = report.render(report.build_data(tmp_path, prompt_fp="pre-change"))
    assert "Stub-mode numbers are circular" in text and "not** that the model or the pipeline is accurate" in text
    assert "are **circular**" in README.read_text(encoding="utf-8") and "not accuracy evidence" in README.read_text(encoding="utf-8")
    run_scenarios.main(["--stub", "--only", "S1"])
    assert "Stub mode is circular" in capsys.readouterr().out
