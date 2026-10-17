"""Builds ``eval/results/comparison.md`` and ``comparison.json`` from whatever results exist.

Works on partial results: the coverage banner at the top says how many items of each dataset
are done. Inputs (all optional): ``<dataset>.jsonl`` (run_eval), ``scenarios_live.jsonl``,
``safety.json``, ``robustness.json`` and ``baseline_current.json``.

    python -m llm_approach.eval.report

No key, token, message text or model reasoning is read or written: records hold ids,
labels, numbers and verdicts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_approach.eval import datasets, label_map, metrics  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINE = "baseline_current.json"
POOLED_133 = ["golden", "fresh", "matrix55"]  # the baseline's pooled set
POOLED_166 = POOLED_133 + ["ctxY"]
INTENT_SETS = datasets.INTENT_DATASETS
TARGETS = [  # PLAN.md section 8.3
    ("Safety block rate", "100%"),
    ("Invalid-search rate", "0"),
    ("E1 scenario pass rate", "100% of turns, or each failure explained"),
    ("Intent accuracy, E2 pooled (133)", "clearly above the current pipeline's 53.4% (harness: 57.9%)"),
    ("Intent accuracy, E2 ctxY", "clearly above 21.2% (harness: 27.3%)"),
    ("Consistency across 3 runs", ">= 95% identical decisions"),
    ("p95 LLM latency", "<= 3 s (rooms API excluded)"),
]


# ---- helpers ---------------------------------------------------------------------


def pct(p: dict[str, Any] | None) -> str:
    if not p or not p.get("n"):
        return "n/a"
    return f"{100 * p['value']:.1f}% [{100 * p['lo']:.1f}-{100 * p['hi']:.1f}] ({p['k']}/{p['n']})"


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: Path) -> list[dict]:
    from llm_approach.eval.run_eval import read_records

    return read_records(path)


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    return {name: read_jsonl(results_dir / f"{name}.jsonl") for name in datasets.DATASETS}


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def ms(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f} ms"


# ---- sections --------------------------------------------------------------------


def scenario_summary(records: list[dict]) -> dict[str, Any]:
    latest: dict[tuple, dict] = {}
    for r in records:
        latest[(r["scenario"], int(r["turn"]))] = r
    per: dict[str, dict[str, Any]] = {}
    for (sid, turn), r in sorted(latest.items(), key=lambda kv: (int(kv[0][0][1:]), kv[0][1])):
        d = per.setdefault(sid, {"turns": 0, "pass": 0, "fail": 0, "skipped": 0, "failed": []})
        d["turns"] += 1
        d[r["status"]] = d.get(r["status"], 0) + 1
        if r["status"] == "fail":
            d["failed"].append(f"turn {turn}: " + ", ".join(r.get("failed_checks") or ["?"]))
    ran = sum(d["pass"] + d["fail"] for d in per.values())
    return {"per_scenario": per, "turns_run": ran, "turns_passed": sum(d["pass"] for d in per.values())}


def build_data(results_dir: Path) -> dict[str, Any]:
    by_ds = load_results(results_dir)
    all_items = {n: datasets.load_dataset(n) for n in datasets.DATASETS}
    totals = {n: len(v) for n, v in all_items.items()}
    flat = [r for recs in by_ds.values() for r in recs]
    baseline = load_json(results_dir / BASELINE) or {}
    base_rows = (baseline.get("harness_setup_A") or {}).get("rows") or []
    # a golden-set row of the baseline is keyed 'golden:gN' and a twins row 'twins:...'; same as our item ids.

    def pooled(names: list[str]) -> list[dict]:
        return [r for n in names for r in by_ds[n]]

    data: dict[str, Any] = {"coverage": metrics.coverage(flat, totals), "baseline_loaded": bool(base_rows)}

    acc: dict[str, Any] = {}
    for label, names in [(n, [n]) for n in INTENT_SETS] + [("pooled_133", POOLED_133), ("pooled_166", POOLED_166)]:
        recs = pooled(names)
        first = metrics.select_run(recs)
        acc[label] = {"strict": metrics.intent_accuracy(first), "lenient": metrics.intent_accuracy(first, True),
                      "operation_strict": metrics.operation_accuracy(first),
                      "operation_lenient": metrics.operation_accuracy(first, True),
                      "paired_strict": metrics.paired_comparison(first, base_rows) if base_rows else None,
                      "paired_lenient": metrics.paired_comparison(first, base_rows, True) if base_rows else None}
    data["intent_accuracy"] = acc

    first_166 = metrics.select_run(pooled(POOLED_166))
    data["by_category"] = {"strict": metrics.breakdown(first_166, "category"),
                           "lenient": metrics.breakdown(first_166, "category", True)}
    data["by_label"] = metrics.breakdown(first_166, "gold_label")
    data["clarification"] = metrics.clarification_pr(first_166)
    data["invalid_slot_rate"] = metrics.invalid_slot_rate(first_166 + metrics.select_run(by_ds["golden_slots"]))
    data["invalid_search_rate"] = metrics.invalid_search_rate(first_166 + metrics.select_run(by_ds["golden_slots"]))
    every = [r for r in flat]
    data["invalid_json_rate"] = metrics.invalid_json_rate(every)
    data["consistency"] = metrics.consistency(every)
    data["calibration"] = metrics.calibration(first_166)
    data["latency"] = metrics.latency_percentiles(every)
    data["tokens"] = metrics.tokens_per_call(every)
    data["rate_limit_misses"] = sum(r.get("status") == "rate_limit_miss" for r in flat)
    data["slots_golden_slots"] = metrics.slot_accuracy(metrics.select_run(by_ds["golden_slots"]))
    data["slots_golden_slots_n"] = len([r for r in by_ds["golden_slots"] if r.get("gold_slots") is not None])

    # items that could not be represented faithfully
    caveats: dict[str, dict[str, list[str]]] = {}
    for n in INTENT_SETS:
        for it in all_items[n]:
            if it["caveat"]:
                caveats.setdefault(n, {}).setdefault(it["caveat"], []).append(it["item"])
    data["unfaithful_items"] = caveats
    flagged = {i for d in caveats.values() for ids in d.values() for i in ids}
    ctxy = [r for r in first_166 if r.get("ctx")]
    data["ctx_flagged_vs_clean"] = {
        "flagged": metrics.intent_accuracy([r for r in ctxy if r["item"] in flagged]),
        "clean": metrics.intent_accuracy([r for r in ctxy if r["item"] not in flagged]),
    }

    # baseline figures as stored
    h = baseline.get("harness_setup_A") or {}
    e2 = (baseline.get("E2_evaluate_py_pipeline_as_is") or {}).get("sections") or {}
    data["baseline"] = {
        "harness_pooled_133_strict_pct": (h.get("pooled_matrix55_fresh_golden") or {}).get("strict_pct"),
        "harness_ctxY_strict_pct": (h.get("twins") or {}).get("strict_pct"),
        "harness_replay_misses": h.get("replay_misses"),
        "evaluate_py_pooled_133_strict_pct": (e2.get("POOLED matrix+fresh+golden") or {}).get("strict_pct"),
        "evaluate_py_ctxY_strict_pct": (e2.get("ctxY twins (NEW_SEARCH, ctx=Y)") or {}).get("strict_pct"),
    }

    data["scenarios"] = scenario_summary(read_jsonl(results_dir / "scenarios_live.jsonl"))
    data["safety"] = load_json(results_dir / "safety.json")
    data["robustness"] = load_json(results_dir / "robustness.json")
    return data


# ---- markdown --------------------------------------------------------------------


def banner(data: dict[str, Any]) -> list[str]:
    cov = data["coverage"]
    done, total = sum(c["done"] for c in cov.values()), sum(c["total"] for c in cov.values())
    lines = ["# LLM approach: evaluation and comparison", ""]
    if done < total:
        lines += [f"> **PARTIAL RESULTS: {done} of {total} items evaluated.** Every number below covers only the items done so far.", ""]
    lines += [table(["Dataset", "Items done", "Total", "Rate-limit misses"],
                    [[n, cov[n]["done"], cov[n]["total"], cov[n]["rate_limit_misses"]] for n in datasets.DATASETS]), ""]
    if data["rate_limit_misses"]:
        lines += [f"Rate-limit misses ({data['rate_limit_misses']}) are not counted as wrong answers anywhere. "
                  "Re-run with `--only-missing` to retry them.", ""]
    return lines


def method_notes(data: dict[str, Any]) -> list[str]:
    b = data["baseline"]
    return [
        "## What is compared, and how", "",
        "- **All data is synthetic** (the POC test sets). Results show direction, not production accuracy.",
        "- **Level of comparison.** The new decision layer is scored at the **intent level** (10 intents, "
        "`label_map.py`). The old router's published figures are **end-to-end routing** against the 23 POC labels. "
        "The paired table below maps the old router's per-item prediction to the same 10 intents, so it is like for like; "
        "the stored label-level figures are shown only for reference. Mapping to 10 intents merges GREETING, ACKNOWLEDGMENT "
        "and CLOSING into GENERAL_CHAT, which makes intent-level scores slightly easier than label-level ones.",
        f"- **Baseline used.** The real `run_graph` harness (setup A): **{b['harness_pooled_133_strict_pct']}% pooled (133: golden + fresh + matrix55)** "
        f"and **{b['harness_ctxY_strict_pct']}% on ctxY (33)**. For reference, the `evaluate.py` figures quoted earlier: "
        f"{b['evaluate_py_pooled_133_strict_pct']}% pooled and {b['evaluate_py_ctxY_strict_pct']}% ctxY. "
        f"The harness golden rows had {b['harness_replay_misses']} replay misses, so its golden figure is less certain.",
        "- **Pooled sets.** *pooled 133* = golden + fresh + matrix55 (the baseline's pooled set); *pooled 166* adds the 33 ctxY items. Each table says which one it uses.",
        "- **Strict / lenient.** Strict: predicted intent equals the gold label's intent. Lenient: the gold intent or the intent of an `acceptable` label.",
        "- **Brackets** are Wilson 95% confidence intervals. Headline numbers use the first run of every item.",
        "- **ROOM_TABLE** is not in the PLAN 6.4 mapping. It was added as ROOM_FOLLOWUP / ROOM_DETAILS (decision recorded in the README).",
        "- **Rooms.** Items with rooms on screen use the 10 fixture rooms (Bangalore, nothing selected, no history).",
        "- Never stored or shown here: keys, tokens, message text, model reasoning.", "",
    ]


def accuracy_section(data: dict[str, Any]) -> list[str]:
    acc = data["intent_accuracy"]
    rows = []
    for label in INTENT_SETS + ["pooled_133", "pooled_166"]:
        a = acc[label]
        rows.append([label, pct(a["strict"]), pct(a["lenient"])])
    lines = ["## Intent accuracy (new decision layer)", "",
             table(["Set", "Strict", "Lenient"], rows), ""]
    lines += ["### Paired comparison with the old router (same items, intent level, strict)", ""]
    rows = []
    for label in INTENT_SETS + ["pooled_133", "pooled_166"]:
        p = acc[label]["paired_strict"]
        if not p or not p["n"]:
            rows.append([label, 0, "-", "-", "-", "-", "-", "-"])
            continue
        rows.append([label, p["n"], pct(p["new"]), pct(p["old"]), p["new_right_old_wrong"], p["new_wrong_old_right"],
                     p["both_right"], p["both_wrong"]])
    lines += [table(["Set", "n", "New", "Old", "New right, old wrong", "New wrong, old right", "Both right", "Both wrong"], rows), ""]
    lines += ["Lenient version (pooled sets):", ""]
    rows = []
    for label in ("pooled_133", "pooled_166"):
        p = acc[label]["paired_lenient"]
        if p and p["n"]:
            rows.append([label, p["n"], pct(p["new"]), pct(p["old"]), p["new_right_old_wrong"], p["new_wrong_old_right"],
                         p["both_right"], p["both_wrong"]])
    lines += [table(["Set", "n", "New", "Old", "New right, old wrong", "New wrong, old right", "Both right", "Both wrong"], rows) if rows else "_no data yet_", ""]
    lines += ["### Rooms on screen versus none (pooled 166, strict)", "",
              table(["Category", "Accuracy"], [[k, pct(v)] for k, v in data["by_category"]["strict"].items()] or [["-", "n/a"]]), "",
              "### By gold label (pooled 166, strict)", "",
              table(["Gold label", "Accuracy"], [[k, pct(v)] for k, v in data["by_label"].items()] or [["-", "n/a"]]), ""]
    return lines


def other_metrics_section(data: dict[str, Any]) -> list[str]:
    a = data["intent_accuracy"]["pooled_166"]
    clar, cal = data["clarification"], data["calibration"]
    lat, tok = data["latency"], data["tokens"]
    lines = ["## Operation, clarification, validity", "",
             table(["Metric", "Value"], [
                 ["Operation accuracy, ROOM_FOLLOWUP items (pooled 166, strict)", pct(a["operation_strict"])],
                 ["Operation accuracy (lenient)", pct(a["operation_lenient"])],
                 ["Clarification precision (gold ask = UNCLEAR)", pct(clar["precision"])],
                 ["Clarification recall", pct(clar["recall"])],
                 ["Invalid-slot rate (search intents, validators rejected a slot)", pct(data["invalid_slot_rate"])],
                 ["Invalid-search rate (a search would carry a rejected slot; target 0)", pct(data["invalid_search_rate"])],
                 ["No usable decision (invalid JSON twice, timeout, API error)", pct(data["invalid_json_rate"])],
                 ["Consistency across runs (items with 2+ runs, identical decision)", pct(data["consistency"])],
             ]), "",
             "The clarification gold is only the UNCLEAR label; the sets have no other 'should ask' labels, so read these as a proxy. "
             "The invalid-search rate is evaluated with the pipeline's own rule (a rejected city, BHK or price makes it ask instead of searching); "
             "the end-to-end check of that rule is in the pipeline tests and the robustness report.", "",
             "### Calibration (pooled 166, strict)", "",
             table(["Confidence", "Accuracy"], [[c["bucket"], pct(c)] for c in cal]), "",
             "## Cost", "",
             table(["Metric", "Value"], [
                 ["Latency p50 / p95 (LLM call incl. retry)", f"{ms(lat['p50'])} / {ms(lat['p95'])} (n={lat['n']})"],
                 ["Tokens in / out per call", f"{tok['mean_in']:.0f} / {tok['mean_out']:.0f}" if tok["n"] else "n/a"],
                 ["Cached tokens per call", f"{tok['mean_cached']:.0f}" if tok.get("mean_cached") is not None else "not reported"],
             ]), ""]
    sl = data["slots_golden_slots"]
    n = data["slots_golden_slots_n"]
    lines += ["## Slot accuracy (`golden_slots`, reported separately from intent accuracy)", ""]
    if n:
        lines += [table(["Field", "Accuracy"], [[k, pct(v)] for k, v in sl.items()]), "",
                  f"Over the {n} search items that have slot gold (`tests/golden_set.jsonl`, `expected_args`). Compared on the validated slots; "
                  "no gold value means the field must be empty. Gold `room_types` other than N-BHK (PG, Single, 1RK) is not a slot of the decision schema, "
                  "and gold cities such as 'HSR Layout' are localities (the model returns them as `area`), so those items lose city accuracy by construction. "
                  "There is no paired comparison for this set.", ""]
    else:
        lines += ["_no data yet_", ""]
    return lines


def faithfulness_section(data: dict[str, Any]) -> list[str]:
    lines = ["## Items that could not be represented faithfully", "",
             "They were still run. The data records only a yes/no flag for rooms on screen, so the following cannot be rebuilt exactly:",
             "- a **selected room** (follow-ups such as 'does it have wifi' may assume one);",
             "- a **pending bot question** (a bare 'pune' or 'ok' may answer one);",
             "- **room-dependent gold** written against the old harness's recorded rooms, while the new run uses the 10 fixture rooms.", ""]
    rows = []
    for n, reasons in data["unfaithful_items"].items():
        for reason, ids in reasons.items():
            rows.append([n, len(ids), reason])
    lines += [table(["Dataset", "Items", "Reason"], rows) if rows else "_none_", ""]
    f, c = data["ctx_flagged_vs_clean"]["flagged"], data["ctx_flagged_vs_clean"]["clean"]
    lines += [f"Intent accuracy on rooms-on-screen items: flagged {pct(f)}; not flagged {pct(c)}. "
              "A large gap means part of the result depends on the missing context rather than on the model.", ""]
    return lines


def scenario_section(data: dict[str, Any]) -> list[str]:
    s = data["scenarios"]
    lines = ["## Scenario pass/fail (live, `scenarios_live.jsonl`)", ""]
    if not s["per_scenario"]:
        return lines + ["_no live scenario results yet_", ""]
    rows = [[sid, d["turns"], d["pass"], d["fail"], d.get("skipped", 0), "; ".join(d["failed"]) or "-"]
            for sid, d in s["per_scenario"].items()]
    return lines + [f"Turns passed: {s['turns_passed']} of {s['turns_run']} run.", "",
                    table(["Scenario", "Turns", "Pass", "Fail", "Skipped", "Failed checks"], rows), ""]


def safety_section(data: dict[str, Any]) -> list[str]:
    s = data["safety"]
    lines = ["## Safety (offline, `safety.json`)", ""]
    if not s:
        return lines + ["_not run yet: `python -m llm_approach.eval.run_safety`_", ""]
    a, fp, inj = s["attack_block_rate"], s["general_messages_false_positives"], s["room_text_injection"]
    lines += [table(["Check", "Result"], [
        ["Real guardrail used", s["real_guardrail_used"]],
        ["Attack block rate (s1-s6, s8, PR #25 phrases, other attacks)", f"{a['blocked']}/{a['n']}"],
        ["s1-s8 attacks not blocked", s["noguard_s1_s8_attacks"]["not_blocked"] or "none"],
        ["s7 (payment request, not a security block)", "blocked" if s["s7_payment_request"]["blocked_either"] else "not blocked (handled as UNSUPPORTED_ACTION)"],
        ["False positives on general messages", f"{fp['blocked_either']}/{fp['n']} {fp['blocked_ids'] or ''}"],
        ["Room-text injection neutralised", f"{inj['neutralised']}/{inj['n']}"],
    ]), ""]
    bad = [r["payload"] for r in inj["rows"] if not r["neutralised"]]
    if bad:
        lines += ["**Not neutralised** (the room sanitiser leaves these fence-breaking payloads in the prompt):", ""] + [f"- `{p!r}`" for p in bad] + [""]
    lines += [inj["note"], ""]
    return lines


def robustness_section(data: dict[str, Any]) -> list[str]:
    r = data["robustness"]
    lines = ["## Robustness (offline mocks, `robustness.json`)", ""]
    if not r:
        return lines + ["_not run yet: `python -m llm_approach.eval.run_robustness`_", ""]
    rows = [[x["case"], x["reply_kind"], "yes" if x["history_consistent"] else "NO",
             "yes" if x["state_unchanged_where_required"] else "NO", "LEAK" if x["raw_error_leaked"] else "no"] for x in r["rows"]]
    return lines + [f"{r['clean']} of {r['n']} cases clean.", "",
                    table(["Case", "Reply type", "History consistent", "State kept where required", "Raw error leaked"], rows), ""]


def targets_section(data: dict[str, Any]) -> list[str]:
    acc = data["intent_accuracy"]
    s, rb = data["safety"], data["robustness"]
    status = {
        "Safety block rate": (f"{s['attack_block_rate']['blocked']}/{s['attack_block_rate']['n']}" if s else "not run"),
        "Invalid-search rate": pct(data["invalid_search_rate"]),
        "E1 scenario pass rate": (f"{data['scenarios']['turns_passed']}/{data['scenarios']['turns_run']} turns"
                                  if data["scenarios"]["turns_run"] else "not run live"),
        "Intent accuracy, E2 pooled (133)": pct(acc["pooled_133"]["strict"]),
        "Intent accuracy, E2 ctxY": pct(acc["ctxY"]["strict"]),
        "Consistency across 3 runs": pct(data["consistency"]),
        "p95 LLM latency": ms(data["latency"]["p95"]),
    }
    lines = ["## Proposed targets (PLAN section 8.3)", "",
             table(["Metric", "Proposed target", "Current"], [[k, t, status[k]] for k, t in TARGETS]), ""]
    if rb:
        lines += [f"Robustness: {rb['clean']}/{rb['n']} cases clean.", ""]
    return lines


def render(data: dict[str, Any]) -> str:
    parts = banner(data) + method_notes(data) + accuracy_section(data) + other_metrics_section(data) + \
        faithfulness_section(data) + scenario_section(data) + safety_section(data) + robustness_section(data) + targets_section(data)
    return "\n".join(parts).rstrip() + "\n"


def build_report(results_dir: Path = RESULTS_DIR) -> tuple[str, dict[str, Any]]:
    data = build_data(results_dir)
    return render(data), data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m llm_approach.eval.report")
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rd = Path(args.results_dir)
    md, data = build_report(rd)
    (rd / "comparison.md").write_text(md, encoding="utf-8")
    (rd / "comparison.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    done = sum(c["done"] for c in data["coverage"].values())
    total = sum(c["total"] for c in data["coverage"].values())
    print(f"Wrote {rd / 'comparison.md'} and comparison.json ({done} of {total} items evaluated).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
