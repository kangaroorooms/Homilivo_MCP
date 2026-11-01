"""Phase 0 baseline: score the CURRENT router on the evaluation sets.

Read-only use of existing code (imports and subprocesses only):
  * poc_classifier/noguard/harness.py  -> real run_graph, replay rooms client,
    Groq and pricing stubbed, setup A (all guardrails on). Per-row results.
  * poc_classifier/evaluate.py         -> its "a) pipeline as-is" lines (E2 numbers).
  * tests/run_golden_eval.py           -> E4 golden set accuracy.
  * poc_classifier/noguard/results/    -> existing A/B/F files (E3), summarised only.

No Groq call, no rooms API call. Writes only llm_approach/eval/results/baseline_current.json.

Run from the repo root:  python -m llm_approach.eval.baseline_current
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POC = ROOT / "poc_classifier"
NOGUARD = POC / "noguard"
OUT = Path(__file__).resolve().parent / "results" / "baseline_current.json"

for p in (ROOT, POC, NOGUARD):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _score(recs):
    n = len(recs)
    strict = sum(r["pred"] == r["gold"] for r in recs)
    lenient = sum(r["pred"] == r["gold"] or r["pred"] in (r["acceptable"] or []) for r in recs)
    return {"n": n, "strict_correct": strict, "lenient_correct": lenient,
            "strict_pct": round(100 * strict / n, 1) if n else None,
            "lenient_pct": round(100 * lenient / n, 1) if n else None}


def run_harness_setup_a():
    import harness  # noqa: E402  (poc_classifier/noguard/harness.py; sets offline env vars)
    from common import load_rows, load_recorded  # noqa: E402

    recorded = load_recorded()
    rooms = harness.load_base_rooms(recorded)
    if rooms is None:
        return None, "no recorded base search in poc_classifier/noguard/recorded"

    rows = load_rows()
    with open(POC / "data" / "test_golden.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                rows.append(dict(set="golden", id=f"g{i}", text=r["text"], label=r["label"],
                                 acceptable=r.get("acceptable", []), ctx=bool(r["has_room_context"]),
                                 known_fail=False))

    from fake_client import ReplayClient  # noqa: E402

    async def go():
        client = ReplayClient(recorded)
        with harness.patched("A"):  # warm-up: loads the embedding model, not recorded
            with contextlib.redirect_stdout(io.StringIO()):
                await harness.run_graph(user_query="hello", mode="fast", client=client,
                                        state=harness.make_state(False, rooms))
        client.calls.clear()
        return await harness.run_set(rows, "A", client, rooms)

    return asyncio.run(go()), None


def summarise(recs):
    by_set = collections.defaultdict(list)
    for r in recs:
        by_set[r["set"]].append(r)
    out = {name: _score(rs) for name, rs in by_set.items() if name != "safety"}
    pooled = [r for r in recs if r["set"] in ("matrix55", "fresh", "golden")]
    out["pooled_matrix55_fresh_golden"] = _score(pooled)
    safety = by_set.get("safety", [])
    out["safety"] = {
        "n": len(safety),
        "blocked_or_refused": sum(r["response_type"] == "REFUSAL" for r in safety),
        "hard_blocked": sum(r["blocked_by"] == "hard_block" for r in safety),
        "not_blocked_uids": [r["uid"] for r in safety if r["response_type"] != "REFUSAL"],
    }
    out["errors"] = sum(1 for r in recs if r["error"])
    out["replay_misses"] = sum(r["replay_misses"] for r in recs)
    out["known_fail_rows"] = [r["uid"] for r in recs if r["known_fail"]]
    return out


def compare_with_existing_a(recs):
    path = NOGUARD / "results" / "A.jsonl"
    if not path.exists():
        return None
    old = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                old[d["uid"]] = d["pred"]
    common = [r for r in recs if r["uid"] in old]
    diff = [r["uid"] for r in common if r["pred"] != old[r["uid"]]]
    return {"existing_rows": len(old), "compared": len(common), "pred_differs": diff}


def existing_noguard_summary():
    out = {}
    for setup in ("A", "B", "F"):
        path = NOGUARD / "results" / f"{setup}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        general = [r for r in recs if r["set"] != "safety"]
        out[setup] = {"rows_total": len(recs), "general": _score(general),
                      "safety_refusals": sum(1 for r in recs if r["set"] == "safety"
                                             and r["response_type"] == "REFUSAL"),
                      "safety_n": sum(1 for r in recs if r["set"] == "safety")}
    return out


def _run(cmd):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=900)
    return p.returncode, p.stdout


def evaluate_py_lines():
    code, text = _run([sys.executable, str(POC / "evaluate.py")])
    result, section = {}, None
    for line in text.splitlines():
        m = re.match(r"=== (.+?)\s+n=(\d+)", line.strip()) or re.match(r"=== (POOLED[^=]*?)\s+n=(\d+)", line.strip())
        if m:
            section = m.group(1).strip()
            result[section] = {"n": int(m.group(2))}
            continue
        if section and "a) pipeline as-is" in line and "strict" not in result[section]:
            nums = re.findall(r"(\d+)/(\d+)\s+([\d.]+)%", line)
            if len(nums) >= 2:
                result[section].update(strict=f"{nums[0][0]}/{nums[0][1]}", strict_pct=float(nums[0][2]),
                                       lenient=f"{nums[1][0]}/{nums[1][1]}", lenient_pct=float(nums[1][2]))
    return {"exit_code": code, "sections": result}


def golden_eval():
    code, text = _run([sys.executable, str(ROOT / "tests" / "run_golden_eval.py")])
    tail = [l for l in text.splitlines() if re.search(r"accuracy|passed|overall", l, re.I)]
    return {"exit_code": code, "summary_lines": tail[-8:]}


def main():
    started = time.time()
    result = {"note": "Current router (read-only). No Groq or rooms API calls. Synthetic sets.",
              "branch_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                            capture_output=True, text=True).stdout.strip()}
    recs, err = run_harness_setup_a()
    if recs is None:
        result["harness_setup_A"] = {"skipped": err}
    else:
        result["harness_setup_A"] = summarise(recs)
        result["harness_setup_A"]["vs_existing_A_jsonl"] = compare_with_existing_a(recs)
        result["harness_setup_A"]["rows"] = [
            {k: r[k] for k in ("uid", "set", "gold", "acceptable", "ctx", "pred", "response_type",
                               "route", "intent", "operation", "latency_ms")} for r in recs]
    result["E2_evaluate_py_pipeline_as_is"] = evaluate_py_lines()
    result["E3_existing_noguard_results"] = existing_noguard_summary()
    result["E4_golden_set_run_golden_eval"] = golden_eval()
    result["E5_hinglish"] = {"skipped": "no labelled 32-query file found; only unit tests in "
                                         "tests/test_p2_09_hinglish_handling.py"}
    result["elapsed_s"] = round(time.time() - started, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
