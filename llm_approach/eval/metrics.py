"""Pure metric functions over evaluation result records. Offline, no I/O.

A result record (one line of ``results/<dataset>.jsonl``) has at least:
``dataset, item, run, status, intent, operation, validated_slots, rejected,
room_reference, confidence, needs_clarification, would_search, latency_ms,
tokens_in, tokens_out, cached_tokens`` plus the gold fields copied from the
dataset (``gold_intent, gold_operation, gold_slots, acceptable_intents,
acceptable_ops, category, gold_label``).

``status`` is ``ok`` (decision parsed), ``invalid`` (no decision: invalid JSON
twice, timeout or another Groq error) or ``rate_limit_miss`` (Groq throttled us
twice). A rate-limit miss is not a model answer: it is left out of every
denominator here and reported separately by ``coverage``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

from llm_approach.eval import label_map

SEARCH_INTENTS = {"NEW_SEARCH", "REFINEMENT"}
SEARCH_SLOT_FIELDS = ("city", "bhk", "min_price", "max_price")  # a rejected one stops the search
SLOT_FIELDS = ("city", "bhk", "min_price", "max_price", "ideal_for")
BUCKETS = [(0.0, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.0001)]


# ---- basics ---------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval for k successes in n trials (fractions)."""
    if n <= 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def proportion(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "value": (k / n) if n else None, "lo": lo, "hi": hi}


def scored(results: Iterable[dict]) -> list[dict]:
    """Records that count toward accuracy: everything except rate-limit misses."""
    return [r for r in results if r.get("status") in ("ok", "invalid")]


def select_run(results: Iterable[dict], run: int | None = None) -> list[dict]:
    """One record per item: the given run, or each item's lowest scored run number."""
    best: dict[tuple, dict] = {}
    for r in scored(results):
        key = (r["dataset"], r["item"])
        if run is not None:
            if r.get("run") == run:
                best[key] = r
        elif key not in best or r.get("run", 1) < best[key].get("run", 1):
            best[key] = r
    return list(best.values())


def coverage(results: Iterable[dict], totals: dict[str, int]) -> dict[str, dict[str, int]]:
    """Per dataset: items with at least one scored run, rate-limit misses, total items."""
    done: dict[str, set] = defaultdict(set)
    misses: dict[str, int] = defaultdict(int)
    for r in results:
        if r.get("status") in ("rate_limit_miss", "error"):  # no answer was obtained: not done
            misses[r["dataset"]] += 1
        else:
            done[r["dataset"]].add(r["item"])
    return {d: {"done": len(done[d]), "total": t, "rate_limit_misses": misses[d]} for d, t in totals.items()}


# ---- intent / operation ---------------------------------------------------


def intent_correct(r: dict, lenient: bool = False) -> bool:
    if r.get("status") != "ok" or r.get("gold_intent") is None:
        return False
    if r.get("intent") == r["gold_intent"]:
        return True
    return lenient and r.get("intent") in (r.get("acceptable_intents") or [])


def intent_accuracy(results: Iterable[dict], lenient: bool = False) -> dict[str, Any]:
    """An invalid record (no decision) counts as wrong; a rate-limit miss is not counted."""
    rows = [r for r in scored(results) if r.get("gold_intent") is not None]
    return proportion(sum(intent_correct(r, lenient) for r in rows), len(rows))


def operation_accuracy(results: Iterable[dict], lenient: bool = False) -> dict[str, Any]:
    """Among ROOM_FOLLOWUP gold items: predicted intent ROOM_FOLLOWUP and operation equals the gold operation."""
    rows = [r for r in scored(results) if r.get("gold_intent") == "ROOM_FOLLOWUP"]

    def ok(r: dict) -> bool:
        if r.get("status") != "ok" or r.get("intent") != "ROOM_FOLLOWUP":
            return False
        allowed = {r.get("gold_operation")}
        if lenient:
            allowed |= set(r.get("acceptable_ops") or [])
        return r.get("operation") in allowed

    return proportion(sum(ok(r) for r in rows), len(rows))


def breakdown(results: Iterable[dict], key: str = "category", lenient: bool = False) -> dict[str, dict[str, Any]]:
    """Intent accuracy per value of ``key`` (``category`` = ctxY/ctxN, or ``gold_label``)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in scored(results):
        if r.get("gold_intent") is not None:
            groups[str(r.get(key))].append(r)
    return {g: intent_accuracy(rows, lenient) for g, rows in sorted(groups.items())}


# ---- slots ----------------------------------------------------------------


def _norm(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


def slot_accuracy(results: Iterable[dict], fields: Iterable[str] = SLOT_FIELDS) -> dict[str, dict[str, Any]]:
    """Per field, over ok records with slot gold: validated value equals gold (no gold value means None).
    ``overall`` counts every (item, field) pair; ``exact`` counts items with all fields right."""
    fields = tuple(fields)
    rows = [r for r in results if r.get("status") == "ok" and r.get("gold_slots") is not None]

    def right(r: dict, f: str) -> bool:
        return _norm(r["gold_slots"].get(f)) == _norm((r.get("validated_slots") or {}).get(f))

    out: dict[str, dict[str, Any]] = {f: proportion(sum(right(r, f) for r in rows), len(rows)) for f in fields}
    out["overall"] = proportion(sum(right(r, f) for r in rows for f in fields), len(rows) * len(fields))
    out["exact"] = proportion(sum(all(right(r, f) for f in fields) for r in rows), len(rows))
    return out


def asked_for_clarification(r: dict) -> bool:
    return r.get("intent") == "UNCLEAR" or bool(r.get("needs_clarification"))


def clarification_pr(results: Iterable[dict]) -> dict[str, Any]:
    """Precision/recall of asking. Gold ask = gold intent UNCLEAR (the only clarification gold the sets have)."""
    rows = [r for r in results if r.get("status") == "ok" and r.get("gold_intent") is not None]
    tp = sum(asked_for_clarification(r) and r["gold_intent"] == "UNCLEAR" for r in rows)
    pred = sum(asked_for_clarification(r) for r in rows)
    gold = sum(r["gold_intent"] == "UNCLEAR" for r in rows)
    return {"precision": proportion(tp, pred), "recall": proportion(tp, gold), "tp": tp, "predicted": pred, "gold": gold}


# ---- validity ---------------------------------------------------------------


def invalid_slot_rate(results: Iterable[dict]) -> dict[str, Any]:
    """Share of search-intent decisions in which the validators rejected a slot."""
    rows = [r for r in results if r.get("status") == "ok" and r.get("intent") in SEARCH_INTENTS]
    return proportion(sum(bool(r.get("rejected")) for r in rows), len(rows))


def invalid_search_rate(results: Iterable[dict]) -> dict[str, Any]:
    """Decisions that would trigger a search while carrying a rejected city/BHK/price.
    ``would_search`` is recorded with the pipeline's rule (a rejected search slot makes
    it ask instead), so this is 0 unless that rule is broken."""
    rows = [r for r in results if r.get("status") == "ok" and r.get("intent") in SEARCH_INTENTS]
    bad = sum(bool(r.get("would_search")) and bool(set(r.get("rejected") or []) & set(SEARCH_SLOT_FIELDS)) for r in rows)
    return proportion(bad, len(rows))


def invalid_json_rate(results: Iterable[dict]) -> dict[str, Any]:
    """Calls that produced no usable decision (invalid JSON twice, timeout, API error), rate limits excluded."""
    rows = scored(results)
    return proportion(sum(r["status"] == "invalid" for r in rows), len(rows))


def decision_key(r: dict) -> tuple:
    slots = r.get("validated_slots") or {}
    return (r.get("status"), r.get("intent"), r.get("operation"), r.get("room_reference"),
            tuple(sorted((k, _norm(v)) for k, v in slots.items() if v not in (None, [], ""))))


def consistency(results: Iterable[dict]) -> dict[str, Any]:
    """Items with 2+ scored runs whose decisions are all identical."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in scored(results):
        groups[(r["dataset"], r["item"])].append(r)
    multi = [g for g in groups.values() if len(g) >= 2]
    same = sum(len({decision_key(r) for r in g}) == 1 for g in multi)
    return proportion(same, len(multi))


def calibration(results: Iterable[dict]) -> list[dict[str, Any]]:
    """Intent accuracy per confidence bucket."""
    rows = [r for r in results if r.get("status") == "ok" and r.get("gold_intent") is not None]
    out = []
    for lo, hi in BUCKETS:
        sel = [r for r in rows if lo <= (r.get("confidence") or 0.0) < hi]
        out.append({"bucket": f"{lo:.1f}-{min(hi, 1.0):.1f}", **intent_accuracy(sel)})
    return out


# ---- cost -------------------------------------------------------------------


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (pct in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def latency_percentiles(results: Iterable[dict]) -> dict[str, Any]:
    vals = [r["latency_ms"] for r in scored(results) if r.get("latency_ms") is not None]
    return {"n": len(vals), "p50": percentile(vals, 50), "p95": percentile(vals, 95)}


def tokens_per_call(results: Iterable[dict]) -> dict[str, Any]:
    rows = [r for r in scored(results) if r.get("tokens_in") is not None]
    if not rows:
        return {"n": 0, "mean_in": None, "mean_out": None, "mean_cached": None}
    cached = [r["cached_tokens"] for r in rows if r.get("cached_tokens") is not None]
    return {"n": len(rows), "mean_in": sum(r["tokens_in"] for r in rows) / len(rows),
            "mean_out": sum(r["tokens_out"] or 0 for r in rows) / len(rows),
            "mean_cached": (sum(cached) / len(cached)) if cached else None}


# ---- paired comparison with the old router ---------------------------------


def old_intent(baseline_row: dict) -> str | None:
    """Intent of the old router's prediction (its POC label mapped through the label map)."""
    return label_map.intent_of(baseline_row.get("pred"))


def old_correct(baseline_row: dict, lenient: bool = False) -> bool:
    gold = label_map.intent_of(baseline_row.get("gold"))
    pred = old_intent(baseline_row)
    if gold is None or pred is None:
        return False
    if pred == gold:
        return True
    return lenient and pred in label_map.acceptable_intents(baseline_row.get("acceptable"))


def paired_comparison(results: Iterable[dict], baseline_rows: Iterable[dict], lenient: bool = False) -> dict[str, Any]:
    """New (first scored run per item) versus the old router on the items present in both.
    Both are scored at intent level; ``old`` is the old router's accuracy on the same items."""
    old = {row["uid"]: row for row in baseline_rows}
    counts = {"both_right": 0, "new_right_old_wrong": 0, "new_wrong_old_right": 0, "both_wrong": 0}
    n = 0
    for r in select_run(results):
        base = old.get(r["item"])
        if base is None or r.get("gold_intent") is None:
            continue
        n += 1
        new_ok, old_ok = intent_correct(r, lenient), old_correct(base, lenient)
        if new_ok:
            counts["both_right" if old_ok else "new_right_old_wrong"] += 1
        else:
            counts["new_wrong_old_right" if old_ok else "both_wrong"] += 1
    new_k = counts["both_right"] + counts["new_right_old_wrong"]
    old_k = counts["both_right"] + counts["new_wrong_old_right"]
    return {"n": n, **counts, "new": proportion(new_k, n), "old": proportion(old_k, n)}
