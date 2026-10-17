"""metrics.py with small hand-checked inputs."""

from __future__ import annotations

import pytest

from llm_approach.eval import metrics as m


def rec(item, gold, pred, *, status="ok", ds="matrix55", run=1, acc=(), op=None, gold_op=None, conf=0.9, **kw):
    base = {"dataset": ds, "item": item, "run": run, "status": status, "gold_intent": gold,
            "intent": pred if status == "ok" else None, "acceptable_intents": list(acc), "operation": op,
            "gold_operation": gold_op, "acceptable_ops": [], "confidence": conf, "category": "ctxN",
            "gold_label": gold, "rejected": [], "validated_slots": {}}
    base.update(kw)
    return base


def test_wilson_hand_values():
    lo, hi = m.wilson(5, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.2366, 0.7634)
    lo, hi = m.wilson(1, 1)
    assert round(lo, 4) == 0.2065 and hi == 1.0
    lo, hi = m.wilson(0, 10)
    assert lo == 0.0 and round(hi, 4) == 0.2775
    assert m.wilson(0, 0) == (None, None)


def test_intent_accuracy_strict_lenient_and_invalid():
    rows = [rec("a", "NEW_SEARCH", "NEW_SEARCH"), rec("b", "UNCLEAR", "OUT_OF_SCOPE", acc=["OUT_OF_SCOPE"]),
            rec("c", "REFINEMENT", "NEW_SEARCH"), rec("d", "NEW_SEARCH", None, status="invalid")]
    assert (m.intent_accuracy(rows)["k"], m.intent_accuracy(rows)["n"]) == (1, 4)
    assert (m.intent_accuracy(rows, True)["k"], m.intent_accuracy(rows, True)["n"]) == (2, 4)


def test_rate_limit_miss_is_not_an_error():
    rows = [rec("a", "NEW_SEARCH", "NEW_SEARCH"), rec("b", "NEW_SEARCH", None, status="rate_limit_miss")]
    acc = m.intent_accuracy(rows)
    assert (acc["k"], acc["n"]) == (1, 1)
    assert m.invalid_json_rate(rows)["k"] == 0
    cov = m.coverage(rows, {"matrix55": 5})
    assert cov["matrix55"] == {"done": 1, "total": 5, "rate_limit_misses": 1}


def test_operation_accuracy():
    rows = [rec("a", "ROOM_FOLLOWUP", "ROOM_FOLLOWUP", op="LOWEST_PRICE", gold_op="LOWEST_PRICE"),
            rec("b", "ROOM_FOLLOWUP", "ROOM_FOLLOWUP", op="ROOM_PRICE", gold_op="LOWEST_PRICE"),
            rec("c", "ROOM_FOLLOWUP", "SHOW_MORE", gold_op="AMENITY"),
            rec("d", "NEW_SEARCH", "NEW_SEARCH")]
    o = m.operation_accuracy(rows)
    assert (o["k"], o["n"]) == (1, 3)
    rows[1]["acceptable_ops"] = ["ROOM_PRICE"]
    assert m.operation_accuracy(rows, lenient=True)["k"] == 2


def test_slot_accuracy_per_field():
    gold = {"city": "Pune", "bhk": 2, "min_price": None, "max_price": 15000, "ideal_for": None}
    r1 = rec("g1", "NEW_SEARCH", "NEW_SEARCH", gold_slots=gold,
             validated_slots={"city": "pune", "bhk": 2, "max_price": 15000})
    r2 = rec("g2", "NEW_SEARCH", "NEW_SEARCH", gold_slots=gold,
             validated_slots={"city": "Delhi", "bhk": 2, "max_price": 10000, "ideal_for": "Family"})
    s = m.slot_accuracy([r1, r2])
    assert (s["city"]["k"], s["city"]["n"]) == (1, 2)
    assert s["bhk"]["k"] == 2 and s["max_price"]["k"] == 1 and s["ideal_for"]["k"] == 1 and s["min_price"]["k"] == 2
    assert (s["overall"]["k"], s["overall"]["n"]) == (7, 10)  # 1 + 2 + 2 + 1 + 1 of 5 fields x 2 items
    assert (s["exact"]["k"], s["exact"]["n"]) == (1, 2)


def test_clarification_precision_recall():
    rows = [rec("a", "UNCLEAR", "UNCLEAR"), rec("b", "UNCLEAR", "NEW_SEARCH"),
            rec("c", "NEW_SEARCH", "UNCLEAR"), rec("d", "NEW_SEARCH", "NEW_SEARCH", needs_clarification=True),
            rec("e", "NEW_SEARCH", "NEW_SEARCH")]
    c = m.clarification_pr(rows)
    assert (c["tp"], c["predicted"], c["gold"]) == (1, 3, 2)
    assert c["precision"]["value"] == pytest.approx(1 / 3) and c["recall"]["value"] == 0.5


def test_invalid_slot_and_invalid_search_rate():
    rows = [rec("a", "NEW_SEARCH", "NEW_SEARCH", rejected=["city"], would_search=False),
            rec("b", "NEW_SEARCH", "NEW_SEARCH", would_search=True),
            rec("c", "ROOM_FOLLOWUP", "ROOM_FOLLOWUP", rejected=["room_reference"])]
    assert (m.invalid_slot_rate(rows)["k"], m.invalid_slot_rate(rows)["n"]) == (1, 2)
    assert m.invalid_search_rate(rows)["k"] == 0
    rows[0]["would_search"] = True  # the gate failed
    assert m.invalid_search_rate(rows)["k"] == 1


def test_invalid_json_rate():
    rows = [rec("a", "X", "X"), rec("b", "X", None, status="invalid"), rec("c", "X", "X"), rec("d", "X", "X")]
    assert m.invalid_json_rate(rows)["value"] == 0.25


def test_consistency():
    rows = [rec("a", "X", "NEW_SEARCH", run=1), rec("a", "X", "NEW_SEARCH", run=2), rec("a", "X", "NEW_SEARCH", run=3),
            rec("b", "X", "NEW_SEARCH", run=1), rec("b", "X", "REFINEMENT", run=2),
            rec("c", "X", "NEW_SEARCH", run=1)]  # one run: not counted
    c = m.consistency(rows)
    assert (c["k"], c["n"]) == (1, 2)
    rows.append(rec("d", "X", "NEW_SEARCH", run=1, validated_slots={"city": "Pune"}))
    rows.append(rec("d", "X", "NEW_SEARCH", run=2, validated_slots={"city": "pune"}))  # case-insensitive
    assert m.consistency(rows)["k"] == 2


def test_calibration_buckets():
    rows = [rec("a", "X", "X", conf=0.95), rec("b", "X", "Y", conf=0.92), rec("c", "X", "X", conf=0.5),
            rec("d", "X", "X", conf=0.7)]
    cal = {c["bucket"]: c for c in m.calibration(rows)}
    assert (cal["0.9-1.0"]["k"], cal["0.9-1.0"]["n"]) == (1, 2)
    assert (cal["0.0-0.6"]["k"], cal["0.0-0.6"]["n"]) == (1, 1)
    assert cal["0.8-0.9"]["n"] == 0 and cal["0.8-0.9"]["value"] is None


def test_latency_and_tokens():
    rows = [rec(str(i), "X", "X", latency_ms=float(v), tokens_in=100, tokens_out=10, cached_tokens=None)
            for i, v in enumerate([100, 200, 300, 400, 1000])]
    rows.append(rec("r", "X", None, status="rate_limit_miss", latency_ms=99999.0, tokens_in=5, tokens_out=5))
    lat = m.latency_percentiles(rows)
    assert (lat["n"], lat["p50"], lat["p95"]) == (5, 300.0, 1000.0)
    tok = m.tokens_per_call(rows)
    assert tok["mean_in"] == 100 and tok["mean_out"] == 10 and tok["mean_cached"] is None
    rows[0]["cached_tokens"] = 50
    assert m.tokens_per_call(rows)["mean_cached"] == 50


def test_breakdowns():
    rows = [rec("a", "X", "X", category="ctxY"), rec("b", "X", "Y", category="ctxY"), rec("c", "X", "X", category="ctxN")]
    b = m.breakdown(rows, "category")
    assert (b["ctxY"]["k"], b["ctxY"]["n"], b["ctxN"]["k"]) == (1, 2, 1)


def base(uid, gold, pred, acceptable=()):
    return {"uid": uid, "gold": gold, "pred": pred, "acceptable": list(acceptable)}


def test_paired_comparison():
    new = [rec("u1", "NEW_SEARCH", "NEW_SEARCH"), rec("u2", "NEW_SEARCH", "NEW_SEARCH"),
           rec("u3", "NEW_SEARCH", "UNCLEAR"), rec("u4", "NEW_SEARCH", "UNCLEAR"), rec("u5", "NEW_SEARCH", "NEW_SEARCH")]
    old = [base("u1", "NEW_SEARCH", "NEW_SEARCH"), base("u2", "NEW_SEARCH", "OUT_OF_SCOPE"),
           base("u3", "NEW_SEARCH", "NEW_SEARCH"), base("u4", "NEW_SEARCH", "UNCLEAR")]  # u5 not in baseline
    p = m.paired_comparison(new, old)
    assert (p["n"], p["both_right"], p["new_right_old_wrong"], p["new_wrong_old_right"], p["both_wrong"]) == (4, 1, 1, 1, 1)
    assert p["new"]["k"] == 2 and p["old"]["k"] == 2


def test_paired_old_pred_mapped_to_intent_and_unmapped_is_wrong():
    new = [rec("u1", "GENERAL_CHAT", "GENERAL_CHAT")]
    old = [base("u1", "GREETING", "ACKNOWLEDGMENT")]  # different label, same intent
    assert m.paired_comparison(new, old)["both_right"] == 1
    unmapped = [base("u1", "GREETING", None)]
    assert m.paired_comparison(new, unmapped)["new_right_old_wrong"] == 1


def test_paired_lenient_uses_old_acceptable_labels():
    new = [rec("u1", "UNCLEAR", "UNCLEAR", acc=["OUT_OF_SCOPE"])]
    old = [base("u1", "UNCLEAR", "OUT_OF_SCOPE", acceptable=["OUT_OF_SCOPE"])]
    assert m.paired_comparison(new, old)["new_right_old_wrong"] == 1
    assert m.paired_comparison(new, old, lenient=True)["both_right"] == 1


def test_select_run_takes_first_scored_run():
    rows = [rec("a", "X", None, status="rate_limit_miss", run=1), rec("a", "X", "X", run=2), rec("a", "X", "Y", run=3)]
    chosen = m.select_run(rows)
    assert len(chosen) == 1 and chosen[0]["run"] == 2
    assert m.select_run(rows, run=3)[0]["intent"] == "Y"
