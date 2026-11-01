"""Counting logic of scripts/gate_shadow_measure.py with a fake classifier (no model)."""

from __future__ import annotations

from llm_approach.scripts import gate_shadow_measure as g


def _item(item, intent, ctx=False, general=False, text="x"):
    return {"item": item, "dataset": "d", "gold_intent": intent, "ctx": ctx, "general": general, "text": text}


def test_counts_views_and_false_positives():
    items = [
        _item("a", "NEW_SEARCH", text="find room"),            # allowed
        _item("b", "ROOM_FOLLOWUP", ctx=True, text="wifi?"),   # blocked, false positive, gate would not run
        _item("c", "OUT_OF_SCOPE", text="joke"),               # blocked, correct
        _item("d", "NEW_SEARCH", general=True, text="cheap"),  # blocked, false positive, gate runs
        _item("fresh:f45", "GENERAL_CHAT", text="secret"),     # blocked, safety string: correct
    ]
    labels = {"find room": "ROOM_SEARCH", "wifi?": "OUT_OF_SCOPE", "joke": "OUT_OF_SCOPE",
              "cheap": "UNSUPPORTED_ACTION", "secret": "OUT_OF_SCOPE"}

    rows = g.classify_all(items, lambda text: labels[text])
    s = g.summarise(rows)
    a, b = s["a_every_message"], s["b_gate_runs_only"]
    assert (a["n"], a["would_block"], a["correct_blocks"], a["false_positives"]) == (5, 4, 2, 2)
    assert a["false_positive_ids_by_intent"] == {"NEW_SEARCH": ["d"], "ROOM_FOLLOWUP": ["b"]}
    assert (b["n"], b["would_block"], b["false_positives"]) == (4, 3, 1)
    assert "b" not in b["would_block_ids"]
    assert s["general_144_every_message"]["n"] == 1


def test_classifier_runs_once_per_distinct_text():
    calls = []
    rows = g.classify_all([_item("a", "X", text="same"), _item("b", "X", text="same")],
                          lambda t: calls.append(t) or "GREETING")
    assert len(calls) == 1 and not any(r["would_block"] for r in rows)
