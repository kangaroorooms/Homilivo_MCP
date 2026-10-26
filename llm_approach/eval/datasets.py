"""Loads the existing evaluation sets, read-only, and builds the prompt state.

Nothing here writes, edits or relabels a dataset. Each item is a plain dict:

    dataset, item (unique id), text, gold_label (POC label), gold_intent,
    gold_operation, acceptable (POC labels), acceptable_intents, acceptable_ops,
    ctx (rooms on screen), category, gold_slots (golden_slots only), caveat

``item`` equals the ``uid`` used in ``results/baseline_current.json`` for the
four intent sets, so a result can be paired with the old router per item.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_approach.eval import label_map

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "poc_classifier" / "data"
GOLDEN_SET = ROOT / "tests" / "golden_set.jsonl"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rooms_bangalore.json"

#: recommended run order (cheapest signal first, slot set last)
DATASETS = ["ctxY", "matrix55", "fresh", "golden", "golden_slots"]
INTENT_DATASETS = ["ctxY", "matrix55", "fresh", "golden"]
SLOT_DATASET = "golden_slots"

_FILES = {"ctxY": "test_newsearch_ctxY.jsonl", "matrix55": "test_matrix55.jsonl",
          "fresh": "test_fresh.jsonl", "golden": "test_golden.jsonl"}

CAVEAT_ROOMS = "gold written against the old harness's rooms; fixture rooms used"
CAVEAT_PENDING = "short reply: gold may assume a pending bot question or selected room that the data does not record"


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _caveat(ctx: bool, gold_intent: str | None, text: str) -> str | None:
    if not ctx:
        return None
    notes = []
    if gold_intent in {"ROOM_FOLLOWUP", "REJECT_CURRENT", "SHOW_MORE"}:
        notes.append(CAVEAT_ROOMS)
    if gold_intent in {"ROOM_FOLLOWUP", "REFINEMENT", "UNCLEAR", "GENERAL_CHAT"} and len(text.split()) <= 3:
        notes.append(CAVEAT_PENDING)
    return "; ".join(notes) or None


def _item(dataset: str, item_id: str, row: dict[str, Any], ctx: bool) -> dict[str, Any]:
    label = row["label"]
    intent, op = label_map.map_label(label) or (None, None)
    acc = list(row.get("acceptable") or [])
    acc_ops = {o for o in (label_map.map_label(a) and label_map.map_label(a)[1] for a in acc) if o}
    return {
        "dataset": dataset, "item": item_id, "text": row["text"], "gold_label": label,
        "gold_intent": intent, "gold_operation": op, "acceptable": acc,
        "acceptable_intents": sorted(label_map.acceptable_intents(acc)), "acceptable_ops": sorted(acc_ops),
        "ctx": ctx, "category": "ctxY" if ctx else "ctxN", "gold_slots": None,
        "caveat": _caveat(ctx, intent, row["text"]),
    }


def _golden_slots_items() -> list[dict[str, Any]]:
    items = []
    for n, row in enumerate(_read(GOLDEN_SET), 1):
        args = row.get("expected_args") or {}
        types = args.get("room_types") or []
        bhk = next((int(t[0]) for t in types if len(t) >= 4 and t[0].isdigit() and t.upper().endswith("BHK")), None)
        ideal = (args.get("ideal_for") or [None])[0]
        searching = row.get("expected_tool") == "search_rooms"
        items.append({
            "dataset": SLOT_DATASET, "item": f"golden_slots:{n}", "text": row["query"],
            "gold_label": None, "gold_intent": "NEW_SEARCH" if searching else None, "gold_operation": None,
            "acceptable": [], "acceptable_intents": [], "acceptable_ops": [],
            "ctx": False, "category": row.get("difficulty", "?"),
            "gold_slots": {"city": args.get("city"), "bhk": bhk, "min_price": args.get("min_price"),
                           "max_price": args.get("max_price"), "ideal_for": ideal} if searching else None,
            "caveat": None,
        })
    return items


def load_dataset(name: str) -> list[dict[str, Any]]:
    """Items of one dataset in file order. Unscored matrix55 rows are skipped (as in the baseline)."""
    if name == SLOT_DATASET:
        return _golden_slots_items()
    if name not in _FILES:
        raise KeyError(f"unknown dataset {name!r}; choose from {DATASETS}")
    rows = _read(DATA_DIR / _FILES[name])
    items = []
    for n, row in enumerate(rows, 1):
        if name == "matrix55":
            if not row.get("score"):
                continue
            item_id, ctx = f"matrix55:{row['id']}", bool(row["has_room_context"])
        elif name == "fresh":
            item_id, ctx = f"fresh:f{row['id']}", bool(row["has_room_context"])
        elif name == "ctxY":
            item_id, ctx = f"twins:{row['id']}", True
        else:
            item_id, ctx = f"golden:g{n}", bool(row["has_room_context"])
        items.append(_item(name, item_id, row, ctx))
    return items


def load_rooms() -> list[dict[str, Any]]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["rooms"] if isinstance(data, dict) else data


def build_state(ctx: bool) -> dict[str, Any]:
    """Prompt-state summary for ``llm_client.decide``.

    ctx=False: empty conversation. ctx=True: a finished Bangalore search with the
    fixture rooms on screen, nothing selected, no pending question, no history.
    A fresh state for every item, so items never leak into each other.
    """
    from llm_approach import prompts
    from llm_approach.state import LLMConversationState

    state = LLMConversationState()
    if ctx:
        state.apply_search_result({"city": "Bangalore"}, load_rooms())
    return prompts.build_state_summary(state.to_prompt_state())


def _norm_text(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9 ]", "", str(text).lower()).strip()


def few_shot_item_ids(items: list[dict[str, Any]] | None = None) -> set[str]:
    """Ids of evaluation items whose text is, after lower-casing and dropping punctuation, exactly one of the user messages in
    the prompt's few-shot examples (``prompts.FEW_SHOT_EXAMPLES``). Nothing is edited: the items and the prompt stay as they are."""
    from llm_approach import prompts

    shots = {_norm_text(message) for _state, message, _decision in prompts.FEW_SHOT_EXAMPLES}
    if items is None:
        items = [item for name in DATASETS for item in load_dataset(name)]
    return {item["item"] for item in items if _norm_text(item["text"]) in shots}
