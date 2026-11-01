"""Offline measure of the semantic input gate on the existing test messages. Local embedding model only.

No Groq, Gemini or rooms API call. The gate is run exactly as ``semantic_gate.evaluate`` does: the copied
``classify_user_intent`` on ``normalize_text(text)``, and a would-block when the label is not in
``semantic_gate.ALLOWED_LABELS``.

    python -m llm_approach.scripts.gate_shadow_measure     # writes eval/results/gate_shadow.json

View (a): every message, as if the gate ran with no context. View (b): only messages where the gate runs in
production (no rooms on screen; the datasets do not record a pending question, so that half of the rule cannot
be applied and is noted in the output). The text of a would-block is printed (synthetic test data); the JSON
file holds counts and ids only.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_approach import semantic_gate  # noqa: E402
from llm_approach.eval import datasets, run_safety  # noqa: E402
from llm_approach.guardrails.common import normalize_text  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "eval" / "results" / "gate_shadow.json"
CORRECT_BLOCK_INTENTS = frozenset({"OUT_OF_SCOPE"})


def gate_runs(item: dict[str, Any]) -> bool:
    """The production rule that the data can express: no rooms on screen (a pending question is not recorded)."""
    return not item["ctx"]


def collect_items() -> list[dict[str, Any]]:
    """Every item of the five datasets, with a ``general`` flag for the 144 messages of run_safety."""
    general = {it["item"] for name in run_safety.GENERAL_DATASETS for it in datasets.load_dataset(name)
               if it["item"] not in run_safety.SAFETY_DUPLICATES}
    return [dict(it, general=it["item"] in general) for name in datasets.DATASETS for it in datasets.load_dataset(name)]


def is_correct_block(item: dict[str, Any]) -> bool:
    """A refusal is right for an out-of-scope gold intent or one of the safety strings kept in the fresh set."""
    return item["gold_intent"] in CORRECT_BLOCK_INTENTS or item["item"] in run_safety.SAFETY_DUPLICATES


def classify_all(items: list[dict[str, Any]], classifier: Callable[[str], str]) -> list[dict[str, Any]]:
    """One row per item with the label and would-block; the classifier runs once per distinct text."""
    cache: dict[str, str] = {}
    rows = []
    for it in items:
        text = it["text"]
        if text not in cache:
            cache[text] = classifier(normalize_text(text))
        label = cache[text]
        rows.append({"item": it["item"], "dataset": it["dataset"], "gold_intent": it["gold_intent"], "text": text,
                     "general": it["general"], "runs": gate_runs(it), "label": label,
                     "would_block": label not in semantic_gate.ALLOWED_LABELS,
                     "correct_block": is_correct_block(it)})
    return rows


def _view(rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocks = [r for r in rows if r["would_block"]]
    fps = [r for r in blocks if not r["correct_block"]]
    by_intent: dict[str, list[str]] = defaultdict(list)
    for r in fps:
        by_intent[str(r["gold_intent"])].append(r["item"])
    return {"n": len(rows), "would_block": len(blocks), "correct_blocks": len(blocks) - len(fps),
            "false_positives": len(fps), "false_positive_ids_by_intent": dict(sorted(by_intent.items())),
            "would_block_ids": [r["item"] for r in blocks]}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    running = [r for r in rows if r["runs"]]
    general = [r for r in rows if r["general"]]
    return {"a_every_message": _view(rows), "b_gate_runs_only": _view(running),
            "general_144_every_message": _view(general),
            "general_144_gate_runs_only": _view([r for r in general if r["runs"]]),
            "note": "pending question is not recorded in the datasets, so view (b) is only the no-rooms-on-screen half of the rule"}


def print_blocks(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n== {title} ==")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["would_block"]:
            groups[str(r["gold_intent"])].append(r)
    for intent in sorted(groups):
        print(f"-- gold {intent} ({len(groups[intent])})")
        for r in groups[intent]:
            kind = "ok-block" if r["correct_block"] else "FALSE POSITIVE"
            print(f"  [{kind}] {r['item']} ({r['dataset']}, label {r['label']}, runs={r['runs']}): {r['text']}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from llm_approach import reuse

    print("Loading the embedding model...", flush=True)
    classifier = reuse.get_semantic_classifier()
    classifier("hello")
    rows = classify_all(collect_items(), classifier)
    print_blocks(rows, "view (a): every message")
    print_blocks([r for r in rows if r["runs"]], "view (b): gate runs in production (no rooms on screen)")
    summary = summarise(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for key, v in summary.items():
        if isinstance(v, dict):
            fp = {k: len(x) for k, x in v["false_positive_ids_by_intent"].items()}
            print(f"{key}: n={v['n']} would_block={v['would_block']} correct={v['correct_blocks']} "
                  f"false_positives={v['false_positives']} {fp}")
    print(f"Saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
