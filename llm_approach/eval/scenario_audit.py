"""Which scenario turns assert nothing, and what could be asserted instead.

A turn whose ``expect`` is empty can only fail through the generic "LLM never called" check, so it passes whatever the bot does.
This lists those turns by scenario. The proposals come from the stub baseline (``tests/stub_scenarios_baseline.json``, the
(intent, action) the pipeline produces today for every turn): they describe current behaviour and have NOT been reviewed as
correct. Nothing is written to the scenario files.

    python -m llm_approach.eval.scenario_audit
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
BASELINE = ROOT / "tests" / "stub_scenarios_baseline.json"


def turn_ranges(numbers: list[int]) -> str:
    """[1, 2, 3, 7] -> "1-3, 7"."""
    out, start = [], None
    for i, n in enumerate(numbers):
        if start is None:
            start = n
        if i + 1 == len(numbers) or numbers[i + 1] != n + 1:
            out.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return ", ".join(out)


def audit() -> dict[str, Any]:
    try:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        baseline = {}
    scenarios = [json.loads(p.read_text(encoding="utf-8")) for p in SCENARIOS.glob("S*.json")]
    scenarios.sort(key=lambda s: int(s["id"][1:]))
    total = empty = live_empty = never_only = 0
    by_scenario: dict[str, dict[str, Any]] = {}
    for sc in scenarios:
        for index, turn in enumerate(sc["turns"], 1):
            total += 1
            if turn.get("expect"):
                continue
            if turn.get("llm") == "never":
                never_only += 1  # only the "LLM never called" check
                continue
            empty += 1
            live_empty += not sc.get("stub_only")
            pair = (baseline.get(sc["id"]) or [[None, None]] * index)[index - 1]
            entry = by_scenario.setdefault(sc["id"], {"title": sc.get("title", ""), "stub_only": bool(sc.get("stub_only")), "turns": []})
            entry["turns"].append({"turn": index, "user": turn["user"], "proposed": {"intent": pair[0], "action": pair[1]}})
    return {"total_turns": total, "assert_nothing": empty, "assert_nothing_in_live_scenarios": live_empty,
            "only_the_never_called_check": never_only, "by_scenario": by_scenario}


def proposal_text(entry: dict[str, Any]) -> str:
    """"1-3 (GENERAL_CHAT/general_chat), 4 (NEW_SEARCH/new_search)": consecutive turns with the same proposal are one range."""
    groups: list[tuple[str, list[int]]] = []
    for t in entry["turns"]:
        key = f"{t['proposed']['intent']}/{t['proposed']['action']}"
        if groups and groups[-1][0] == key:
            groups[-1][1].append(t["turn"])
        else:
            groups.append((key, [t["turn"]]))
    return ", ".join(f"{turn_ranges(nums)} ({key})" for key, nums in groups)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = audit()
    print(f"{data['assert_nothing']} of {data['total_turns']} turns assert nothing "
          f"({data['assert_nothing_in_live_scenarios']} of them in scenarios that run live).")
    for sid, entry in data["by_scenario"].items():
        print(f"{sid} {entry['title']}{' [stub only]' if entry['stub_only'] else ''}: turns {proposal_text(entry)}")
    print("Proposed expectations are today's stub-baseline (intent, action); they are not reviewed as correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
