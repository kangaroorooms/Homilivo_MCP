"""
Standalone golden-set runner for intent + argument extraction
(tests/golden_set.jsonl).

Unlike tests/test_golden_eval.py (pytest, hard-gated on calibrated
thresholds, uses its own inline case data), this script is meant to be
run directly and read by a human: it feeds every query in the jsonl
file through the SAME deterministic pipeline the real /chat endpoint
uses for a fresh turn - app.experiments.agent_modes.route_message()
for tool selection, app.fast_parser.extract_search_parameters() for
arguments when a search is selected - and prints a pass/fail table
plus an accuracy score. No Ollama/MCP/network access is required.

route_message() is the DETERMINISTIC router only. A query that fails
here is a real gap in the fast, no-LLM path; it does not necessarily
mean the live system with mode=qwen/gemma/groq (or fast mode's own
Gemma-classifier fallback inside process_message()) also gets it
wrong - see tests/test_golden_eval.py's
test_messy_search_detection_end_to_end_* for that additional layer.

Run from the project root:
    python tests/run_golden_eval.py
    python tests/run_golden_eval.py --verbose   (show every case, not just failures)

Exit code is 0 if overall accuracy is >= 70%, else 1 (usable in CI).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.experiments.agent_modes import (  # noqa: E402
    ConversationState,
    MESSAGE_NEW_SEARCH,
    MESSAGE_REFINEMENT,
    route_message,
)
from app.fast_parser import extract_search_parameters  # noqa: E402

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.jsonl"

ACCURACY_GATE = 0.70

# Keys extract_search_parameters() always returns even when the query
# said nothing about them - excluded from both "actual" args (before
# comparison) and printed output so pass/fail reflects only what the
# query actually specified.
_ALWAYS_PRESENT_DEFAULTS = {"sort": "latest", "limit": 10}


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"{path}:{line_number}: invalid JSON - {error}"
                )
    return cases


def _strip_empty(args: dict[str, Any]) -> dict[str, Any]:
    """Drop None/[]/default values so the printed diff only shows what matters."""
    result = {}
    for key, value in args.items():
        if key in _ALWAYS_PRESENT_DEFAULTS:
            continue
        if value is None or value == []:
            continue
        result[key] = value
    return result


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    query = case["query"]
    expected_tool = case.get("expected_tool")
    expected_args = case.get("expected_args", {})

    route = route_message(query, ConversationState())

    if route.message_type in (MESSAGE_NEW_SEARCH, MESSAGE_REFINEMENT):
        actual_tool = "search_rooms"
        actual_args = _strip_empty(extract_search_parameters(query))
    else:
        actual_tool = None
        actual_args = {}

    tool_ok = actual_tool == expected_tool
    args_ok = all(
        actual_args.get(key) == value for key, value in expected_args.items()
    )
    passed = tool_ok and args_ok

    return {
        "query": query,
        "difficulty": case.get("difficulty", "n/a"),
        "note": case.get("note", ""),
        "route": route.message_type,
        "expected_tool": expected_tool,
        "actual_tool": actual_tool,
        "expected_args": expected_args,
        "actual_args": actual_args,
        "tool_ok": tool_ok,
        "args_ok": args_ok,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", action="store_true", help="print every case, not just failures"
    )
    args = parser.parse_args()

    cases = load_cases(GOLDEN_SET_PATH)
    results = [run_case(case) for case in cases]

    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]

    by_difficulty: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_difficulty.setdefault(r["difficulty"], []).append(r)

    print("=" * 78)
    print("GOLDEN SET EVAL - intent + argument extraction (deterministic pipeline)")
    print("=" * 78)

    for r in results:
        if not args.verbose and r["passed"]:
            continue
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{status}] ({r['difficulty']}) {r['query']!r}")
        print(f"    route={r['route']}")
        print(f"    expected_tool={r['expected_tool']!r}  actual_tool={r['actual_tool']!r}")
        print(f"    expected_args={r['expected_args']}")
        print(f"    actual_args  ={r['actual_args']}")
        if r["note"]:
            print(f"    note: {r['note']}")

    print("\n" + "-" * 78)
    print("SUMMARY")
    print("-" * 78)
    overall = len(passed) / len(results) if results else 0.0
    print(f"Overall accuracy: {overall:.0%} ({len(passed)}/{len(results)})")
    for difficulty, group in by_difficulty.items():
        group_passed = sum(1 for r in group if r["passed"])
        print(f"  {difficulty:8s}: {group_passed}/{len(group)}")

    if failed:
        print(f"\nFailures ({len(failed)}):")
        for r in failed:
            reason = []
            if not r["tool_ok"]:
                reason.append(f"tool {r['actual_tool']!r} != expected {r['expected_tool']!r}")
            if not r["args_ok"]:
                reason.append("args mismatch")
            print(f"  - {r['query']!r}: {', '.join(reason)}")

    print()

    if overall < ACCURACY_GATE:
        print(f"FAILED: accuracy {overall:.0%} is below the {ACCURACY_GATE:.0%} gate.")
        return 1

    print(f"OK: accuracy {overall:.0%} meets the {ACCURACY_GATE:.0%} gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
