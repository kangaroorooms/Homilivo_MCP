"""Batch planner: how many Gemini requests and minutes each live run needs, and a day-by-day order that
keeps every day under the request budget. Makes no call.

    python -m llm_approach.eval.plan

"Worst-case" is the planner's per-prompt assumption (``runs.WORST_CASE_CALLS_PER_PROMPT`` = 3: two decision calls
plus one rate-limit retry; scenario turns 5, because a chat question adds the answerer's two calls). The daily
counter (``LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET``, default 450 of the free tier's 500) enforces the cap
whatever the real use turns out to be. The first day is sized by what is still left today.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_approach import config  # noqa: E402
from llm_approach.eval import datasets, runs  # noqa: E402

#: order in which runs are suggested (cheap signal first, the long scenario run last)
ORDER = ["ctxY", "golden", "golden_slots", "matrix55", "fresh", "scenarios"]


def scenario_turns() -> int:
    """LLM turns of the live scenarios (stub-only scenarios and turns that never reach the LLM excluded)."""
    import json

    total = 0
    for path in (Path(__file__).resolve().parents[1] / "scenarios").glob("S*.json"):
        sc = json.loads(path.read_text(encoding="utf-8"))
        if not sc.get("stub_only"):
            total += sum(1 for t in sc["turns"] if t["llm"] != "never")
    return total


def rows(spacing: float) -> list[dict[str, Any]]:
    """One row per run: prompts, typical requests, worst-case requests, worst-case minutes."""
    out = []
    for name in ORDER:
        if name == "scenarios":
            prompts, per = scenario_turns(), runs.WORST_CASE_CALLS_PER_SCENARIO_TURN
        else:
            prompts, per = len(datasets.load_dataset(name)), runs.WORST_CASE_CALLS_PER_PROMPT
        out.append({"name": name, "prompts": prompts, "typical": prompts, "worst": prompts * per,
                    "minutes": prompts * per * spacing / 60})
    return out


def schedule(items: list[dict[str, Any]], first_day: int, budget: int) -> list[list[tuple[str, int]]]:
    """Pack the runs, in order, into days of at most ``budget`` worst-case requests (the first day: ``first_day``,
    possibly empty). A run bigger than a whole day is split across days (every live run can be resumed).
    Returns, per day, (label, requests) pairs."""
    days: list[list[tuple[str, int]]] = [[]]
    room = max(0, first_day)
    for item in items:
        left, parts = item["worst"], 0
        if left <= budget:  # a run that fits in one day is never split
            if left > room:
                days.append([])
                room = budget
            days[-1].append((item["name"], left))
            room -= left
            continue
        while left > 0:  # bigger than a whole day: split
            if room <= 0:
                days.append([])
                room = budget
            take, parts = min(left, room), parts + 1
            days[-1].append((f"{item['name']} (part {parts})", take))
            room, left = room - take, left - take
    return days


def build_plan(spacing: float, used: int, budget: int) -> list[str]:
    items = rows(spacing)
    remaining = max(0, budget - used)
    lines = [
        f"Gemini free tier: 15 requests/min, 250K tokens/min, 500 requests/day. Planner budget: {budget} requests/day, "
        f"spacing {spacing:g} s. Requests used today: {used}; remaining today: {remaining}.",
        "",
        f"{'run':<14}{'prompts':>8}{'typical req':>13}{'worst-case req':>16}{'worst-case min':>16}",
    ]
    for r in items:
        lines.append(f"{r['name']:<14}{r['prompts']:>8}{r['typical']:>13}{r['worst']:>16}{r['minutes']:>16.1f}")
    total = sum(r["worst"] for r in items)
    lines.append(f"{'total':<14}{sum(r['prompts'] for r in items):>8}{sum(r['typical'] for r in items):>13}"
                 f"{total:>16}{sum(r['minutes'] for r in items):>16.1f}")
    lines.append("")
    days = schedule(items, remaining, budget)
    if len(days) == 1 and days[0]:
        lines.append(f"Everything fits in one day: {total} worst-case requests, {remaining} left today.")
        lines.append("Suggested order: " + " + ".join(label for label, _ in days[0]))
        return lines
    lines.append(f"Suggested order, each day within {budget} worst-case requests "
                 f"(today has {remaining} left):")
    for n, day in enumerate(days, 1):
        when = "today" if n == 1 else ("next day" if n == 2 else f"day {n}")
        what = " + ".join(label for label, _ in day) or "nothing fits in what is left"
        lines.append(f"  {when}: {what}  ({sum(c for _, c in day)} requests)")
    lines.append(f"{len(days)} day(s) in total. Each run stops cleanly at the cap and continues with --resume.")
    return lines


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except Exception:  # noqa: BLE001 - python-dotenv missing or .env unreadable: use the environment
        pass
    from llm_approach import gemini_limits

    used = gemini_limits.get_limiter().counter.used_today()
    print("\n".join(build_plan(config.spacing_s("gemini"), used, config.daily_request_budget())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
