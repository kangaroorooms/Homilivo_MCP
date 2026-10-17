"""Live evaluation runner for the LLM decision layer. Run by hand, never by pytest.

One Groq call per item (``llm_client.decide``), the state built from the dataset's
context flag (see ``datasets.build_state``). Results append to
``eval/results/<dataset>.jsonl`` and a finished (dataset, item, run) is skipped on the
next start, so a run can be stopped and resumed on another day.

    python -m llm_approach.eval.run_eval --dry-run --dataset all
    python -m llm_approach.eval.run_eval --dataset ctxY --spacing 30
    python -m llm_approach.eval.run_eval --dataset all --sample 30 --runs 3

Order for ``--dataset all``: ctxY, matrix55, fresh, golden, golden_slots.
Groq free tier: 8,000 tokens per minute and 200,000 per day; one call is about
4,000 tokens, hence ``--spacing 30`` and a default ``--daily-token-budget`` of 150,000.

Nothing secret is printed or stored: no key, no message text, no model reasoning.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_approach.eval import datasets, metrics  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
USAGE_FILE = "usage.jsonl"  # one line per day-run: {"date", "tokens"}; ignored by git
DEFAULT_SPACING = 30.0
DEFAULT_DAILY_BUDGET = 150_000
RATE_LIMIT_WAIT_S = 60.0
SAMPLE_SEED = 1234
EST_COMPLETION_TOKENS = 250  # low-effort reasoning plus the JSON answer

DecideFn = Callable[[str, dict, Any], Awaitable[Any]]
SleepFn = Callable[[float], Awaitable[None]]


# ---- files ---------------------------------------------------------------------


def result_path(dataset: str, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / f"{dataset}.jsonl"


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written last line after a crash
    return out


def load_done(path: Path, *, only_missing: bool = False) -> set[tuple[str, str, int]]:
    """(dataset, item, run) keys to skip. By default any record counts, a rate-limit
    miss included; with ``only_missing`` a rate-limit miss does not, so it is retried."""
    done = set()
    for r in read_records(path):
        if only_missing and r.get("status") == "rate_limit_miss":
            continue
        done.add((r["dataset"], r["item"], int(r.get("run", 1))))
    return done


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        f.flush()


def tokens_used_today(results_dir: Path, today: str) -> int:
    return sum(int(r.get("tokens", 0)) for r in read_records(results_dir / USAGE_FILE) if r.get("date") == today)


def add_usage(results_dir: Path, today: str, tokens: int) -> None:
    if tokens > 0:
        append_record(results_dir / USAGE_FILE, {"date": today, "tokens": int(tokens)})


# ---- planning --------------------------------------------------------------------


def select_items(name: str, sample: int | None, seed: int = SAMPLE_SEED) -> list[dict]:
    """Items of a dataset, or a fixed-seed random sample of ``sample`` items in file order."""
    items = datasets.load_dataset(name)
    if sample is not None and 0 < sample < len(items):
        keep = set(random.Random(f"{seed}:{name}").sample(range(len(items)), sample))
        items = [it for i, it in enumerate(items) if i in keep]
    return items


def plan_items(
    names: list[str], runs: int, sample: int | None, results_dir: Path = RESULTS_DIR, only_missing: bool = False
) -> list[tuple[dict, int]]:
    """(item, run) pairs still to do, in run order: dataset, then run, then item."""
    todo = []
    for name in names:
        done = load_done(result_path(name, results_dir), only_missing=only_missing)
        items = select_items(name, sample)
        for run in range(1, runs + 1):
            todo += [(it, run) for it in items if (name, it["item"], run) not in done]
    return todo


def estimate_tokens(item: dict, system_chars: int | None = None) -> int:
    """Rough tokens for one call from the prompt size (about 4 characters per token)."""
    if system_chars is None:
        from llm_approach import prompts

        system_chars = len(prompts.system_prompt())
    state_chars = len(json.dumps(datasets.build_state(item["ctx"]), ensure_ascii=False))
    return (system_chars + state_chars + len(item["text"])) // 4 + EST_COMPLETION_TOKENS


def estimate(todo: list[tuple[dict, int]], spacing: float, budget: int) -> dict[str, Any]:
    from llm_approach import prompts

    system_chars = len(prompts.system_prompt())
    per_dataset: dict[str, dict[str, int]] = {}
    for item, _run in todo:
        d = per_dataset.setdefault(item["dataset"], {"calls": 0, "tokens": 0})
        d["calls"] += 1
        d["tokens"] += estimate_tokens(item, system_chars)
    calls = sum(d["calls"] for d in per_dataset.values())
    tokens = sum(d["tokens"] for d in per_dataset.values())
    return {"per_dataset": per_dataset, "calls": calls, "tokens": tokens,
            "minutes": calls * spacing / 60, "days": (tokens / budget) if budget else None}


# ---- one item --------------------------------------------------------------------


def build_record(item: dict, run: int, decision: Any, budget: Any, status: str, rooms_count: int) -> dict[str, Any]:
    from llm_approach import validators

    rec: dict[str, Any] = {
        "dataset": item["dataset"], "item": item["item"], "run": run, "status": status,
        "category": item["category"], "gold_label": item["gold_label"], "gold_intent": item["gold_intent"],
        "gold_operation": item["gold_operation"], "gold_slots": item["gold_slots"],
        "acceptable_intents": item["acceptable_intents"], "acceptable_ops": item["acceptable_ops"],
        "ctx": item["ctx"], "caveat": item["caveat"],
        "intent": None, "operation": None, "slots": None, "validated_slots": None, "rejected": [],
        "room_reference": None, "confidence": None, "needs_clarification": None, "slots_valid": None,
        "would_search": False, "error_class": getattr(budget, "error_class", None),
        "latency_ms": round(getattr(budget, "latency_ms", 0.0), 1), "tokens_in": getattr(budget, "tokens_in", 0),
        "tokens_out": getattr(budget, "tokens_out", 0), "cached_tokens": getattr(budget, "cached_tokens", None),
        "llm_calls": getattr(budget, "calls", 0),
    }
    if decision is not None:
        clean, rejected = validators.validate_slots(
            decision.slots, rooms_count=rooms_count, has_selected=False,
            room_reference=decision.room_reference, compare_references=decision.compare_references,
        )
        search_rejected = bool(set(rejected) & set(metrics.SEARCH_SLOT_FIELDS))
        rec.update(
            intent=decision.intent.value, operation=decision.operation.value if decision.operation else None,
            slots=decision.slots.model_dump(), validated_slots={k: v for k, v in clean.items() if v not in (None, [])},
            rejected=rejected, room_reference=decision.room_reference, confidence=decision.confidence,
            needs_clarification=decision.needs_clarification, slots_valid=not rejected,
            # the pipeline's rule: a rejected city/BHK/price on a search intent asks instead of searching
            would_search=decision.intent.value in metrics.SEARCH_INTENTS and not search_rejected,
        )
    return rec


async def evaluate_item(
    item: dict, run: int, decide: DecideFn, sleep: SleepFn, *, rate_limit_wait: float = RATE_LIMIT_WAIT_S
) -> dict[str, Any]:
    """One decision call. A rate-limit error waits ``rate_limit_wait`` once and retries once;
    a second one is recorded as ``rate_limit_miss`` (never as a wrong answer)."""
    from llm_approach import llm_client

    summary = datasets.build_state(item["ctx"])
    rooms_count = len(summary.get("rooms_on_screen") or [])
    tokens_in = tokens_out = 0
    for attempt in (1, 2):
        budget = llm_client.LLMBudget()
        decision = await decide(item["text"], summary, budget)
        tokens_in += budget.tokens_in
        tokens_out += budget.tokens_out
        if decision is None and budget.error_class == "RateLimitError":
            if attempt == 1:
                await sleep(rate_limit_wait)
                continue
            status = "rate_limit_miss"
        else:
            status = "ok" if decision is not None else "invalid"
        break
    budget.tokens_in, budget.tokens_out = tokens_in, tokens_out
    return build_record(item, run, decision, budget, status, rooms_count)


# ---- the loop ----------------------------------------------------------------------


async def run_eval(
    names: list[str], *, runs: int = 1, sample: int | None = None, spacing: float = DEFAULT_SPACING,
    daily_budget: int = DEFAULT_DAILY_BUDGET, only_missing: bool = False, results_dir: Path = RESULTS_DIR,
    decide: DecideFn | None = None, sleep: SleepFn = asyncio.sleep, today: str | None = None,
    rate_limit_wait: float = RATE_LIMIT_WAIT_S, out: Callable[[str], None] = print,
) -> dict[str, Any]:
    from llm_approach import llm_client

    decide = decide or llm_client.decide
    today = today or dt.date.today().isoformat()
    todo = plan_items(names, runs, sample, results_dir, only_missing)
    used = tokens_used_today(results_dir, today)
    summary = {"planned": len(todo), "calls": 0, "tokens": 0, "rate_limit_misses": 0, "stopped_by_budget": False,
               "completed": 0, "tokens_today": used}
    from llm_approach import prompts

    system_chars = len(prompts.system_prompt())
    totals = {n: sum(1 for it, _ in todo if it["dataset"] == n) for n in names}
    seen: dict[str, int] = {n: 0 for n in names}
    first_call = True
    out(f"{len(todo)} item-runs to do; tokens used today so far: {used}; budget {daily_budget}")

    for item, run in todo:
        need = estimate_tokens(item, system_chars)
        if used + need > daily_budget:
            summary["stopped_by_budget"] = True
            break
        if not first_call and spacing > 0:
            await sleep(spacing)
        first_call = False
        record = await evaluate_item(item, run, decide, sleep, rate_limit_wait=rate_limit_wait)
        append_record(result_path(item["dataset"], results_dir), record)
        spent = record["tokens_in"] + record["tokens_out"]
        used += spent
        add_usage(results_dir, today, spent)
        summary["calls"] += record["llm_calls"] or 1
        summary["tokens"] += spent
        summary["completed"] += 1
        summary["rate_limit_misses"] += record["status"] == "rate_limit_miss"
        seen[item["dataset"]] += 1
        out(f"[{item['dataset']}] {seen[item['dataset']]}/{totals[item['dataset']]} run {run} {item['item']} "
            f"{record['status']} intent={record['intent']} conf={record['confidence']} "
            f"{record['latency_ms']:.0f}ms {spent}tok (today {used})")
    summary["tokens_today"] = used
    return summary


def resume_command(args: argparse.Namespace) -> str:
    parts = ["python -m llm_approach.eval.run_eval", f"--dataset {args.dataset}", f"--runs {args.runs}"]
    if args.sample:
        parts.append(f"--sample {args.sample}")
    parts += [f"--spacing {args.spacing:g}", f"--daily-token-budget {args.daily_token_budget}"]
    if args.only_missing:
        parts.append("--only-missing")
    return " ".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m llm_approach.eval.run_eval", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default="all", help=f"one of {datasets.DATASETS} or 'all' (default)")
    p.add_argument("--runs", type=int, default=1, help="runs per item (default 1)")
    p.add_argument("--sample", type=int, default=None, help="fixed-seed sample of N items per dataset")
    p.add_argument("--spacing", type=float, default=DEFAULT_SPACING, help="seconds between calls (default 30)")
    p.add_argument("--daily-token-budget", type=int, default=DEFAULT_DAILY_BUDGET,
                   help="stop when today's tokens would exceed this (default 150000)")
    p.add_argument("--dry-run", action="store_true", help="no calls: print items, calls, tokens and time")
    p.add_argument("--only-missing", action="store_true", help="also retry rate-limit misses; skip finished items")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    names = datasets.DATASETS if args.dataset == "all" else [args.dataset]
    for n in names:
        if n not in datasets.DATASETS:
            print(f"Unknown dataset {n!r}. Choose from {datasets.DATASETS} or 'all'.")
            return 2
    if args.runs < 1:
        print("--runs must be at least 1.")
        return 2

    if args.dry_run:
        todo = plan_items(names, args.runs, args.sample, RESULTS_DIR, args.only_missing)
        est = estimate(todo, args.spacing, args.daily_token_budget)
        print("DRY RUN: no calls are made.")
        for name in names:
            d = est["per_dataset"].get(name, {"calls": 0, "tokens": 0})
            print(f"  {name:<13} calls {d['calls']:>4}  est. tokens {d['tokens']:>8}  "
                  f"est. time {d['calls'] * args.spacing / 60:>6.1f} min")
        print(f"  {'total':<13} calls {est['calls']:>4}  est. tokens {est['tokens']:>8}  est. time {est['minutes']:.1f} min")
        print(f"At a budget of {args.daily_token_budget} tokens/day this needs about {est['days']:.1f} day(s). "
              f"Tokens already used today: {tokens_used_today(RESULTS_DIR, dt.date.today().isoformat())}.")
        return 0

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Put it in .env at the repo root.")
        return 2
    summary = asyncio.run(run_eval(names, runs=args.runs, sample=args.sample, spacing=args.spacing,
                                   daily_budget=args.daily_token_budget, only_missing=args.only_missing))
    print(f"\nDone: {summary['completed']} of {summary['planned']} item-runs, {summary['tokens']} tokens this run, "
          f"{summary['rate_limit_misses']} rate-limit misses.")
    if summary["stopped_by_budget"]:
        print("Daily token budget reached. Resume tomorrow with exactly:\n  " + resume_command(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
