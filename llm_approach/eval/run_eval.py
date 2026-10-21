"""Live evaluation runner for the LLM decision layer. Run by hand, never by pytest.

One LLM call per item (``llm_client.decide``, the active provider), the state built from the dataset's
context flag (see ``datasets.build_state``). Results append to
``eval/results/<dataset>.jsonl`` and a finished (dataset, item, run) is skipped on the
next start, so a run can be stopped and resumed on another day.

    python -m llm_approach.eval.run_eval --dry-run --dataset all
    python -m llm_approach.eval.run_eval --dataset ctxY --spacing 30
    python -m llm_approach.eval.run_eval --dataset all --sample 30 --runs 3

Order for ``--dataset all``: ctxY, matrix55, fresh, golden, golden_slots.
Spacing and daily token budget are per provider. Groq free tier: 8,000 tokens per minute and
200,000 per day; one call is about 4,000 tokens, hence 30 s and 150,000 (the defaults). Gemini: set
``LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET`` (our own cap; a live run refuses to start without it); spacing is
5 s (``LLM_APPROACH_GEMINI_SPACING_S``, minimum 4 s); ``--spacing`` and ``--daily-token-budget`` override either.
Gemini live runs also print a plan first (prompts x worst-case calls, requests left today) and refuse to start
when the worst case does not fit in the daily request budget (see ``gemini_limits.py``).
Every prompt is checkpointed to its results file. ``--resume`` continues an unfinished run (same provider,
model and dataset only); a run that stops early ends with a ``STOPPED EARLY: X of Y done. Reason: ...
Resume with: <command>`` line.
Every result record carries ``provider`` and ``model``; a run skips only items finished with the same
pair, and ``--dry-run`` is provider-neutral (tokenizers differ between providers).

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

from llm_approach import config  # noqa: E402
from llm_approach.eval import datasets, metrics, runs as runs_mod  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
USAGE_FILE = "usage.jsonl"  # one line per day-run: {"date", "tokens"}; ignored by git
DEFAULT_SPACING = config.GROQ_DEFAULT_SPACING_S  # Groq's values; Gemini's come from the environment
DEFAULT_DAILY_BUDGET = config.GROQ_DEFAULT_DAILY_TOKEN_BUDGET
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


def load_done(
    path: Path, *, only_missing: bool = False, label: tuple[str, str] | None = None
) -> set[tuple[str, str, int]]:
    """(dataset, item, run) keys to skip. By default any record counts, a rate-limit
    miss included; with ``only_missing`` a rate-limit miss does not, so it is retried.
    With ``label`` (provider, model) only records made with that pair count."""
    done = set()
    for r in read_records(path):
        if label is not None and config.record_label(r) != label:
            continue
        if only_missing and r.get("status") in ("rate_limit_miss", "error"):
            continue
        done.add((r["dataset"], r["item"], int(r.get("run", 1))))
    return done


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        f.flush()


def tokens_used_today(results_dir: Path, today: str, label: tuple[str, str] | None = None) -> int:
    """Tokens spent today by one (provider, model) pair; each provider has its own daily budget."""
    label = label or config.label()
    return sum(int(r.get("tokens", 0)) for r in read_records(results_dir / USAGE_FILE)
               if r.get("date") == today and config.record_label(r) == label)


def add_usage(results_dir: Path, today: str, tokens: int, label: tuple[str, str] | None = None) -> None:
    if tokens > 0:
        provider, model = label or config.label()
        append_record(results_dir / USAGE_FILE, {"date": today, "tokens": int(tokens), "provider": provider, "model": model})


# ---- planning --------------------------------------------------------------------


def select_items(name: str, sample: int | None, seed: int = SAMPLE_SEED) -> list[dict]:
    """Items of a dataset, or a fixed-seed random sample of ``sample`` items in file order."""
    items = datasets.load_dataset(name)
    if sample is not None and 0 < sample < len(items):
        keep = set(random.Random(f"{seed}:{name}").sample(range(len(items)), sample))
        items = [it for i, it in enumerate(items) if i in keep]
    return items


def plan_items(
    names: list[str], runs: int, sample: int | None, results_dir: Path = RESULTS_DIR, only_missing: bool = False,
    label: tuple[str, str] | None = None,
) -> list[tuple[dict, int]]:
    """(item, run) pairs still to do for the active provider and model (or ``label``), in run order:
    dataset, then run, then item."""
    todo = []
    label = label or config.label()
    for name in names:
        done = load_done(result_path(name, results_dir), only_missing=only_missing, label=label)
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

    provider, model = config.label()
    rec: dict[str, Any] = {
        "provider": provider, "model": model,
        "dataset": item["dataset"], "item": item["item"], "run": run, "status": status,
        "category": item["category"], "gold_label": item["gold_label"], "gold_intent": item["gold_intent"],
        "gold_operation": item["gold_operation"], "gold_slots": item["gold_slots"],
        "acceptable_intents": item["acceptable_intents"], "acceptable_ops": item["acceptable_ops"],
        "ctx": item["ctx"], "caveat": item["caveat"],
        "intent": None, "operation": None, "slots": None, "validated_slots": None, "rejected": [],
        "room_reference": None, "confidence": None, "needs_clarification": None, "slots_valid": None,
        "would_search": False, "error_class": getattr(budget, "error_class", None),
        "error_code": getattr(budget, "error_code", None), "error_status": getattr(budget, "error_status", None),
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


class RunStop(Exception):
    """The run must stop cleanly now (Gemini daily quota used up, or the local daily request cap)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


async def evaluate_item(
    item: dict, run: int, decide: DecideFn, sleep: SleepFn, *, rate_limit_wait: float = RATE_LIMIT_WAIT_S
) -> dict[str, Any]:
    """One decision call. A rate-limit error waits ``rate_limit_wait`` once and retries once (Gemini already
    did that inside the provider, so it is not repeated); a second one is recorded as ``rate_limit_miss``
    (never as a wrong answer). Any other API/network error is recorded as ``error``. A daily-quota 429 or the
    local request cap raises ``RunStop``: nothing is recorded for the item, which stays "not run"."""
    from llm_approach import llm_client

    summary = datasets.build_state(item["ctx"])
    rooms_count = len(summary.get("rooms_on_screen") or [])
    tokens_in = tokens_out = 0
    for attempt in (1, 2):
        budget = llm_client.LLMBudget()
        decision = await decide(item["text"], summary, budget)
        tokens_in += budget.tokens_in
        tokens_out += budget.tokens_out
        if budget.error_kind == "daily_quota":
            raise RunStop("Gemini says the daily request quota is exhausted (429, not retried)")
        if budget.error_kind == "request_cap":
            raise RunStop(f"the local daily request cap ({config.daily_request_budget()}) is reached")
        if decision is None and budget.error_class == "RateLimitError":
            if attempt == 1 and budget.error_kind is None:
                await sleep(rate_limit_wait)
                continue
            status = "rate_limit_miss"
        elif decision is None and budget.error_class:
            status = "error"  # timeout, network, auth, 5xx: not a wrong answer
        else:
            status = "ok" if decision is not None else "invalid"
        break
    budget.tokens_in, budget.tokens_out = tokens_in, tokens_out
    return build_record(item, run, decision, budget, status, rooms_count)


# ---- the loop ----------------------------------------------------------------------


async def run_eval(
    names: list[str], *, runs: int = 1, sample: int | None = None, spacing: float = DEFAULT_SPACING,
    daily_budget: int | None = None, only_missing: bool = False, results_dir: Path = RESULTS_DIR,
    decide: DecideFn | None = None, sleep: SleepFn = asyncio.sleep, today: str | None = None,
    rate_limit_wait: float = RATE_LIMIT_WAIT_S, out: Callable[[str], None] = print,
    max_failures: int = runs_mod.MAX_CONSECUTIVE_FAILURES, label: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Run the planned items one by one, writing each result before the next call (a checkpoint after every
    prompt). Stops early, cleanly, on: the daily token budget, the local daily request cap, a Gemini
    daily-quota 429, a hard error on the very first call (wrong key, model or quota), or ``max_failures``
    failures in a row. ``summary["stop_reason"]`` says which (None: finished); ``summary["done"]`` and
    ``summary["total"]`` count item-runs over the whole run, earlier sessions included."""
    from llm_approach import llm_client

    decide = decide or llm_client.decide
    if daily_budget is None:
        daily_budget = config.daily_token_budget()
    if daily_budget is None:
        raise ValueError(budget_missing_message())
    today = today or dt.date.today().isoformat()
    todo = plan_items(names, runs, sample, results_dir, only_missing, label)
    total = sum(len(select_items(n, sample)) for n in names) * runs
    used = tokens_used_today(results_dir, today)
    summary = {"planned": len(todo), "calls": 0, "tokens": 0, "rate_limit_misses": 0, "stopped_by_budget": False,
               "completed": 0, "tokens_today": used, "stop_reason": None, "total": total, "done": total - len(todo),
               "errors": 0}
    from llm_approach import prompts

    system_chars = len(prompts.system_prompt())
    totals = {n: sum(1 for it, _ in todo if it["dataset"] == n) for n in names}
    seen: dict[str, int] = {n: 0 for n in names}
    first_call = True
    streak = runs_mod.FailureStreak(max_failures)
    gemini = config.provider() == "gemini"
    out(f"{len(todo)} item-runs to do; tokens used today so far: {used}; budget {daily_budget}")

    def stop(reason: str) -> None:
        summary["stop_reason"] = reason
        out(f"Stopping: {reason}. {len(todo) - summary['completed']} prompts not run.")

    for item, run in todo:
        need = estimate_tokens(item, system_chars)
        if used + need > daily_budget:
            summary["stopped_by_budget"] = True
            stop("daily token budget reached")
            break
        if gemini and runs_mod.request_status()[2] <= 0:
            stop(f"the local daily request cap ({config.daily_request_budget()}) is reached")
            break
        if not first_call and spacing > 0:
            await sleep(spacing)
        try:
            record = await evaluate_item(item, run, decide, sleep, rate_limit_wait=rate_limit_wait)
        except RunStop as exc:
            stop(exc.reason)
            break
        append_record(result_path(item["dataset"], results_dir), record)
        spent = record["tokens_in"] + record["tokens_out"]
        used += spent
        add_usage(results_dir, today, spent)
        summary["calls"] += record["llm_calls"] or 1
        summary["tokens"] += spent
        summary["completed"] += 1
        summary["done"] += 1
        summary["rate_limit_misses"] += record["status"] == "rate_limit_miss"
        failed = record["status"] in ("rate_limit_miss", "error")
        summary["errors"] += failed
        streak.add(failed)
        seen[item["dataset"]] += 1
        out(f"[{item['dataset']}] {seen[item['dataset']]}/{totals[item['dataset']]} run {run} {item['item']} "
            f"{record['status']} intent={record['intent']} conf={record['confidence']} "
            f"{record['latency_ms']:.0f}ms {spent}tok (today {used})")
        if first_call and record["status"] == "error":  # the first real prompt is the connectivity check
            hint = runs_mod.error_hint(record["error_class"], record["error_code"], record["error_status"])
            stop(f"the first request failed ({hint}); check the key, the model name and the quota")
            break
        first_call = False
        if streak.tripped:
            stop(f"{streak.count} failures in a row (last: "
                 f"{runs_mod.error_hint(record['error_class'], record['error_code'], record['error_status'])})")
            break
    summary["tokens_today"] = used
    return summary


def resume_command(args: argparse.Namespace) -> str:
    parts = ["python -m llm_approach.eval.run_eval", f"--dataset {args.dataset}", f"--runs {args.runs}"]
    if args.sample:
        parts.append(f"--sample {args.sample}")
    parts += [f"--spacing {args.spacing:g}", f"--daily-token-budget {args.daily_token_budget}"]
    if args.only_missing:
        parts.append("--only-missing")
    parts.append("--resume")
    return " ".join(parts)


def budget_missing_message() -> str:
    return (f"No daily token budget is configured for provider {config.provider()!r}. Set "
            f"{config.GEMINI_DAILY_BUDGET_ENV} (our own safety cap, tokens per day) or pass --daily-token-budget.")


def resolve_limits(args: argparse.Namespace) -> tuple[float, int | None]:
    """(spacing, daily token budget) for a live run: the flag if given, else the active provider's setting."""
    spacing = args.spacing if args.spacing is not None else config.spacing_s()
    budget = args.daily_token_budget if args.daily_token_budget is not None else config.daily_token_budget()
    return spacing, budget


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m llm_approach.eval.run_eval", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default="all", help=f"one of {datasets.DATASETS} or 'all' (default)")
    p.add_argument("--runs", type=int, default=1, help="runs per item (default 1)")
    p.add_argument("--sample", type=int, default=None, help="fixed-seed sample of N items per dataset")
    p.add_argument("--spacing", type=float, default=None,
                   help="seconds between calls (default: the provider's setting; Groq 30, Gemini 5)")
    p.add_argument("--daily-token-budget", type=int, default=None,
                   help="stop when today's tokens would exceed this (default: the provider's setting; "
                        "Groq 150000, Gemini from the environment)")
    p.add_argument("--dry-run", action="store_true", help="no calls: print items, calls, tokens and time")
    p.add_argument("--only-missing", action="store_true", help="also retry rate-limit misses; skip finished items")
    p.add_argument("--resume", action="store_true",
                   help="continue the unfinished run (same provider, model and dataset): finished prompts are "
                        "skipped, failed ones retried")
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
        # Provider-neutral: no key, no provider needed. Without flags the reference values are Groq's.
        spacing = args.spacing if args.spacing is not None else DEFAULT_SPACING
        budget = args.daily_token_budget if args.daily_token_budget is not None else DEFAULT_DAILY_BUDGET
        todo = plan_items(names, args.runs, args.sample, RESULTS_DIR, args.only_missing)
        est = estimate(todo, spacing, budget)
        print("DRY RUN: no calls are made. Provider-neutral: token counts are estimated from prompt size "
              "(about 4 characters per token); real tokenizers differ between providers.")
        for name in names:
            d = est["per_dataset"].get(name, {"calls": 0, "tokens": 0})
            print(f"  {name:<13} calls {d['calls']:>4}  est. tokens {d['tokens']:>8}  "
                  f"est. time {d['calls'] * spacing / 60:>6.1f} min")
        print(f"  {'total':<13} calls {est['calls']:>4}  est. tokens {est['tokens']:>8}  est. time {est['minutes']:.1f} min")
        print(f"At a budget of {budget} tokens/day this needs about {est['days']:.1f} day(s). "
              f"Tokens already used today: {tokens_used_today(RESULTS_DIR, dt.date.today().isoformat())}.")
        return 0

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    problem = config.setup_problem()
    if problem:
        print(problem)
        return 2
    args.spacing, args.daily_token_budget = resolve_limits(args)
    if args.daily_token_budget is None:
        print(budget_missing_message())
        return 2
    sig = runs_mod.signature("run_eval", args.dataset, args.runs, args.sample)
    refusal = runs_mod.begin(RESULTS_DIR, sig, args.resume)
    if refusal:
        print(refusal)
        return 2
    only_missing = args.only_missing or args.resume
    todo = plan_items(names, args.runs, args.sample, RESULTS_DIR, only_missing)
    if config.provider() == "gemini":
        allowed, lines = runs_mod.preflight(len(todo), runs_mod.WORST_CASE_CALLS_PER_PROMPT, args.spacing)
        print("\n".join(lines))
        if not allowed:
            return 2
    print("LLM: {} ({})".format(*config.label()))
    total = sum(len(select_items(n, args.sample)) for n in names) * args.runs
    runs_mod.save_entry(RESULTS_DIR, sig, "running", total - len(todo), total)
    summary = asyncio.run(run_eval(names, runs=args.runs, sample=args.sample, spacing=args.spacing,
                                   daily_budget=args.daily_token_budget, only_missing=only_missing,
                                   results_dir=RESULTS_DIR))
    print(f"\nDone: {summary['completed']} of {summary['planned']} item-runs, {summary['tokens']} tokens this run, "
          f"{summary['rate_limit_misses']} rate-limit misses.")
    reason = summary.get("stop_reason")
    done, total = summary.get("done", 0), summary.get("total", 0)
    runs_mod.save_entry(RESULTS_DIR, sig, "stopped" if reason else "complete", done, total, reason)
    if reason:
        print(runs_mod.stopped_line(done, total, reason, resume_command(args)))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
