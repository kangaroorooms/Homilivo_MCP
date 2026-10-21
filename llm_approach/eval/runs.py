"""Shared pieces of the live eval runs: the run-state file, the pre-run planner, the failure streak and the
STOPPED EARLY line.

``eval/results/run_state.json`` (git-ignored) remembers, per tool (``run_eval`` / ``run_scenarios``), which run
is unfinished: its provider, model, dataset, runs and sample, how many prompts are done, and why it stopped.
``--resume`` only continues a run whose provider, model, dataset, runs and sample match. The report refuses
to compare an unfinished run. The file holds ids and numbers only: no key, no prompt, no reply.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_approach import config, gemini_limits, prompt_fingerprint

STATE_FILE = "run_state.json"
# Planner assumption, per prompt: up to 2 decision calls (the invalid-JSON retry) plus 1 rate-limit retry.
# A scenario turn can also run the conversation answerer (2 more calls). One prompt can in theory use 4 requests
# (a rate-limit retry on both decision calls); the daily request counter enforces the cap either way.
WORST_CASE_CALLS_PER_PROMPT = 3
WORST_CASE_CALLS_PER_SCENARIO_TURN = 5
MAX_CONSECUTIVE_FAILURES = 5
SIGNATURE_KEYS = ("provider", "model", "prompt_fp", "dataset", "runs", "sample")


class IncompleteRunError(ValueError):
    """The results come from a run that did not finish; the message names the missing prompts."""


# ---- run state --------------------------------------------------------------------------


def signature(tool: str, dataset: str, runs: int = 1, sample: int | None = None) -> dict[str, Any]:
    provider, model = config.label()
    return {"tool": tool, "provider": provider, "model": model, "prompt_fp": prompt_fingerprint.current(),
            "dataset": dataset, "runs": runs, "sample": sample}


def load_state(results_dir: Path) -> dict[str, dict]:
    try:
        data = json.loads((results_dir / STATE_FILE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_entry(results_dir: Path, sig: dict, status: str, done: int, total: int, reason: str | None = None) -> None:
    state = load_state(results_dir)
    state[sig["tool"]] = {"signature": sig, "status": status, "done": done, "total": total, "reason": reason}
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / STATE_FILE).write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def begin(results_dir: Path, sig: dict, resume: bool) -> str | None:
    """None when the run may start, else the one-line reason it must not. Does not write anything."""
    entry = load_state(results_dir).get(sig["tool"])
    unfinished = bool(entry) and entry.get("status") != "complete"
    if resume:
        if not unfinished:
            return f"Nothing to resume: no unfinished {sig['tool']} run is recorded."
        old = entry["signature"]
        diffs = [k for k in SIGNATURE_KEYS if old.get(k) != sig.get(k)]
        if diffs:
            was = ", ".join(f"{k} {old.get(k)}" for k in diffs)
            now = ", ".join(f"{k} {sig.get(k)}" for k in diffs)
            return (f"Cannot resume: the unfinished run differs in {', '.join(diffs)} "
                    f"(unfinished run: {was}; this run: {now}). Only the same provider, model and dataset can be resumed.")
        return None
    if unfinished and entry["signature"] == sig:
        return (f"An unfinished run with these exact settings exists ({entry.get('done')} of {entry.get('total')} done). "
                "Continue it with --resume.")
    return None


def unfinished_runs(results_dir: Path) -> list[dict]:
    return [e for e in load_state(results_dir).values() if e.get("status") != "complete"]


# ---- stop line, failures ------------------------------------------------------------------


def stopped_line(done: int, total: int, reason: str, command: str) -> str:
    return f"STOPPED EARLY: {done} of {total} done. Reason: {reason}. Resume with: {command}"


class FailureStreak:
    """Counts failures in a row; ``add(True)`` is a failure, ``add(False)`` resets. ``tripped`` at 5."""

    def __init__(self, limit: int = MAX_CONSECUTIVE_FAILURES) -> None:
        self.limit, self.count = limit, 0

    def add(self, failed: bool) -> None:
        self.count = self.count + 1 if failed else 0

    @property
    def tripped(self) -> bool:
        return self.count >= self.limit


# ---- planner ----------------------------------------------------------------------------------


def request_status() -> tuple[int, int, int]:
    """(used today, daily request budget, remaining) from the shared counter."""
    used = gemini_limits.get_limiter().counter.used_today()
    budget = config.daily_request_budget()
    return used, budget, max(0, budget - used)


def preflight(prompts: int, worst_calls_per_prompt: int, spacing: float) -> tuple[bool, list[str]]:
    """Planner for a Gemini run: (allowed, lines to print). Refuses when the worst case exceeds the remaining
    requests today. There is deliberately no override."""
    used, budget, remaining = request_status()
    calls = prompts * worst_calls_per_prompt
    lines = [
        f"Plan: {prompts} prompts x {worst_calls_per_prompt} worst-case calls = {calls} requests (estimate).",
        f"Requests used today (Pacific day): {used} of {budget}; remaining: {remaining}.",
        f"Estimated time at {spacing:g} s spacing: about {prompts * spacing / 60:.1f} min typical, "
        f"{calls * spacing / 60:.1f} min worst case.",
    ]
    if calls > remaining:
        lines.append(f"REFUSED: the worst case needs {calls} requests but only {remaining} remain today. "
                     "Run fewer prompts (--dataset / --sample), use `python -m llm_approach.eval.plan` for a day-by-day "
                     "split, or continue tomorrow.")
        return False, lines
    return True, lines


def error_hint(error_class: str | None, code: int | None, status: str | None) -> str:
    """A clear, secret-free description of a failed call, e.g. 'RateLimitError, HTTP 429 RESOURCE_EXHAUSTED'."""
    parts = [error_class or "unknown error"]
    if code or status:
        parts.append("HTTP " + " ".join(str(x) for x in (code, status) if x))
    return ", ".join(parts)
