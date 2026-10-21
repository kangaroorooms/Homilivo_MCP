"""Run the multi-turn scenarios in ``scenarios/`` through ``pipeline.handle_turn``.

Two modes:

* ``--stub``: scripted decisions, fake guardrails/knowledge base, offline rooms.
  No network, no torch. This is what pytest uses.
* ``--live``: the real LLM provider (Gemini by default; the key comes from ``.env``) with the offline rooms
  fixture and the real guardrails. Never run by pytest. ``--spacing`` waits
  between LLM calls to respect the tokens-per-minute limit.

    python -m llm_approach.eval.run_scenarios --stub
    python -m llm_approach.eval.run_scenarios --live --only S1,S4 --spacing 30

``--live`` appends one line per turn to ``eval/results/scenarios_live.jsonl`` and skips scenarios
already fully recorded, so it can be resumed on the next day (``--daily-token-budget``).

Output is a pass/fail line per turn. No keys, tokens or ``.env`` values are printed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_approach import config, llm_client, pipeline, reuse, rooms_client  # noqa: E402
from llm_approach import gemini_limits  # noqa: E402
from llm_approach.eval import run_eval, runs as runs_mod  # noqa: E402
from llm_approach.schema import parse_decision  # noqa: E402
from llm_approach.state import SessionStore  # noqa: E402

LIVE_RESULTS = Path(__file__).resolve().parent / "results" / "scenarios_live.jsonl"
EST_TOKENS_PER_TURN = 4_500  # one decision call is about 4,000 tokens plus the reply

STUB_KB_ANSWER = {
    "grounded": True,
    "message": "Homilivo earns from owner subscriptions. (stub knowledge base)",
    "sources": [],
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class TurnReport:
    scenario: str
    index: int
    user: str
    checks: list[Check] = field(default_factory=list)
    skipped: str | None = None
    intent: str | None = None
    action: str | None = None
    tokens: int = 0
    latency_ms: float = 0.0
    llm_calls: int = 0

    @property
    def passed(self) -> bool:
        return self.skipped is None and all(c.ok for c in self.checks)


class ScenarioStop(Exception):
    """A live run must stop cleanly now (daily quota or request cap reached, repeated failures)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class LiveGuard:
    """Stop rules of a live run. ``before_turn`` / ``hard_stop`` stop without recording the turn (Gemini quota
    or local cap: the turn counts as not run); ``soft_stop`` runs after the turn is recorded (five failed LLM
    turns in a row, a failed turn being one whose call returned no token usage)."""

    def __init__(self, limiter: gemini_limits.GeminiLimiter | None = None, max_failures: int = runs_mod.MAX_CONSECUTIVE_FAILURES) -> None:
        self.gemini = config.provider() == "gemini"
        self.limiter = limiter or gemini_limits.get_limiter()
        self.streak = runs_mod.FailureStreak(max_failures)

    def before_turn(self) -> str | None:
        if self.gemini and runs_mod.request_status()[2] <= 0:
            return f"the local daily request cap ({config.daily_request_budget()}) is reached"
        return None

    def hard_stop(self, report: "TurnReport") -> str | None:
        if self.gemini and self.limiter.quota_exhausted:
            return "Gemini says the daily request quota is exhausted (429, not retried)"
        return None

    def soft_stop(self, report: "TurnReport") -> str | None:
        if report.llm_calls:
            self.streak.add(report.tokens == 0)
        return f"{self.streak.count} failed LLM turns in a row" if self.streak.tripped else None


def load_scenarios(only: list[str] | None = None) -> list[dict[str, Any]]:
    scenarios = [json.loads(p.read_text(encoding="utf-8")) for p in SCENARIO_DIR.glob("S*.json")]
    scenarios.sort(key=lambda s: int(s["id"][1:]))
    if only:
        wanted = {name.strip().upper() for name in only}
        scenarios = [s for s in scenarios if s["id"].upper() in wanted]
    return scenarios


# ---- stub environment ------------------------------------------------------


class _Spy:
    """Counts LLM calls and records the searches sent to the rooms client."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.llm_calls_this_turn = 0
        self.answer_calls = 0  # calls of the conversation answerer (the second call of a chat question)
        self.answer_calls_this_turn = 0
        self.searches: list[dict[str, Any]] = []


@contextlib.contextmanager
def _patched(target: Any, name: str, value: Any) -> Iterator[None]:
    with mock.patch.object(target, name, value):
        yield


@contextlib.contextmanager
def _env(**values: str | None) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status_code = status

    def json(self) -> dict:
        return {"success": False, "message": "internal error"}


@contextlib.contextmanager
def runtime(
    scenario: dict[str, Any], mode: str, script: list[Any], spy: _Spy, answers: list[Any] | None = None
) -> Iterator[None]:
    """Patch what a run needs. ``script`` is the queue of scripted LLM replies (stub); ``answers`` the
    queue of scripted replies of the conversation answerer (a dict is dumped to JSON, a str is raw model
    text, "rate_limit" / "timeout" set the budget's error class like the real client)."""
    real_search = rooms_client.search_rooms

    def spying_search(search, budget, **kw):
        spy.searches.append(dict(search))
        return real_search(search, budget, **kw)

    with contextlib.ExitStack() as stack:
        stack.enter_context(_patched(rooms_client, "search_rooms", spying_search))
        if mode == "stub":

            async def fake_decide(user_message, state_summary, budget):
                spy.llm_calls += 1
                item = script.pop(0)
                if item == "rate_limit":  # the provider answered 429: the client reports it on the budget
                    budget.error_class = "RateLimitError"
                    return None
                return None if item in (None, "timeout") else parse_decision(item)

            async def fake_generate_json(messages, budget):
                spy.answer_calls += 1
                if not budget.try_consume():
                    return None
                budget.record_latency(1.0)
                item = (answers or []).pop(0)
                if item in ("rate_limit", "timeout"):
                    budget.error_class = "RateLimitError" if item == "rate_limit" else "TimeoutError"
                    return None
                budget.tokens_in += 300
                budget.tokens_out += 40
                return json.dumps(item) if isinstance(item, dict) else item

            stack.enter_context(_patched(llm_client, "decide", fake_decide))
            stack.enter_context(_patched(llm_client, "generate_json", fake_generate_json))
            stack.enter_context(_patched(reuse, "get_is_security_blocked", lambda: (lambda m: (False, ""))))
            stack.enter_context(_patched(reuse, "get_output_guardrail", lambda: (lambda t: (True, t))))
            kb_answer = {**STUB_KB_ANSWER, **scenario.get("kb_answer", {})}  # a scenario may stub a different answer
            stack.enter_context(_patched(reuse, "get_answer_platform_question", lambda: (lambda q: dict(kb_answer))))
            if scenario.get("rooms_http_status"):
                status = int(scenario["rooms_http_status"])
                stack.enter_context(_patched(rooms_client, "_httpx_get", lambda *a, **k: _FakeResponse(status)))
                stack.enter_context(_env(**{config.OFFLINE_ROOMS_ENV: None, "KANGAROO_TOKEN": "stub-token-not-real"}))
            elif scenario.get("rooms_fixture"):  # a file under fixtures/, e.g. a larger synthetic list
                fixture = Path(__file__).resolve().parents[1] / "fixtures" / scenario["rooms_fixture"]
                stack.enter_context(_env(**{config.OFFLINE_ROOMS_ENV: str(fixture)}))
            else:
                stack.enter_context(_env(**{config.OFFLINE_ROOMS_ENV: "1"}))
            stack.enter_context(_env(LLM_APPROACH_METRICS=None))
        else:
            real_decide = llm_client.decide

            async def counting_decide(user_message, state_summary, budget):
                spy.llm_calls += 1
                return await real_decide(user_message, state_summary, budget)

            real_generate = llm_client.generate_json

            async def counting_generate(messages, budget):
                spy.answer_calls += 1
                return await real_generate(messages, budget)

            stack.enter_context(_patched(llm_client, "decide", counting_decide))
            stack.enter_context(_patched(llm_client, "generate_json", counting_generate))
            stack.enter_context(_env(**{config.OFFLINE_ROOMS_ENV: "1"}))
        yield


# ---- checking --------------------------------------------------------------


def _check_turn(
    expect: dict[str, Any], result, state, spy: _Spy, searches_before: int, mode: str
) -> list[Check]:
    m = result.metrics
    reply = result.reply
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name, bool(ok), "" if ok else detail))

    def eq(name: str, expected: Any, actual: Any) -> None:
        add(name, expected == actual, f"expected {expected!r}, got {actual!r}")

    for key, metric in (("intent", "intent"), ("operation", "operation"), ("action", "action"),
                        ("tool_called", "tool_called"), ("error_class", "error_class"),
                        ("fallback_used", "fallback_used")):
        if key in expect:
            eq(key, expect[key], m.get(metric))
    if "success" in expect:
        eq("success", expect["success"], result.success)
    if "llm_called" in expect:
        called = spy.llm_calls_this_turn > 0 if mode == "stub" else (m.get("llm_ms", 0) > 0 or m.get("tokens_in", 0) > 0)
        eq("llm_called", expect["llm_called"], called)
    if "answer_called" in expect:  # the conversation answerer (second call) ran
        eq("answer_called", expect["answer_called"], spy.answer_calls_this_turn > 0)
    if "rooms_shown" in expect:
        eq("rooms_shown", expect["rooms_shown"], bool(result.rooms))
    if "rooms_on_screen" in expect:
        eq("rooms_on_screen", expect["rooms_on_screen"], len(state.rooms_on_screen) if state else None)
    if "selected" in expect:
        eq("selected", expect["selected"], state.selected_room_index if state else None)
    if "pending" in expect:
        eq("pending", expect["pending"], (state.pending_question or {}).get("type") if state else None)
    if "facts" in expect:  # session memory: stored facts, {key: value}
        have = {k: state.get_fact(k) for k in expect["facts"]} if state else {}
        add("facts", have == expect["facts"], f"facts differ (expected, stored): {expect['facts']} vs {have}")
    if "no_facts" in expect:  # these keys must not be stored
        stored = [k for k in expect["no_facts"] if state and state.get_fact(k) is not None]
        add("no_facts", not stored, f"stored but should not be: {stored}")
    if "queries_logged" in expect:
        eq("queries_logged", expect["queries_logged"], state.query_count if state else None)
    if "search" in expect:
        sent = spy.searches[-1] if len(spy.searches) > searches_before else {}
        wrong = {k: (v, sent.get(k)) for k, v in expect["search"].items() if sent.get(k) != v}
        add("search", not wrong, f"search differs (expected, sent): {wrong}")
    contains = list(expect.get("reply_contains", []))
    if mode == "stub":
        contains += expect.get("stub_reply_contains", [])
    for text in contains:
        add(f"reply has {text!r}", text.lower() in reply.lower(), f"missing in reply: {reply[:120]!r}")
    for text in expect.get("reply_not_contains", []):
        add(f"reply lacks {text!r}", text.lower() not in reply.lower(), f"found in reply: {reply[:120]!r}")
    return checks


async def run_scenario(
    scenario: dict[str, Any], mode: str = "stub", *, spacing: float = 0.0, log=None, guard: LiveGuard | None = None
) -> list[TurnReport]:
    """Run one scenario in a fresh session. Returns one report per turn."""
    reports: list[TurnReport] = []
    if mode == "live" and scenario.get("stub_only"):
        return [
            TurnReport(scenario["id"], i, t["user"], skipped="stub only")
            for i, t in enumerate(scenario["turns"], 1)
        ]

    script = [t["llm"] for t in scenario["turns"] if t["llm"] != "never"]
    answers: list[Any] = []
    for t in scenario["turns"]:  # "answer": one scripted answerer reply, or a list (a retry)
        given = t.get("answer")
        answers += given if isinstance(given, list) else ([] if given is None else [given])
    spy = _Spy()
    store = SessionStore()
    session_id = str(uuid.uuid4())
    last_llm_turn_at: float | None = None

    with runtime(scenario, mode, script, spy, answers):
        for i, turn in enumerate(scenario["turns"], 1):
            if turn.get("new_session"):  # a brand-new session: nothing from before may be visible
                session_id = str(uuid.uuid4())
            if mode == "live" and spacing and last_llm_turn_at is not None:
                wait = spacing - (time.monotonic() - last_llm_turn_at)
                if wait > 0:
                    await asyncio.sleep(wait)
            if guard and (reason := guard.before_turn()):
                raise ScenarioStop(reason)
            calls_before, searches_before, answers_before = spy.llm_calls, len(spy.searches), spy.answer_calls
            result = await pipeline.handle_turn(session_id, turn["user"], store=store)
            spy.llm_calls_this_turn = spy.llm_calls - calls_before
            spy.answer_calls_this_turn = spy.answer_calls - answers_before
            if spy.llm_calls_this_turn:
                last_llm_turn_at = time.monotonic()
            report = TurnReport(
                scenario["id"], i, turn["user"],
                intent=result.metrics.get("intent"), action=result.metrics.get("action"),
                tokens=int(result.metrics.get("tokens_in", 0) or 0) + int(result.metrics.get("tokens_out", 0) or 0),
                latency_ms=float(result.metrics.get("total_ms", 0.0) or 0.0),
                llm_calls=int(result.metrics.get("llm_calls", 0) or 0),
            )
            report.checks = _check_turn(
                turn.get("expect", {}), result, store.peek(session_id), spy, searches_before, mode
            )
            if turn["llm"] == "never":
                report.checks.append(
                    Check("llm never called", spy.llm_calls_this_turn == 0, "the LLM was called for a blocked turn")
                )
            if guard and (reason := guard.hard_stop(report)):
                raise ScenarioStop(reason)
            reports.append(report)
            if log:
                log(report)
            if guard and (reason := guard.soft_stop(report)):
                raise ScenarioStop(reason)
    return reports


# ---- CLI -------------------------------------------------------------------


def live_record(report: TurnReport) -> dict[str, Any]:
    """One line of scenarios_live.jsonl: ids, verdicts and check names only. No message text, no reply."""
    provider, model = config.label()
    return {
        "provider": provider, "model": model,
        "scenario": report.scenario, "turn": report.index,
        "status": "skipped" if report.skipped else ("pass" if report.passed else "fail"),
        "intent": report.intent, "action": report.action,
        "failed_checks": [c.name for c in report.checks if not c.ok],
        "tokens": report.tokens, "latency_ms": round(report.latency_ms, 1),
    }


def format_report(report: TurnReport) -> str:
    if report.skipped:
        return f"{report.scenario}.{report.index}  SKIP  ({report.skipped})"
    status = "PASS" if report.passed else "FAIL"
    line = f"{report.scenario}.{report.index}  {status}  intent={report.intent} action={report.action}"
    for check in report.checks:
        if not check.ok:
            line += f"\n        - {check.name}: {check.detail}"
    return line


def finished_scenarios(wanted: dict[str, int], label: tuple[str, str] | None = None) -> set[str]:
    """Ids of scenarios whose every turn is recorded in scenarios_live.jsonl for the (provider, model) pair."""
    label = label or config.label()
    seen: dict[str, set[int]] = {}
    for rec in run_eval.read_records(LIVE_RESULTS):
        if config.record_label(rec) == label:  # only turns recorded with this provider and model
            seen.setdefault(rec["scenario"], set()).add(int(rec["turn"]))
    return {sid for sid, turns in seen.items() if sid in wanted and len(turns) >= wanted[sid]}


def resume_command(args: argparse.Namespace) -> str:
    only_arg = f" --only {args.only}" if args.only else ""
    return (f"python -m llm_approach.eval.run_scenarios --live{only_arg} --spacing {args.spacing:g} "
            f"--daily-token-budget {args.daily_token_budget} --resume")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm_approach.eval.run_scenarios")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stub", action="store_true", help="scripted decisions, offline (used by pytest)")
    mode.add_argument("--live", action="store_true", help="real LLM provider; offline rooms; real guardrails")
    parser.add_argument("--only", default="", help="comma-separated ids, e.g. S1,S4")
    parser.add_argument("--spacing", type=float, default=None,
                        help="seconds between LLM calls in --live (default: the provider's setting; Groq 30, Gemini 5)")
    parser.add_argument("--daily-token-budget", type=int, default=None,
                        help="--live: stop between scenarios when today's tokens would pass this "
                             "(default: the provider's setting; Groq 150000, Gemini from the environment)")
    parser.add_argument("--resume", action="store_true",
                        help="--live: continue the unfinished run (same provider, model and dataset); "
                             "scenarios already fully recorded are skipped")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    only = [s for s in args.only.split(",") if s.strip()]
    scenarios = load_scenarios(only)
    if not scenarios:
        print("No scenarios matched.")
        return 2
    if args.resume and not args.live:
        print("--resume only applies to --live.")
        return 2

    run_mode = "live" if args.live else "stub"
    today = dt.date.today().isoformat()
    finished: set[str] = set()
    stop_reason: list[str] = []
    sig: dict[str, Any] = {}
    live_total = 0
    if run_mode == "live":
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        problem = config.setup_problem()
        if problem:
            print(problem)
            return 2
        args.spacing, args.daily_token_budget = run_eval.resolve_limits(args)
        if args.daily_token_budget is None:
            print(run_eval.budget_missing_message())
            return 2
        sig = runs_mod.signature("run_scenarios", "scenarios:" + (",".join(s.strip().upper() for s in only) or "all"))
        refusal = runs_mod.begin(run_eval.RESULTS_DIR, sig, args.resume)
        if refusal:
            print(refusal)
            return 2
        wanted = {s["id"]: len(s["turns"]) for s in scenarios}
        finished = finished_scenarios(wanted)
        live = [s for s in scenarios if not s.get("stub_only")]
        live_total = len(live)
        todo_turns = sum(1 for s in live if s["id"] not in finished for t in s["turns"] if t["llm"] != "never")
        if config.provider() == "gemini":
            allowed, lines = runs_mod.preflight(todo_turns, runs_mod.WORST_CASE_CALLS_PER_SCENARIO_TURN, args.spacing)
            print("\n".join(lines))
            if not allowed:
                return 2
        print("LLM: {} ({})".format(*config.label()))
        runs_mod.save_entry(run_eval.RESULTS_DIR, sig, "running",
                            sum(1 for s in live if s["id"] in finished), live_total)
        print("Loading guardrails (can take ~40 s)...", flush=True)
        pipeline.preload()
    used = [run_eval.tokens_used_today(LIVE_RESULTS.parent, today)] if run_mode == "live" else [0]
    guard = LiveGuard() if run_mode == "live" else None

    def checkpoint(report: TurnReport) -> None:
        """Print the turn and, live, write its record and usage at once (a checkpoint after every turn)."""
        print(format_report(report), flush=True)
        if run_mode == "live":
            run_eval.append_record(LIVE_RESULTS, live_record(report))
            used[0] += report.tokens
            run_eval.add_usage(LIVE_RESULTS.parent, today, report.tokens)

    async def run_all() -> list[TurnReport]:
        every: list[TurnReport] = []
        for scenario in scenarios:
            if run_mode == "live":
                if scenario["id"] in finished:
                    print(f"\n{scenario['id']}: already recorded in {LIVE_RESULTS.name}, skipped")
                    continue
                need = EST_TOKENS_PER_TURN * sum(1 for t in scenario["turns"] if t["llm"] != "never")
                if used[0] + need > args.daily_token_budget:
                    stop_reason.append("daily token budget reached")
                    break
            print(f"\n{scenario['id']}: {scenario['title']}")
            try:
                reports = await run_scenario(
                    scenario, run_mode, spacing=args.spacing or 0.0, log=checkpoint, guard=guard
                )
            except ScenarioStop as exc:
                stop_reason.append(exc.reason)
                print(f"Stopping: {exc.reason}.")
                break
            every += reports
        return every

    reports = asyncio.run(run_all())
    ran = [r for r in reports if not r.skipped]
    failed = [r for r in ran if not r.passed]
    print(f"\n{len(ran) - len(failed)}/{len(ran)} turns passed, {len(reports) - len(ran)} skipped ({run_mode} mode)")
    if run_mode == "live":
        now_done = finished_scenarios({s["id"]: len(s["turns"]) for s in scenarios})
        done = sum(1 for s in scenarios if not s.get("stub_only") and s["id"] in now_done)
        reason = stop_reason[0] if stop_reason else None
        runs_mod.save_entry(run_eval.RESULTS_DIR, sig, "stopped" if reason else "complete", done, live_total, reason)
        if reason:
            not_run = [s["id"] for s in scenarios if not s.get("stub_only") and s["id"] not in now_done]
            print(f"Not run: {', '.join(not_run)}")
            print(runs_mod.stopped_line(done, live_total, reason, resume_command(args)))
            return 3
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
