"""Run the multi-turn scenarios in ``scenarios/`` through ``pipeline.handle_turn``.

Two modes:

* ``--stub``: scripted decisions, fake guardrails/knowledge base, offline rooms.
  No network, no torch. This is what pytest uses.
* ``--live``: real Groq (``GROQ_API_KEY`` from ``.env``) with the offline rooms
  fixture and the real guardrails. Never run by pytest. ``--spacing`` waits
  between LLM calls to respect the tokens-per-minute limit.

    python -m llm_approach.eval.run_scenarios --stub
    python -m llm_approach.eval.run_scenarios --live --only S1,S4 --spacing 30

Output is a pass/fail line per turn. No keys, tokens or ``.env`` values are printed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
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
from llm_approach.schema import parse_decision  # noqa: E402
from llm_approach.state import SessionStore  # noqa: E402

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

    @property
    def passed(self) -> bool:
        return self.skipped is None and all(c.ok for c in self.checks)


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
def runtime(scenario: dict[str, Any], mode: str, script: list[Any], spy: _Spy) -> Iterator[None]:
    """Patch what a run needs. ``script`` is the queue of scripted LLM replies (stub)."""
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
                return None if item in (None, "timeout") else parse_decision(item)

            stack.enter_context(_patched(llm_client, "decide", fake_decide))
            stack.enter_context(_patched(reuse, "get_is_security_blocked", lambda: (lambda m: (False, ""))))
            stack.enter_context(_patched(reuse, "get_output_guardrail", lambda: (lambda t: (True, t))))
            stack.enter_context(_patched(reuse, "get_answer_platform_question", lambda: (lambda q: dict(STUB_KB_ANSWER))))
            if scenario.get("rooms_http_status"):
                status = int(scenario["rooms_http_status"])
                stack.enter_context(_patched(rooms_client, "_httpx_get", lambda *a, **k: _FakeResponse(status)))
                stack.enter_context(_env(**{config.OFFLINE_ROOMS_ENV: None, "KANGAROO_TOKEN": "stub-token-not-real"}))
            else:
                stack.enter_context(_env(**{config.OFFLINE_ROOMS_ENV: "1"}))
            stack.enter_context(_env(LLM_APPROACH_METRICS=None))
        else:
            real_decide = llm_client.decide

            async def counting_decide(user_message, state_summary, budget):
                spy.llm_calls += 1
                return await real_decide(user_message, state_summary, budget)

            stack.enter_context(_patched(llm_client, "decide", counting_decide))
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
    if "rooms_shown" in expect:
        eq("rooms_shown", expect["rooms_shown"], bool(result.rooms))
    if "rooms_on_screen" in expect:
        eq("rooms_on_screen", expect["rooms_on_screen"], len(state.rooms_on_screen) if state else None)
    if "selected" in expect:
        eq("selected", expect["selected"], state.selected_room_index if state else None)
    if "pending" in expect:
        eq("pending", expect["pending"], (state.pending_question or {}).get("type") if state else None)
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
    scenario: dict[str, Any], mode: str = "stub", *, spacing: float = 0.0, log=None
) -> list[TurnReport]:
    """Run one scenario in a fresh session. Returns one report per turn."""
    reports: list[TurnReport] = []
    if mode == "live" and scenario.get("stub_only"):
        return [
            TurnReport(scenario["id"], i, t["user"], skipped="stub only")
            for i, t in enumerate(scenario["turns"], 1)
        ]

    script = [t["llm"] for t in scenario["turns"] if t["llm"] != "never"]
    spy = _Spy()
    store = SessionStore()
    session_id = str(uuid.uuid4())
    last_llm_turn_at: float | None = None

    with runtime(scenario, mode, script, spy):
        for i, turn in enumerate(scenario["turns"], 1):
            if mode == "live" and spacing and last_llm_turn_at is not None:
                wait = spacing - (time.monotonic() - last_llm_turn_at)
                if wait > 0:
                    await asyncio.sleep(wait)
            calls_before, searches_before = spy.llm_calls, len(spy.searches)
            result = await pipeline.handle_turn(session_id, turn["user"], store=store)
            spy.llm_calls_this_turn = spy.llm_calls - calls_before
            if spy.llm_calls_this_turn:
                last_llm_turn_at = time.monotonic()
            report = TurnReport(
                scenario["id"], i, turn["user"],
                intent=result.metrics.get("intent"), action=result.metrics.get("action"),
            )
            report.checks = _check_turn(
                turn.get("expect", {}), result, store.peek(session_id), spy, searches_before, mode
            )
            if turn["llm"] == "never":
                report.checks.append(
                    Check("llm never called", spy.llm_calls_this_turn == 0, "the LLM was called for a blocked turn")
                )
            reports.append(report)
            if log:
                log(report)
    return reports


# ---- CLI -------------------------------------------------------------------


def format_report(report: TurnReport) -> str:
    if report.skipped:
        return f"{report.scenario}.{report.index}  SKIP  ({report.skipped})"
    status = "PASS" if report.passed else "FAIL"
    line = f"{report.scenario}.{report.index}  {status}  intent={report.intent} action={report.action}"
    for check in report.checks:
        if not check.ok:
            line += f"\n        - {check.name}: {check.detail}"
    return line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm_approach.eval.run_scenarios")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stub", action="store_true", help="scripted decisions, offline (used by pytest)")
    mode.add_argument("--live", action="store_true", help="real Groq; offline rooms; real guardrails")
    parser.add_argument("--only", default="", help="comma-separated ids, e.g. S1,S4")
    parser.add_argument("--spacing", type=float, default=30.0, help="seconds between LLM calls in --live (default 30)")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    only = [s for s in args.only.split(",") if s.strip()]
    scenarios = load_scenarios(only)
    if not scenarios:
        print("No scenarios matched.")
        return 2

    run_mode = "live" if args.live else "stub"
    if run_mode == "live":
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        if not os.getenv("GROQ_API_KEY"):
            print("GROQ_API_KEY is not set. Put it in .env at the repo root.")
            return 2
        print("Loading guardrails (can take ~40 s)...", flush=True)
        pipeline.preload()

    async def run_all() -> list[TurnReport]:
        every: list[TurnReport] = []
        for scenario in scenarios:
            print(f"\n{scenario['id']}: {scenario['title']}")
            every += await run_scenario(
                scenario, run_mode, spacing=args.spacing, log=lambda r: print(format_report(r), flush=True)
            )
        return every

    reports = asyncio.run(run_all())
    ran = [r for r in reports if not r.skipped]
    failed = [r for r in ran if not r.passed]
    print(f"\n{len(ran) - len(failed)}/{len(ran)} turns passed, {len(reports) - len(ran)} skipped ({run_mode} mode)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
