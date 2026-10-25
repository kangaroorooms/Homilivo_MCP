"""Offline robustness report: failures injected with mocks, run through ``pipeline.handle_turn``.

No Groq or Gemini call, no rooms API call, no torch (fake guardrails). Each case runs two turns in one
session: turn 1 is a normal Bangalore search (rooms on screen), turn 2 has the failure.
For each case the report lists the reply kind, whether history and state stayed
consistent and whether any raw error text leaked into the reply.

    python -m llm_approach.eval.run_robustness      # writes eval/results/robustness.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_approach import config, pipeline, providers, reuse, rooms_client  # noqa: E402
from llm_approach import state as state_mod  # noqa: E402
from llm_approach.eval import fake_rooms_api  # noqa: E402
from llm_approach.state import SessionStore  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "robustness.json"
PRISMA_MESSAGE = ("Invalid `prisma.room.findMany()` invocation: Timed out fetching a new connection "
                  "from the connection pool. (Current connection pool timeout: 10, connection limit: 1)")
LEAK_MARKERS = ["prisma", "findmany", "connection pool", "traceback", "exception", "timeouterror", "ratelimiterror",
                "httpx", "groq", "http 500", "status 500", "internal server error", "stub-token", "api key",
                "bearer", "errno"]
GOOD_SEARCH = json.dumps({"intent": "NEW_SEARCH", "slots": {"city": "Bangalore"}, "confidence": 0.95})
SEARCH_PUNE = json.dumps({"intent": "NEW_SEARCH", "slots": {"city": "Pune"}, "confidence": 0.95})
SEARCH_ATLANTIS = json.dumps({"intent": "NEW_SEARCH", "slots": {"city": "Atlantis"}, "confidence": 0.95})
CHAT = json.dumps({"intent": "GENERAL_CHAT", "reply": "Hello! Happy to help.", "confidence": 0.95})


class RateLimitError(Exception):  # same class name as groq's, which is how the client detects it
    status_code = 429


class _FakeGroq:
    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


# case name -> (turn-2 user message, scripted Groq replies for turn 2, rooms mode, output guardrail ok)
# rooms mode: offline (fixture), http500, timeout
CASES: list[dict[str, Any]] = [
    {"name": "groq_timeout", "message": "rooms in pune", "groq": [asyncio.TimeoutError()], "rooms": "offline",
     "describe": "Groq call times out; deterministic fallback runs"},
    {"name": "groq_invalid_json_twice", "message": "rooms in pune", "groq": ["not json", "{still not json"], "rooms": "offline",
     "describe": "Groq answers invalid JSON twice; fallback runs"},
    {"name": "groq_rate_limit", "message": "rooms in pune", "groq": [RateLimitError("429 too many requests")], "rooms": "offline",
     "describe": "Groq answers 429; friendly busy reply, no fallback search"},
    {"name": "rooms_api_500_prisma", "message": "rooms in pune", "groq": [SEARCH_PUNE], "rooms": "http500",
     "describe": "rooms API returns 500 with a Prisma-style message; state must stay unchanged"},
    {"name": "rooms_api_timeout", "message": "rooms in pune", "groq": [SEARCH_PUNE], "rooms": "timeout",
     "describe": "rooms API times out twice (one retry); state must stay unchanged"},
    {"name": "empty_results", "message": "rooms in atlantis", "groq": [SEARCH_ATLANTIS], "rooms": "offline",
     "describe": "search succeeds with zero rooms; honest no-rooms reply"},
    {"name": "output_guardrail_blocks", "message": "hello there", "groq": [CHAT], "rooms": "offline", "output_ok": False,
     "describe": "the output guardrail rejects the reply; fixed safe reply, no cards"},
]


@contextlib.contextmanager
def _env(**values: str | None) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def reply_kind(result) -> str:
    m = result.metrics
    if m.get("error_class") == "RateLimitError":
        return "rate_limit_busy"
    if result.reply == pipeline.OUTPUT_BLOCKED_REPLY:
        return "output_blocked"
    if result.reply == rooms_client.UNAVAILABLE_MESSAGE:
        return "rooms_unavailable"
    if result.reply == state_mod.ERROR_REPLY:
        return "generic_error"
    if m.get("fallback_used"):
        return "fallback:" + str(m.get("action"))
    return str(m.get("action"))


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    store, sid = SessionStore(), str(uuid.uuid4())
    client = _FakeGroq([GOOD_SEARCH] + case["groq"])
    patches = [
        mock.patch.object(providers, "_get_groq_client", lambda: client),
        mock.patch.object(reuse, "get_is_security_blocked", lambda: (lambda m: (False, ""))),
        mock.patch.object(reuse, "get_output_guardrail",
                          lambda: (lambda t: (True, t)) if case.get("output_ok", True) else (lambda t: (False, "blocked"))),
        mock.patch.object(config, "ROOMS_RETRY_JITTER_S", (0.0, 0.0)),
    ]
    if case["rooms"] == "http500":
        patches.append(fake_rooms_api.failing_rooms_api(500, {"success": False, "message": PRISMA_MESSAGE}))
    elif case["rooms"] == "timeout":
        patches.append(fake_rooms_api.failing_rooms_api(None))

    with contextlib.ExitStack() as stack, _env(LLM_APPROACH_METRICS=None, **{config.PROVIDER_ENV: "groq"}):
        for p in patches:
            stack.enter_context(p)
        pipeline.configure(full_guardrails=True)
        with _env(**{config.OFFLINE_ROOMS_ENV: "1"}):
            first = await pipeline.handle_turn(sid, "rooms in bangalore", store=store)
        before = store.peek(sid)
        rooms_before, selected_before = len(before.rooms_on_screen), before.selected_room_index
        hist_before = len(before.history)
        online = case["rooms"] in ("http500", "timeout")
        env = {config.OFFLINE_ROOMS_ENV: None, "KANGAROO_TOKEN": "stub-token-not-real"} if online else {config.OFFLINE_ROOMS_ENV: "1"}
        with _env(**env):
            result = await pipeline.handle_turn(sid, case["message"], store=store)
    state = store.peek(sid)
    lowered = result.reply.lower()
    leaks = [m for m in LEAK_MARKERS if m in lowered]
    history_ok = (len(state.history) == hist_before + 2 and state.history[-2]["role"] == "user"
                  and state.history[-1]["role"] == "assistant" and state.last_bot_message == result.reply)
    expects_unchanged = case["name"] in ("rooms_api_500_prisma", "rooms_api_timeout")
    state_ok = (len(state.rooms_on_screen) == rooms_before and state.selected_room_index == selected_before) if expects_unchanged else True
    return {
        "case": case["name"], "describe": case["describe"], "first_turn_ok": first.success,
        "reply_kind": reply_kind(result), "success": result.success, "fallback_used": bool(result.metrics.get("fallback_used")),
        "error_class": result.metrics.get("error_class"), "tool_called": bool(result.metrics.get("tool_called")),
        "history_consistent": history_ok, "state_unchanged_where_required": state_ok,
        "rooms_before": rooms_before, "rooms_after": len(state.rooms_on_screen),
        "raw_error_leaked": bool(leaks), "leak_markers": leaks, "reply_chars": len(result.reply),
        "groq_replies_unused": len(client.replies),
    }


def run_all() -> dict[str, Any]:
    async def go() -> list[dict[str, Any]]:
        return [await run_case(c) for c in CASES]

    rows = asyncio.run(go())
    ok = [r for r in rows if r["history_consistent"] and r["state_unchanged_where_required"] and not r["raw_error_leaked"]]
    return {"provider": "offline-mock", "model": "none", "n": len(rows), "clean": len(ok), "rows": rows,
            "note": "Mocks only: fake Groq client (through the real llm_client), fake rooms HTTP, fake guardrails. "
                    "Reply text is not stored; only its kind, length and leak check."}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m llm_approach.eval.run_robustness")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = run_all()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in report["rows"]:
        flag = "OK  " if (r["history_consistent"] and r["state_unchanged_where_required"] and not r["raw_error_leaked"]) else "FAIL"
        print(f"{flag} {r['case']:<26} reply={r['reply_kind']:<28} history={r['history_consistent']} "
              f"leak={r['raw_error_leaked']} rooms {r['rooms_before']}->{r['rooms_after']}")
    print(f"{report['clean']}/{report['n']} cases clean. Saved {args.out}")
    return 0 if report["clean"] == report["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
