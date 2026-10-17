"""Terminal chat: ``python -m llm_approach.cli``.

Uses real Groq (``GROQ_API_KEY`` from ``.env``). Commands: /state, /reset,
/debug on|off, /quit. Never prints the key or any ``.env`` value.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

from llm_approach import config, pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
HELP = "Commands: /state  /reset  /debug on|off  /quit"
SKIP_WARNING = (
    "WARNING: --skip-guardrail-load is on. Only the extra regex security layer runs.\n"
    "         This is NOT the full safety stack. Do not use it for evaluation."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m llm_approach.cli", description="Homilivo LLM-approach chat")
    parser.add_argument("--offline-rooms", action="store_true", help="use fixtures/rooms_bangalore.json instead of the rooms API")
    parser.add_argument("--debug", action="store_true", help="print the decision (intent, operation, slots, confidence) each turn")
    parser.add_argument(
        "--skip-guardrail-load",
        action="store_true",
        help="skip the heavy guardrails (faster start). NOT the full safety stack; never use for evaluation",
    )
    return parser.parse_args(argv)


def format_rooms(cards: list[dict[str, Any]]) -> str:
    """Room cards as a simple readable table."""
    if not cards:
        return ""
    rows = [("#", "Title", "Type", "Price/month", "Location")]
    for i, card in enumerate(cards, 1):
        price = card.get("price_per_month")
        rows.append(
            (
                str(i),
                str(card.get("title") or ""),
                str(card.get("room_type") or "-"),
                f"Rs {price:,.0f}" if isinstance(price, (int, float)) else "-",
                str(card.get("location") or "-"),
            )
        )
    widths = [min(max(len(r[c]) for r in rows), 38) for c in range(5)]
    lines = []
    for n, row in enumerate(rows):
        lines.append("  ".join(cell[: widths[c]].ljust(widths[c]) for c, cell in enumerate(row)).rstrip())
        if n == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def format_debug(debug: dict[str, Any] | None, metrics: dict[str, Any]) -> str:
    if debug is None:
        return f"[debug] no decision (action={metrics.get('action')}, fallback={metrics.get('fallback_used')})"
    slots = {k: v for k, v in debug["slots"].items() if v is not None}
    return (
        f"[debug] intent={debug['intent']} operation={debug['operation']} "
        f"confidence={debug['confidence']:.2f} room_reference={debug['room_reference']}\n"
        f"[debug] slots={json.dumps(slots, ensure_ascii=False)} rejected={debug['rejected']} "
        f"action={metrics.get('action')} tool_called={metrics.get('tool_called')}"
    )


def run_chat(
    *,
    debug: bool = False,
    turn_fn: Callable[..., Any] | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
    state_fn: Callable[[str], dict[str, Any] | None] | None = None,
    reset_fn: Callable[[str], Any] | None = None,
) -> None:
    """The chat loop. The pipeline hooks are injectable so tests can mock them."""
    turn_fn = turn_fn or pipeline.handle_turn
    state_fn = state_fn or _peek_state
    reset_fn = reset_fn or pipeline.reset_session
    session_id = str(uuid.uuid4())
    loop = asyncio.new_event_loop()  # one loop for the whole chat (the Groq client is bound to it)
    print_fn(HELP)
    try:
        while True:
            try:
                line = input_fn("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print_fn()
                break
            if not line:
                continue
            if line.startswith("/"):
                command, _, arg = line.partition(" ")
                command, arg = command.lower(), arg.strip().lower()
                if command == "/quit":
                    break
                if command == "/state":
                    state = state_fn(session_id)
                    print_fn(json.dumps(state, indent=2, ensure_ascii=False, default=str) if state is not None else "(no session yet)")
                elif command == "/reset":
                    reset_fn(session_id)
                    session_id = str(uuid.uuid4())
                    print_fn("Conversation reset.")
                elif command == "/debug" and arg in ("on", "off"):
                    debug = arg == "on"
                    print_fn(f"debug {arg}")
                else:
                    print_fn(HELP)
                continue
            result = loop.run_until_complete(turn_fn(session_id, line))
            print_fn(f"bot> {result.reply}")
            table = format_rooms(result.rooms or [])
            if table:
                print_fn(table)
            if debug:
                print_fn(format_debug(result.debug, result.metrics))
    finally:
        loop.close()


def _peek_state(session_id: str) -> dict[str, Any] | None:
    state = pipeline.get_store().peek(session_id)
    return state.to_prompt_state() if state is not None else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Put it in .env at the repo root.")
        return 2
    if args.offline_rooms:
        os.environ[config.OFFLINE_ROOMS_ENV] = "1"
        print("Rooms: offline fixture (Bangalore only).")
    elif not os.getenv("KANGAROO_TOKEN"):
        print("KANGAROO_TOKEN is not set: room searches will fail. Use --offline-rooms or set it in .env.")

    if args.skip_guardrail_load:
        pipeline.configure(full_guardrails=False)
        print(SKIP_WARNING)
    else:
        print("Loading guardrails (can take ~40 s)...", flush=True)
        pipeline.preload()
        print("Guardrails loaded.")
    run_chat(debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
