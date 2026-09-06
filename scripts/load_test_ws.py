"""
Phase 8 dev tool - WebSocket load test for /ws/chat.

NOT part of the pytest suite (see docs/Homilivo_Implementation_Prompt.md
Phase 8: "a dev tool, not part of the pytest suite"). Run it against a
REAL, already-running server (`uvicorn app.api:app`) - it does not mock
anything, unlike tests/test_streaming.py and tests/test_websocket.py.

What it checks, at whatever concurrency you ask for:

  1. Baseline throughput/latency for a batch of concurrent WS sessions,
     each running several turns (default mode "fast" - no local model
     or GROQ_API_KEY required to load-test the endpoint itself; pass
     --mode qwen/gemma/groq to also load-test a real model call).
  2. Heartbeat: at least one {"event": "ping"} arrives on a session
     that is deliberately left idle past WS_HEARTBEAT_INTERVAL_SECONDS.
  3. Reconnect: a session's connection is dropped mid-run and a NEW
     connection resumes the SAME session_id - the follow-up turn must
     still see the room/session context the first connection built.
     This is the same session-continuity behavior
     tests/test_websocket.py checks functionally; here it is checked
     under concurrent load instead of in isolation.
  4. Shared state under load: when the server is deployed with
     STATE_BACKEND=redis (see .env.example), the reconnect check above
     is the same proof it always is - session_id resolves through
     Redis, not an in-process dict - but now happening on many
     sessions at once. There is no separate Redis-specific code path
     here on purpose: the point of the Redis-backed SessionStore
     (Phase 2) is that the client-facing behavior does not change.

Usage:
    python scripts/load_test_ws.py --url ws://localhost:8000/ws/chat \
        --token <homilivo-or-demo-token> --sessions 20 --turns 5

Exit code is non-zero if any session hit an unexpected error, if no
ping was observed on the idle probe session, or if a reconnected
session failed to resume its prior context.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import websockets


@dataclass
class SessionResult:
    session_index: int
    turn_latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    saw_ping: bool = False
    reconnect_ok: bool | None = None  # None = not attempted


async def _send_turn_and_wait(ws, message: str, session_id: str | None, mode: str) -> tuple[dict, float]:
    started = time.monotonic()
    await ws.send(json.dumps({"message": message, "session_id": session_id, "mode": mode}))

    while True:
        raw = await ws.recv()
        event = json.loads(raw)
        if event["event"] in ("done", "error"):
            return event, (time.monotonic() - started) * 1000
        # "typing" / "token" - keep draining until the turn finishes.


async def _run_idle_probe_session(url: str, token: str, heartbeat_wait_s: float) -> SessionResult:
    """One session that sends nothing and just waits for a heartbeat ping."""
    result = SessionResult(session_index=-1)
    try:
        async with websockets.connect(f"{url}?token={token}") as ws:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=heartbeat_wait_s)
                event = json.loads(raw)
                result.saw_ping = event.get("event") == "ping"
            except asyncio.TimeoutError:
                result.saw_ping = False
    except Exception as error:  # noqa: BLE001 - this is a load-test script, not app code
        result.errors.append(f"idle probe connection failed: {error}")
    return result


async def _run_session(
    url: str,
    token: str,
    index: int,
    turns: int,
    mode: str,
    reconnect: bool,
) -> SessionResult:
    result = SessionResult(session_index=index)
    session_id: str | None = None

    try:
        async with websockets.connect(f"{url}?token={token}") as ws:
            event, latency_ms = await _send_turn_and_wait(
                ws, "find rooms in Jaipur", session_id, mode
            )
            result.turn_latencies_ms.append(latency_ms)
            if event["event"] == "error":
                result.errors.append(event.get("detail", "unknown error"))
            else:
                session_id = event["session_id"]

            remaining = max(0, turns - 1)
            # Reserve the last turn (if reconnecting) to run over a
            # brand-new connection instead of this one.
            turns_here = remaining - 1 if reconnect and remaining > 0 else remaining

            for _ in range(turns_here):
                event, latency_ms = await _send_turn_and_wait(
                    ws, "which one is cheapest?", session_id, mode
                )
                result.turn_latencies_ms.append(latency_ms)
                if event["event"] == "error":
                    result.errors.append(event.get("detail", "unknown error"))

        # Connection closed above (simulates a drop). If requested,
        # open a NEW connection and prove the session resumes.
        if reconnect and session_id is not None:
            try:
                async with websockets.connect(f"{url}?token={token}") as ws2:
                    event, latency_ms = await _send_turn_and_wait(
                        ws2, "which one is cheapest?", session_id, mode
                    )
                    result.turn_latencies_ms.append(latency_ms)
                    result.reconnect_ok = (
                        event["event"] == "done"
                        and event.get("session_id") == session_id
                        and event.get("selected_room") is not None
                    )
                    if not result.reconnect_ok:
                        result.errors.append(
                            f"reconnect did not resume session context: {event}"
                        )
            except Exception as error:  # noqa: BLE001
                result.reconnect_ok = False
                result.errors.append(f"reconnect failed: {error}")

    except Exception as error:  # noqa: BLE001
        result.errors.append(f"session failed: {error}")

    return result


def _summarize(results: list[SessionResult]) -> int:
    all_latencies = [ms for r in results for ms in r.turn_latencies_ms]
    total_errors = sum(len(r.errors) for r in results)
    reconnect_attempts = [r for r in results if r.reconnect_ok is not None]
    reconnect_successes = [r for r in reconnect_attempts if r.reconnect_ok]

    print("\n=== Load test summary ===")
    print(f"Sessions run:        {len(results)}")
    print(f"Turns completed:     {len(all_latencies)}")
    if all_latencies:
        ordered = sorted(all_latencies)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
        print(f"Turn latency (ms):   avg={statistics.mean(ordered):.1f} p50={p50:.1f} p95={p95:.1f} max={ordered[-1]:.1f}")
    print(f"Errors:              {total_errors}")
    if reconnect_attempts:
        print(f"Reconnect resumed:   {len(reconnect_successes)}/{len(reconnect_attempts)}")

    for r in results:
        if r.errors:
            label = "idle-probe" if r.session_index == -1 else f"session {r.session_index}"
            print(f"  [{label}] {r.errors}")

    ok = total_errors == 0 and (
        not reconnect_attempts or len(reconnect_successes) == len(reconnect_attempts)
    )
    return 0 if ok else 1


async def _main_async(args: argparse.Namespace) -> int:
    tasks = [
        _run_session(
            args.url, args.token, index, args.turns, args.mode, reconnect=(index % 4 == 0)
        )
        for index in range(args.sessions)
    ]

    idle_probe = _run_idle_probe_session(
        args.url, args.token, heartbeat_wait_s=args.heartbeat_wait
    )

    results = list(await asyncio.gather(*tasks, idle_probe))

    idle_result = results[-1]
    if not idle_result.saw_ping:
        idle_result.errors.append(
            f"expected a heartbeat ping within {args.heartbeat_wait}s of an idle "
            "connection - none arrived (check WS_HEARTBEAT_INTERVAL_SECONDS on the "
            "server matches --heartbeat-wait)"
        )

    return _summarize(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://localhost:8000/ws/chat")
    parser.add_argument("--token", default="local-dev-token", help="Homilivo or demo token")
    parser.add_argument("--sessions", type=int, default=20, help="concurrent WS sessions")
    parser.add_argument("--turns", type=int, default=5, help="turns per session")
    parser.add_argument("--mode", default="fast", choices=["fast", "qwen", "gemma", "groq"])
    parser.add_argument(
        "--heartbeat-wait",
        type=float,
        default=25.0,
        help="seconds to wait on the idle probe session for a ping "
        "(must exceed the server's WS_HEARTBEAT_INTERVAL_SECONDS, default 20)",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(_main_async(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
