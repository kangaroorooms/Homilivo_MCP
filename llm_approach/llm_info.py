"""Prints which LLM provider and model will be used and whether its key is set. Never the key itself.

    python -m llm_approach.llm_info
    python -m llm_approach.llm_info --seed-used 204 [--seed-tokens 150000] [--seed-at 12:36]

Reads the repo-root ``.env`` like the server does (variables already in the environment win).
Also prints the Gemini requests and tokens used in the last 24 hours (or since the fixed daily reset, when
``LLM_APPROACH_GEMINI_DAY_RESET`` is set), the request cap and token budget, the spacing, and when capacity
starts to free up. ``--seed-used N`` records N requests (and ``--seed-tokens T`` tokens) as already used at the
given local time (default: now), to copy the number from Google's dashboard; N may not exceed 500.
Used by ``scripts/start-server.ps1``. Makes no network call.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from llm_approach import config

REPO_ROOT = Path(__file__).resolve().parents[1]


def describe() -> str:
    try:
        provider = config.provider()
    except ValueError as exc:
        return str(exc)
    state = "set" if config.key_present(provider) else "missing"
    return f"LLM provider: {provider} | model: {config.model(provider)} | {config.key_env(provider)}: {state}"


def _local(when: dt.datetime) -> str:
    return when.astimezone().strftime("%Y-%m-%d %H:%M")


def limits_lines() -> list[str]:
    """Requests and tokens used in the window, the caps, spacing and the free-up time (Gemini limits).
    No key, no network call."""
    from llm_approach import gemini_limits

    counter = gemini_limits.get_limiter().counter
    used, tokens_used = counter.used_requests(), counter.used_tokens()
    budget = config.daily_request_budget()
    tokens = config.daily_token_budget("gemini")
    try:
        reset = config.day_reset()
    except ValueError:
        reset = None
    window = f"since the {reset:%H:%M} UTC reset" if reset else "in the last 24 h"
    free = counter.free_up_time()
    now = counter.wall()
    if free is None:
        free_line = "Gemini capacity: nothing counted, all of it is free"
    else:
        mins = max(0, int((free - now).total_seconds() // 60))
        what = "the counters reset" if reset else "the oldest entries drop out and capacity starts to free up"
        free_line = f"Gemini {what} at: {_local(free)} local ({mins // 60} h {mins % 60:02d} min from now)"
    return [
        f"Gemini requests used {window}: {used} of {budget} (free tier allows {gemini_limits.RPD_LIMIT})",
        f"Gemini tokens used {window}: {tokens_used} of " + (f"{tokens} (our own budget)" if tokens else "no budget set"),
        free_line,
        f"Gemini spacing: {config.spacing_s('gemini'):g} s between live calls "
        f"(window guard: {gemini_limits.window_max_requests()} requests and {gemini_limits.window_max_tokens()} tokens per {gemini_limits.window_seconds():g} s)",
    ]


def seed(requests: int, tokens: int = 0, at: str | None = None) -> str:
    """Record ``requests`` (and ``tokens``) as used at local time ``at`` (HH:MM, today, or yesterday when that is
    still in the future; default now). Returns what was written. Raises ValueError on bad input. No API call."""
    from llm_approach import gemini_limits

    if not 1 <= requests <= gemini_limits.MAX_SEED_REQUESTS:
        raise ValueError(f"--seed-used must be between 1 and {gemini_limits.MAX_SEED_REQUESTS} "
                         f"(the free tier's daily limit); got {requests}")
    if tokens < 0:
        raise ValueError("--seed-tokens cannot be negative")
    counter = gemini_limits.get_limiter().counter
    now = counter.wall().astimezone()
    when = now
    if at:
        try:
            clock = dt.datetime.strptime(at, "%H:%M").time()
        except ValueError:
            raise ValueError("--seed-at must be a local time like 12:36 (HH:MM)") from None
        when = now.replace(hour=clock.hour, minute=clock.minute, second=0, microsecond=0)
        if when > now:
            when -= dt.timedelta(days=1)
    counter.add(requests, tokens=tokens, at=when)
    return f"Recorded {requests} requests and {tokens} tokens as used at {_local(when)} (local time)."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm_approach.llm_info", description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed-used", type=int, metavar="N", help="record N requests as already used (max 500)")
    parser.add_argument("--seed-tokens", type=int, default=0, metavar="T", help="with --seed-used: also T tokens")
    parser.add_argument("--seed-at", metavar="HH:MM", help="with --seed-used: local time they were used (default now)")
    args = parser.parse_args([] if argv is None else argv)  # the __main__ block passes the real command line
    if args.seed_used is None and (args.seed_tokens or args.seed_at):
        parser.error("--seed-tokens and --seed-at need --seed-used")
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except Exception:  # noqa: BLE001 - python-dotenv missing or .env unreadable: report from the environment
        pass
    if args.seed_used is not None:
        try:
            print(seed(args.seed_used, args.seed_tokens, args.seed_at))
        except ValueError as exc:
            print(f"Refused: {exc}")
            return 2
    print(describe())
    for line in limits_lines():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
