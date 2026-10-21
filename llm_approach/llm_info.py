"""Prints which LLM provider and model will be used and whether its key is set. Never the key itself.

    python -m llm_approach.llm_info

Reads the repo-root ``.env`` like the server does (variables already in the environment win).
Also prints the Gemini request counter, request budget, spacing and token budget.
Used by ``scripts/start-server.ps1``. Makes no network call.
"""

from __future__ import annotations

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


def limits_lines() -> list[str]:
    """Request counter, request budget, spacing and token budget (Gemini limits). No key, no network call."""
    from llm_approach import gemini_limits

    used = gemini_limits.get_limiter().counter.used_today()
    budget = config.daily_request_budget()
    tokens = config.daily_token_budget("gemini")
    return [
        f"Gemini requests used today (Pacific day): {used} of {budget} (free tier allows {gemini_limits.RPD_LIMIT})",
        f"Gemini spacing: {config.spacing_s('gemini'):g} s between live calls "
        f"(window guard: {gemini_limits.WINDOW_MAX_REQUESTS} requests and {gemini_limits.WINDOW_MAX_TOKENS} tokens per 60 s)",
        "Gemini daily token budget (our own cap): " + (f"{tokens}" if tokens else "not set"),
    ]


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except Exception:  # noqa: BLE001 - python-dotenv missing or .env unreadable: report from the environment
        pass
    print(describe())
    for line in limits_lines():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
