"""Live smoke test for the Gemini provider: ONE request through the real pipeline's ``decide()``.

    python -m llm_approach.smoke_gemini

Needs ``GEMINI_API_KEY`` in the environment or in ``.env`` (loaded here, never overriding the environment)
and ``pip install -r llm_approach/requirements-gemini.txt``. Uses the model from
``LLM_APPROACH_GEMINI_MODEL`` (default ``gemini-3.5-flash-lite``) whatever ``LLM_APPROACH_PROVIDER`` says.

Prints only: provider, model, the thinking setting, the request shape (structure and non-text values), OK or
the error class (with HTTP code, status and a redacted message cut to 200 characters), whether the reply
parsed as a valid decision, finish reason, token usage (with thinking tokens) and latency in ms. Never the
key, the prompt or the reply text. One call, no retry.

If that call fails with a 4xx other than 429, up to 6 extra calls retry the SAME request with one change each
(no thinking config, no JSON mode, system text in the first user message, no output cap, last user message
only, no timeout option) and print OK or FAILED with code and status for each, then the first one that worked.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from llm_approach import config, prompts, providers
from llm_approach.llm_client import LLMBudget, decide

REPO_ROOT = Path(__file__).resolve().parents[1]
MESSAGE = "find me a room in bangalore"  # a fixed, harmless message


def _variations(client: Any, contents: list[dict[str, Any]], cfg: dict[str, Any]):
    """(label, client, contents, config) for each single change, in the fixed order."""
    def without(*keys: str) -> dict[str, Any]:
        return {k: v for k, v in cfg.items() if k not in keys}

    system = cfg.get("system_instruction")
    merged = [dict(c, parts=[dict(p) for p in c["parts"]]) for c in contents]
    if system and merged:
        merged[0]["parts"][0]["text"] = system + "\n\n" + merged[0]["parts"][0]["text"]
    last_user = [c for c in contents if c["role"] == "user"][-1:]
    return [
        ("without the thinking config", client, contents, without("thinking_config")),
        ("without JSON response mode", client, contents, without("response_mime_type")),
        ("without the system instruction", client, merged, without("system_instruction")),
        ("without the output cap", client, contents, without("max_output_tokens")),
        ("with only the last user message", client, last_user, cfg),
        ("without the timeout option", None, contents, cfg),  # None: build a client without the timeout
    ]


def _failure(exc: BaseException) -> str:
    code, status = providers.error_details(exc)
    return f"FAILED {type(exc).__name__} code {code} status {status}"


async def _narrow(contents: list[dict[str, Any]], cfg: dict[str, Any], out: Callable[[str], None]) -> None:
    client = providers._get_gemini_client()
    first_ok = None
    for label, cl, cts, cf in _variations(client, contents, cfg):
        try:
            if cl is None:
                cl = providers._new_gemini_client(None)
            await providers.send_gemini(cl, cts, cf)
            result = "OK"
            first_ok = first_ok or label
        except asyncio.TimeoutError:
            result = "FAILED TimeoutError"
        except Exception as exc:  # noqa: BLE001 - class, code and status only
            result = _failure(exc)
        out(f"  {label}: {result}")
    out(f"first variation that succeeded: {first_ok or 'none succeeded'}")


async def run(decide_fn: Callable[..., Awaitable[Any]] = decide, out: Callable[[str], None] = print) -> int:
    os.environ[config.PROVIDER_ENV] = "gemini"  # this script tests Gemini, whatever the server is set to
    if not config.key_present("gemini"):
        out(f"{config.key_env('gemini')} is not set. Put it in .env at the repo root. Nothing was called.")
        return 2
    try:
        thinking = config.gemini_thinking_label()
    except ValueError as exc:  # invalid LLM_APPROACH_GEMINI_THINKING: names the allowed values, nothing else
        out(str(exc) + ". Nothing was called.")
        return 2
    budget = LLMBudget(max_calls=1)  # one request, no JSON retry
    state = {"current_search": None, "last_bot_message": None, "rooms_on_screen": []}
    contents, cfg = providers.build_gemini_request(
        prompts.build_messages(MESSAGE, prompts.build_state_summary(state)), config.gemini_max_output_tokens()
    )
    out(f"request:   {providers.request_shape(contents, cfg)}")
    decision = await decide_fn(MESSAGE, prompts.build_state_summary(state), budget)
    ok = budget.error_class is None and decision is not None
    outcome = "OK" if ok else (budget.error_class or "InvalidOrEmptyReply")
    out(f"provider:  gemini\nmodel:     {config.model('gemini')}\nthinking:  {thinking}\nresult:    {outcome}\n"
        f"valid decision: {'yes' if decision is not None else 'no'}\n"
        f"finish:    {budget.finish_reason or 'n/a'}\n"
        f"tokens:    in {budget.tokens_in}, out {budget.tokens_out}, thinking {budget.thinking_tokens}\n"
        f"latency:   {budget.latency_ms:.0f} ms")
    if budget.error_class is not None:
        out(f"error:     code {budget.error_code}, status {budget.error_status}, "
            f"message {providers.redact(budget.error_message or '')!r}")
    if budget.error_code is not None and 400 <= budget.error_code < 500 and budget.error_code != 429:
        out("narrowing down: same request, one change at a time")
        await _narrow(contents, cfg, out)
    return 0 if ok else 1


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:
        pass
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
