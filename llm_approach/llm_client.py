"""One Groq call per decision, JSON mode, one retry, per-turn budget.

Logs metadata only (latency, tokens, intent, confidence, valid/invalid). The
user message and the raw model output are never logged.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from llm_approach import config, prompts
from llm_approach.schema import Decision, parse_decision

logger = logging.getLogger(__name__)

RETRY_REMINDER = (
    "Your last reply was not valid JSON. Return valid JSON only: one object "
    "following the schema, no prose, no code fences."
)


@dataclass
class LLMBudget:
    """Counts LLM calls (and usage) for one turn."""

    max_calls: int = config.MAX_LLM_CALLS_PER_TURN
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0

    def try_consume(self) -> bool:
        if self.calls >= self.max_calls:
            return False
        self.calls += 1
        return True


_client: Any = None


def _get_client() -> Any:
    """Create the async Groq client on first use; the key comes from the environment."""
    global _client
    if _client is None:
        from groq import AsyncGroq

        _client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"], max_retries=0)  # fail fast; we retry once ourselves
    return _client


async def _call_groq(messages: list[dict[str, str]]) -> tuple[str, int, int]:
    kwargs: dict[str, Any] = dict(
        model=config.groq_model(),
        messages=messages,
        temperature=config.TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    if config.REASONING_EFFORT:
        kwargs["reasoning_effort"] = config.REASONING_EFFORT
    response = await asyncio.wait_for(
        _get_client().chat.completions.create(**kwargs), timeout=config.LLM_TIMEOUT_S
    )
    usage = getattr(response, "usage", None)
    return (
        response.choices[0].message.content or "",
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


async def decide(
    user_message: str, state_summary: dict[str, Any], budget: LLMBudget
) -> Decision | None:
    """Return a Decision, or None after a timeout, an API error or two invalid replies.

    Every attempt is counted on ``budget``; token and latency totals accumulate
    there too.
    """
    messages = prompts.build_messages(user_message, state_summary)
    for attempt in (1, 2):
        if not budget.try_consume():
            logger.info("llm_decision skipped: call budget exhausted")
            return None
        started = time.perf_counter()
        try:
            raw, tokens_in, tokens_out = await _call_groq(messages)
        except asyncio.TimeoutError:
            budget.latency_ms += (time.perf_counter() - started) * 1000
            logger.warning("llm_decision attempt=%d outcome=timeout", attempt)
            return None
        except Exception as exc:  # network, auth, rate limit: class name only
            budget.latency_ms += (time.perf_counter() - started) * 1000
            logger.warning("llm_decision attempt=%d outcome=error class=%s", attempt, type(exc).__name__)
            return None

        latency_ms = (time.perf_counter() - started) * 1000
        budget.latency_ms += latency_ms
        budget.tokens_in += tokens_in
        budget.tokens_out += tokens_out

        decision = parse_decision(raw)
        if decision is not None:
            logger.info(
                "llm_decision attempt=%d valid=True intent=%s confidence=%.2f latency_ms=%.0f tokens_in=%d tokens_out=%d",
                attempt, decision.intent.value, decision.confidence, latency_ms, tokens_in, tokens_out,
            )
            return decision

        logger.info(
            "llm_decision attempt=%d valid=False latency_ms=%.0f tokens_in=%d tokens_out=%d",
            attempt, latency_ms, tokens_in, tokens_out,
        )
        if attempt == 1:
            messages = messages + [{"role": "user", "content": RETRY_REMINDER}]
    return None
