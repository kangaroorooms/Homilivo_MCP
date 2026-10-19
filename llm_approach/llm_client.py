"""One Groq call per decision, JSON mode, one retry, per-turn budget.

Logs metadata only (latency, tokens, intent, confidence, valid/invalid). The
user message and the raw model output are never logged.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
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
    cached_tokens: int | None = None  # None: the usage object never reported cached tokens
    latency_ms: float = 0.0
    call_ms: list[float] = field(default_factory=list)  # latency of each Groq call, in order
    error_class: str | None = None  # "RateLimitError" when Groq answered 429, else the last error class

    def record_latency(self, ms: float) -> None:
        self.latency_ms += ms
        self.call_ms.append(ms)

    def try_consume(self) -> bool:
        if self.calls >= self.max_calls:
            return False
        self.calls += 1
        return True


def is_rate_limit_error(exc: BaseException) -> bool:
    """Groq RateLimitError, or any error carrying HTTP status 429."""
    if type(exc).__name__ == "RateLimitError":
        return True
    return getattr(exc, "status_code", None) == 429 or getattr(getattr(exc, "response", None), "status_code", None) == 429


_client: Any = None


def _get_client() -> Any:
    """Create the async Groq client on first use; the key comes from the environment."""
    global _client
    if _client is None:
        from groq import AsyncGroq

        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:  # missing or blank: fail here, before any request is made
            raise KeyError("GROQ_API_KEY")  # same error as before for a missing key
        _client = AsyncGroq(api_key=key, max_retries=0)  # fail fast; we retry once ourselves
    return _client


async def _call_groq(
    messages: list[dict[str, str]], max_tokens: int | None = None
) -> tuple[str, int, int, int | None]:
    kwargs: dict[str, Any] = dict(
        model=config.groq_model(),
        messages=messages,
        temperature=config.TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
        response_format={"type": "json_object"},
    )
    if config.REASONING_EFFORT:
        kwargs["reasoning_effort"] = config.REASONING_EFFORT
    response = await asyncio.wait_for(
        _get_client().chat.completions.create(**kwargs), timeout=config.LLM_TIMEOUT_S
    )
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None)
    cached = cached if isinstance(cached, int) and not isinstance(cached, bool) else None
    return (
        response.choices[0].message.content or "",
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
        cached,
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
            raw, tokens_in, tokens_out, cached = await _call_groq(messages)
        except asyncio.TimeoutError:
            budget.record_latency((time.perf_counter() - started) * 1000)
            budget.error_class = "TimeoutError"
            logger.warning("llm_decision attempt=%d outcome=timeout", attempt)
            return None
        except Exception as exc:  # network, auth, rate limit: class name only
            budget.record_latency((time.perf_counter() - started) * 1000)
            budget.error_class = "RateLimitError" if is_rate_limit_error(exc) else type(exc).__name__
            logger.warning("llm_decision attempt=%d outcome=error class=%s", attempt, type(exc).__name__)
            return None

        latency_ms = (time.perf_counter() - started) * 1000
        budget.record_latency(latency_ms)
        budget.tokens_in += tokens_in
        budget.tokens_out += tokens_out
        if cached is not None:
            budget.cached_tokens = (budget.cached_tokens or 0) + cached

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


async def generate_json(messages: list[dict[str, str]], budget: LLMBudget) -> str | None:
    """One more JSON-mode call on the same per-turn budget (used by the conversation answerer).

    Returns the raw model text, or None after a timeout, an API error or an exhausted budget
    (``budget.error_class`` says which). Parsing and validation are the caller's job. Latency and
    token totals accumulate on ``budget`` like for ``decide``; logs hold metadata only."""
    if not budget.try_consume():
        logger.info("llm_answer skipped: call budget exhausted")
        return None
    started = time.perf_counter()
    try:
        raw, tokens_in, tokens_out, cached = await _call_groq(messages, config.ANSWERER_MAX_TOKENS)
    except asyncio.TimeoutError:
        budget.record_latency((time.perf_counter() - started) * 1000)
        budget.error_class = "TimeoutError"
        logger.warning("llm_answer outcome=timeout")
        return None
    except Exception as exc:  # network, auth, rate limit: class name only
        budget.record_latency((time.perf_counter() - started) * 1000)
        budget.error_class = "RateLimitError" if is_rate_limit_error(exc) else type(exc).__name__
        logger.warning("llm_answer outcome=error class=%s", type(exc).__name__)
        return None
    latency_ms = (time.perf_counter() - started) * 1000
    budget.record_latency(latency_ms)
    budget.tokens_in += tokens_in
    budget.tokens_out += tokens_out
    if cached is not None:
        budget.cached_tokens = (budget.cached_tokens or 0) + cached
    logger.info("llm_answer latency_ms=%.0f tokens_in=%d tokens_out=%d", latency_ms, tokens_in, tokens_out)
    return raw
