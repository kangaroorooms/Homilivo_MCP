"""One LLM call per decision (Gemini or Groq, see providers.py), JSON mode, one retry, per-turn budget.

Logs metadata only (latency, tokens, intent, confidence, valid/invalid). The
user message and the raw model output are never logged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from llm_approach import config, gemini_limits, observability, prompts, providers
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
    call_ms: list[float] = field(default_factory=list)  # latency of each LLM call, in order
    error_class: str | None = None  # "RateLimitError" when the provider answered 429, else the last error class
    thinking_tokens: int = 0  # Gemini thinking tokens; already part of tokens_out
    finish_reason: str | None = None  # finish reason of the last reply (Gemini only)
    error_code: int | None = None  # HTTP code of the last API error (Gemini), e.g. 400
    error_status: str | None = None  # its status string, e.g. "INVALID_ARGUMENT"
    error_kind: str | None = None  # Gemini limits: "request_cap", "daily_quota" or "rate_limit_retried" (see gemini_limits)
    error_message: str | None = None  # redacted, cut to 200 chars; for the smoke test only, never logged

    def record_latency(self, ms: float) -> None:
        self.latency_ms += ms
        self.call_ms.append(ms)
        _observe_latency(ms)

    def try_consume(self) -> bool:
        if self.calls >= self.max_calls:
            return False
        # Gemini calls can take 20 s each: once a turn's calls have used the time budget, no further call
        # (the JSON retry or the answerer's second call). Groq keeps its call-count-only limit.
        if self.calls and config.provider() == "gemini" and self.latency_ms >= config.GEMINI_TURN_LLM_BUDGET_S * 1000:
            return False
        self.calls += 1
        return True


def is_rate_limit_error(exc: BaseException) -> bool:
    """Groq RateLimitError, or any error carrying HTTP status 429 (Gemini: ``code`` 429 / RESOURCE_EXHAUSTED)."""
    if type(exc).__name__ == "RateLimitError":
        return True
    if getattr(exc, "status_code", None) == 429 or getattr(getattr(exc, "response", None), "status_code", None) == 429:
        return True
    return getattr(exc, "code", None) == 429 or getattr(exc, "status", None) == "RESOURCE_EXHAUSTED"


def _record_error(budget: LLMBudget, exc: BaseException) -> None:
    budget.error_class = "RateLimitError" if is_rate_limit_error(exc) else type(exc).__name__
    budget.error_code, budget.error_status = providers.error_details(exc)
    budget.error_message = providers.redact(str(exc))
    budget.error_kind = getattr(exc, "kind", None) if isinstance(exc, gemini_limits.GeminiLimitError) else None


def _error_log(exc: BaseException) -> str:
    code, status = providers.error_details(exc)
    return f"class={type(exc).__name__} code={code} status={status}"


async def _complete(
    messages: list[dict[str, str]], max_tokens: int | None = None
) -> tuple[str, int, int, int | None, providers.Completion]:
    """The one provider call: (text, tokens in, tokens out, cached tokens, the full Completion).
    A reply cut off by the output cap (or blocked) comes back with empty text: it is retried like invalid JSON."""
    c = await providers.complete(messages, max_tokens)
    return c.text, c.tokens_in, c.tokens_out, c.cached, c


def _observe_latency(ms: float) -> None:
    """Feed the production latency metric (provider and model names only). Never raises."""
    try:
        observability.record_llm_latency(ms, config.model())
    except Exception:  # noqa: BLE001
        pass


def _observe_usage(messages: list[dict[str, str]], completion_text: str) -> None:
    """Feed the production usage estimate; it keeps token counts only, no text. Never raises."""
    try:
        prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
        observability.record_llm_usage(config.provider(), config.model(), prompt_text, completion_text)
    except Exception:  # noqa: BLE001
        pass


def _add_usage(budget: LLMBudget, tokens_in: int, tokens_out: int, cached: int | None, c: providers.Completion) -> None:
    budget.tokens_in += tokens_in
    budget.tokens_out += tokens_out  # includes Gemini thinking tokens
    budget.thinking_tokens += c.thinking_tokens
    budget.finish_reason = c.finish_reason
    if cached is not None:
        budget.cached_tokens = (budget.cached_tokens or 0) + cached


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
            raw, tokens_in, tokens_out, cached, completion = await _complete(messages)
        except asyncio.TimeoutError:
            budget.record_latency((time.perf_counter() - started) * 1000)
            budget.error_class = "TimeoutError"
            logger.warning("llm_decision attempt=%d outcome=timeout", attempt)
            return None
        except Exception as exc:  # network, auth, rate limit: class name only
            budget.record_latency((time.perf_counter() - started) * 1000)
            _record_error(budget, exc)
            logger.warning("llm_decision attempt=%d outcome=error %s", attempt, _error_log(exc))
            return None

        latency_ms = (time.perf_counter() - started) * 1000
        budget.record_latency(latency_ms)
        _add_usage(budget, tokens_in, tokens_out, cached, completion)

        _observe_usage(messages, raw)
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
        raw, tokens_in, tokens_out, cached, completion = await _complete(messages, config.ANSWERER_MAX_TOKENS)
    except asyncio.TimeoutError:
        budget.record_latency((time.perf_counter() - started) * 1000)
        budget.error_class = "TimeoutError"
        logger.warning("llm_answer outcome=timeout")
        return None
    except Exception as exc:  # network, auth, rate limit: class name only
        budget.record_latency((time.perf_counter() - started) * 1000)
        _record_error(budget, exc)
        logger.warning("llm_answer outcome=error %s", _error_log(exc))
        return None
    latency_ms = (time.perf_counter() - started) * 1000
    budget.record_latency(latency_ms)
    _add_usage(budget, tokens_in, tokens_out, cached, completion)
    _observe_usage(messages, raw)
    logger.info("llm_answer latency_ms=%.0f tokens_in=%d tokens_out=%d", latency_ms, tokens_in, tokens_out)
    return raw
