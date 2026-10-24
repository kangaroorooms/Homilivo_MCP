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

from llm_approach import config, gemini_limits, llm_resilience, observability, prompts, providers
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
    failure_kind: str | None = None  # llm_resilience kind of the last failure: timeout, rate_limit, bad_json, ...
    use_provider: str | None = None  # set once the turn switched to the fallback provider: later calls stay on it
    breaker_open: bool = False  # the primary was skipped (breaker open) and no fallback answered

    def record_latency(self, ms: float, provider: str | None = None) -> None:
        self.latency_ms += ms
        self.call_ms.append(ms)
        _observe_latency(ms, provider)

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
    messages: list[dict[str, str]], max_tokens: int | None = None, provider: str | None = None
) -> tuple[str, int, int, int | None, providers.Completion]:
    """The one provider call: (text, tokens in, tokens out, cached tokens, the full Completion).
    A reply cut off by the output cap (or blocked) comes back with empty text: it is retried like invalid JSON.
    ``provider`` is the fallback provider's name; None means the active one."""
    c = await (providers.complete(messages, max_tokens) if provider is None
               else providers.complete(messages, max_tokens, provider))
    return c.text, c.tokens_in, c.tokens_out, c.cached, c


def _observe_latency(ms: float, provider: str | None = None) -> None:
    """Feed the production latency metric (provider and model names only). Never raises."""
    try:
        observability.record_llm_latency(ms, config.model(provider))
    except Exception:  # noqa: BLE001
        pass


def _observe_usage(messages: list[dict[str, str]], completion_text: str, provider: str | None = None) -> None:
    """Feed the production usage estimate; it keeps token counts only, no text. Never raises."""
    try:
        prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
        name = provider or config.provider()
        observability.record_llm_usage(name, config.model(name), prompt_text, completion_text)
    except Exception:  # noqa: BLE001
        pass


def _add_usage(budget: LLMBudget, tokens_in: int, tokens_out: int, cached: int | None, c: providers.Completion) -> None:
    budget.tokens_in += tokens_in
    budget.tokens_out += tokens_out  # includes Gemini thinking tokens
    budget.thinking_tokens += c.thinking_tokens
    budget.finish_reason = c.finish_reason
    if cached is not None:
        budget.cached_tokens = (budget.cached_tokens or 0) + cached


def _primary_gate(budget: LLMBudget) -> str | None:
    """Where this turn's next call goes: None = the active provider, else the fallback provider's name.
    Returns "skip" when the breaker is open and no fallback is configured (the turn gets the polite reply)."""
    if budget.use_provider:
        return budget.use_provider
    primary = config.provider()
    if llm_resilience.breaker(primary).allow():
        return None
    observability.record_llm_event("breaker_skip")
    fb = config.fallback_provider()
    if fb:
        llm_resilience.note_switch(primary, fb, "breaker_open")
        budget.use_provider = fb
        return fb
    logger.warning("llm primary skipped (breaker open), no fallback provider")
    budget.breaker_open, budget.error_class, budget.failure_kind = True, "BreakerOpen", "breaker_open"
    return "skip"


def _fail(budget: LLMBudget, kind: str, prov: str | None) -> None:
    """Note one failure of the active provider (``prov`` None) or of the fallback provider."""
    budget.failure_kind = kind
    llm_resilience.note_failure(kind, prov or config.provider(), primary=prov is None)


def _switch_to_fallback(budget: LLMBudget) -> str | None:
    """The fallback provider to retry this turn on, once: None when it is off, already used, or the failure
    kind is one the fallback does not cover."""
    fb = config.fallback_provider()
    if not fb or budget.use_provider or budget.failure_kind not in llm_resilience.FALLBACK_KINDS:
        return None
    llm_resilience.note_switch(config.provider(), fb, budget.failure_kind or "")
    budget.use_provider = fb
    return fb


def _clear_error(budget: LLMBudget) -> None:
    """The fallback answered: the active provider's error no longer describes this turn."""
    budget.error_class = budget.error_kind = budget.error_code = budget.error_status = None
    budget.error_message = budget.failure_kind = None


async def _decide_on(messages: list[dict[str, str]], budget: LLMBudget, prov: str | None) -> Decision | None:
    """The decision call on one provider: the active one (``prov`` None: two attempts, the second with a JSON
    reminder) or the fallback provider (one attempt). Failures are counted; the active provider's feed the breaker."""
    name = prov or config.provider()
    budget.failure_kind = None
    for attempt in ((1, 2) if prov is None else (1,)):
        if prov is None:
            if not budget.try_consume():
                logger.info("llm_decision skipped: call budget exhausted")
                return None
        else:
            budget.calls += 1  # the fallback call has its own allowance: it is not part of the active provider's cap
        started = time.perf_counter()
        try:
            raw, tokens_in, tokens_out, cached, completion = await (
                _complete(messages) if prov is None else _complete(messages, provider=prov))
        except asyncio.TimeoutError:
            budget.record_latency((time.perf_counter() - started) * 1000, name)
            budget.error_class = "TimeoutError"
            logger.warning("llm_decision attempt=%d outcome=timeout", attempt)
            _fail(budget, "timeout", prov)
            return None
        except Exception as exc:  # network, auth, rate limit: class name only
            budget.record_latency((time.perf_counter() - started) * 1000, name)
            _record_error(budget, exc)
            logger.warning("llm_decision attempt=%d outcome=error %s", attempt, _error_log(exc))
            _fail(budget, llm_resilience.classify(exc), prov)
            return None

        latency_ms = (time.perf_counter() - started) * 1000
        budget.record_latency(latency_ms, name)
        _add_usage(budget, tokens_in, tokens_out, cached, completion)

        _observe_usage(messages, raw, name)
        decision = parse_decision(raw)
        if decision is not None:
            logger.info(
                "llm_decision attempt=%d valid=True intent=%s confidence=%.2f latency_ms=%.0f tokens_in=%d tokens_out=%d",
                attempt, decision.intent.value, decision.confidence, latency_ms, tokens_in, tokens_out,
            )
            if prov is None:
                llm_resilience.breaker(name).record_success()
            return decision

        logger.info(
            "llm_decision attempt=%d valid=False latency_ms=%.0f tokens_in=%d tokens_out=%d",
            attempt, latency_ms, tokens_in, tokens_out,
        )
        if attempt == 1 and prov is None:
            messages = messages + [{"role": "user", "content": RETRY_REMINDER}]
    _fail(budget, llm_resilience.BAD_JSON, prov)  # every attempt ran and none parsed
    return None


async def decide(
    user_message: str, state_summary: dict[str, Any], budget: LLMBudget
) -> Decision | None:
    """Return a Decision, or None after a timeout, an API error or two invalid replies.

    Every attempt is counted on ``budget``; token and latency totals accumulate
    there too. With ``LLM_APPROACH_FALLBACK_PROVIDER`` set, a timeout, 429, 5xx or bad JSON on the active
    provider is retried once on the fallback provider. After repeated failures the circuit breaker skips the
    active provider for a while (``budget.breaker_open`` tells the pipeline to answer politely).
    """
    messages = prompts.build_messages(user_message, state_summary)
    gate = _primary_gate(budget)
    if gate == "skip":
        return None
    decision = await _decide_on(messages, budget, gate)
    if decision is None and gate is None and _switch_to_fallback(budget):
        decision = await _decide_on(messages, budget, budget.use_provider)
        if decision is not None:
            _clear_error(budget)
    return decision


async def _generate_on(messages: list[dict[str, str]], budget: LLMBudget, prov: str | None) -> str | None:
    name = prov or config.provider()
    budget.failure_kind = None
    if prov is None:
        if not budget.try_consume():
            logger.info("llm_answer skipped: call budget exhausted")
            return None
    else:
        budget.calls += 1
    started = time.perf_counter()
    try:
        raw, tokens_in, tokens_out, cached, completion = await (
            _complete(messages, config.ANSWERER_MAX_TOKENS) if prov is None
            else _complete(messages, config.ANSWERER_MAX_TOKENS, prov))
    except asyncio.TimeoutError:
        budget.record_latency((time.perf_counter() - started) * 1000, name)
        budget.error_class = "TimeoutError"
        logger.warning("llm_answer outcome=timeout")
        _fail(budget, "timeout", prov)
        return None
    except Exception as exc:  # network, auth, rate limit: class name only
        budget.record_latency((time.perf_counter() - started) * 1000, name)
        _record_error(budget, exc)
        logger.warning("llm_answer outcome=error %s", _error_log(exc))
        _fail(budget, llm_resilience.classify(exc), prov)
        return None
    latency_ms = (time.perf_counter() - started) * 1000
    budget.record_latency(latency_ms, name)
    _add_usage(budget, tokens_in, tokens_out, cached, completion)
    _observe_usage(messages, raw, name)
    if prov is None:
        llm_resilience.breaker(name).record_success()
    logger.info("llm_answer latency_ms=%.0f tokens_in=%d tokens_out=%d", latency_ms, tokens_in, tokens_out)
    return raw


async def generate_json(messages: list[dict[str, str]], budget: LLMBudget) -> str | None:
    """One more JSON-mode call on the same per-turn budget (used by the conversation answerer).

    Returns the raw model text, or None after a timeout, an API error or an exhausted budget
    (``budget.error_class`` says which). Parsing and validation are the caller's job. Latency and
    token totals accumulate on ``budget`` like for ``decide``; logs hold metadata only. A turn that
    already moved to the fallback provider stays on it; otherwise a failure here gets the same one-time
    fallback retry as ``decide``."""
    gate = _primary_gate(budget)
    if gate == "skip":
        return None
    raw = await _generate_on(messages, budget, gate)
    if raw is None and gate is None and _switch_to_fallback(budget):
        raw = await _generate_on(messages, budget, budget.use_provider)
        if raw is not None:
            _clear_error(budget)
    return raw
