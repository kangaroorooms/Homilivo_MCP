"""LLM failure handling: failure kinds, the circuit breaker and the fixed polite reply.

``llm_client`` calls into this module; nothing here talks to a provider. Logs and counters carry fixed
words only (kind, provider name), never user text, replies or keys.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from llm_approach import config, gemini_limits, observability

logger = logging.getLogger(__name__)

LLM_UNAVAILABLE_REPLY = "Sorry, I'm unavailable right now. Please try again shortly."

BAD_JSON = "bad_json"
# Failure kinds the fallback provider is tried for (an invalid key or a request error is not one of them).
FALLBACK_KINDS = frozenset({"timeout", "rate_limit", "daily_quota", "server_error", BAD_JSON})


def _http_code(exc: BaseException) -> int | None:
    for obj in (exc, getattr(exc, "response", None)):
        for attr in ("status_code", "code"):
            value = getattr(obj, attr, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def classify(exc: BaseException) -> str:
    """timeout | rate_limit | daily_quota | server_error | auth | other."""
    if isinstance(exc, gemini_limits.GeminiLimitError):
        return "rate_limit" if exc.kind in ("rate_limit_retried", "wait_too_long") else "daily_quota"
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or "Timeout" in name:  # asyncio.TimeoutError is TimeoutError
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "server_error"  # the provider cannot be reached
    if gemini_limits.is_rate_limit(exc):
        return "daily_quota" if gemini_limits.is_daily_quota(exc) else "rate_limit"
    code = _http_code(exc)
    status = getattr(exc, "status", None)
    if (code is not None and 500 <= code <= 599) or status in ("INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED"):
        return "server_error"
    if name in ("InternalServerError", "APIConnectionError", "ServiceUnavailableError"):
        return "server_error"
    if code in (401, 403) or isinstance(exc, KeyError) or name in ("AuthenticationError", "PermissionDeniedError"):
        return "auth"
    if code == 400 and "api key" in str(exc).lower():  # Gemini answers an invalid key with 400 INVALID_ARGUMENT
        return "auth"
    return "other"


RETRY_5XX_CODES = frozenset({500, 502, 503, 504})


def is_retryable_5xx(exc: BaseException) -> bool:
    """A 500, 502, 503 or 504 answer from the provider (never a 4xx, 429, auth error, timeout or lost connection)."""
    if isinstance(exc, (TimeoutError, ConnectionError)) or isinstance(exc, gemini_limits.GeminiLimitError):
        return False
    return _http_code(exc) in RETRY_5XX_CODES


class CircuitBreaker:
    """Closed until ``failures`` calls in a row fail; then open for ``cooldown_s``. After the cooldown one trial
    call is let through (half-open): a success closes the breaker, a failure opens it for another cooldown.
    A trial that never reports back is replaced after one more cooldown."""

    def __init__(self, failures: Callable[[], int] = config.breaker_failures,
                 cooldown_s: Callable[[], float] = config.breaker_cooldown_s,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._failures_limit, self._cooldown, self._clock = failures, cooldown_s, clock
        self._lock = threading.Lock()
        self.consecutive = 0
        self.opened_at: float | None = None
        self.trial_at: float | None = None

    def allow(self) -> bool:
        """May the primary be called now? True when closed, or for the single trial after the cooldown."""
        with self._lock:
            if self.opened_at is None:
                return True
            now = self._clock()
            if now - self.opened_at < self._cooldown():
                return False
            if self.trial_at is not None and now - self.trial_at < self._cooldown():
                return False  # a trial is already out
            self.trial_at = now
            return True

    def is_open(self) -> bool:
        with self._lock:
            return self.opened_at is not None

    def record_success(self) -> None:
        with self._lock:
            was_open = self.opened_at is not None
            self.consecutive, self.opened_at, self.trial_at = 0, None, None
        if was_open:
            logger.warning("llm circuit breaker closed")
            observability.record_llm_event("breaker_close")

    def record_failure(self) -> None:
        with self._lock:
            self.consecutive += 1
            if self.opened_at is None and self.consecutive < self._failures_limit():
                return
            reopened = self.opened_at is not None
            self.opened_at, self.trial_at = self._clock(), None
            count = self.consecutive
        logger.warning("llm circuit breaker %s after %d consecutive failures",
                       "reopened (trial failed)" if reopened else "opened", count)
        observability.record_llm_event("breaker_open")


_breakers: dict[str, CircuitBreaker] = {}


def breaker(provider: str) -> CircuitBreaker:
    """The breaker of one primary provider (a process-wide singleton per provider name)."""
    if provider not in _breakers:
        _breakers[provider] = CircuitBreaker()
    return _breakers[provider]


def reset() -> None:
    """Forget every breaker. Tests only."""
    _breakers.clear()


def note_failure(kind: str, provider: str, *, primary: bool) -> None:
    """Count and log one failure (a fixed kind and the provider name only) and, for the primary, feed the breaker."""
    observability.record_llm_failure(kind)
    logger.warning("llm failure kind=%s provider=%s role=%s", kind, provider, "primary" if primary else "fallback")
    if primary:
        breaker(provider).record_failure()


def note_switch(from_provider: str, to_provider: str, reason: str) -> None:
    logger.warning("llm fallback switch from=%s to=%s reason=%s", from_provider, to_provider, reason)
    observability.record_llm_event("fallback_switch")
