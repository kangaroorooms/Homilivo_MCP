"""Gemini free-tier limits, enforced in one place (used by ``providers`` for every Gemini request).

gemini-3.5-flash-lite free tier (AI Studio): 15 requests per minute (RPM), 250K tokens per minute (TPM),
500 requests per day (RPD), no daily token limit. RPD is the binding one. Three guards sit in front of
every request, shared by live chat and the eval runners (same process: one limiter; other processes: one
counter file):

1. Daily request cap (``LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET``, default 450 of 500). The counter file
   ``eval/results/gemini_requests.json`` holds only a date and a count. The day ends at midnight US Pacific
   time. Every request counts, retries and failed calls included. At the cap no request starts.
2. Sliding 60-second window: at most 12 requests and 200K tokens (20% under 15 and 250K). The limiter waits
   instead of exceeding either. The token estimate for the next call is the previous call's count, or the
   prompt-size estimate before any call.
3. On a 429: wait the retry delay named in the error (at most 60 s) and retry the same call once. A 429 that
   says the daily quota is used up is never retried.

Nothing secret is stored or logged: no key, no prompt text, no reply text.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable

RPM_LIMIT, TPM_LIMIT, RPD_LIMIT = 15, 250_000, 500  # what Google lists for the free tier (documentation only)
WINDOW_S = 60.0
WINDOW_MAX_REQUESTS = 12  # 15 RPM minus a 20% margin
WINDOW_MAX_TOKENS = 200_000  # 250K TPM minus a 20% margin
DEFAULT_DAILY_REQUEST_BUDGET = 450  # 500 RPD minus a safety margin
MAX_RETRY_WAIT_S = 60.0
DEFAULT_RETRY_WAIT_S = 60.0  # a 429 that names no delay
EST_COMPLETION_TOKENS = 250

COUNTER_PATH = Path(__file__).resolve().parent / "eval" / "results" / "gemini_requests.json"  # git-ignored

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]
Wall = Callable[[], dt.datetime]  # timezone-aware "now"


class GeminiLimitError(Exception):
    """Base of the errors raised here. They carry HTTP code 429 / RESOURCE_EXHAUSTED, so the rest of the code
    treats them like a provider rate-limit error (the pipeline's normal rate-limit fallback)."""

    code = 429
    status = "RESOURCE_EXHAUSTED"
    kind = "rate_limit"


class RequestCapReached(GeminiLimitError):
    kind = "request_cap"

    def __init__(self) -> None:
        super().__init__("local daily Gemini request cap reached")


class DailyQuotaExhausted(GeminiLimitError):
    kind = "daily_quota"

    def __init__(self, original: BaseException | None = None) -> None:
        super().__init__(f"Gemini reports the daily request quota is exhausted: {original}" if original else
                         "Gemini reports the daily request quota is exhausted")


class RateLimitRetried(GeminiLimitError):
    """The call was already retried once after a 429 and got a second one."""

    kind = "rate_limit_retried"

    def __init__(self, original: BaseException | None = None) -> None:
        super().__init__(str(original) if original else "Gemini rate limit (429) persisted after one retry")


# ---- US Pacific day boundary ---------------------------------------------------------


def _nth_sunday(year: int, month: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    return first + dt.timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))


def pacific_date(now: dt.datetime) -> str:
    """ISO date in US Pacific time of an aware ``now``. DST: second Sunday of March 02:00 to first Sunday of
    November 02:00 local time (10:00 and 09:00 UTC). Computed here so no tz database is needed."""
    utc = now.astimezone(dt.timezone.utc)
    start = dt.datetime.combine(_nth_sunday(utc.year, 3, 2), dt.time(10), tzinfo=dt.timezone.utc)
    end = dt.datetime.combine(_nth_sunday(utc.year, 11, 1), dt.time(9), tzinfo=dt.timezone.utc)
    offset = -7 if start <= utc < end else -8
    return (utc + dt.timedelta(hours=offset)).date().isoformat()


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ---- daily request counter -------------------------------------------------------------


class DailyCounter:
    """A tiny file ``{"date": "YYYY-MM-DD", "count": n}`` (Pacific date). A new day starts at 0."""

    def __init__(self, path: Path | None = None, wall: Wall = _utc_now) -> None:
        self._path = path
        self.wall = wall
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path or COUNTER_PATH

    def _read(self) -> int:
        today = pacific_date(self.wall())
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("date") == today and isinstance(data.get("count"), int) and data["count"] >= 0:
                return data["count"]
        except (OSError, ValueError, AttributeError):
            pass
        return 0

    def used_today(self) -> int:
        with self._lock:
            return self._read()

    def add(self, n: int = 1) -> int:
        """Count ``n`` more requests today; returns the new total. A write failure never blocks a call."""
        with self._lock:
            count = self._read() + n
            tmp = self.path.with_suffix(".tmp")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps({"date": pacific_date(self.wall()), "count": count}), encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError:
                pass
            return count


# ---- error parsing ----------------------------------------------------------------------

_DELAY_RES = (
    re.compile(r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s", re.I),
    re.compile(r"retry_?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", re.I),
)


def is_rate_limit(exc: BaseException) -> bool:
    if type(exc).__name__ == "RateLimitError":
        return True
    resp = getattr(exc, "response", None)
    return (getattr(exc, "code", None) == 429 or getattr(exc, "status", None) == "RESOURCE_EXHAUSTED"
            or getattr(exc, "status_code", None) == 429 or getattr(resp, "status_code", None) == 429)


def _error_text(exc: BaseException) -> str:
    return f"{exc} {getattr(exc, 'details', '')}"


def is_daily_quota(exc: BaseException) -> bool:
    """A 429 whose text names the per-day quota (``...RequestsPerDay...`` or "daily")."""
    text = _error_text(exc).lower()
    return "perday" in text or "per day" in text or "daily" in text


def retry_delay(exc: BaseException) -> float | None:
    """Seconds the error asks us to wait ("retry in 23.4s" or ``retryDelay: 23s``), else None."""
    text = _error_text(exc)
    for pattern in _DELAY_RES:
        m = pattern.search(text)
        if m:
            return float(m.group(1))
    return None


# ---- the limiter --------------------------------------------------------------------------


class GeminiLimiter:
    """Daily cap + sliding window + single 429 retry. ``clock`` (monotonic seconds), ``sleep`` and ``wall``
    can be faked in tests."""

    def __init__(
        self, counter: DailyCounter | None = None, clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep, wall: Wall = _utc_now,
    ) -> None:
        self.counter = counter or DailyCounter(wall=wall)
        self.clock, self.sleep = clock, sleep
        self._window: deque[list[float]] = deque()  # [start time, tokens] of each request in the last 60 s
        self._last_tokens: int | None = None
        self.quota_exhausted = False  # set when Gemini said the daily quota is used up; runs stop on it

    # -- helpers

    def requests_in_window(self) -> int:
        self._trim()
        return len(self._window)

    def tokens_in_window(self) -> int:
        self._trim()
        return int(sum(t for _, t in self._window))

    def _trim(self) -> None:
        now = self.clock()
        while self._window and now - self._window[0][0] >= WINDOW_S:
            self._window.popleft()

    def _wait_needed(self, estimate: int) -> float:
        """Seconds until the next call fits in the window (0 when it fits now): wait for the oldest entries
        to leave the window, one by one, until the request and token margins both hold."""
        self._trim()
        now = self.clock()
        entries = list(self._window)
        requests, tokens = len(entries), sum(t for _, t in entries)
        wait, i = 0.0, 0
        while requests + 1 > WINDOW_MAX_REQUESTS or tokens + estimate > WINDOW_MAX_TOKENS:
            if i >= len(entries):  # the estimate alone is over the margin: nothing left to wait for
                break
            started, spent = entries[i]
            i += 1
            requests, tokens = requests - 1, tokens - spent
            wait = started + WINDOW_S - now
        return max(0.0, wait)

    def remaining_requests(self, budget: int) -> int:
        return max(0, budget - self.counter.used_today())

    # -- one guarded request

    async def _begin(self, budget: int, prompt_estimate: int) -> list[float]:
        """Cap check, window wait, then count the request. Returns the window entry to fix up afterwards."""
        if self.counter.used_today() >= budget:
            raise RequestCapReached()
        estimate = self._last_tokens if self._last_tokens is not None else prompt_estimate
        wait = self._wait_needed(estimate)
        while wait > 0:
            await self.sleep(wait)
            wait = self._wait_needed(estimate)
        if self.counter.used_today() >= budget:  # another task may have used the last request while we waited
            raise RequestCapReached()
        self.counter.add(1)  # counted before it is sent: failed calls and retries count too
        entry = [self.clock(), float(estimate)]
        self._window.append(entry)
        return entry

    async def call(self, send: Callable[[], Awaitable[Any]], *, budget: int, prompt_estimate: int,
                   tokens_of: Callable[[Any], int | None] = lambda _r: None) -> Any:
        """Run ``send()`` under the guards. ``tokens_of(result)`` gives the call's real token count."""
        for attempt in (1, 2):
            entry = await self._begin(budget, prompt_estimate)
            try:
                result = await send()
            except Exception as exc:
                if not is_rate_limit(exc) or isinstance(exc, GeminiLimitError):
                    raise
                if is_daily_quota(exc):
                    self.quota_exhausted = True
                    raise DailyQuotaExhausted(exc) from exc
                if attempt == 2:
                    raise RateLimitRetried(exc) from exc
                delay = retry_delay(exc)
                await self.sleep(min(DEFAULT_RETRY_WAIT_S if delay is None else delay, MAX_RETRY_WAIT_S))
                continue
            used = tokens_of(result)
            if used:
                entry[1] = float(used)
                self._last_tokens = int(used)
            return result
        raise RateLimitRetried()  # not reached


_limiter: GeminiLimiter | None = None


def get_limiter() -> GeminiLimiter:
    """The process-wide limiter (live chat and the eval runner share it)."""
    global _limiter
    if _limiter is None:
        _limiter = GeminiLimiter()
    return _limiter


def set_limiter(limiter: GeminiLimiter | None) -> None:
    """Install (or, with None, reset) the process-wide limiter. Tests only."""
    global _limiter
    _limiter = limiter


def estimate_prompt_tokens(chars: int) -> int:
    """Rough tokens for one call from the prompt size (about 4 characters per token) plus the reply."""
    return chars // 4 + EST_COMPLETION_TOKENS
