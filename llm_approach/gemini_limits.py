"""Gemini free-tier limits, enforced in one place (used by ``providers`` for every Gemini request).

gemini-3.5-flash-lite free tier (AI Studio): 15 requests per minute (RPM), 250K tokens per minute (TPM),
500 requests per day (RPD), no daily token limit. RPD is the binding one. Three guards sit in front of
every request, shared by live chat and the eval runners (same process: one limiter; other processes: one
counter file):

1. Request cap over a rolling 24-hour window (``LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET``, default 450 of 500).
   The counter file ``eval/results/gemini_requests.json`` holds only timestamps, request counts and token counts;
   entries 24 hours old or older are dropped on every read. Every request counts, retries and failed calls
   included (a failed call keeps its estimated tokens). The same file holds the 24-hour token total, so chat
   and the eval runners share both numbers. At the cap no request starts. Google's real reset boundary is not
   known: with ``LLM_APPROACH_GEMINI_DAY_RESET`` (a UTC time such as 07:00) the window becomes a fixed daily
   boundary at that time instead; unset, the rolling window is used.
2. Sliding 60-second window: at most 12 requests and 200K tokens (20% under 15 and 250K). The limiter waits
   instead of exceeding either. The token estimate for the next call is the previous call's count, or the
   prompt-size estimate before any call.
3. On a 429: wait the retry delay named in the error (at most 60 s) and retry the same call once. A 429 that
   says the daily quota is used up is never retried.
4. A turn deadline (``turn_deadline``, set by the pipeline at the start of the LLM step, about 20 s): before any
   sleep (window wait, 429 retry delay) the limiter checks that the wait still fits. If not it raises
   ``WaitTooLong`` instead of sleeping, so the rule-based fallback answers at once. A request that was never
   sent is not counted against the daily cap.

Nothing secret is stored or logged: no key, no prompt text, no reply text.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable

from llm_approach import config

logger = logging.getLogger(__name__)

RPM_LIMIT, TPM_LIMIT, RPD_LIMIT = 15, 250_000, 500  # what Google lists for the free tier (documentation only)
WINDOW_S = 60.0
WINDOW_MAX_REQUESTS = 12  # 15 RPM minus a 20% margin
WINDOW_MAX_TOKENS = 200_000  # 250K TPM minus a 20% margin
DEFAULT_DAILY_REQUEST_BUDGET = 450  # 500 RPD minus a safety margin
MAX_RETRY_WAIT_S = 60.0
DEFAULT_RETRY_WAIT_S = 60.0  # a 429 that names no delay
EST_COMPLETION_TOKENS = 250
USAGE_WINDOW_S = 24 * 3600.0  # the rolling window for requests and tokens
MAX_SEED_REQUESTS = RPD_LIMIT

DEFAULT_TURN_DEADLINE_S = 20.0  # v2: how long a turn may spend in the LLM step, waits included
TURN_DEADLINE_ENV = "LLM_APPROACH_GEMINI_TURN_DEADLINE_S"

COUNTER_PATH = Path(__file__).resolve().parent / "eval" / "results" / "gemini_requests.json"  # git-ignored; the default


def counter_path() -> Path:
    """The counter file: ``LLM_APPROACH_GEMINI_COUNTER_PATH`` when set, else the default inside the package.
    On a server point it at a writable volume (a read-only or throwaway filesystem would lose the daily count)."""
    value = os.getenv(config.COUNTER_PATH_ENV, "").strip()
    return Path(value) if value else COUNTER_PATH

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]
Wall = Callable[[], dt.datetime]  # timezone-aware "now"


class GeminiLimitError(Exception):
    """Base of the errors raised here. They carry HTTP code 429 / RESOURCE_EXHAUSTED, so the rest of the code
    treats them like a provider rate-limit error (the pipeline's normal rate-limit fallback)."""

    code = 429
    status = "RESOURCE_EXHAUSTED"
    kind = "rate_limit"
    local = False  # True: raised by this limiter without any request reaching Google (not a provider failure)


class RequestCapReached(GeminiLimitError):
    kind = "request_cap"
    local = True

    def __init__(self) -> None:
        super().__init__("local daily Gemini request cap reached")


class DailyQuotaExhausted(GeminiLimitError):
    kind = "daily_quota"

    def __init__(self, original: BaseException | None = None) -> None:
        super().__init__(f"Gemini reports the daily request quota is exhausted: {original}" if original else
                         "Gemini reports the daily request quota is exhausted")


class WaitTooLong(GeminiLimitError):
    """The wait the limiter needs (window or 429 retry delay) would pass the turn's deadline, so it did not sleep."""

    kind = "wait_too_long"
    local = True

    def __init__(self, wait: float, remaining: float) -> None:
        super().__init__(f"Gemini limiter would wait {wait:.1f}s but only {max(0.0, remaining):.1f}s of the turn remain")


class RateLimitRetried(GeminiLimitError):
    """The call was already retried once after a 429 and got a second one."""

    kind = "rate_limit_retried"

    def __init__(self, original: BaseException | None = None) -> None:
        super().__init__(str(original) if original else "Gemini rate limit (429) persisted after one retry")


def turn_deadline_s() -> float:
    """Seconds a turn may spend in the LLM step (env override; an unreadable or non-positive value uses the default)."""
    try:
        value = float(os.getenv(TURN_DEADLINE_ENV, "").strip())
    except ValueError:
        return DEFAULT_TURN_DEADLINE_S
    return value if value > 0 else DEFAULT_TURN_DEADLINE_S


# The time (on the limiter's clock, monotonic seconds) by which the current turn's LLM step must be done, or None.
# A context variable, so the pipeline sets it once and the limiter sees it without any signature change.
_turn_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("gemini_turn_deadline", default=None)


def start_turn_deadline(seconds: float | None = None, now: float | None = None) -> None:
    """Start the deadline for the current turn (``seconds`` default: ``turn_deadline_s()``)."""
    now = get_limiter().clock() if now is None else now
    _turn_deadline.set(now + (turn_deadline_s() if seconds is None else seconds))


def clear_turn_deadline() -> None:
    _turn_deadline.set(None)


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ---- request and token counter (rolling 24 h, or a fixed daily boundary) -------------------------------

_bad_reset_warned = False
_write_warned = False
# Fixed wording, no path: the log must not reveal the layout of the host.
WRITE_FAILED_WARNING = ("The Gemini request counter file cannot be written: the local daily request cap is not "
                        "enforced across restarts. Set LLM_APPROACH_GEMINI_COUNTER_PATH to a writable file.")


def _day_reset() -> dt.time | None:
    """The fixed daily boundary (UTC) from the environment, or None for the rolling window. An unusable value
    falls back to the rolling window with one warning (``config.setup_problem`` reports it for live runs)."""
    global _bad_reset_warned
    try:
        return config.day_reset()
    except ValueError as exc:
        if not _bad_reset_warned:
            _bad_reset_warned = True
            logger.warning("%s; using the rolling 24-hour window", exc)
        return None


class UsageCounter:
    """A tiny file ``{"version": 2, "entries": [{"id", "t", "n", "tokens"}]}``: ``t`` is a UTC epoch time, ``n`` the
    requests and ``tokens`` the tokens of that entry. Numbers only: no text, no keys. An entry counts while it is
    inside the window: the last 24 hours (it drops out once it is 24 hours old), or, with a day reset set, since
    the latest reset boundary."""

    def __init__(self, path: Path | None = None, wall: Wall = _utc_now) -> None:
        self._path = path
        self.wall = wall
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path or counter_path()

    # -- window

    def window_start(self, now: dt.datetime | None = None) -> dt.datetime:
        """Entries strictly after this time count (rolling), or at/after it (fixed boundary)."""
        now = now or self.wall()
        reset = _day_reset()
        if reset is None:
            return now - dt.timedelta(seconds=USAGE_WINDOW_S)
        utc = now.astimezone(dt.timezone.utc)
        boundary = dt.datetime.combine(utc.date(), reset, tzinfo=dt.timezone.utc)
        return boundary if boundary <= utc else boundary - dt.timedelta(days=1)

    def _in_window(self, entries: list[dict]) -> list[dict]:
        start = self.window_start().timestamp()
        fixed = _day_reset() is not None
        return [e for e in entries if e["t"] > start or (fixed and e["t"] == start)]

    # -- file

    def _load(self) -> list[dict]:
        """Every well-formed entry in the file, converting the old ``{"date", "count"}`` format when found."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, dict):
            return []
        if "entries" not in data:
            count = data.get("count")
            if "date" in data and isinstance(count, int) and not isinstance(count, bool) and count > 0:
                # old format: the real times are unknown, so the old count is treated as used right now
                logger.warning("Gemini counter file was in the old date+count format: counted %d requests as used "
                               "now (conservative) and converted it to the rolling-window format", count)
                entries = [{"id": uuid.uuid4().hex[:8], "t": self.wall().timestamp(), "n": count, "tokens": 0}]
                self._write(entries)
                return entries
            return []
        out = []
        for e in data["entries"] if isinstance(data["entries"], list) else []:
            try:
                t, n, tokens = float(e["t"]), int(e["n"]), int(e.get("tokens", 0))
            except (KeyError, TypeError, ValueError):
                continue
            if n >= 0 and tokens >= 0:
                out.append({"id": str(e.get("id", "")), "t": t, "n": n, "tokens": tokens})
        return out

    def _write(self, entries: list[dict]) -> None:
        """Write the file atomically: a private temp file in the same folder (unique per call, so two processes never
        share one), flushed to disk, then renamed over the real file. A failure never blocks a call."""
        global _write_warned
        keep = self.wall().timestamp() - USAGE_WINDOW_S
        entries = [e for e in entries if e["t"] > keep]
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"version": 2, "entries": entries}))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            if not _write_warned:
                _write_warned = True
                logger.warning(WRITE_FAILED_WARNING)

    def writable(self) -> bool:
        """Whether the counter file can be created and replaced (a probe file in its folder is written and removed).
        For the startup check; never raises."""
        probe = self.path.with_name(f"{self.path.name}.{os.getpid()}.probe")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return True
        except OSError:
            return False

    # -- reads

    def used_requests(self) -> int:
        with self._lock:
            return sum(e["n"] for e in self._in_window(self._load()))

    used_today = used_requests  # the old name

    def used_tokens(self) -> int:
        with self._lock:
            return sum(e["tokens"] for e in self._in_window(self._load()))

    def free_up_time(self) -> dt.datetime | None:
        """When capacity starts to free up: 24 hours after the oldest counted entry (rolling), or the next reset
        boundary (fixed). None when nothing is counted."""
        with self._lock:
            live = self._in_window(self._load())
        if not live:
            return None
        if _day_reset() is None:
            return dt.datetime.fromtimestamp(min(e["t"] for e in live) + USAGE_WINDOW_S, dt.timezone.utc)
        return self.window_start() + dt.timedelta(days=1)

    # -- writes

    def add(self, n: int = 1, tokens: int = 0, at: dt.datetime | None = None, eid: str | None = None) -> int:
        """Count ``n`` more requests (and ``tokens``) at ``at`` (default now); returns the requests now in the window."""
        with self._lock:
            entries = self._load()
            entries.append({"id": eid or uuid.uuid4().hex[:8], "t": (at or self.wall()).timestamp(), "n": n,
                            "tokens": max(0, int(tokens))})
            self._write(entries)
            return sum(e["n"] for e in self._in_window(entries))

    def set_tokens(self, eid: str, tokens: int) -> None:
        """Replace the estimated tokens of entry ``eid`` with the real count."""
        with self._lock:
            entries = self._load()
            for e in entries:
                if e["id"] == eid:
                    e["tokens"] = max(0, int(tokens))
                    self._write(entries)
                    return


DailyCounter = UsageCounter  # the old name


# ---- error parsing ----------------------------------------------------------------------

_DELAY_RES = (
    re.compile(r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s", re.I),
    re.compile(r"retry_?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", re.I),
)


def is_local_limit(exc: BaseException) -> bool:
    """An error this limiter raised itself (our own cap or wait bound): Google was never asked, so it says nothing
    about the provider's health."""
    return isinstance(exc, GeminiLimitError) and exc.local


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
    """Request cap + sliding window + single 429 retry. ``clock`` (monotonic seconds), ``sleep`` and ``wall``
    can be faked in tests."""

    def __init__(
        self, counter: UsageCounter | None = None, clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep, wall: Wall = _utc_now,
    ) -> None:
        self.counter = counter or UsageCounter(wall=wall)
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

    async def _sleep_within_deadline(self, wait: float) -> None:
        """Sleep ``wait`` seconds, or raise ``WaitTooLong`` (without sleeping) when that would pass the turn deadline."""
        deadline = _turn_deadline.get()
        if deadline is not None:
            remaining = deadline - self.clock()
            if wait > remaining:
                raise WaitTooLong(wait, remaining)
        await self.sleep(wait)

    def remaining_requests(self, budget: int) -> int:
        return max(0, budget - self.counter.used_requests())

    # -- one guarded request

    async def _begin(self, budget: int, prompt_estimate: int) -> tuple[list[float], str]:
        """Cap check, window wait, then count the request. Returns the window entry and the counter entry id,
        to fix up afterwards."""
        if self.counter.used_requests() >= budget:
            raise RequestCapReached()
        estimate = self._last_tokens if self._last_tokens is not None else prompt_estimate
        wait = self._wait_needed(estimate)
        while wait > 0:
            await self._sleep_within_deadline(wait)
            wait = self._wait_needed(estimate)
        if self.counter.used_requests() >= budget:  # another task may have used the last request while we waited
            raise RequestCapReached()
        eid = uuid.uuid4().hex[:8]
        self.counter.add(1, tokens=estimate, eid=eid)  # counted before it is sent: failed calls and retries count too
        entry = [self.clock(), float(estimate)]
        self._window.append(entry)
        return entry, eid

    async def call(self, send: Callable[[], Awaitable[Any]], *, budget: int, prompt_estimate: int,
                   tokens_of: Callable[[Any], int | None] = lambda _r: None) -> Any:
        """Run ``send()`` under the guards. ``tokens_of(result)`` gives the call's real token count."""
        for attempt in (1, 2):
            entry, eid = await self._begin(budget, prompt_estimate)
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
                await self._sleep_within_deadline(min(DEFAULT_RETRY_WAIT_S if delay is None else delay, MAX_RETRY_WAIT_S))
                continue
            used = tokens_of(result)
            if used:
                entry[1] = float(used)
                self._last_tokens = int(used)
                self.counter.set_tokens(eid, int(used))
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
