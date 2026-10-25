# Parts copied from app/session_store.py at commit 7dbcba3 (copied, not imported): the key helper and the
# idle-expiry and LRU-cap methods of RedisSessionStore (_key, _evict_expired, _enforce_cap).
# Changes: key prefixes (marked '# v2:'); everything else in this file is new for v2 (async turn with a
# per-session Redis lock, versioned JSON state, fail-open handling).
# Drift is checked by llm_approach/tests/test_copy_drift.py.
# ==== end of v2 provenance header ====
"""Session stores for the LLM approach: the protocol, the Redis store and the factory.

The default store is the in-memory ``state.SessionStore`` (unchanged). ``STATE_BACKEND=redis`` selects
``RedisSessionStore``, so sessions survive a restart and work behind several servers.

Redis layout (v2 prefix, so v2 never reads or deletes app/ sessions, which use ``homilivo:session:``):
- ``homilivo:v2:session:<id>``  the state as JSON (``LLMConversationState.to_dict``), key TTL = idle TTL
- ``homilivo:v2:sessions:lru``  sorted set id -> last access, shared by every server (the LRU cap)
- ``homilivo:v2:lock:<id>``     the per-session turn lock: ``SET NX PX`` with a random token
- ``homilivo:v2:reset:<id>``    set by ``reset`` while a turn holds the lock, so that turn does not save the
                                session back after the user cleared it

A turn: take the in-process asyncio lock, then the Redis lock (both bounded by one wait), load, run, save,
release. While the turn runs a background task refreshes the lock, so a long turn (a Gemini 429 wait is up to
60 s) never loses it; if the server dies the lock lapses after its TTL. The refresh stops after
``LOCK_MAX_HOLD_S``, so a hung turn cannot hold a session for ever.

Fail-open: Redis unreachable -> the turn runs on a fresh in-memory state, nothing is saved (a fresh state
must never overwrite the real session if Redis comes back mid-turn), an error is logged and counted. A
corrupt, wrong-version or unreadable blob is replaced by a fresh session with a WARNING that holds no
contents. Nothing here logs a URL, a key or a message.

The Redis client must be built with ``decode_responses=True`` (``redis.Redis.from_url`` is, below).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncContextManager, Callable, Protocol

from llm_approach import config, observability, turn_context
from llm_approach.state import (
    ERROR_REPLY,
    LLMConversationState,
    SessionBusyError,
    SessionStore,
    StateDecodeError,
    normalize_session_id,
)

try:
    import redis as redis_lib
except ImportError:  # pragma: no cover - redis is in requirements.txt
    redis_lib = None  # STATE_BACKEND=redis raises a clear error if reached

logger = logging.getLogger(__name__)

_LOCK_POLL_S = 0.05


class SessionStoreBackend(Protocol):
    """What the pipeline, the server and the CLI need from a store. ``state.SessionStore`` (memory) and
    ``RedisSessionStore`` both implement it. ``get_or_create`` is the protocol's ``get``."""

    is_shared: bool  # True: state lives outside this process (mutating a returned state saves nothing)

    def get_or_create(self, session_id: str) -> LLMConversationState: ...
    def peek(self, session_id: str) -> LLMConversationState | None: ...
    def save(self, session_id: str, state: LLMConversationState) -> None: ...
    def reset(self, session_id: str) -> bool: ...
    def count(self) -> int: ...
    def turn(self, session_id: str, user_message: str | None = None) -> AsyncContextManager[LLMConversationState]: ...


@dataclass
class _Local:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active: int = 0  # turns running or waiting on this server


@dataclass
class _Lease:
    token: str | None = None
    lost: bool = False  # a refresh found the lock gone or taken over
    capped: bool = False  # the max-hold cap stopped the refresh


def _s(value) -> str | None:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class RedisSessionStore:
    """Redis-backed sessions with the same interface as ``state.SessionStore``."""

    is_shared = True

    _KEY_PREFIX = "homilivo:v2:session:"  # v2: was: _KEY_PREFIX = "homilivo:session:"
    _LRU_KEY = "homilivo:v2:sessions:lru"  # v2: was: _LRU_KEY = "homilivo:sessions:lru"
    _LOCK_PREFIX = "homilivo:v2:lock:"
    _RESET_PREFIX = "homilivo:v2:reset:"

    def __init__(
        self,
        *,
        ttl_seconds: float | None = None,
        max_sessions: int | None = None,
        redis_client=None,
        url: str | None = None,
        clock: Callable[[], float] = time.time,
        lock_ttl_ms: int | None = None,
        lock_wait_s: float | None = None,
        lock_max_hold_s: float | None = None,
        sleep: Callable[[float], "asyncio.Future"] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,  # v2: injectable for the max-hold test
    ) -> None:
        if redis_client is None and redis_lib is None:
            raise RuntimeError("STATE_BACKEND=redis requires the 'redis' package (pip install -r requirements.txt).")
        self._ttl = config.SESSION_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        self._max = config.SESSION_MAX_COUNT if max_sessions is None else max_sessions
        if redis_client is None:
            timeout = config.redis_timeout_s()
            redis_client = redis_lib.Redis.from_url(
                url or config.redis_url(), decode_responses=True,
                socket_connect_timeout=timeout, socket_timeout=timeout,
            )
        self._redis = redis_client
        self._clock = clock
        self._lock_ttl_ms = config.session_lock_ttl_ms() if lock_ttl_ms is None else int(lock_ttl_ms)
        self._lock_wait_s = config.session_lock_wait_s() if lock_wait_s is None else lock_wait_s
        self._lock_max_hold_s = config.session_lock_max_hold_s() if lock_max_hold_s is None else lock_max_hold_s
        self._sleep = sleep
        self._mono = monotonic
        self._locals: dict[str, _Local] = {}

    # ---- copied from app/session_store.py -----------------------------------

    def _key(self, session_id: str) -> str:
        return f"{self._KEY_PREFIX}{session_id}"

    def _evict_expired(self, now: float) -> None:
        stale_ids = self._redis.zrangebyscore(self._LRU_KEY, "-inf", now - self._ttl)
        if not stale_ids:
            return
        pipe = self._redis.pipeline()
        for sid in stale_ids:
            pipe.delete(self._key(sid))
        pipe.zrem(self._LRU_KEY, *stale_ids)
        pipe.execute()

    def _enforce_cap(self) -> None:
        count = self._redis.zcard(self._LRU_KEY)
        if count <= self._max:
            return
        oldest = self._redis.zrange(self._LRU_KEY, 0, count - self._max - 1)
        if not oldest:
            return
        pipe = self._redis.pipeline()
        for sid in oldest:
            pipe.delete(self._key(sid))
        pipe.zrem(self._LRU_KEY, *oldest)
        pipe.execute()

    # ---- helpers ------------------------------------------------------------------

    def _lock_key(self, session_id: str) -> str:
        return f"{self._LOCK_PREFIX}{session_id}"

    def _reset_key(self, session_id: str) -> str:
        return f"{self._RESET_PREFIX}{session_id}"

    @staticmethod
    def _event(event: str) -> None:
        try:
            observability.record_session_store_event(event)
        except Exception:  # noqa: BLE001 - counting must never break a turn
            pass

    def _redis_down(self, op: str, exc: BaseException) -> None:
        """Log (class name only: the message of a connection error can hold the URL) and count."""
        logger.error("session store: redis unavailable during %s: %s", op, type(exc).__name__)
        self._event("redis_unavailable")

    def _decode(self, session_id: str, raw) -> LLMConversationState | None:
        """The state in ``raw``, or None (with a WARNING and a counter, no contents) when it is unreadable."""
        sid = turn_context.hash_session_id(session_id)
        try:
            return LLMConversationState.from_dict(json.loads(raw))
        except StateDecodeError as exc:
            wrong = str(exc) == "wrong_schema_version"
            logger.warning("session store: stored state discarded (%s) session=%s", exc, sid)
            self._event("state_wrong_version" if wrong else "state_corrupt")
        except (ValueError, TypeError, RecursionError) as exc:  # bad JSON, not text, nested too deep
            logger.warning("session store: stored state discarded (unreadable %s) session=%s", type(exc).__name__, sid)
            self._event("state_corrupt")
        return None

    def _undecodable(self, session_id: str) -> None:
        logger.warning("session store: stored state discarded (not text) session=%s", turn_context.hash_session_id(session_id))
        self._event("state_corrupt")

    def _fresh(self) -> LLMConversationState:
        return LLMConversationState(updated_at=time.time())

    # ---- blocking Redis work (run in a thread from async code) ----------------------

    def _load_blocking(self, session_id: str) -> str | None:
        """Raw blob or None. Mirrors ``get`` of app/: evict, read, refresh the key TTL and the LRU score."""
        now = self._clock()
        self._evict_expired(now)
        raw = self._redis.get(self._key(session_id))
        if raw is not None:
            self._redis.expire(self._key(session_id), int(self._ttl))
            self._redis.zadd(self._LRU_KEY, {session_id: now})
        return raw

    def _save_blocking(self, session_id: str, state: LLMConversationState) -> None:
        """Mirrors ``save`` of app/."""
        now = self._clock()
        payload = json.dumps(state.to_dict())
        self._evict_expired(now)
        self._redis.set(self._key(session_id), payload, ex=int(self._ttl))
        self._redis.zadd(self._LRU_KEY, {session_id: now})
        self._enforce_cap()

    def _finish_blocking(self, session_id: str, state: LLMConversationState) -> bool:
        """Save unless the session was reset while the turn ran. Returns whether it was saved."""
        if self._redis.get(self._reset_key(session_id)) is not None:
            self._redis.delete(self._reset_key(session_id))
            return False
        self._save_blocking(session_id, state)
        return True

    def _compare_and_run(self, key: str, token: str, action: Callable[[object], None]) -> bool:
        """Run ``action(pipe)`` inside MULTI only while ``key`` still holds ``token`` (WATCH, no Lua needed)."""
        with self._redis.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    if _s(pipe.get(key)) != token:
                        pipe.unwatch()
                        return False
                    pipe.multi()
                    action(pipe)
                    pipe.execute()
                    return True
                except redis_lib.WatchError:
                    continue

    def _refresh_blocking(self, session_id: str, token: str) -> bool:
        return self._compare_and_run(
            self._lock_key(session_id), token, lambda pipe: pipe.pexpire(self._lock_key(session_id), self._lock_ttl_ms)
        )

    def _release_blocking(self, session_id: str, token: str) -> bool:
        return self._compare_and_run(self._lock_key(session_id), token, lambda pipe: pipe.delete(self._lock_key(session_id)))

    def _try_lock_blocking(self, session_id: str, token: str) -> bool:
        return bool(self._redis.set(self._lock_key(session_id), token, nx=True, px=self._lock_ttl_ms))

    # ---- lock ----------------------------------------------------------------------

    def _busy(self, session_id: str, where: str) -> SessionBusyError:
        logger.warning("session store: lock wait timed out (%s) session=%s", where, turn_context.hash_session_id(session_id))
        self._event("lock_timeout")
        return SessionBusyError("session busy")

    async def _take_lock(self, session_id: str, deadline: float) -> str | None:
        """The lock token, or None when Redis is unreachable. Raises ``SessionBusyError`` at the deadline."""
        token = uuid.uuid4().hex
        while True:
            try:
                if await asyncio.to_thread(self._try_lock_blocking, session_id, token):
                    return token
            except Exception as exc:  # noqa: BLE001 - any redis/socket error: run without Redis
                self._redis_down("lock", exc)
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._busy(session_id, "redis")
            await self._sleep(min(_LOCK_POLL_S, remaining))

    async def _keep_lock(self, session_id: str, token: str, lease: _Lease) -> None:
        """Refresh the lock every third of its life until cancelled, lost, or past the max hold."""
        interval = self._lock_ttl_ms / 3000.0
        give_up = self._mono() + self._lock_max_hold_s
        while True:
            await self._sleep(interval)
            if self._mono() >= give_up:
                lease.capped = True
                logger.warning("session store: turn passed the max lock hold, no longer refreshing session=%s",
                               turn_context.hash_session_id(session_id))
                self._event("lock_hold_cap")
                return
            try:
                still_ours = await asyncio.to_thread(self._refresh_blocking, session_id, token)
            except Exception as exc:  # noqa: BLE001
                self._redis_down("lock refresh", exc)
                continue
            if not still_ours:
                lease.lost = True
                logger.warning("session store: lock lost during a turn session=%s", turn_context.hash_session_id(session_id))
                self._event("lock_lost")
                return

    async def _release(self, session_id: str, token: str) -> None:
        try:
            await asyncio.to_thread(self._release_blocking, session_id, token)
        except Exception as exc:  # noqa: BLE001 - the lock lapses by itself after its TTL
            self._redis_down("lock release", exc)

    # ---- load / save ---------------------------------------------------------------

    async def _load(self, session_id: str) -> LLMConversationState | None:
        """The stored state, a fresh one when there is none or it is unreadable, None when Redis is unreachable."""
        try:
            await asyncio.to_thread(self._redis.delete, self._reset_key(session_id))  # a reset before this turn is already in effect
            raw = await asyncio.to_thread(self._load_blocking, session_id)
        except UnicodeDecodeError:  # the stored bytes are not text: a corrupt blob, not an outage
            self._undecodable(session_id)
            return self._fresh()
        except Exception as exc:  # noqa: BLE001
            self._redis_down("load", exc)
            return None
        if raw is None:
            return self._fresh()
        return self._decode(session_id, raw) or self._fresh()

    async def _save_after_turn(self, session_id: str, state: LLMConversationState, lease: _Lease) -> None:
        try:
            saved = await asyncio.to_thread(self._finish_blocking, session_id, state)
            if not saved:
                self._event("save_skipped_after_reset")
        except Exception as exc:  # noqa: BLE001 - the user still gets the reply; the turn is just not remembered
            if isinstance(exc, TypeError):
                logger.error("session store: state could not be encoded: %s", type(exc).__name__)
                self._event("state_unencodable")
            else:
                self._redis_down("save", exc)

    # ---- public --------------------------------------------------------------------

    @asynccontextmanager
    async def turn(self, session_id: str, user_message: str | None = None):
        """Run one turn with exclusive access to the session, across servers. Raises ``SessionBusyError``
        before yielding when the session stays busy for ``LOCK_WAIT_S``. Same error handling as
        ``state.SessionStore.turn``: an exception in the body records ``ERROR_REPLY`` and is re-raised."""
        sid = normalize_session_id(session_id)  # validates the id
        local = self._locals.setdefault(sid, _Local())
        local.active += 1
        try:
            deadline = time.monotonic() + self._lock_wait_s
            try:
                await asyncio.wait_for(local.lock.acquire(), self._lock_wait_s)
            except asyncio.TimeoutError:
                raise self._busy(sid, "this server") from None
            try:
                lease, heartbeat = _Lease(), None
                try:
                    lease.token = await self._take_lock(sid, deadline)
                    state = None
                    if lease.token is not None:
                        heartbeat = asyncio.create_task(self._keep_lock(sid, lease.token, lease))
                        state = await self._load(sid)
                    degraded = state is None
                    if degraded:
                        state = self._fresh()  # Redis unreachable: this turn answers from a fresh state and saves nothing
                    turns_before = state.turn_count
                    try:
                        yield state
                    except BaseException:
                        if user_message is not None and state.turn_count == turns_before:
                            state.record_turn(user_message, ERROR_REPLY)
                        raise
                    finally:
                        if not degraded:
                            await self._save_after_turn(sid, state, lease)
                finally:
                    if heartbeat is not None:
                        heartbeat.cancel()
                        await asyncio.gather(heartbeat, return_exceptions=True)
                    if lease.token is not None:
                        await self._release(sid, lease.token)
            finally:
                local.lock.release()
        finally:
            local.active -= 1
            if local.active == 0 and self._locals.get(sid) is local:
                del self._locals[sid]

    def get_or_create(self, session_id: str) -> LLMConversationState:
        """The stored state or a fresh one. Not locked and **not saved**: changes to the returned state are
        lost unless ``save`` is called (use ``turn`` for read-modify-write). Redis unreachable -> a fresh state."""
        sid = normalize_session_id(session_id)
        return self._peek(sid, refresh=True) or self._fresh()

    def peek(self, session_id: str) -> LLMConversationState | None:
        """The stored state, or None when there is none, it is unreadable, or Redis is unreachable.
        Never creates and never refreshes the TTL."""
        try:
            sid = normalize_session_id(session_id)
        except ValueError:
            return None
        return self._peek(sid, refresh=False)

    def _peek(self, sid: str, *, refresh: bool) -> LLMConversationState | None:
        try:
            raw = self._load_blocking(sid) if refresh else self._redis.get(self._key(sid))
        except UnicodeDecodeError:
            self._undecodable(sid)
            return None
        except Exception as exc:  # noqa: BLE001
            self._redis_down("read", exc)
            return None
        return None if raw is None else self._decode(sid, raw)

    def save(self, session_id: str, state: LLMConversationState) -> None:
        """Write a state (not locked: for tests and single-writer tools). Raises on a Redis error."""
        self._save_blocking(normalize_session_id(session_id), state)

    def reset(self, session_id: str) -> bool:
        """Delete a session. Returns whether it existed (False, logged, when Redis is unreachable). A turn
        running on it finishes and does not save the session back."""
        sid = normalize_session_id(session_id)
        try:
            removed = self._redis.delete(self._key(sid))
            self._redis.zrem(self._LRU_KEY, sid)
            if self._redis.exists(self._lock_key(sid)):
                self._redis.set(self._reset_key(sid), "1", px=int(self._lock_max_hold_s * 1000) + self._lock_ttl_ms)
            return bool(removed)
        except Exception as exc:  # noqa: BLE001
            self._redis_down("reset", exc)
            return False

    def count(self) -> int:
        try:
            self._evict_expired(self._clock())
            return int(self._redis.zcard(self._LRU_KEY))
        except Exception as exc:  # noqa: BLE001
            self._redis_down("count", exc)
            return 0

    def ping(self) -> bool:
        """Whether Redis answers (for /ready). Never raises."""
        try:
            return bool(self._redis.ping())
        except Exception as exc:  # noqa: BLE001
            self._redis_down("ping", exc)
            return False


def make_store(*, backend: str | None = None, redis_client=None) -> SessionStoreBackend:
    """The store for ``STATE_BACKEND`` (memory by default). An unknown backend raises ``ValueError``."""
    chosen = backend if backend is not None else config.state_backend()
    if chosen == "redis":
        return RedisSessionStore(redis_client=redis_client)
    if chosen != "memory":
        raise ValueError(f"{config.STATE_BACKEND_ENV} must be one of: {', '.join(config.STATE_BACKENDS)}")
    return SessionStore()
