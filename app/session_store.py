"""
Per-session conversational state storage.

Two backends share one narrow interface (get/save/reset/__len__) so
app/api.py's `SESSIONS = SessionStore()` keeps working unchanged no
matter which one is active - selected via STATE_BACKEND=memory|redis
(default memory, so nothing breaks without Redis):

- InMemorySessionStore: today's implementation, unchanged. Fine for a
  single-process deployment.
- RedisSessionStore: same TTL/cap/history-trim semantics, backed by
  Redis so multiple app instances behind a load balancer share
  sessions (and the "conversation memory" ConversationState carries
  for follow-ups - see app.experiments.agent_modes.ConversationState's
  own docstring, "Structured conversational memory" - is the same
  object being persisted here; this project does not have a second,
  separate memory store).

`SessionStore(...)` is a factory function, not a class, so every
existing call site (`SessionStore()`, `SessionStore(ttl_seconds=..,
max_sessions=..)`) keeps working byte-for-byte; `backend=`/
`redis_client=` are additive, test-only override points that
production code never passes.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict
from typing import Protocol

from app.experiments.agent_modes import ConversationState

try:
    import redis as redis_lib
except ImportError:  # pragma: no cover - redis is always in requirements.txt
    redis_lib = None  # STATE_BACKEND=redis will raise a clear error if reached

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
SESSION_MAX_COUNT = int(os.getenv("SESSION_MAX_COUNT", "1000"))
SESSION_MAX_HISTORY_MESSAGES = int(os.getenv("SESSION_MAX_HISTORY_MESSAGES", "40"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class SessionStoreBackend(Protocol):
    def get(self, session_id: str) -> ConversationState: ...
    def save(self, session_id: str, state: ConversationState) -> None: ...
    def reset(self, session_id: str) -> bool: ...
    def __len__(self) -> int: ...


class InMemorySessionStore:
    def __init__(
        self,
        ttl_seconds: int = SESSION_TTL_SECONDS,
        max_sessions: int = SESSION_MAX_COUNT,
        max_history_messages: int = SESSION_MAX_HISTORY_MESSAGES,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._max_history = max_history_messages
        self._data: dict[str, list] = {}  # session_id -> [last_access, state]
        self._lock = threading.Lock()

    def _evict_expired(self, now: float) -> None:
        expired = [
            sid for sid, (ts, _) in self._data.items() if now - ts > self._ttl
        ]
        for sid in expired:
            del self._data[sid]

    def _enforce_cap(self) -> None:
        if len(self._data) <= self._max:
            return
        ordered = sorted(self._data.items(), key=lambda kv: kv[1][0])
        for sid, _ in ordered[: len(self._data) - self._max]:
            del self._data[sid]

    def _trim_history(self, state: ConversationState) -> None:
        if self._max_history <= 0:
            return
        history = state.conversation_history
        if len(history) > self._max_history:
            del history[: len(history) - self._max_history]

    def get(self, session_id: str) -> ConversationState:
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            item = self._data.get(session_id)
            if item is None:
                state = ConversationState()
                self._data[session_id] = [now, state]
                self._enforce_cap()
                return state
            item[0] = now
            return item[1]

    def save(self, session_id: str, state: ConversationState) -> None:
        now = time.time()
        self._trim_history(state)
        with self._lock:
            self._evict_expired(now)
            self._data[session_id] = [now, state]
            self._enforce_cap()

    def reset(self, session_id: str) -> bool:
        with self._lock:
            return self._data.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class RedisSessionStore:
    """
    Redis-backed equivalent of InMemorySessionStore.

    Idle-TTL and LRU-cap eviction are computed from our OWN `now`
    (module-level `time.time()`, monkeypatchable exactly like the
    in-memory store) rather than Redis's native key TTL, so both
    backends behave identically under a test's mocked clock. Redis's
    own EXPIRE is set too, purely as a storage-hygiene backstop so an
    idle key is eventually reclaimed even if nothing ever calls back
    into this store to notice it.

    A session's last-access time is tracked in a Redis sorted set
    (`_LRU_KEY`) shared by every instance pointed at the same Redis, so
    the LRU cap is enforced consistently across processes instead of
    each instance only knowing about the sessions it personally
    touched.
    """

    _KEY_PREFIX = "homilivo:session:"
    _LRU_KEY = "homilivo:sessions:lru"

    def __init__(
        self,
        ttl_seconds: int = SESSION_TTL_SECONDS,
        max_sessions: int = SESSION_MAX_COUNT,
        max_history_messages: int = SESSION_MAX_HISTORY_MESSAGES,
        redis_client=None,
        url: str | None = None,
    ) -> None:
        if redis_client is None and redis_lib is None:
            raise RuntimeError(
                "STATE_BACKEND=redis requires the 'redis' package "
                "(pip install -r requirements.txt)."
            )
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._max_history = max_history_messages
        self._redis = redis_client or redis_lib.Redis.from_url(
            url or REDIS_URL, decode_responses=True
        )

    def _key(self, session_id: str) -> str:
        return f"{self._KEY_PREFIX}{session_id}"

    def _serialize(self, state: ConversationState) -> str:
        return json.dumps(asdict(state))

    def _deserialize(self, raw: str) -> ConversationState:
        return ConversationState(**json.loads(raw))

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

    def _trim_history(self, state: ConversationState) -> None:
        if self._max_history <= 0:
            return
        history = state.conversation_history
        if len(history) > self._max_history:
            del history[: len(history) - self._max_history]

    def get(self, session_id: str) -> ConversationState:
        now = time.time()
        self._evict_expired(now)
        raw = self._redis.get(self._key(session_id))
        if raw is None:
            state = ConversationState()
            self._redis.set(self._key(session_id), self._serialize(state), ex=self._ttl)
            self._redis.zadd(self._LRU_KEY, {session_id: now})
            self._enforce_cap()
            return state
        self._redis.expire(self._key(session_id), self._ttl)
        self._redis.zadd(self._LRU_KEY, {session_id: now})
        return self._deserialize(raw)

    def save(self, session_id: str, state: ConversationState) -> None:
        now = time.time()
        self._trim_history(state)
        self._evict_expired(now)
        self._redis.set(self._key(session_id), self._serialize(state), ex=self._ttl)
        self._redis.zadd(self._LRU_KEY, {session_id: now})
        self._enforce_cap()

    def reset(self, session_id: str) -> bool:
        removed = self._redis.delete(self._key(session_id))
        self._redis.zrem(self._LRU_KEY, session_id)
        return bool(removed)

    def __len__(self) -> int:
        self._evict_expired(time.time())
        return int(self._redis.zcard(self._LRU_KEY))


def SessionStore(
    ttl_seconds: int = SESSION_TTL_SECONDS,
    max_sessions: int = SESSION_MAX_COUNT,
    max_history_messages: int = SESSION_MAX_HISTORY_MESSAGES,
    *,
    backend: str | None = None,
    redis_client=None,
    url: str | None = None,
) -> SessionStoreBackend:
    """
    Build a session store for the configured backend.

    `backend` defaults to the STATE_BACKEND env var (memory|redis,
    default memory) read fresh on every call, so app/api.py's
    `SessionStore()` picks up STATE_BACKEND without any change to that
    call site. `backend=`/`redis_client=`/`url=` let tests force a
    specific backend (and inject a fake Redis client, or point at a
    non-default Redis) without touching the environment.
    """
    chosen = (backend or os.getenv("STATE_BACKEND", "memory")).strip().lower()
    if chosen == "redis":
        return RedisSessionStore(
            ttl_seconds=ttl_seconds,
            max_sessions=max_sessions,
            max_history_messages=max_history_messages,
            redis_client=redis_client,
            url=url,
        )
    return InMemorySessionStore(
        ttl_seconds=ttl_seconds,
        max_sessions=max_sessions,
        max_history_messages=max_history_messages,
    )
