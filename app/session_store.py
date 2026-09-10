"""
Per-session conversational state storage.

In-memory today, but the public surface (get/save/reset/__len__) is
the only thing app/api.py depends on, so a Redis-backed (or any other)
implementation can replace SessionStore later - matching how a caller
uses it - without touching the chatbot pipeline in
app/experiments/agent_modes.py.

Bounded on three axes so a long-running process cannot grow without
limit:
- TTL eviction: a session idle for longer than SESSION_TTL_SECONDS is
  dropped.
- Session cap: SESSION_MAX_COUNT sessions total; the least-recently-
  used session is evicted first when the cap is exceeded.
- Conversation history cap: each session's conversation_history is
  trimmed to the last SESSION_MAX_HISTORY_MESSAGES turns on every
  save(), regardless of where the append happened.
"""

from __future__ import annotations

import os
import threading
import time

from app.experiments.agent_modes import ConversationState

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
SESSION_MAX_COUNT = int(os.getenv("SESSION_MAX_COUNT", "1000"))
SESSION_MAX_HISTORY_MESSAGES = int(os.getenv("SESSION_MAX_HISTORY_MESSAGES", "40"))


class SessionStore:
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
