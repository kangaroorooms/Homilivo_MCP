"""Conversation state and the in-memory session store for the LLM approach.

``LLMConversationState`` holds everything one chat session needs between
turns. ``SessionStore`` keeps those states in memory with a TTL, an LRU cap
and one ``asyncio.Lock`` per session.

Assumptions (single process, asyncio only):
- All access happens on one event loop in one process. The store is not
  thread-safe and is not shared between uvicorn workers. Run one worker, or
  move the state to Redis (not done here).
- Store methods are synchronous and never await, so they cannot interleave
  inside one event loop. The only awaiting is the per-session lock taken by
  ``SessionStore.turn``.
- Sessions are lost on restart.

Indexing: ``selected_room_index`` is **1-based**, like ``room_reference`` in
the decision schema and the ``index`` field ``prompts.build_state_summary``
shows the model. ``None`` means no room in focus.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from llm_approach import config, reuse

ERROR_REPLY = "Sorry, something went wrong on my side. Please try again."

# Fields a room may keep in session state. Everything else (owner, contact,
# phone, email, coordinates, ...) is dropped before the room is stored.
ROOM_ALLOWED_FIELDS = frozenset(
    {
        "id",
        "title",
        "description",
        "city",
        "location",
        "landmark",
        "pricePerMonth",
        "price",
        "roomType",
        "type",
        "idealFor",
        "amenities",
        "amenity",
        "facilities",
        "facility",
        "features",
        "rating",
        "availabilityStatus",
    }
)

SEARCH_FIELDS = ("city", "area", "bhk", "min_price", "max_price", "room_types", "ideal_for", "amenity")
PENDING_TYPES = ("city", "clarify", "reject")

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def normalize_session_id(session_id: Any) -> str:
    """Return the lowercase UUID string, or raise ``ValueError``."""
    if not isinstance(session_id, str) or not _UUID_RE.match(session_id.strip().lower()):
        raise ValueError("session_id must be a UUID string (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)")
    return session_id.strip().lower()


def clean_room(room: dict[str, Any]) -> dict[str, Any]:
    """Allowlist the fields, then run the reused prompt sanitiser."""
    kept = {k: v for k, v in room.items() if k in ROOM_ALLOWED_FIELDS}
    return reuse.get_sanitize_room()(kept)


def _empty_search() -> dict[str, Any]:
    return {key: None for key in SEARCH_FIELDS}


@dataclass
class LLMConversationState:
    history: list[dict[str, str]] = field(default_factory=list)
    last_bot_message: str = ""
    current_search: dict[str, Any] = field(default_factory=_empty_search)
    rooms_on_screen: list[dict[str, Any]] = field(default_factory=list)
    selected_room_index: int | None = None  # 1-based
    pending_question: dict[str, Any] | None = None
    rejected_room_ids: list[Any] = field(default_factory=list)
    api_room_count: int | None = None  # rooms the API returned, before client-side filtering
    rooms_truncated: bool = False  # matching rooms were cut to the display limit
    search_page: int = 1  # page of the last search (1-based)
    has_next: bool | None = None  # meta.hasNextPage of the last result (None: unknown)
    total_rooms: int | None = None  # meta.total of the last result (None: unknown)
    turn_count: int = 0
    updated_at: float = 0.0

    # ---- search / rooms ------------------------------------------------

    def apply_search_result(
        self,
        search: dict[str, Any] | None,
        rooms: list[dict[str, Any]] | None,
        api_count: int | None = None,
        truncated: bool = False,
        page: int = 1,
        has_next: bool | None = None,
        total: int | None = None,
    ) -> bool:
        """Store a finished search. ``rooms=None`` means the rooms API failed:
        nothing changes (previous rooms, selection, search and pending stay).
        An empty list is a real search with no results and does replace state.
        ``api_count`` is how many rooms the API returned before client-side
        filtering (amenity, area); SHOW_MORE uses it. Defaults to ``len(rooms)``.
        ``truncated``: more matching rooms exist than were kept for display.
        ``page``, ``has_next`` and ``total`` come from the API paging info; a new
        search passes the defaults, which resets the page to 1.
        Returns True when the state was updated."""
        if rooms is None:
            return False
        self.current_search = {key: (search or {}).get(key) for key in SEARCH_FIELDS}
        self.rooms_on_screen = [clean_room(r) for r in rooms if isinstance(r, dict)]
        self.api_room_count = len(self.rooms_on_screen) if api_count is None else api_count
        self.rooms_truncated = truncated
        self.set_paging(page, has_next, total)
        self.selected_room_index = None
        self.pending_question = None
        self.rejected_room_ids = []
        return True

    def set_paging(self, page: int, has_next: bool | None, total: int | None) -> None:
        self.search_page = page if isinstance(page, int) and page >= 1 else 1
        self.has_next = has_next
        self.total_rooms = total

    def select_room(self, index: int) -> None:
        """Focus the room at 1-based ``index``."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("room index must be an integer")
        if not 1 <= index <= len(self.rooms_on_screen):
            raise ValueError(f"room index {index} is not on screen ({len(self.rooms_on_screen)} rooms)")
        self.selected_room_index = index

    def clear_selection(self) -> None:
        self.selected_room_index = None

    def selected_room(self) -> dict[str, Any] | None:
        if self.selected_room_index is None:
            return None
        return self.rooms_on_screen[self.selected_room_index - 1]

    def reject_rooms(self) -> None:
        """Remember the rooms on screen as turned down by the user."""
        for room in self.rooms_on_screen:
            room_id = room.get("id")
            if room_id is not None and room_id not in self.rejected_room_ids:
                self.rejected_room_ids.append(room_id)

    # ---- pending question ----------------------------------------------

    def set_pending(self, type: str, context: Any = None) -> None:
        if type not in PENDING_TYPES:
            raise ValueError(f"pending type must be one of {PENDING_TYPES}, got {type!r}")
        self.pending_question = {"type": type, "context": context}

    def clear_pending(self) -> None:
        self.pending_question = None

    # ---- history -------------------------------------------------------

    def record_turn(self, user_msg: str, bot_msg: str, now: float | None = None) -> None:
        """The only writer of history: call once per finished turn."""
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": bot_msg})
        cap = config.SESSION_MAX_HISTORY_MESSAGES
        if cap > 0 and len(self.history) > cap:
            self.history = self.history[-cap:]
        self.last_bot_message = bot_msg
        self.turn_count += 1
        self.updated_at = time.time() if now is None else now

    # ---- prompt view ---------------------------------------------------

    def to_prompt_state(self) -> dict[str, Any]:
        """Input for ``prompts.build_state_summary`` (same keys).

        ``pending_question`` is the type string when there is no context (the
        form the few-shot examples use), otherwise ``{"type", "context"}``.
        ``current_search`` is None until a search has any value."""
        pending = self.pending_question
        if pending is not None and pending.get("context") is None:
            pending = pending["type"]
        search = {k: v for k, v in self.current_search.items() if v is not None}
        return {
            "current_search": search or None,
            "pending_question": pending,
            "last_bot_message": self.last_bot_message or None,
            "rooms_on_screen": [dict(r) for r in self.rooms_on_screen],
            "selected_room_index": self.selected_room_index,
            "recent_turns": [dict(m) for m in self.history],
        }


# ---- store ---------------------------------------------------------------


@dataclass
class _Entry:
    state: LLMConversationState
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_access: float = 0.0
    active: int = 0  # turns running or waiting for the lock


class SessionStore:
    """In-memory sessions: TTL, LRU cap, one lock per session."""

    def __init__(
        self,
        *,
        ttl_seconds: float | None = None,
        max_sessions: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = config.SESSION_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        self.max_sessions = config.SESSION_MAX_COUNT if max_sessions is None else max_sessions
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()  # oldest access first

    # ---- internals -----------------------------------------------------

    def _expired(self, entry: _Entry, now: float) -> bool:
        return entry.active == 0 and now - entry.last_access > self.ttl_seconds

    def _sweep(self, now: float) -> None:
        """Drop expired idle sessions together with their locks."""
        for sid in [s for s, e in self._entries.items() if self._expired(e, now)]:
            del self._entries[sid]

    def _evict_to_cap(self, keep: str) -> None:
        """LRU eviction. Sessions with a running or waiting turn are kept, and so is
        ``keep`` (the session being used right now), so the store can briefly exceed
        the cap when every session is busy."""
        for sid in list(self._entries):
            if len(self._entries) <= self.max_sessions:
                break
            if sid != keep and self._entries[sid].active == 0:
                del self._entries[sid]

    def _get_entry(self, session_id: Any) -> _Entry:
        sid = normalize_session_id(session_id)
        now = self._clock()
        self._sweep(now)
        entry = self._entries.get(sid)
        if entry is None:
            entry = _Entry(state=LLMConversationState(updated_at=time.time()))
            self._entries[sid] = entry
        entry.last_access = now
        self._entries.move_to_end(sid)
        self._evict_to_cap(sid)
        return entry

    # ---- public --------------------------------------------------------

    def get_or_create(self, session_id: str) -> LLMConversationState:
        """The session's state (created if missing). Refreshes its TTL.
        Raises ``ValueError`` for a session id that is not a UUID."""
        return self._get_entry(session_id).state

    def peek(self, session_id: str) -> LLMConversationState | None:
        """The state if the session exists and is not expired. Never creates,
        never refreshes the TTL, never reorders the LRU. For debugging."""
        try:
            sid = normalize_session_id(session_id)
        except ValueError:
            return None
        entry = self._entries.get(sid)
        if entry is None or self._expired(entry, self._clock()):
            return None
        return entry.state

    def reset(self, session_id: str) -> bool:
        """Delete a session. Returns whether it existed. A turn already
        running on it finishes on the old state, which is then discarded."""
        sid = normalize_session_id(session_id)
        self._sweep(self._clock())
        return self._entries.pop(sid, None) is not None

    def count(self) -> int:
        self._sweep(self._clock())
        return len(self._entries)

    @asynccontextmanager
    async def turn(
        self, session_id: str, user_message: str | None = None
    ) -> AsyncIterator[LLMConversationState]:
        """Run one turn with exclusive access to the session.

        The lock is held for the whole ``async with`` body, so overlapping
        turns on one session run one after another, in arrival order. Turns
        on different sessions do not block each other.

        The body must call ``state.record_turn`` when it finishes. If it
        raises before that and ``user_message`` was given, the user message and
        ``ERROR_REPLY`` are recorded so history never ends on an unanswered
        user message; the exception is then re-raised. The lock is always
        released."""
        entry = self._get_entry(session_id)  # validates the id
        entry.active += 1
        try:
            async with entry.lock:
                state = entry.state
                turns_before = state.turn_count
                try:
                    yield state
                except BaseException:
                    if user_message is not None and state.turn_count == turns_before:
                        state.record_turn(user_message, ERROR_REPLY)
                    raise
                finally:
                    entry.last_access = self._clock()
        finally:
            entry.active -= 1
