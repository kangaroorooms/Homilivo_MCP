"""Conversation state and the in-memory session store for the LLM approach.

``LLMConversationState`` holds everything one chat session needs between
turns. ``SessionStore`` keeps those states in memory with a TTL, an LRU cap
and one ``asyncio.Lock`` per session.

Assumptions (single process, asyncio only):
- All access happens on one event loop in one process. The store is not
  thread-safe and is not shared between uvicorn workers. Run one worker, or
  use ``session_store.RedisSessionStore`` (``STATE_BACKEND=redis``) for restarts and several servers.
- Store methods are synchronous and never await, so they cannot interleave
  inside one event loop. The only awaiting is the per-session lock taken by
  ``SessionStore.turn``.
- Sessions are lost on restart (this store only).

Indexing: ``selected_room_index`` is **1-based**, like ``room_reference`` in
the decision schema and the ``index`` field ``prompts.build_state_summary``
shows the model. ``None`` means no room in focus.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import re
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from llm_approach import config, observability, reuse, sorting, textclean
from llm_approach.actions import city_change
from llm_approach.schema import FACT_KEYS

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

# Session memory (see record_query / set_fact). Caps keep one session small.
QUERY_LOG_MAX = 200
QUERY_TEXT_MAX_CHARS = 500
QUERY_BLOCKED = "BLOCKED"  # intent stored for messages that were blocked, too long or failed before a decision
QUERY_PLACEHOLDER = "[message withheld]"  # stored instead of the text of such messages
USER_FACT_KEYS = frozenset(FACT_KEYS)

# Full transcript (conversation memory): every message of both roles, in memory only.
TRANSCRIPT_MAX_TURNS = 200
TRANSCRIPT_USER_MAX_CHARS = 1000
TRANSCRIPT_BOT_MAX_CHARS = 800
TRANSCRIPT_ROOM_TITLES_MAX = 10
TRANSCRIPT_TITLE_MAX_CHARS = 80

SEARCH_FIELDS = ("city", "area", "bhk", "min_price", "max_price", "room_types", "ideal_for", "amenity", "sort")
PENDING_TYPES = ("city", "clarify", "reject", "city_change")

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def normalize_session_id(session_id: Any) -> str:
    """Return the lowercase UUID string, or raise ``ValueError``."""
    if not isinstance(session_id, str) or not _UUID_RE.match(session_id.strip().lower()):
        raise ValueError("session_id must be a UUID string (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)")
    return session_id.strip().lower()


# Length caps of the free-text fields of a stored room (characters); any other string gets ROOM_TEXT_DEFAULT_CAP.
ROOM_TEXT_CAPS = {"title": 200, "description": 500, "location": 200, "landmark": 200, "city": 100, "roomType": 50,
                  "type": 50, "availabilityStatus": 40, "id": 100}
ROOM_TEXT_DEFAULT_CAP = 100  # an item of a list field (amenities, idealFor, ...)


def _clean_room_text(key: str, value: Any) -> Any:
    """Third-party text through the shared plain-text cleaner (tags, links, domains, markdown, control and invisible
    characters out, length capped). A string that is empty after cleaning is dropped; non-text values pass through."""
    cap = ROOM_TEXT_CAPS.get(key, ROOM_TEXT_DEFAULT_CAP)
    if isinstance(value, str):
        cleaned = textclean.clean_text(value, cap)
        if key == "city" and cleaned != " ".join(value.split()):
            return None  # an odd-looking city (markup, a fragment, ...) is dropped, not tidied into a plausible one
        return cleaned
    if isinstance(value, list):
        return [c for c in (textclean.clean_text(i, cap) if isinstance(i, str) else i for i in value) if c is not None]
    return value


def clean_room(room: dict[str, Any]) -> dict[str, Any]:
    """Allowlist the fields, run the reused prompt sanitiser, then the shared plain-text cleaner on every string.
    Nothing a listing owner typed reaches state, cards or replies as markup (v2)."""
    kept = {k: v for k, v in room.items() if k in ROOM_ALLOWED_FIELDS}
    sanitized = reuse.get_sanitize_room()(kept)
    cleaned = {k: _clean_room_text(k, v) for k, v in sanitized.items()}
    return {k: v for k, v in cleaned.items() if v is not None}


def _empty_search() -> dict[str, Any]:
    return {key: None for key in SEARCH_FIELDS}


# ---- persistence helpers ---------------------------------------------------

STATE_SCHEMA_VERSION = 1  # bump when a field changes meaning; an old blob is then dropped, not guessed at


class StateDecodeError(ValueError):
    """A stored state could not be read. The message is a fixed kind (and field name), never content."""


class SessionBusyError(RuntimeError):
    """The session lock could not be taken within the wait bound (another turn on the session is still running)."""


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _list_of_dicts(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(i, dict) for i in v)


def _opt(check: Callable[[Any], bool]) -> Callable[[Any], bool]:
    return lambda v: v is None or check(v)


def _valid_pending(v: Any) -> bool:
    """A known type; a ``city_change`` question must also carry its context (see actions/city_change.py)."""
    if not isinstance(v, dict) or v.get("type") not in PENDING_TYPES:
        return False
    return v["type"] != "city_change" or city_change.valid_context(v.get("context"))


# One check per dataclass field (the round-trip test fails when a field is added without one).
_FIELD_CHECKS: dict[str, Callable[[Any], bool]] = {
    "history": lambda v: _list_of_dicts(v) and all(isinstance(i.get("content"), str) for i in v),
    "last_bot_message": lambda v: isinstance(v, str),
    "current_search": lambda v: isinstance(v, dict),
    "rooms_on_screen": _list_of_dicts,
    "selected_room_index": _opt(_is_int),
    "pending_question": _opt(_valid_pending),
    "rejected_room_ids": lambda v: isinstance(v, list),
    "api_room_count": _opt(_is_int),
    "rooms_truncated": lambda v: isinstance(v, bool),
    "search_page": lambda v: _is_int(v) and v >= 1,
    "has_next": _opt(lambda v: isinstance(v, bool)),
    "total_rooms": _opt(_is_int),
    "turn_count": lambda v: _is_int(v) and v >= 0,
    "updated_at": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "user_queries": lambda v: _list_of_dicts(v) and all(isinstance(i.get("text"), str) for i in v),
    "query_count": lambda v: _is_int(v) and v >= 0,
    "user_facts": lambda v: isinstance(v, dict) and all(isinstance(e, dict) and "value" in e for e in v.values()),
    "transcript": lambda v: _list_of_dicts(v)
    and all(i.get("role") in ("user", "bot") and isinstance(i.get("text"), str) for i in v),
    "transcript_seq": lambda v: _is_int(v) and v >= 0,
}


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
    # Session memory. user_queries: every user message in order, {turn, text, intent}; ``turn``
    # counts all messages of the session (1-based) and keeps counting when old entries are dropped.
    # user_facts: allowlisted keys only, {key: {value, turn}}.
    user_queries: list[dict[str, Any]] = field(default_factory=list)
    query_count: int = 0
    user_facts: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Conversation memory: every message of both roles, {turn, role "user"|"bot", text, intent[, rooms]}.
    # ``turn`` is a running id over all entries (user and bot alike); it keeps counting when the
    # oldest entries are dropped. Never logged, never sent anywhere, gone with the session.
    transcript: list[dict[str, Any]] = field(default_factory=list)
    transcript_seq: int = 0

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

    # ---- session memory ------------------------------------------------

    def record_query(self, text: str | None, intent: str = "UNCLEAR") -> int:
        """Log one user message and return its turn number. Call once per message.

        ``text=None`` (blocked, too long, failed security check) stores a placeholder, never the
        text. Text longer than QUERY_TEXT_MAX_CHARS is cut. The oldest entry is dropped past
        QUERY_LOG_MAX."""
        self.query_count += 1
        if text is None:
            text, intent = QUERY_PLACEHOLDER, QUERY_BLOCKED
        self.user_queries.append(
            {"turn": self.query_count, "text": str(text)[:QUERY_TEXT_MAX_CHARS], "intent": intent}
        )
        if len(self.user_queries) > QUERY_LOG_MAX:
            del self.user_queries[: len(self.user_queries) - QUERY_LOG_MAX]
        self._add_transcript("user", str(text)[:TRANSCRIPT_USER_MAX_CHARS], intent)
        return self.query_count

    def set_last_query_intent(self, intent: str) -> None:
        """Set the intent of the latest logged message (known only after the decision)."""
        if self.user_queries and self.user_queries[-1]["intent"] != QUERY_BLOCKED:
            self.user_queries[-1]["intent"] = intent
            last = self.last_transcript_entry("user")
            if last is not None:
                last["intent"] = intent

    # ---- conversation memory (transcript) --------------------------------

    def _add_transcript(self, role: str, text: str, intent: str, rooms: list[str] | None = None) -> None:
        self.transcript_seq += 1
        entry: dict[str, Any] = {"turn": self.transcript_seq, "role": role, "text": text, "intent": intent}
        if rooms:
            entry["rooms"] = rooms
        self.transcript.append(entry)
        if len(self.transcript) > TRANSCRIPT_MAX_TURNS:
            del self.transcript[: len(self.transcript) - TRANSCRIPT_MAX_TURNS]

    def last_transcript_entry(self, role: str) -> dict[str, Any] | None:
        for entry in reversed(self.transcript):
            if entry["role"] == role:
                return entry
        return None

    def record_bot_reply(self, text: str, rooms: list[dict[str, Any]] | None = None) -> None:
        """Log the bot's reply (call after the output guard). ``rooms`` are the cards shown with it;
        only their titles are kept. The reply takes the intent of the user message it answers."""
        last_user = self.last_transcript_entry("user")
        intent = last_user["intent"] if last_user else "UNCLEAR"
        titles = [
            str(r.get("title"))[:TRANSCRIPT_TITLE_MAX_CHARS]
            for r in (rooms or [])[:TRANSCRIPT_ROOM_TITLES_MAX]
            if isinstance(r, dict) and r.get("title")
        ]
        self._add_transcript("bot", str(text)[:TRANSCRIPT_BOT_MAX_CHARS], intent, titles)

    def set_fact(self, key: str, value: Any) -> bool:
        """Store (or overwrite) one already validated fact. Unknown keys are refused."""
        if key not in USER_FACT_KEYS:
            return False
        self.user_facts[key] = {"value": value, "turn": self.query_count}
        return True

    def get_fact(self, key: str) -> Any:
        entry = self.user_facts.get(key)
        return entry["value"] if entry else None

    # ---- persistence (Redis) ---------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe copy of every field plus ``schema_version``. Raises ``TypeError`` when a value is not JSON-safe."""
        data = {f.name: copy.deepcopy(getattr(self, f.name)) for f in dataclasses.fields(self)}
        json.dumps(data)  # fail here, not half way into a Redis write
        data["schema_version"] = STATE_SCHEMA_VERSION
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "LLMConversationState":
        """Rebuild a state from ``to_dict`` output. Raises ``StateDecodeError`` (a ValueError) for anything that is
        not a dict of the current schema version, has a field of the wrong type, or whose selected room is not on
        screen. The message names the problem kind and field, never a value."""
        if not isinstance(data, dict):
            raise StateDecodeError("not_an_object")
        if data.get("schema_version") != STATE_SCHEMA_VERSION or isinstance(data.get("schema_version"), bool):
            raise StateDecodeError("wrong_schema_version")
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in data:
                continue  # an older blob of the same version: that field keeps its default
            value = data[f.name]
            check = _FIELD_CHECKS[f.name]
            if not check(value):
                raise StateDecodeError(f"bad_field:{f.name}")
            kwargs[f.name] = copy.deepcopy(value)
        state = cls(**kwargs)
        state.current_search = {key: state.current_search.get(key) for key in SEARCH_FIELDS}
        state.current_search["sort"] = sorting.clean(state.current_search["sort"])  # an older blob has none
        index = state.selected_room_index
        if index is not None and not 1 <= index <= len(state.rooms_on_screen):
            raise StateDecodeError("bad_field:selected_room_index")
        return state

    # ---- prompt view ---------------------------------------------------

    def to_prompt_state(self) -> dict[str, Any]:
        """Input for ``prompts.build_state_summary`` (same keys).

        ``pending_question`` is the type string when there is no context (the
        form the few-shot examples use), otherwise ``{"type", "context"}``.
        ``current_search`` is None until a search has any value."""
        pending = self.pending_question
        if pending is not None and pending["type"] == "city_change":
            pending = "clarify"  # the model sees a plain question; the remembered filters stay out of the prompt
        elif pending is not None and pending.get("context") is None:
            pending = pending["type"]
        search = {k: v for k, v in self.current_search.items() if v is not None and k != "sort"}  # the model never sees it
        return {
            "current_search": search or None,
            "pending_question": pending,
            "last_bot_message": self.last_bot_message or None,
            "rooms_on_screen": [dict(r) for r in self.rooms_on_screen],
            "selected_room_index": self.selected_room_index,
            "recent_turns": [dict(m) for m in self.history],
            "user_facts": {k: e["value"] for k, e in self.user_facts.items()},
        }


# ---- store ---------------------------------------------------------------


@dataclass
class _Entry:
    state: LLMConversationState
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_access: float = 0.0
    active: int = 0  # turns running or waiting for the lock


class SessionStore:
    """In-memory sessions: TTL, LRU cap, one lock per session. The default store; ``session_store.RedisSessionStore``
    is the shared one (same interface, see ``session_store.SessionStoreBackend``)."""

    is_shared = False  # state lives in this process: a state returned by get_or_create is the live one

    def __init__(
        self,
        *,
        ttl_seconds: float | None = None,
        max_sessions: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        lock_wait_s: float | None = None,
    ) -> None:
        self.lock_wait_s = config.session_lock_wait_s() if lock_wait_s is None else lock_wait_s  # longest a turn waits for its session
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

    def save(self, session_id: str, state: LLMConversationState) -> None:
        """Store ``state`` as the session's state (the interface the Redis store needs; a no-op for a state that
        came from ``get_or_create``, which is already the live one)."""
        self._get_entry(session_id).state = state

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
        released. A turn waits at most ``lock_wait_s`` for the session, then
        ``SessionBusyError`` is raised before anything runs."""
        entry = self._get_entry(session_id)  # validates the id
        entry.active += 1
        try:
            try:
                await asyncio.wait_for(entry.lock.acquire(), self.lock_wait_s)
            except asyncio.TimeoutError:
                try:
                    observability.record_session_store_event("lock_timeout")
                except Exception:  # noqa: BLE001 - counting must never break a turn
                    pass
                raise SessionBusyError("session busy") from None
            try:
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
                entry.lock.release()
        finally:
            entry.active -= 1
