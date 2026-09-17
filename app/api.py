import sys
import threading
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from app.chatbot.routes import router as chatbot_router


# The conversational engine logs with Unicode symbols (arrows, checkmarks).
# On Windows, stdout defaults to cp1252 whenever it isn't an interactive
# UTF-8 console (piped, redirected to a log file, run under a process
# manager) and crashes every request with UnicodeEncodeError. Force UTF-8
# before anything else can print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from mcp import Client
from app.server import mcp, set_kangaroo_token

# The real conversational engine (routing, follow-ups, refinements,
# clarifications, FAST/QWEN3/GEMMA, embedding fallback) lives here.
from app.experiments.agent_modes import process_message, ConversationState


# ============================================================
# SESSION STORE (inlined - no separate file needed)
# ============================================================
# Maps a session_id to a ConversationState so the API can hold a multi-turn
# conversation the same way the terminal loop does. TTL eviction + a soft cap
# keep memory bounded.

class SessionStore:
    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 1000) -> None:
        self._ttl = ttl_seconds
        self._max = max_sessions
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

        with self._lock:
            self._data[session_id] = [now, state]
            self._enforce_cap()

    def reset(self, session_id: str) -> bool:
        with self._lock:
            return self._data.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Homilivo Room Search API",
    description="FastAPI service for the Homilivo MCP AI agent",
    version="2.0.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Separate Google File Search chatbot router.
# This does NOT modify the existing /chat endpoint.
app.include_router(chatbot_router)


SESSIONS = SessionStore()


# Single source of truth for the valid response modes. Pydantic enforces
# this on ChatRequest.mode below, so an invalid mode is rejected with a
# normal 422 before the request ever reaches the conversational engine
# - it is never silently coerced to "fast".
ChatMode = Literal["fast", "qwen", "gemma", "groq"]


# ------------------------------------------------------------
# AUTH: this renders the green "Authorize" button in /docs.
# Paste the Homilivo API token there (no "Bearer " prefix).
# It is applied to the room search for ALL modes (fast/qwen/gemma/groq),
# because every mode goes through the /chat endpoint below.
# ------------------------------------------------------------
bearer_scheme = HTTPBearer(description="Homilivo API token (KANGAROO_TOKEN)")


@app.on_event("startup")
async def _warmup_models() -> None:
    # Load the shared embedding model, plus both modules' precomputed
    # embeddings (interpreter's operation anchors, the input guardrail's
    # intent examples), once at boot so the first user request isn't slow.
    try:
        from app.experiments.interpreter import warmup as warmup_interpreter

        warmup_interpreter()
    except Exception as error:  # noqa: BLE001
        print(f"⚠ interpreter warmup skipped: {error}")

    try:
        from app.guardrails.input_guardrail import get_intent_embeddings

        get_intent_embeddings()
    except Exception as error:  # noqa: BLE001
        print(f"⚠ input guardrail warmup skipped: {error}")


# ============================================================
# MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Room-search request or follow-up from the user",
    )

    session_id: str | None = Field(
        None,
        description="Omit on the first call; reuse the returned value for follow-ups.",
    )

    mode: ChatMode = Field(
        "fast",
        description="fast | qwen | gemma | groq",
    )


class RoomResponse(BaseModel):
    id: str
    title: str
    city: str
    location: str | None = None
    price_per_month: float | None = None
    room_type: str | None = None
    ideal_for: list[str] = []


class ChatResponse(BaseModel):
    success: bool
    session_id: str
    mode: str
    message: str
    room_count: int = 0
    rooms: list[RoomResponse] = []
    selected_room: RoomResponse | None = None
    pending_clarification: bool = False


# ============================================================
# HELPERS
# ============================================================

def _structure_room(room: dict) -> dict:
    price = room.get("pricePerMonth")

    if price is None:
        price = room.get("price")

    return {
        "id": str(room.get("id")),
        "title": room.get("title", "Untitled Room"),
        "city": room.get("city", "Unknown"),
        "location": room.get("location"),
        "price_per_month": price,
        "room_type": room.get("roomType"),
        "ideal_for": room.get("idealFor", []),
    }


def _apply_result_to_state(
    state: ConversationState,
    result: dict,
) -> str | None:
    """Mirror the memory-update logic from the terminal loop."""

    response = result.get("response")

    if response:
        state.conversation_history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

    rooms = result.get("rooms")

    if isinstance(rooms, list):
        state.previous_rooms = rooms

    search = result.get("search")

    if isinstance(search, dict):
        state.previous_search = search

    selected_room = result.get("selected_room")

    state.selected_room = (
        selected_room
        if isinstance(selected_room, dict)
        else None
    )

    state.selected_room_id = result.get("selected_room_id")

    pending = result.get("pending_search")

    if isinstance(pending, dict):
        state.pending_search = pending

    return response


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "status": "success",
        "message": "Homilivo Room Search API is running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "active_sessions": len(SESSIONS),
    }


# Demo chat UI: a static page (no build step, no framework) that talks to
# the /chat endpoint below exactly like any other client - it holds no
# chatbot/room-search logic of its own. Served same-origin, so no CORS
# middleware is needed.
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/chat-ui")
def chat_ui():
    return FileResponse(_STATIC_DIR / "chat_ui.html")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    # Token pasted in the Authorize box -> bound to THIS request's task
    # only (see app/server.py's ContextVar) -> made available to the MCP
    # room search. This applies to every mode, since fast/qwen/gemma/groq
    # all run through here. Never logged, never written to a process-global.
    set_kangaroo_token(credentials.credentials)

    session_id = request.session_id or str(uuid4())

    # request.mode is already validated to be one of ChatMode by Pydantic.
    mode = request.mode

    state = SESSIONS.get(session_id)

    state.conversation_history.append(
        {
            "role": "user",
            "content": request.message,
        }
    )

    async with Client(mcp) as client:
        result = await process_message(
            user_query=request.message,
            mode=mode,
            client=client,
            state=state,
        )

    response_text = _apply_result_to_state(state, result)

    SESSIONS.save(session_id, state)

    rooms = [
        _structure_room(r)
        for r in state.previous_rooms
    ]

    selected = (
        _structure_room(state.selected_room)
        if state.selected_room
        else None
    )

    return ChatResponse(
        success=response_text is not None,
        session_id=session_id,
        mode=mode,
        message=response_text
        or "Sorry, I couldn't process that request. Please rephrase.",
        room_count=len(rooms),
        rooms=rooms,
        selected_room=selected,
        pending_clarification=bool(state.pending_search),
    )


@app.post("/chat/reset")
def reset_session(session_id: str):
    existed = SESSIONS.reset(session_id)

    return {
        "status": "success",
        "session_id": session_id,
        "existed": existed,
    }