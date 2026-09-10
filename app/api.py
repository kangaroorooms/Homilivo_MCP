import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal
from uuid import uuid4

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

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from mcp import Client
from app.server import mcp, set_kangaroo_token
from app.rate_limit import check_rate_limit, client_host_identifier, enforce_rate_limit
from app.demo_auth import issue_demo_token, redeem_demo_token, server_credential_configured
from app.streaming import chunk_for_streaming
from app.observability import get_metrics_snapshot, record_turn
from app.feedback_store import record_feedback

# The real conversational engine (routing, follow-ups, refinements,
# clarifications, FAST/QWEN3/GEMMA, embedding fallback) lives here.
from app.experiments.agent_modes import process_message, ConversationState
from app.session_store import SessionStore


# ============================================================
# FASTAPI APP
# ============================================================

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Load the shared embedding model, plus every module's precomputed
    # embeddings (interpreter's operation anchors, the input guardrail's
    # intent examples, the knowledge base's document index), once at
    # boot so the first user request isn't slow. Replaces the
    # deprecated @app.on_event("startup") hook.
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

    try:
        from app.knowledge.store import warmup as warmup_knowledge
        warmup_knowledge()
    except Exception as error:  # noqa: BLE001
        print(f"⚠ knowledge base warmup skipped: {error}")

    yield


app = FastAPI(
    title="Homilivo Room Search API",
    description="FastAPI service for the Homilivo MCP AI agent",
    version="2.0.0",
    lifespan=_lifespan,
)

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
    mode: ChatMode = Field("fast", description="fast | qwen | gemma | groq")


class RoomResponse(BaseModel):
    id: str
    title: str
    city: str
    location: str | None = None
    price_per_month: float | None = None
    room_type: str | None = None
    ideal_for: list[str] = []
    amenities: list[str] = []


class ChatResponse(BaseModel):
    success: bool
    session_id: str
    request_id: str = Field(
        description=(
            "Identifies this turn for observability (app/observability.py) "
            "and for POST /feedback, which tags feedback back to the turn "
            "it refers to."
        ),
    )
    mode: str
    message: str
    room_count: int = 0
    rooms: list[RoomResponse] = []
    selected_room: RoomResponse | None = None
    pending_clarification: bool = False
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Knowledge-base document titles the message was grounded in, "
            "when it answered a platform question (verification, fees, "
            "contacting an owner, booking) rather than a live search. "
            "Empty for room-search/follow-up responses."
        ),
    )
    search_performed: bool = Field(
        False,
        description=(
            "True only when a room search actually completed this turn. "
            "Lets a client distinguish a genuine zero-result search from "
            "any other turn (greeting, platform question, follow-up) "
            "that also happens to carry an empty/unchanged rooms list."
        ),
    )


class FeedbackRequest(BaseModel):
    request_id: str = Field(
        ...,
        min_length=1,
        description="The request_id returned by /chat or /ws/chat's done event for the turn being rated.",
    )
    rating: Literal["up", "down"]
    note: str | None = Field(
        None,
        max_length=500,
        description="Optional free-text note. Never put secrets or PII here.",
    )


class FeedbackResponse(BaseModel):
    status: str
    request_id: str


# ============================================================
# HELPERS
# ============================================================

def _structure_room(room: dict) -> dict:
    price = room.get("pricePerMonth")
    if price is None:
        price = room.get("price")

    # The Homilivo API's exact amenities field name is not documented
    # anywhere in this repo (see app/experiments/agent_modes.py's
    # answer_room_fact(), which checks the same set of candidates for
    # the same reason). Whichever of these is present wins; none
    # present just means no amenities to show.
    amenities = None
    for field in ("amenities", "amenity", "facilities", "facility", "features"):
        value = room.get(field)
        if value is not None:
            amenities = value
            break
    if isinstance(amenities, str):
        amenities = [amenities]
    elif not isinstance(amenities, list):
        amenities = []

    return {
        "id": str(room.get("id")),
        "title": room.get("title", "Untitled Room"),
        "city": room.get("city", "Unknown"),
        "location": room.get("location"),
        "price_per_month": price,
        "room_type": room.get("roomType"),
        "ideal_for": room.get("idealFor", []),
        "amenities": amenities,
    }


def _apply_result_to_state(state: ConversationState, result: dict) -> str | None:
    """Mirror the memory-update logic from the terminal loop."""
    response = result.get("response")

    if response:
        state.conversation_history.append({"role": "assistant", "content": response})

    rooms = result.get("rooms")
    if isinstance(rooms, list):
        state.previous_rooms = rooms

    search = result.get("search")
    if isinstance(search, dict):
        state.previous_search = search

    selected_room = result.get("selected_room")
    state.selected_room = selected_room if isinstance(selected_room, dict) else None
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
    return {"status": "success", "message": "Homilivo Room Search API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "active_sessions": len(SESSIONS)}


@app.get("/knowledge/health")
def knowledge_health():
    from app.knowledge.store import index_stats

    stats = index_stats()
    return {
        "status": "healthy" if stats["embedding_model_available"] else "degraded",
        **stats,
    }


@app.get("/metrics")
def metrics():
    """
    Lightweight in-memory metrics snapshot (request/error counts,
    tool failures, rate-limit events, API/LLM/turn latency stats).
    Not Prometheus text format - JSON, matching this project's
    "no heavyweight monitoring stack" scope. Resets on process
    restart, same as the rest of this demo's in-memory state.
    """
    return {"active_sessions": len(SESSIONS), **get_metrics_snapshot()}


# Demo chat UI: a static page (no build step, no framework) that talks to
# the /chat endpoint below exactly like any other client - it holds no
# chatbot/room-search logic of its own. Served same-origin, so no CORS
# middleware is needed.
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/chat-ui")
def chat_ui():
    return FileResponse(_STATIC_DIR / "chat_ui.html")


@app.get("/chat-ui/demo-token")
def chat_ui_demo_token():
    """
    Opt-in, opaque credential exchange for the bundled demo page - see
    app/demo_auth.py. Lets /chat-ui skip asking a presenter to paste a
    real Homilivo token, WITHOUT weakening /chat's own Authorization
    requirement or exposing the real server-side credential to the
    browser. 404s (rather than 200 with a null token) when no
    server-side credential is configured, so the demo page can fall
    back to its manual-token flow without guessing.
    """
    token = issue_demo_token()

    if token is None:
        raise HTTPException(
            status_code=404,
            detail="No server-side demo credential is configured.",
        )

    return {"token": token}


async def _run_chat_turn(session_id: str, mode: str, message: str) -> dict:
    """
    One full turn through the conversational engine, shared by the
    REST /chat endpoint and the /ws/chat WebSocket below so neither
    duplicates or forks the session/state-update logic. The caller is
    responsible for binding the Homilivo credential (set_kangaroo_token)
    before calling this.

    Also the single place a turn is recorded for observability
    (app/observability.py) - request id, intent/operation/tool, mode,
    status, latency. No message text or credentials are logged.
    """
    request_id = str(uuid4())
    started_at = time.monotonic()

    state = SESSIONS.get(session_id)
    state.conversation_history.append({"role": "user", "content": message})

    try:
        async with Client(mcp) as client:
            result = await process_message(
                user_query=message,
                mode=mode,
                client=client,
                state=state,
            )
    except Exception:  # noqa: BLE001
        record_turn(
            request_id=request_id,
            session_id=session_id,
            intent=None,
            operation=None,
            tool=None,
            mode=mode,
            status="error",
            error_category="unhandled_exception",
            latency_ms=(time.monotonic() - started_at) * 1000,
        )
        raise

    response_text = _apply_result_to_state(state, result)
    SESSIONS.save(session_id, state)

    # A turn can ask to display a filtered subset of the current room
    # set (e.g. "which rooms have AC?") without that filter becoming
    # the new session-wide room set - state.previous_rooms (used for
    # ordinal/rank follow-ups) is untouched either way.
    displayed_rooms = result.get("displayed_rooms")
    rooms_to_show = (
        displayed_rooms
        if isinstance(displayed_rooms, list)
        else state.previous_rooms
    )

    rooms = [_structure_room(r) for r in rooms_to_show]
    selected = _structure_room(state.selected_room) if state.selected_room else None
    sources = result.get("sources")

    status = "success" if response_text is not None else "blocked"

    record_turn(
        request_id=request_id,
        session_id=session_id,
        intent=result.get("intent"),
        operation=result.get("operation"),
        tool=result.get("tool"),
        mode=mode,
        status=status,
        error_category=None if status == "success" else "blocked_or_unclassified",
        latency_ms=(time.monotonic() - started_at) * 1000,
    )

    return {
        "success": response_text is not None,
        "session_id": session_id,
        "request_id": request_id,
        "mode": mode,
        "message": response_text or "Sorry, I couldn't process that request. Please rephrase.",
        "room_count": len(rooms),
        "rooms": rooms,
        "selected_room": selected,
        "pending_clarification": bool(state.pending_search),
        "sources": sources if isinstance(sources, list) else [],
        "search_performed": bool(result.get("search_performed")),
    }


def _resolve_homilivo_token(bearer_value: str) -> str:
    """
    If the caller's bearer value is a currently-valid opaque demo
    token (app/demo_auth.py), redeem it for the server-side credential
    instead - the demo page never sees the real secret. Any other
    value (a real Homilivo token) passes through unchanged: the BYOT
    contract is unaffected. Never logged.
    """
    return redeem_demo_token(bearer_value) or bearer_value


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(enforce_rate_limit)])
async def chat_endpoint(
    request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    # Token pasted in the Authorize box (or demo-page equivalent) ->
    # bound to THIS request's task only (see app/server.py's
    # ContextVar) -> made available to the MCP room search. This
    # applies to every mode, since fast/qwen/gemma/groq all run
    # through here. Never logged, never written to a process-global.
    set_kangaroo_token(_resolve_homilivo_token(credentials.credentials))

    session_id = request.session_id or str(uuid4())
    # request.mode is already validated to be one of ChatMode by Pydantic.
    turn = await _run_chat_turn(session_id, request.mode, request.message)

    return ChatResponse(**turn)


@app.post("/chat/reset")
def reset_session(session_id: str):
    existed = SESSIONS.reset(session_id)
    return {"status": "success", "session_id": session_id, "existed": existed}


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(feedback: FeedbackRequest):
    """
    Thumbs up/down on a turn, tagged with the request_id already
    returned by /chat and /ws/chat's "done" event (see
    app.api._run_chat_turn and app/observability.py). Stored append-only
    (app/feedback_store.py) - no new infra, never secrets/PII.
    """
    record_feedback(feedback.request_id, feedback.rating, feedback.note)
    return FeedbackResponse(status="success", request_id=feedback.request_id)


# ============================================================
# WEBSOCKET STREAMING (additive - /chat above is unchanged and stays
# the reliable fallback if a client cannot use WebSockets)
# ============================================================
# Protocol: connect to /ws/chat?token=<bearer>, then send one JSON
# object per turn: {"message": str, "session_id": str | null, "mode":
# "fast"|"qwen"|"gemma"|"groq"}. The server replies with a sequence of
# JSON events for that turn:
#   {"event": "typing"}
#   {"event": "token", "text": "..."}   (repeated, in order)
#   {"event": "done", ...same fields as ChatResponse...}
# or, on a problem with that specific turn (not the connection):
#   {"event": "error", "detail": "..."}
# During any quiet period longer than WS_HEARTBEAT_INTERVAL_SECONDS
# (no turn in flight), the server also sends {"event": "ping"} so the
# client can detect a dead connection; clients should ignore it.
# A browser's WebSocket API cannot set an Authorization header, so the
# token travels as a query parameter here instead - either a real BYOT
# Homilivo token or an opaque demo token (app/demo_auth.py), exactly
# like /chat's Authorization header accepts either.
#
# Full contract (connect URL, every event shape, close codes, reconnect
# expectations, REST fallback parity) is documented for frontend
# integrators in docs/WEBSOCKET_CONTRACT.md (v1) - keep both in sync.

WS_MAX_MESSAGE_LENGTH = 500
STREAM_CHUNK_DELAY_SECONDS = float(os.getenv("STREAM_CHUNK_DELAY_SECONDS", "0.02"))
WS_HEARTBEAT_INTERVAL_SECONDS = float(os.getenv("WS_HEARTBEAT_INTERVAL_SECONDS", "20"))
_WS_VALID_MODES = {"fast", "qwen", "gemma", "groq"}


@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket) -> None:
    token = websocket.query_params.get("token")

    if not token:
        # No standard "401" close code exists for WebSockets; 4401 is
        # in the app-reserved range (4000-4999) and documented in
        # app/static/chat_ui.html's reconnect handling.
        await websocket.close(code=4401, reason="Missing token")
        return

    client_key = client_host_identifier(
        websocket.client.host if websocket.client else None
    )
    rate_limit_result = check_rate_limit(client_key)

    if not rate_limit_result.allowed:
        await websocket.close(code=4429, reason="Rate limit exceeded")
        return

    await websocket.accept()

    session_id: str | None = None

    try:
        while True:
            try:
                payload = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=WS_HEARTBEAT_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                # Heartbeat: proves the connection (and this handler)
                # is still alive during a quiet period, and gives the
                # frontend something to detect a truly dead socket.
                await websocket.send_json({"event": "ping"})
                continue
            except ValueError:
                await websocket.send_json(
                    {"event": "error", "detail": "Message must be valid JSON."}
                )
                continue

            if not isinstance(payload, dict):
                await websocket.send_json(
                    {"event": "error", "detail": "Message must be a JSON object."}
                )
                continue

            message = payload.get("message")
            mode = payload.get("mode", "fast")
            session_id = payload.get("session_id") or session_id or str(uuid4())

            if (
                not isinstance(message, str)
                or not message.strip()
                or len(message) > WS_MAX_MESSAGE_LENGTH
            ):
                await websocket.send_json(
                    {
                        "event": "error",
                        "detail": f"message must be a non-empty string of at most {WS_MAX_MESSAGE_LENGTH} characters.",
                    }
                )
                continue

            if mode not in _WS_VALID_MODES:
                await websocket.send_json(
                    {"event": "error", "detail": f"Invalid mode: {mode!r}"}
                )
                continue

            turn_rate_limit = check_rate_limit(client_key)
            if not turn_rate_limit.allowed:
                await websocket.send_json(
                    {"event": "error", "detail": "Too many requests. Please slow down."}
                )
                continue

            await websocket.send_json({"event": "typing"})

            set_kangaroo_token(_resolve_homilivo_token(token))

            try:
                turn = await _run_chat_turn(session_id, mode, message)
            except Exception:  # noqa: BLE001 - never leak a stack trace to the client
                await websocket.send_json(
                    {
                        "event": "error",
                        "detail": "Something went wrong processing that message. Please try again.",
                    }
                )
                continue

            for chunk in chunk_for_streaming(turn["message"]):
                await websocket.send_json({"event": "token", "text": chunk})
                if STREAM_CHUNK_DELAY_SECONDS > 0:
                    await asyncio.sleep(STREAM_CHUNK_DELAY_SECONDS)

            await websocket.send_json({"event": "done", **turn})

    except WebSocketDisconnect:
        pass