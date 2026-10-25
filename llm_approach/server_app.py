"""FastAPI server for the LLM-decision pipeline (port 8001).

Speaks the same protocol as the current backend (``app/api.py``) so the website
works unchanged: ``GET /health``, ``/ready``, ``/chat-ui/demo-token``,
``POST /chat``, ``POST /chat/reset`` and ``WS /ws/chat?token=``. Every turn goes
through ``pipeline.handle_turn``; the ``mode`` field is validated and ignored.

Run with ``--no-access-log``: the WebSocket token travels in the query string and
uvicorn's access log would record it. As a second line of defence a filter also
scrubs ``token=...`` from that logger. Nothing here logs a URL, token or message.

Token handling mirrors the current backend and is not a new security choice: any
non-empty token connects; a valid demo token is redeemed for the server credential.
The rooms API always uses ``KANGAROO_TOKEN`` from the server environment.

Heavy imports (torch) happen only inside ``pipeline.preload()`` and the knowledge-base
preload, unless ``LLM_APPROACH_SKIP_PRELOAD=1``. They run in the background: the server listens at once,
``/health`` answers 200 from the first second, ``/ready`` and ``/chat`` return 503 until both are loaded.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from pathlib import Path

from llm_approach import config, observability, pipeline, responder, reuse, rooms_client, switches, turn_context
from llm_approach.security.secrets_check import validate_secrets

logger = logging.getLogger(__name__)

SERVER_ERROR_REPLY = "Something went wrong on our side, please try again."
WS_MAX_MESSAGE_LENGTH = 500
VALID_MODES = ("fast", "qwen", "gemma", "groq")
SKIP_PRELOAD_ENV = "LLM_APPROACH_SKIP_PRELOAD"
SKIP_SECRET_CHECK_ENV = "LLM_APPROACH_SKIP_SECRET_CHECK"  # tests only: never set it on a real server
# The server always runs configure_production(), so the variable alone never skips the check. Only the test
# suite flips this switch (tests/conftest.py); nothing in the app does.
ALLOW_SECRET_CHECK_SKIP = False
CORS_ENV = "LLM_APPROACH_CORS_ORIGINS"
METRICS_TOKEN_ENV = "LLM_APPROACH_METRICS_TOKEN"  # optional: without it /metrics is off (404)

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z", re.IGNORECASE)  # \Z, not $: "$" would accept a trailing newline
_TOKEN_IN_TEXT = re.compile(r"(token=)[^&\s\"']*", re.IGNORECASE)

REPO_ROOT = Path(__file__).resolve().parents[1]  # same .env lookup as cli.py


def load_env(path: Path | str | None = None) -> bool:
    """Load the repo-root ``.env`` into ``os.environ`` without overriding variables that are
    already set. Returns whether a file was read. Logs and prints nothing from the file."""
    try:
        from dotenv import load_dotenv

        return bool(load_dotenv(Path(path) if path else REPO_ROOT / ".env", override=False))
    except Exception as exc:  # noqa: BLE001 - class name only, never file content
        logger.error(".env could not be loaded: %s", type(exc).__name__)
        return False


# Before anything below reads the environment (rate limiter, demo auth, LLM provider, rooms client).
load_env()

_rl = reuse.get_rate_limit_module()
_demo = reuse.get_demo_auth_module()

# What /ready reports. Set by the startup preload.
_readiness = {"guardrails": False, "knowledge_base": False}
# True only while the background preload runs. Chat is refused (503) while it is set; once it ends, even after a
# failed preload, chat is served as before (the models then load on first use). Never set when the preload is skipped.
_startup = {"loading": False}
STARTING_REPLY = "I'm just starting up. Please try again in a moment."
STARTING_RETRY_AFTER_S = 5


class _RedactTokenFilter(logging.Filter):
    """Scrub ``token=...`` from a record (defence in depth; run with --no-access-log)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            text = record.getMessage()
        except Exception:
            return True
        if "token=" in text.lower():
            record.msg, record.args = _TOKEN_IN_TEXT.sub(r"\1REDACTED", text), ()
        return True


def _llm_key_present() -> bool:
    try:
        return config.key_present()
    except ValueError:  # bad LLM_APPROACH_PROVIDER: not ready
        return False


def _llm_label() -> dict[str, str]:
    provider, model = config.label()
    return {"provider": provider, "model": model}


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def _preload_blocking() -> None:
    """Load the guardrails and the knowledge base once. Failures are logged by class only."""
    started = time.monotonic()
    try:
        pipeline.preload()
        _readiness["guardrails"] = True
        logger.info("startup step=guardrails_and_gate seconds=%.1f", time.monotonic() - started)
    except Exception as exc:  # noqa: BLE001
        logger.error("guardrail preload failed: %s", type(exc).__name__)
    started = time.monotonic()
    try:
        reuse.get_knowledge_preload()()
        _readiness["knowledge_base"] = True
        logger.info("startup step=knowledge_base seconds=%.1f", time.monotonic() - started)
    except Exception as exc:  # noqa: BLE001
        logger.error("knowledge base preload failed: %s", type(exc).__name__)


async def _preload_background() -> None:
    """Run the blocking preload off the event loop; always clear the loading flag. Never raises."""
    try:
        await asyncio.to_thread(_preload_blocking)
    except Exception as exc:  # noqa: BLE001
        logger.error("startup preload crashed: %s", type(exc).__name__)
    finally:
        _startup["loading"] = False


def _still_loading() -> list[str]:
    """Names of the components not loaded yet, while the background preload runs (else empty)."""
    if not _startup["loading"]:
        return []
    return [name for name, loaded in _readiness.items() if not loaded]


def _starting_response() -> JSONResponse | None:
    """The polite 503 for a chat request that arrives before the models are loaded."""
    if not _startup["loading"]:
        return None
    return JSONResponse(
        status_code=503,
        content={"success": False, "status": "starting", "message": STARTING_REPLY, "detail": STARTING_REPLY},
        headers={"Retry-After": str(STARTING_RETRY_AFTER_S)},
    )


def _metrics_token() -> str:
    """The dedicated metrics token, or "" when metrics are off. Blank or whitespace counts as not set."""
    return (os.getenv(METRICS_TOKEN_ENV) or "").strip()


def _metrics_authorized(request: Request, expected: str) -> bool:
    """Only the dedicated metrics token opens /metrics: not a demo token, not the server credential."""
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token or not expected:
        return False
    return hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if _env_flag(SKIP_SECRET_CHECK_ENV) and not ALLOW_SECRET_CHECK_SKIP:
        logger.error("%s is ignored in production mode: the secret check still runs", SKIP_SECRET_CHECK_ENV)
    if _env_flag(SKIP_SECRET_CHECK_ENV) and ALLOW_SECRET_CHECK_SKIP:
        logger.warning("secret check skipped by %s: do not use this on a real server", SKIP_SECRET_CHECK_ENV)
    else:
        validate_secrets()  # refuses to start when the provider key or KANGAROO_TOKEN is missing (names only, never values)
    if not _metrics_token():
        logger.warning("%s is not set: /metrics is disabled (404)", METRICS_TOKEN_ENV)
    before = (pipeline._full_guardrails, switches.snapshot())
    pipeline.get_store()  # fail fast on a bad STATE_BACKEND or a missing redis package; Redis being down is not fatal
    logger.info("session store backend=%s", "redis" if pipeline.get_store().is_shared else "memory")
    pipeline.configure_production()  # full guardrails + room data check + search-argument check, for this app's lifetime
    preload_task = None
    try:
        logging.getLogger("uvicorn.access").addFilter(_RedactTokenFilter())
        provider, model = config.label()
        logger.info("llm provider=%s model=%s key_present=%s", provider, model, _llm_key_present())  # never the key
        if _env_flag(SKIP_PRELOAD_ENV):
            logger.info("startup preload skipped by %s", SKIP_PRELOAD_ENV)
        else:
            _startup["loading"] = True  # listen now; /ready and /chat answer 503 until the background load ends
            preload_task = asyncio.create_task(_preload_background())
        yield
    finally:
        if preload_task is not None:
            await preload_task  # a shutdown during the load waits for it: the worker thread cannot be cancelled
        pipeline.configure(full_guardrails=before[0])
        switches.configure(**before[1])


# ---- models (same fields and types as the current backend) ---------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=WS_MAX_MESSAGE_LENGTH)
    session_id: Optional[str] = None
    mode: Literal["fast", "qwen", "gemma", "groq"] = "fast"


class RoomResponse(BaseModel):
    id: str
    title: str
    city: str
    location: Optional[str] = None
    price_per_month: Optional[float] = None
    room_type: Optional[str] = None
    ideal_for: list[str] = []
    amenities: list[str] = []


class ChatResponse(BaseModel):
    success: bool
    session_id: str
    request_id: str
    mode: str
    message: str
    room_count: int = 0
    rooms: list[RoomResponse] = []
    selected_room: Optional[RoomResponse] = None
    pending_clarification: bool = False
    sources: list[str] = []
    search_performed: bool = False


# ---- helpers -------------------------------------------------------------------


def normalize_session_id(candidate: Any) -> str:
    """A UUID-format id (lowercased), else a fresh one. The invalid value is never logged."""
    if candidate is None or candidate == "":
        return str(uuid.uuid4())
    if isinstance(candidate, str) and _SESSION_ID_RE.match(candidate):
        return candidate.lower()
    logger.warning("Invalid session_id format received - generating new session_id")
    return str(uuid.uuid4())


async def run_turn(session_id: str, message: str, mode: str = "fast") -> pipeline.TurnResult:
    """One pipeline turn. Never raises: any escape becomes a normal bot message."""
    turn_context.set_requested(str(uuid.uuid4()), mode)  # read by handle_turn (observability) and build_done
    try:
        turn = await pipeline.handle_turn(session_id, message)
        store = pipeline.get_store()
        if store.is_shared and not turn.state_resolved:  # a shared store is a network read: keep it off the event loop
            turn.state, turn.state_resolved = await asyncio.to_thread(store.peek, session_id), True
        return turn
    except Exception as exc:  # noqa: BLE001
        logger.error("turn failed outside the pipeline: %s", type(exc).__name__)
        return pipeline.TurnResult(reply=SERVER_ERROR_REPLY, rooms=None, success=False)


def build_done(turn: pipeline.TurnResult, session_id: str, mode: str) -> dict[str, Any]:
    """The ChatResponse payload (also the ``done`` event body), field names as the frontend reads them."""
    state = turn.state if turn.state_resolved else pipeline.get_store().peek(session_id)
    on_screen = list(state.rooms_on_screen) if state is not None else []
    rooms = turn.rooms if turn.rooms is not None else responder.room_cards(on_screen)
    selected = state.selected_room() if state is not None else None
    return {
        "success": bool(turn.success),
        "session_id": session_id,
        "request_id": turn_context.requested_request_id() or str(uuid.uuid4()),  # the id pipeline.handle_turn recorded
        "mode": mode,
        "message": turn.reply,
        "room_count": len(rooms),
        "rooms": rooms,
        "selected_room": responder.room_card(selected) if selected else None,
        "pending_clarification": bool(state is not None and state.pending_question),
        "sources": [],
        "search_performed": bool((turn.metrics or {}).get("tool_called")),
    }


def chunk_text(text: str) -> list[str]:
    """Whitespace-split chunks that concatenate back to the text."""
    return re.findall(r"\S+\s*", text) or [text]


def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    """A finite float >= ``minimum`` from the environment, else the default."""
    try:
        value = float(os.getenv(name, default))
    except ValueError:
        return default
    return value if value >= minimum and value == value and value != float("inf") else default


def _short(value: Any, limit: int = 32) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---- app -----------------------------------------------------------------------

_bearer = HTTPBearer(description="Homilivo API token (a demo token or the caller's own)")


def create_app() -> FastAPI:
    app = FastAPI(title="Homilivo LLM-approach API", version="0.1.0", debug=False, lifespan=_lifespan)

    origins = [o.strip() for o in os.getenv(CORS_ENV, "").split(",") if o.strip()]
    if origins:  # off by default: the current backend has no CORS middleware either
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"]
        )

    @app.get("/")
    def root():
        return {"status": "success", "message": "Homilivo LLM-approach API is running"}

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "homilivo-chatbot-llm"}

    @app.get("/ready")
    def ready():
        checks = {
            "llm_key_present": _llm_key_present(),  # the active provider's key; booleans only, never values
            "kangaroo_token_present": bool(os.getenv("KANGAROO_TOKEN")),
            "knowledge_base": _readiness["knowledge_base"],
            "guardrails": _readiness["guardrails"],
        }
        # The rooms token is reported but not required: offline-fixture runs work without it.
        is_ready = all(v for k, v in checks.items() if k != "kangaroo_token_present")
        content = {"status": "ready" if is_ready else "not_ready", "checks": checks, **_llm_label()}
        loading = _still_loading()
        if loading:
            content["loading"] = loading  # which components the background load has not finished
            content["message"] = "Starting up: still loading " + ", ".join(loading) + "."
        store = pipeline.get_store()
        if hasattr(store, "ping"):  # Redis sessions: report the state. Not part of "ready": with Redis down the
            # turns still answer (from a fresh state each time), so a load balancer should not drain the server.
            reachable = store.ping()
            content["session_store"] = {"backend": "redis", "redis_reachable": reachable,
                                        "sessions_survive_restart": reachable}
        return JSONResponse(status_code=200 if is_ready else 503, content=content)

    @app.get("/metrics")
    async def metrics(request: Request, _limit: None = Depends(_rl.enforce_rate_limit)):
        """Counters and latency stats, JSON. Opened only by LLM_APPROACH_METRICS_TOKEN; 404 when it is not set.
        Fixed words and numbers only: no message text, no room contents, no keys, no session ids."""
        expected = _metrics_token()
        if not expected:
            raise HTTPException(status_code=404, detail="Not Found")
        if not _metrics_authorized(request, expected):
            raise HTTPException(status_code=401, detail="Authentication required.", headers={"WWW-Authenticate": "Bearer"})
        try:
            active = await asyncio.to_thread(pipeline.get_store().count)
        except Exception:  # noqa: BLE001
            active = None
        return {"active_sessions": active, **observability.get_metrics_snapshot()}

    @app.get("/chat-ui/demo-token")
    def demo_token():
        token = _demo.issue_demo_token()
        if token is None:
            raise HTTPException(status_code=404, detail="No server-side demo credential is configured.")
        return {"token": token}

    def _session_limit(session_id: str) -> None:
        result = _rl.check_session_rate_limit(session_id)
        if not result.allowed:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please slow down and try again shortly.",
                headers={"Retry-After": str(max(1, int(result.retry_after_seconds) + 1))},
            )

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(_rl.enforce_rate_limit)])
    async def chat(request: ChatRequest, credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
        starting = _starting_response()
        if starting is not None:
            return starting
        # Same as the current backend: the bearer value is a demo token (redeemed for the server credential) or
        # the caller's own Homilivo token. It is bound to this request only and never stored or logged.
        handle = rooms_client.set_rooms_token(_demo.redeem_demo_token(credentials.credentials) or credentials.credentials)
        try:
            session_id = normalize_session_id(request.session_id)
            if request.session_id:
                _session_limit(session_id)
            turn = await run_turn(session_id, request.message, request.mode)
            return ChatResponse(**build_done(turn, session_id, request.mode))
        finally:
            rooms_client.reset_rooms_token(handle)

    @app.post("/chat/reset")
    def reset(session_id: str):
        existed = pipeline.reset_session(session_id) if _SESSION_ID_RE.match(session_id) else False
        return {"status": "success", "session_id": session_id, "existed": existed}

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket) -> None:
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=4401, reason="Missing token")
            return

        client_key = _rl.client_host_identifier(websocket.client.host if websocket.client else None)
        if not _rl.check_rate_limit(client_key).allowed:
            await websocket.close(code=4429, reason="Rate limit exceeded")
            return

        await websocket.accept()
        rooms_token = _demo.redeem_demo_token(token) or token  # held for this connection only, never logged
        session_id: str | None = None

        async def send_error(detail: str) -> None:
            await websocket.send_json({"event": "error", "detail": detail})

        try:
            while True:
                try:
                    payload = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=_float_env("WS_HEARTBEAT_INTERVAL_SECONDS", 20.0, minimum=0.01),
                    )
                except asyncio.TimeoutError:
                    await websocket.send_json({"event": "ping"})
                    continue
                except (ValueError, KeyError):  # not JSON, or a binary frame (no "text" key)
                    await send_error("Message must be valid JSON.")
                    continue

                if not isinstance(payload, dict):
                    await send_error("Message must be a JSON object.")
                    continue

                message = payload.get("message")
                mode = payload.get("mode", "fast")
                supplied = payload.get("session_id")
                session_id = normalize_session_id(supplied) if supplied else (session_id or str(uuid.uuid4()))

                if not isinstance(message, str) or not message.strip() or len(message) > WS_MAX_MESSAGE_LENGTH:
                    await send_error(
                        f"message must be a non-empty string of at most {WS_MAX_MESSAGE_LENGTH} characters."
                    )
                    continue
                if mode not in VALID_MODES:
                    await send_error(f"Invalid mode: {_short(mode)}")
                    continue
                if not _rl.check_rate_limit(client_key).allowed or not _rl.check_session_rate_limit(session_id).allowed:
                    await send_error("Too many requests. Please slow down.")
                    continue

                if _startup["loading"]:  # not ready yet: tell the user, keep the connection
                    await send_error(STARTING_REPLY)
                    continue
                await websocket.send_json({"event": "typing"})
                handle = rooms_client.set_rooms_token(rooms_token)
                try:
                    turn = await run_turn(session_id, message, mode)
                finally:
                    rooms_client.reset_rooms_token(handle)
                done = build_done(turn, session_id, mode)

                delay = _float_env("STREAM_CHUNK_DELAY_SECONDS", 0.02)
                for chunk in chunk_text(turn.reply):
                    await websocket.send_json({"event": "token", "text": chunk})
                    if delay > 0:
                        await asyncio.sleep(delay)
                await websocket.send_json({"event": "done", **done})
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 - send after a disconnect, a failure building the reply
            logger.error("websocket closed after an error: %s", type(exc).__name__)
            try:
                await websocket.close(code=1011)
            except Exception:  # noqa: BLE001 - already closed
                pass

    return app


app = create_app()
