"""Settings for the LLM decision pipeline. Safe to import without a key."""

from __future__ import annotations

import datetime as dt
import logging
import os

from llm_approach import reuse

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.6
LLM_TIMEOUT_S = 8  # Groq only; Gemini has its own timeout below
GEMINI_TIMEOUT_ENV = "LLM_APPROACH_GEMINI_TIMEOUT_S"
GEMINI_DEFAULT_TIMEOUT_S = 20.0
GEMINI_MIN_TIMEOUT_S = 10.0  # Gemini rejects a request deadline under 10 s (400 INVALID_ARGUMENT)
GEMINI_TURN_LLM_BUDGET_S = 25.0  # Gemini: no further call in a turn once its calls have used this much time
LLM_MAX_TOKENS = 500
TEMPERATURE = 0
REASONING_EFFORT: str | None = "low"  # gpt-oss: keeps reasoning from eating max_tokens
MAX_LLM_CALLS_PER_TURN = 2
MAX_TOOL_CALLS_PER_TURN = 1
ANSWERER_MAX_CALLS = 2  # CONVERSATION_QUESTION: the answer call and one retry, on top of the decision calls
ANSWERER_MAX_TOKENS = 500
ANSWERER_CONTEXT_CHARS = 24000  # about 6,000 tokens of chat record per answer call
ANSWERER_HEAD_TURNS = 20  # beyond ANSWERER_TURN_LIMIT turns: keep this many oldest ...
ANSWERER_TAIL_TURNS = 100  # ... and this many newest
ANSWERER_TURN_LIMIT = 120
HISTORY_TURNS_IN_PROMPT = 6  # one turn = user message + bot reply
MAX_ROOMS_IN_PROMPT = 10
MAX_USER_MESSAGE_CHARS = 500
LAST_BOT_MESSAGE_CHARS = 300
HISTORY_MESSAGE_CHARS = 200


def _int_env(name: str, default: int) -> int:
    """A positive integer from the environment, else the default (a bad value is ignored, never echoed)."""
    try:
        value = int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


# Mirrors app/session_store.py and app/experiments/agent_modes.py: same env
# vars, same defaults (not imported, because those modules are heavy).
SESSION_TTL_SECONDS = _int_env("SESSION_TTL_SECONDS", 3600)
SESSION_MAX_COUNT = _int_env("SESSION_MAX_COUNT", 1000)
SESSION_MAX_HISTORY_MESSAGES = _int_env("SESSION_MAX_HISTORY_MESSAGES", 40)
MAX_LLM_CONTEXT_CHARS = 8000

# Session storage. STATE_BACKEND and REDIS_URL are the variables app/session_store.py reads (same names, same
# defaults); memory stays the default. The lock settings are new and v2-only.
STATE_BACKEND_ENV = "STATE_BACKEND"
STATE_BACKENDS = ("memory", "redis")
REDIS_URL_ENV = "REDIS_URL"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
REDIS_TIMEOUT_ENV = "LLM_APPROACH_REDIS_TIMEOUT_S"  # connect and read timeout; a down Redis must not stall a turn
LOCK_TTL_ENV = "LLM_APPROACH_SESSION_LOCK_TTL_MS"  # life of the Redis lock key; refreshed while the turn runs
LOCK_WAIT_ENV = "LLM_APPROACH_SESSION_LOCK_WAIT_S"  # longest a turn waits for the session (this server + Redis)
LOCK_MAX_HOLD_ENV = "LLM_APPROACH_SESSION_LOCK_MAX_HOLD_S"  # a turn that runs longer stops refreshing and loses the lock
DEFAULT_REDIS_TIMEOUT_S = 1.0
DEFAULT_LOCK_TTL_MS = 30_000
DEFAULT_LOCK_WAIT_S = 10.0  # also the bound of the in-memory store (was 30 s for Redis, unbounded in memory)
DEFAULT_LOCK_MAX_HOLD_S = 120.0  # above the 60 s Gemini 429 wait plus the 25 s call budget


def state_backend() -> str:
    """``STATE_BACKEND`` (memory or redis), read fresh each call; default memory. An unknown value is an error
    (it is not echoed), so a typo never silently falls back to memory on a multi-server deployment."""
    value = os.getenv(STATE_BACKEND_ENV, "").strip().lower() or "memory"
    if value not in STATE_BACKENDS:
        raise ValueError(f"{STATE_BACKEND_ENV} must be one of: {', '.join(STATE_BACKENDS)}")
    return value


def redis_url() -> str:
    """The Redis URL. May contain a password: never log it."""
    return os.getenv(REDIS_URL_ENV, "").strip() or DEFAULT_REDIS_URL


def _positive_float_env_default(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, "").strip())
    except ValueError:
        return default
    return value if value > 0 and value == value and value != float("inf") else default


def redis_timeout_s() -> float:
    return _positive_float_env_default(REDIS_TIMEOUT_ENV, DEFAULT_REDIS_TIMEOUT_S)


def session_lock_ttl_ms() -> int:
    return _int_env(LOCK_TTL_ENV, DEFAULT_LOCK_TTL_MS)


TURN_DEADLINE_ENV = "LLM_APPROACH_TURN_DEADLINE_S"  # the server's wall for one whole turn, lock wait included
DEFAULT_TURN_DEADLINE_S = 40.0  # below the usual 60 s proxy limit


def turn_deadline_s() -> float:
    """Seconds one turn may take end to end on the server (default 40; an unreadable or non-positive value uses it)."""
    return _positive_float_env_default(TURN_DEADLINE_ENV, DEFAULT_TURN_DEADLINE_S)


def session_lock_wait_s() -> float:
    return _positive_float_env_default(LOCK_WAIT_ENV, DEFAULT_LOCK_WAIT_S)


def session_lock_max_hold_s() -> float:
    return _positive_float_env_default(LOCK_MAX_HOLD_ENV, DEFAULT_LOCK_MAX_HOLD_S)


# Deployment mode and the server's edge settings (server_app.py). Development is the default so nothing changes for
# a developer; a real server sets LLM_APPROACH_ENV=production (docs off, stricter startup checks).
ENV_ENV = "LLM_APPROACH_ENV"
ENV_MODES = ("development", "production")
DEMO_TOKEN_ENABLED_ENV = "LLM_APPROACH_DEMO_TOKEN_ENABLED"
DEMO_TOKEN_LIMIT_ENV = "LLM_APPROACH_DEMO_TOKEN_RATE_LIMIT"  # per IP per minute for /chat-ui/demo-token
DEFAULT_DEMO_TOKEN_LIMIT = 10
TRUSTED_PROXY_HOPS_ENV = "LLM_APPROACH_TRUSTED_PROXY_HOPS"
COUNTER_PATH_ENV = "LLM_APPROACH_GEMINI_COUNTER_PATH"  # where the Gemini request counter file lives


def environment() -> str:
    """``LLM_APPROACH_ENV`` (development or production), default development. An unknown value is an error (not
    echoed), so a typo never silently leaves a production server in development mode."""
    value = os.getenv(ENV_ENV, "").strip().lower() or "development"
    if value not in ENV_MODES:
        raise ValueError(f"{ENV_ENV} must be one of: {', '.join(ENV_MODES)}")
    return value


def production_mode() -> bool:
    return environment() == "production"


def demo_token_enabled() -> bool:
    """``LLM_APPROACH_DEMO_TOKEN_ENABLED`` (default true). Only 0/false/no/off turn the endpoint off."""
    return os.getenv(DEMO_TOKEN_ENABLED_ENV, "").strip().lower() not in ("0", "false", "no", "off")


def demo_token_limit() -> int:
    return _int_env(DEMO_TOKEN_LIMIT_ENV, DEFAULT_DEMO_TOKEN_LIMIT)


def trusted_proxy_hops() -> int:
    """How many reverse proxies in front of the server append to X-Forwarded-For (default 0: use the TCP peer).
    A value that is not a non-negative integer is an error (not echoed): a typo must not silently trust nothing."""
    raw = os.getenv(TRUSTED_PROXY_HOPS_ENV, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(f"{TRUSTED_PROXY_HOPS_ENV} must be a non-negative integer")
    return value


# Rooms API (mirrors API_URL in app/server.py, which is heavy to import).
ROOMS_API_URL = os.getenv("LLM_APPROACH_ROOMS_API_URL", "https://homilivo.onrender.com/api/rooms")
ROOMS_API_TIMEOUT_S = 15.0
ROOMS_RETRY_JITTER_S = (2.0, 4.0)  # one retry, timeout or connection error only
ROOMS_LIMIT = 10
ROOMS_SORT = "latest"
SEARCH_TIMEOUT_ENV = "LLM_APPROACH_SEARCH_TIMEOUT_SECONDS"  # whole search, retry included (app/server.py MCP_TOOL_TIMEOUT_SECONDS)
DEFAULT_SEARCH_TIMEOUT_S = 15.0


def search_timeout_s() -> float:
    """The most one rooms search may take end to end, read fresh each call. A bad value falls back to 15 s."""
    return _positive_float_env_default(SEARCH_TIMEOUT_ENV, DEFAULT_SEARCH_TIMEOUT_S)


# How the rooms search reaches the Rooms API. mcp: through the copied in-process MCP tool (tools/mcp_search.py, which owns
# the retries); direct: rooms_client's own httpx client. Anything else falls back to mcp.
SEARCH_TRANSPORT_ENV = "LLM_APPROACH_SEARCH_TRANSPORT"
SEARCH_TRANSPORTS = ("mcp", "direct")


def search_transport() -> str:
    value = os.getenv(SEARCH_TRANSPORT_ENV, "").strip().lower()
    return value if value in SEARCH_TRANSPORTS else "mcp"


# Semantic input gate (copied input guardrail's intent classifier). off: not run; shadow: run and record, never block;
# enforce: block off-topic first messages. configure_production() sets shadow; bare handle_turn and tests run off.
SEMANTIC_GATE_ENV = "LLM_APPROACH_SEMANTIC_GATE"
SEMANTIC_GATE_MODES = ("off", "shadow", "enforce")


def semantic_gate_setting() -> str | None:
    """The mode named by ``LLM_APPROACH_SEMANTIC_GATE``, or None when unset. An unknown value is an error
    (not echoed), so a typo never silently leaves the gate off in production."""
    value = os.getenv(SEMANTIC_GATE_ENV, "").strip().lower()
    if not value:
        return None
    if value not in SEMANTIC_GATE_MODES:
        raise ValueError(f"{SEMANTIC_GATE_ENV} must be one of: {', '.join(SEMANTIC_GATE_MODES)}")
    return value


# The API has no area filter, so an area search fetches this many rooms and
# filters them here. ASSUMPTION: the real API accepts this limit (unverified).
ROOMS_AREA_FETCH_LIMIT = _int_env("LLM_APPROACH_ROOMS_AREA_FETCH_LIMIT", 30)
OFFLINE_ROOMS_ENV = "LLM_APPROACH_OFFLINE_ROOMS"  # "1" -> fixtures/rooms_bangalore.json, or a path


def groq_model() -> str:
    """Model name, read lazily from the app's own constant."""
    return reuse.get_groq_model()


# ---- LLM provider ---------------------------------------------------------------
# Only llm_approach/providers.py acts on these; every other module sends the same messages and
# gets text plus token usage back. Keys come from the environment only and are never printed.

PROVIDER_ENV = "LLM_APPROACH_PROVIDER"
PROVIDERS = ("gemini", "groq")
DEFAULT_PROVIDER = "gemini"
GEMINI_MODEL_ENV = "LLM_APPROACH_GEMINI_MODEL"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"  # gemini-2.5-flash(-lite) return 404 NOT_FOUND for new projects
GEMINI_THINKING_ENV = "LLM_APPROACH_GEMINI_THINKING"
GEMINI_THINKING_LEVELS = ("LOW", "MINIMAL", "MEDIUM", "HIGH")
GEMINI_THINKING_ALLOWED = "level:LOW, level:MINIMAL, level:MEDIUM, level:HIGH, budget:<int> or none"
DEFAULT_GEMINI_THINKING_LEVEL = "LOW"  # gemini-3.x family
GEMINI_MAX_OUTPUT_ENV = "LLM_APPROACH_GEMINI_MAX_OUTPUT_TOKENS"
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = 1024  # thinking tokens count toward the cap, so 500 can truncate the JSON
KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}

# Failure handling (llm_resilience.py, llm_client.py). Fallback provider: off unless set, because the
# answer quality on the fallback has not been evaluated.
FALLBACK_PROVIDER_ENV = "LLM_APPROACH_FALLBACK_PROVIDER"
BREAKER_FAILURES_ENV = "LLM_APPROACH_BREAKER_FAILURES"
BREAKER_COOLDOWN_ENV = "LLM_APPROACH_BREAKER_COOLDOWN_SECONDS"
RETRY_5XX_DELAY_ENV = "LLM_APPROACH_5XX_RETRY_DELAY_SECONDS"
DEFAULT_RETRY_5XX_DELAY_S = 1.0
DEFAULT_BREAKER_FAILURES = 5
DEFAULT_BREAKER_COOLDOWN_S = 60

# Daily token budget and call spacing are per provider. Groq: the values the eval runners always had.
# Gemini: read from the environment on purpose, no limit numbers are hard-coded here.
GROQ_DEFAULT_SPACING_S = 30.0
GROQ_DEFAULT_DAILY_TOKEN_BUDGET = 150_000
GEMINI_SPACING_ENV = "LLM_APPROACH_GEMINI_SPACING_S"  # seconds between live calls
GEMINI_DEFAULT_SPACING_S = 5.0  # free tier is 15 requests per minute: 4 s would do, 5 s leaves room
GEMINI_MIN_SPACING_S = 4.0  # never below this (15 RPM)
# Our own safety cap: Google lists NO daily token limit for the free tier. Unset = live evals refuse.
GEMINI_DAILY_BUDGET_ENV = "LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET"
# Requests per day is the binding Gemini limit (500 on the free tier); the default leaves a safety margin.
GEMINI_REQUEST_BUDGET_ENV = "LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET"
GEMINI_DEFAULT_REQUEST_BUDGET = 450
GEMINI_DAY_RESET_ENV = "LLM_APPROACH_GEMINI_DAY_RESET"  # optional fixed daily boundary, a UTC time such as 07:00

# What result files written before the provider switch were produced with.
LEGACY_PROVIDER = "groq"
LEGACY_MODEL = "openai/gpt-oss-20b"


def provider() -> str:
    """The active provider: ``LLM_APPROACH_PROVIDER`` (gemini or groq), default gemini.
    Anything else raises ValueError (the bad value is not echoed)."""
    value = os.getenv(PROVIDER_ENV, "").strip().lower() or DEFAULT_PROVIDER
    if value not in PROVIDERS:
        raise ValueError(f"{PROVIDER_ENV} must be one of: {', '.join(PROVIDERS)}")
    return value


def fallback_provider() -> str | None:
    """``LLM_APPROACH_FALLBACK_PROVIDER`` (gemini or groq), or None: unset, the same as the primary, or not a
    known provider (that last case logs one warning that does not echo the value)."""
    value = os.getenv(FALLBACK_PROVIDER_ENV, "").strip().lower()
    if not value:
        return None
    if value not in PROVIDERS:
        logger.warning("%s is not one of: %s; fallback provider stays off", FALLBACK_PROVIDER_ENV, ", ".join(PROVIDERS))
        return None
    return None if value == provider() else value


def breaker_failures() -> int:
    """Consecutive primary failures that open the circuit breaker (default 5)."""
    return _int_env(BREAKER_FAILURES_ENV, DEFAULT_BREAKER_FAILURES)


def breaker_cooldown_s() -> int:
    """Seconds the open breaker skips the primary provider (default 60)."""
    return _int_env(BREAKER_COOLDOWN_ENV, DEFAULT_BREAKER_COOLDOWN_S)


def retry_5xx_delay_s() -> float | None:
    """Seconds to wait before the one retry of the primary provider after a 5xx (default 1.0; 0 retries at once;
    a negative value turns the retry off; an unreadable value uses the default)."""
    try:
        value = float(os.getenv(RETRY_5XX_DELAY_ENV, "").strip())
    except ValueError:
        return DEFAULT_RETRY_5XX_DELAY_S
    return value if value >= 0 else None


def model(name: str | None = None) -> str:
    """Model name of ``name`` (default: the active provider)."""
    name = name or provider()
    if name == "groq":
        return groq_model()
    return os.getenv(GEMINI_MODEL_ENV, "").strip() or DEFAULT_GEMINI_MODEL


def gemini_thinking(model_id: str | None = None) -> dict | None:
    """The ``thinking_config`` to send, or None for no thinking config at all.

    ``LLM_APPROACH_GEMINI_THINKING`` wins when set. Otherwise by model family: ``gemini-3*`` gets
    ``thinking_level`` LOW, ``gemini-2.5*`` gets ``thinking_budget`` 0, any other model gets nothing.
    An invalid override raises ValueError that names the allowed values (the bad value is not echoed)."""
    override = os.getenv(GEMINI_THINKING_ENV, "").strip()
    if override:
        low = override.lower()
        if low == "none":
            return None
        kind, _, arg = override.partition(":")
        if kind.lower() == "level" and arg.strip().upper() in GEMINI_THINKING_LEVELS:
            return {"thinking_level": arg.strip().upper()}
        if kind.lower() == "budget":
            try:
                return {"thinking_budget": int(arg.strip())}
            except ValueError:
                pass
        raise ValueError(f"{GEMINI_THINKING_ENV} must be one of: {GEMINI_THINKING_ALLOWED}")
    model_id = model_id if model_id is not None else model("gemini")
    if model_id.startswith("gemini-3"):
        return {"thinking_level": DEFAULT_GEMINI_THINKING_LEVEL}
    if model_id.startswith("gemini-2.5"):
        return {"thinking_budget": 0}
    return None


def gemini_thinking_label(model_id: str | None = None) -> str:
    """Human-readable form of ``gemini_thinking`` for the smoke test, e.g. ``level LOW``."""
    cfg = gemini_thinking(model_id)
    if not cfg:
        return "none (no thinking config sent)"
    (key, value), = cfg.items()
    return f"{key.removeprefix('thinking_')} {value}"


def gemini_max_output_tokens() -> int:
    return _int_env(GEMINI_MAX_OUTPUT_ENV, DEFAULT_GEMINI_MAX_OUTPUT_TOKENS)


def key_env(name: str | None = None) -> str:
    """Name of the environment variable that holds the key (never the key itself)."""
    return KEY_ENV[name or provider()]


def key_present(name: str | None = None) -> bool:
    return bool(os.environ.get(key_env(name), "").strip())


def label() -> tuple[str, str]:
    """(provider, model) for result files and metrics. Never raises: a bad provider setting is
    labelled ``invalid`` (the entry points reject it before any call)."""
    try:
        return provider(), model()
    except Exception:  # noqa: BLE001 - labelling must not break a metrics write
        return "invalid", ""


def record_label(record: dict) -> tuple[str, str]:
    """(provider, model) a result record was produced with; records written before the provider
    switch carry neither field and count as Groq / gpt-oss-20b."""
    return str(record.get("provider") or LEGACY_PROVIDER), str(record.get("model") or LEGACY_MODEL)


def setup_problem() -> str | None:
    """A one-line, secret-free message when the active provider cannot be used (bad provider
    setting or missing/blank key); None when it is ready."""
    try:
        name = provider()
    except ValueError as exc:
        return str(exc)
    if not key_present(name):
        return f"{key_env(name)} is not set. Put it in .env at the repo root."
    if name == "gemini":
        try:
            gemini_thinking()
            day_reset()
        except ValueError as exc:
            return str(exc)
    return None


def gemini_timeout_s() -> float:
    """Seconds for one Gemini call: ``LLM_APPROACH_GEMINI_TIMEOUT_S``, default 20, never below 10.
    A smaller (or unusable) env value is raised to 10 with one warning that names no value but the minimum."""
    raw = os.getenv(GEMINI_TIMEOUT_ENV, "").strip()
    if not raw:
        return GEMINI_DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return GEMINI_DEFAULT_TIMEOUT_S
    if value < GEMINI_MIN_TIMEOUT_S:
        logger.warning("%s is below the Gemini minimum; using %g s", GEMINI_TIMEOUT_ENV, GEMINI_MIN_TIMEOUT_S)
        return GEMINI_MIN_TIMEOUT_S
    return value


def _positive_float_env(name: str) -> float | None:
    try:
        value = float(os.getenv(name, "").strip())
    except ValueError:
        return None
    return value if value > 0 else None


_spacing_warned = False


def spacing_s(name: str | None = None) -> float:
    """Seconds between live calls for a provider. Gemini: ``LLM_APPROACH_GEMINI_SPACING_S``, default 5, never
    below 4 (a lower value is raised to 4 with one warning)."""
    global _spacing_warned
    name = name or provider()
    if name == "groq":
        return GROQ_DEFAULT_SPACING_S
    value = _positive_float_env(GEMINI_SPACING_ENV)
    if value is None:
        return GEMINI_DEFAULT_SPACING_S
    if value < GEMINI_MIN_SPACING_S:
        if not _spacing_warned:
            _spacing_warned = True
            logger.warning("%s is below the 15-requests-per-minute minimum; using %g s",
                           GEMINI_SPACING_ENV, GEMINI_MIN_SPACING_S)
        return GEMINI_MIN_SPACING_S
    return value


def daily_request_budget() -> int:
    """Gemini requests allowed per 24 hours: ``LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET``, default 450."""
    return _int_env(GEMINI_REQUEST_BUDGET_ENV, GEMINI_DEFAULT_REQUEST_BUDGET)


def day_reset() -> dt.time | None:
    """The fixed daily boundary (a UTC time, ``LLM_APPROACH_GEMINI_DAY_RESET=HH:MM``), or None (the default): the
    rolling 24-hour window is used. A value that is not HH:MM raises ValueError."""
    raw = os.getenv(GEMINI_DAY_RESET_ENV, "").strip()
    if not raw:
        return None
    try:
        return dt.datetime.strptime(raw, "%H:%M").time()
    except ValueError:
        raise ValueError(f"{GEMINI_DAY_RESET_ENV} must be a UTC time like 07:00 (HH:MM)") from None


def daily_token_budget(name: str | None = None) -> int | None:
    """Tokens per day for a provider; None when it is not configured (Gemini only)."""
    name = name or provider()
    if name == "groq":
        return GROQ_DEFAULT_DAILY_TOKEN_BUDGET
    value = _positive_float_env(GEMINI_DAILY_BUDGET_ENV)
    return int(value) if value else None
