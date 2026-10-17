"""Settings for the LLM decision pipeline. Safe to import without a key."""

from __future__ import annotations

import os

from llm_approach import reuse

CONFIDENCE_THRESHOLD = 0.6
LLM_TIMEOUT_S = 8
LLM_MAX_TOKENS = 500
TEMPERATURE = 0
REASONING_EFFORT: str | None = "low"  # gpt-oss: keeps reasoning from eating max_tokens
MAX_LLM_CALLS_PER_TURN = 2
MAX_TOOL_CALLS_PER_TURN = 1
HISTORY_TURNS_IN_PROMPT = 6  # one turn = user message + bot reply
MAX_ROOMS_IN_PROMPT = 10
MAX_USER_MESSAGE_CHARS = 500
LAST_BOT_MESSAGE_CHARS = 300
HISTORY_MESSAGE_CHARS = 200

# Mirrors app/session_store.py and app/experiments/agent_modes.py: same env
# vars, same defaults (not imported, because those modules are heavy).
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
SESSION_MAX_COUNT = int(os.getenv("SESSION_MAX_COUNT", "1000"))
SESSION_MAX_HISTORY_MESSAGES = int(os.getenv("SESSION_MAX_HISTORY_MESSAGES", "40"))
MAX_LLM_CONTEXT_CHARS = 8000


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Rooms API (mirrors API_URL in app/server.py, which is heavy to import).
ROOMS_API_URL = os.getenv("LLM_APPROACH_ROOMS_API_URL", "https://homilivo.onrender.com/api/rooms")
ROOMS_API_TIMEOUT_S = 15.0
ROOMS_RETRY_JITTER_S = (2.0, 4.0)  # one retry, timeout or connection error only
ROOMS_LIMIT = 10
ROOMS_SORT = "latest"
OFFLINE_ROOMS_ENV = "LLM_APPROACH_OFFLINE_ROOMS"  # "1" -> fixtures/rooms_bangalore.json, or a path

DEBUG = _flag("LLM_APPROACH_DEBUG")  # may print decision JSON in the CLI
LIVE = _flag("LLM_APPROACH_LIVE")  # enables live tests


def groq_model() -> str:
    """Model name, read lazily from the app's own constant."""
    return reuse.get_groq_model()
