# Copied from app/api.py (validate_secrets at line 102, _normalize_session_id at line 68) at commit 7dbcba3 (copied, not imported).
# Changes: validate_secrets requires the active provider's key (GEMINI_API_KEY or GROQ_API_KEY) plus KANGAROO_TOKEN;
# _SESSION_ID_RE and _normalize_session_id are verbatim; imports and logger added.
# Not called from server_app.py yet (Phase 2).
# Drift is checked by llm_approach/tests/test_copy_drift.py.
# ==== end of v2 provenance header ====
"""Fail-fast secret check and session-id normaliser, copied from app/api.py."""

from __future__ import annotations  # v2: file header

import logging  # v2: needed by the copied functions
import os  # v2: needed by validate_secrets
import re  # v2: needed by _SESSION_ID_RE
from uuid import uuid4  # v2: needed by _normalize_session_id

from llm_approach import config  # v2: provider() decides which API key is required

logger = logging.getLogger(__name__)  # v2: module logger (api.py defines its own)

# P3-18: only UUID-format session ids may reach logs or the session
# store. A client-supplied id is otherwise free text (could be an email
# or name), and record_turn() logs session_id.
_SESSION_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
    r"-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _normalize_session_id(candidate: object) -> str:
    """
    Return `candidate` if it is a UUID-format string, else a fresh
    server-side UUID. The invalid value itself is never logged.
    """
    if candidate is None or candidate == "":
        return str(uuid4())

    if isinstance(candidate, str) and _SESSION_ID_RE.match(candidate):
        # Case-fold so ABC... and abc... resolve to the same session.
        return candidate.lower()

    logger.warning(
        "Invalid session_id format received - generating new session_id"
    )
    return str(uuid4())


def validate_secrets() -> None:
    """
    Fail fast at startup (P3-13) if a required secret is missing, rather
    than crashing mid-request. Only secret NAMES are ever logged or put
    in an error message - never a value.
    """
    key_name = config.KEY_ENV[config.provider()]  # v2: GEMINI_API_KEY or GROQ_API_KEY, by LLM_APPROACH_PROVIDER
    required = {"KANGAROO_TOKEN": os.getenv("KANGAROO_TOKEN"), key_name: os.getenv(key_name)}  # v2: was: required = {"KANGAROO_TOKEN": os.getenv("KANGAROO_TOKEN")}
    optional: dict[str, str | None] = {}  # v2: was: optional = {"GROQ_API_KEY": os.getenv("GROQ_API_KEY")}

    for name, value in required.items():
        if not value:
            logger.error("Required secret %s is not set.", name)
            raise RuntimeError(
                f"Required secret {name} is not set. "
                f"Cannot start the application."
            )

    for name, value in optional.items():
        if not value:
            logger.warning(
                "Optional secret %s is not set — "
                "dependent features will be unavailable.",
                name,
            )
