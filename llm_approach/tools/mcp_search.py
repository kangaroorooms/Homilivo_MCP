# Copied from app/server.py at commit ed0406a (the last commit that changed it; the other copies cite 7dbcba3) (copied, not imported).
# Changes (each marked '# v2:'): the observability import points at llm_approach.observability; the API URL comes from
# llm_approach.config (so LLM_APPROACH_ROOMS_API_URL still works); the tool takes a `page` argument and returns the API's `meta`
# (total, hasNextPage), which SHOW_MORE needs. Retries, timeouts, status handling and token handling are unchanged.
# Drift is checked by llm_approach/tests/test_copy_drift.py.
# ==== end of v2 provenance header ====
import contextvars
import os
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

from mcp.server import MCPServer

from llm_approach import config  # v2: URL comes from config
from llm_approach.observability import record_api_latency  # v2: was: from app.observability import record_api_latency

logger = logging.getLogger(__name__)

mcp = MCPServer("Homilivo Room Search MCP")

API_URL = config.ROOMS_API_URL  # v2: was: API_URL = "https://homilivo.onrender.com/api/rooms"

# ============================================================
# OUTBOUND API RELIABILITY
# ============================================================
# urllib.request has no separate connect/read timeout knobs - `timeout=`
# bounds the whole call (socket connect + read). That is the best this
# stdlib client can offer without switching HTTP libraries, and is a
# real improvement over the previous unbounded call, which could hang
# the request indefinitely on a stalled connection.
HOMILIVO_API_TIMEOUT_SECONDS = float(
    os.getenv("HOMILIVO_API_TIMEOUT_SECONDS", "10")
)

# Upper bound (P2-11) on how long a caller may await the search_rooms
# MCP tool end to end, retries included. Applied with asyncio.wait_for()
# at the client.call_tool() site in app/experiments/agent_modes.py;
# defined here beside the other search_rooms limits.
MCP_TOOL_TIMEOUT_SECONDS = 15

# search_rooms is a read-only GET, so limited retries are safe here -
# this must never be applied to a write/non-idempotent operation.
HOMILIVO_API_MAX_ATTEMPTS = max(
    1, int(os.getenv("HOMILIVO_API_MAX_ATTEMPTS", "3"))
)

# Exponential backoff base between retries (0.3s, 0.6s, 1.2s, ...).
HOMILIVO_API_RETRY_BACKOFF_SECONDS = float(
    os.getenv("HOMILIVO_API_RETRY_BACKOFF_SECONDS", "0.3")
)


def _failure(error: str) -> dict:
    """
    Single safe failure shape for every Homilivo API failure. Never
    contains rooms - results are never fabricated after an API failure.
    """
    return {
        "success": False,
        "count": 0,
        "rooms": [],
        "error": error,
    }


_HTTP_ERROR_MESSAGES = {
    400: "Homilivo API bad request (400)",
    401: "Homilivo API unauthorized (401) - check KANGAROO_TOKEN",
    403: "Homilivo API forbidden (403)",
    404: "Homilivo API endpoint not found (404)",
    429: "Homilivo API rate limited (429)",
}


def _log_http_error(code: int) -> None:
    message = _HTTP_ERROR_MESSAGES.get(code)

    if message is None:
        message = (
            f"Homilivo API server error ({code})"
            if code in (500, 502, 503)
            else f"Homilivo API error ({code})"
        )

    logger.error(message)


def _record_api_latency(started_at: float, success: bool) -> None:
    record_api_latency((time.monotonic() - started_at) * 1000, success)


def _is_retryable_error(error: Exception) -> bool:
    """
    Retry only on transient network-level failures and 5xx server
    errors. Never retry a 4xx (bad request, auth failure, etc.) since
    the same request would just fail again identically.
    """
    if isinstance(error, urllib.error.HTTPError):
        return error.code >= 500

    if isinstance(error, (TimeoutError, urllib.error.URLError)):
        return True

    return False


# ============================================================
# PER-REQUEST TOKEN CONTEXT
# ============================================================
# The Homilivo token must never be a process-global (e.g. os.environ),
# because FastAPI serves concurrent requests as concurrent asyncio tasks
# sharing one process: a global write from request A can be overwritten
# by request B before A's outbound HTTP call fires, leaking B's token
# into A's search. A ContextVar is bound per-task (and inherited by any
# child tasks spawned from it, e.g. MCP's internal task group), so each
# request keeps its own isolated value regardless of concurrency.
_kangaroo_token_var: "contextvars.ContextVar[str | None]" = contextvars.ContextVar(
    "kangaroo_token",
    default=None,
)


def set_kangaroo_token(token: str) -> contextvars.Token:
    """Bind the Homilivo API token to the current request's task only."""
    return _kangaroo_token_var.set(token)


def _get_kangaroo_token() -> str | None:
    """
    Per-request token if one was bound (the FastAPI path), otherwise
    fall back to the process environment variable (CLI / manual usage,
    e.g. `python -m app.agent`, where there is only one caller and no
    concurrency to protect against).
    """
    token = _kangaroo_token_var.get()
    if token:
        return token
    return os.getenv("KANGAROO_TOKEN")


@mcp.tool()
def search_rooms(
    city: str | None = None,
    min_price: int | None = None,
    max_price: int | None = None,
    room_types: list[str] | None = None,
    gender_preference: str | None = None,
    ideal_for: list[str] | None = None,
    sort: str = "latest",
    limit: int = 10,  # v2: was: limit: int = 10
    page: int = 1,  # v2: SHOW_MORE asks for page 2 and up
) -> dict:
    """
    Search rooms using the Homilivo rooms API.

    Parameters:
    city: City to search in.
    min_price: Minimum monthly rent.
    max_price: Maximum monthly rent.
    room_types: Room types such as PG, 1BHK, 2BHK.
    gender_preference: Gender preference.
    ideal_for: Suitable users such as Students or Working Professionals.
    sort: Sorting method.
    limit: Maximum number of results.
    """

    token = _get_kangaroo_token()

    if not token:
        return _failure("KANGAROO_TOKEN environment variable is not set.")

    params = {
        "city": city,
        "minPrice": min_price,
        "maxPrice": max_price,
        "roomTypes": ",".join(room_types) if room_types else None,
        "genderPreference": gender_preference,
        "idealFor": ",".join(ideal_for) if ideal_for else None,
        "sort": sort,
        "limit": limit,  # v2: was: "limit": limit
        "page": page if page > 1 else None,  # v2: sent only for page > 1, like rooms_client.build_params
    }

    # Remove parameters that were not provided
    params = {
        key: value
        for key, value in params.items()
        if value is not None
    }

    query_string = urllib.parse.urlencode(params)

    url = f"{API_URL}?{query_string}"

    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
    )

    last_error: Exception | None = None
    _started_at = time.monotonic()

    for attempt in range(1, HOMILIVO_API_MAX_ATTEMPTS + 1):

        try:

            with urllib.request.urlopen(
                request,
                timeout=HOMILIVO_API_TIMEOUT_SECONDS,
            ) as response:

                raw_response = response.read().decode()

            # Malformed / empty bodies are not transient: retrying the
            # same request would not change the body, so fail at once.
            if not raw_response.strip():
                logger.warning("Homilivo API returned empty body")
                _record_api_latency(_started_at, success=False)
                return _failure("Homilivo API returned an empty response.")

            try:
                data = json.loads(raw_response)
            except json.JSONDecodeError:
                logger.warning(
                    "Homilivo API returned malformed JSON: %s",
                    raw_response[:200],
                )
                _record_api_latency(_started_at, success=False)
                return _failure("Homilivo API returned malformed JSON.")

            rooms = data.get("data", []) if isinstance(data, dict) else None

            if not isinstance(rooms, list):
                logger.warning(
                    "Homilivo API returned unexpected payload shape: %s",
                    raw_response[:200],
                )
                _record_api_latency(_started_at, success=False)
                return _failure("Homilivo API returned an unexpected response.")

            _record_api_latency(_started_at, success=True)

            return {
                "success": True,
                "count": len(rooms),
                "rooms": rooms,  # v2: was: "rooms": rooms
                "meta": data.get("meta"),  # v2: total and hasNextPage, read by rooms_client.parse_meta
            }

        except Exception as e:

            last_error = e

            if isinstance(e, urllib.error.HTTPError):
                _log_http_error(e.code)

            is_last_attempt = attempt == HOMILIVO_API_MAX_ATTEMPTS

            if is_last_attempt or not _is_retryable_error(e):
                break

            time.sleep(
                HOMILIVO_API_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            )

    _record_api_latency(_started_at, success=False)

    # Controlled failure: never let an exception (or a hang) propagate
    # out of the tool. The caller (app/experiments/agent_modes.py) turns
    # success=False into a user-facing "temporarily unavailable" message
    # rather than a stack trace.
    return _failure(
        "Homilivo API request timed out."
        if isinstance(last_error, TimeoutError)
        else str(last_error)
    )


if __name__ == "__main__":
    mcp.run()