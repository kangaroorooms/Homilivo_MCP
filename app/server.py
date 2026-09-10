import contextvars
import os
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from mcp.server import MCPServer

from app.observability import record_api_latency

mcp = MCPServer("Homilivo Room Search MCP")

API_URL = "https://homilivo.onrender.com/api/rooms"

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

# search_rooms is a read-only GET, so limited retries are safe here -
# this must never be applied to a write/non-idempotent operation.
HOMILIVO_API_MAX_ATTEMPTS = max(
    1, int(os.getenv("HOMILIVO_API_MAX_ATTEMPTS", "3"))
)

# Exponential backoff base between retries (0.3s, 0.6s, 1.2s, ...).
HOMILIVO_API_RETRY_BACKOFF_SECONDS = float(
    os.getenv("HOMILIVO_API_RETRY_BACKOFF_SECONDS", "0.3")
)


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
    limit: int = 10
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
        return {
            "success": False,
            "error": "KANGAROO_TOKEN environment variable is not set."
        }

    params = {
        "city": city,
        "minPrice": min_price,
        "maxPrice": max_price,
        "roomTypes": ",".join(room_types) if room_types else None,
        "genderPreference": gender_preference,
        "idealFor": ",".join(ideal_for) if ideal_for else None,
        "sort": sort,
        "limit": limit
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

                data = json.loads(
                    response.read().decode()
                )

            rooms = data.get("data", [])

            _record_api_latency(_started_at, success=True)

            return {
                "success": True,
                "count": len(rooms),
                "rooms": rooms
            }

        except Exception as e:

            last_error = e

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
    return {
        "success": False,
        "error": (
            "Homilivo API request timed out."
            if isinstance(last_error, TimeoutError)
            else str(last_error)
        ),
    }


if __name__ == "__main__":
    mcp.run()