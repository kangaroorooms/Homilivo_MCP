import contextvars
import os
import json
import urllib.parse
import urllib.request

from mcp.server import MCPServer


mcp = MCPServer("Homilivo Room Search MCP")

API_URL = "https://homilivo.onrender.com/api/rooms"


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

    try:

        with urllib.request.urlopen(request) as response:

            data = json.loads(
                response.read().decode()
            )

        rooms = data.get("data", [])

        return {
            "success": True,
            "count": len(rooms),
            "rooms": rooms
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    mcp.run()