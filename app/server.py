import os
import json
import urllib.parse
import urllib.request

from mcp.server import MCPServer


mcp = MCPServer("Homilivo Room Search MCP")

API_URL = "https://homilivo.onrender.com/api/rooms"


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

    token = os.getenv("KANGAROO_TOKEN")

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