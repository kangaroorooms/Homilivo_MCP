"""Reply text and room cards in the format the current frontend renders.

Card shape = ``_structure_room`` in app/api.py (see README Phase 0 notes).
Every room fact comes from stored room data, never from the model.
"""

from __future__ import annotations

from typing import Any

from llm_approach.rooms_client import UNAVAILABLE_MESSAGE  # noqa: F401  (re-export)

_AMENITY_KEYS = ("amenities", "amenity", "facilities", "facility", "features")


def price_of(room: dict[str, Any]) -> float | None:
    for key in ("pricePerMonth", "price"):
        value = room.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def price_text(value: float | None) -> str:
    return "price not listed" if value is None else f"₹{value:,.0f}/month"


def room_card(room: dict[str, Any]) -> dict[str, Any]:
    amenities: Any = None
    for key in _AMENITY_KEYS:
        if room.get(key) is not None:
            amenities = room[key]
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
        "price_per_month": price_of(room),
        "room_type": room.get("roomType"),
        "ideal_for": room.get("idealFor", []),
        "amenities": amenities,
    }


def room_cards(rooms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [room_card(r) for r in rooms]


def room_line(index: int, room: dict[str, Any]) -> str:
    parts = [str(room.get("title") or "Untitled Room")]
    if room.get("roomType"):
        parts.append(str(room["roomType"]))
    if room.get("location"):
        parts.append(str(room["location"]))
    parts.append(price_text(price_of(room)))
    return f"{index}. " + ", ".join(parts)


def _city_label(search: dict[str, Any]) -> str:
    return str(search.get("city") or "").strip().title()


def format_search_reply(rooms: list[dict[str, Any]], search: dict[str, Any]) -> str:
    city = _city_label(search)
    noun = "room" if len(rooms) == 1 else "rooms"
    head = f"I found {len(rooms)} {noun}" + (f" in {city}" if city else "") + ":"
    lines = [room_line(i, r) for i, r in enumerate(rooms, 1)]
    tail = "Ask me about any of them, for example the cheapest one, or tell me what to change."
    return "\n".join([head, *lines, tail])


def format_no_rooms_reply(search: dict[str, Any]) -> str:
    city = _city_label(search)
    where = f" in {city}" if city else ""
    return (
        f"I couldn't find any rooms{where} matching that. "
        "You could try a higher budget, a different room type or another area."
    )
