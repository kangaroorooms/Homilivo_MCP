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


def place_label(search: dict[str, Any]) -> str:
    """'Koramangala, Bangalore', 'Bangalore' or ''."""
    area = str(search.get("area") or "").strip().title()
    return ", ".join(part for part in (area, _city_label(search)) if part)


def format_search_reply(
    rooms: list[dict[str, Any]],
    search: dict[str, Any],
    *,
    total: int | None = None,
    has_next: bool | None = None,
    page: int = 1,
    more: bool = False,
    note: str | None = None,
) -> str:
    """``more``: these rooms are a further page ("Here are more rooms (page N)").
    ``total`` is mentioned only when more rooms exist ("showing 10 of 24"); with an
    area it counts the whole city, so the caller passes ``note`` instead."""
    place = place_label(search)
    noun = "room" if len(rooms) == 1 else "rooms"
    if more:
        head = f"Here are more rooms (page {page})" + (f" in {place}" if place else "")
    else:
        head = f"I found {len(rooms)} {noun}" + (f" in {place}" if place else "")
    if has_next and total is not None and not search.get("area"):
        head += f" (showing {len(rooms)} of {total})"
    head += ":"
    lines = [room_line(i, r) for i, r in enumerate(rooms, 1)]
    tail = "Ask me about any of them, for example the cheapest one, or tell me what to change."
    if has_next:
        tail += " There are more rooms, say \"show more\" to see the next ones."
    if note:
        tail += " " + note
    return "\n".join([head, *lines, tail])


def format_no_rooms_reply(search: dict[str, Any]) -> str:
    place = place_label(search)
    where = f" in {place}" if place else ""
    return (
        f"I couldn't find any rooms{where} matching that. "
        "You could try a higher budget, a different room type or another area."
    )
