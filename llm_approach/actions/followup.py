"""ROOM_FOLLOWUP: answers computed in code from ``state.rooms_on_screen``. No LLM.

Targeting rules:
- Single-room operations (price, type, location, details) use the explicit
  reference, else the selected room, else ask which room.
- AMENITY and the suitability operations use a room only when a reference was
  given (a number or "selected"); otherwise they answer across all rooms.
- LOWEST/HIGHEST_PRICE select the winner when it is unique; on a tie nothing
  is selected.
"""

from __future__ import annotations

from typing import Any

from llm_approach import reuse
from llm_approach.actions import ActionResult, TurnBudget
from llm_approach.responder import price_of, price_text, room_line
from llm_approach.rooms_client import has_amenity, room_amenities
from llm_approach.schema import Decision, Operation
from llm_approach.state import LLMConversationState

SEARCH_FIRST = "I don't have any rooms to talk about yet. Tell me the city you're looking for and I'll search first."
NOT_LISTED = "not listed"

_SUITABILITY = {
    Operation.STUDENT_SUITABILITY: ("Students", "students"),
    Operation.FAMILY_SUITABILITY: ("Family", "families"),
    Operation.PROFESSIONAL_SUITABILITY: ("Working Professionals", "working professionals"),
}


def _result(reply: str, action: str) -> ActionResult:
    return ActionResult(reply, None, action)


def _name(index: int, room: dict[str, Any]) -> str:
    return f"Room {index} ({room.get('title') or 'Untitled Room'})"


def _numbers(indexes: list[int]) -> str:
    parts = [str(i) for i in indexes]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _ask_which(state: LLMConversationState) -> ActionResult:
    return _result(f"Which room do you mean? Please pick a number from 1 to {len(state.rooms_on_screen)}.", "ask_which_room")


def _explicit_target(validated, state) -> tuple[int, dict] | None:
    ref = validated.get("room_reference")
    if ref == "selected":
        ref = state.selected_room_index
    if isinstance(ref, int) and 1 <= ref <= len(state.rooms_on_screen):
        return ref, state.rooms_on_screen[ref - 1]
    return None


def _target(validated, state) -> tuple[int, dict] | None:
    """Explicit reference, else the selected room."""
    found = _explicit_target(validated, state)
    if found is None and state.selected_room_index is not None:
        found = state.selected_room_index, state.selected_room()
    return found


def _priced(state) -> list[tuple[int, dict, float]]:
    return [
        (i, r, p)
        for i, r in enumerate(state.rooms_on_screen, 1)
        if (p := price_of(r)) is not None
    ]


def _extreme(state, lowest: bool) -> ActionResult:
    priced = _priced(state)
    if not priced:
        return _result("Prices are not listed for these rooms.", "followup_price")
    best = (min if lowest else max)(p for _, _, p in priced)
    winners = [(i, r) for i, r, p in priced if p == best]
    word = "cheapest" if lowest else "most expensive"
    if len(winners) == 1:
        index, room = winners[0]
        state.select_room(index)
        return _result(f"The {word} is {_name(index, room)} at {price_text(best)}.", "followup_price")
    names = _numbers([i for i, _ in winners])
    return _result(
        f"Rooms {names} are tied as the {word}, each at {price_text(best)}. Which one would you like to know more about?",
        "followup_price",
    )


def _amenity(validated, decision, state) -> ActionResult:
    amenity = validated.get("amenity")
    if amenity is None:
        if decision.slots.amenity:  # asked for something outside the vocabulary
            return _result(reuse.get_not_verified_amenity_response(), "followup_amenity")
        return _result("Which amenity are you asking about?", "ask_amenity")
    target = _explicit_target(validated, state)
    if target:
        index, room = target
        if room_amenities(room) is None:
            return _result(f"Amenities are {NOT_LISTED} for {_name(index, room)}.", "followup_amenity")
        state.select_room(index)
        has = has_amenity(room, amenity)
        return _result(
            f"{_name(index, room)} {'has' if has else 'does not list'} {amenity}.", "followup_amenity"
        )
    rooms = state.rooms_on_screen
    with_it = [i for i, r in enumerate(rooms, 1) if has_amenity(r, amenity)]
    if with_it:
        verb = "has" if len(with_it) == 1 else "have"
        noun = "Room" if len(with_it) == 1 else "Rooms"
        return _result(f"{noun} {_numbers(with_it)} {verb} {amenity}.", "followup_amenity")
    if all(room_amenities(r) is None for r in rooms):
        return _result(f"Amenities are {NOT_LISTED} for these rooms.", "followup_amenity")
    return _result(f"None of these rooms list {amenity}.", "followup_amenity")


def _suitability(op: Operation, validated, state) -> ActionResult:
    value, plural = _SUITABILITY[op]

    def listed(room) -> list[str] | None:
        ideal = room.get("idealFor")
        return [str(i).lower() for i in ideal] if isinstance(ideal, list) else None

    target = _explicit_target(validated, state)
    if target:
        index, room = target
        ideal = listed(room)
        if ideal is None:
            return _result(f"Who {_name(index, room)} is ideal for is {NOT_LISTED}.", "followup_suitability")
        state.select_room(index)
        fits = value.lower() in ideal
        return _result(
            f"{_name(index, room)} is {'listed as suitable' if fits else 'not listed as suitable'} for {plural}.",
            "followup_suitability",
        )
    rooms = state.rooms_on_screen
    fits = [i for i, r in enumerate(rooms, 1) if value.lower() in (listed(r) or [])]
    if fits:
        noun = "Room" if len(fits) == 1 else "Rooms"
        return _result(f"{noun} {_numbers(fits)} {'is' if len(fits) == 1 else 'are'} listed as suitable for {plural}.", "followup_suitability")
    if all(listed(r) is None for r in rooms):
        return _result(f"Who these rooms are ideal for is {NOT_LISTED}.", "followup_suitability")
    return _result(f"None of these rooms are listed as suitable for {plural}.", "followup_suitability")


def _details(index: int, room: dict[str, Any]) -> str:
    place = ", ".join(str(x) for x in (room.get("location"), room.get("landmark")) if x)
    amenities = room_amenities(room)
    ideal = room.get("idealFor")
    lines = [
        f"{_name(index, room)}:",
        f"- Type: {room.get('roomType') or NOT_LISTED}",
        f"- Price: {price_text(price_of(room))}",
        f"- Location: {place or NOT_LISTED}",
        f"- Amenities: {', '.join(amenities) if amenities else NOT_LISTED}",
        f"- Ideal for: {', '.join(map(str, ideal)) if isinstance(ideal, list) and ideal else NOT_LISTED}",
    ]
    if isinstance(room.get("rating"), (int, float)) and room["rating"] > 0:
        lines.append(f"- Rating: {room['rating']}")
    return "\n".join(lines)


def handle_followup(
    decision: Decision,
    validated: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
) -> ActionResult:
    if not state.rooms_on_screen:
        return _result(SEARCH_FIRST, "followup_no_rooms")
    op = decision.operation
    if op is None:
        return _result("What would you like to know about these rooms?", "followup_ask")

    if op is Operation.LOWEST_PRICE:
        return _extreme(state, lowest=True)
    if op is Operation.HIGHEST_PRICE:
        return _extreme(state, lowest=False)
    if op is Operation.AMENITY:
        return _amenity(validated, decision, state)
    if op in _SUITABILITY:
        return _suitability(op, validated, state)

    if op is Operation.COMPARE:
        refs = list(dict.fromkeys(validated.get("compare_references") or []))
        if len(refs) < 2:
            return _result("Which two rooms would you like to compare? For example, 1 and 2.", "ask_compare")
        a, b = refs[:2]
        rooms = state.rooms_on_screen
        return _result(
            "Comparison:\n" + room_line(a, rooms[a - 1]) + "\n" + room_line(b, rooms[b - 1]),
            "followup_compare",
        )

    target = _target(validated, state)
    if target is None:
        return _ask_which(state)
    index, room = target
    state.select_room(index)
    name = _name(index, room)
    if op is Operation.ROOM_PRICE:
        price = price_of(room)
        text = f"{name} costs {price_text(price)}." if price is not None else f"The price of {name} is {NOT_LISTED}."
    elif op is Operation.ROOM_TYPE:
        rtype = room.get("roomType") or room.get("type")
        text = f"{name} is a {rtype}." if rtype else f"The room type of {name} is {NOT_LISTED}."
    elif op is Operation.LOCATION:
        place = ", ".join(str(x) for x in (room.get("location"), room.get("landmark")) if x)
        text = f"{name} is in {place}." if place else f"The location of {name} is {NOT_LISTED}."
    else:  # ROOM_DETAILS
        text = _details(index, room)
    return _result(text, "followup_" + op.value.lower())
