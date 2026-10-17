"""NEW_SEARCH, REFINEMENT and SHOW_MORE. The only module that calls the rooms API."""

from __future__ import annotations

from typing import Any

from llm_approach import config, reuse, responder, rooms_client
from llm_approach.actions import ActionResult, TurnBudget
from llm_approach.actions.clarify import ASK_CITY
from llm_approach.schema import Decision
from llm_approach.state import LLMConversationState

ASK_AREA = "Which area would you like?"
ALL_SHOWN = (
    "These are all the matching rooms I found. You can widen the budget "
    "or the area and I'll search again."
)
MAX_AREA_EXAMPLES = 3
MAX_AREAS_LISTED = 5
ASK_CITY_FOR_AREA = "Which city is that area in?"
SEARCH_KEYS = ("city", "area", "bhk", "min_price", "max_price", "ideal_for", "amenity")


def area_examples(state: LLMConversationState) -> list[str]:
    """Up to 3 distinct locations of the rooms on screen, in order of appearance. Never invented."""
    seen: list[str] = []
    for room in state.rooms_on_screen:
        location = room.get("location")
        if isinstance(location, str) and location.strip() and location.strip().lower() not in {s.lower() for s in seen}:
            seen.append(location.strip())
        if len(seen) == MAX_AREA_EXAMPLES:
            break
    return seen


def area_question(state: LLMConversationState, repeated_city: str | None = None) -> str:
    """'Which area would you like?' with example areas from the rooms on screen.

    ``repeated_city``: the user only repeated the city already being searched."""
    text = ASK_AREA
    examples = area_examples(state)
    if examples:
        names = examples[0] if len(examples) == 1 else ", ".join(examples[:-1]) + " or " + examples[-1]
        text += f" For example {names}."
    if repeated_city:
        text = f"You're already searching in {repeated_city}. " + text
    return text


def _ask_city(state: LLMConversationState) -> ActionResult:
    state.set_pending("city")
    return ActionResult(ASK_CITY, None, "ask_city")


def _ask_city_for_area(state: LLMConversationState, area: str) -> ActionResult:
    """An area but no city anywhere: keep the area in the search and ask for the city."""
    state.current_search["area"] = area
    state.set_pending("city")
    return ActionResult(ASK_CITY_FOR_AREA, None, "ask_city")


def no_area_reply(search: dict[str, Any], areas: list[str]) -> str:
    """Honest 'no rooms in that area' message; areas are real locations of fetched rooms."""
    text = f"I couldn't find rooms in {responder.place_label(search)}."
    if areas:
        text += " Areas with rooms: " + ", ".join(areas[:MAX_AREAS_LISTED]) + "."
    return text


def _run_search(
    search: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
    action: str,
) -> ActionResult:
    """Call the rooms API once with validated values (amenity included) and update state."""
    result = rooms_client.search_rooms(search, budget)
    if not result.ok:
        state.apply_search_result(search, None)  # keeps previous state
        return ActionResult(
            rooms_client.UNAVAILABLE_MESSAGE, None, action, True, result.error_class
        )
    state.apply_search_result(search, result.rooms, result.api_count, result.truncated)
    rooms = state.rooms_on_screen
    if not rooms:
        if search.get("area") and result.area_matches == 0:
            return ActionResult(no_area_reply(search, result.areas), [], action, True)
        return ActionResult(responder.format_no_rooms_reply(search), [], action, True)
    return ActionResult(
        responder.format_search_reply(rooms, search), responder.room_cards(rooms), action, True
    )


def handle_new_search(
    decision: Decision,
    validated: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
) -> ActionResult:
    validated = dict(validated)
    stashed = state.current_search
    if not validated.get("city"):
        if validated.get("area") and stashed.get("city"):
            validated["city"] = stashed["city"]  # "rooms in koramangala": the city is the current one
        elif validated.get("area"):
            return _ask_city_for_area(state, validated["area"])
        else:
            return _ask_city(state)
    elif not validated.get("area") and stashed.get("area") and not stashed.get("city"):
        validated["area"] = stashed["area"]  # the city answers "Which city is that area in?"
    search = {key: validated.get(key) for key in SEARCH_KEYS}
    return _run_search(search, state, budget, "new_search")


def handle_refinement(
    decision: Decision,
    validated: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
) -> ActionResult:
    previous = {key: state.current_search.get(key) for key in SEARCH_KEYS}
    merged = dict(previous)
    changed = False
    for key in SEARCH_KEYS:
        value = validated.get(key)
        if value is not None and value != merged[key]:
            merged[key] = value
            changed = True
    new_city, old_city = validated.get("city"), previous.get("city")
    if new_city and old_city and new_city.lower() != str(old_city).lower() and not validated.get("area"):
        merged["area"] = None  # a different city: the old area does not apply
    if validated.get("area"):
        changed = True  # an area reply always runs the search
    if not merged.get("city"):
        if merged.get("area"):
            return _ask_city_for_area(state, merged["area"])
        return _ask_city(state)
    if not changed:
        state.set_pending("clarify", "refinement")
        city = str(merged["city"])
        repeated = city.title() if str(validated.get("city") or "").lower() == city.lower() else None
        return ActionResult(area_question(state, repeated), None, "ask_area")
    return _run_search(merged, state, budget, "refinement")


def handle_show_more(
    decision: Decision,
    validated: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
) -> ActionResult:
    """No pagination is known for the API, so no tool call. Honest answer only:
    fewer rooms than the page limit means these are all of them."""
    if not state.rooms_on_screen:
        if state.current_search.get("city"):
            return ActionResult(
                "I have no rooms to show right now. Try a higher budget or another area.",
                None,
                "show_more_none",
            )
        return _ask_city(state)
    # Count what the API returned, not what survived the client-side amenity
    # filter: a full page filtered down may still have more rooms behind it.
    # An area search fetches more rooms than it shows, so compare with that limit.
    shown = state.api_room_count if state.api_room_count is not None else len(state.rooms_on_screen)
    fetch_limit = config.ROOMS_AREA_FETCH_LIMIT if state.current_search.get("area") else config.ROOMS_LIMIT
    if shown < fetch_limit and not state.rooms_truncated:
        return ActionResult(ALL_SHOWN, None, "show_more_all")
    return ActionResult(reuse.get_show_alternatives_response(), None, "show_more_limit")
