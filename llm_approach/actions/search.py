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
SEARCH_KEYS = ("city", "bhk", "min_price", "max_price", "ideal_for", "amenity")


def _ask_city(state: LLMConversationState) -> ActionResult:
    state.set_pending("city")
    return ActionResult(ASK_CITY, None, "ask_city")


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
    state.apply_search_result(search, result.rooms, result.api_count)
    rooms = state.rooms_on_screen
    if not rooms:
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
    if not validated.get("city"):
        return _ask_city(state)
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
    if not merged.get("city"):
        return _ask_city(state)
    if not changed:
        state.set_pending("clarify", "refinement")
        return ActionResult(ASK_AREA, None, "ask_area")
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
    shown = state.api_room_count if state.api_room_count is not None else len(state.rooms_on_screen)
    if shown < config.ROOMS_LIMIT:
        return ActionResult(ALL_SHOWN, None, "show_more_all")
    return ActionResult(reuse.get_show_alternatives_response(), None, "show_more_limit")
