"""NEW_SEARCH, REFINEMENT and SHOW_MORE. The only module that calls the rooms API."""

from __future__ import annotations

import logging
from typing import Any

from llm_approach import config, observability, responder, reuse, rooms_client, switches, turn_context
from llm_approach.actions import ActionResult, TurnBudget
from llm_approach.actions.clarify import ASK_CITY, state_aware_question
from llm_approach.guardrails.parameter_guardrail import validate_search_arguments
from llm_approach.schema import Decision
from llm_approach.state import LLMConversationState

logger = logging.getLogger(__name__)

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


def area_partial_reply(search: dict[str, Any], searched: int | None) -> str:
    """The area was not found in the rooms fetched, but the city has more rooms."""
    count = f"the first {searched} rooms" if searched else "only the first page of rooms"
    return (
        f"I searched {count} of {_city(search)} and none matched {search.get('area')}, "
        "but there are more rooms I did not check. You can say \"show more\" to keep "
        "looking, or try a budget or a room type."
    )


def _city(search: dict[str, Any]) -> str:
    return str(search.get("city") or "the city").strip().title()


# The guardrail's reason can echo a value, so only the argument name is logged. Prefix -> argument.
_ARG_REASON_FIELDS = (
    ("City", "city"),
    ("Minimum price", "min_price"),
    ("Maximum price", "max_price"),
    ("Invalid room type", "room_types"),
    ("Each room type", "room_types"),
    ("Room types", "room_types"),
    ("Duplicate room types", "room_types"),
    ("Invalid ideal-for", "ideal_for"),
    ("Each ideal-for", "ideal_for"),
    ("Ideal-for", "ideal_for"),
    ("Duplicate ideal-for", "ideal_for"),
    ("Unexpected search parameter", "parameter"),
    ("Sort", "sort"),
    ("Invalid sort", "sort"),
    ("Result limit", "limit"),
)


def guardrail_arguments(search: dict[str, Any]) -> dict[str, Any]:
    """The search as the parameter guardrail reads it (the same values the rooms API gets)."""
    room_types = rooms_client._room_types_param(search)
    args: dict[str, Any] = {
        "city": search.get("city"),
        "min_price": search.get("min_price"),
        "max_price": search.get("max_price"),
        "room_types": room_types.split(",") if room_types else None,
        "ideal_for": [search["ideal_for"]] if search.get("ideal_for") else None,
        "sort": config.ROOMS_SORT,
    }
    return {k: v for k, v in args.items() if v is not None}


def _check_arguments(search: dict[str, Any], state: LLMConversationState, action: str) -> ActionResult | None:
    """Second line of defence after ``validators.validate_slots``: ask instead of searching."""
    if not switches.search_arg_check():
        return None
    valid, reason = validate_search_arguments(guardrail_arguments(search))
    if valid:
        return None
    field = next((name for prefix, name in _ARG_REASON_FIELDS if reason.startswith(prefix)), "unknown")
    observability.record_parameter_rejection(field)
    logger.warning(
        "search arguments rejected field=%s request_id=%s session=%s",
        field, turn_context.request_id(), turn_context.session_hash(),
    )
    state.set_pending("clarify")
    if field in ("min_price", "max_price"):
        reply = "I didn't catch the budget. Could you say it again?"
    elif field == "room_types":
        reply = "I didn't catch the number of BHK. Could you say it again?"
    else:
        reply = state_aware_question(state)
    return ActionResult(reply, None, "ask_slot")


def _run_search(
    search: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
    action: str,
    page: int = 1,
) -> ActionResult:
    """Call the rooms API once with validated values (amenity included) and update state.
    ``page`` > 1 is SHOW_MORE: the page's rooms replace the ones on screen."""
    more = page > 1
    rejected = _check_arguments(search, state, action)
    if rejected is not None:
        return rejected
    result = rooms_client.search_rooms(search, budget, page=page)
    if not result.ok:
        state.apply_search_result(search, None)  # keeps previous state
        return ActionResult(
            rooms_client.UNAVAILABLE_MESSAGE, None, action, True, result.error_class
        )
    area = search.get("area")
    partial = bool(area) and result.has_next is True
    if more and not result.rooms:
        # Nothing on this page matched: keep the rooms on screen, remember the paging.
        state.set_paging(page, result.has_next, result.total)
        reply = (
            area_partial_reply(search, result.api_count)
            if partial
            else "There are no more matching rooms. " + ALL_SHOWN.split(". ", 1)[1]
        )
        return ActionResult(reply, None, action, True)
    state.apply_search_result(
        search, result.rooms, result.api_count, result.truncated,
        page=page, has_next=result.has_next, total=result.total,
    )
    rooms = state.rooms_on_screen
    if not rooms:
        if partial:
            return ActionResult(area_partial_reply(search, result.api_count), [], action, True)
        if area and result.area_matches == 0:
            return ActionResult(no_area_reply(search, result.areas), [], action, True)
        return ActionResult(responder.format_no_rooms_reply(search), [], action, True)
    note = None
    if partial:
        note = (
            f"I only searched the first {result.api_count} rooms of {_city(search)}; "
            "try a budget or room type to narrow it down."
        )
    reply = responder.format_search_reply(
        rooms, search, total=result.total, has_next=result.has_next,
        page=page, more=more, note=note,
    )
    return ActionResult(reply, responder.room_cards(rooms), action, True)


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
    """When the last result said there is a next page, fetch it (one tool call) and
    show it as the new rooms on screen. Otherwise no call and an honest answer."""
    city = state.current_search.get("city")
    if state.has_next is True and city:
        search = {key: state.current_search.get(key) for key in SEARCH_KEYS}
        return _run_search(search, state, budget, "show_more", page=state.search_page + 1)
    if not state.rooms_on_screen:
        if city:
            return ActionResult(
                "I have no rooms to show right now. Try a higher budget or another area.",
                None,
                "show_more_none",
            )
        return _ask_city(state)
    if state.has_next is False:
        # Complete unless matches were cut for display (area searches show at most 10).
        if state.rooms_truncated:
            return ActionResult(reuse.get_show_alternatives_response(), None, "show_more_limit")
        return ActionResult(ALL_SHOWN, None, "show_more_all")
    # No paging info (meta missing): count what the API returned, not what survived
    # the client-side filters. An area search fetches more than it shows.
    shown = state.api_room_count if state.api_room_count is not None else len(state.rooms_on_screen)
    fetch_limit = config.ROOMS_AREA_FETCH_LIMIT if state.current_search.get("area") else config.ROOMS_LIMIT
    if shown < fetch_limit and not state.rooms_truncated:
        return ActionResult(ALL_SHOWN, None, "show_more_all")
    return ActionResult(reuse.get_show_alternatives_response(), None, "show_more_limit")
