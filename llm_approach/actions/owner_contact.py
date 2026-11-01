"""Owner-contact requests ("how do I contact the owner"): show the Homilivo subscription plans of the city.

Runs only when the model's validated intent is UNSUPPORTED_ACTION or PLATFORM_INFO and the user's own message (or the model's
``standalone_query``) names a contact request. Nothing runs before the LLM call. The plans come from the pricing API
(``pricing_client``) for a city taken from the rooms on screen or the current search, never from the user's text.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from llm_approach import observability, pricing_client
from llm_approach.actions import ActionResult, TurnBudget
from llm_approach.schema import Decision
from llm_approach.state import LLMConversationState

logger = logging.getLogger(__name__)

OWNER_CONTACT = "To connect with the owner, you'll need a Homilivo subscription."
SUPPORT_FALLBACK = "Please contact the Homilivo support team for assistance."
NO_ROOMS = "Please search for a room first so I can help you with the available subscription plans."
NO_CITY = f"{OWNER_CONTACT} {SUPPORT_FALLBACK}"
FAILED = "I wasn't able to retrieve subscription plans right now. Please visit homilivo.com to view plans."
MAX_CITIES = 3

# ---- detection -----------------------------------------------------------------
# Narrow on purpose: a contact request that names the owner. "call the owner" alone, "contact support" and
# questions about subscriptions in general do not match.

_OWNER = r"(?:owner|landlord|malik)s?(?:'s|’s)?"
_PARTICLE = r"(?:\s+(?:ka|ke|ki|ko))?"
_NOUN = r"(?:number|phone|contact|details?|info|information|email|mobile|whatsapp)"
_FILLER = r"(?:\s+[a-z']+){0,3}\s+"  # up to three words between the verb and the owner ("the property")

_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # contact / reach / talk to ... the owner
        rf"\b(?:contact|reach|reach out to|get in touch(?: with)?|talk to|talk with|speak to|speak with|connect with|connect me with|message){_FILLER}{_OWNER}\b",
        # owner number, owner's phone number, owner contact details, owner ka number
        rf"\b{_OWNER}{_PARTICLE}\s+(?:{_NOUN}\s+)?{_NOUN}\b",
        # phone number of the owner, contact details for the owner
        rf"\b{_NOUN}(?:\s+{_NOUN})?\s+(?:of|for){_FILLER}{_OWNER}\b",
        # owner se baat karni hai, owner ko kaise contact karun
        rf"\b{_OWNER}\s+(?:se|ko)\b(?:\s+[a-z]+){{0,3}}\s+(?:contact|baat|bat|sampark|connect|milna)\b",
    )
)


def is_owner_contact(text: str | None) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return any(p.search(text[:500]) for p in _PATTERNS)


def wants_owner_contact(message: str | None, decision: Decision) -> bool:
    """The user's own message or the model's rewrite of it names an owner-contact request."""
    return is_owner_contact(message) or is_owner_contact(decision.standalone_query)


# ---- which city / cities ---------------------------------------------------------


def room_city(room: Any) -> str | None:
    """A usable city of one room on screen (its ``city``, or ``location.city``), else None."""
    if not isinstance(room, dict):
        return None
    city = pricing_client.clean_city(room.get("city"))
    if city is None and isinstance(room.get("location"), dict):
        city = pricing_client.clean_city(room["location"].get("city"))
    return city


def resolve_cities(validated: dict[str, Any], state: LLMConversationState) -> tuple[list[str], str]:
    """``(cities, case)``. Case: pointed (a), one_city (b), many_cities (c), search (d), no_rooms (e), no_city (f)."""
    rooms = state.rooms_on_screen
    ref = validated.get("room_reference")
    if ref == "selected":
        ref = state.selected_room_index
    if ref is None:
        ref = state.selected_room_index
    if isinstance(ref, int) and not isinstance(ref, bool) and 1 <= ref <= len(rooms):
        city = room_city(rooms[ref - 1])
        return ([city], "pointed") if city else ([], "no_city")
    if rooms:
        cities: list[str] = []
        for room in rooms:
            city = room_city(room)
            if city and city not in cities:
                cities.append(city)
        if not cities:
            return [], "no_city"
        return (cities, "one_city") if len(cities) == 1 else (cities[:MAX_CITIES], "many_cities")
    city = pricing_client.clean_city((state.current_search or {}).get("city"))
    if city:
        return [city], "search"
    return [], "no_rooms"


# ---- the reply ---------------------------------------------------------------------


def _count(event: str) -> None:
    try:
        observability.record_pricing(event)
    except Exception:  # noqa: BLE001 - metrics never break a reply
        pass


def handle_owner_contact(
    decision: Decision,
    validated: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
    **kwargs: Any,
) -> ActionResult:
    """Fixed wording plus the API's plans. Blocking (the pricing call): the pipeline runs it in a thread."""
    cities, case = resolve_cities(validated, state)
    if not cities:
        _count("pricing_no_city")
        return ActionResult(NO_ROOMS if case == "no_rooms" else NO_CITY, None, "owner_contact_no_city")  # an answer, not an error
    for _ in cities:
        _count("pricing_requested")
    results = pricing_client.fetch_plans(cities, **kwargs)
    blocks: list[str] = []
    failed = 0
    for city in cities:
        result = results[city]
        if not result.ok:
            failed += 1
            _count("pricing_failed")
            blocks.append(FAILED if len(cities) == 1 else f"**{city.title()}**: {FAILED}")
            continue
        _count("pricing_ok" if result.plans else "pricing_empty")
        blocks.append(pricing_client.format_subscription_plans(result.plans, city))
    if failed == len(cities):
        reply, error_class = FAILED, "pricing_failed"
    else:
        reply, error_class = "\n\n".join(blocks), ("pricing_partial" if failed else None)
    return ActionResult(reply, None, "owner_contact", error_class=error_class)
