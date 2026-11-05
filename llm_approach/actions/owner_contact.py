"""Owner-contact requests ("how do I contact the owner"): show the Homilivo subscription plans of the city.

Runs only when the model's validated intent is UNSUPPORTED_ACTION or PLATFORM_INFO and the user's own message (or the model's
``standalone_query``) names a contact request. Nothing runs before the LLM call. The plans come from the pricing API
(``pricing_client``) for a city taken from the rooms on screen or the current search, never from the user's text.

Design change (price questions): a price or plan-cost question ("platinum plan price in Bangalore") takes the same route. Its city is the
model's ``slots.city`` or a city in the user's message, only when it is one of the known cities (the canonical name, never the user's
text), else the session city.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from llm_approach import observability, pricing_client, reuse, turn_deadline
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
TURN_RESERVE_S = 3.0  # kept free of the turn deadline for the output guard, the session save and the reply
PRICE_HEAD = "Here are the Homilivo plans for **{city}**:"  # first line of a price answer (the View plans button reads it)
OWNER_CONTACT_HEAD = "To connect with the owner, you'll need a Homilivo subscription for"  # opening of the shared plan format

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


# ---- price questions -----------------------------------------------------------
# A price cue together with a plan word (plan, plans, subscription, membership); "gold" or "platinum" alone is not one (jewellery).
# "what are the plans" and "do I need a separate plan for each city" have no price cue and stay on the knowledge base. A message about
# refunds, cancellation or a failed payment never takes this route. Matched on the user's own message only (a rewrite can drift).

_PRICE_CUE = re.compile(r"\b(?:price|prices|pricing|cost|costs|how much|charge|charges|fee|fees|rate|rates)\b", re.IGNORECASE)
_PLAN_WORD = re.compile(r"\b(?:plan|plans|subscription|subscriptions|membership|memberships)\b", re.IGNORECASE)
_NOT_PRICE = re.compile(r"\b(?:refund\w*|cancel\w*|deducted|failed payment|payment (?:has |had )?failed)\b", re.IGNORECASE)


def is_price_question(text: str | None) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    text = text[:500]
    return bool(_PRICE_CUE.search(text) and _PLAN_WORD.search(text) and not _NOT_PRICE.search(text))


def named_city(raw: Any) -> str | None:
    """The model's ``slots.city`` as a cleaned canonical city (lowercase, as ``clean_city``) only when it is a known city or a
    known alias of one; None otherwise. The returned value comes from the known-city list, never from the text."""
    if not isinstance(raw, str):
        return None
    key = re.sub(r"\s+", " ", raw).strip(" .,;:!?\"'").lower()
    known = {c.lower() for c in reuse.get_known_cities()}
    if key in known:
        return pricing_client.clean_city(key)
    canonical = reuse.get_city_aliases().get(key)
    if isinstance(canonical, str) and canonical.lower() in known:
        return pricing_client.clean_city(canonical)
    return None


def message_city(message: str | None) -> str | None:
    """A known city named in the user's message, found by ``fast_parser``'s city extraction and validated like ``named_city``."""
    if not isinstance(message, str) or not message.strip():
        return None
    try:
        return named_city(reuse.get_extract_search_parameters()(message[:500]).get("city"))
    except Exception:  # noqa: BLE001 - the extractor never breaks a turn; the session city is used instead
        return None


def price_cities(message: str | None, decision: Decision, validated: dict[str, Any], state: LLMConversationState) -> list[str]:
    """Cities for a price question; ``[]`` when it is not one or no city is known (the knowledge base answers then).
    Order: the model's ``slots.city`` if known, a known city in the user's message, then the session city (rooms on screen or
    the current search)."""
    if not is_price_question(message):
        return []
    named = named_city(decision.slots.city) or message_city(message)
    if named:
        return [named]
    return resolve_cities(validated, state)[0]


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
    return _plans_reply(cities, "owner_contact", **kwargs)


def handle_price_question(cities: list[str], **kwargs: Any) -> ActionResult:
    """The plans of the given cities (already cleaned) for a price question. Blocking, like ``handle_owner_contact``."""
    return _plans_reply(cities, "price_question", **kwargs)


def _plans_reply(cities: list[str], action: str, **kwargs: Any) -> ActionResult:
    for _ in cities:
        _count("pricing_requested")
    left = turn_deadline.remaining()  # None outside the server's deadline (CLI, eval)
    if left is not None and "total_timeout" not in kwargs:
        # never wait past the turn deadline: keep a reserve for the output guard, the save and the reply
        kwargs["total_timeout"] = min(pricing_client.PRICING_TOTAL_TIMEOUT_S, left - TURN_RESERVE_S)
        if kwargs["total_timeout"] <= 0:  # no time left for even one request: the usual failure reply, not "took too long"
            for _ in cities:
                _count("pricing_failed")
            return ActionResult(FAILED, None, action, error_class="pricing_failed")
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
        text = pricing_client.format_subscription_plans(result.plans, city)
        if action == "price_question" and result.plans:
            # the shared builder opens with the owner-contact sentence; a price answer opens with the plain heading
            text = text.replace(OWNER_CONTACT_HEAD + f" **{city.title()}**:", PRICE_HEAD.format(city=city.title()), 1)
        blocks.append(text)
    if failed == len(cities):
        reply, error_class = FAILED, "pricing_failed"
    else:
        reply, error_class = "\n\n".join(blocks), ("pricing_partial" if failed else None)
    return ActionResult(reply, None, action, error_class=error_class)
