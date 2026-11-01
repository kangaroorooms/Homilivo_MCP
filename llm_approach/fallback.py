"""Deterministic path used when Groq is down, times out or returns invalid JSON twice.

Uses the app's rule-based slot extractor (``fast_parser.extract_search_parameters``)
through ``reuse``. Every extracted value still goes through ``validators``, so an
invalid or missing city never reaches the rooms client. No Ollama, no Gemma.
"""

from __future__ import annotations

import re
from typing import Any

from llm_approach import observability, reuse, validators
from llm_approach.actions import ActionResult, TurnBudget
from llm_approach.actions.clarify import ASK_AFTER_RESULTS
from llm_approach.actions.search import handle_new_search
from llm_approach.schema import Decision, Intent
from llm_approach.state import LLMConversationState

ASK_CITY_AND_BUDGET = "Could you tell me the city and budget you're looking for?"

_BHK_RE = re.compile(r"(\d)\s*bhk", re.IGNORECASE)


def _extract(message: str) -> dict[str, Any]:
    try:
        return reuse.get_extract_search_parameters()(message) or {}
    except Exception:
        return {}


def _bhk_from(room_types: Any) -> int | None:
    for item in room_types or []:
        match = _BHK_RE.fullmatch(str(item).strip())
        if match:
            return int(match.group(1))
    return None


def fallback_slots(message: str, state: LLMConversationState) -> dict[str, Any] | None:
    """Validated slots for a search, or None when no usable city was found."""
    params = _extract(message)
    raw = {
        "city": params.get("city"),
        "bhk": _bhk_from(params.get("room_types")),
        "min_price": params.get("min_price"),
        "max_price": params.get("max_price"),
        "ideal_for": (params.get("ideal_for") or [None])[0],
    }
    clean, _ = validators.validate_slots(raw)
    if clean.get("city"):
        return clean
    # A bare city name answering a pending "which city?" question.
    if (state.pending_question or {}).get("type") == "city":
        city = validators.validate_city(message)
        if city:
            clean["city"] = city
            return clean
    return None


def run_fallback(message: str, state: LLMConversationState, budget: TurnBudget) -> ActionResult:
    """Search when a valid city can be found by rules, otherwise ask for it.

    Blocking (the search calls the rooms API): run through ``asyncio.to_thread``.
    """
    if (state.pending_question or {}).get("type") == "city_change":
        state.clear_pending()  # no model to read the reply: the rule-based path always searches fresh
        observability.record_city_change("abandoned")
    clean = fallback_slots(message, state)
    if clean is None and state.rooms_on_screen:
        # Rooms are on screen: asking for a city again would ignore them.
        state.set_pending("clarify")
        return ActionResult(ASK_AFTER_RESULTS, None, "fallback_ask", error_class="llm_unavailable")
    if clean is None:
        state.set_pending("city")
        return ActionResult(ASK_CITY_AND_BUDGET, None, "fallback_ask", error_class="llm_unavailable")
    decision = Decision(intent=Intent.NEW_SEARCH, confidence=1.0)
    result = handle_new_search(decision, clean, state, budget)
    result.action = "fallback_" + result.action
    result.error_class = result.error_class or "llm_unavailable"
    return result
