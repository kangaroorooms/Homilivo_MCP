"""D4: a REFINEMENT that names a different city while the current search has other filters.

Pure helpers (no rooms calls, no state writes): the check, the summary of the remembered filters, the
question, and the classifier for the user's reply. ``search.py`` asks the question and runs the search.
"""

from __future__ import annotations

import re
from typing import Any

from llm_approach import sorting
from llm_approach.schema import Intent

PENDING_TYPE = "city_change"
MAX_ASKS = 2  # the question is asked once more when the reply is unclear, then the search is fresh

# What can be carried over to another city. The area never is: it belongs to the old city.
FILTER_KEYS = ("bhk", "min_price", "max_price", "ideal_for", "amenity", "sort")
REPLY_KEYS = ("city", "area") + FILTER_KEYS

KEEP, FRESH, REQUIREMENTS, UNCLEAR, OTHER = "keep", "fresh", "requirements", "unclear", "other"

_KEEP_STRONG = frozenset("same keep wahi purane purana".split())
_FRESH_STRONG = frozenset("new fresh naya nayi naye start scratch restart reset".split())
_YES = frozenset("yes yeah yep yup sure ok okay haan han ha ji thik theek".split())
_NO = frozenset("no nope nahi nahin nai".split())
_FILLER = frozenset(
    "please pls plz the a filters filter them it all search one wala wale hai karo kar do rakho rakhna "
    "with old from over again".split()
)
_MAX_KEYWORD_TOKENS = 10
_SEARCH_LIKE = (Intent.NEW_SEARCH, Intent.REFINEMENT)
_ANSWER_LIKE = (Intent.NEW_SEARCH, Intent.REFINEMENT, Intent.UNCLEAR, Intent.GENERAL_CHAT)


def _norm_city(city: Any) -> str:
    return re.sub(r"\s+", " ", str(city or "")).strip().lower()


def carried_filters(search: dict[str, Any]) -> dict[str, Any]:
    return {k: search.get(k) for k in FILTER_KEYS if search.get(k) is not None}


def needs_question(validated: dict[str, Any], current_search: dict[str, Any]) -> bool:
    """A different city (aliases are already canonical in ``validated``) and at least one filter to carry."""
    new_city, old_city = _norm_city(validated.get("city")), _norm_city(current_search.get("city"))
    return bool(new_city and old_city and new_city != old_city and carried_filters(current_search))


def _rupees(value: Any) -> str:
    return f"₹{value:,.0f}"


def filters_summary(search: dict[str, Any]) -> str:
    """'2 BHK rooms for family in Bangalore under ₹20,000 with wifi' (only what is remembered)."""
    bhk = search.get("bhk")
    text = f"{bhk} BHK rooms" if bhk else "rooms"
    if search.get("ideal_for"):
        text += f" for {str(search['ideal_for']).strip().lower()}"
    if search.get("city"):
        text += f" in {str(search['city']).strip().title()}"
    low, high = search.get("min_price"), search.get("max_price")
    if low is not None and high is not None:
        text += f" between {_rupees(low)} and {_rupees(high)}"
    elif high is not None:
        text += f" under {_rupees(high)}"
    elif low is not None:
        text += f" above {_rupees(low)}"
    if search.get("amenity"):
        text += f" with {str(search['amenity']).replace('-', ' ').replace('_', ' ').strip().lower()}"
    if sorting.label(search.get("sort")):
        text += f", {sorting.label(search['sort'])}"
    return text


def question(old_search: dict[str, Any], new_city: str) -> str:
    city = str(new_city).strip().title()
    return (
        f"You were looking at {filters_summary(old_search)}. Do you want to search in {city} with the "
        f"same filters, or start a new search for {city}? You can also tell me your new requirements."
    )


def pending_context(validated: dict[str, Any], asked: int = 1) -> dict[str, Any]:
    """What the original message asked for (JSON-safe); the old filters stay in ``current_search``."""
    return {"new": {k: validated[k] for k in REPLY_KEYS if validated.get(k) is not None}, "asked": asked}


def valid_context(value: Any) -> bool:
    """Shape check for a stored pending context (used when a session is loaded)."""
    if not isinstance(value, dict) or not isinstance(value.get("new"), dict):
        return False
    asked = value.get("asked")
    return isinstance(asked, int) and not isinstance(asked, bool) and 1 <= asked <= MAX_ASKS and bool(
        value["new"].get("city")
    )


def classify_reply(message: str, intent: Intent, validated: dict[str, Any]) -> str:
    """KEEP, FRESH, REQUIREMENTS, UNCLEAR, or OTHER (a different intent: the question is dropped)."""
    tokens = re.findall(r"[a-z]+", (message or "").lower())
    words = set(tokens) if len(tokens) <= _MAX_KEYWORD_TOKENS else set()
    keep_strong, fresh_strong = bool(words & _KEEP_STRONG), bool(words & _FRESH_STRONG)
    if keep_strong and fresh_strong:
        return UNCLEAR
    if keep_strong:
        return KEEP
    if fresh_strong:
        return FRESH
    if words and words <= (_YES | _NO | _FILLER):
        yes, no = bool(words & _YES), bool(words & _NO)
        if yes != no:
            return KEEP if yes else FRESH
        return UNCLEAR
    if intent in _SEARCH_LIKE and any(validated.get(k) is not None for k in REPLY_KEYS):
        return REQUIREMENTS
    return UNCLEAR if intent in _ANSWER_LIKE else OTHER
