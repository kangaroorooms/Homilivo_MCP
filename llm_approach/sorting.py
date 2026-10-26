"""The sort preference of a search ("cheapest first"), kept per session in ``current_search["sort"]``.

The rooms API and the parameter guardrail know one sort value, ``latest``, so the preference is applied here, to
the rooms a search returned (one page: it orders those rooms, it does not re-rank the whole city). The value sent
to the API comes from ``api_sort``; when the API gains price sorts, only ``_API_SORT`` changes.
"""

from __future__ import annotations

import re
from typing import Any

from llm_approach import config

LABELS = {
    "price_asc": "cheapest first",
    "price_desc": "most expensive first",
    "rating_desc": "highest rated first",
}
RESET = "latest"  # a cue that clears the preference (newest first is the API's own order)

_API_SORT: dict[str, str] = {}  # preference -> API sort value; none are supported yet, so all use ROOMS_SORT

_CUES = (
    ("price_asc", re.compile(
        r"\b(?:cheapest|cheaper|lowest\s+(?:price|rent|cost)|least\s+expensive|low\s+to\s+high|sasta|sasti|sabse\s+sasta)\b")),
    ("price_desc", re.compile(r"\b(?:most\s+expensive|costliest|highest\s+(?:price|rent)|high\s+to\s+low)\b")),
    ("rating_desc", re.compile(r"\b(?:highest\s+rated|best\s+rated|top\s+rated|highest\s+rating|best\s+reviewed)\b")),
    (RESET, re.compile(r"\b(?:newest|latest\s+first|most\s+recent)\b")),
)


def detect(message: Any) -> str | None:
    """The sort the message asks for (``RESET`` to clear it), or None when it says nothing about sorting."""
    if not isinstance(message, str):
        return None
    text = message.lower()
    for value, pattern in _CUES:
        if pattern.search(text):
            return value
    return None


def clean(value: Any) -> str | None:
    """A stored value that is a known preference, else None."""
    return value if value in LABELS else None


def label(value: Any) -> str | None:
    return LABELS.get(value) if isinstance(value, str) else None


def api_sort(search: dict[str, Any]) -> str:
    return _API_SORT.get(search.get("sort"), config.ROOMS_SORT)


def _key(room: dict[str, Any], field: str) -> float | None:
    from llm_approach import responder

    if field == "price":
        return responder.price_of(room)
    value = room.get("rating")
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def apply(rooms: list[dict[str, Any]], sort: Any) -> list[dict[str, Any]]:
    """The rooms in the preferred order (stable; rooms without a price or rating go last)."""
    sort = clean(sort)
    if sort is None:
        return rooms
    field = "price" if sort.startswith("price") else "rating"
    sign = 1 if sort.endswith("asc") else -1
    known = [r for r in rooms if _key(r, field) is not None]
    unknown = [r for r in rooms if _key(r, field) is None]
    return sorted(known, key=lambda r: sign * _key(r, field)) + unknown
