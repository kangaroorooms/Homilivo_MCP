"""POC label -> (new intent, operation), PLAN.md section 6.4.

Checked against ``poc_classifier/labels.py`` (23 labels). Two differences from
PLAN 6.4 as written:

* ``ROOM_TABLE`` is in the POC label file but not in the PLAN table. Added here as
  ROOM_FOLLOWUP / ROOM_DETAILS (the schema has no table operation).
* The POC name ``SHOW_MORE_OPTIONS`` is the new intent ``SHOW_MORE``.

Pure data and functions; no imports from ``app/`` or ``poc_classifier/``.
"""

from __future__ import annotations

from llm_approach.schema import Intent, Operation

ADDED_TO_PLAN = {"ROOM_TABLE": "ROOM_FOLLOWUP / ROOM_DETAILS (not in PLAN 6.4)"}

POC_TO_NEW: dict[str, tuple[str, str | None]] = {
    "NEW_SEARCH": ("NEW_SEARCH", None),
    "REFINEMENT": ("REFINEMENT", None),
    "SHOW_MORE_OPTIONS": ("SHOW_MORE", None),
    "REJECT_CURRENT": ("REJECT_CURRENT", None),
    "LOWEST_PRICE": ("ROOM_FOLLOWUP", "LOWEST_PRICE"),
    "HIGHEST_PRICE": ("ROOM_FOLLOWUP", "HIGHEST_PRICE"),
    "ROOM_PRICE": ("ROOM_FOLLOWUP", "ROOM_PRICE"),
    "ROOM_TYPE": ("ROOM_FOLLOWUP", "ROOM_TYPE"),
    "LOCATION": ("ROOM_FOLLOWUP", "LOCATION"),
    "AMENITY": ("ROOM_FOLLOWUP", "AMENITY"),
    "STUDENT_SUITABILITY": ("ROOM_FOLLOWUP", "STUDENT_SUITABILITY"),
    "FAMILY_SUITABILITY": ("ROOM_FOLLOWUP", "FAMILY_SUITABILITY"),
    "PROFESSIONAL_SUITABILITY": ("ROOM_FOLLOWUP", "PROFESSIONAL_SUITABILITY"),
    "ROOM_DETAILS": ("ROOM_FOLLOWUP", "ROOM_DETAILS"),
    "ROOM_TABLE": ("ROOM_FOLLOWUP", "ROOM_DETAILS"),
    "COMPARE_ROOMS": ("ROOM_FOLLOWUP", "COMPARE"),
    "GREETING": ("GENERAL_CHAT", None),
    "ACKNOWLEDGMENT": ("GENERAL_CHAT", None),
    "CLOSING": ("GENERAL_CHAT", None),
    "PLATFORM_INFO": ("PLATFORM_INFO", None),
    "UNSUPPORTED_ACTION": ("UNSUPPORTED_ACTION", None),
    "OUT_OF_SCOPE": ("OUT_OF_SCOPE", None),
    "UNCLEAR": ("UNCLEAR", None),
}

# Intents the new decision layer has that no POC label maps to (the POC label set was frozen
# before session memory). No gold label is ever CONVERSATION_RECALL, so the baseline comparison
# is unchanged: the 186 evaluation items contain no recall question, and a prediction of this
# intent for a POC item is simply counted as a wrong intent. The old router has no equivalent
# ("what is my name" went to OUT_OF_SCOPE), so there is nothing to map it to.
NEW_ONLY_INTENTS = frozenset({"CONVERSATION_RECALL"})

assert all(i in Intent.__members__ for i, _ in POC_TO_NEW.values())
assert NEW_ONLY_INTENTS <= set(Intent.__members__)
assert {i for i, _ in POC_TO_NEW.values()} | NEW_ONLY_INTENTS == set(Intent.__members__)
assert all(o is None or o in Operation.__members__ for _, o in POC_TO_NEW.values())


def map_label(poc_label: str | None) -> tuple[str, str | None] | None:
    """(intent, operation) for a POC label, or None for an unknown/missing label."""
    if poc_label is None:
        return None
    return POC_TO_NEW.get(poc_label)


def intent_of(poc_label: str | None) -> str | None:
    mapped = map_label(poc_label)
    return mapped[0] if mapped else None


def acceptable_intents(acceptable: list[str] | None) -> set[str]:
    """Intents of the ``acceptable`` POC labels (unknown labels ignored)."""
    return {i for i in (intent_of(a) for a in acceptable or []) if i}
