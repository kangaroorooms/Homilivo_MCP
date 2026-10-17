"""Decision model and tolerant parser for the model's JSON output."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field


class Intent(str, Enum):
    NEW_SEARCH = "NEW_SEARCH"
    REFINEMENT = "REFINEMENT"
    ROOM_FOLLOWUP = "ROOM_FOLLOWUP"
    SHOW_MORE = "SHOW_MORE"
    REJECT_CURRENT = "REJECT_CURRENT"
    PLATFORM_INFO = "PLATFORM_INFO"
    GENERAL_CHAT = "GENERAL_CHAT"
    UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNCLEAR = "UNCLEAR"


class Operation(str, Enum):
    LOWEST_PRICE = "LOWEST_PRICE"
    HIGHEST_PRICE = "HIGHEST_PRICE"
    ROOM_PRICE = "ROOM_PRICE"
    ROOM_TYPE = "ROOM_TYPE"
    LOCATION = "LOCATION"
    AMENITY = "AMENITY"
    STUDENT_SUITABILITY = "STUDENT_SUITABILITY"
    FAMILY_SUITABILITY = "FAMILY_SUITABILITY"
    PROFESSIONAL_SUITABILITY = "PROFESSIONAL_SUITABILITY"
    ROOM_DETAILS = "ROOM_DETAILS"
    COMPARE = "COMPARE"


REPLY_INTENTS = frozenset(
    {Intent.GENERAL_CHAT, Intent.OUT_OF_SCOPE, Intent.UNSUPPORTED_ACTION}
)

RoomRef = Union[int, str, None]  # 1-based index, "selected" or None
RawNumber = Union[int, str, None]  # numbers stay numbers; strings go to validators


class Slots(BaseModel):
    model_config = ConfigDict(extra="ignore")

    city: str | None = None
    bhk: RawNumber = None
    min_price: RawNumber = None
    max_price: RawNumber = None
    amenity: str | None = None
    ideal_for: str | None = None


class Decision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: Intent
    operation: Operation | None = None
    slots: Slots = Field(default_factory=Slots)
    room_reference: RoomRef = None
    compare_references: list[int] = Field(default_factory=list)
    standalone_query: str | None = None
    confidence: float = 0.0
    needs_clarification: bool = False
    clarifying_question: str | None = None
    reply: str | None = None


# ---- coercion helpers ---------------------------------------------------


def _clean_str(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def _clean_number(value: Any) -> RawNumber:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == value and abs(value) != float("inf") else None
    return _clean_str(value)


def _ref_index(value: Any) -> int | None:
    """Positive 1-based index from an int, whole float or digit string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, int) and value >= 1:
        return value
    return None


def _room_reference(value: Any) -> RoomRef:
    if isinstance(value, str) and value.strip().lower() == "selected":
        return "selected"
    return _ref_index(value)


def _confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return 0.0
    if isinstance(value, (int, float)) and value == value:
        return min(1.0, max(0.0, float(value)))
    return 0.0


def _enum_or_none(enum_cls: type[Enum], value: Any):
    if isinstance(value, str):
        try:
            return enum_cls(value.strip().upper())
        except ValueError:
            return None
    return None


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _load_object(raw: Any) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def parse_decision(raw: Any) -> Decision | None:
    """Turn model output (JSON string or dict) into a Decision.

    Unknown keys are dropped, wrong types are coerced or nulled, confidence is
    clamped to 0..1. Returns None when there is no valid intent.
    """
    data = _load_object(raw)
    if data is None:
        return None

    intent = _enum_or_none(Intent, data.get("intent"))
    if intent is None:
        return None

    raw_slots = data.get("slots")
    raw_slots = raw_slots if isinstance(raw_slots, dict) else {}
    slots = Slots(
        city=_clean_str(raw_slots.get("city")),
        bhk=_clean_number(raw_slots.get("bhk")),
        min_price=_clean_number(raw_slots.get("min_price")),
        max_price=_clean_number(raw_slots.get("max_price")),
        amenity=_clean_str(raw_slots.get("amenity")),
        ideal_for=_clean_str(raw_slots.get("ideal_for")),
    )

    refs = data.get("compare_references")
    compare = [r for r in map(_ref_index, refs) if r] if isinstance(refs, list) else []

    needs = data.get("needs_clarification")
    return Decision(
        intent=intent,
        operation=_enum_or_none(Operation, data.get("operation")),
        slots=slots,
        room_reference=_room_reference(data.get("room_reference")),
        compare_references=compare,
        standalone_query=_clean_str(data.get("standalone_query")),
        confidence=_confidence(data.get("confidence")),
        needs_clarification=needs if isinstance(needs, bool) else False,
        clarifying_question=_clean_str(data.get("clarifying_question")),
        reply=_clean_str(data.get("reply")) if intent in REPLY_INTENTS else None,
    )
