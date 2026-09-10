"""
Strict Pydantic schema for LLM-extracted room-search parameters.

This is the Phase 6 "Pydantic Extraction" box in the target
architecture diagram: it sits between the LLM extraction fallback
(app/experiments/agent_modes.py, used only when the deterministic
regex fast_parser finds nothing useful) and the UNCHANGED parameter
guardrail (guardrails/parameter_guardrail.py).

The field set and every allowed value are mirrored EXACTLY from
guardrails/parameter_guardrail.py's ALLOWED_PARAMETERS, VALID_ROOM_TYPES,
VALID_IDEAL_FOR and VALID_SORT_VALUES - per this project's rule against
inventing enum values or parameters the guardrail does not recognize.

Note: the guardrail has no "gender_preference" parameter at all (it is
not part of ALLOWED_PARAMETERS), so this schema deliberately does not
include one either - adding it would either be silently dropped or
rejected outright as an "Unexpected search parameter" by the guardrail.
A query implying a gender preference (e.g. "for gurls") has no
representation in the current search contract and is correctly left
unexpressed, exactly like the deterministic fast_parser already does.

This schema is a second line of defense, not the authority: the
parameter guardrail still runs, unchanged, on whatever this schema
produces.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.guardrails.common import MAX_SEARCH_RESULTS
from app.guardrails.parameter_guardrail import (
    VALID_IDEAL_FOR,
    VALID_ROOM_TYPES,
    VALID_SORT_VALUES,
)

# ============================================================
# DISPLAY-FORM MAPS
# ============================================================
# The guardrail's VALID_ROOM_TYPES/VALID_IDEAL_FOR sets are the
# canonical (upper-case) form used only for membership checks. The rest
# of the pipeline (fast_parser, session state, the Homilivo API call)
# uses a specific display casing - mirrored here so LLM-extracted
# values merge into search_request in the same shape fast_parser would
# have produced, never a differently-cased duplicate.

_ROOM_TYPE_DISPLAY: dict[str, str] = {value: value for value in VALID_ROOM_TYPES}
_ROOM_TYPE_DISPLAY["SINGLE"] = "Single"

_IDEAL_FOR_DISPLAY: dict[str, str] = {
    "STUDENTS": "Students",
    "WORKING PROFESSIONALS": "Working Professionals",
    "FAMILY": "Family",
}


class SearchParameters(BaseModel):
    """
    Validated room-search parameters extracted by the LLM fallback.

    Unknown/extra fields are rejected outright (extra="forbid") so a
    hallucinated field (e.g. "gender_preference") fails validation
    instead of silently reaching the parameter guardrail.
    """

    model_config = ConfigDict(extra="forbid")

    city: Optional[str] = None
    min_price: Optional[float] = Field(default=None, ge=0)
    max_price: Optional[float] = Field(default=None, ge=0)
    room_types: list[str] = Field(default_factory=list)
    ideal_for: list[str] = Field(default_factory=list)
    sort: str = "latest"
    limit: int = Field(default=10, ge=1, le=MAX_SEARCH_RESULTS)

    @field_validator("city", mode="before")
    @classmethod
    def _validate_city(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("City must be a string.")
        value = value.strip()
        if not value:
            return None
        if len(value) > 100:
            raise ValueError("City name is too long.")
        return value

    @field_validator("room_types", mode="before")
    @classmethod
    def _validate_room_types(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("Room types must be a list.")

        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Each room type must be a non-empty string.")
            candidate = item.strip().upper()
            if candidate not in VALID_ROOM_TYPES:
                raise ValueError(f"Invalid room type: {item}")
            display = _ROOM_TYPE_DISPLAY[candidate]
            if display not in normalized:
                normalized.append(display)
        return normalized

    @field_validator("ideal_for", mode="before")
    @classmethod
    def _validate_ideal_for(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("Ideal-for values must be a list.")

        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Each ideal-for value must be a non-empty string.")
            candidate = " ".join(item.strip().upper().split())
            if candidate not in VALID_IDEAL_FOR:
                raise ValueError(f"Invalid ideal-for value: {item}")
            display = _IDEAL_FOR_DISPLAY[candidate]
            if display not in normalized:
                normalized.append(display)
        return normalized

    @field_validator("sort", mode="before")
    @classmethod
    def _validate_sort(cls, value: Any) -> Any:
        if value is None:
            return "latest"
        if not isinstance(value, str):
            raise ValueError("Sort must be a string.")
        candidate = value.strip().lower()
        if candidate not in VALID_SORT_VALUES:
            raise ValueError(f"Invalid sort value: {value}")
        return candidate

    @model_validator(mode="after")
    def _validate_price_range(self) -> "SearchParameters":
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price cannot exceed max_price.")
        return self

    def to_search_arguments(self) -> dict[str, Any]:
        """
        Render this model back into the same dict shape
        app.fast_parser.extract_search_parameters() produces, so callers
        can merge/compare it identically regardless of which tier
        (regex or LLM) produced it.
        """

        min_price = None if self.min_price is None else int(self.min_price)
        max_price = None if self.max_price is None else int(self.max_price)

        return {
            "city": self.city,
            "min_price": min_price,
            "max_price": max_price,
            "room_types": list(self.room_types),
            "ideal_for": list(self.ideal_for),
            "sort": self.sort,
            "limit": self.limit,
        }
