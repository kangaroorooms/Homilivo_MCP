# Copied from app/guardrails/parameter_guardrail.py at commit 7dbcba3 (copied, not imported).
# Changes: VALID_ROOM_TYPES also allows 4BHK and 5BHK (two lines marked "# v2:"; validators.py already allows 1 to 5 BHK)
# Drift is checked by llm_approach/tests/test_copy_drift.py.
# ==== end of v2 provenance header ====
"""
Parameter Guardrail.

Validates structured search parameters before they reach MCP.

Optional filters such as room_types and ideal_for may be empty.
An empty list means that the user did not specify that filter.
"""

from __future__ import annotations

import math
from typing import Any

from .common import MAX_SEARCH_RESULTS


# ============================================================
# ALLOWED PARAMETERS
# ============================================================

ALLOWED_PARAMETERS = {
    "city",
    "min_price",
    "max_price",
    "room_types",
    "ideal_for",
    "sort",
    "limit",
}


# ============================================================
# VALID VALUES
# ============================================================

VALID_ROOM_TYPES = {
    "PG",
    "1BHK",
    "2BHK",
    "3BHK",
    "4BHK",  # v2: team lead decision, 4 and 5 BHK are valid searches
    "5BHK",  # v2: same
    "1RK",
    "SINGLE",
}


VALID_IDEAL_FOR = {
    "STUDENTS",
    "WORKING PROFESSIONALS",
    "FAMILY",
}


VALID_SORT_VALUES = {
    "latest",
}


# ============================================================
# HELPERS
# ============================================================

def _is_valid_number(value: Any) -> bool:
    """
    Validate finite integer or float values.

    Boolean values are rejected explicitly because bool
    is a subclass of int in Python.
    """

    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _normalize_room_type(value: str) -> str:
    return value.strip().upper()


def _normalize_ideal_for(value: str) -> str:
    return " ".join(
        value.strip().upper().split()
    )


def _normalize_sort(value: str) -> str:
    return value.strip().lower()


# ============================================================
# MAIN PARAMETER GUARDRAIL
# ============================================================

def validate_search_arguments(
    arguments: dict,
) -> tuple[bool, str]:
    """
    Validate structured room-search parameters.

    Empty room_types and ideal_for lists are allowed because
    these are optional search filters.

    Examples:

        room_types: []
        ideal_for: []

    mean that the user did not specify those filters.
    """

    # --------------------------------------------------------
    # Container validation
    # --------------------------------------------------------

    if not isinstance(arguments, dict):

        return (
            False,
            "Search arguments must be a dictionary.",
        )

    # --------------------------------------------------------
    # Unexpected parameter detection
    # --------------------------------------------------------

    unexpected_parameters = (
        set(arguments.keys())
        - ALLOWED_PARAMETERS
    )

    if unexpected_parameters:

        parameter = sorted(
            unexpected_parameters
        )[0]

        return (
            False,
            f"Unexpected search parameter: {parameter}",
        )

    # --------------------------------------------------------
    # CITY
    # --------------------------------------------------------

    city = arguments.get("city")

    if city is not None:

        if not isinstance(city, str):

            return (
                False,
                "City must be a string.",
            )

        if not city.strip():

            return (
                False,
                "City cannot be empty.",
            )

        if len(city.strip()) > 100:

            return (
                False,
                "City name is too long.",
            )

    # --------------------------------------------------------
    # MINIMUM PRICE
    # --------------------------------------------------------

    min_price = arguments.get("min_price")

    if min_price is not None:

        if not _is_valid_number(min_price):

            return (
                False,
                "Minimum price must be a valid number.",
            )

        if min_price < 0:

            return (
                False,
                "Minimum price cannot be negative.",
            )

    # --------------------------------------------------------
    # MAXIMUM PRICE
    # --------------------------------------------------------

    max_price = arguments.get("max_price")

    if max_price is not None:

        if not _is_valid_number(max_price):

            return (
                False,
                "Maximum price must be a valid number.",
            )

        if max_price < 0:

            return (
                False,
                "Maximum price cannot be negative.",
            )

    # --------------------------------------------------------
    # PRICE RANGE
    # --------------------------------------------------------

    if (
        min_price is not None
        and max_price is not None
        and min_price > max_price
    ):

        return (
            False,
            "Minimum price cannot be greater than maximum price.",
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    sort = arguments.get(
        "sort",
        "latest",
    )

    if not isinstance(sort, str):

        return (
            False,
            "Sort must be a string.",
        )

    normalized_sort = _normalize_sort(sort)

    if normalized_sort not in VALID_SORT_VALUES:

        return (
            False,
            f"Invalid sort value: {sort}",
        )

    # --------------------------------------------------------
    # RESULT LIMIT
    # --------------------------------------------------------

    limit = arguments.get(
        "limit",
        10,
    )

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
    ):

        return (
            False,
            "Result limit must be an integer.",
        )

    if limit < 1:

        return (
            False,
            "Result limit must be at least 1.",
        )

    if limit > MAX_SEARCH_RESULTS:

        return (
            False,
            f"Result limit cannot exceed {MAX_SEARCH_RESULTS}.",
        )

    # --------------------------------------------------------
    # ROOM TYPES
    # --------------------------------------------------------

    room_types = arguments.get(
        "room_types"
    )

    if room_types is not None:

        if not isinstance(room_types, list):

            return (
                False,
                "Room types must be a list.",
            )

        # Empty list is allowed.
        # It means no room-type filter was specified.

        normalized_room_types = []

        for room_type in room_types:

            if not isinstance(
                room_type,
                str,
            ):

                return (
                    False,
                    "Each room type must be a string.",
                )

            if not room_type.strip():

                return (
                    False,
                    "Room types cannot contain empty values.",
                )

            normalized_room_type = (
                _normalize_room_type(
                    room_type
                )
            )

            if normalized_room_type not in VALID_ROOM_TYPES:

                return (
                    False,
                    f"Invalid room type: {room_type}",
                )

            normalized_room_types.append(
                normalized_room_type
            )

        # Duplicate room types are still rejected.

        if len(normalized_room_types) != len(
            set(normalized_room_types)
        ):

            return (
                False,
                "Duplicate room types are not allowed.",
            )

    # --------------------------------------------------------
    # IDEAL FOR
    # --------------------------------------------------------

    ideal_for = arguments.get(
        "ideal_for"
    )

    if ideal_for is not None:

        if not isinstance(
            ideal_for,
            list,
        ):

            return (
                False,
                "Ideal-for values must be a list.",
            )

        # Empty list is allowed.
        # It means no ideal-for filter was specified.

        normalized_ideal_for = []

        for value in ideal_for:

            if not isinstance(
                value,
                str,
            ):

                return (
                    False,
                    "Each ideal-for value must be a string.",
                )

            if not value.strip():

                return (
                    False,
                    "Ideal-for values cannot be empty.",
                )

            normalized_value = (
                _normalize_ideal_for(
                    value
                )
            )

            if normalized_value not in VALID_IDEAL_FOR:

                return (
                    False,
                    f"Invalid ideal-for value: {value}",
                )

            normalized_ideal_for.append(
                normalized_value
            )

        # Duplicate ideal-for values are still rejected.

        if len(normalized_ideal_for) != len(
            set(normalized_ideal_for)
        ):

            return (
                False,
                "Duplicate ideal-for values are not allowed.",
            )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return (
        True,
        "Search parameters accepted.",
    )