"""
API Data Guardrail.

Validates room records returned by MCP/Homilivo before they are
used by the chatbot response-generation pipeline.

This guardrail protects the application from:
- malformed API responses
- missing required room fields
- invalid identifiers
- invalid prices
- non-string/empty room types (the specific set of room-type VALUES
  is intentionally NOT enforced here - see the comment above the
  roomType check in validate_room_data() for why)
- malformed ideal-for data
- invalid optional room metadata
"""

from __future__ import annotations

import math
from typing import Any


# ============================================================
# REQUIRED ROOM FIELDS
# ============================================================

REQUIRED_ROOM_FIELDS = {
    "id",
    "title",
    "city",
    "pricePerMonth",
    "roomType",
}


# ============================================================
# VALID IDEAL-FOR VALUES
# ============================================================

VALID_IDEAL_FOR = {
    "STUDENTS",
    "WORKING PROFESSIONALS",
    "FAMILY",
    # Confirmed via live Homilivo API testing across 7 cities + Mumbai
    # (2026-09-18): the backend returns this value in room idealFor
    # data. Incoming-data value only - NOT added to
    # parameter_guardrail.py/schemas/chat.py, since there is no
    # confirmation Homilivo accepts it as a search FILTER parameter.
    "SMALL FAMILY",
}


# ============================================================
# VALID AVAILABILITY VALUES
# ============================================================

VALID_AVAILABILITY_STATUS = {
    "AVAILABLE",
    "FILLED",
    "OWNER_PAUSED",
    "NEEDS_RECONFIRMATION",
}


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _is_valid_string(
    value: Any,
) -> bool:

    return (
        isinstance(value, str)
        and bool(value.strip())
    )


def _normalize_ideal_for(
    value: str,
) -> str:

    return " ".join(
        value.strip().upper().split()
    )


def _normalize_availability(
    value: str,
) -> str:

    return value.strip().upper()


def _is_valid_number(
    value: Any,
) -> bool:

    return (
        not isinstance(value, bool)
        and isinstance(
            value,
            (int, float),
        )
        and math.isfinite(
            float(value)
        )
    )


# ============================================================
# MAIN API DATA GUARDRAIL
# ============================================================

def validate_room_data(
    rooms: list,
) -> tuple[bool, str]:

    # --------------------------------------------------------
    # TOP-LEVEL RESPONSE
    # --------------------------------------------------------

    if not isinstance(
        rooms,
        list,
    ):

        return (
            False,
            "Invalid room data format.",
        )

    # --------------------------------------------------------
    # EMPTY RESULT
    # --------------------------------------------------------

    if not rooms:

        return (
            True,
            "No matching rooms returned.",
        )

    # --------------------------------------------------------
    # ROOM RECORDS
    # --------------------------------------------------------

    for index, room in enumerate(
        rooms
    ):

        # ----------------------------------------------------
        # ROOM OBJECT
        # ----------------------------------------------------

        if not isinstance(
            room,
            dict,
        ):

            return (
                False,
                f"Invalid room record at index {index}.",
            )

        # ----------------------------------------------------
        # REQUIRED FIELDS
        # ----------------------------------------------------

        missing_fields = (
            REQUIRED_ROOM_FIELDS
            - room.keys()
        )

        if missing_fields:

            field = sorted(
                missing_fields
            )[0]

            return (
                False,
                f"Room data missing required field: {field}",
            )

        # ----------------------------------------------------
        # ROOM ID
        # ----------------------------------------------------

        room_id = room.get(
            "id"
        )

        if not _is_valid_string(
            room_id
        ):

            return (
                False,
                f"Invalid room ID: {room_id}",
            )

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        title = room.get(
            "title"
        )

        if not _is_valid_string(
            title
        ):

            return (
                False,
                f"Invalid title for room {room_id}",
            )

        # ----------------------------------------------------
        # CITY
        # ----------------------------------------------------

        city = room.get(
            "city"
        )

        if not _is_valid_string(
            city
        ):

            return (
                False,
                f"Invalid city for room {room_id}",
            )

        # ----------------------------------------------------
        # PRICE
        # ----------------------------------------------------

        price = room.get(
            "pricePerMonth"
        )

        if not _is_valid_number(
            price
        ):

            return (
                False,
                f"Invalid price for room {room_id}",
            )

        if price < 0:

            return (
                False,
                f"Invalid negative price for room {room_id}",
            )

        # ----------------------------------------------------
        # ROOM TYPE
        #
        # roomType is intentionally NOT restricted to a closed enum
        # here. Homilivo's real API returns room types beyond the
        # small set this project has seen so far (e.g. "2RK"), and
        # rejecting the ENTIRE search response because one room used
        # an unrecognized-but-well-formed type is a false rejection,
        # not a security control - the enum lived only in this
        # guardrail's assumptions, not in any documented backend
        # contract. The original backend value is preserved verbatim
        # (never rewritten/mapped); only structural validity (a real,
        # non-empty string) is enforced, matching how title/city/
        # location are validated just above and below.
        # ----------------------------------------------------

        room_type = room.get(
            "roomType"
        )

        if not _is_valid_string(
            room_type
        ):

            return (
                False,
                f"Invalid room type for room {room_id}",
            )

        # ----------------------------------------------------
        # IDEAL FOR
        # ----------------------------------------------------

        ideal_for = room.get(
            "idealFor"
        )

        if ideal_for is not None:

            if not isinstance(
                ideal_for,
                list,
            ):

                return (
                    False,
                    f"Invalid idealFor data "
                    f"for room {room_id}",
                )

            for value in ideal_for:

                if not _is_valid_string(
                    value
                ):

                    return (
                        False,
                        f"Invalid idealFor value "
                        f"for room {room_id}",
                    )

                normalized_value = (
                    _normalize_ideal_for(
                        value
                    )
                )

                if normalized_value not in (
                    VALID_IDEAL_FOR
                ):

                    return (
                        False,
                        f"Invalid idealFor value "
                        f"'{value}' "
                        f"for room {room_id}",
                    )

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        location = room.get(
            "location"
        )

        if location is not None:

            if not _is_valid_string(
                location
            ):

                return (
                    False,
                    f"Invalid location "
                    f"for room {room_id}",
                )

        # ----------------------------------------------------
        # AVAILABILITY STATUS
        # ----------------------------------------------------

        availability_status = room.get(
            "availabilityStatus"
        )

        if availability_status is not None:

            if not _is_valid_string(
                availability_status
            ):

                return (
                    False,
                    f"Invalid availability status "
                    f"for room {room_id}",
                )

            normalized_status = (
                _normalize_availability(
                    availability_status
                )
            )

            if normalized_status not in (
                VALID_AVAILABILITY_STATUS
            ):

                return (
                    False,
                    f"Invalid availability status "
                    f"'{availability_status}' "
                    f"for room {room_id}",
                )

        # ----------------------------------------------------
        # IMAGES
        # ----------------------------------------------------

        images = room.get(
            "images"
        )

        if images is not None:

            if not isinstance(
                images,
                list,
            ):

                return (
                    False,
                    f"Invalid images data "
                    f"for room {room_id}",
                )

            for image in images:

                if not _is_valid_string(
                    image
                ):

                    return (
                        False,
                        f"Invalid image URL "
                        f"for room {room_id}",
                    )

        # ----------------------------------------------------
        # RATING
        # ----------------------------------------------------

        rating = room.get(
            "rating"
        )

        if rating is not None:

            if not _is_valid_number(
                rating
            ):

                return (
                    False,
                    f"Invalid rating "
                    f"for room {room_id}",
                )

            if rating < 0 or rating > 5:

                return (
                    False,
                    f"Rating must be between 0 and 5 "
                    f"for room {room_id}",
                )

        # ----------------------------------------------------
        # REVIEW COUNT
        # ----------------------------------------------------

        reviews_count = room.get(
            "reviewsCount"
        )

        if reviews_count is not None:

            if (
                isinstance(
                    reviews_count,
                    bool,
                )
                or not isinstance(
                    reviews_count,
                    int,
                )
            ):

                return (
                    False,
                    f"Invalid review count "
                    f"for room {room_id}",
                )

            if reviews_count < 0:

                return (
                    False,
                    f"Review count cannot be negative "
                    f"for room {room_id}",
                )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return (
        True,
        "Room data validated.",
    )