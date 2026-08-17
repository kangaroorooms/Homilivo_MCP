# ============================================================
# INPUT GUARDRAIL
# ============================================================

ALLOWED_KEYWORDS = [
    "room",
    "rooms",
    "pg",
    "1bhk",
    "2bhk",
    "3bhk",
    "1rk",
    "flat",
    "apartment",
    "rent",
    "stay",
    "accommodation",
    "hostel",
    "house",
    "student",
    "students",
    "professional",
    "professionals",
    "working professional",
    "working professionals"
]


BLOCKED_PATTERNS = [
    "api key",
    "api token",
    "bearer token",
    "access token",
    "password",
    "secret",
    "hack",
    "malware",
    "ransomware",
    "exploit",
    "sql injection"
]


def validate_user_input(user_query: str) -> tuple[bool, str]:

    # Empty request
    if not user_query or not user_query.strip():

        return (
            False,
            "Please enter a room-search request."
        )

    # Length protection
    if len(user_query) > 500:

        return (
            False,
            "The request is too long. Please provide a shorter room-search request."
        )

    query = user_query.lower()

    # Sensitive/security-related requests
    for pattern in BLOCKED_PATTERNS:

        if pattern in query:

            return (
                False,
                "I can only help with accommodation and room searches."
            )

    # Room/accommodation relevance check
    if not any(
        keyword in query
        for keyword in ALLOWED_KEYWORDS
    ):

        return (
            False,
            "I can only help with accommodation and room searches."
        )

    return True, "Input accepted."


# ============================================================
# PARAMETER GUARDRAIL
# ============================================================

VALID_ROOM_TYPES = {
    "PG",
    "1BHK",
    "2BHK",
    "3BHK",
    "1RK",
    "Single"
}


def validate_search_arguments(
    arguments: dict
) -> tuple[bool, str]:

    if not isinstance(arguments, dict):

        return (
            False,
            "Search arguments must be a dictionary."
        )

    # --------------------------------------------------------
    # Validate minimum price
    # --------------------------------------------------------

    min_price = arguments.get("min_price")

    if min_price is not None:

        if isinstance(min_price, bool) or not isinstance(min_price, int):

            return (
                False,
                "Minimum price must be an integer."
            )

        if min_price < 0:

            return (
                False,
                "Minimum price cannot be negative."
            )

    # --------------------------------------------------------
    # Validate maximum price
    # --------------------------------------------------------

    max_price = arguments.get("max_price")

    if max_price is not None:

        if isinstance(max_price, bool) or not isinstance(max_price, int):

            return (
                False,
                "Maximum price must be an integer."
            )

        if max_price < 0:

            return (
                False,
                "Maximum price cannot be negative."
            )

    # --------------------------------------------------------
    # Validate price range
    # --------------------------------------------------------

    if (
        min_price is not None
        and max_price is not None
        and min_price > max_price
    ):

        return (
            False,
            "Minimum price cannot be greater than maximum price."
        )

    # --------------------------------------------------------
    # Validate result limit
    # --------------------------------------------------------

    limit = arguments.get("limit", 10)

    if isinstance(limit, bool) or not isinstance(limit, int):

        return (
            False,
            "Result limit must be an integer."
        )

    if limit < 1 or limit > 50:

        return (
            False,
            "Result limit must be between 1 and 50."
        )

    # --------------------------------------------------------
    # Validate room types
    # --------------------------------------------------------

    room_types = arguments.get("room_types")

    if room_types is not None:

        if not isinstance(room_types, list):

            return (
                False,
                "Room types must be a list."
            )

        for room_type in room_types:

            if room_type not in VALID_ROOM_TYPES:

                return (
                    False,
                    f"Invalid room type: {room_type}"
                )

    return True, "Search parameters accepted."


# ============================================================
# API DATA GUARDRAIL
# ============================================================

# These fields are confirmed from the real Homilivo API.
REQUIRED_ROOM_FIELDS = [
    "id",
    "title",
    "city",
    "pricePerMonth",
    "roomType"
]


def validate_room_data(
    rooms: list
) -> tuple[bool, str]:

    if not isinstance(rooms, list):

        return (
            False,
            "Invalid room data format."
        )

    for room in rooms:

        if not isinstance(room, dict):

            return (
                False,
                "Invalid room record."
            )

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        for field in REQUIRED_ROOM_FIELDS:

            if field not in room:

                return (
                    False,
                    f"Room data missing required field: {field}"
                )

        # ----------------------------------------------------
        # Validate ID
        # ----------------------------------------------------

        if not isinstance(room.get("id"), str):

            return (
                False,
                f"Invalid room ID: {room.get('id')}"
            )

        # ----------------------------------------------------
        # Validate title
        # ----------------------------------------------------

        if not isinstance(room.get("title"), str):

            return (
                False,
                f"Invalid title for room {room.get('id')}"
            )

        # ----------------------------------------------------
        # Validate city
        # ----------------------------------------------------

        if not isinstance(room.get("city"), str):

            return (
                False,
                f"Invalid city for room {room.get('id')}"
            )

        # ----------------------------------------------------
        # Validate price
        # REAL API FIELD = pricePerMonth
        # ----------------------------------------------------

        price = room.get("pricePerMonth")

        if isinstance(price, bool) or not isinstance(
            price,
            (int, float)
        ):

            return (
                False,
                f"Invalid price for room {room.get('id')}"
            )

        if price < 0:

            return (
                False,
                f"Invalid negative price for room {room.get('id')}"
            )

        # ----------------------------------------------------
        # Validate room type
        # ----------------------------------------------------

        room_type = room.get("roomType")

        if not isinstance(room_type, str):

            return (
                False,
                f"Invalid room type for room {room.get('id')}"
            )

    return True, "Room data validated."


# ============================================================
# OUTPUT GUARDRAIL
# ============================================================

def validate_ai_response(
    response: str
) -> tuple[bool, str]:

    # Empty response
    if not response or not response.strip():

        return (
            False,
            "The AI returned an empty response."
        )

    # --------------------------------------------------------
    # Prevent accidental exposure of credentials
    # --------------------------------------------------------

    sensitive_patterns = [
        "kangaroo_token",
        "authorization: bearer",
        "bearer token",
        "api key",
        "access token",
        "password",
        "secret"
    ]

    response_lower = response.lower()

    for pattern in sensitive_patterns:

        if pattern in response_lower:

            return (
                False,
                "The response contained potentially sensitive information."
            )

    # --------------------------------------------------------
    # Basic length protection
    # --------------------------------------------------------

    if len(response) > 5000:

        return (
            False,
            "The AI response is unexpectedly long."
        )

    return True, "Response validated."