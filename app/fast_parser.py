import re


# ============================================================
# FAST USER REQUEST PARSER
# ============================================================

ROOM_TYPE_PATTERNS = [
    "PG",
    "1BHK",
    "2BHK",
    "3BHK",
    "1RK",
    "Single"
]


IDEAL_FOR_PATTERNS = {
    "student": "Students",
    "students": "Students",
    "working professional": "Working Professionals",
    "working professionals": "Working Professionals",
    "professional": "Working Professionals",
    "professionals": "Working Professionals",
    "family": "Family",
    "families": "Family"
}


def extract_search_parameters(
    user_query: str
) -> dict:

    query = user_query.strip()

    result = {
        "city": None,
        "min_price": None,
        "max_price": None,
        "room_types": [],
        "ideal_for": [],
        "sort": "latest",
        "limit": 10
    }

    # ========================================================
    # CITY
    # ========================================================

    # Handles:
    # "in Jaipur"
    # "from Jaipur"
    # "Jaipur rooms"

    city_match = re.search(
        r"\b(?:in|from|at)\s+([A-Za-z]+)",
        query,
        re.IGNORECASE
    )

    if city_match:

        city = city_match.group(1)

        # Avoid treating these as cities
        ignored_words = {
            "under",
            "below",
            "above",
            "around",
            "near"
        }

        if city.lower() not in ignored_words:

            result["city"] = city.title()

    # ========================================================
    # PRICE
    # ========================================================

    # Handles:
    # under 15000
    # below 15000
    # under ₹15,000
    # below ₹15000

    max_price_match = re.search(
        r"(?:under|below|upto|up to)\s*[₹rs.\s]*([\d,]+)",
        query,
        re.IGNORECASE
    )

    if max_price_match:

        price_text = max_price_match.group(1)

        price_text = price_text.replace(",", "")

        result["max_price"] = int(price_text)

    # ========================================================
    # MINIMUM PRICE
    # ========================================================

    min_price_match = re.search(
        r"(?:above|over|from)\s*[₹rs.\s]*([\d,]+)",
        query,
        re.IGNORECASE
    )

    if min_price_match:

        price_text = min_price_match.group(1)

        price_text = price_text.replace(",", "")

        result["min_price"] = int(price_text)

    # ========================================================
    # ROOM TYPES
    # ========================================================

    upper_query = query.upper()

    for room_type in ROOM_TYPE_PATTERNS:

        if room_type.upper() in upper_query:

            result["room_types"].append(room_type)

    # Remove duplicates
    result["room_types"] = list(
        dict.fromkeys(result["room_types"])
    )

    # ========================================================
    # IDEAL FOR
    # ========================================================

    lower_query = query.lower()

    for pattern, normalized_value in IDEAL_FOR_PATTERNS.items():

        if pattern in lower_query:

            if normalized_value not in result["ideal_for"]:

                result["ideal_for"].append(
                    normalized_value
                )

    return result