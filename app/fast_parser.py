import re


# ============================================================
# FAST USER REQUEST PARSER
# ============================================================

# Matches BHK/RK room types for any N and either spacing ("3BHK" or
# "3 bhk"), plus PG and Single as fixed, word-boundary-safe tokens.
_ROOM_TYPE_RE = re.compile(
    r"\b(?P<bhk>\d+)\s*BHK\b"
    r"|\b(?P<rk>\d+)\s*RK\b"
    r"|\b(?P<pg>PG)\b"
    r"|\b(?P<single>Single)\b",
    re.IGNORECASE
)


# Common real-world shorthand for a city/locality name, as users
# actually type it (not full official names). Deliberately small and
# explicit - anything not listed here just falls through to .title().
CITY_ALIASES = {
    "blr": "Bangalore",
    "blore": "Bangalore",
    "bombay": "Mumbai",
    "hyd": "Hyderabad",
    "hsr": "HSR Layout",
}

# Words that end a city/locality capture even though they are not
# capitalized in casual text - so "in HSR Layout for a family" stops
# at "for" instead of swallowing the rest of the sentence, while still
# allowing a locality name to span more than one word (e.g. "HSR
# Layout", "Electronic City").
CITY_STOP_WORDS = (
    "under", "below", "above", "around", "near", "upto", "for",
    "between", "with", "without", "and", "over",
)

_CITY_STOP_PATTERN = "|".join(CITY_STOP_WORDS)


def _resolve_city_name(candidate: str) -> str:
    """
    Resolve a captured city/locality phrase to its display form: a
    known alias expansion, or the phrase itself, title-cased.
    """

    alias = CITY_ALIASES.get(candidate.strip().lower())

    if alias is not None:

        return alias

    return candidate.strip().title()


def _parse_price_token(raw: str) -> int:
    """
    Parse a captured price token into an integer rupee amount.

    Handles plain numbers ("15000"), comma-grouped numbers ("15,000")
    and the "k" shorthand ("15k" -> 15000), for any digit run.
    """

    cleaned = raw.strip().replace(",", "").replace(" ", "")

    if cleaned and cleaned[-1] in ("k", "K"):

        return int(float(cleaned[:-1]) * 1000)

    return int(cleaned)


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
        rf"\b(?:in|from|at)\s+([A-Za-z][A-Za-z\s]*?)"
        rf"(?=\s+(?:{_CITY_STOP_PATTERN})\b|[?.!,]|$)",
        query,
        re.IGNORECASE
    )

    if city_match:

        city = city_match.group(1).strip()

        # Avoid treating these as cities
        ignored_words = {
            "under",
            "below",
            "above",
            "around",
            "near"
        }

        if city.lower() not in ignored_words:

            result["city"] = _resolve_city_name(city)

    # Fallback for phrasing that omits a preposition entirely
    # ("any rooms Bangalore?", "room in Bangalore" already matched
    # above, but "rooms Bangalore" would not). Only attempted when the
    # message clearly is a room/accommodation request (so a bare
    # capitalized word in an unrelated sentence is never mistaken for
    # a city) AND the candidate is capitalized in the ORIGINAL query
    # (a reasonable, bounded proxy for "this looks like a proper noun
    # / place name", not a guess at arbitrary Homilivo city data).
    if result["city"] is None and re.search(
        r"\b(room|rooms|accommodation|flat|flats|pg|apartment|apartments)\b",
        query,
        re.IGNORECASE,
    ):

        trailing_capitalized = re.search(
            r"\b([A-Z][a-zA-Z]+)\s*\??\s*$",
            query,
        )

        if trailing_capitalized:

            candidate = trailing_capitalized.group(1)

            if candidate.lower() not in {
                "room", "rooms", "accommodation", "flat", "flats",
                "pg", "apartment", "apartments",
            }:

                result["city"] = _resolve_city_name(candidate)

    # ========================================================
    # PRICE
    # ========================================================

    # Handles:
    # under 15000
    # below 15000
    # under ₹15,000
    # below ₹15000

    # Handles:
    # between 15000 and 25000
    # between ₹15,000 and ₹25,000
    # between 15k and 25k

    range_match = re.search(
        r"between\s*[₹rs.\s]*([\d,]+\s*[kK]?)\s*"
        r"(?:and|to)\s*[₹rs.\s]*([\d,]+\s*[kK]?)",
        query,
        re.IGNORECASE
    )

    if range_match:

        result["min_price"] = _parse_price_token(
            range_match.group(1)
        )

        result["max_price"] = _parse_price_token(
            range_match.group(2)
        )

    max_price_match = re.search(
        r"(?:under|below|upto|up to)\s*[₹rs.\s]*([\d,]+\s*[kK]?)",
        query,
        re.IGNORECASE
    )

    if max_price_match and result["max_price"] is None:

        result["max_price"] = _parse_price_token(
            max_price_match.group(1)
        )

    # ========================================================
    # MINIMUM PRICE
    # ========================================================

    min_price_match = re.search(
        r"(?:above|over|from)\s*[₹rs.\s]*([\d,]+\s*[kK]?)",
        query,
        re.IGNORECASE
    )

    if min_price_match and result["min_price"] is None:

        result["min_price"] = _parse_price_token(
            min_price_match.group(1)
        )

    # ========================================================
    # ROOM TYPES
    # ========================================================

    room_types = []

    for match in _ROOM_TYPE_RE.finditer(query):

        if match.group("bhk"):

            room_types.append(f"{match.group('bhk')}BHK")

        elif match.group("rk"):

            room_types.append(f"{match.group('rk')}RK")

        elif match.group("pg"):

            room_types.append("PG")

        elif match.group("single"):

            room_types.append("Single")

    # Remove duplicates
    result["room_types"] = list(
        dict.fromkeys(room_types)
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