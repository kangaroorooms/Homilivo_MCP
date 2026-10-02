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
    # Hindi/Hinglish spellings and common typos for the same cities,
    # so users mixing Hindi and English still resolve to the
    # canonical English city name used by the Homilivo API.
    "bengaluru": "Bangalore",
    "bengalore": "Bangalore",
    "banglore": "Bangalore",
    "bangalor": "Bangalore",
    "bangalre": "Bangalore",
    "bangaloru": "Bangalore",
    "dilli": "Delhi",
    "dilhi": "Delhi",
    "new dilli": "New Delhi",
    "bambai": "Mumbai",
    "mumba": "Mumbai",
    "mumbia": "Mumbai",
    "mumabi": "Mumbai",
    "chenai": "Chennai",
    "madras": "Chennai",
    "calcutta": "Kolkata",
    "kolkatta": "Kolkata",
    "hydrabad": "Hyderabad",
    "hydarabad": "Hyderabad",
    "punne": "Pune",
}

# Canonical (already-correct) city names the trailing-city fallback
# below may also match verbatim in an all-lowercase Hinglish query -
# mirrors the known-city set the "X mein" pattern above uses, kept as
# its own constant so that pattern's existing code is never touched by
# this addition.
KNOWN_CANONICAL_CITIES = {
    "bangalore", "mumbai", "delhi", "chennai", "kolkata",
    "hyderabad", "pune", "jaipur", "udaipur",
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


# Hindi/Hinglish number words needed for spoken-style budget and BHK
# phrases ("das hazaar", "do BHK"). Deliberately small - only the
# words actually seen in real user queries for this task's scope.
HINDI_NUMBER_WORDS = {
    "ek": 1,
    "do": 2,
    "teen": 3,
    "char": 4,
    "paanch": 5,
    "das": 10,
    "bees": 20,
}


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

    # Handles Hindi/Hinglish postposition phrasing, where the city
    # comes BEFORE the locator word instead of after a preposition:
    # "dilli mein", "bambai mein", "chenai mein dikhao". Tried first
    # since these queries typically have no "in/from/at" for the
    # English-preposition pattern below to anchor on.
    hinglish_city_match = re.search(
        r"\b([A-Za-z]+)\s+mein\b",
        query,
        re.IGNORECASE
    )

    if hinglish_city_match:

        candidate = hinglish_city_match.group(1).strip()

        # Only treat the word before "mein" as a city if it is a known
        # alias/typo or an already-canonical city name - otherwise an
        # unrelated word immediately before "mein" (e.g. "room mein")
        # would be misread as a city.
        known_canonical_cities = {
            "bangalore", "mumbai", "delhi", "chennai", "kolkata",
            "hyderabad", "pune", "jaipur", "udaipur",
        }

        if (
            candidate.lower() in CITY_ALIASES
            or candidate.lower() in known_canonical_cities
        ):

            result["city"] = _resolve_city_name(candidate)

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

    if city_match and result["city"] is None:

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

        # Hinglish queries are typically typed all-lowercase ("...room
        # chahiye bangalore"), so the capitalized-word heuristic above
        # never fires for them. Only trust a lowercase trailing word
        # when it is a KNOWN city/alias (never an arbitrary guess), so
        # this stays exactly as safe as the alias check the "X mein"
        # pattern above already relies on.
        if result["city"] is None:

            trailing_word = re.search(
                r"\b([A-Za-z]+)\s*\??\s*$",
                query,
            )

            if trailing_word:

                candidate = trailing_word.group(1)

                if (
                    candidate.lower() in CITY_ALIASES
                    or candidate.lower() in KNOWN_CANONICAL_CITIES
                ):

                    result["city"] = _resolve_city_name(candidate)

        # Position-aware check for the other end of the same lowercase
        # Hinglish phrasing: a city mentioned as the FIRST word instead
        # of trailing ("bengaluru jaana hai room dhundh do"). Same
        # safety rule as the trailing-word check above - only trusted
        # when it is a KNOWN city/alias, never an arbitrary guess.
        if result["city"] is None:

            leading_word = re.search(
                r"^\s*([A-Za-z]+)\b",
                query,
            )

            if leading_word:

                candidate = leading_word.group(1)

                if (
                    candidate.lower() in CITY_ALIASES
                    or candidate.lower() in KNOWN_CANONICAL_CITIES
                ):

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

    # --------------------------------------------------------
    # Mixed English/Hindi "under N hazaar" budget phrasing - must run
    # BEFORE the plain English under/below/upto pattern further down,
    # otherwise that pattern captures the bare digit run ("10" out of
    # "under 10 hazaar") and sets max_price before the hazaar
    # multiplier ever gets a chance to apply.
    # --------------------------------------------------------

    hazaar_max_match = re.search(
        r"(?:under|below|upto|up to)\s+(\d+)\s*hazaar\b",
        query,
        re.IGNORECASE
    )

    if hazaar_max_match and result["max_price"] is None:

        result["max_price"] = int(hazaar_max_match.group(1)) * 1000

    hazaar_min_match = re.search(
        r"(?:above|over)\s+(\d+)\s*hazaar\b",
        query,
        re.IGNORECASE
    )

    if hazaar_min_match and result["min_price"] is None:

        result["min_price"] = int(hazaar_min_match.group(1)) * 1000

    max_price_match = re.search(
        r"(?:under|below|upto|up to)\s*[₹rs.\s]*([\d,]+\s*[kK]?)",
        query,
        re.IGNORECASE
    )

    if max_price_match and result["max_price"] is None:

        result["max_price"] = _parse_price_token(
            max_price_match.group(1)
        )

    # "budget 10000", "budget is 8k" - a bare number after "budget" with
    # no under/below/upto keyword, common Hinglish phrasing.
    budget_match = re.search(
        r"budget\s+(?:is\s+|of\s+)?[₹rs.\s]*([\d,]+\s*[kK]?)",
        query,
        re.IGNORECASE
    )

    if budget_match and result["max_price"] is None:

        result["max_price"] = _parse_price_token(
            budget_match.group(1)
        )

    # --------------------------------------------------------
    # Hindi/Hinglish budget phrasing (additional patterns only -
    # none of the English patterns above are replaced)
    # --------------------------------------------------------

    # "15k se kam", "15000 se neeche" -> max_price, same meaning as
    # English "under X".
    hinglish_max_match = re.search(
        r"([\d,]+\s*[kK]?)\s*se\s*(?:kam|neeche)\b",
        query,
        re.IGNORECASE
    )

    if hinglish_max_match and result["max_price"] is None:

        result["max_price"] = _parse_price_token(
            hinglish_max_match.group(1)
        )

    # "10 hazaar", "20 hazaar" -> digit run * 1000
    digit_hazaar_match = re.search(
        r"\b(\d+)\s*hazaar\b",
        query,
        re.IGNORECASE
    )

    if digit_hazaar_match and result["max_price"] is None:

        result["max_price"] = int(digit_hazaar_match.group(1)) * 1000

    # "das hazaar", "paanch hazaar", "bees hazaar" -> Hindi number
    # word * 1000
    word_hazaar_match = re.search(
        r"\b(" + "|".join(HINDI_NUMBER_WORDS) + r")\s+hazaar\b",
        query,
        re.IGNORECASE
    )

    if word_hazaar_match and result["max_price"] is None:

        result["max_price"] = (
            HINDI_NUMBER_WORDS[word_hazaar_match.group(1).lower()] * 1000
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

    # --------------------------------------------------------
    # Hindi/Hinglish BHK phrasing (additional patterns only -
    # none of the existing digit-based patterns above are modified)
    # --------------------------------------------------------

    # "do BHK", "teen BHK", "char BHK", "do bedroom" -> 2BHK/3BHK/4BHK
    hinglish_bhk_match = re.search(
        r"\b(" + "|".join(HINDI_NUMBER_WORDS) + r")\s+(?:BHK|bedroom)\b",
        query,
        re.IGNORECASE
    )

    if hinglish_bhk_match:

        bhk_label = (
            f"{HINDI_NUMBER_WORDS[hinglish_bhk_match.group(1).lower()]}BHK"
        )

        if bhk_label not in room_types:

            room_types.append(bhk_label)

    # "ek room" -> Single
    if re.search(r"\bek\s+room\b", query, re.IGNORECASE):

        if "Single" not in room_types:

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