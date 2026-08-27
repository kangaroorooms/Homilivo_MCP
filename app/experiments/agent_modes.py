"""
Experimental conversational Homilivo chatbot.

IMPORTANT:
- This file modifies ONLY the experimental pipeline.
- app/agent.py is not modified.
- Existing guardrails are preserved.
- MCP is preserved.
- FAST, QWEN3, GEMMA3 and GROQ are preserved.
- All modes share the same orchestration pipeline.
- LLM intent classification is used only as a fallback.
- Deterministic/contextual routing is preferred for follow-ups.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

# Windows stdout defaults to cp1252 whenever it isn't an interactive
# UTF-8 console (piped, redirected, run under a process manager), which
# crashes on the Unicode symbols (arrows, checkmarks) this module prints.
# Harmless no-op when a caller (e.g. app/api.py) already reconfigured it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Load GROQ_API_KEY (and any other secrets) from a .env file if present.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from ollama import chat
from mcp import Client

from app.fast_parser import extract_search_parameters

# Semantic operation classifier (embedding fallback for the long tail of
# room-follow-up phrasings). Returns the SAME operation constants that
# detect_room_operation() already uses, or None when not confident.
from app.experiments.interpreter import classify_operation

from app.guardrails import (
    validate_user_input,
    validate_search_arguments,
    validate_room_data,
    validate_ai_response,
)

from app.llm.gemma import (
    classify_intent,
    GemmaError,
)

from app.server import mcp


# ============================================================
# CONSTANTS
# ============================================================

MODE_FAST = "fast"
MODE_QWEN = "qwen"
MODE_GEMMA = "gemma"
MODE_GROQ = "groq"

# Groq model used for the GROQ mode (fast, free-tier, hosted).
GROQ_MODEL = "openai/gpt-oss-20b"


MESSAGE_GREETING = "GREETING"
MESSAGE_INFO = "CHATBOT_INFO"
MESSAGE_NEW_SEARCH = "NEW_SEARCH"
MESSAGE_REFINEMENT = "SEARCH_REFINEMENT"
MESSAGE_FOLLOWUP = "ROOM_FOLLOWUP"
MESSAGE_CLARIFICATION = "CLARIFICATION_RESPONSE"
MESSAGE_UNKNOWN = "UNKNOWN"
MESSAGE_EXIT = "EXIT"


EXIT_COMMANDS = {
    "bye",
    "goodbye",
    "exit",
    "quit",
    "stop",
    "ok",
    "okay",
    "thanks",
    "thank you",
}


GREETING_RESPONSE = (
    "Hello! I'm the Homilivo room-search assistant. "
    "How can I help you find a room today?"
)


CHATBOT_INFO_RESPONSE = (
    "I'm the Homilivo room-search assistant. "
    "I can help you search for rooms and accommodation "
    "based on location, budget, room type and preferences."
)


UNKNOWN_RESPONSE = (
    "I can help you search for rooms and accommodation. "
    "Please tell me what kind of place you're looking for."
)


# ============================================================
# CONVERSATION STATE
# ============================================================

@dataclass
class ConversationState:
    """
    Structured conversational memory.

    Conversation history is useful for language context,
    but room identity and search state are stored separately.
    """

    conversation_history: list[dict[str, Any]] = field(
        default_factory=list
    )

    previous_rooms: list[dict[str, Any]] = field(
        default_factory=list
    )

    previous_search: dict[str, Any] = field(
        default_factory=dict
    )

    selected_room: dict[str, Any] | None = None

    selected_room_id: str | None = None

    pending_search: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class MessageRoute:
    """
    Result of the lightweight conversational router.

    This object describes WHAT the user is asking.
    It does not execute MCP or generate the final answer.
    """

    message_type: str

    confidence: float = 1.0

    reason: str = ""

    operation: str | None = None


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header(
    title: str,
) -> None:

    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(
    message: str,
) -> None:

    print(f"\n→ {message}")


def print_success(
    message: str,
) -> None:

    print(f"✓ {message}")


def print_failure(
    message: str,
) -> None:

    print(f"✗ {message}")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text: str,
) -> str:

    text = text.strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def normalized_without_punctuation(
    text: str,
) -> str:

    text = normalize_text(text)

    return text.rstrip(
        " .!?,'\""
    )


# ============================================================
# EXIT
# ============================================================

def should_end_conversation(
    user_query: str,
) -> bool:

    normalized = normalized_without_punctuation(
        user_query
    )

    return normalized in EXIT_COMMANDS


# ============================================================
# BASIC ROOM HELPERS
# ============================================================

def get_room_id(
    room: dict[str, Any],
) -> str | None:

    for key in (
        "id",
        "_id",
        "roomId",
        "room_id",
    ):

        value = room.get(key)

        if value is not None:

            value = str(value).strip()

            if value:

                return value

    return None


def get_room_price(
    room: dict[str, Any],
) -> float | None:

    value = room.get(
        "pricePerMonth"
    )

    if value is None:

        value = room.get(
            "price"
        )

    if value is None:
        return None

    try:

        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return None


def get_room_title(
    room: dict[str, Any],
) -> str:

    return str(
        room.get(
            "title",
            room.get(
                "name",
                "Untitled Room",
            ),
        )
    )


def get_room_type(
    room: dict[str, Any],
) -> Any:

    for key in (
        "roomType",
        "room_type",
        "type",
    ):

        value = room.get(key)

        if value is not None:

            return value

    return None


def get_room_location(
    room: dict[str, Any],
) -> str | None:

    values: list[str] = []

    for key in (
        "location",
        "area",
        "locality",
        "city",
    ):

        value = room.get(key)

        if value is None:
            continue

        if isinstance(
            value,
            str,
        ):

            if value.strip():

                values.append(
                    value.strip()
                )

        elif isinstance(
            value,
            dict,
        ):

            for nested_key in (
                "name",
                "city",
                "area",
                "locality",
            ):

                nested = value.get(
                    nested_key
                )

                if nested:

                    values.append(
                        str(nested)
                    )

        else:

            values.append(
                str(value)
            )

    unique: list[str] = []

    for value in values:

        if value not in unique:

            unique.append(value)

    if not unique:

        return None

    return ", ".join(unique)


def get_room_ideal_for(
    room: dict[str, Any],
) -> list[Any]:

    value = room.get(
        "idealFor",
        room.get(
            "ideal_for",
            [],
        ),
    )

    if value is None:
        return []

    if isinstance(
        value,
        list,
    ):

        return value

    return [value]


# ============================================================
# GENERIC TOKEN / CONCEPT HELPERS
# ============================================================

def contains_any(
    text: str,
    phrases: tuple[str, ...] | list[str],
) -> bool:

    return any(
        phrase in text
        for phrase in phrases
    )


def has_price_language(
    text: str,
) -> bool:

    return contains_any(
        text,
        (
            "price",
            "rent",
            "cost",
            "costly",
            "expensive",
            "cheap",
            "cheapest",
            "affordable",
            "monthly payment",
            "monthly rent",
            "pay every month",
            "pay monthly",
            "how much",
        ),
    )


def has_room_reference_language(
    text: str,
) -> bool:

    return contains_any(
        text,
        (
            "it",
            "its",
            "this",
            "that",
            "one",
            "room",
            "property",
            "option",
            "place",
        ),
    )


def has_suitability_language(
    text: str,
) -> bool:

    return contains_any(
        text,
        (
            "suitable",
            "suitability",
            "allowed",
            "allow",
            "can ",
            "would ",
            "good for",
            "ideal for",
            "appropriate",
            "work for",
            "okay for",
            "ok for",
            "stay there",
            "live there",
        ),
    )


# ============================================================
# ROOM OPERATION DETECTION
# ============================================================

# High-precision deterministic cores. These stay SHORT on purpose: they cover
# the obvious/common phrasings (and everything in the golden conversation).
# The long tail of natural-language variations is handled by the embedding
# fallback (classify_operation) at the end of detect_room_operation(), so these
# tuples never need to grow again.

_LOWEST_CORE = (
    "cheapest",
    "least expensive",
    "lowest price",
    "lowest rent",
    "lowest priced",
    "most affordable",
    "least costly",
)

_HIGHEST_CORE = (
    "most expensive",
    "highest price",
    "highest rent",
    "highest priced",
    "priciest",
    "costliest",
    "most costly",
    "costs the most",
    "cost the most",
)

_TYPE_CORE = (
    "room type",
    "what type",
    "what kind of room",
    "which type",
    "bhk type",
    "how many bhk",
    "what bhk",
)

_LOCATION_CORE = (
    "where is",
    "where's",
    "location",
    "located",
    "which city",
    "what city",
    "which area",
    "what area",
    "where can i find",
)

_AMENITY_CORE = (
    "parking",
    "furnished",
    "furniture",
    "refrigerator",
    "fridge",
    "washroom",
    "bathroom",
    "wifi",
    "wi-fi",
    "air conditioning",
    "amenities",
    "amenity",
    "facilities",
    "facility",
    "bed",
)

_DETAILS_CORE = (
    "tell me about",
    "more about",
    "more information",
    "details about",
    "information about",
    "describe",
    "what about this",
    "what about that",
)

_LAST_CORE = (
    "last one",
    "last option",
    "last room",
    "last property",
)

# Ordinal word/numeral -> 1-indexed rank. A fixed, small map plus one numeral
# regex covers "second"/"2nd"/"3rd" etc. without a branch per phrasing, and
# extends to any N (not just first/second/third) for free.
_ORDINAL_RANKS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_ORDINAL_WORD_PATTERN = "|".join(_ORDINAL_RANKS.keys())
_NUMERIC_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")

# "second cheapest", "third most expensive", "2nd cheapest": the ordinal must
# sit directly in front of the superlative phrase. This is what distinguishes
# a RANKED price query from unrelated co-occurrences like "cheapest of the
# first two" (superlative before the ordinal, and modifying a different
# noun), which must keep resolving as plain LOWEST_PRICE / ORDINAL_1.
_ORDINAL_PREFIX_PATTERN = rf"(?:{_ORDINAL_WORD_PATTERN}|\d+(?:st|nd|rd|th))"
_RANKED_LOWEST_RE = re.compile(
    _ORDINAL_PREFIX_PATTERN
    + r"\s+(?:"
    + "|".join(re.escape(phrase) for phrase in _LOWEST_CORE)
    + r")"
)
_RANKED_HIGHEST_RE = re.compile(
    _ORDINAL_PREFIX_PATTERN
    + r"\s+(?:"
    + "|".join(re.escape(phrase) for phrase in _HIGHEST_CORE)
    + r")"
)


def _extract_rank(text: str) -> int | None:
    """
    Extract a 1-indexed rank from an ordinal word or numeral appearing in
    (already-normalized, lowercase) text. Returns None if none is present.
    """

    match = _NUMERIC_ORDINAL_RE.search(text)

    if match:
        return int(match.group(1))

    for word, rank in _ORDINAL_RANKS.items():
        if re.search(rf"\b{word}\b", text):
            return rank

    return None


def _ordinal_label(rank: int) -> str:
    for word, value in _ORDINAL_RANKS.items():
        if value == rank:
            return word

    return f"{rank}th"


def detect_room_operation(
    user_query: str,
) -> str | None:
    """
    Determine the semantic operation for a question about an already verified
    room or the current set of verified rooms.

    Two tiers:

    TIER A - deterministic core
        Ordinals plus a short, high-precision set of keyword cases. Fast, no
        model cost, and it covers the common path (including every turn in the
        golden conversation). Superlative/comparison intent is resolved BEFORE
        the generic price case, so "most affordable" does not become ROOM_PRICE.

    TIER B - semantic fallback
        When the deterministic core is silent, classify_operation() maps the
        wording onto the SAME operation constants using embedding similarity.
        This handles arbitrary phrasing without adding keywords. It returns None
        when not confident, in which case behaviour is exactly as before.

    This function determines WHAT the user is asking. It does not decide WHICH
    room is being referenced; that is handled by resolve_room_reference().

    Return values are unchanged, so resolve_room_reference(), answer_room_fact(),
    answer_room_comparison() and is_likely_followup() keep working as-is.
    """

    text = normalize_text(user_query)

    # --------------------------------------------------------
    # TIER A - DETERMINISTIC CORE
    # --------------------------------------------------------

    # Ranked price ("second cheapest", "third most expensive", "2nd
    # cheapest"). Must be checked BEFORE the plain ordinal checks below,
    # otherwise "second cheapest" would match on "second" alone and pick
    # the 2nd room in API order instead of the 2nd-lowest-priced room.
    # The adjacency requirement (ordinal directly before the superlative)
    # is what keeps "cheapest of the first two" resolving as ORDINAL_1,
    # since there the superlative comes first and modifies "two", not a
    # rank-of-cheapness.
    if _RANKED_LOWEST_RE.search(text):
        return "RANKED_LOWEST_PRICE"

    if _RANKED_HIGHEST_RE.search(text):
        return "RANKED_HIGHEST_PRICE"

    # "last one" / "last option" - the last room in the current room
    # context, in the same natural (API-return) order the plain ordinals
    # below use for "first"/"second"/"third".
    if contains_any(text, _LAST_CORE):
        return "LAST_ROOM"

    # Ordinals
    if re.search(r"\b(first|1st)\b", text):
        return "ORDINAL_1"

    if re.search(r"\b(second|2nd)\b", text):
        return "ORDINAL_2"

    if re.search(r"\b(third|3rd)\b", text):
        return "ORDINAL_3"

    # Lowest-price superlative (must come before the generic price case)
    if contains_any(text, _LOWEST_CORE):
        return "LOWEST_PRICE"

    # Highest-price superlative. "Is A more expensive than B?" is a pairwise
    # comparison, not a request for the priciest room in the set; leave such
    # ambiguous questions to the fallback path.
    if contains_any(text, _HIGHEST_CORE) and " than " not in f" {text} ":
        return "HIGHEST_PRICE"

    # Price of the referenced/selected room
    if has_price_language(text):
        return "ROOM_PRICE"

    # Room type
    if contains_any(text, _TYPE_CORE):
        return "ROOM_TYPE"

    # Location
    if contains_any(text, _LOCATION_CORE):
        return "LOCATION"

    # Student suitability
    if contains_any(
        text,
        (
            "student",
            "students",
            "college student",
            "college",
            "university student",
            "university",
        ),
    ) and has_suitability_language(text):
        return "STUDENT_SUITABILITY"

    # Family suitability
    if contains_any(
        text,
        (
            "family",
            "families",
        ),
    ) and has_suitability_language(text):
        return "FAMILY_SUITABILITY"

    # Working-professional suitability
    if contains_any(
        text,
        (
            "working professional",
            "working professionals",
            "professional",
            "professionals",
        ),
    ) and has_suitability_language(text):
        return "PROFESSIONAL_SUITABILITY"

    # Amenities / facilities
    if contains_any(text, _AMENITY_CORE):
        return "AMENITY"

    # General room information
    if contains_any(text, _DETAILS_CORE):
        return "ROOM_DETAILS"

    # --------------------------------------------------------
    # TIER B - SEMANTIC FALLBACK (long tail)
    # --------------------------------------------------------
    # Reached only when the deterministic core matched nothing. Returns one of
    # the operation constants above, or None (behaviour identical to before).

    return classify_operation(text)


# ============================================================
# ROOM REFERENCE RESOLUTION
# ============================================================


def resolve_room_reference(
    user_query: str,
    previous_rooms: list[dict[str, Any]],
    selected_room: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Deterministically resolve which verified room the user
    means.

    Priority:

    1. cheapest / most expensive
    2. ordinal
    3. explicit selected/this/that reference
    4. pronoun reference
    5. existing selected room for contextual questions
    """

    if not previous_rooms and selected_room is None:

        return None

    text = normalize_text(
        user_query
    )

    operation = detect_room_operation(
        user_query
    )

    # --------------------------------------------------------
    # LOWEST PRICE
    # --------------------------------------------------------

    if operation == "LOWEST_PRICE":

        priced_rooms = [
            (
                get_room_price(room),
                room,
            )
            for room in previous_rooms
            if get_room_price(room) is not None
        ]

        if priced_rooms:

            return min(
                priced_rooms,
                key=lambda item: item[0],
            )[1]

    # --------------------------------------------------------
    # HIGHEST PRICE
    # --------------------------------------------------------

    if operation == "HIGHEST_PRICE":

        priced_rooms = [
            (
                get_room_price(room),
                room,
            )
            for room in previous_rooms
            if get_room_price(room) is not None
        ]

        if priced_rooms:

            return max(
                priced_rooms,
                key=lambda item: item[0],
            )[1]

    # --------------------------------------------------------
    # RANKED PRICE ("second cheapest", "third most expensive")
    # --------------------------------------------------------

    if operation in ("RANKED_LOWEST_PRICE", "RANKED_HIGHEST_PRICE"):

        rank = _extract_rank(text)

        if rank is not None:

            priced_rooms = [
                (get_room_price(room), room)
                for room in previous_rooms
                if get_room_price(room) is not None
            ]

            priced_rooms.sort(
                key=lambda item: item[0],
                reverse=(operation == "RANKED_HIGHEST_PRICE"),
            )

            index = rank - 1

            if 0 <= index < len(priced_rooms):

                return priced_rooms[index][1]

    # --------------------------------------------------------
    # LAST ROOM
    # --------------------------------------------------------

    if operation == "LAST_ROOM":

        if previous_rooms:

            return previous_rooms[-1]

    # --------------------------------------------------------
    # ORDINAL
    # --------------------------------------------------------

    ordinal_map = {
        "ORDINAL_1": 0,
        "ORDINAL_2": 1,
        "ORDINAL_3": 2,
    }

    if operation in ordinal_map:

        index = ordinal_map[
            operation
        ]

        if index < len(
            previous_rooms
        ):

            return previous_rooms[
                index
            ]

    # --------------------------------------------------------
    # EXPLICIT REFERENCES
    # --------------------------------------------------------

    explicit_reference = contains_any(
        text,
        (
            "this one",
            "that one",
            "this room",
            "that room",
            "this property",
            "that property",
            "this option",
            "that option",
            "selected room",
            "selected one",
            "selected property",
        ),
    )

    if explicit_reference:

        if selected_room is not None:

            return selected_room

    # --------------------------------------------------------
    # PRONOUN REFERENCES
    # --------------------------------------------------------

    pronoun_reference = contains_any(
        text,
        (
            "its ",
            "it ",
            "it?",
            "is it",
            "does it",
            "would it",
            "can it",
            "could it",
            "where is it",
            "what is it",
            "how much is it",
            "about it",
            "there",
        ),
    )

    if pronoun_reference:

        if selected_room is not None:

            return selected_room

    # --------------------------------------------------------
    # CONTEXTUAL ROOM QUESTION
    # --------------------------------------------------------

    operation_exists = (
        operation is not None
    )

    if (
        selected_room is not None
        and operation_exists
    ):

        return selected_room

    # --------------------------------------------------------
    # GENERIC FOLLOW-UP
    # --------------------------------------------------------

    if selected_room is not None:

        if contains_any(
            text,
            (
                "which one",
                "which is",
                "what about",
                "tell me more",
                "more details",
                "more information",
            ),
        ):

            return selected_room

    return None


# ============================================================
# ROOM FACT ANSWERING
# ============================================================

def answer_room_fact(
    user_query: str,
    selected_room: dict[str, Any] | None,
) -> str | None:
    """
    Answer structured room questions directly from verified
    room data.

    Returns None when the question requires natural-language
    generation.
    """

    if selected_room is None:

        return None

    operation = detect_room_operation(
        user_query
    )

    # --------------------------------------------------------
    # ROOM PRICE
    # --------------------------------------------------------

    if operation == "ROOM_PRICE":

        price = get_room_price(
            selected_room
        )

        if price is None:

            return (
                "The available room data does not "
                "specify the price."
            )

        return (
            f"The monthly rent is "
            f"₹{price:,.0f}."
        )

    # --------------------------------------------------------
    # ROOM TYPE
    # --------------------------------------------------------

    if operation == "ROOM_TYPE":

        room_type = get_room_type(
            selected_room
        )

        if room_type is None:

            return (
                "The available room data does not "
                "specify the room type."
            )

        return (
            f"The room type is **{room_type}**."
        )

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    if operation == "LOCATION":

        location = get_room_location(
            selected_room
        )

        if location is None:

            return (
                "The available room data does not "
                "specify the room location."
            )

        return (
            f"It is located in **{location}**."
        )

    # --------------------------------------------------------
    # STUDENT SUITABILITY
    # --------------------------------------------------------

    if operation == "STUDENT_SUITABILITY":

        ideal_for = get_room_ideal_for(
            selected_room
        )

        values = [
            str(value).lower()
            for value in ideal_for
        ]

        if any(
            "student" in value
            for value in values
        ):

            return (
                "Yes. The verified room data shows "
                "that this room is suitable for students."
            )

        if ideal_for:

            return (
                "The verified room data does not list "
                "students as a target group for this room."
            )

        return (
            "The available room data does not specify "
            "whether this room is suitable for students."
        )

    # --------------------------------------------------------
    # FAMILY SUITABILITY
    # --------------------------------------------------------

    if operation == "FAMILY_SUITABILITY":

        ideal_for = get_room_ideal_for(
            selected_room
        )

        values = [
            str(value).lower()
            for value in ideal_for
        ]

        if any(
            "famil" in value
            for value in values
        ):

            return (
                "Yes. The verified room data shows "
                "that this room is suitable for families."
            )

        if ideal_for:

            return (
                "The verified room data does not list "
                "families as a target group for this room."
            )

        return (
            "The available room data does not specify "
            "whether this room is suitable for families."
        )

    # --------------------------------------------------------
    # PROFESSIONAL SUITABILITY
    # --------------------------------------------------------

    if operation == "PROFESSIONAL_SUITABILITY":

        ideal_for = get_room_ideal_for(
            selected_room
        )

        values = [
            str(value).lower()
            for value in ideal_for
        ]

        if any(
            "professional" in value
            for value in values
        ):

            return (
                "Yes. The verified room data shows "
                "that this room is suitable for working "
                "professionals."
            )

        if ideal_for:

            return (
                "The verified room data does not list "
                "working professionals as a target group "
                "for this room."
            )

        return (
            "The available room data does not specify "
            "whether this room is suitable for working "
            "professionals."
        )

    # --------------------------------------------------------
    # AMENITY
    # --------------------------------------------------------

    if operation == "AMENITY":

        requested = _extract_amenity_term(user_query)

        display_term = requested or "requested amenity"

        if requested is not None and _room_has_amenity(
            selected_room,
            requested,
        ):

            # Try to retrieve a useful explicit value.
            for field in (
                "amenities",
                "amenity",
                "facilities",
                "facility",
                "features",
            ):

                value = selected_room.get(
                    field
                )

                if value is not None:

                    return (
                        f"The verified room data includes "
                        f"the following {field}: "
                        f"{value}."
                    )

            return (
                f"The verified room data contains information "
                f"about {display_term}."
            )

        return (
            "The available room data does not specify "
            f"whether {display_term} is provided."
        )

    return None


# ============================================================
# AMENITY TERM HELPERS
# ============================================================
# Shared by answer_room_fact() (single selected room) and
# answer_amenity_across_rooms() (the whole current room set) below, so
# there is exactly one place that maps a question onto an amenity term
# and exactly one place that checks a room for it.

AMENITY_TERMS = (
    "parking",
    "furnished",
    "furniture",
    "refrigerator",
    "fridge",
    "washroom",
    "bathroom",
    "bed",
    "wifi",
    "wi-fi",
    "ac",
)


def _extract_amenity_term(
    user_query: str,
) -> str | None:

    text = normalize_text(
        user_query
    )

    for term in AMENITY_TERMS:

        if term in text:

            return term

    return None


def _room_has_amenity(
    room: dict[str, Any],
    term: str,
) -> bool:

    room_json = json.dumps(
        room,
        ensure_ascii=False,
    ).lower()

    return term in room_json


# ============================================================
# SET-WIDE AMENITY ANSWER
# ============================================================

def answer_amenity_across_rooms(
    user_query: str,
    rooms: list[dict[str, Any]],
) -> str | None:
    """
    Answer a SET-WIDE amenity question ("does any room have ac?") from
    verified session room data when no single room has been selected or
    resolved. answer_room_fact() only ever answers about one
    selected_room; this is the collective counterpart for the same
    AMENITY operation, so a question about the whole current result set
    is not silently dropped to the generic capability message.

    Returns None when this isn't an AMENITY question or there are no
    rooms to check, so callers fall through unchanged.
    """

    if detect_room_operation(user_query) != "AMENITY":

        return None

    if not rooms:

        return None

    requested = _extract_amenity_term(
        user_query
    )

    if requested is None:

        return (
            "I can check amenities such as parking, furnishing, a "
            "fridge, wifi or AC across the current rooms - could you "
            "specify which one you mean?"
        )

    matches = [
        room
        for room in rooms
        if _room_has_amenity(room, requested)
    ]

    if not matches:

        return (
            f"The available room data does not specify {requested} "
            f"for any of the current rooms."
        )

    if len(matches) == len(rooms):

        return (
            f"Yes, all {len(rooms)} current rooms include "
            f"{requested}."
        )

    titles = ", ".join(
        get_room_title(room)
        for room in matches
    )

    return (
        f"Yes, {len(matches)} of {len(rooms)} current rooms include "
        f"{requested}: {titles}."
    )


# ============================================================
# ROOM COMPARISON RESPONSE
# ============================================================

def answer_room_comparison(
    user_query: str,
    rooms: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    """
    Perform deterministic comparisons.

    Returns:
        (response, selected_room)

    The LLM is never asked to decide which room wins.
    """

    operation = detect_room_operation(
        user_query
    )

    if operation not in {
        "LOWEST_PRICE",
        "HIGHEST_PRICE",
        "RANKED_LOWEST_PRICE",
        "RANKED_HIGHEST_PRICE",
    }:

        return None, None

    priced_rooms = []

    for room in rooms:

        price = get_room_price(
            room
        )

        if price is not None:

            priced_rooms.append(
                (
                    price,
                    room,
                )
            )

    if not priced_rooms:

        return (
            "The previous room results do not contain "
            "valid price information.",
            None,
        )

    if operation == "LOWEST_PRICE":

        price, room = min(
            priced_rooms,
            key=lambda item: item[0],
        )

        return (
            f"The cheapest option is "
            f"**{get_room_title(room)}**, "
            f"at ₹{price:,.0f}/month.",
            room,
        )

    if operation == "HIGHEST_PRICE":

        price, room = max(
            priced_rooms,
            key=lambda item: item[0],
        )

        return (
            f"The most expensive option is "
            f"**{get_room_title(room)}**, "
            f"at ₹{price:,.0f}/month.",
            room,
        )

    # --------------------------------------------------------
    # RANKED PRICE ("second cheapest", "third most expensive")
    # --------------------------------------------------------

    rank = _extract_rank(
        normalize_text(user_query)
    )

    if rank is None:

        return None, None

    priced_rooms.sort(
        key=lambda item: item[0],
        reverse=(operation == "RANKED_HIGHEST_PRICE"),
    )

    label = (
        "cheapest"
        if operation == "RANKED_LOWEST_PRICE"
        else "most expensive"
    )

    index = rank - 1

    if index >= len(priced_rooms):

        return (
            f"There "
            f"{'is' if len(priced_rooms) == 1 else 'are'} only "
            f"{len(priced_rooms)} room(s) with price information, "
            f"so there is no {_ordinal_label(rank)} {label} option.",
            None,
        )

    price, room = priced_rooms[index]

    return (
        f"The {_ordinal_label(rank)} {label} option is "
        f"**{get_room_title(room)}**, "
        f"at ₹{price:,.0f}/month.",
        room,
    )


# ============================================================
# CONTEXTUAL FOLLOW-UP DETECTION
# ============================================================

def is_likely_followup(
    user_query: str,
    previous_rooms: list[dict[str, Any]],
    selected_room: dict[str, Any] | None,
) -> bool:
    """
    Detect a room-context question without calling an LLM.

    This is intentionally broad enough to handle natural
    phrasing while requiring existing room context.
    """

    if not previous_rooms and selected_room is None:

        return False

    text = normalize_text(
        user_query
    )

    # A clear search CONSTRAINT (BHK / price band / audience qualifier) with no
    # room reference is a refinement, not a follow-up. This stops the semantic
    # fallback from weakly matching an operation and hijacking the refinement
    # route (e.g. "only 2bhk", "for students").
    if not has_room_reference_language(text):
        if (
            re.search(r"\b\d+\s*bhk\b", text)
            or re.search(r"₹?\s*\d{3,}", text)
            or contains_any(
                text,
                (
                    "only ", "just ", "under ", "below ", "less than ",
                    "above ", "over ", "more than ", "between ",
                    "for students", "for student", "for families",
                    "for family", "for professionals", "furnished",
                    "unfurnished", " pg", "hostel",
                ),
            )
        ):
            return False

    operation = detect_room_operation(
        user_query
    )

    if operation is not None:

        return True

    if has_room_reference_language(
        text
    ):

        contextual_phrases = (
            "tell me",
            "what",
            "where",
            "how much",
            "would",
            "can",
            "could",
            "does",
            "is",
            "which",
            "about",
        )

        if contains_any(
            text,
            contextual_phrases,
        ):

            return True

    return False

# ============================================================
# SEARCH REFINEMENT DETECTION
# ============================================================

def is_likely_search_refinement(
    user_query: str,
    previous_search: dict[str, Any],
    previous_rooms: list[dict[str, Any]],
) -> bool:

    if not previous_search:

        return False

    text = normalize_text(
        user_query
    )

    if is_likely_followup(
        user_query,
        previous_rooms,
        None,
    ):

        return False

    refinement_patterns = (
        "only ",
        "just ",
        "under ",
        "below ",
        "less than ",
        "above ",
        "over ",
        "more than ",
        "between ",
        "for students",
        "for student",
        "for families",
        "for family",
        "for professionals",
        "working professional",
        "furnished",
        "unfurnished",
        "1bhk",
        "2bhk",
        "3bhk",
        "4bhk",
        "pg",
        "hostel",
        "near ",
        # NOTE: bare "with "/"without " were removed here on purpose.
        # fast_parser.py has no amenity-filter extraction, so they never
        # produced a real parameter change - they only ever matched
        # ordinary English ("how to contact WITH the owner") and, because
        # refinement routes bypass the input guardrail's rejection, that
        # false match silently re-ran the previous search for clearly
        # out-of-scope messages. See tests/test_operations.py for the
        # regression test.
        "remove ",
        "exclude ",
        "change the budget",
        "increase the budget",
        "decrease the budget",
    )

    if contains_any(
        text,
        refinement_patterns,
    ):

        return True

    # A message that contains clear search constraints
    # but no explicit new-search verb is also a refinement.
    if (
        re.search(
            r"₹\s*\d+",
            text,
        )
        and previous_search
    ):

        return True

    if (
        re.search(
            r"\b\d+\s*bhk\b",
            text,
        )
        and previous_search
    ):

        return True

    return False


# ============================================================
# CLARIFICATION RESPONSE DETECTION
# ============================================================

def is_clarification_response(
    user_query: str,
    pending_search: dict[str, Any],
) -> bool:

    if not pending_search:

        return False

    text = normalize_text(
        user_query
    )

    if not text:

        return False

    # Currently the main clarification requested by the
    # pipeline is city.
    if pending_search.get(
        "city"
    ) is None:

        # "Jaipur"
        # "Chennai"
        # "Bangalore"
        if len(
            text.split()
        ) <= 4:

            return True

        if contains_any(
            text,
            (
                "in ",
                "at ",
                "near ",
                "around ",
                "looking in ",
                "search in ",
            ),
        ):

            return True

    return False


# ============================================================
# FAST HEURISTIC INTENT DETECTION
# ============================================================

def detect_simple_intent(
    user_query: str,
) -> str | None:

    text = normalize_text(
        user_query
    )

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    greeting_patterns = (
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "how are you",
    )

    if text in greeting_patterns:

        return MESSAGE_GREETING

    # --------------------------------------------------------
    # CHATBOT INFO
    # --------------------------------------------------------

    info_patterns = (
        "what does this chatbot do",
        "what can you help me with",
        "what can you help with",
        "what can i search for",
        "what can this chatbot do",
        "how does homilivo work",
        "what is homilivo",
        "what do you do",
        "what are you",
    )

    if contains_any(
        text,
        info_patterns,
    ):

        return MESSAGE_INFO

    # --------------------------------------------------------
    # OBVIOUS ROOM SEARCH
    # --------------------------------------------------------

    search_patterns = (
        "find room",
        "find rooms",
        "search room",
        "search rooms",
        "looking for a room",
        "looking for rooms",
        "looking for accommodation",
        "need accommodation",
        "need a room",
        "need somewhere to stay",
        "place to stay",
        "place to live",
        "looking for a place",
        "find me a place",
        "find me accommodation",
        "show me rooms",
        "show rooms",
    )

    if contains_any(
        text,
        search_patterns,
    ):

        return MESSAGE_NEW_SEARCH

    return None


# ============================================================
# CONTEXTUAL ROUTER
# ============================================================

def route_message(
    user_query: str,
    state: ConversationState,
) -> MessageRoute:
    """
    Main conversational router.

    IMPORTANT:

    This function deliberately does NOT call the LLM.

    It first uses:
        - deterministic commands
        - pending state
        - room context
        - search context
        - semantic-ish structured patterns

    Only process_message() calls the existing Gemma intent
    classifier when this router cannot confidently determine
    the message type.

    This prevents unnecessary Ollama calls.
    """

    text = normalize_text(
        user_query
    )

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if should_end_conversation(
        user_query
    ):

        return MessageRoute(
            message_type=MESSAGE_EXIT,
            reason="Explicit exit command.",
        )

    # --------------------------------------------------------
    # SIMPLE STATIC INTENTS
    # --------------------------------------------------------

    simple_intent = detect_simple_intent(
        user_query
    )

    if simple_intent is not None:

        return MessageRoute(
            message_type=simple_intent,
            reason="Deterministic intent match.",
        )

    # --------------------------------------------------------
    # PENDING SEARCH HAS HIGHEST CONTEXTUAL PRIORITY
    # --------------------------------------------------------

    if is_clarification_response(
        user_query,
        state.pending_search,
    ):

        return MessageRoute(
            message_type=MESSAGE_CLARIFICATION,
            reason="Message answers a pending search clarification.",
        )

    # --------------------------------------------------------
    # ROOM CONTEXT
    # --------------------------------------------------------

    if is_likely_followup(
        user_query,
        state.previous_rooms,
        state.selected_room,
    ):

        operation = detect_room_operation(
            user_query
        )

        return MessageRoute(
            message_type=MESSAGE_FOLLOWUP,
            operation=operation,
            reason="Existing verified room context makes this a room follow-up.",
        )

    # --------------------------------------------------------
    # SEARCH REFINEMENT
    # --------------------------------------------------------

    if is_likely_search_refinement(
        user_query,
        state.previous_search,
        state.previous_rooms,
    ):

        return MessageRoute(
            message_type=MESSAGE_REFINEMENT,
            reason="Message contains constraints that can refine the previous search.",
        )

    # --------------------------------------------------------
    # SHORT VALUES AFTER A SEARCH
    # --------------------------------------------------------

    if state.previous_search:

        # "2bhk"
        # "under 10000"
        # "students"
        # "furnished"
        if (
            re.search(
                r"\b\d+\s*bhk\b",
                text,
            )
            or re.search(
                r"₹?\s*\d{3,}",
                text,
            )
            or text in {
                "students",
                "student",
                "families",
                "family",
                "furnished",
                "unfurnished",
            }
        ):

            return MessageRoute(
                message_type=MESSAGE_REFINEMENT,
                reason="Short message interpreted as a refinement of the previous search.",
            )

    # --------------------------------------------------------
    # UNKNOWN FOR NOW
    # --------------------------------------------------------

    return MessageRoute(
        message_type=MESSAGE_UNKNOWN,
        confidence=0.0,
        reason="No deterministic contextual route found.",
    )


# ============================================================
# MERGE SEARCH PARAMETERS
# ============================================================

def merge_search_parameters(
    previous: dict[str, Any],
    new: dict[str, Any],
) -> dict[str, Any]:

    merged = deepcopy(
        previous
    )

    for key, value in new.items():

        if value is None:
            continue

        if isinstance(
            value,
            list,
        ) and not value:

            continue

        # Parser defaults should not erase previous values.
        if (
            key == "sort"
            and value == "latest"
            and key in merged
        ):

            continue

        if (
            key == "limit"
            and value == 10
            and key in merged
        ):

            continue

        merged[key] = value

    return merged


# ============================================================
# CLARIFICATION PARAMETER EXTRACTION
# ============================================================

def extract_clarification_parameters(
    user_query: str,
    pending_search: dict[str, Any],
) -> dict[str, Any]:

    text = user_query.strip()

    # First use the existing parser.
    try:

        parsed = extract_search_parameters(
            text
        )

        if isinstance(
            parsed,
            dict,
        ):

            useful: dict[str, Any] = {}

            for key, value in parsed.items():

                if value is None:
                    continue

                if (
                    isinstance(
                        value,
                        list,
                    )
                    and not value
                ):

                    continue

                useful[key] = value

            if useful.get(
                "city"
            ):

                return useful

    except Exception as error:

        print(
            f"⚠ Parser clarification extraction "
            f"failed, using fallback: {error}"
        )

    # Simple city response.
    cleaned = (
        text
        .strip()
        .strip(",.!?")
    )

    if (
        pending_search.get(
            "city"
        ) is None
        and cleaned
        and len(
            cleaned.split()
        ) <= 4
    ):

        return {
            "city": cleaned.title()
        }

    patterns = (
        r"\bin\s+([A-Za-z][A-Za-z .'-]+)$",
        r"\bat\s+([A-Za-z][A-Za-z .'-]+)$",
        r"\bnear\s+([A-Za-z][A-Za-z .'-]+)$",
        r"\baround\s+([A-Za-z][A-Za-z .'-]+)$",
    )

    for pattern in patterns:

        match = re.search(
            pattern,
            cleaned,
            re.IGNORECASE,
        )

        if match:

            city = match.group(
                1
            ).strip()

            if city:

                return {
                    "city": city.title()
                }

    return {}


# ============================================================
# EXTRACT ROOMS FROM MCP
# ============================================================

def extract_rooms_from_response(
    api_data: dict[str, Any],
) -> list[dict[str, Any]]:

    rooms = api_data.get(
        "rooms"
    )

    if isinstance(
        rooms,
        list,
    ):

        return rooms

    data = api_data.get(
        "data"
    )

    if isinstance(
        data,
        list,
    ):

        return data

    if isinstance(
        data,
        dict,
    ):

        rooms = data.get(
            "rooms"
        )

        if isinstance(
            rooms,
            list,
        ):

            return rooms

        rooms = data.get(
            "data"
        )

        if isinstance(
            rooms,
            list,
        ):

            return rooms

    return []


# ============================================================
# FAST SEARCH RESPONSE
# ============================================================

def generate_fast_response(
    user_query: str,
    rooms: list[dict[str, Any]],
    search_request: dict[str, Any],
) -> str:

    if not rooms:

        city = search_request.get(
            "city"
        )

        max_price = search_request.get(
            "max_price"
        )

        if city and max_price:

            return (
                f"I couldn't find any matching rooms "
                f"in {city} under ₹{max_price:,}."
            )

        if city:

            return (
                f"I couldn't find any matching rooms "
                f"in {city}."
            )

        return (
            "I couldn't find any matching rooms."
        )

    city = search_request.get(
        "city"
    )

    max_price = search_request.get(
        "max_price"
    )

    lines: list[str] = []

    if city and max_price:

        lines.append(
            f"Here are the matching options "
            f"in {city} under ₹{max_price:,}:"
        )

    elif city:

        lines.append(
            f"Here are the available options "
            f"in {city}:"
        )

    else:

        lines.append(
            "Here are the matching options:"
        )

    lines.append("")

    for index, room in enumerate(
        rooms,
        start=1,
    ):

        lines.append(
            f"{index}. **{get_room_title(room)}**"
        )

        price = get_room_price(
            room
        )

        if price is not None:

            lines.append(
                f"   ₹{price:,.0f}/month"
            )

        room_type = get_room_type(
            room
        )

        if room_type is not None:

            lines.append(
                f"   Room type: {room_type}"
            )

        location = get_room_location(
            room
        )

        if location:

            lines.append(
                f"   Location: {location}"
            )

        ideal_for = get_room_ideal_for(
            room
        )

        if ideal_for:

            lines.append(
                "   Ideal for: "
                + ", ".join(
                    str(value)
                    for value in ideal_for
                )
            )

        lines.append("")

    return "\n".join(
        lines
    ).strip()


# ============================================================
# FAST FOLLOW-UP RESPONSE
# ============================================================

def generate_fast_followup(
    user_query: str,
    rooms: list[dict[str, Any]],
    selected_room: dict[str, Any] | None,
) -> str:

    # Comparison across all rooms.
    comparison_response, comparison_room = (
        answer_room_comparison(
            user_query,
            rooms,
        )
    )

    if comparison_response is not None:

        return comparison_response

    # Specific room fact.
    if selected_room is not None:

        fact = answer_room_fact(
            user_query,
            selected_room,
        )

        if fact is not None:

            return fact

    # If a deterministic reference exists, give a useful
    # response about that room.
    resolved = resolve_room_reference(
        user_query,
        rooms,
        selected_room,
    )

    if resolved is not None:

        title = get_room_title(
            resolved
        )

        price = get_room_price(
            resolved
        )

        if price is not None:

            return (
                f"**{title}** is listed at "
                f"₹{price:,.0f}/month."
            )

        return (
            f"The selected room is **{title}**."
        )

    return (
        "I can answer questions about the verified "
        "rooms I found, including their price, room "
        "type, location, suitability and available "
        "amenities."
    )


# ============================================================
# LLM RESPONSE CONTEXT
# ============================================================

def build_llm_room_context(
    rooms: list[dict[str, Any]],
    selected_room: dict[str, Any] | None,
    selected_room_id: str | None,
    search_request: dict[str, Any],
    conversation_history: list[dict[str, Any]],
) -> str:

    return f"""
RECENT CONVERSATION:
{json.dumps(
    conversation_history[-10:],
    indent=2,
    ensure_ascii=False,
)}

CURRENT SEARCH PARAMETERS:
{json.dumps(
    search_request,
    indent=2,
    ensure_ascii=False,
)}

SELECTED ROOM ID:
{selected_room_id}

SELECTED ROOM:
{json.dumps(
    selected_room,
    indent=2,
    ensure_ascii=False,
)}

ALL VERIFIED ROOM DATA:
{json.dumps(
    rooms,
    indent=2,
    ensure_ascii=False,
)}
"""


# ============================================================
# QWEN RESPONSE
# ============================================================

def generate_qwen_response(
    user_query: str,
    rooms: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]],
    selected_room: dict[str, Any] | None,
    selected_room_id: str | None,
    search_request: dict[str, Any],
) -> str:

    context = build_llm_room_context(
        rooms,
        selected_room,
        selected_room_id,
        search_request,
        conversation_history,
    )

    system_prompt = """
You are a concise Homilivo room-search assistant.

Use ONLY verified room data.

IMPORTANT:
The application has already resolved room references.

If SELECTED ROOM is present, it is authoritative.

Never replace the selected room with another room.

Never invent:
- price
- rent
- room type
- location
- amenities
- facilities
- suitability
- availability
- ratings
- property details

If information is unavailable, explicitly say that
the available room data does not specify it.

For comparison questions, trust the application-selected
room and structured context.

Do not mention internal implementation details.

Answer the user's actual question directly.
"""

    user_prompt = f"""
{context}

CURRENT USER QUESTION:
{user_query}

Answer using only verified room information.
"""

    response = chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.message.content


# ============================================================
# GEMMA RESPONSE
# ============================================================

def generate_gemma_response(
    user_query: str,
    rooms: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]],
    selected_room: dict[str, Any] | None,
    selected_room_id: str | None,
    search_request: dict[str, Any],
) -> str:

    context = build_llm_room_context(
        rooms,
        selected_room,
        selected_room_id,
        search_request,
        conversation_history,
    )

    system_prompt = """
You are the Homilivo room-search assistant.

Use ONLY verified room data.

IMPORTANT:
The application has already resolved room references.

If SELECTED ROOM is present, it is authoritative.

Never choose another room when a selected room is supplied.

Never invent property facts.

If requested information is unavailable,
say so clearly.

Keep the response concise and directly answer
the user's question.

Do not mention internal implementation details.
"""

    user_prompt = f"""
{context}

CURRENT USER QUESTION:
{user_query}

Answer using only verified room information.
"""

    response = chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.message.content


# ============================================================
# GROQ RESPONSE
# ============================================================

_groq_client = None


def _get_groq_client():
    """Lazily create one Groq client, reading the key from the environment."""
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def generate_groq_response(
    user_query: str,
    rooms: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]],
    selected_room: dict[str, Any] | None,
    selected_room_id: str | None,
    search_request: dict[str, Any],
) -> str:

    context = build_llm_room_context(
        rooms,
        selected_room,
        selected_room_id,
        search_request,
        conversation_history,
    )

    system_prompt = """
You are the Homilivo room-search assistant.

Use ONLY verified room data.

IMPORTANT:
The application has already resolved room references.

If SELECTED ROOM is present, it is authoritative.

Never choose another room when a selected room is supplied.

Never invent property facts.

If requested information is unavailable,
say so clearly.

Keep the response concise and directly answer
the user's question.

Do not mention internal implementation details.
"""

    user_prompt = f"""
{context}

CURRENT USER QUESTION:
{user_query}

Answer using only verified room information.
"""

    client = _get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.choices[0].message.content


# ============================================================
# SHARED RESPONSE GENERATOR
# ============================================================

def generate_response(
    mode: str,
    user_query: str,
    rooms: list[dict[str, Any]],
    search_request: dict[str, Any],
    conversation_history: list[dict[str, Any]],
    is_followup: bool,
    selected_room: dict[str, Any] | None,
    selected_room_id: str | None,
) -> str:

    # --------------------------------------------------------
    # Deterministic structured answer ALWAYS gets priority.
    #
    # This prevents unnecessary LLM calls.
    # --------------------------------------------------------

    if is_followup:

        comparison_response, _ = (
            answer_room_comparison(
                user_query,
                rooms,
            )
        )

        if comparison_response is not None:

            return comparison_response

        if selected_room is not None:

            fact = answer_room_fact(
                user_query,
                selected_room,
            )

            if fact is not None:

                return fact

        else:

            # No single room was selected/resolved (e.g. "does any room
            # have ac?" - a set-wide question, not a reference to one
            # room). Check the amenity across the whole current room
            # set from verified session data before falling through to
            # any mode-specific (and possibly LLM-backed) handling.
            set_wide_amenity_answer = answer_amenity_across_rooms(
                user_query,
                rooms,
            )

            if set_wide_amenity_answer is not None:

                return set_wide_amenity_answer

    # --------------------------------------------------------
    # FAST
    # --------------------------------------------------------

    if mode == MODE_FAST:

        if is_followup:

            return generate_fast_followup(
                user_query,
                rooms,
                selected_room,
            )

        return generate_fast_response(
            user_query,
            rooms,
            search_request,
        )

    # --------------------------------------------------------
    # QWEN3
    # --------------------------------------------------------

    if mode == MODE_QWEN:

        return generate_qwen_response(
            user_query=user_query,
            rooms=rooms,
            conversation_history=conversation_history,
            selected_room=selected_room,
            selected_room_id=selected_room_id,
            search_request=search_request,
        )

    # --------------------------------------------------------
    # GEMMA3
    # --------------------------------------------------------

    if mode == MODE_GEMMA:

        return generate_gemma_response(
            user_query=user_query,
            rooms=rooms,
            conversation_history=conversation_history,
            selected_room=selected_room,
            selected_room_id=selected_room_id,
            search_request=search_request,
        )

    # --------------------------------------------------------
    # GROQ (falls back to FAST if the Groq call fails)
    # --------------------------------------------------------

    if mode == MODE_GROQ:

        try:

            return generate_groq_response(
                user_query=user_query,
                rooms=rooms,
                conversation_history=conversation_history,
                selected_room=selected_room,
                selected_room_id=selected_room_id,
                search_request=search_request,
            )

        except Exception as error:

            print_failure(
                f"Groq failed, falling back to FAST: {error}"
            )

            if is_followup:

                return generate_fast_followup(
                    user_query,
                    rooms,
                    selected_room,
                )

            return generate_fast_response(
                user_query,
                rooms,
                search_request,
            )

    raise ValueError(
        f"Unsupported mode: {mode}"
    )


# ============================================================
# OUTPUT GUARDRAIL
# ============================================================

def print_validated_response(
    final_text: str,
) -> bool:

    print_step(
        "Running output guardrail..."
    )

    valid, message = (
        validate_ai_response(
            final_text
        )
    )

    if not valid:

        print_failure(
            f"Output guardrail blocked response: "
            f"{message}"
        )

        return False

    print_success(
        "Output guardrail passed"
    )

    print_header(
        "FINAL RESPONSE"
    )

    print(final_text)

    return True


# ============================================================
# FALLBACK LLM INTENT CLASSIFICATION
# ============================================================

def fallback_llm_intent(
    user_query: str,
) -> str | None:
    """
    Existing Gemma intent classifier is deliberately kept
    as a FALLBACK only.

    It should NOT run for normal contextual room questions.
    """

    print_step(
        "Using semantic LLM intent classification "
        "as fallback..."
    )

    try:

        intent_result = classify_intent(
            user_query
        )

    except GemmaError as error:

        print_failure(
            f"Intent classification failed: {error}"
        )

        return None

    intent = intent_result.get(
        "intent"
    )

    confidence = float(
        intent_result.get(
            "confidence",
            0.0,
        )
    )

    print_success(
        f"Fallback intent: {intent}"
    )

    print(
        f"  Confidence : {confidence:.2f}"
    )

    print(
        f"  Reason     : "
        f"{intent_result.get('reason', '')}"
    )

    if confidence < 0.70:

        return None

    if intent == "GREETING":

        return MESSAGE_GREETING

    if intent == "CHATBOT_INFO":

        return MESSAGE_INFO

    if intent == "ROOM_SEARCH":

        return MESSAGE_NEW_SEARCH

    return MESSAGE_UNKNOWN


# ============================================================
# PROCESS ONE MESSAGE
# ============================================================

async def process_message(
    user_query: str,
    mode: str,
    client,
    state: ConversationState,
) -> dict[str, Any]:

    result_state = {
        "response": None,
        "rooms": state.previous_rooms,
        "search": state.previous_search,
        "selected_room": state.selected_room,
        "selected_room_id": state.selected_room_id,
        "pending_search": state.pending_search,
    }

    # ========================================================
    # 1. EXIT
    # ========================================================

    if should_end_conversation(
        user_query
    ):

        return result_state

    # ========================================================
    # 2. INPUT GUARDRAIL
    # ========================================================

    print_step(
        "Running input guardrail..."
    )

    input_allowed, input_message = (
        validate_user_input(
            user_query
        )
    )

    # ========================================================
    # 3. ROUTE BEFORE EXPENSIVE LLM CLASSIFICATION
    # ========================================================

    route = route_message(
        user_query,
        state,
    )

    print_step(
        f"Detected message type: "
        f"{route.message_type}"
    )

    if route.operation:

        print(
            f"  Operation: {route.operation}"
        )

    # --------------------------------------------------------
    # Contextual messages may be incomplete standalone
    # sentences. They are still validated against the
    # conversation context and existing room data.
    # --------------------------------------------------------

    contextual_message = (
        route.message_type
        in {
            MESSAGE_FOLLOWUP,
            MESSAGE_REFINEMENT,
            MESSAGE_CLARIFICATION,
        }
    )

    if not input_allowed:

        if contextual_message:

            print_success(
                "Contextual message accepted "
                "using existing conversation state"
            )

        else:

            # Do NOT immediately reject a semantically valid
            # accommodation request.
            #
            # Use the existing semantic classifier only when
            # the lightweight router could not establish a
            # contextual route.
            fallback_intent = fallback_llm_intent(
                user_query
            )

            if fallback_intent is None:

                print_failure(
                    f"Input blocked: {input_message}"
                )

                return result_state

            if fallback_intent in {
                MESSAGE_NEW_SEARCH,
                MESSAGE_GREETING,
                MESSAGE_INFO,
            }:

                print_success(
                    "Input accepted after semantic "
                    "fallback classification"
                )

                route = MessageRoute(
                    message_type=fallback_intent,
                    reason="Fallback semantic classification.",
                )

            else:

                print_failure(
                    f"Input blocked: {input_message}"
                )

                return result_state

    else:

        print_success(
            "Input guardrail passed"
        )

    # ========================================================
    # 4. STATIC RESPONSE ROUTES
    # ========================================================

    if route.message_type == MESSAGE_GREETING:

        final_text = GREETING_RESPONSE

        if print_validated_response(
            final_text
        ):

            result_state["response"] = final_text

        return result_state

    if route.message_type == MESSAGE_INFO:

        final_text = CHATBOT_INFO_RESPONSE

        if print_validated_response(
            final_text
        ):

            result_state["response"] = final_text

        return result_state

    if route.message_type == MESSAGE_UNKNOWN:

        # Only now use the LLM classifier.
        fallback_intent = fallback_llm_intent(
            user_query
        )

        if fallback_intent == MESSAGE_GREETING:

            final_text = GREETING_RESPONSE

            if print_validated_response(
                final_text
            ):

                result_state["response"] = final_text

            return result_state

        if fallback_intent == MESSAGE_INFO:

            final_text = CHATBOT_INFO_RESPONSE

            if print_validated_response(
                final_text
            ):

                result_state["response"] = final_text

            return result_state

        if fallback_intent == MESSAGE_NEW_SEARCH:

            route = MessageRoute(
                message_type=MESSAGE_NEW_SEARCH,
                reason="LLM fallback identified room search.",
            )

        else:

            final_text = UNKNOWN_RESPONSE

            if print_validated_response(
                final_text
            ):

                result_state["response"] = final_text

            return result_state

    # ========================================================
    # 5. ROOM FOLLOW-UP
    # ========================================================

    if route.message_type == MESSAGE_FOLLOWUP:

        print_step(
            "Resolving room reference..."
        )

        resolved_room = resolve_room_reference(
            user_query=user_query,
            previous_rooms=state.previous_rooms,
            selected_room=state.selected_room,
        )

        if resolved_room is not None:

            result_state["selected_room"] = (
                resolved_room
            )

            result_state["selected_room_id"] = (
                get_room_id(
                    resolved_room
                )
            )

            print_success(
                "Room reference resolved"
            )

            print(
                f"  Selected room : "
                f"{get_room_title(resolved_room)}"
            )

            price = get_room_price(
                resolved_room
            )

            if price is not None:

                print(
                    f"  Price         : "
                    f"₹{price:,.0f}/month"
                )

            print(
                f"  Room ID       : "
                f"{get_room_id(resolved_room)}"
            )

        elif state.selected_room is not None:

            print_success(
                "Using previously selected room."
            )

        else:

            print(
                "No deterministic room reference found."
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # If a structured answer exists, do NOT call an LLM.
        # ----------------------------------------------------

        final_text = generate_response(
            mode=mode,
            user_query=user_query,
            rooms=state.previous_rooms,
            search_request=state.previous_search,
            conversation_history=state.conversation_history,
            is_followup=True,
            selected_room=result_state[
                "selected_room"
            ],
            selected_room_id=result_state[
                "selected_room_id"
            ],
        )

        if print_validated_response(
            final_text
        ):

            result_state["response"] = final_text

        return result_state

    # ========================================================
    # 6. SEARCH PARAMETERS
    # ========================================================

    if route.message_type == MESSAGE_CLARIFICATION:

        print_step(
            "Processing clarification response..."
        )

        clarification_parameters = (
            extract_clarification_parameters(
                user_query,
                state.pending_search,
            )
        )

        if not clarification_parameters:

            final_text = (
                "I still need the missing search "
                "information. Please provide the city."
            )

            if print_validated_response(
                final_text
            ):

                result_state["response"] = final_text

            return result_state

        search_request = merge_search_parameters(
            state.pending_search,
            clarification_parameters,
        )

        print_success(
            "Clarification merged into pending search"
        )

    elif route.message_type == MESSAGE_REFINEMENT:

        print_step(
            "Extracting refinement parameters..."
        )

        try:

            extracted_parameters = (
                extract_search_parameters(
                    user_query
                )
            )

        except Exception as error:

            print_failure(
                f"Parameter extraction failed: {error}"
            )

            return result_state

        search_request = merge_search_parameters(
            state.previous_search,
            extracted_parameters,
        )

        print_success(
            "Previous search parameters merged "
            "with new constraints"
        )

    else:

        # ----------------------------------------------------
        # NEW SEARCH
        # ----------------------------------------------------

        print_step(
            "Extracting search parameters..."
        )

        try:

            extracted_parameters = (
                extract_search_parameters(
                    user_query
                )
            )

        except Exception as error:

            print_failure(
                f"Parameter extraction failed: {error}"
            )

            return result_state

        search_request = (
            extracted_parameters
        )

    print(
        "\nSearch Parameters:"
    )

    for key, value in (
        search_request.items()
    ):

        print(
            f"  {key:15}: {value}"
        )

    # ========================================================
    # 7. REQUIRED INFORMATION
    # ========================================================

    if not search_request.get(
        "city"
    ):

        print(
            "\n⚠ Search location is missing."
        )

        result_state["pending_search"] = (
            search_request
        )

        final_text = (
            "Sure. Which city are you looking for?"
        )

        if print_validated_response(
            final_text
        ):

            result_state["response"] = final_text

        return result_state

    # ========================================================
    # 8. PARAMETER GUARDRAIL
    # ========================================================

    print_step(
        "Validating search parameters..."
    )

    parameters_valid, parameter_message = (
        validate_search_arguments(
            search_request
        )
    )

    if not parameters_valid:

        print_failure(
            parameter_message
        )

        return result_state

    print_success(
        "Search parameters validated"
    )

    # ========================================================
    # 9. MCP
    # ========================================================

    print_header(
        "EXECUTING MCP TOOL"
    )

    print(
        "Calling: search_rooms(...)"
    )

    try:

        mcp_result = await client.call_tool(
            "search_rooms",
            search_request,
        )

    except Exception as error:

        print_failure(
            f"MCP execution failed: {error}"
        )

        return result_state

    print_success(
        "MCP tool executed"
    )

    # ========================================================
    # 10. MCP RESPONSE
    # ========================================================

    if not mcp_result.content:

        print_failure(
            "MCP returned no content."
        )

        return result_state

    try:

        response_text = (
            mcp_result.content[0].text
        )

        api_data = json.loads(
            response_text
        )

    except (
        AttributeError,
        IndexError,
        json.JSONDecodeError,
    ) as error:

        print_failure(
            f"Invalid MCP response: {error}"
        )

        return result_state

    if api_data.get(
        "success"
    ) is False:

        print_failure(
            api_data.get(
                "error",
                "Homilivo API returned an error.",
            )
        )

        return result_state

    # ========================================================
    # 11. EXTRACT ROOMS
    # ========================================================

    rooms = extract_rooms_from_response(
        api_data
    )

    print_success(
        f"MCP returned {len(rooms)} room(s)"
    )

    # ========================================================
    # 12. API DATA GUARDRAIL
    # ========================================================

    print_step(
        "Validating room data..."
    )

    data_valid, data_message = (
        validate_room_data(
            rooms
        )
    )

    if not data_valid:

        print_failure(
            data_message
        )

        return result_state

    print_success(
        "API room data validated"
    )

    # ========================================================
    # 13. UPDATE VERIFIED SEARCH STATE
    # ========================================================

    result_state["rooms"] = rooms

    result_state["search"] = (
        search_request
    )

    result_state["pending_search"] = {}

    # New completed search = new room universe.
    result_state["selected_room"] = None
    result_state["selected_room_id"] = None

    # ========================================================
    # 14. RESPONSE GENERATION
    # ========================================================

    print_header(
        "GENERATING RESPONSE"
    )

    print(
        f"Mode: {mode.upper()}"
    )

    try:

        final_text = generate_response(
            mode=mode,
            user_query=user_query,
            rooms=rooms,
            search_request=search_request,
            conversation_history=state.conversation_history,
            is_followup=False,
            selected_room=None,
            selected_room_id=None,
        )

    except Exception as error:

        print_failure(
            f"{mode.upper()} response generation failed: "
            f"{error}"
        )

        return result_state

    # ========================================================
    # 15. OUTPUT GUARDRAIL
    # ========================================================

    if print_validated_response(
        final_text
    ):

        result_state["response"] = final_text

    return result_state


# ============================================================
# MODE SELECTION
# ============================================================

def choose_mode() -> str | None:

    print_header(
        "HOMILIVO CHATBOT - RESPONSE MODE"
    )

    print(
        "Choose response generation mode:"
    )

    print(
        "\n1. FAST"
        "\n2. QWEN3"
        "\n3. GEMMA3"
        "\n4. GROQ"
        "\n5. EXIT"
    )

    choice = input(
        "\nEnter choice: "
    ).strip()

    return {
        "1": MODE_FAST,
        "2": MODE_QWEN,
        "3": MODE_GEMMA,
        "4": MODE_GROQ,
    }.get(choice)


# ============================================================
# MAIN CONVERSATION LOOP
# ============================================================

async def conversational_main():

    mode = choose_mode()

    if mode is None:

        print(
            "\nExiting."
        )

        return

    print_header(
        f"{mode.upper()} MODE"
    )

    print(
        "✓ Continuous conversation enabled."
    )

    print(
        "✓ Previous room results will be remembered."
    )

    print(
        "✓ Selected room context is preserved."
    )

    print(
        "✓ Search refinement is supported."
    )

    print(
        "✓ Deterministic room questions are enabled."
    )

    print(
        "✓ LLM classification is fallback-only."
    )

    print(
        "\nType 'bye', 'exit', 'quit', "
        "'ok', 'okay', 'thanks' or "
        "'thank you' to end."
    )

    print(
        "\nConversation started."
    )

    state = ConversationState()

    async with Client(mcp) as client:

        print_success(
            "MCP server connected"
        )

        while True:

            print(
                "\n" + "-" * 70
            )

            user_query = input(
                "\nUser: "
            ).strip()

            if not user_query:

                print(
                    "Please enter a message."
                )

                continue

            if should_end_conversation(
                user_query
            ):

                print(
                    "\nAssistant: "
                    "Thanks for chatting with Homilivo. "
                    "Goodbye!"
                )

                break

            # ------------------------------------------------
            # STORE USER MESSAGE
            # ------------------------------------------------

            state.conversation_history.append(
                {
                    "role": "user",
                    "content": user_query,
                }
            )

            # ------------------------------------------------
            # PROCESS
            # ------------------------------------------------

            result = await process_message(
                user_query=user_query,
                mode=mode,
                client=client,
                state=state,
            )

            # ------------------------------------------------
            # UPDATE STRUCTURED MEMORY
            # ------------------------------------------------

            response = result.get(
                "response"
            )

            if response:

                state.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )

            rooms = result.get(
                "rooms"
            )

            if isinstance(
                rooms,
                list,
            ):

                state.previous_rooms = rooms

            search = result.get(
                "search"
            )

            if isinstance(
                search,
                dict,
            ):

                state.previous_search = search

            selected_room = result.get(
                "selected_room"
            )

            state.selected_room = (
                selected_room
                if isinstance(
                    selected_room,
                    dict,
                )
                else None
            )

            state.selected_room_id = (
                result.get(
                    "selected_room_id"
                )
            )

            pending_search = result.get(
                "pending_search"
            )

            if isinstance(
                pending_search,
                dict,
            ):

                state.pending_search = (
                    pending_search
                )

            print(
                "\n[Memory:"
                f" {len(state.conversation_history)} messages"
                f" | Previous rooms:"
                f" {len(state.previous_rooms)}"
                f" | Selected room:"
                f" {'YES' if state.selected_room else 'NO'}"
                f" | Pending search:"
                f" {'YES' if state.pending_search else 'NO'}"
                "]"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        conversational_main()
    )