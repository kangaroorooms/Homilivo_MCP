"""
Local Gemma integration.

Gemma is used only for natural-language understanding and intent
classification.

It does not:
- access the Homilivo API
- call MCP tools
- generate room data
- invent search parameters

The original user query remains the source of truth for downstream
parameter extraction.
"""

from __future__ import annotations

import json
from typing import Any

import ollama


MODEL_NAME = "gemma3:4b"


SYSTEM_PROMPT = """
You are the intent-understanding component of the Homilivo room-search
assistant.

Your ONLY job is to understand the user's request and return structured
JSON.

You must classify the request into exactly one of these intents:

1. GREETING
   Examples:
   - hello
   - hi
   - good morning
   - hey

2. CHATBOT_INFO
   Examples:
   - what does this chatbot do?
   - how can you help me?
   - tell me about this assistant
   - what can you help me with?

3. ROOM_SEARCH
   Examples:
   - find me a room in Jaipur
   - I need a 2BHK under 15000
   - find accommodation for students
   - I need an affordable place to stay in Jaipur
   - looking for a place for me and my family in Jaipur

   IMPORTANT:
   A request can still be ROOM_SEARCH even when the user has not
   provided enough information to perform an actual search.

   For example:
   "Can you help me find accommodation?"
   must be classified as ROOM_SEARCH.

4. UNKNOWN
   Use this when the request is unrelated to greetings, chatbot
   information, or room/accommodation search.

Return ONLY valid JSON using exactly this structure:

{
  "intent": "GREETING | CHATBOT_INFO | ROOM_SEARCH | UNKNOWN",
  "confidence": 0.0,
  "reason": "short explanation",
  "search_query": "interpreted search request or null"
}

Rules:

- confidence must be a number between 0 and 1.
- Do not invent a city.
- Do not invent a budget.
- Do not invent a room type.
- Do not invent a user category.
- Do not invent room information.
- Do not call tools.
- Do not access external APIs.
- Do not answer the user directly.
- Return JSON only.

For ROOM_SEARCH:

- If the user provides enough information to describe the search,
  preserve the meaning of the request in search_query.
- If the user provides only a general accommodation request,
  search_query may be null.
- Do not remove important constraints such as city, budget, room type,
  students, professionals, or family.
- Do not add constraints that the user did not mention.

For GREETING, CHATBOT_INFO, and UNKNOWN:

- search_query must be null.
"""


class GemmaError(Exception):
    """Raised when Gemma cannot produce a valid response."""


VALID_INTENTS = {
    "GREETING",
    "CHATBOT_INFO",
    "ROOM_SEARCH",
    "UNKNOWN",
}


def classify_intent(user_query: str) -> dict[str, Any]:
    """
    Classify a user query using the local Gemma model.

    Args:
        user_query: Raw user message.

    Returns:
        A validated dictionary containing:
        - intent
        - confidence
        - reason
        - search_query

    Raises:
        ValueError:
            If the user query is invalid or empty.

        GemmaError:
            If Gemma fails or returns an invalid response.
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if not isinstance(user_query, str):
        raise ValueError(
            "User query must be a string."
        )

    user_query = user_query.strip()

    if not user_query:
        raise ValueError(
            "User query cannot be empty."
        )

    # ========================================================
    # GEMMA REQUEST
    # ========================================================

    try:

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_query,
                },
            ],
            format="json",
        )

    except Exception as error:

        raise GemmaError(
            f"Gemma request failed: {error}"
        ) from error

    # ========================================================
    # PARSE GEMMA RESPONSE
    # ========================================================

    try:

        content = response["message"]["content"]

        result = json.loads(content)

    except (
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:

        raise GemmaError(
            f"Gemma returned invalid JSON: {error}"
        ) from error

    # ========================================================
    # RESPONSE CONTAINER VALIDATION
    # ========================================================

    if not isinstance(result, dict):

        raise GemmaError(
            "Gemma response must be a JSON object."
        )

    # ========================================================
    # INTENT VALIDATION
    # ========================================================

    intent = result.get("intent")

    if intent not in VALID_INTENTS:

        raise GemmaError(
            f"Invalid intent returned by Gemma: {intent}"
        )

    # ========================================================
    # CONFIDENCE VALIDATION
    # ========================================================

    confidence = result.get("confidence")

    if (
        isinstance(confidence, bool)
        or not isinstance(
            confidence,
            (int, float),
        )
        or not 0 <= confidence <= 1
    ):

        raise GemmaError(
            "Gemma returned an invalid confidence score."
        )

    # ========================================================
    # REASON VALIDATION
    # ========================================================

    reason = result.get("reason")

    if not isinstance(reason, str):

        raise GemmaError(
            "Gemma returned an invalid reason."
        )

    reason = reason.strip()

    if not reason:

        raise GemmaError(
            "Gemma returned an empty reason."
        )

    # ========================================================
    # SEARCH QUERY VALIDATION
    # ========================================================

    search_query = result.get("search_query")

    if intent == "ROOM_SEARCH":

        # A general search request is valid even when it does not
        # contain enough information for an actual room search.
        if search_query is None:

            normalized_search_query = None

        elif isinstance(search_query, str):

            normalized_search_query = (
                search_query.strip()
            )

            if not normalized_search_query:
                normalized_search_query = None

        else:

            raise GemmaError(
                "ROOM_SEARCH search_query must be "
                "a string or null."
            )

    else:

        # Non-search intents must never carry search information.
        normalized_search_query = None

    # ========================================================
    # RETURN NORMALIZED RESULT
    # ========================================================

    return {
        "intent": intent,
        "confidence": float(confidence),
        "reason": reason,
        "search_query": normalized_search_query,
    }