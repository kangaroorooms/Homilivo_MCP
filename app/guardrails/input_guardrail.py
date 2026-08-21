"""
Input guardrail and semantic intent classification.

Supported intents:

ROOM_SEARCH
    Requests for rooms, PGs, apartments, accommodation, etc.

CHATBOT_INFO
    Questions about what the chatbot does or how it can help.

GREETING
    Basic greetings.

OUT_OF_SCOPE
    Requests unrelated to the chatbot's purpose.

SENSITIVE
    Requests involving credentials, tokens, hacking, malware, etc.
"""

from __future__ import annotations

import re
from functools import lru_cache

from sentence_transformers import SentenceTransformer, util

from .common import (
    MAX_USER_INPUT_LENGTH,
    normalize_text,
)


# ============================================================
# MODEL
# ============================================================

SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"

INTENT_THRESHOLD = 0.48
INTENT_MARGIN = 0.03


@lru_cache(maxsize=1)
def get_semantic_model() -> SentenceTransformer:
    """
    Load the embedding model once.

    Subsequent requests reuse the same model instance.
    """
    return SentenceTransformer(SEMANTIC_MODEL_NAME)


# ============================================================
# INTENT EXAMPLES
# ============================================================

ROOM_SEARCH_EXAMPLES = [
    "Find me a room to rent",
    "I need a place to stay",
    "I am looking for accommodation",
    "I need somewhere to live",
    "Find housing for me",
    "I need a place near my college",
    "I am looking for student accommodation",
    "I need a PG",
    "Find a paying guest accommodation",
    "I need a flat for rent",
    "Find an apartment for me",
    "I am searching for a house to rent",
    "Where can I stay in this city",
    "I need affordable housing",
    "Find somewhere to stay for college",
    "I need a rental property",
    "Looking for a place near my university",
    "I need accommodation for next month",
    "Help me find a place to live",
    "I want to rent a 2BHK",
]


CHATBOT_INFO_EXAMPLES = [
    "What does this chatbot do",
    "What can this chatbot do",
    "How can you help me",
    "What can you help me with",
    "What is this chatbot for",
    "What services do you provide",
    "What can I search for",
    "How does this chatbot work",
    "Can you help me find accommodation",
    "What kind of rooms can you find",
    "What information can you provide",
    "Tell me about this chatbot",
    "Explain what you can do",
    "What are your capabilities",
    "How do I use this chatbot",
]


GREETING_EXAMPLES = [
    "Hello",
    "Hi",
    "Hey",
    "Good morning",
    "Good afternoon",
    "Good evening",
    "Hi there",
    "Hello there",
]


OUT_OF_SCOPE_EXAMPLES = [
    "Tell me a joke",
    "Write Python code for me",
    "Explain blockchain",
    "What is machine learning",
    "Tell me today's weather",
    "Write an email for me",
    "Solve this mathematics problem",
    "Explain quantum computing",
    "Create a resume",
    "Translate this paragraph",
    "What is the capital of France",
    "Give me stock market advice",
]


# ============================================================
# SECURITY PATTERNS
# ============================================================

BLOCKED_PATTERNS = [
    r"\bapi[\s_-]*key\b",
    r"\bapi[\s_-]*token\b",
    r"\bbearer[\s_-]*token\b",
    r"\baccess[\s_-]*token\b",
    r"\bpassword\b",
    r"\bsecret\b",
    r"\bcredential[s]?\b",
    r"\bkangaroo[\s_-]*token\b",
    r"\bhack(?:ing)?\b",
    r"\bmalware\b",
    r"\bransomware\b",
    r"\bexploit(?:ation)?\b",
    r"\bsql[\s_-]*injection\b",
]


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

INTENT_EXAMPLES = {
    "ROOM_SEARCH": ROOM_SEARCH_EXAMPLES,
    "CHATBOT_INFO": CHATBOT_INFO_EXAMPLES,
    "GREETING": GREETING_EXAMPLES,
    "OUT_OF_SCOPE": OUT_OF_SCOPE_EXAMPLES,
}


@lru_cache(maxsize=1)
def get_intent_embeddings():
    """
    Precompute embeddings once instead of encoding the examples
    on every request.
    """

    model = get_semantic_model()

    return {
        intent: model.encode(
            examples,
            normalize_embeddings=True,
        )
        for intent, examples in INTENT_EXAMPLES.items()
    }


def classify_user_intent(
    user_query: str,
) -> str:

    model = get_semantic_model()
    embeddings = get_intent_embeddings()

    query_embedding = model.encode(
        user_query,
        normalize_embeddings=True,
    )

    scores = {}

    for intent, intent_embeddings in embeddings.items():

        similarity_scores = util.cos_sim(
            query_embedding,
            intent_embeddings,
        )[0]

        scores[intent] = float(
            similarity_scores.max()
        )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_intent, best_score = ranked[0]

    second_score = (
        ranked[1][1]
        if len(ranked) > 1
        else 0.0
    )

    if best_score < INTENT_THRESHOLD:
        return "OUT_OF_SCOPE"

    if best_score - second_score < INTENT_MARGIN:
        return "OUT_OF_SCOPE"

    return best_intent


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_user_input(
    user_query: str,
) -> tuple[bool, str]:

    if not isinstance(user_query, str):
        return (
            False,
            "Please enter a valid request.",
        )

    if not user_query.strip():
        return (
            False,
            "Please enter a room-search request.",
        )

    if len(user_query) > MAX_USER_INPUT_LENGTH:
        return (
            False,
            "The request is too long. Please provide a shorter request.",
        )

    query = normalize_text(user_query)

    # --------------------------------------------------------
    # Security check FIRST
    # --------------------------------------------------------

    for pattern in BLOCKED_PATTERNS:

        if re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        ):
            return (
                False,
                "I can only help with accommodation and room searches.",
            )

    # --------------------------------------------------------
    # Semantic intent
    # --------------------------------------------------------

    try:

        intent = classify_user_intent(query)

    except Exception:

        return (
            False,
            "I couldn't verify the type of request. Please try an accommodation-related query.",
        )

    # --------------------------------------------------------
    # Valid chatbot intents
    # --------------------------------------------------------

    if intent in {
        "ROOM_SEARCH",
        "CHATBOT_INFO",
        "GREETING",
    }:
        return True, intent

    return (
        False,
        "I can only help with accommodation and room searches.",
    )