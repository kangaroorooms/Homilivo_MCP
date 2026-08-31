"""
Grounds a PLATFORM_KNOWLEDGE question in retrieved knowledge-base
content (app/knowledge/store.py). Deterministic - no LLM involved, so
there is nothing here that can invent a fee, a verification step, or a
booking rule that was not actually retrieved.
"""

from __future__ import annotations

from typing import Any

from app.knowledge.store import retrieve

NOT_AVAILABLE_RESPONSE = (
    "I'm sorry, I don't have verified information about that at the "
    "moment. Please contact the Homilivo support team for assistance."
)


def answer_platform_question(query: str) -> dict[str, Any]:
    """
    Returns:
        {
            "grounded": bool,       # True only if retrieval found a
                                     # relevant, indexed document
            "message": str,         # user-facing text
            "sources": list[str],   # titles of the documents used
        }
    """
    results = retrieve(query)

    if not results:
        return {
            "grounded": False,
            "message": NOT_AVAILABLE_RESPONSE,
            "sources": [],
        }

    seen_texts: list[str] = []
    sources: list[str] = []

    for result in results:
        if result.chunk.text not in seen_texts:
            seen_texts.append(result.chunk.text)
        if result.chunk.title not in sources:
            sources.append(result.chunk.title)

    return {
        "grounded": True,
        "message": "\n\n".join(seen_texts),
        "sources": sources,
    }
