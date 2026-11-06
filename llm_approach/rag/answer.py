# Copied from app/knowledge/answer.py at commit 7dbcba3 (copied, not imported).
# Changes: import path; the retrieved chunks pass through rag/reply.py before the reply is joined; an optional second query (the model's
# rewrite) is retrieved too and takes over from the raw message only by the rule in choose_results; both result lists are kept in
# LAST_RETRIEVAL for the debug log (marked '# v2:')
# Drift is checked by llm_approach/tests/test_copy_drift.py.
# ==== end of v2 provenance header ====
"""
Grounds a PLATFORM_KNOWLEDGE question in retrieved knowledge-base
content (app/knowledge/store.py). Deterministic - no LLM involved, so
there is nothing here that can invent a fee, a verification step, or a
booking rule that was not actually retrieved.
"""

from __future__ import annotations

import contextvars  # v2: new
from typing import Any

from llm_approach.rag.reply import select_for_reply  # v2: new
from llm_approach.rag.store import TOP_K, RetrievedChunk, retrieve  # v2: was: from app.knowledge.store import retrieve

NOT_AVAILABLE_RESPONSE = (
    "I'm sorry, I don't have verified information about that at the "
    "moment. Please contact the Homilivo support team for assistance."
)
REWRITE_WIN_MARGIN = 0.1  # v2: new, how much higher the rewrite's best score must be than the raw message's for the rewrite's chunks to be used
_EPSILON = 1e-9  # v2: new, so a margin of exactly 0.1 is not lost to float rounding
# v2: (raw results, rewrite results) of the last answer_platform_question call in this context, for the debug log (no second retrieval)
LAST_RETRIEVAL: contextvars.ContextVar[tuple[list[RetrievedChunk], list[RetrievedChunk]] | None] = contextvars.ContextVar(  # v2: new
    "rag_last_retrieval", default=None)  # v2: new


def answer_platform_question(query: str, extra_query: str | None = None) -> dict[str, Any]:  # v2: was: def answer_platform_question(query: str) -> dict[str, Any]:
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
    rewrite: list[RetrievedChunk] = []  # v2: new
    if extra_query and extra_query != query:  # v2: new, each retrieval applied the 0.35 floor
        rewrite = retrieve(extra_query)  # v2: new
    LAST_RETRIEVAL.set((results, rewrite))  # v2: new
    if extra_query and extra_query != query:  # v2: new
        results = choose_results(results, rewrite)  # v2: new

    if not results:
        return {
            "grounded": False,
            "message": NOT_AVAILABLE_RESPONSE,
            "sources": [],
        }

    results = select_for_reply(results)  # v2: best chunk (and a close one from another document), label and question removed

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


def choose_results(raw: list[RetrievedChunk], rewrite: list[RetrievedChunk]) -> list[RetrievedChunk]:  # v2: new
    # v2: the raw message's own results, unless it has none at or above the floor or the rewrite's best score beats the raw best by the margin
    if not raw:  # v2: new
        return rewrite  # v2: new
    if rewrite and rewrite[0].score - raw[0].score >= REWRITE_WIN_MARGIN - _EPSILON:  # v2: new
        return rewrite  # v2: new
    return raw  # v2: new
