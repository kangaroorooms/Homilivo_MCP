"""PLATFORM_INFO from the existing knowledge base. The LLM never answers this."""

from __future__ import annotations

import functools
import logging
import re
from typing import Any

from llm_approach import reuse
from llm_approach.actions import ActionResult, TurnBudget
from llm_approach.schema import Decision
from llm_approach.state import LLMConversationState

logger = logging.getLogger(__name__)

NOT_GROUNDED = "I don't have that information, please contact Homilivo support."
MIN_LETTERS = 2  # a raw message with fewer letters (any script) than this never reaches the knowledge base

# Placeholder or internal-note text must never reach a user, even if retrieval calls it grounded.
_UNSAFE_MARKERS = re.compile(r"placeholder|not verified|confirm wording|\btodo\b|\btbd\b|lorem", re.IGNORECASE)


def has_internal_marker(text: str) -> bool:
    return bool(_UNSAFE_MARKERS.search(text))


def _chunk_positions(chunks) -> dict[tuple[str, str], int]:
    """(document, chunk text) -> the chunk's index in its document."""
    position: dict[str, int] = {}
    index: dict[tuple[str, str], int] = {}
    for chunk in chunks:
        index[(chunk.doc_id, chunk.text)] = position.get(chunk.doc_id, 0)
        position[chunk.doc_id] = position.get(chunk.doc_id, 0) + 1
    return index


@functools.lru_cache(maxsize=1)
def _cached_chunk_positions() -> dict[tuple[str, str], int]:
    """The documents are read once per process for the debug log (the index itself is built once per process too)."""
    from llm_approach.rag import store

    return _chunk_positions(store._load_documents())


def _log_retrieval(query: str, extra_query: str | None = None, results=None) -> None:
    """Debug only: the raw and rewrite top 3 (document, chunk index in it, score) and which query's chunks the reply uses. Changes nothing.
    ``results``: the (raw, rewrite) lists the answer path already retrieved, so nothing is retrieved again; without them (a direct
    call) both queries are retrieved here."""
    try:
        from llm_approach.rag import store
        from llm_approach.rag.answer import choose_results

        if results is not None:
            index = _cached_chunk_positions()
            raw, rewrite = results
        else:
            index = _chunk_positions(store._load_documents())
            raw = store.retrieve(query)
            rewrite = store.retrieve(extra_query) if extra_query else []

        def label(found):
            return [f"{r.chunk.doc_id}#{index.get((r.chunk.doc_id, r.chunk.text), '?')}={r.score:.3f}" for r in found[:3]]

        used = choose_results(raw, rewrite) if extra_query else raw
        logger.debug("knowledge query=%r rewritten=%r raw_top3=%s rewrite_top3=%s used=%s", query[:120], (extra_query or "")[:120],
                     label(raw), label(rewrite), "rewrite" if extra_query and used is rewrite else "raw")
    except Exception as exc:  # noqa: BLE001 - a log line never breaks a reply
        logger.debug("knowledge query log failed: %s", type(exc).__name__)


def handle_platform_info(
    decision: Decision,
    validated: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
    *,
    user_message: str | None = None,
) -> ActionResult:
    # the user's own words first, the model's rewrite as a second query when it differs; the rewrite's chunks are used only when the raw
    # message finds nothing or the rewrite scores at least REWRITE_WIN_MARGIN higher (rag/answer.choose_results)
    query = user_message or decision.standalone_query
    extra = decision.standalone_query if user_message and decision.standalone_query != user_message else None
    if not query:
        return ActionResult(NOT_GROUNDED, None, "platform_info", error_class="no_query")
    if user_message is not None and sum(ch.isalpha() for ch in user_message) < MIN_LETTERS:
        # no words of the user's own (emoji, digits, punctuation): the knowledge base is not asked, whatever the rewrite says
        return ActionResult(NOT_GROUNDED, None, "platform_info", error_class="no_letters")
    debug = logger.isEnabledFor(logging.DEBUG)
    if debug:
        from llm_approach.rag.answer import LAST_RETRIEVAL

        LAST_RETRIEVAL.set(None)  # filled by the answer call below; still None when the answer function is not the real one
    try:
        answer_fn = reuse.get_answer_platform_question()
        answer = answer_fn(query, extra_query=extra) if extra else answer_fn(query)
    except Exception as exc:
        logger.error("knowledge base failed: %s", type(exc).__name__)
        return ActionResult(NOT_GROUNDED, None, "platform_info", error_class="kb_error")
    if debug:
        _log_retrieval(query, extra, LAST_RETRIEVAL.get())  # the answer's own results: no second retrieval
    message = answer.get("message") if isinstance(answer, dict) else None
    if isinstance(answer, dict) and answer.get("grounded") is True and isinstance(message, str) and message.strip():
        if has_internal_marker(message):
            logger.warning("knowledge base text refused: internal marker")
            return ActionResult(NOT_GROUNDED, None, "platform_info", error_class="placeholder_text")
        return ActionResult(message, None, "platform_info")
    return ActionResult(NOT_GROUNDED, None, "platform_info", error_class="ungrounded")
