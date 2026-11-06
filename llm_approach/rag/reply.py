"""What the user sees for a knowledge answer: the best chunk, cleaned.

Retrieval returns up to three chunks (store.TOP_K, floor 0.40). The reply uses the best one only. The text shown is cleaned (display only; what is embedded
and indexed is untouched): the "<Title> - <Section>: " label at the start of a section chunk is removed, and in the FAQ the
question before the first "? " is dropped, so the reply starts with the answer. No LLM, no network.
"""

from __future__ import annotations

import dataclasses
import re

from llm_approach.rag.store import RetrievedChunk

FAQ_DOC_ID = "homilivo_faq"
_LABEL = re.compile(r"^[^:?\n]{1,80} - [^:?\n]{1,80}: ")  # "Homilivo terms and conditions - What Homilivo is: "


def clean_text(text: str, doc_id: str) -> str:
    """Display text of a chunk: label prefix removed, and for the FAQ the leading question too."""
    text = _LABEL.sub("", text, count=1)
    if doc_id == FAQ_DOC_ID:
        head, sep, rest = text.partition("? ")
        if sep and rest.strip():
            text = rest
    return text.strip()


def select_for_reply(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Only the best chunk (``results`` is ranked best first), with cleaned text; empty when nothing was retrieved."""
    if not results:
        return []
    best = results[0]
    return [dataclasses.replace(best, chunk=dataclasses.replace(best.chunk, text=clean_text(best.chunk.text, best.chunk.doc_id)))]
