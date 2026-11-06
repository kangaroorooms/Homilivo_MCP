"""Whether the knowledge base still holds placeholder content.

A document is a placeholder while its text carries the banner "PLACEHOLDER DOCUMENT" (the FAQ has it at the top; the
documents were byte-for-byte copies from ``app/``, so the marker is the one they contained). Replacing the FAQ with
approved content removes the banner and the flag goes false by itself. ``unverified()`` is a second, separate flag: true
while any document says ``verified: false`` in its front matter (the shipped website drafts do, until approved content
replaces them). Both are status only. Reads the files once; no model, no network. ``skipped_documents()`` is the number of
documents the knowledge store's last load skipped because they could not be parsed (reported in /metrics).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

DOCUMENTS = Path(__file__).resolve().parent / "rag" / "documents"
_MARKER = re.compile(r"PLACEHOLDER\s+DOCUMENT", re.IGNORECASE)
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---", re.DOTALL)
_VERIFIED_FALSE = re.compile(r"^verified:\s*false\s*$", re.IGNORECASE | re.MULTILINE)


@lru_cache(maxsize=1)
def placeholder() -> bool:
    """True when any knowledge document carries the placeholder banner."""
    try:
        return any(_MARKER.search(path.read_text(encoding="utf-8")) for path in sorted(DOCUMENTS.glob("*.md")))
    except OSError:
        return False


def skipped_documents() -> int:
    """How many documents the last knowledge-base load skipped because they could not be parsed (0 before the first load)."""
    from llm_approach.rag import store

    return len(store.skipped_documents())


@lru_cache(maxsize=1)
def unverified() -> bool:
    """True when any knowledge document has ``verified: false`` in its front matter."""
    try:
        for path in sorted(DOCUMENTS.glob("*.md")):
            head = _FRONT_MATTER.match(path.read_text(encoding="utf-8"))
            if head and _VERIFIED_FALSE.search(head.group(1)):
                return True
        return False
    except OSError:
        return False
