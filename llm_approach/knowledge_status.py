"""Whether the knowledge base still holds placeholder content.

A document is a placeholder while its text carries the banner "PLACEHOLDER DOCUMENT" (the FAQ has it at the top; the
documents are byte-for-byte copies from ``app/``, so the marker is the one they already contain). Replacing the FAQ with
approved content removes the banner and the flag goes false by itself. Reads the files once; no model, no network.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

DOCUMENTS = Path(__file__).resolve().parent / "rag" / "documents"
_MARKER = re.compile(r"PLACEHOLDER\s+DOCUMENT", re.IGNORECASE)


@lru_cache(maxsize=1)
def placeholder() -> bool:
    """True when any knowledge document carries the placeholder banner."""
    try:
        return any(_MARKER.search(path.read_text(encoding="utf-8")) for path in sorted(DOCUMENTS.glob("*.md")))
    except OSError:
        return False
