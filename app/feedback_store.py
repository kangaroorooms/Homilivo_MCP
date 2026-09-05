"""
Append-only storage for POST /feedback (app/api.py).

Mirrors this project's "no heavyweight infra" scope (see
app/observability.py's in-memory metrics registry): each feedback
submission is appended as one JSON line to a local file, keyed to the
request_id app/observability.py already generates per turn (see
app.api._run_chat_turn) so feedback can be joined back to the turn it
refers to.

Never store secrets/credentials/PII here - only request_id, rating, an
optional free-text note, and a server timestamp. The note is
caller-supplied free text; callers must not put PII/secrets in it, and
nothing here inspects or redacts its contents beyond length limits
already enforced by app.api.FeedbackRequest.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

FEEDBACK_LOG_PATH = Path(os.getenv("FEEDBACK_LOG_PATH", "data/feedback.jsonl"))

_lock = threading.Lock()


def record_feedback(request_id: str, rating: str, note: str | None) -> dict[str, Any]:
    """Append one feedback record. Returns the record that was written."""
    entry = {
        "request_id": request_id,
        "rating": rating,
        "note": note,
        "recorded_at": time.time(),
    }

    with _lock:
        FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def read_feedback(path: Path | None = None) -> list[dict[str, Any]]:
    """
    Read all stored feedback records. Test/ops helper - the running
    service never needs to read its own feedback log back.
    """
    target = path or FEEDBACK_LOG_PATH

    if not target.exists():
        return []

    records = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records
