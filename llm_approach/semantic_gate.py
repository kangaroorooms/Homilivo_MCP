"""The semantic input gate: the copied intent classifier, run before the LLM call in shadow or enforce mode.

``guardrails/input_guardrail.py`` (copied) has no conversation-context parameter, so it would refuse follow-ups
such as "does it have wifi". The pipeline therefore asks this gate only when no rooms are on screen and no question
is pending (see ``pipeline._semantic_gate``). The classifier's label and the allowed set are the copy's own
(``validate_user_input`` accepts ROOM_SEARCH, CHATBOT_INFO, GREETING and ACKNOWLEDGMENT).

Privacy: the message text goes into the classifier and nowhere else. Counters and the one INFO log line carry the
hashed session id, the request id, a fixed verdict word and the classifier's label. Never the text.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from llm_approach import observability, reuse, turn_context
from llm_approach.guardrails.common import normalize_text

logger = logging.getLogger(__name__)

ALLOWED_LABELS = frozenset({"ROOM_SEARCH", "CHATBOT_INFO", "GREETING", "ACKNOWLEDGMENT"})


@dataclass(frozen=True)
class Verdict:
    label: str  # the classifier's intent word
    would_block: bool  # the copied guardrail would refuse this message
    blocked: bool  # it was refused (enforce mode only)
    ms: float


def evaluate(text: str, mode: str) -> Verdict | None:
    """Classify ``text``; ``mode`` is "shadow" or "enforce". Records the counters and one log line.

    Returns None when the classifier failed: the turn then goes on unblocked (fail open) and ``gate_error`` counts it.
    """
    started = time.perf_counter()
    try:
        label = reuse.get_semantic_classifier()(normalize_text(text))
    except Exception as exc:  # noqa: BLE001 - a missing model must not take the chatbot down
        observability.record_semantic_gate_error()
        logger.warning(
            "semantic gate failed: %s request_id=%s session=%s",
            type(exc).__name__, turn_context.request_id(), turn_context.session_hash(),
        )
        return None
    ms = (time.perf_counter() - started) * 1000
    would_block = label not in ALLOWED_LABELS
    verdict = Verdict(label, would_block, would_block and mode == "enforce", ms)
    observability.record_semantic_gate(label, would_block, verdict.blocked, ms)
    logger.info(
        "semantic gate mode=%s verdict=%s label=%s request_id=%s session=%s",
        mode, "blocked" if verdict.blocked else ("would_block" if would_block else "pass"), label,
        turn_context.request_id(), turn_context.session_hash(),
    )
    return verdict
