"""REJECT_CURRENT, UNCLEAR, low confidence and rejected required slots."""

from __future__ import annotations

import re
from typing import Any

from llm_approach.actions import ActionResult, TurnBudget
from llm_approach.schema import Decision
from llm_approach.state import LLMConversationState

ASK_CITY = "Which city are you looking for?"
ASK_CITY_BUDGET = "Which city and budget are you looking for?"
ASK_AFTER_RESULTS = "Would you like to search again, or ask about these rooms?"
ASK_REJECT = "Sorry those didn't suit you. What would you like to change: the area, budget or room type?"

MAX_SAFE_CHARS = 200

# Words that would make free LLM text a room, price or platform claim.
_UNSAFE_WORDS = re.compile(
    r"\b(?:rs|inr|rupees?|price[sd]?|pricing|cost[s]?|fees?|charges?|rent|rental|budget|"
    r"lakhs?|cheap(?:er|est)?|expensive|subscriptions?|premium|commission|owners?|"
    r"bhk|wi-?fi|amenit\w*|availab\w*|ratings?|rated|verified|guarantee\w*|refund\w*|"
    r"discount\w*|offers?|deposit)\b",
    re.IGNORECASE,
)
_UNSAFE_CHARS = re.compile(r"[\d₹$€£]|https?:|www\.")


def is_safe_text(text: Any) -> bool:
    """True if free text has no digits, currency, price/room/platform words."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_SAFE_CHARS:
        return False
    return not (_UNSAFE_CHARS.search(text) or _UNSAFE_WORDS.search(text))


def state_aware_question(state: LLMConversationState) -> str:
    return ASK_AFTER_RESULTS if state.rooms_on_screen else ASK_CITY_BUDGET


def handle_reject(
    decision: Decision,
    validated: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
) -> ActionResult:
    if not state.rooms_on_screen:
        state.set_pending("clarify", "no_rooms")
        return ActionResult(ASK_CITY_BUDGET, None, "reject_no_rooms")
    state.reject_rooms()
    state.set_pending("reject")
    return ActionResult(ASK_REJECT, None, "reject_current")


def handle_unclear(
    decision: Decision,
    validated: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
    *,
    error_class: str | None = None,
) -> ActionResult:
    """Ask a question. Never searches. A pending city question is asked again."""
    pending = (state.pending_question or {}).get("type")
    if pending == "city":
        state.set_pending("city")
        return ActionResult(ASK_CITY, None, "ask_city", error_class=error_class)
    question = decision.clarifying_question
    reply = question.strip() if is_safe_text(question) else state_aware_question(state)
    state.set_pending("clarify")
    return ActionResult(reply, None, "clarify", error_class=error_class)
