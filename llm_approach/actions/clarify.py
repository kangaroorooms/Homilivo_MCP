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

HELP_WITH_ROOMS = (
    "I can search rooms, answer questions about the rooms shown, and remember what you tell me in this chat. "
    'Try "2 BHK in Pune under 15k" or ask which room is cheapest.'
)
HELP_NO_ROOMS = (
    "I can search rooms, answer questions about them once you search, and remember what you tell me in this chat. "
    'Try "2 BHK in Pune under 15k".'
)
HELP_SHORT = 'Just tell me a city to search, for example "rooms in Pune", or ask what I remember about you.'

MAX_SAFE_CHARS = 200

# A bare reply that answers none of our generic questions. Matched on the whole message.
_BARE_REPLIES = frozenset(
    "yes no ok okay k kk yeah yep yup sure haan ha nahi nope fine thik theek".split()
    + ["thik hai", "theek hai", "ok ok"]
)
_ROOMS_WORDS = re.compile(r"\b(?:these|the|those)\s+rooms?\b|\brooms?\s+(?:shown|on\s+screen)\b", re.IGNORECASE)

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


def is_bare_reply(message: Any) -> bool:
    """A message that is only "yes", "no", "ok" and the like."""
    if not isinstance(message, str):
        return False
    return re.sub(r"[^\w\s]", "", message.lower()).strip() in _BARE_REPLIES


def was_generic_prompt(state: LLMConversationState) -> bool:
    """The last bot message ended with one of our generic questions or steers."""
    from llm_approach.actions import conversation

    last = (state.last_bot_message or "").strip()
    tails = (ASK_AFTER_RESULTS, ASK_CITY_BUDGET, conversation.STEER_WITH_ROOMS, conversation.STEER_NO_ROOMS)
    return bool(last) and last.endswith(tails) or last in (HELP_WITH_ROOMS, HELP_NO_ROOMS, HELP_SHORT)


def help_message(state: LLMConversationState) -> str:
    """What the bot can do. Alternates with a short form so it never repeats word for word."""
    full = HELP_WITH_ROOMS if state.rooms_on_screen else HELP_NO_ROOMS
    return HELP_SHORT if (state.last_bot_message or "") == full else full


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
    user_message: str | None = None,
) -> ActionResult:
    """Ask a question. Never searches. A pending city question is asked again.

    Any other question is never repeated word for word, and a bare "yes"/"no"/"ok" (it answers
    nothing we asked) gets the help message instead of the same question again."""
    pending = (state.pending_question or {}).get("type")
    if pending == "city":
        state.set_pending("city")
        return ActionResult(ASK_CITY, None, "ask_city", error_class=error_class)
    question = decision.clarifying_question
    usable = is_safe_text(question) and (state.rooms_on_screen or not _ROOMS_WORDS.search(question))
    reply = question.strip() if usable else state_aware_question(state)
    if is_bare_reply(user_message) or reply == (state.last_bot_message or "").strip():
        reply = help_message(state)
    state.set_pending("clarify")
    return ActionResult(reply, None, "clarify", error_class=error_class)
