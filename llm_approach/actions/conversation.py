"""GENERAL_CHAT, UNSUPPORTED_ACTION and OUT_OF_SCOPE."""

from __future__ import annotations

import re
from typing import Any

from llm_approach import reuse
from llm_approach.actions import ActionResult, TurnBudget
from llm_approach.actions.clarify import is_safe_text
from llm_approach.schema import Decision
from llm_approach.state import LLMConversationState

SAFE_CHAT = "Happy to chat!"
STEER_NO_ROOMS = "Tell me a city and I'll help you find a room."
STEER_WITH_ROOMS = "Would you like to ask about these rooms or search again?"
OUT_OF_SCOPE = "I can only help with finding rooms and answering questions about them."

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _steer(state: LLMConversationState) -> str:
    return STEER_WITH_ROOMS if state.rooms_on_screen else STEER_NO_ROOMS


def cap_sentences(text: str, limit: int = 2) -> str:
    return " ".join(_SENTENCE_SPLIT.split(text.strip())[:limit])


def handle_general_chat(
    decision: Decision,
    validated: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
) -> ActionResult:
    reply = cap_sentences(decision.reply or "")
    error_class = None
    if not is_safe_text(reply):
        reply, error_class = SAFE_CHAT, "unsafe_chat_reply"
    return ActionResult(f"{reply} {_steer(state)}", None, "general_chat", error_class=error_class)


def handle_unsupported(
    decision: Decision,
    validated: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
) -> ActionResult:
    """Fixed wording only: nothing from ``decision.reply``. Owner-contact requests never get here: the pipeline sends them
    to ``owner_contact`` first."""
    return ActionResult(reuse.get_unsupported_action_response(), None, "unsupported_action")


def handle_out_of_scope(
    decision: Decision,
    validated: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
) -> ActionResult:
    return ActionResult(f"{OUT_OF_SCOPE} {_steer(state)}", None, "out_of_scope")
