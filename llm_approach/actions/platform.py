"""PLATFORM_INFO from the existing knowledge base. The LLM never answers this."""

from __future__ import annotations

import logging
from typing import Any

from llm_approach import reuse
from llm_approach.actions import ActionResult, TurnBudget
from llm_approach.schema import Decision
from llm_approach.state import LLMConversationState

logger = logging.getLogger(__name__)

NOT_GROUNDED = "I don't have that information, please contact Homilivo support."


def handle_platform_info(
    decision: Decision,
    validated: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
    *,
    user_message: str | None = None,
) -> ActionResult:
    query = decision.standalone_query or user_message
    if not query:
        return ActionResult(NOT_GROUNDED, None, "platform_info", error_class="no_query")
    try:
        answer = reuse.get_answer_platform_question()(query)
    except Exception as exc:
        logger.error("knowledge base failed: %s", type(exc).__name__)
        return ActionResult(NOT_GROUNDED, None, "platform_info", error_class="kb_error")
    message = answer.get("message") if isinstance(answer, dict) else None
    if isinstance(answer, dict) and answer.get("grounded") is True and isinstance(message, str) and message.strip():
        return ActionResult(message, None, "platform_info")
    return ActionResult(NOT_GROUNDED, None, "platform_info", error_class="ungrounded")
