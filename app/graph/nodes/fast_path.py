"""
Fast-path routing node.

WRAPS app.experiments.agent_modes.route_message() - the existing,
tested, deterministic router - and nothing else. route_message()
itself already calls detect_room_operation() internally (to populate
MessageRoute.operation for ROOM_FOLLOWUP messages), so reusing it here
reuses that whole chain: deterministic commands, pending-state
clarification, follow-up/refinement/ordinal detection, city
extraction, platform-knowledge matching, and unsupported-action
detection - none of it is reimplemented or copied.

route_message() never calls an LLM and never touches session state
(see its own docstring in agent_modes.py) - it is pure classification.
This node's only job is to map its MessageRoute.message_type onto one
of the target diagram's four explicit graph edges. Which existing code
path actually runs for the chosen branch is decided entirely inside
agent_modes.process_message() (app/graph/nodes/execute.py calls it
unchanged) - this node never executes a turn itself.
"""

from __future__ import annotations

from app.experiments.agent_modes import (
    MESSAGE_CLARIFICATION,
    MESSAGE_EXIT,
    MESSAGE_FOLLOWUP,
    MESSAGE_GREETING,
    MESSAGE_INFO,
    MESSAGE_NEW_SEARCH,
    MESSAGE_PLATFORM_KNOWLEDGE,
    MESSAGE_REFINEMENT,
    MESSAGE_UNKNOWN,
    MESSAGE_UNSUPPORTED_ACTION,
    MessageRoute,
    route_message,
)
from app.graph.state import GraphState

# Branch names double as the graph's node names for the four
# downstream nodes (app/graph/nodes/execute.py) and as the values
# add_conditional_edges() maps MessageRoute.message_type onto.
BRANCH_FAQ = "faq"
BRANCH_CONVERSATION = "conversation"
BRANCH_API_REQUEST = "api_request"
# Reached only when route_message() cannot classify the message
# deterministically (MESSAGE_UNKNOWN) - the one case the target
# architecture's Gemma-fallback classifier exists for. The node itself
# still executes via the unchanged process_message() pipeline, which
# is the sole place fallback_llm_intent() is called - see
# app/graph/nodes/execute.py's module docstring.
BRANCH_LLM_FALLBACK = "llm_fallback"

# Every MessageRoute.message_type route_message() can return, mapped
# onto exactly one of the diagram's branches. GREETING/CHATBOT_INFO/
# EXIT are the deterministic Conversation branch; PLATFORM_KNOWLEDGE is
# FAQ/RAG; NEW_SEARCH/SEARCH_REFINEMENT/ROOM_FOLLOWUP/
# CLARIFICATION_RESPONSE are all API-request turns (a fresh search, a
# refinement, a follow-up on already-fetched rooms, or an answer to a
# pending clarification question all end up at extraction -> parameter
# guardrail -> MCP -> api-data guardrail inside process_message).
# UNSUPPORTED_ACTION is a fixed, deterministic refusal - no LLM, no
# MCP call - so it belongs with Conversation, not API-request.
_MESSAGE_TYPE_TO_BRANCH: dict[str, str] = {
    MESSAGE_GREETING: BRANCH_CONVERSATION,
    MESSAGE_INFO: BRANCH_CONVERSATION,
    MESSAGE_EXIT: BRANCH_CONVERSATION,
    MESSAGE_UNSUPPORTED_ACTION: BRANCH_CONVERSATION,
    MESSAGE_PLATFORM_KNOWLEDGE: BRANCH_FAQ,
    MESSAGE_NEW_SEARCH: BRANCH_API_REQUEST,
    MESSAGE_REFINEMENT: BRANCH_API_REQUEST,
    MESSAGE_FOLLOWUP: BRANCH_API_REQUEST,
    MESSAGE_CLARIFICATION: BRANCH_API_REQUEST,
    MESSAGE_UNKNOWN: BRANCH_LLM_FALLBACK,
}


def classify_branch(route: MessageRoute) -> str:
    """
    Map a route_message() result onto one of the four graph branches.
    Defaults unrecognized message types to API_REQUEST rather than
    raising, matching process_message()'s own default (`else`)
    handling of any message_type it doesn't have an explicit branch
    for (see agent_modes.py's "STATIC RESPONSE ROUTES" block).
    """
    return _MESSAGE_TYPE_TO_BRANCH.get(route.message_type, BRANCH_API_REQUEST)


def fast_path_node(state: GraphState) -> dict:
    route = route_message(state["user_query"], state["conversation_state"])
    return {"branch": classify_branch(route)}
