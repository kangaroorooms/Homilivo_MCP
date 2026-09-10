"""
Graph state schema for the LangGraph router.

Deliberately thin: it carries exactly the inputs
agent_modes.process_message() already takes (user_query, mode, client,
conversation state), plus two fields the graph itself owns (which
branch the fast-path node selected, and the turn result once a branch
node has run process_message()). No routing/session data is
duplicated here - ConversationState (app/experiments/agent_modes.py)
remains the single source of truth for conversation memory.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from app.experiments.agent_modes import ConversationState


class GraphState(TypedDict):
    user_query: str
    mode: str
    client: Any  # an open mcp.Client, as process_message() already expects
    conversation_state: ConversationState
    branch: str | None
    result: dict[str, Any] | None
    # Phase 8 true streaming: forwarded unchanged to process_message()'s
    # own on_token - see app/graph/nodes/execute.py. None on every REST
    # turn, golden eval, and existing test - only a live WebSocket LLM
    # turn (app.api.chat_websocket) ever sets this.
    on_token: Callable[[str], None] | None
