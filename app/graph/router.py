"""
LangGraph router - Phase 5 target-architecture conformance.

Wraps app.experiments.agent_modes.route_message() (the tested
deterministic router, which itself calls detect_room_operation()) as
the graph's fast-path classification node, then routes to one of four
explicit branch nodes matching the target diagram: FAQ/RAG,
Conversation, API-request, and the ambiguous-intent LLM-fallback path
(reached only when route_message() cannot classify the message
deterministically). Every branch node executes the turn via the
UNCHANGED process_message() pipeline - see app/graph/nodes/execute.py
- which is the single place all four guardrails, extraction (regex +
Phase 6 Pydantic LLM fallback), resolve_room_reference(), the MCP tool
call, and response generation actually run. This module adds explicit
graph structure around that pipeline; it does not reimplement any part
of it (see docs/Homilivo_Project_Plan.md #6 and app/experiments/
agent_modes.py, both unmodified by this phase).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.experiments.agent_modes import ConversationState
from app.graph.nodes.execute import (
    api_request_node,
    conversation_node,
    faq_node,
    llm_fallback_node,
)
from app.graph.nodes.fast_path import (
    BRANCH_API_REQUEST,
    BRANCH_CONVERSATION,
    BRANCH_FAQ,
    BRANCH_LLM_FALLBACK,
    fast_path_node,
)
from app.graph.state import GraphState

NODE_FAST_PATH = "fast_path"


def _select_branch(state: GraphState) -> str:
    return state["branch"]  # type: ignore[return-value]


def build_graph():
    """Construct (uncompiled) the router graph. Exposed mainly for tests."""
    graph = StateGraph(GraphState)

    graph.add_node(NODE_FAST_PATH, fast_path_node)
    graph.add_node(BRANCH_FAQ, faq_node)
    graph.add_node(BRANCH_CONVERSATION, conversation_node)
    graph.add_node(BRANCH_API_REQUEST, api_request_node)
    graph.add_node(BRANCH_LLM_FALLBACK, llm_fallback_node)

    graph.set_entry_point(NODE_FAST_PATH)
    graph.add_conditional_edges(
        NODE_FAST_PATH,
        _select_branch,
        {
            BRANCH_FAQ: BRANCH_FAQ,
            BRANCH_CONVERSATION: BRANCH_CONVERSATION,
            BRANCH_API_REQUEST: BRANCH_API_REQUEST,
            BRANCH_LLM_FALLBACK: BRANCH_LLM_FALLBACK,
        },
    )
    graph.add_edge(BRANCH_FAQ, END)
    graph.add_edge(BRANCH_CONVERSATION, END)
    graph.add_edge(BRANCH_API_REQUEST, END)
    graph.add_edge(BRANCH_LLM_FALLBACK, END)

    return graph


_COMPILED_GRAPH = None


def get_compiled_graph():
    """Lazily build and cache the compiled graph (it is stateless/reentrant)."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph().compile()
    return _COMPILED_GRAPH


async def run_graph(
    user_query: str,
    mode: str,
    client: Any,
    state: ConversationState,
) -> dict[str, Any]:
    """
    Entry point for app/api.py - same call shape as
    agent_modes.process_message(), so it is a drop-in replacement at
    the one call site in _run_chat_turn(). Returns the identical
    result dict process_message() produces, since every branch node
    delegates to it unchanged; observable behavior, response shapes,
    and session handling are therefore unchanged by routing through
    the graph.
    """
    compiled = get_compiled_graph()
    outcome = await compiled.ainvoke(
        {
            "user_query": user_query,
            "mode": mode,
            "client": client,
            "conversation_state": state,
            "branch": None,
            "result": None,
        }
    )
    return outcome["result"]
