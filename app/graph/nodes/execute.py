"""
Branch-execution nodes.

Each node here is one of the target architecture diagram's explicit
edges out of the LangGraph Router - FAQ/RAG, Conversation, API-request
- plus the ambiguous-intent path the Gemma classifier serves as a
fallback for. All four delegate to the SAME existing, unchanged
process_message() (app/experiments/agent_modes.py):

  - FAQ/RAG:       process_message() resolves PLATFORM_KNOWLEDGE
                    messages via app.knowledge.answer.answer_platform_question().
  - Conversation:  process_message() returns the fixed GREETING_RESPONSE /
                    CHATBOT_INFO_RESPONSE / UNSUPPORTED_ACTION_RESPONSE strings.
  - API-request:   process_message() runs extract_search_parameters()
                    (regex) -> extract_search_parameters_via_llm()
                    (Phase 6 Pydantic fallback, on a regex miss) ->
                    validate_search_arguments() (parameter guardrail)
                    -> the search_rooms MCP tool -> validate_room_data()
                    (api-data guardrail) -> generate_response().
  - LLM fallback:  process_message() is also the only place
                    fallback_llm_intent() (the Gemma classifier) is
                    called, and only when route_message() and the
                    input guardrail could not classify the message.

process_message() is the tested pipeline all four of the above already
live in - splitting it apart branch-by-branch would mean copying its
control flow (which stage runs the input guardrail, which stages are
allowed to bypass it for contextual messages, etc.), which is exactly
the reimplementation this phase must not do. Keeping four thin,
separately named nodes (rather than one) is what gives the graph
explicit, diagram-matching edges without touching that control flow.
"""

from __future__ import annotations

from typing import Any

from app.experiments.agent_modes import process_message
from app.graph.state import GraphState


async def _run_turn(state: GraphState) -> dict[str, Any]:
    result = await process_message(
        user_query=state["user_query"],
        mode=state["mode"],
        client=state["client"],
        state=state["conversation_state"],
        on_token=state.get("on_token"),
    )
    return {"result": result}


async def faq_node(state: GraphState) -> dict[str, Any]:
    return await _run_turn(state)


async def conversation_node(state: GraphState) -> dict[str, Any]:
    return await _run_turn(state)


async def api_request_node(state: GraphState) -> dict[str, Any]:
    return await _run_turn(state)


async def llm_fallback_node(state: GraphState) -> dict[str, Any]:
    return await _run_turn(state)
