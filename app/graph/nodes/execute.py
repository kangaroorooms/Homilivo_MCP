"""
Branch-execution nodes.

Each node here is one of the target architecture diagram's explicit
edges out of the LangGraph Router - FAQ/RAG, Conversation, API-request
- plus the ambiguous-intent path the Gemma classifier serves as a
fallback for.

  - FAQ/RAG:       faq_node() calls
                    app.knowledge.answer.answer_platform_question()
                    directly - process_message() is never entered on
                    a grounded hit. Falls back to process_message()
                    only when retrieval finds nothing grounded (or
                    raises), so PLATFORM_KNOWLEDGE questions are never
                    left unanswered.
  - Conversation:  conversation_node() returns the fixed
                    GREETING_RESPONSE directly for a detected greeting
                    (route.message_type == MESSAGE_GREETING), skipping
                    search/extraction/MCP entirely. Falls back to
                    process_message() for CHATBOT_INFO / EXIT /
                    UNSUPPORTED_ACTION so those responses stay defined
                    in exactly one place.
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

Both faq_node() and conversation_node() run the SAME output guardrail
process_message() uses (print_validated_response()) before returning a
response built outside process_message(), and seed the exact same
result_state shape process_message() seeds at its top (see
_seed_result_state() below) so app/api.py's _run_chat_turn() sees an
identical contract regardless of which branch answered the turn.
API-request and LLM-fallback are unchanged: process_message() remains
the tested pipeline they live in, and splitting its search/guardrail
control flow apart branch-by-branch is exactly the reimplementation
this phase must not do.
"""

from __future__ import annotations

from typing import Any

from app.experiments import agent_modes
from app.experiments.agent_modes import (
    GREETING_RESPONSE,
    MESSAGE_GREETING,
    ConversationState,
    print_validated_response,
    process_message,
)
from app.graph.state import GraphState


async def _run_turn(state: GraphState) -> dict[str, Any]:
    result = await process_message(
        user_query=state["user_query"],
        mode=state["mode"],
        client=state["client"],
        state=state["conversation_state"],
        on_token=state.get("on_token"),
        route=state.get("route"),
    )
    return {"result": result}


def _seed_result_state(conversation_state: ConversationState) -> dict[str, Any]:
    """
    Mirrors the result_state seeded at the top of
    agent_modes.process_message() (lines 4684-4702) so a handler that
    bypasses process_message() still returns the exact same shape
    app/api.py's _run_chat_turn() reads from.
    """
    return {
        "response": None,
        "rooms": conversation_state.previous_rooms,
        "search": conversation_state.previous_search,
        "selected_room": conversation_state.selected_room,
        "selected_room_id": conversation_state.selected_room_id,
        "pending_search": conversation_state.pending_search,
        "sources": [],
        "search_performed": False,
        "intent": None,
        "operation": None,
        "tool": None,
        "displayed_rooms": None,
    }


async def faq_node(state: GraphState) -> dict[str, Any]:
    route = state.get("route")

    try:
        # Looked up as a module attribute (not imported directly into
        # this module's namespace) so tests that monkeypatch
        # app.experiments.agent_modes.answer_platform_question (the
        # same call process_message()'s own PLATFORM_KNOWLEDGE branch
        # makes) still take effect here.
        knowledge_result = agent_modes.answer_platform_question(state["user_query"])
    except Exception:  # noqa: BLE001
        knowledge_result = None

    if knowledge_result is not None and knowledge_result.get("grounded"):
        result_state = _seed_result_state(state["conversation_state"])
        if route is not None:
            result_state["intent"] = route.message_type
            result_state["operation"] = route.operation

        final_text = knowledge_result["message"]

        if print_validated_response(final_text):
            result_state["response"] = final_text
            result_state["sources"] = knowledge_result["sources"]

        return {"result": result_state}

    # Fallback: answer_platform_question() found no grounded
    # answer. Delegating to process_message() as safety net.
    return await _run_turn(state)


async def conversation_node(state: GraphState) -> dict[str, Any]:
    route = state.get("route")

    if route is not None and route.message_type == MESSAGE_GREETING:
        result_state = _seed_result_state(state["conversation_state"])
        result_state["intent"] = route.message_type
        result_state["operation"] = route.operation

        if print_validated_response(GREETING_RESPONSE):
            result_state["response"] = GREETING_RESPONSE

        return {"result": result_state}

    # Fallback: message not a clear greeting.
    # Delegating to process_message() as safety net.
    return await _run_turn(state)


async def api_request_node(state: GraphState) -> dict[str, Any]:
    return await _run_turn(state)


async def llm_fallback_node(state: GraphState) -> dict[str, Any]:
    return await _run_turn(state)
