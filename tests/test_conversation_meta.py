"""
Regression tests for the conversation meta query handler - lets a user
ask about their own query history within the current session ("what
was my last query", "show my history", ...).

Covers the same deterministic, LLM-free, no-MCP shape as every other
static route in app/experiments/agent_modes.py: Tier A phrase
detection (detect_conversation_meta_query()), route_message() priority
placement (after CLARIFICATION, before ROOM_FOLLOWUP), state.user_queries
bookkeeping in process_message(), and the MESSAGE_CONVERSATION_META
static handler itself.
"""

from __future__ import annotations

import asyncio

from app.experiments.agent_modes import (
    MODE_FAST,
    ConversationState,
    detect_conversation_meta_query,
    process_message,
)


# ============================================================
# TIER A DETECTION
# ============================================================

def test_detect_conversation_meta_positive():
    positive_queries = (
        "what was my last query",
        "what was my first query",
        "what did i ask",
        "what have i asked",
        "show my history",
        "my previous query",
        "my last query",
    )

    for query in positive_queries:
        assert detect_conversation_meta_query(query), query


def test_detect_conversation_meta_negative():
    negative_queries = (
        "show me rooms in bangalore",
        "under 10k",
        "show in table",
        "I want to contact the owner",
        "compare rooms",
        "hi",
    )

    for query in negative_queries:
        assert not detect_conversation_meta_query(query), query


# ============================================================
# process_message() INTEGRATION
# ============================================================

def _run_turn(user_query: str, state: ConversationState) -> dict:
    async def run():
        return await process_message(
            user_query=user_query,
            mode=MODE_FAST,
            client=None,
            state=state,
        )

    return asyncio.run(run())


def test_user_queries_appended_to_state():
    # "hi" is used (rather than a room search) so this turn resolves
    # via the deterministic GREETING static route with no MCP/network
    # call - the append behaviour being tested here is unconditional,
    # so any route proves it, and this one keeps the test fast and
    # hermetic.
    state = ConversationState()

    _run_turn("hi", state)

    assert state.user_queries == ["hi"]


def test_last_query_response():
    state = ConversationState(
        user_queries=[
            "show me rooms in bangalore",
            "under 10k",
        ],
    )

    result = _run_turn("what was my last query", state)

    assert "under 10k" in result["response"]


def test_first_query_response():
    state = ConversationState(
        user_queries=[
            "show me rooms in bangalore",
            "under 10k",
        ],
    )

    result = _run_turn("what was my first query", state)

    assert "show me rooms in bangalore" in result["response"]


def test_full_history_response():
    state = ConversationState(
        user_queries=[
            "show me rooms in bangalore",
            "under 10k",
        ],
    )

    result = _run_turn("what have i asked", state)

    assert "show me rooms in bangalore" in result["response"]
    assert "under 10k" in result["response"]


def test_empty_history_response():
    state = ConversationState()

    result = _run_turn("what was my last query", state)

    assert "haven't asked anything yet" in result["response"]
