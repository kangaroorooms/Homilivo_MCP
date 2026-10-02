# P2-11: LLM / tool cost controls
# Verifies the input length constant, the LLM context size cap, and the
# per-turn LLM / MCP tool call budgets.

import asyncio
import contextvars
import logging
from unittest.mock import patch

from app import api
from app.api import MAX_USER_MESSAGE_LENGTH, _run_chat_turn
from app.experiments import agent_modes
from app.experiments.agent_modes import (
    MAX_LLM_CONTEXT_CHARS,
    build_llm_room_context,
)


def _run_turn(message: str, blocked: bool = True):
    """Call _run_chat_turn() directly (POST /chat would hit Pydantic's
    500-char limit first). The security check is stubbed so the turn
    stops right after the length check and never reaches the engine."""
    with patch.object(
        api, "is_security_blocked", return_value=(blocked, "stub-blocked")
    ) as security:
        turn = asyncio.run(_run_chat_turn("s-p2-11", "fast", message))
    return turn, security


def _rooms(count: int) -> list[dict]:
    return [
        {
            "id": str(i),
            "title": f"Room {i} " + "T" * 150,
            "description": "D" * 450,
            "city": "Bangalore",
        }
        for i in range(count)
    ]


def _history(count: int) -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i:02d} " + "H" * 500}
        for i in range(count)
    ]


def _build(rooms, history) -> str:
    return build_llm_room_context(
        rooms=rooms,
        selected_room=None,
        selected_room_id=None,
        search_request={"city": "Bangalore"},
        conversation_history=history,
    )


# ---------------- Fix 1: message length ----------------

def test_message_too_long_rejected():
    turn, security = _run_turn("a" * (MAX_USER_MESSAGE_LENGTH + 1))

    assert turn["success"] is False
    assert "too long" in turn["message"].lower()
    assert turn["search_performed"] is False
    assert turn["rooms"] == []
    # Rejected before the security guardrail ever ran.
    security.assert_not_called()


def test_message_at_limit_accepted():
    turn, security = _run_turn("a" * MAX_USER_MESSAGE_LENGTH)

    # Not rejected for length: it got past the length check and reached
    # the (stubbed) security guardrail.
    security.assert_called_once()
    assert "too long" not in turn["message"].lower()


def test_max_message_length_constant_exists():
    assert MAX_USER_MESSAGE_LENGTH == 2000


# ---------------- Fix 2: LLM context cap ----------------

def test_llm_context_truncated_when_too_large(caplog):
    with caplog.at_level(logging.WARNING):
        context = _build(_rooms(20), _history(10))

    assert len(context) <= MAX_LLM_CONTEXT_CHARS * 1.1
    assert "LLM context truncated" in caplog.text


def test_llm_context_keeps_recent_history():
    history = _history(10)

    # Small enough that dropping some (not all) old history fits it.
    context = _build(_rooms(4), history)

    assert len(context) <= MAX_LLM_CONTEXT_CHARS
    assert "msg-09" in context  # newest kept
    assert "msg-00" not in context  # oldest dropped


def test_llm_context_untouched_when_small():
    context = _build(_rooms(2), _history(2))

    assert "msg-00" in context and "msg-01" in context
    assert len(context) < MAX_LLM_CONTEXT_CHARS


def test_max_context_chars_constant_exists():
    assert MAX_LLM_CONTEXT_CHARS == 8000


# ---------------- Fix 4: per-turn LLM call budget ----------------

def test_llm_calls_limited_per_turn(caplog):
    def scenario():
        agent_modes._start_turn_budget()
        with patch.object(
            agent_modes,
            "classify_intent",
            return_value={"intent": "GREETING", "confidence": 0.99, "reason": ""},
        ) as classify:
            first = agent_modes.fallback_llm_intent("hi")
            second = agent_modes.fallback_llm_intent("hi")
            third = agent_modes.fallback_llm_intent("hi")
        return classify.call_count, first, second, third

    with caplog.at_level(logging.WARNING):
        calls, first, second, third = contextvars.copy_context().run(scenario)

    assert calls == 2
    assert first is not None and second is not None
    assert third is None  # deterministic fallback, no raise
    assert "LLM call limit" in caplog.text


def test_llm_budget_shared_across_both_call_types():
    def scenario():
        agent_modes._start_turn_budget()
        with patch.object(
            agent_modes,
            "classify_intent",
            return_value={"intent": "GREETING", "confidence": 0.99, "reason": ""},
        ), patch.object(
            agent_modes,
            "_extract_search_parameters_via_gemma",
            return_value={"city": "Bangalore"},
        ) as gemma:
            agent_modes.fallback_llm_intent("hi")
            agent_modes.extract_search_parameters_via_llm("room in bangalore")
            over_limit = agent_modes.extract_search_parameters_via_llm("again")
        return gemma.call_count, over_limit

    gemma_calls, over_limit = contextvars.copy_context().run(scenario)

    assert gemma_calls == 1
    assert over_limit is None


def test_llm_budget_resets_each_turn():
    def scenario():
        results = []
        for _ in range(2):  # two turns
            agent_modes._start_turn_budget()
            with patch.object(
                agent_modes,
                "classify_intent",
                return_value={"intent": "GREETING", "confidence": 0.99, "reason": ""},
            ):
                results.append(agent_modes.fallback_llm_intent("hi"))
                results.append(agent_modes.fallback_llm_intent("hi"))
        return results

    assert all(r is not None for r in contextvars.copy_context().run(scenario))


def test_no_limit_outside_a_turn():
    def scenario():
        with patch.object(
            agent_modes,
            "classify_intent",
            return_value={"intent": "GREETING", "confidence": 0.99, "reason": ""},
        ) as classify:
            for _ in range(5):
                agent_modes.fallback_llm_intent("hi")
        return classify.call_count

    assert contextvars.copy_context().run(scenario) == 5


# ---------------- Fix 3: MCP tool call guard ----------------

def test_tool_call_guard_warns_on_second_call(caplog):
    def scenario():
        agent_modes._start_turn_budget()
        agent_modes._note_tool_call()
        agent_modes._note_tool_call()

    with caplog.at_level(logging.WARNING):
        contextvars.copy_context().run(scenario)

    assert "search_rooms MCP tool called 2 times" in caplog.text


def test_tool_call_guard_silent_on_single_call(caplog):
    def scenario():
        agent_modes._start_turn_budget()
        agent_modes._note_tool_call()

    with caplog.at_level(logging.WARNING):
        contextvars.copy_context().run(scenario)

    assert "search_rooms MCP tool called" not in caplog.text


# ---------------- Timeouts ----------------

def test_llm_timeout_constant_exists():
    from app.experiments.agent_modes import LLM_CALL_TIMEOUT_SECONDS

    assert LLM_CALL_TIMEOUT_SECONDS == 30


def test_mcp_timeout_constant_exists():
    from app.server import MCP_TOOL_TIMEOUT_SECONDS

    assert MCP_TOOL_TIMEOUT_SECONDS == 15


class _HangingClient:
    async def call_tool(self, name, arguments):
        await asyncio.sleep(60)


class _TimeoutClient:
    async def call_tool(self, name, arguments):
        raise asyncio.TimeoutError()


def test_mcp_tool_times_out_gracefully(caplog):
    # The wait_for wrapper itself: a hung call is cut off and logged.
    with patch.object(agent_modes, "MCP_TOOL_TIMEOUT_SECONDS", 0.05):
        with caplog.at_level(logging.WARNING):
            try:
                asyncio.run(
                    agent_modes._call_search_rooms_with_timeout(
                        _HangingClient(), {"city": "Bangalore"}
                    )
                )
                raised = None
            except asyncio.TimeoutError as error:
                raised = error

    assert raised is not None
    assert "timed out" in caplog.text

    # End to end: process_message() does not raise; the user gets the
    # standard "temporarily unavailable" reply and no rooms.
    state = agent_modes.ConversationState()
    result = asyncio.run(
        agent_modes.process_message(
            "show me rooms in Bangalore under 15000",
            "fast",
            _TimeoutClient(),
            state,
        )
    )

    assert result["tool"] == "search_rooms"
    assert result["response"] == agent_modes.HOMILIVO_UNAVAILABLE_RESPONSE
    assert not result["rooms"]
    assert result["search_performed"] is False
