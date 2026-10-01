"""
Regression tests for:
- short acknowledgments ("okey", "achha", ...) getting a neutral reply,
  never the out-of-scope refusal, a greeting, or a hijacked price answer
- bare "expensive one" resolving to the most expensive room instead of a
  stale selected_room from an earlier "cheapest" turn
"""

from __future__ import annotations

import asyncio

import pytest

from app.experiments.agent_modes import (
    ACKNOWLEDGMENT_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
    ConversationState,
    detect_room_operation,
    process_message,
    resolve_room_reference,
)
from app.guardrails.input_guardrail import validate_user_input

ROOMS = [
    {"id": "r1", "title": "1 BHK Flat", "pricePerMonth": 7000, "city": "Bangalore"},
    {"id": "r2", "title": "2 BHK Flat", "pricePerMonth": 15000, "city": "Bangalore"},
    {"id": "r3", "title": "Studio", "pricePerMonth": 10000, "city": "Bangalore"},
]

ACKS = ["okey", "ok", "got it", "hmm", "achha", "theek hai", "alright", "cool"]


def _state(selected=None):
    state = ConversationState()
    state.previous_rooms = list(ROOMS)
    state.previous_search = {"city": "Bangalore"}
    state.selected_room = selected
    state.selected_room_id = selected["id"] if selected else None
    return state


def _run(message, state):
    return asyncio.run(process_message(message, "fast", None, state))


@pytest.mark.parametrize("message", ACKS)
def test_guardrail_accepts_acknowledgment(message):
    allowed, label = validate_user_input(message)
    assert allowed and label == "ACKNOWLEDGMENT"


@pytest.mark.parametrize("message", ACKS)
def test_ack_without_context_gets_neutral_reply(message):
    result = _run(message, ConversationState())
    assert result["response"] is not None
    assert result["response"] != OUT_OF_SCOPE_RESPONSE
    assert not result["response"].startswith("Hello")


@pytest.mark.parametrize("message", ACKS)
def test_ack_with_room_context_is_not_hijacked(message):
    state = _state(selected=ROOMS[0])
    result = _run(message, state)
    assert result["response"] == ACKNOWLEDGMENT_RESPONSE or "welcome" in result["response"].lower() or "let me know" in result["response"].lower()
    assert "₹" not in result["response"]
    assert result["selected_room"] == ROOMS[0]


def test_pending_clarification_is_not_treated_as_ack():
    state = ConversationState()
    state.pending_search = {"city": None}
    from app.experiments.agent_modes import is_conversational_acknowledgment

    assert not is_conversational_acknowledgment("sure", "ACKNOWLEDGMENT", state)


@pytest.mark.parametrize(
    "message",
    [
        "expensive one",
        "tell me about expensive one",
        "the expensive one",
        "costliest one",
        "most expensive one",
        "sabse mehanga wala",
        "mehanga wala",
    ],
)
def test_expensive_phrasings_are_highest_price(message):
    assert detect_room_operation(message) == "HIGHEST_PRICE"


@pytest.mark.parametrize(
    "message",
    [
        "is this one expensive?",
        "that is too expensive",
        "is it more expensive than the first one",
        "less expensive one",
    ],
)
def test_pairwise_or_selected_room_questions_are_not_highest(message):
    assert detect_room_operation(message) != "HIGHEST_PRICE"


def test_expensive_one_resolves_over_stale_cheapest_selection():
    resolved = resolve_room_reference("expensive one", ROOMS, ROOMS[0])
    assert resolved["id"] == "r2"


def test_cheapest_then_expensive_one_returns_actual_most_expensive():
    state = _state()
    first = _run("cheapest", state)
    assert "7,000" in first["response"]
    state.selected_room = first["selected_room"]
    state.selected_room_id = first["selected_room_id"]

    for message in ("tell me about expensive one", "expensive one"):
        result = _run(message, state)
        assert "15,000" in result["response"], result["response"]
        assert "7,000" not in result["response"]
        assert result["selected_room"]["id"] == "r2"
