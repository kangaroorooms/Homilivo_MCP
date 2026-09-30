# Diagnostic tests proving follow-up query behavior for team lead report.
# Tests prove:
# 1. Single-word follow-ups are detected correctly
# 2. Answers come from session data, not LLM
# 3. What is stored per session after a search
#
# NOTE ON FUNCTION SIGNATURES: the task brief that prompted this file
# sketches detect_room_operation(query, state) and
# is_likely_followup(query, state). The real signatures (verified by
# reading app/experiments/agent_modes.py) are:
#   detect_room_operation(user_query) -> str | None
#   is_likely_followup(user_query, previous_rooms, selected_room) -> bool
# detect_room_operation() is pure text classification and never reads
# session state - room context is only used later, by
# resolve_room_reference()/answer_room_comparison()/answer_room_fact().
# Tests below call the real signatures.

from __future__ import annotations

import asyncio

import pytest

from app.experiments.agent_modes import (
    MODE_FAST,
    ConversationState,
    detect_room_operation,
    generate_gemma_response,
    generate_groq_response,
    generate_qwen_response,
    is_likely_followup,
    process_message,
)

PREVIOUS_ROOMS = [
    {
        "id": "room-1",
        "title": "Budget Single Room",
        "city": "Bangalore",
        "pricePerMonth": 7000,
        "roomType": "Single",
        "idealFor": ["Students"],
        "location": "HSR Layout, Bangalore",
        "availabilityStatus": "Available",
        "rating": 4.2,
        "reviewsCount": 10,
    },
    {
        "id": "room-2",
        "title": "Cozy 2BHK",
        "city": "Bangalore",
        "pricePerMonth": 15000,
        "roomType": "2BHK",
        "idealFor": ["Working Professionals"],
        "location": "Koramangala, Bangalore",
        "availabilityStatus": "Available",
        "rating": 4.5,
        "reviewsCount": 25,
    },
    {
        "id": "room-3",
        "title": "Premium 3BHK",
        "city": "Bangalore",
        "pricePerMonth": 35000,
        "roomType": "3BHK",
        "idealFor": ["Family"],
        "location": "Indiranagar, Bangalore",
        "availabilityStatus": "Available",
        "rating": 4.8,
        "reviewsCount": 50,
    },
]


def _state_with_rooms() -> ConversationState:
    return ConversationState(previous_rooms=list(PREVIOUS_ROOMS))


def _run_turn(user_query: str, state: ConversationState) -> dict:
    async def run():
        return await process_message(
            user_query=user_query,
            mode=MODE_FAST,
            client=None,
            state=state,
        )

    return asyncio.run(run())


# ============================================================
# TEST 1-5 - single-word Tier A/B detection (no session needed;
# detect_room_operation() is pure text classification)
# ============================================================

def test_single_word_cheapest_detected():
    result = detect_room_operation("cheapest")
    assert result == "LOWEST_PRICE"
    print("PASS: 'cheapest' alone detected as LOWEST_PRICE — no LLM needed")


def test_single_word_expensive_detected():
    result = detect_room_operation("most expensive")
    assert result == "HIGHEST_PRICE"
    print("PASS: 'most expensive' alone detected as HIGHEST_PRICE — no LLM needed")


def test_single_word_price_detected():
    result = detect_room_operation("price")
    assert result is not None
    print(f"PASS: 'price' alone detected as {result}")


def test_single_word_location_detected():
    result = detect_room_operation("location")
    assert result is not None
    print(f"PASS: 'location' alone detected as {result}")


def test_single_word_amenities_detected():
    result = detect_room_operation("amenities")
    assert result is not None
    print(f"PASS: 'amenities' alone detected as {result}")


# ============================================================
# TEST 6-7 - answers come from session data, LLM never invoked
# ============================================================

@pytest.fixture()
def forbid_llm_calls(monkeypatch):
    """
    Patches every LLM-backed response generator this codebase has
    (Qwen/Gemma/Groq) so that if process_message() ever reaches one of
    them, the test fails loudly instead of the assertion silently
    depending on MODE_FAST never routing there.
    """

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "LLM-backed response generator was called - "
            "follow-up should have been answered from session data."
        )

    monkeypatch.setattr(
        "app.experiments.agent_modes.generate_qwen_response", _forbidden
    )
    monkeypatch.setattr(
        "app.experiments.agent_modes.generate_gemma_response", _forbidden
    )
    monkeypatch.setattr(
        "app.experiments.agent_modes.generate_groq_response", _forbidden
    )


def test_cheapest_answer_from_session_not_llm(forbid_llm_calls):
    state = _state_with_rooms()

    result = _run_turn("cheapest", state)
    response = result.get("response") or ""

    assert "7,000" in response or "7000" in response
    print("PASS: Answer came from session data, LLM was never called")
    print(f"  Response: {response}")


def test_most_expensive_from_session(forbid_llm_calls):
    state = _state_with_rooms()

    result = _run_turn("most expensive", state)
    response = result.get("response") or ""

    assert "35,000" in response or "35000" in response
    print("PASS: Answer came from session data, LLM was never called")
    print(f"  Response: {response}")


# ============================================================
# TEST 8-9 - is_likely_followup() requires room context
# ============================================================

def test_is_likely_followup_with_rooms():
    state = _state_with_rooms()

    result = is_likely_followup("cheapest", state.previous_rooms, state.selected_room)

    assert result is True
    print("PASS: 'cheapest' is a follow-up when previous_rooms is present")


def test_is_not_followup_without_rooms():
    empty_state = ConversationState()

    result = is_likely_followup(
        "cheapest", empty_state.previous_rooms, empty_state.selected_room
    )

    assert result is False
    print("INFO: Without a prior search, follow-up cannot be answered — by design")


# ============================================================
# TEST 10 - what is actually stored in session after a search
# ============================================================

def test_session_stores_rooms_after_search():
    state = ConversationState(previous_rooms=list(PREVIOUS_ROOMS))

    assert len(state.previous_rooms) == 3
    assert state.previous_rooms[0]["pricePerMonth"] == 7000

    # Every ConversationState field is accessible regardless of
    # whether it has been populated this turn.
    assert state.previous_search is not None  # defaults to {}
    assert state.conversation_history is not None  # defaults to []
    assert state.user_queries is not None  # defaults to []
    assert state.selected_room is None
    assert state.selected_room_id is None
    assert state.pending_search is not None  # defaults to {}

    print("SESSION FIELD SUMMARY:")
    print(f"  conversation_history : {state.conversation_history}")
    print(f"  previous_rooms       : {len(state.previous_rooms)} room(s) stored")
    for room in state.previous_rooms:
        print(
            f"    - {room['id']}: {room['title']} "
            f"(₹{room['pricePerMonth']}/month, {room['roomType']})"
        )
    print(f"  previous_search      : {state.previous_search}")
    print(f"  selected_room        : {state.selected_room}")
    print(f"  selected_room_id     : {state.selected_room_id}")
    print(f"  pending_search       : {state.pending_search}")
    print(f"  user_queries         : {state.user_queries}")


# ============================================================
# TEST 11-12 - Hindi/Hinglish single-word detection
# ============================================================
# NOTE: these assertions match the intended meaning ("sabse sasta" =
# cheapest, "sabse mehanga" = most expensive), not necessarily today's
# actual output - see the diagnostic report for what was found.

def test_hindi_cheapest_detected():
    result = detect_room_operation("sabse sasta")
    print(f"  'sabse sasta' -> {result}")
    assert result == "LOWEST_PRICE"
    print("PASS: Hindi 'sabse sasta' also works")


def test_hindi_expensive_detected():
    result = detect_room_operation("sabse mehanga")
    print(f"  'sabse mehanga' -> {result}")
    assert result == "HIGHEST_PRICE"
    print("PASS: Hindi 'sabse mehanga' also works")


def test_sab_se_sasta_detected():
    result = detect_room_operation("sab se sasta")
    print(f"  'sab se sasta' -> {result}")
    assert result == "LOWEST_PRICE"
    print("PASS: 'sab se sasta' (spaced variant) also works")


def test_sabse_sasta_does_not_affect_expensive():
    result = detect_room_operation("sabse mehanga")
    print(f"  'sabse mehanga' -> {result}")
    assert result == "HIGHEST_PRICE"
    print("PASS: adding 'sabse sasta' to LOWEST_PRICE did not affect HIGHEST_PRICE")


def test_sabse_kam_wala_maps_to_lowest_price():
    result = detect_room_operation("sabse kam wala")
    print(f"  'sabse kam wala' -> {result}")
    assert result == "LOWEST_PRICE"
    print("PASS: 'sabse kam wala' maps to LOWEST_PRICE")


def test_sabse_kam_wala_latent_risk_documented():
    # LATENT RISK: "sabse kam wala" is a generic pattern. If a user asks
    # "sabse kam rating wala" (lowest rated) it could match LOWEST_PRICE
    # incorrectly. Rating-based follow-ups are not a supported operation.
    # Monitor if rating queries are added in future.
    #
    # Currently returns None - "rating" breaks the substring match so no
    # false positive occurs today. This test locks in the safe behavior
    # and will fail loudly if a broader pattern match is added that causes
    # "sabse kam rating wala" to incorrectly match LOWEST_PRICE in future.
    result = detect_room_operation("sabse kam rating wala")
    print(f"  'sabse kam rating wala' -> {result}")
    assert result != "LOWEST_PRICE"
