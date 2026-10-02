# Tests for the tiered Tier-B confidence rule (price slots need >= 0.65)
# and the clarification reply for genuinely unmatched follow-ups.

from __future__ import annotations

import pytest

from app.experiments.agent_modes import (
    SHOW_ALTERNATIVES_RESPONSE,
    detect_room_operation,
    generate_fast_followup,
)
from app.experiments.interpreter import classify_operation

ROOMS = [
    {"id": "r1", "title": "Budget Single Room", "pricePerMonth": 7000,
     "roomType": "Single", "location": "HSR Layout, Bangalore"},
    {"id": "r2", "title": "3 BHK Flat", "pricePerMonth": 30000,
     "roomType": "3BHK", "location": "Indiranagar, Bangalore"},
]

FALSE_POSITIVES = [
    "ye wala nahi chahiye",
    "i dont understand",
    "what do you mean",
    "kuch aur dikhao",
    "dusra dikhao",
]


@pytest.mark.parametrize("query", FALSE_POSITIVES)
def test_false_positives_no_longer_resolve_to_price_operation(query):
    assert detect_room_operation(query) not in {"HIGHEST_PRICE", "LOWEST_PRICE"}


@pytest.mark.parametrize("query", ["kuch aur dikhao", "dusra dikhao",
                                   "ye wala nahi chahiye", "i dont understand"])
def test_false_positives_are_rejected_by_tier_b(query):
    assert classify_operation(query) is None


@pytest.mark.parametrize("query,expected", [
    ("which one is cheapest", "LOWEST_PRICE"),
    ("most expensive one", "HIGHEST_PRICE"),
    ("sabse sasta", "LOWEST_PRICE"),
    ("sabse kam", "LOWEST_PRICE"),
    ("sabse kam wala", "LOWEST_PRICE"),
    ("mehanga wala", "HIGHEST_PRICE"),
    ("expensive one", "HIGHEST_PRICE"),
])
def test_true_positives_still_work(query, expected):
    assert detect_room_operation(query) == expected


@pytest.mark.parametrize("query,expected", [
    ("which one is easiest on my budget", "LOWEST_PRICE"),
    ("lightest on my pocket", "LOWEST_PRICE"),
    ("which one would set me back the least", "LOWEST_PRICE"),
    ("show me the most budget friendly option", "LOWEST_PRICE"),
    ("which one hits my wallet the hardest", "HIGHEST_PRICE"),
    ("which will burn the biggest hole in my pocket", "HIGHEST_PRICE"),
    ("which is the biggest expense", "HIGHEST_PRICE"),
])
def test_tier_b_price_paraphrases_clear_price_floor(query, expected):
    assert classify_operation(query) == expected


def test_unmatched_followup_gets_clarification_reply():
    reply = generate_fast_followup("i dont understand", ROOMS, None)
    assert "I'm not sure what you're asking" in reply
    assert "most expensive" not in reply


def test_show_alternatives_followup_gets_honest_reply():
    reply = generate_fast_followup("kuch aur dikhao", ROOMS, None)
    assert reply == SHOW_ALTERNATIVES_RESPONSE


def test_matched_followup_is_not_clarification():
    reply = generate_fast_followup("which one is cheapest", ROOMS, None)
    assert "cheapest option" in reply


# ---------------------------------------------------------------------------
# "Show alternatives" routing: reaches the follow-up route (trusted
# contextual allowlist) instead of UNKNOWN -> OUT_OF_SCOPE refusal.
# ---------------------------------------------------------------------------

import asyncio

from app.experiments.agent_modes import (
    MESSAGE_FOLLOWUP,
    MESSAGE_UNKNOWN,
    MODE_FAST,
    MODE_GEMMA,
    MODE_GROQ,
    MODE_QWEN,
    OUT_OF_SCOPE_RESPONSE,
    ConversationState,
    process_message,
    route_message,
)

CLARIFICATION = SHOW_ALTERNATIVES_RESPONSE


def _state_with_rooms(selected=False):
    return ConversationState(
        previous_rooms=list(ROOMS),
        previous_search={"city": "Bangalore"},
        selected_room=ROOMS[0] if selected else None,
        selected_room_id=ROOMS[0]["id"] if selected else None,
    )


def test_kuch_aur_dikhao_routes_as_followup():
    route = route_message("kuch aur dikhao", _state_with_rooms())
    assert route.message_type == MESSAGE_FOLLOWUP


def test_something_else_routes_as_followup():
    route = route_message("something else", _state_with_rooms())
    assert route.message_type == MESSAGE_FOLLOWUP


@pytest.mark.parametrize("query", ["kuch aur dikhao", "something else"])
def test_show_alternatives_without_context_stays_unknown(query):
    route = route_message(query, ConversationState())
    assert route.message_type == MESSAGE_UNKNOWN


def _run(query, state, mode=MODE_FAST):
    return asyncio.run(process_message(query, mode, None, state))


def test_end_to_end_no_wrong_price_answer():
    result = _run("kuch aur dikhao", _state_with_rooms())
    assert CLARIFICATION in result["response"]
    assert result["response"] != OUT_OF_SCOPE_RESPONSE
    assert "₹" not in result["response"]


def test_clarification_wins_over_stale_selection():
    state = _state_with_rooms(selected=True)
    result = _run("kuch aur dikhao", state)
    assert CLARIFICATION in result["response"]
    assert "₹" not in result["response"]
    assert "Budget Single Room" not in result["response"]


@pytest.mark.parametrize("mode", [MODE_FAST, MODE_QWEN, MODE_GEMMA, MODE_GROQ])
@pytest.mark.parametrize("selected", [False, True])
def test_show_alternatives_short_circuits_before_any_llm(
    monkeypatch, mode, selected
):
    def _forbidden(*args, **kwargs):
        raise AssertionError("LLM generator called for show-alternatives")

    for name in ("qwen", "gemma", "groq"):
        monkeypatch.setattr(
            f"app.experiments.agent_modes.generate_{name}_response", _forbidden
        )

    result = _run("kuch aur dikhao", _state_with_rooms(selected), mode)
    assert result["response"] == SHOW_ALTERNATIVES_RESPONSE
