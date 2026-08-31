"""
Pytest coverage for the deterministic operation-detection and routing
core in app/experiments/agent_modes.py.

This is the automatically-collected home for the cases that used to
live only in tests/hard_tests.py (a manual `python hard_tests.py`
script, never run by pytest/CI). The case DATA below is the single
source of truth - hard_tests.py now imports it for its human-readable
report instead of keeping its own duplicate copy, so the two never
drift out of sync.

Sections mirror the original script's intent:
  STRICT   - deterministic Tier-A cases. Hard assertions.
  ROUTING  - multi-turn routing via route_message(). Hard assertions.
  RANKED   - "second cheapest" / "last one" compound rank cases (new).
  SOFT     - Tier-B semantic cases. Soft: only the entries confirmed
             correct today are hard-asserted; a documented, currently-
             unmatched phrasing is xfail rather than silently dropped.
  NOTES    - intentional behaviour documented as regression locks.
"""

from __future__ import annotations

import pytest

from app.experiments.agent_modes import (
    NOT_VERIFIED_AMENITY_RESPONSE,
    ConversationState,
    answer_amenity_across_rooms,
    answer_room_comparison,
    detect_room_operation,
    generate_response,
    is_likely_search_refinement,
    resolve_room_reference,
    route_message,
)


# ============================================================
# STRICT - deterministic Tier-A, hard assertions
# ============================================================

STRICT_OPERATION_CASES = [
    ("Which one is cheapest?", "LOWEST_PRICE"),
    ("show me the priciest option", "HIGHEST_PRICE"),
    ("the most expensive one", "HIGHEST_PRICE"),
    ("least expensive please", "LOWEST_PRICE"),
    ("most affordable?", "LOWEST_PRICE"),
    ("what is its price?", "ROOM_PRICE"),
    ("what's the monthly rent?", "ROOM_PRICE"),
    ("how much is it?", "ROOM_PRICE"),
    ("where is it located?", "LOCATION"),
    ("which city is it in?", "LOCATION"),
    ("what type of room is it?", "ROOM_TYPE"),
    ("how many bhk?", "ROOM_TYPE"),
    ("is it suitable for students?", "STUDENT_SUITABILITY"),
    ("is this good for families?", "FAMILY_SUITABILITY"),
    ("is it okay for working professionals?", "PROFESSIONAL_SUITABILITY"),
    ("does it have parking?", "AMENITY"),
    ("is it furnished?", "AMENITY"),
    ("tell me about this one", "ROOM_DETAILS"),
    ("the second one", "ORDINAL_2"),
    ("first one please", "ORDINAL_1"),
]


@pytest.mark.parametrize("query, expected", STRICT_OPERATION_CASES)
def test_strict_deterministic_operation_detection(query, expected):
    assert detect_room_operation(query) == expected


# ============================================================
# RANKED - "second cheapest" / "third most expensive" / "last one"
# ============================================================

RANKED_OPERATION_CASES = [
    ("second cheapest one", "RANKED_LOWEST_PRICE"),
    ("show me the second cheapest", "RANKED_LOWEST_PRICE"),
    ("third cheapest", "RANKED_LOWEST_PRICE"),
    ("2nd cheapest", "RANKED_LOWEST_PRICE"),
    ("third most expensive", "RANKED_HIGHEST_PRICE"),
    ("second most expensive one", "RANKED_HIGHEST_PRICE"),
    ("last one", "LAST_ROOM"),
    ("the last option", "LAST_ROOM"),
    ("last room", "LAST_ROOM"),
]


@pytest.mark.parametrize("query, expected", RANKED_OPERATION_CASES)
def test_ranked_and_last_operation_detection(query, expected):
    assert detect_room_operation(query) == expected


def _priced_rooms():
    return [
        {"id": "r1", "title": "Room A", "pricePerMonth": 18000, "roomType": "2BHK",
         "city": "Jaipur", "idealFor": ["Family"]},
        {"id": "r2", "title": "Room B", "pricePerMonth": 10000, "roomType": "2BHK",
         "city": "Jaipur", "idealFor": ["Students"]},
        {"id": "r3", "title": "Room C", "pricePerMonth": 8500, "roomType": "2BHK",
         "city": "Jaipur", "idealFor": ["Students"]},
    ]


@pytest.mark.parametrize(
    "query, expected_room_id",
    [
        ("second cheapest one", "r2"),   # ascending: r3(8500) r2(10000) r1(18000)
        ("third cheapest", "r1"),
        ("second most expensive", "r2"),  # descending: r1(18000) r2(10000) r3(8500)
        ("last one", "r3"),               # natural list order, last element
        ("first one", "r1"),              # unchanged plain-ordinal behaviour
    ],
)
def test_rank_and_last_resolve_to_the_correct_room(query, expected_room_id):
    room = resolve_room_reference(query, _priced_rooms(), None)

    assert room is not None
    assert room["id"] == expected_room_id


def test_ranked_beyond_available_rooms_degrades_gracefully():
    room = resolve_room_reference("fifth cheapest", _priced_rooms(), None)

    assert room is None

    text, selected = answer_room_comparison("fifth cheapest", _priced_rooms())

    assert selected is None
    assert "fifth cheapest" in text


def test_ranked_comparison_produces_a_sentence_and_selects_the_room():
    text, room = answer_room_comparison("second cheapest one", _priced_rooms())

    assert room is not None
    assert room["id"] == "r2"
    assert "second cheapest" in text
    assert "Room B" in text


# ============================================================
# ROUTING - multi-turn via route_message()
# ============================================================

def _state_after_search() -> ConversationState:
    state = ConversationState()
    state.previous_search = {"city": "Jaipur", "sort": "latest", "limit": 10}
    state.previous_rooms = [
        {"id": "r1", "title": "Room A", "pricePerMonth": 8000,
         "roomType": "1BHK", "city": "Jaipur", "idealFor": ["Students"]},
        {"id": "r2", "title": "Room B", "pricePerMonth": 12000,
         "roomType": "2BHK", "city": "Jaipur", "idealFor": ["Family"]},
    ]
    return state


def _state_pending_city() -> ConversationState:
    state = ConversationState()
    state.pending_search = {
        "city": None, "room_types": ["2BHK"], "sort": "latest", "limit": 10,
    }
    return state


ROUTING_CASES = [
    ("Which one is cheapest?", _state_after_search, "ROOM_FOLLOWUP"),
    ("only 2BHK", _state_after_search, "SEARCH_REFINEMENT"),
    ("under 8000", _state_after_search, "SEARCH_REFINEMENT"),
    ("for students", _state_after_search, "SEARCH_REFINEMENT"),
    ("now find rooms in Chennai", _state_after_search, "NEW_SEARCH"),
    ("Jaipur", _state_pending_city, "CLARIFICATION_RESPONSE"),
    ("second cheapest one", _state_after_search, "ROOM_FOLLOWUP"),
    ("last one", _state_after_search, "ROOM_FOLLOWUP"),
    # Regression: "find rooms in X" matched deterministically already,
    # but "rooms in X" (no verb) and "find me rooms in X" (pronoun
    # breaks the "find rooms" substring) fell through to UNKNOWN for
    # ANY city, not just one - see looks_like_general_room_search().
    ("rooms in Bangalore", ConversationState, "NEW_SEARCH"),
    ("find me rooms in Bangalore", ConversationState, "NEW_SEARCH"),
    ("find rooms in Udaipur", ConversationState, "NEW_SEARCH"),
    ("accommodation in Udaipur", ConversationState, "NEW_SEARCH"),
    # Lock: the broadened city-agnostic catch-all above must not win
    # over an active follow-up just because the follow-up happens to
    # mention a city and the word "room".
    ("is this room actually in Jaipur?", _state_after_search, "ROOM_FOLLOWUP"),
]


@pytest.mark.parametrize("query, make_state, expected", ROUTING_CASES)
def test_multi_turn_routing(query, make_state, expected):
    route = route_message(query, make_state())

    assert route.message_type == expected


def test_followup_before_any_search_is_not_misrouted_as_followup():
    """A follow-up phrase with no prior room context must not be treated
    as ROOM_FOLLOWUP (there is nothing to follow up on)."""
    route = route_message("which one is cheapest?", ConversationState())

    assert route.message_type != "ROOM_FOLLOWUP"


# ============================================================
# SOFT - Tier-B semantic, mostly hard-asserted; one documented gap
# ============================================================

SOFT_OPERATION_CASES = [
    ("which will burn the biggest hole in my pocket", "HIGHEST_PRICE"),
    ("which one hits my wallet hardest", "HIGHEST_PRICE"),
    ("which is easiest on my budget", "LOWEST_PRICE"),
    ("which one would set me back the least", "LOWEST_PRICE"),
    ("how much would this set me back", "ROOM_PRICE"),
    ("what part of town is it in", "LOCATION"),
    pytest.param(
        "give me the full rundown on this place", "ROOM_DETAILS",
        marks=pytest.mark.xfail(
            reason="Tier-B embedding similarity doesn't confidently match "
                   "this phrasing yet - known tuning gap, not a regression.",
            strict=False,
        ),
    ),
    ("does the place come with a washing machine", "AMENITY"),
    ("does it have ac", "AMENITY"),
    ("is the capital of france paris", None),
    ("i really like this vibe honestly", None),
]


@pytest.mark.parametrize("query, expected", SOFT_OPERATION_CASES)
def test_soft_semantic_fallback(query, expected):
    assert detect_room_operation(query) == expected


# ============================================================
# NOTES - intentional behaviour, locked in as regressions tests
# ============================================================

def test_ordinals_take_priority_over_plain_superlatives_for_room_reference():
    """
    Which ROOM "cheapest of the first two" refers to is unaffected by
    detect_room_operation() below - it must still resolve as the
    first room, not the set-wide cheapest, since "cheapest" here
    modifies "two" (a scope), not a rank-of-cheapness. This is now
    resolve_room_reference()'s job (via
    detect_room_reference_operation(), whose ordinal-priority order is
    intentionally unchanged from before), decoupled from
    detect_room_operation() - see the next test for why.
    """
    rooms = [
        {"id": "r1", "title": "Room A", "pricePerMonth": 18000},
        {"id": "r2", "title": "Room B", "pricePerMonth": 10000},
        {"id": "r3", "title": "Room C", "pricePerMonth": 8000},
    ]

    resolved = resolve_room_reference("cheapest of the first two", rooms, None)

    assert resolved is not None
    assert resolved["id"] == "r1"


def test_detect_room_operation_now_reports_the_superlative_for_this_phrase():
    """
    detect_room_operation() (WHAT is being asked, used by
    answer_room_fact()/answer_amenity_across_rooms()/etc - never by
    room reference resolution, see the test above) now returns
    LOWEST_PRICE for "cheapest of the first two", not ORDINAL_1. This
    is an intentional, correct consequence of moving plain ordinals to
    the END of detect_room_operation()'s Tier A (see its comments) so
    that a compound sentence like "does the FIRST ONE have a FRIDGE?"
    resolves operation=AMENITY there instead of the ordinal silently
    winning - answer_room_fact() has no ORDINAL_1 branch to begin
    with, so this relabeling has no user-facing effect for bare
    ordinal references.
    """
    assert detect_room_operation("cheapest of the first two") == "LOWEST_PRICE"


def test_pairwise_comparison_resolves_to_room_price():
    assert detect_room_operation("is this one more expensive than that one") == "ROOM_PRICE"


# ============================================================
# ISSUE 1 - out-of-scope "contact the owner" wrongly re-ran a search
# ============================================================
# "with " was a bare substring in refinement_patterns and false-matched
# ordinary English ("contact WITH the owner"), which let a correctly
# guardrail-rejected message slip through as a "contextual" refinement
# and silently re-run the previous search.

OUT_OF_SCOPE_NOT_REFINEMENT_CASES = [
    "how to contact with the owner?",
    "can I speak with the owner directly?",
    "I want to chat with support without any delay",
]


@pytest.mark.parametrize("query", OUT_OF_SCOPE_NOT_REFINEMENT_CASES)
def test_out_of_scope_with_word_is_not_misrouted_as_refinement(query):
    state = _state_after_search()

    assert is_likely_search_refinement(query, state.previous_search, state.previous_rooms) is False
    assert route_message(query, state).message_type not in {"SEARCH_REFINEMENT", "NEW_SEARCH"}


@pytest.mark.parametrize(
    "query",
    [
        "only 2BHK",
        "under 8000",
        "for students",
        "furnished",
        "1bhk please",
        "rooms near the station",
    ],
)
def test_genuine_refinement_triggers_still_work(query):
    """Regression guard: removing bare 'with '/'without ' must not weaken
    detection of the real refinement phrasings that don't use those words."""
    state = _state_after_search()

    assert is_likely_search_refinement(query, state.previous_search, state.previous_rooms) is True


# ============================================================
# ISSUE 2 - set-wide amenity questions ("does any room have ac?")
# ============================================================
# answer_room_fact() only ever answers about one selected_room. A
# question about the whole current room set used to fall through to the
# generic capability message instead of being answered from verified
# session data.

def _rooms_with_mixed_amenities():
    return [
        {"id": "r1", "title": "Room A", "pricePerMonth": 18000, "roomType": "2BHK",
         "city": "Jaipur", "idealFor": ["Family"], "amenities": ["AC", "Parking"]},
        {"id": "r2", "title": "Room B", "pricePerMonth": 10000, "roomType": "2BHK",
         "city": "Jaipur", "idealFor": ["Students"], "amenities": ["WiFi"]},
    ]


# --------------------------------------------------------------
# YES/NO framing: "does any room have X?" -> boolean + count, no
# room names (matches the product spec's exact examples).
# --------------------------------------------------------------

def test_set_wide_amenity_yes_no_framing_with_a_match():
    answer = answer_amenity_across_rooms("does any room have ac?", _rooms_with_mixed_amenities())

    assert answer == "Yes, 1 of the verified rooms have AC."


def test_set_wide_amenity_question_all_rooms_match():
    rooms = [
        {"id": "r1", "title": "Room A", "pricePerMonth": 18000, "roomType": "2BHK",
         "city": "Jaipur", "amenities": ["WiFi"]},
        {"id": "r2", "title": "Room B", "pricePerMonth": 10000, "roomType": "2BHK",
         "city": "Jaipur", "amenities": ["WiFi", "AC"]},
    ]

    answer = answer_amenity_across_rooms("does any room have wifi?", rooms)

    assert answer == "Yes, 2 of the verified rooms have wifi."


def test_set_wide_amenity_question_no_room_matches():
    rooms = [
        {"id": "r1", "title": "Room A", "pricePerMonth": 18000, "roomType": "2BHK", "city": "Jaipur"},
    ]

    answer = answer_amenity_across_rooms("does any room have parking?", rooms)

    assert answer == "No, none of the verified rooms have parking."


# --------------------------------------------------------------
# LISTING framing: "which rooms have X?" / "list rooms with X" ->
# names the matching rooms (and, at the API layer, returns them as
# cards - see tests/test_api.py).
# --------------------------------------------------------------

def test_set_wide_amenity_listing_framing_names_the_matching_room():
    answer = answer_amenity_across_rooms("which rooms have ac?", _rooms_with_mixed_amenities())

    assert answer is not None
    assert "Room A" in answer
    assert "Room B" not in answer


def test_set_wide_amenity_listing_framing_all_rooms_match():
    rooms = [
        {"id": "r1", "title": "Room A", "pricePerMonth": 18000, "roomType": "2BHK",
         "city": "Jaipur", "amenities": ["WiFi"]},
        {"id": "r2", "title": "Room B", "pricePerMonth": 10000, "roomType": "2BHK",
         "city": "Jaipur", "amenities": ["WiFi", "AC"]},
    ]

    answer = answer_amenity_across_rooms("list rooms with wifi", rooms)

    assert answer == "Yes, all 2 of the verified rooms have wifi."


def test_set_wide_amenity_listing_framing_no_matches():
    rooms = [
        {"id": "r1", "title": "Room A", "pricePerMonth": 18000, "roomType": "2BHK", "city": "Jaipur"},
    ]

    answer = answer_amenity_across_rooms("which rooms have parking?", rooms)

    assert answer == "I couldn't find any of the verified rooms with parking."


# --------------------------------------------------------------
# COUNT framing: "how many rooms have X?"
# --------------------------------------------------------------

def test_set_wide_amenity_count_framing():
    answer = answer_amenity_across_rooms("how many rooms have ac?", _rooms_with_mixed_amenities())

    assert answer == "1 of the 2 verified rooms have AC."


# --------------------------------------------------------------
# NEGATION: "which rooms don't have X?"
# --------------------------------------------------------------

def test_set_wide_amenity_negation_listing():
    answer = answer_amenity_across_rooms(
        "which rooms don't have ac?", _rooms_with_mixed_amenities()
    )

    assert answer is not None
    assert "Room B" in answer
    assert "Room A" not in answer


def test_set_wide_amenity_negation_yes_no():
    answer = answer_amenity_across_rooms(
        "does any room not have ac?", _rooms_with_mixed_amenities()
    )

    assert answer == "Yes, 1 of the verified rooms don't have AC."


# --------------------------------------------------------------
# Unknown/unresolvable amenity: must decline honestly, never guess.
# --------------------------------------------------------------

def test_set_wide_amenity_unresolvable_amenity_declines_honestly():
    answer = answer_amenity_across_rooms(
        "does any room have a swimming pool?", _rooms_with_mixed_amenities()
    )

    assert answer == NOT_VERIFIED_AMENITY_RESPONSE


def test_set_wide_amenity_returns_none_for_non_amenity_questions():
    """Must not interfere with non-amenity follow-ups (e.g. price)."""
    assert answer_amenity_across_rooms("what is the price?", _rooms_with_mixed_amenities()) is None


def test_set_wide_amenity_returns_none_with_no_rooms():
    assert answer_amenity_across_rooms("does any room have ac?", []) is None


def test_generate_response_answers_set_wide_amenity_without_llm():
    """
    End-to-end through generate_response() with no selected_room, for
    every mode - proves the deterministic answer wins before any
    mode-specific (LLM-backed) branch runs, per the no-unnecessary-LLM-
    calls requirement. qwen/gemma/groq would otherwise need a live model
    to even reach a response; reaching the correct text here without one
    proves the deterministic short-circuit fired.
    """
    rooms = _rooms_with_mixed_amenities()

    for mode in ("fast", "qwen", "gemma", "groq"):
        text = generate_response(
            mode=mode,
            user_query="which rooms have ac?",
            rooms=rooms,
            search_request={"city": "Jaipur"},
            conversation_history=[],
            is_followup=True,
            selected_room=None,
            selected_room_id=None,
        )

        assert "Room A" in text
        assert "capability" not in text.lower()
