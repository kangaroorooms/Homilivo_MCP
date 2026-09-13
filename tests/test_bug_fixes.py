"""
Regression tests for two live-test bugs.

BUG 1 - the output guardrail's SENSITIVE_OUTPUT_PATTERNS (e.g.
\\bsecret\\b, \\bpassword\\b) are meant to catch the AI/system leaking
its OWN credentials in free-form generated text, not to censor
verified, deterministically-rendered room data (format_rooms_as_table()
in app/experiments/agent_modes.py) whose title/location/amenity text
happens to contain an ordinary English word that also matches one of
those patterns. app/guardrails/output_guardrail.py now skips
contains_sensitive_information()/contains_internal_information() only
for a response that is a structured markdown table (starts with "| ") -
every other response, and every other guardrail check on a table
(length, empty, control characters, excessive blank lines), is
unaffected.

BUG 2 - "I want to contact the owner" (and the rest of
_PROPERTY_INTEREST_PHRASES) also appears in _PLATFORM_KNOWLEDGE_PATTERNS
(for the generic, no-room-context "how do I contact an owner on this
platform" policy question), and is_likely_followup() had no knowledge
of property-interest phrasing at all. With active room context, that
meant route_message() lost the message to is_platform_knowledge_question()
before detect_property_interest() (called deep inside generate_response(),
only reachable via MESSAGE_FOLLOWUP) ever got a chance to run. is_likely_
followup() now also returns True when detect_property_interest() matches
and there is active room context - mirroring exactly how a non-None
detect_room_operation() result already causes it to return True.
"""

from __future__ import annotations

from app.experiments.agent_modes import (
    ConversationState,
    MESSAGE_FOLLOWUP,
    MESSAGE_PLATFORM_KNOWLEDGE,
    detect_room_operation,
    format_rooms_as_table,
    is_likely_followup,
    route_message,
)
from app.guardrails.output_guardrail import validate_ai_response


# ============================================================
# BUG 1 - table guardrail exemption is table-only
# ============================================================

def test_table_response_not_blocked_by_guardrail():
    rooms = [
        {
            "id": "room-1",
            "title": "The Secret Garden PG",
            "city": "Bangalore",
            "location": "Koramangala",
            "pricePerMonth": 18000,
            "roomType": "2BHK",
            "idealFor": ["Family"],
        },
    ]

    table = format_rooms_as_table(rooms)
    assert table is not None
    assert table.lstrip().startswith("| ")

    valid, message = validate_ai_response(table)

    assert valid is True
    assert message == "Response validated."


def test_non_table_with_secret_still_blocked():
    valid, message = validate_ai_response(
        "My secret plan is to leak the api key to everyone."
    )

    assert valid is False
    assert message == "The response contained potentially sensitive information."


# ============================================================
# BUG 2 - property interest treated as follow-up with room context
# ============================================================

def test_property_interest_treated_as_followup():
    state = ConversationState()
    state.previous_rooms = [
        {
            "id": "room-1",
            "title": "Sunrise 2BHK",
            "city": "Bangalore",
            "pricePerMonth": 18000,
        },
    ]
    state.previous_search = {"city": "Bangalore"}

    assert is_likely_followup(
        "I want to contact the owner",
        state.previous_rooms,
        state.selected_room,
    ) is True

    route = route_message("I want to contact the owner", state)

    assert route.message_type == MESSAGE_FOLLOWUP


def test_property_interest_without_context_not_followup():
    state = ConversationState()

    assert is_likely_followup(
        "I want to contact the owner",
        state.previous_rooms,
        state.selected_room,
    ) is False

    route = route_message("I want to contact the owner", state)

    # No active room context - this is the generic platform-knowledge
    # question, unchanged from before this fix.
    assert route.message_type == MESSAGE_PLATFORM_KNOWLEDGE


# ============================================================
# TABLE PHRASING GAP - "in table" without "a"
# ============================================================

def test_table_phrase_variant_routes_correctly():
    assert detect_room_operation("show me the results in table") == "ROOM_TABLE"
