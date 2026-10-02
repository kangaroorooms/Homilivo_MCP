"""Regression: clear new searches must not be hijacked as follow-ups/refinements."""

import pytest

from app.experiments.agent_modes import (
    ConversationState,
    MESSAGE_FOLLOWUP,
    MESSAGE_NEW_SEARCH,
    MESSAGE_UNKNOWN,
    route_message,
)


def _state():
    rooms = [
        {"id": str(i), "city": "Bangalore", "price": 8000 + i * 1000}
        for i in range(3)
    ]
    return ConversationState(
        previous_rooms=rooms, previous_search={"city": "Bangalore"}
    )


@pytest.mark.parametrize(
    "query",
    [
        "pg for working professionals",
        "show me rooms in mumbai",
        "i need a 2bhk in pune",
    ],
)
def test_clear_new_search_with_context(query):
    assert route_message(query, _state()).message_type == MESSAGE_NEW_SEARCH


@pytest.mark.parametrize("query", ["find me a room", "room room room room"])
def test_bare_room_message_is_unknown_with_context(query):
    assert route_message(query, _state()).message_type == MESSAGE_UNKNOWN


@pytest.mark.parametrize(
    "query",
    ["is it also available in koramangala", "is this room good for students"],
)
def test_real_followups_stay_followups(query):
    assert route_message(query, _state()).message_type == MESSAGE_FOLLOWUP
