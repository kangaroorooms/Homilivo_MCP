"""Amenity vocabulary (AC, laundry), honest amenity search replies, the rule-based fallback's amenity and the
provider field of the turn log. Mocks only: a scripted model and an offline rooms fixture, no Gemini call."""

from __future__ import annotations

import json
import logging

import pytest

from llm_approach import fallback, responder, validators
from llm_approach.pipeline import AMENITY_NOT_FILTERED
from llm_approach.state import LLMConversationState
from llm_approach.tests.conftest import dec

# Live Jaipur rooms' amenity lists; none lists AC.
JAIPUR = [
    {"id": "j1", "title": "Pink City Studio", "city": "Jaipur", "location": "Malviya Nagar", "pricePerMonth": 8000,
     "roomType": "1BHK", "amenities": ["fan", "attached-bathroom", "furnished", "fridge", "water-purifier", "laundry"]},
    {"id": "j2", "title": "Vaishali Nest", "city": "Jaipur", "location": "Vaishali Nagar", "pricePerMonth": 9500,
     "roomType": "1BHK", "amenities": ["parking", "lighting"]},
    {"id": "j3", "title": "Mansarovar Room", "city": "Jaipur", "location": "Mansarovar", "pricePerMonth": 7000,
     "roomType": "1BHK", "amenities": ["semi-furnished"]},
]

# The reply to a search with no amenity, recorded before this change: it must not move.
NO_AMENITY_REPLY = (
    "I found 3 rooms in Jaipur:\n"
    "1. Pink City Studio, 1BHK, Malviya Nagar, ₹8,000/month\n"
    "2. Vaishali Nest, 1BHK, Vaishali Nagar, ₹9,500/month\n"
    "3. Mansarovar Room, 1BHK, Mansarovar, ₹7,000/month\n"
    "Ask me about any of them, for example the cheapest one, or tell me what to change."
)
NO_ROOMS_REPLY = (
    "I couldn't find any rooms in Jaipur matching that. "
    "You could try a higher budget, a different room type or another area."
)


@pytest.fixture
def jaipur(pipe, monkeypatch, tmp_path):
    path = tmp_path / "jaipur.json"
    path.write_text(json.dumps({"rooms": JAIPUR}), encoding="utf-8")
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", str(path))
    return pipe


@pytest.mark.parametrize(
    "raw", ["AC", "ac", "a/c", "A/C", "air conditioning", "Air Conditioning", "air-conditioning", "air conditioner", "aircon"]
)
def test_every_ac_alias_validates_to_ac(raw):
    assert validators.validate_amenity(raw) == "AC"


@pytest.mark.parametrize("raw", ["laundry", "Laundry"])
def test_laundry_validates(raw):
    assert validators.validate_amenity(raw) == "laundry"


@pytest.mark.parametrize(
    "raw", ["fan", "attached-bathroom", "furnished", "semi-furnished", "fridge", "water-purifier", "laundry", "parking", "lighting"]
)
def test_every_live_jaipur_amenity_is_in_the_vocabulary(raw):
    assert validators.validate_amenity(raw) == raw


def test_followup_says_no_room_lists_ac(jaipur):
    jaipur.turn("rooms in jaipur", dec("NEW_SEARCH", city="Jaipur"))
    result = jaipur.turn("does all of these rooms have AC?", dec("ROOM_FOLLOWUP", op="AMENITY", amenity="AC"))
    assert result.reply == "None of these rooms list AC."


def test_followup_on_one_room_with_laundry(jaipur):
    jaipur.turn("rooms in jaipur", dec("NEW_SEARCH", city="Jaipur"))
    result = jaipur.turn("does the first room have laundry?", dec("ROOM_FOLLOWUP", op="AMENITY", ref=1, amenity="laundry"))
    assert result.reply == "Room 1 (Pink City Studio) has laundry."


def test_amenity_search_names_the_amenity_in_the_header(jaipur):
    result = jaipur.turn("rooms in jaipur with parking", dec("NEW_SEARCH", city="Jaipur", amenity="parking"))
    assert result.reply.split("\n")[0] == "I found 1 room in Jaipur with parking:"
    assert [r["id"] for r in result.rooms] == ["j2"]


def test_zero_result_amenity_search_says_so(jaipur):
    result = jaipur.turn("rooms in jaipur with AC", dec("NEW_SEARCH", city="Jaipur", amenity="AC"))
    assert result.reply.startswith("I couldn't find any rooms in Jaipur that list AC.")
    assert result.rooms == []
    assert jaipur.searches[-1]["amenity"] == "AC"


def test_unvalidated_amenity_says_results_are_unfiltered(jaipur):
    result = jaipur.turn("rooms in jaipur with jacuzzi", dec("NEW_SEARCH", city="Jaipur", amenity="jacuzzi"))
    assert result.reply == NO_AMENITY_REPLY + " " + AMENITY_NOT_FILTERED
    assert "jacuzzi" not in result.reply


def test_search_without_amenity_reply_is_unchanged(jaipur):
    result = jaipur.turn("rooms in jaipur", dec("NEW_SEARCH", city="Jaipur"))
    assert result.reply == NO_AMENITY_REPLY


def test_zero_result_search_without_amenity_reply_is_unchanged(jaipur):
    result = jaipur.turn("2bhk in jaipur", dec("NEW_SEARCH", city="Jaipur", bhk=2))
    assert result.reply == NO_ROOMS_REPLY


def test_paged_search_without_amenity_still_shows_the_total():
    rooms = JAIPUR[:2]
    reply = responder.format_search_reply(rooms, {"city": "jaipur"}, total=24, has_next=True)
    assert reply.split("\n")[0] == "I found 2 rooms in Jaipur (showing 2 of 24):"


@pytest.mark.parametrize(
    "message,amenity",
    [
        ("rooms in jaipur with AC", "AC"),
        ("rooms in jaipur with a/c", "AC"),
        ("rooms in jaipur with air conditioning", "AC"),
        ("rooms in jaipur with laundry", "laundry"),
        ("rooms in jaipur with parking", "parking"),
    ],
)
def test_fallback_extracts_the_amenity(message, amenity):
    assert fallback.fallback_slots(message, LLMConversationState())["amenity"] == amenity


@pytest.mark.parametrize(
    "message",
    [
        "rooms in jaipur near Acme tower",  # "ac" inside a word
        "rooms in jaipur near the lighthouse",  # "light" inside a word
        "rooms in jaipur with a table",  # ambiguous alias
        "rooms in jaipur with good light",  # ambiguous alias
        "non-ac rooms in jaipur",
        "rooms in jaipur without AC",
    ],
)
def test_fallback_ignores_substrings_ambiguous_words_and_negations(message):
    assert fallback.fallback_slots(message, LLMConversationState())["amenity"] is None


@pytest.mark.parametrize("message", ["rooms in jaipur", "2bhk in jaipur under 10000 for students", "pg in pune"])
def test_fallback_without_amenity_word_gives_the_same_slots(message, monkeypatch):
    state = LLMConversationState()
    with monkeypatch.context() as patched:
        patched.setattr(fallback, "_amenity_from", lambda text: None)  # the slots as they were before this change
        before = fallback.fallback_slots(message, state)
    assert fallback.fallback_slots(message, state) == before


def test_turn_log_includes_the_provider(jaipur, caplog):
    with caplog.at_level(logging.INFO, logger="chatbot"):
        jaipur.turn("rooms in jaipur", dec("NEW_SEARCH", city="Jaipur"))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("turn request_id=")]
    assert lines and " mode=fast provider=gemini status=success " in lines[-1]


@pytest.mark.parametrize(
    "message,amenity",
    [
        ("rooms in jaipur with security deposit under 10000", None),
        ("rooms in jaipur with Security Deposit under 10000", None),
        ("rooms in jaipur with security amount 5000", None),
        ("rooms in jaipur with low security charges", None),
        ("rooms in jaipur with less security money", None),
        ("rooms in jaipur with security deposit and parking", "parking"),
        ("rooms in jaipur with security", "Security"),
        ("rooms in jaipur with 24x7 security", "Security"),
        ("rooms in jaipur with low deposit", None),
        ("rooms in jaipur with no advance", None),
    ],
)
def test_fallback_security_deposit_is_not_the_security_amenity(message, amenity):
    assert fallback.fallback_slots(message, LLMConversationState())["amenity"] == amenity


@pytest.mark.parametrize("message", ["deposit", "advance", "Deposit 5000 advance 2000", "how much advance and deposit"])
def test_bare_deposit_and_advance_never_set_an_amenity(message):
    assert fallback._amenity_from(message) is None
    assert validators.validate_amenity("deposit") is None and validators.validate_amenity("advance") is None
