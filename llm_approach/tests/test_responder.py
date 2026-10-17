"""Responder tests: card format and null handling."""

from __future__ import annotations

from llm_approach import responder

CARD_KEYS = {"id", "title", "city", "location", "price_per_month", "room_type", "ideal_for", "amenities"}


def test_card_format_from_fixture_room(rooms):
    card = responder.room_card(rooms[1])
    assert set(card) == CARD_KEYS
    assert card["id"] == "fixture-2" and card["price_per_month"] == 7500
    assert card["room_type"] == "Single" and card["location"] == "Koramangala"
    assert card["ideal_for"] == ["Students", "Working Professionals", "Family"]
    assert "balcony" in card["amenities"]


def test_null_location_and_room_type_stay_null():
    card = responder.room_card({"id": 5, "price": 9000})
    assert card["location"] is None and card["room_type"] is None
    assert card["price_per_month"] == 9000
    assert card["title"] == "Untitled Room" and card["city"] == "Unknown"
    assert card["ideal_for"] == [] and card["amenities"] == []
    assert card["id"] == "5"


def test_amenity_field_variants():
    assert responder.room_card({"amenity": "wifi"})["amenities"] == ["wifi"]
    assert responder.room_card({"facilities": ["gym"]})["amenities"] == ["gym"]
    assert responder.room_card({"amenities": 5})["amenities"] == []
    assert responder.room_card({"amenities": [], "features": ["x"]})["amenities"] == []


def test_price_prefers_price_per_month():
    assert responder.room_card({"pricePerMonth": 100, "price": 200})["price_per_month"] == 100
    assert responder.room_card({"pricePerMonth": None, "price": 200})["price_per_month"] == 200


def test_search_reply_lists_rooms_with_facts_from_data(rooms):
    reply = responder.format_search_reply(rooms[:2], {"city": "bangalore"})
    assert reply.startswith("I found 2 rooms in Bangalore:")
    assert "1. 1 BHK Flat, 2RK, Koramangala, ₹25,000/month" in reply
    assert "2. " in reply and "₹7,500/month" in reply


def test_search_reply_singular_and_missing_fields():
    reply = responder.format_search_reply([{"id": 1}], {"city": "pune"})
    assert reply.startswith("I found 1 room in Pune:")
    assert "Untitled Room, price not listed" in reply


def test_no_rooms_reply():
    assert "couldn't find any rooms in Pune" in responder.format_no_rooms_reply({"city": "pune"})


def test_unavailable_message():
    assert responder.UNAVAILABLE_MESSAGE == "Room search is unavailable right now, please try again shortly."


def test_cards_are_json_serialisable(rooms):
    import json
    json.dumps(responder.room_cards(rooms))
