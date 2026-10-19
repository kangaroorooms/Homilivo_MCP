import json

import pytest

from llm_approach.schema import Decision, Intent, Operation, parse_decision


def test_valid_full_decision():
    raw = json.dumps({
        "intent": "REFINEMENT", "operation": None,
        "slots": {"city": "Udupi", "bhk": None, "min_price": None, "max_price": 15000,
                  "amenity": None, "ideal_for": None},
        "room_reference": None, "compare_references": [], "standalone_query": "find rooms in Udupi",
        "confidence": 0.91, "needs_clarification": False, "clarifying_question": None, "reply": None,
    })
    d = parse_decision(raw)
    assert isinstance(d, Decision)
    assert d.intent is Intent.REFINEMENT
    assert d.slots.city == "Udupi" and d.slots.max_price == 15000
    assert d.confidence == 0.91


def test_accepts_dict():
    assert parse_decision({"intent": "SHOW_MORE"}).intent is Intent.SHOW_MORE


def test_partial_json_gets_defaults():
    d = parse_decision('{"intent": "UNCLEAR"}')
    assert d.slots.city is None and d.confidence == 0.0
    assert d.compare_references == [] and d.needs_clarification is False


@pytest.mark.parametrize("raw", [
    "", "not json", "[1, 2]", "null", '{"intent": "DANCE"}', '{"intent": 5}', '{"slots": {}}', None, 42,
])
def test_invalid_returns_none(raw):
    assert parse_decision(raw) is None


def test_extra_keys_dropped():
    d = parse_decision({"intent": "NEW_SEARCH", "evil": "x", "slots": {"city": "Pune", "zip": 1}})
    assert not hasattr(d, "evil") and not hasattr(d.slots, "zip")
    assert d.slots.city == "Pune"


def test_code_fence_tolerated():
    assert parse_decision('```json\n{"intent": "GENERAL_CHAT"}\n```').intent is Intent.GENERAL_CHAT


def test_intent_case_insensitive_and_operation():
    d = parse_decision({"intent": "room_followup", "operation": "lowest_price"})
    assert d.intent is Intent.ROOM_FOLLOWUP and d.operation is Operation.LOWEST_PRICE
    assert parse_decision({"intent": "ROOM_FOLLOWUP", "operation": "NOPE"}).operation is None


def test_enum_sizes():
    assert len(Intent) == 11 and len(Operation) == 11


@pytest.mark.parametrize("intent", ["GENERAL_CHAT", "OUT_OF_SCOPE", "UNSUPPORTED_ACTION"])
def test_reply_kept_for_chat_intents(intent):
    assert parse_decision({"intent": intent, "reply": "Hello!"}).reply == "Hello!"


@pytest.mark.parametrize("intent", [
    "NEW_SEARCH", "REFINEMENT", "ROOM_FOLLOWUP", "SHOW_MORE", "REJECT_CURRENT", "PLATFORM_INFO", "UNCLEAR",
])
def test_reply_dropped_for_other_intents(intent):
    assert parse_decision({"intent": intent, "reply": "Room 1 costs 5000"}).reply is None


@pytest.mark.parametrize("value,expected", [
    (1.7, 1.0), (-3, 0.0), (0.5, 0.5), ("0.8", 0.8), ("high", 0.0), (None, 0.0), (True, 0.0),
])
def test_confidence_clamped(value, expected):
    assert parse_decision({"intent": "UNCLEAR", "confidence": value}).confidence == expected


@pytest.mark.parametrize("value,expected", [
    (2, 2), ("selected", "selected"), ("SELECTED", "selected"), ("3", 3), (2.0, 2),
    (0, None), (-1, None), (2.5, None), ("second", None), (True, None), (None, None), ([1], None),
])
def test_room_reference(value, expected):
    assert parse_decision({"intent": "ROOM_FOLLOWUP", "room_reference": value}).room_reference == expected


def test_compare_references_filtered():
    d = parse_decision({"intent": "ROOM_FOLLOWUP", "compare_references": [1, 0, "2", -4, "x", 3.0]})
    assert d.compare_references == [1, 2, 3]
    assert parse_decision({"intent": "UNCLEAR", "compare_references": "1,2"}).compare_references == []


def test_slot_type_coercion():
    d = parse_decision({"intent": "NEW_SEARCH", "slots": {
        "city": 5, "bhk": 2.0, "min_price": "15k", "max_price": True, "amenity": "  ", "ideal_for": ["x"]}})
    s = d.slots
    assert s.city is None and s.bhk == 2 and s.min_price == "15k"
    assert s.max_price is None and s.amenity is None and s.ideal_for is None


def test_non_dict_slots_and_bool_flag():
    d = parse_decision({"intent": "NEW_SEARCH", "slots": "Pune", "needs_clarification": "yes"})
    assert d.slots.city is None and d.needs_clarification is False
