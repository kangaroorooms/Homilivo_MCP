"""Room type slot, gender preference slot and filter-clearing. Offline, torch-free, no network.

Covers: validators.validate_room_type(s)/validate_gender_preference, the schema's clear_slots
parsing, the search merge/refinement wiring (SEARCH_KEYS, CLEAR_SLOT_SEARCH_KEYS), the exact
roomTypes/genderPreference strings rooms_client sends, the fallback path's room type derivation,
and the four previously-failing scenarios (PG/single room/independent house, boys-only PG, add AC,
remove the budget limit) run through the real pipeline with a fake LLM decision.
"""

from __future__ import annotations

import json

import pytest

from llm_approach import fallback, pipeline, responder, rooms_client, validators
from llm_approach.actions import search as search_actions
from llm_approach.pipeline import ROOM_TYPE_NOT_FILTERED
from llm_approach.schema import Decision, parse_decision
from llm_approach.state import LLMConversationState
from llm_approach.tests.conftest import dec


# ---- validators.validate_room_type / validate_room_types -------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("PG", "PG"), ("pg", "PG"), ("paying guest", "PG"),
        ("Single", "Single"), ("single room", "Single"), ("akela room", "Single"),
        ("Shared", "Shared"), ("shared room", "Shared"), ("sharing", "Shared"), ("room sharing", "Shared"),
        ("1RK", "1RK"), ("1 rk", "1RK"), ("2RK", "2RK"), ("2 rk", "2RK"),
        ("1BHK", "1BHK"), ("2bhk", "2BHK"), ("3 BHK", "3BHK"), ("4bhk", "4BHK"),
        ("Independent House", "Independent House"), ("independent house", "Independent House"),
        ("house", "Independent House"), ("poora ghar", "Independent House"), ("alag ghar", "Independent House"),
    ],
)
def test_validate_room_type_valid(raw, expected):
    assert validators.validate_room_type(raw) == expected


@pytest.mark.parametrize("raw", ["villa", "room", "flat", "apartment", "studio", "", "5bhk", "triple sharing", 5, None, True])
def test_validate_room_type_rejects_ambiguous_or_unknown(raw):
    assert validators.validate_room_type(raw) is None


def test_validate_room_types_list_partial_accept():
    clean, had_unknown = validators.validate_room_types(["PG", "villa"])
    assert clean == ["PG"] and had_unknown is True


def test_validate_room_types_all_valid_no_unknown():
    clean, had_unknown = validators.validate_room_types(["PG", "Single"])
    assert clean == ["PG", "Single"] and had_unknown is False


def test_validate_room_types_all_unknown():
    clean, had_unknown = validators.validate_room_types(["villa", "bungalow"])
    assert clean is None and had_unknown is True


def test_validate_room_types_caps_raw_items_to_three_before_dedup():
    """The first 3 RAW items are considered; a dup among them collapses, items beyond 3 are never looked at."""
    clean, had_unknown = validators.validate_room_types(["PG", "pg", "Single", "Shared", "1RK"])
    assert clean == ["PG", "Single"] and had_unknown is False


def test_validate_room_types_accepts_a_bare_string():
    clean, had_unknown = validators.validate_room_types("PG")
    assert clean == ["PG"] and had_unknown is False


def test_validate_room_types_non_list_non_str_is_none():
    assert validators.validate_room_types(5) == (None, False)


# ---- validators.validate_gender_preference ----------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("boy", "MALE_ONLY"), ("boys", "MALE_ONLY"), ("male", "MALE_ONLY"), ("ladkon", "MALE_ONLY"),
        ("only boys", "MALE_ONLY"), ("boys only", "MALE_ONLY"), ("male only", "MALE_ONLY"),
        ("girl", "FEMALE_ONLY"), ("girls", "FEMALE_ONLY"), ("female", "FEMALE_ONLY"), ("ladies", "FEMALE_ONLY"),
        ("ladkiyon", "FEMALE_ONLY"), ("only girls", "FEMALE_ONLY"), ("girls only", "FEMALE_ONLY"),
        ("female only", "FEMALE_ONLY"),
        ("MALE_ONLY", "MALE_ONLY"), ("FEMALE_ONLY", "FEMALE_ONLY"),  # already-canonical tokens pass through
    ],
)
def test_validate_gender_preference_valid(raw, expected):
    assert validators.validate_gender_preference(raw) == expected


@pytest.mark.parametrize("raw", ["anyone", "any", "both", "either", "", 5, None, True])
def test_validate_gender_preference_anyone_and_garbage_rejected(raw):
    """"Anyone"/"both" is not a gender value: the model is expected to use clear_slots instead."""
    assert validators.validate_gender_preference(raw) is None


# ---- validate_slots wiring --------------------------------------------------------------------


def test_validate_slots_room_types_and_gender_pass_through():
    clean, rejected = validators.validate_slots({"room_types": ["PG"], "gender": "boys only"})
    assert clean["room_types"] == ["PG"] and clean["gender"] == "MALE_ONLY"
    assert "room_types" not in rejected and "gender" not in rejected


def test_validate_slots_unknown_room_type_is_rejected_but_known_ones_in_the_same_list_survive():
    clean, rejected = validators.validate_slots({"room_types": ["PG", "villa"]})
    assert clean["room_types"] == ["PG"] and "room_types" in rejected


def test_validate_slots_unknown_gender_is_rejected():
    clean, rejected = validators.validate_slots({"gender": "nonsense"})
    assert clean["gender"] is None and "gender" in rejected


def test_validate_slots_room_types_absent_is_not_rejected():
    clean, rejected = validators.validate_slots({"city": "Pune"})
    assert clean["room_types"] is None and "room_types" not in rejected


# ---- schema.Decision.clear_slots ---------------------------------------------------------------


def test_clear_slots_filters_to_known_names():
    d = parse_decision({"intent": "REFINEMENT", "clear_slots": ["budget", "city", "nonsense", "gender", "budget"]})
    assert d.clear_slots == ["budget", "gender"]  # "city" and "nonsense" dropped, no duplicates


def test_clear_slots_accepts_a_bare_string():
    d = parse_decision({"intent": "REFINEMENT", "clear_slots": "room_type"})
    assert d.clear_slots == ["room_type"]


def test_clear_slots_defaults_to_empty():
    assert parse_decision({"intent": "UNCLEAR"}).clear_slots == []


def test_decision_room_types_slot_caps_at_three_and_strips():
    d = parse_decision({"intent": "NEW_SEARCH", "slots": {"room_types": [" PG ", "Single", "Shared", "1RK"]}})
    assert d.slots.room_types == ["PG", "Single", "Shared"]


# ---- rooms_client: exact params sent -----------------------------------------------------------


def test_build_params_sends_room_types_csv_in_exact_case():
    params = rooms_client.build_params({"city": "Bangalore", "room_types": ["PG", "Single"]})
    assert params["roomTypes"] == "PG,Single"


def test_build_params_sends_gender_preference_exact_string():
    params = rooms_client.build_params({"city": "Bangalore", "gender": "MALE_ONLY"})
    assert params["genderPreference"] == "MALE_ONLY"


def test_build_params_omits_gender_when_not_set():
    params = rooms_client.build_params({"city": "Bangalore"})
    assert "genderPreference" not in params


def test_room_types_explicit_wins_over_bhk_derived():
    """bhk and room_types both set: the explicit room_types list is sent, not an "NBHK" derived from bhk."""
    params = rooms_client.build_params({"city": "Bangalore", "bhk": 2, "room_types": ["PG"]})
    assert params["roomTypes"] == "PG"


def test_bhk_alone_still_derives_room_type_param():
    params = rooms_client.build_params({"city": "Bangalore", "bhk": 2})
    assert params["roomTypes"] == "2BHK"


def test_guardrail_arguments_includes_gender_preference():
    args = search_actions.guardrail_arguments({"city": "Bangalore", "gender": "FEMALE_ONLY"})
    assert args["gender_preference"] == "FEMALE_ONLY"


# ---- the parameter guardrail copy: new values accepted/rejected --------------------------------


from llm_approach.guardrails import parameter_guardrail as pg  # noqa: E402


@pytest.mark.parametrize("room_type", ["SHARED", "2RK", "INDEPENDENT HOUSE", "PG", "1BHK"])
def test_parameter_guardrail_accepts_new_room_types(room_type):
    ok, _ = pg.validate_search_arguments({"room_types": [room_type]})
    assert ok is True


def test_parameter_guardrail_rejects_unknown_room_type():
    ok, reason = pg.validate_search_arguments({"room_types": ["VILLA"]})
    assert ok is False and "room type" in reason.lower()


@pytest.mark.parametrize("value", ["MALE_ONLY", "FEMALE_ONLY", "male_only", "female_only"])
def test_parameter_guardrail_accepts_gender_preference(value):
    ok, _ = pg.validate_search_arguments({"gender_preference": value})
    assert ok is True


def test_parameter_guardrail_rejects_unknown_gender_preference():
    ok, reason = pg.validate_search_arguments({"gender_preference": "NONBINARY"})
    assert ok is False and "gender preference" in reason.lower()


def test_parameter_guardrail_rejects_non_string_gender_preference():
    ok, reason = pg.validate_search_arguments({"gender_preference": 5})
    assert ok is False and "gender preference" in reason.lower()


# ---- actions/search.py: SEARCH_KEYS and CLEAR_SLOT_SEARCH_KEYS ---------------------------------


def test_search_keys_includes_room_types_and_gender():
    assert "room_types" in search_actions.SEARCH_KEYS and "gender" in search_actions.SEARCH_KEYS


def test_clear_slot_search_keys_maps_room_type_name_to_room_types_key():
    assert search_actions.CLEAR_SLOT_SEARCH_KEYS["room_type"] == ("room_types",)


def test_clear_slot_search_keys_has_no_city_entry():
    assert "city" not in search_actions.CLEAR_SLOT_SEARCH_KEYS


# ---- fallback.py: rule-based room type derivation ----------------------------------------------


def test_fallback_derives_pg_room_type():
    state = LLMConversationState()
    clean = fallback.fallback_slots("pg in bangalore", state)
    assert clean is not None and clean["city"] == "Bangalore" and clean["room_types"] == ["PG"]


def test_fallback_derives_single_room_type_and_keeps_bhk_behaviour():
    state = LLMConversationState()
    clean = fallback.fallback_slots("2bhk in pune", state)
    assert clean["bhk"] == 2 and clean["room_types"] == ["2BHK"]  # bhk unchanged; room_types mirrors it, no contradiction


def test_fallback_has_no_gender_support_but_does_not_crash():
    state = LLMConversationState()
    clean = fallback.fallback_slots("boys only pg in pune", state)
    assert clean is not None and clean.get("gender") is None  # not wired in the rule-based path; must not crash


def test_fallback_clear_phrasing_is_ignored_safely():
    """The fallback has no concept of clear_slots; "remove the budget" must not crash it."""
    state = LLMConversationState()
    clean = fallback.fallback_slots("remove the budget limit in pune", state)
    assert clean is not None and clean["city"] == "Pune"


# ---- real pipeline, fake LLM decision: the four previously-failing scenarios -------------------


def test_pg_search_sets_room_type_and_sends_it_to_the_rooms_api(pipe):
    r = pipe.turn("any pg in bangalore", dec("NEW_SEARCH", city="Bangalore", room_types=["PG"]))
    assert r.metrics["action"] == "new_search" and r.metrics["tool_called"] is True
    assert pipe.searches[-1]["room_types"] == ["PG"]


def test_single_room_search(pipe):
    r = pipe.turn("single room in bangalore", dec("NEW_SEARCH", city="Bangalore", room_types=["Single"]))
    assert pipe.searches[-1]["room_types"] == ["Single"]


def test_independent_house_search(pipe):
    r = pipe.turn(
        "independent house in bangalore",
        dec("NEW_SEARCH", city="Bangalore", room_types=["Independent House"]),
    )
    assert pipe.searches[-1]["room_types"] == ["Independent House"]
    assert r.metrics["tool_called"] is True


def test_boys_only_pg_search_sets_both_room_type_and_gender(pipe):
    pipe.turn("any pg in bangalore", dec("NEW_SEARCH", city="Bangalore", room_types=["PG"]))
    r = pipe.turn("boys only please", dec("REFINEMENT", gender="boys only"))
    assert r.metrics["action"] == "refinement"
    sent = pipe.searches[-1]
    assert sent["room_types"] == ["PG"] and sent["gender"] == "MALE_ONLY"


def test_add_ac_refinement_already_works_on_this_branch(pipe):
    """AC is already in AMENITY_VOCAB on this branch (cut from fix/amenity-vocabulary-and-logging);
    this just confirms the previously-failing scenario now behaves correctly end to end."""
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    r = pipe.turn("add AC", dec("REFINEMENT", amenity="AC"))
    assert r.metrics["action"] == "refinement"
    assert pipe.searches[-1]["amenity"] == "AC"
    assert ROOM_TYPE_NOT_FILTERED not in r.reply


def test_remove_the_budget_limit_clears_min_and_max_price(pipe):
    pipe.turn("rooms in bangalore under 20000", dec("NEW_SEARCH", city="Bangalore", max_price=20000))
    assert pipe.searches[-1]["max_price"] == 20000
    r = pipe.turn("remove the budget limit", dec("REFINEMENT", clear_slots=["budget"]))
    assert r.metrics["action"] == "refinement"
    sent = pipe.searches[-1]
    assert sent["min_price"] is None and sent["max_price"] is None


def test_clear_slots_can_clear_and_set_a_different_filter_in_the_same_turn(pipe):
    pipe.turn("2bhk in bangalore", dec("NEW_SEARCH", city="Bangalore", bhk=2))
    r = pipe.turn(
        "actually any bhk, but make it a PG",
        dec("REFINEMENT", clear_slots=["bhk"], room_types=["PG"]),
    )
    sent = pipe.searches[-1]
    assert sent["bhk"] is None and sent["room_types"] == ["PG"]
    assert r.metrics["action"] == "refinement"


def test_clear_slots_ignores_unknown_name_safely(pipe):
    pipe.turn("rooms in bangalore under 20000", dec("NEW_SEARCH", city="Bangalore", max_price=20000))
    r = pipe.turn(
        "change something",
        dec("REFINEMENT", clear_slots=["not_a_real_slot"], bhk=2),
    )
    assert r.metrics["action"] == "refinement"  # bhk=2 is a real change even though the clear name was bogus
    assert pipe.searches[-1]["max_price"] == 20000  # untouched: the bogus clear name did nothing


def test_unknown_room_type_gets_an_honest_note_and_the_search_still_runs(pipe):
    r = pipe.turn("villa in bangalore", dec("NEW_SEARCH", city="Bangalore", room_types=["villa"]))
    assert r.metrics["action"] == "new_search" and r.metrics["tool_called"] is True
    assert ROOM_TYPE_NOT_FILTERED in r.reply
    assert pipe.searches[-1]["room_types"] is None  # the unrecognised type was not sent


def test_room_type_and_gender_appear_in_the_reply_wording(pipe):
    pipe.turn("any pg in bangalore", dec("NEW_SEARCH", city="Bangalore", room_types=["PG"]))
    r = pipe.turn("boys only please", dec("REFINEMENT", gender="boys only"))
    assert "PG" in r.reply and "boys only" in r.reply


# ---- responder.py helpers ------------------------------------------------------------------------


def test_room_type_label_joins_multiple_types():
    assert responder.room_type_label({"room_types": ["PG", "Single"]}) == "PG/Single"


def test_room_type_label_empty_when_absent():
    assert responder.room_type_label({}) == ""


def test_gender_label_mapping():
    assert responder.gender_label({"gender": "MALE_ONLY"}) == "boys only"
    assert responder.gender_label({"gender": "FEMALE_ONLY"}) == "girls only"
    assert responder.gender_label({}) == ""


def test_no_rooms_reply_mentions_room_type_and_gender():
    text = responder.format_no_rooms_reply({"city": "Pune", "room_types": ["PG"], "gender": "MALE_ONLY"})
    assert "PG" in text and "boys only" in text and "Pune" in text


def test_no_rooms_reply_unaffected_when_room_type_and_gender_absent():
    assert responder.format_no_rooms_reply({"city": "pune"}) == (
        "I couldn't find any rooms in Pune matching that. "
        "You could try a higher budget, a different room type or another area."
    )
