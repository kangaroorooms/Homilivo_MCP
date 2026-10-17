"""Phase 3 action tests. Offline: fixture rooms, mocked client, no torch."""

from __future__ import annotations

import pytest

from llm_approach import reuse, rooms_client, validators
from llm_approach.actions import ActionResult, TurnBudget
from llm_approach.actions import clarify, conversation, followup, platform, search
from llm_approach.schema import Decision, Intent, Operation, Slots
from llm_approach.state import LLMConversationState


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)


def make(intent, operation=None, *, ref=None, compare=(), reply=None, question=None, query=None, **slots):
    return Decision(
        intent=Intent(intent),
        operation=Operation(operation) if operation else None,
        slots=Slots(**slots),
        room_reference=ref,
        compare_references=list(compare),
        standalone_query=query,
        clarifying_question=question,
        reply=reply,
        confidence=0.9,
    )


def validated(decision, state):
    clean, _ = validators.validate_slots(
        decision.slots,
        rooms_count=len(state.rooms_on_screen),
        has_selected=state.selected_room_index is not None,
        room_reference=decision.room_reference,
        compare_references=decision.compare_references,
    )
    return clean


def run(handler, decision, state, budget=None, **kw):
    return handler(decision, validated(decision, state), state, budget or TurnBudget(), **kw)


@pytest.fixture
def state(rooms):
    s = LLMConversationState()
    s.apply_search_result({"city": "bangalore"}, rooms)
    return s


def followup_run(state, op, **kw):
    return run(followup.handle_followup, make("ROOM_FOLLOWUP", op, **kw), state)


# ---- follow-ups -----------------------------------------------------------


def test_cheapest_selects_room_and_s8_wifi_uses_same_room(state):
    r = followup_run(state, "LOWEST_PRICE")
    assert "Room 6" in r.reply and "₹7,000" in r.reply
    assert state.selected_room_index == 6
    # S8: next turn "does it have wifi" -> the model sends "selected"
    r2 = followup_run(state, "AMENITY", ref="selected", amenity="wifi")
    assert "Room 6" in r2.reply and "does not list" in r2.reply
    assert state.selected_room_index == 6


def test_most_expensive(state):
    r = followup_run(state, "HIGHEST_PRICE")
    assert "Room 7" in r.reply and "₹30,000" in r.reply and state.selected_room_index == 7


def test_price_tie_selects_nothing(state):
    state.rooms_on_screen[0]["pricePerMonth"] = 7000
    r = followup_run(state, "LOWEST_PRICE")
    assert "tied" in r.reply and "1 and 6" in r.reply
    assert state.selected_room_index is None


def test_wifi_across_rooms(state):
    r = followup_run(state, "AMENITY", amenity="wifi")
    assert r.reply == "Room 10 has WiFi."


def test_amenity_normalised_and_none_found(state):
    assert "Rooms" in followup_run(state, "AMENITY", amenity="Hot Water").reply
    assert followup_run(state, "AMENITY", amenity="gym").reply == "Room 10 has Gym."
    state.rooms_on_screen.pop()  # drop the room with wifi and gym
    assert "None of these rooms list" in followup_run(state, "AMENITY", amenity="gym").reply


def test_unknown_amenity_is_not_verified(state):
    r = followup_run(state, "AMENITY", amenity="jacuzzi")
    assert r.reply == reuse.get_not_verified_amenity_response()


def test_second_room_price_and_selection_persists(state):
    r = followup_run(state, "ROOM_DETAILS", ref=2)
    assert "Room 2" in r.reply and "Koramangala" in r.reply
    assert state.selected_room_index == 2
    r2 = followup_run(state, "ROOM_PRICE")  # "how much is it" -> selected
    assert "Room 2" in r2.reply and "₹7,500" in r2.reply


def test_room_price_without_selection_asks_which(state):
    r = followup_run(state, "ROOM_PRICE")
    assert "Which room" in r.reply and state.selected_room_index is None


def test_room_type_and_location(state):
    assert "Single" in followup_run(state, "ROOM_TYPE", ref=2).reply
    loc = followup_run(state, "LOCATION", ref=1).reply
    assert "Koramangala" in loc and "6 block Koramangala" in loc


def test_compare_one_and_two(state):
    r = followup_run(state, "COMPARE", compare=[1, 2])
    assert "1. 1 BHK Flat" in r.reply and "2. " in r.reply and "₹25,000" in r.reply and "₹7,500" in r.reply
    assert state.selected_room_index is None


def test_compare_needs_two_valid_rooms(state):
    assert "Which two rooms" in followup_run(state, "COMPARE", compare=[1, 99]).reply


def test_family_suitability(state):
    r = followup_run(state, "FAMILY_SUITABILITY")
    assert "Room 3" not in r.reply.split("suitable")[0] and "Rooms 1, 2" in r.reply
    assert "not listed as suitable" in followup_run(state, "FAMILY_SUITABILITY", ref=3).reply


def test_student_and_professional_suitability(state):
    assert "suitable for students" in followup_run(state, "STUDENT_SUITABILITY", ref=1).reply
    assert "working professionals" in followup_run(state, "PROFESSIONAL_SUITABILITY", ref=1).reply


def test_missing_data_is_not_listed(state):
    del state.rooms_on_screen[1]["pricePerMonth"]
    del state.rooms_on_screen[1]["landmark"]
    assert "not listed" in followup_run(state, "ROOM_PRICE", ref=2).reply
    state.rooms_on_screen[2].pop("amenities")
    assert "not listed" in followup_run(state, "AMENITY", ref=3, amenity="fan").reply
    state.rooms_on_screen[2].pop("idealFor")
    assert "not listed" in followup_run(state, "STUDENT_SUITABILITY", ref=3).reply


def test_no_rooms_on_screen_asks_to_search_first():
    s = LLMConversationState()
    r = followup_run(s, "LOWEST_PRICE")
    assert "search first" in r.reply and r.rooms is None and not r.tool_called


def test_out_of_range_reference_asks_which(state):
    assert "Which room" in followup_run(state, "ROOM_TYPE", ref=42).reply


def test_selected_with_no_selection_asks_which(state):
    assert "Which room" in followup_run(state, "ROOM_PRICE", ref="selected").reply


# ---- search ---------------------------------------------------------------


def test_new_search_replaces_state(state):
    state.select_room(3)
    d = make("NEW_SEARCH", city="Bangalore", bhk=2, max_price="20k")
    r = run(search.handle_new_search, d, state)
    assert r.tool_called and r.action == "new_search"
    assert [c["price_per_month"] for c in r.rooms] == [17500, 20000]
    assert state.selected_room_index is None
    assert state.current_search["bhk"] == 2 and state.current_search["max_price"] == 20000
    assert len(state.rooms_on_screen) == 2


def test_new_search_without_city_asks_and_sets_pending():
    s = LLMConversationState()
    r = run(search.handle_new_search, make("NEW_SEARCH", bhk=2), s)
    assert r.reply == "Which city are you looking for?" and not r.tool_called
    assert s.pending_question["type"] == "city"


def test_rejected_city_is_not_searched():
    s = LLMConversationState()
    r = run(search.handle_new_search, make("NEW_SEARCH", city="pls urgent"), s)
    assert r.reply == "Which city are you looking for?" and not r.tool_called


def test_refinement_keeps_bhk_and_price_when_only_city_changes(rooms):
    s = LLMConversationState()
    run(search.handle_new_search, make("NEW_SEARCH", city="Bangalore", bhk=2, max_price=25000), s)
    r = run(search.handle_refinement, make("REFINEMENT", city="Udupi"), s)
    assert r.tool_called and s.current_search["city"] == "Udupi"
    assert s.current_search["bhk"] == 2 and s.current_search["max_price"] == 25000
    assert "couldn't find any rooms in Udupi" in r.reply and r.rooms == []  # fixture is Bangalore only


def test_refinement_adds_budget_to_current_search(state):
    state.current_search["city"] = "bangalore"
    r = run(search.handle_refinement, make("REFINEMENT", max_price="10k"), state)
    assert state.current_search["max_price"] == 10000
    assert {c["price_per_month"] for c in r.rooms} == {7000, 7500, 8000}


def test_refinement_nothing_changed_asks(state):
    state.current_search["city"] = "bangalore"
    r = run(search.handle_refinement, make("REFINEMENT"), state)
    assert r.reply == "Which area would you like? For example Koramangala or Adugodi." and not r.tool_called
    assert len(state.rooms_on_screen) == 10


def test_refinement_without_any_search_asks_city():
    s = LLMConversationState()
    r = run(search.handle_refinement, make("REFINEMENT", max_price=9000), s)
    assert r.reply == "Which city are you looking for?" and s.pending_question["type"] == "city"


def test_search_amenity_filters_client_side():
    s = LLMConversationState()
    r = run(search.handle_new_search, make("NEW_SEARCH", city="Bangalore", amenity="wifi"), s)
    assert len(r.rooms) == 1 and r.rooms[0]["id"] == "fixture-10"


def test_search_failure_keeps_previous_rooms_and_selection(state, monkeypatch):
    state.select_room(2)
    monkeypatch.setattr(
        rooms_client, "search_rooms",
        lambda *a, **k: rooms_client.RoomsResult(ok=False, error_class="http_500", status=500),
    )
    r = run(search.handle_new_search, make("NEW_SEARCH", city="Pune"), state)
    assert r.reply == "Room search is unavailable right now, please try again shortly."
    assert r.error_class == "http_500" and r.rooms is None and r.tool_called
    assert len(state.rooms_on_screen) == 10 and state.selected_room_index == 2
    assert state.current_search["city"] == "bangalore"


def test_empty_result_reported_honestly():
    s = LLMConversationState()
    r = run(search.handle_new_search, make("NEW_SEARCH", city="Pune"), s)
    assert "couldn't find any rooms in Pune" in r.reply and r.rooms == []
    assert s.rooms_on_screen == []


def test_show_more_honest_paths(state, monkeypatch):
    r = run(search.handle_show_more, make("SHOW_MORE"), state)
    assert not r.tool_called
    assert r.reply == reuse.get_show_alternatives_response()  # 10 rooms == limit
    state.rooms_on_screen.pop()
    state.api_room_count = len(state.rooms_on_screen)  # API itself returned 9
    r = run(search.handle_show_more, make("SHOW_MORE"), state)
    assert r.reply.startswith("These are all the matching rooms I found")
    assert run(search.handle_show_more, make("SHOW_MORE"), LLMConversationState()).reply == "Which city are you looking for?"


# ---- platform -------------------------------------------------------------


def test_platform_grounded_returns_kb_text_only(state, monkeypatch):
    monkeypatch.setattr(reuse, "get_answer_platform_question",
                        lambda: lambda q: {"grounded": True, "message": "KB ANSWER", "sources": []})
    d = make("PLATFORM_INFO", query="how does homilivo make money", reply="LLM INVENTED THIS")
    r = run(platform.handle_platform_info, d, state)
    assert r.reply == "KB ANSWER" and "INVENTED" not in r.reply


@pytest.mark.parametrize("answer", [
    {"grounded": False, "message": "something", "sources": []},
    {"grounded": True, "message": "", "sources": []},
    "garbage",
])
def test_platform_ungrounded(state, monkeypatch, answer):
    monkeypatch.setattr(reuse, "get_answer_platform_question", lambda: lambda q: answer)
    d = make("PLATFORM_INFO", query="q", reply="LLM INVENTED THIS")
    r = run(platform.handle_platform_info, d, state)
    assert r.reply == "I don't have that information, please contact Homilivo support."


def test_platform_kb_error(state, monkeypatch):
    def boom():
        raise RuntimeError("chroma down")
    monkeypatch.setattr(reuse, "get_answer_platform_question", lambda: boom)
    d = make("PLATFORM_INFO", query="q")
    r = run(platform.handle_platform_info, d, state)
    assert r.error_class == "kb_error" and "contact Homilivo support" in r.reply


# ---- conversation ---------------------------------------------------------


def test_general_chat_uses_reply_capped_and_steers():
    s = LLMConversationState()
    d = make("GENERAL_CHAT", reply="Hello there! Nice to meet you. I like sunny days.")
    r = run(conversation.handle_general_chat, d, s)
    assert r.reply == "Hello there! Nice to meet you. Tell me a city and I'll help you find a room."


@pytest.mark.parametrize("bad", [
    "Rooms start at 5000 rupees!", "Our rent is cheap.", "Homilivo charges a fee.",
    "Try the 2 bhk near the owner.", "That costs ₹7,000.", "", None,
])
def test_general_chat_unsafe_reply_replaced(bad):
    d = make("GENERAL_CHAT", reply=bad)
    r = run(conversation.handle_general_chat, d, LLMConversationState())
    assert r.reply.startswith(conversation.SAFE_CHAT)
    assert not any(ch.isdigit() for ch in r.reply)


def test_general_chat_steer_with_rooms(state):
    r = run(conversation.handle_general_chat, make("GENERAL_CHAT", reply="You're welcome!"), state)
    assert r.reply.endswith("Would you like to ask about these rooms or search again?")


def test_unsupported_booking_uses_existing_wording(state):
    d = make("UNSUPPORTED_ACTION", reply="Sure, booked for 5000!", query="book this room")
    r = run(conversation.handle_unsupported, d, state)
    assert r.reply == reuse.get_unsupported_action_response()


def test_unsupported_owner_contact(state):
    d = make("UNSUPPORTED_ACTION", reply="Here is the number 99999", query="call the owner")
    r = run(conversation.handle_unsupported, d, state)
    assert r.reply == "To connect with the owner, you'll need a Homilivo subscription."


def test_out_of_scope_redirects(state):
    r = run(conversation.handle_out_of_scope, make("OUT_OF_SCOPE", reply="Light is 299792458 m/s"), state)
    assert "299" not in r.reply and "finding rooms" in r.reply


# ---- clarify --------------------------------------------------------------


def test_reject_records_rooms_and_sets_pending(state):
    r = run(clarify.handle_reject, make("REJECT_CURRENT"), state)
    assert len(state.rejected_room_ids) == 10
    assert state.pending_question["type"] == "reject"
    assert "area, budget or room type" in r.reply
    assert len(state.rooms_on_screen) == 10  # still available for follow-ups


def test_reject_without_rooms_asks_city_and_budget():
    s = LLMConversationState()
    r = run(clarify.handle_reject, make("REJECT_CURRENT"), s)
    assert r.reply == "Which city and budget are you looking for?" and s.rejected_room_ids == []


def test_unclear_uses_safe_question_only():
    s = LLMConversationState()
    r = run(clarify.handle_unclear, make("UNCLEAR", question="Could you tell me more?"), s)
    assert r.reply == "Could you tell me more?"
    r = run(clarify.handle_unclear, make("UNCLEAR", question="Is 5000 your budget?"), s)
    assert r.reply == "Which city and budget are you looking for?"


def test_unclear_state_aware_template(state):
    r = run(clarify.handle_unclear, make("UNCLEAR"), state)
    assert r.reply == "Would you like to search again, or ask about these rooms?"


def test_idk_after_city_question_asks_again_and_never_searches(monkeypatch):
    calls = []
    monkeypatch.setattr(rooms_client, "search_rooms", lambda *a, **k: calls.append(a))
    s = LLMConversationState()
    run(search.handle_new_search, make("NEW_SEARCH"), s)
    assert s.pending_question["type"] == "city"
    r = run(clarify.handle_unclear, make("UNCLEAR", question="Do you want rooms?"), s)
    assert r.reply == "Which city are you looking for?" and s.pending_question["type"] == "city"
    r = run(search.handle_new_search, make("NEW_SEARCH", city="I Dont Know"), s)
    assert r.reply == "Which city are you looking for?"
    assert calls == []


# ---- no unvalidated slot reaches the client -------------------------------


def test_client_only_called_with_validated_city(state, monkeypatch):
    seen = []

    def spy(search_dict, budget, **kw):
        seen.append(search_dict)
        return rooms_client.RoomsResult(ok=True, rooms=[])

    monkeypatch.setattr(rooms_client, "search_rooms", spy)
    bad = [
        make("NEW_SEARCH"), make("NEW_SEARCH", city="pls urgent"), make("NEW_SEARCH", city="123"),
        make("REFINEMENT"), make("REFINEMENT", city="different area"),
    ]
    for d in bad:
        run(search.handle_new_search if d.intent is Intent.NEW_SEARCH else search.handle_refinement,
            d, LLMConversationState())
    assert seen == []
    # junk bhk / price are dropped before the client sees them
    run(search.handle_new_search, make("NEW_SEARCH", city="Bangalore", bhk=99, max_price="free"), LLMConversationState())
    assert len(seen) == 1
    assert seen[0]["city"] == "Bangalore" and seen[0]["bhk"] is None and seen[0]["max_price"] is None


def test_action_result_defaults():
    r = ActionResult("hi")
    assert r.rooms is None and r.tool_called is False and r.error_class is None


# ---- Phase 4 carry-over fixes ---------------------------------------------


def test_show_more_uses_api_count_before_amenity_filter():
    """Full API page, filtered to 2 by amenity: more may exist, so no 'all' claim."""
    s = LLMConversationState()
    r = run(search.handle_new_search, make("NEW_SEARCH", city="Bangalore", amenity="WiFi"), s)
    assert r.tool_called and len(s.rooms_on_screen) == 1  # only fixture-10 lists WiFi
    assert s.api_room_count == 10
    more = run(search.handle_show_more, make("SHOW_MORE"), s)
    assert more.reply == reuse.get_show_alternatives_response()
    assert not more.reply.startswith("These are all")


def test_show_more_all_when_api_returned_fewer_than_limit():
    s = LLMConversationState()
    run(search.handle_new_search, make("NEW_SEARCH", city="Bangalore", max_price=10000), s)
    assert s.api_room_count == len(s.rooms_on_screen) == 3
    assert run(search.handle_show_more, make("SHOW_MORE"), s).reply.startswith("These are all")


def test_amenity_is_kept_by_refinement():
    s = LLMConversationState()
    run(search.handle_new_search, make("NEW_SEARCH", city="Bangalore", bhk=1, amenity="hot-water"), s)
    assert s.current_search["amenity"] == "hot-water"
    seen = {}
    real = rooms_client.search_rooms

    def spy(search_, budget, **kw):
        seen.update(search_)
        return real(search_, budget, **kw)

    import pytest as _p
    mp = _p.MonkeyPatch()
    mp.setattr(rooms_client, "search_rooms", spy)
    try:
        run(search.handle_refinement, make("REFINEMENT", city="Pune"), s)
    finally:
        mp.undo()
    assert seen["city"] == "Pune" and seen["bhk"] == 1 and seen["amenity"] == "hot-water"
    assert s.current_search["amenity"] == "hot-water"
