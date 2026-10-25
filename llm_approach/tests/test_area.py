"""Phase 5b: area (locality) support. Offline, torch-free."""

from __future__ import annotations

import json

import pytest

from llm_approach import config, prompts, responder, rooms_client, validators
from llm_approach.actions import TurnBudget, search
from llm_approach.schema import Decision, Intent, Slots, parse_decision
from llm_approach.state import LLMConversationState
from llm_approach.tests.conftest import dec


# ---- validate_area ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("koramangala", "Koramangala"),
        ("  INDIRANAGAR  ", "Indiranagar"),
        ("hsr layout", "Hsr Layout"),
        ("6 block", "6 Block"),
        ("st. marks road", "St. Marks Road"),
    ],
)
def test_validate_area_valid(raw, expected):
    assert validators.validate_area(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["different area", "any", "anywhere", "other", "somewhere", "i dont know", "pls", "urgent",
     "something", "area", "", " ", "x", None, 12, "koramangala!!@#", "<script>"],
)
def test_validate_area_rejects_filler_and_junk(raw):
    assert validators.validate_area(raw) is None


@pytest.mark.parametrize("raw", ["bangalore", "Bengaluru", "PUNE", "bombay"])
def test_validate_area_city_is_not_an_area(raw):
    assert validators.validate_area(raw) is None


def test_validate_area_too_long():
    assert validators.validate_area("a" * 41) is None
    assert validators.validate_area("one two three four five") is None
    assert validators.validate_area("one two three four") == "One Two Three Four"


def test_validate_slots_area_and_city_in_area_slot():
    clean, rejected = validators.validate_slots(Slots(city="Bangalore", area="koramangala"))
    assert clean["area"] == "Koramangala" and not rejected
    clean, rejected = validators.validate_slots(Slots(area="different area"))
    assert clean["area"] is None and rejected == ["area"]
    clean, rejected = validators.validate_slots(Slots(area="Pune"))
    assert clean["city"] == "Pune" and clean["area"] is None and not rejected
    clean, _ = validators.validate_slots(Slots(city="Bangalore", area="Pune"))
    assert clean["city"] == "Bangalore" and clean["area"] is None


def test_parse_decision_keeps_area_as_clean_string():
    d = parse_decision({"intent": "REFINEMENT", "slots": {"area": "  Koramangala "}})
    assert d.slots.area == "Koramangala"
    assert parse_decision({"intent": "REFINEMENT", "slots": {"area": 5}}).slots.area is None


# ---- the shared area filter, on the fixture ----------------------------------------


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv(config.OFFLINE_ROOMS_ENV, "1")
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)


def _search(area=None, **kw):
    return rooms_client.search_rooms({"city": "Bangalore", "area": area, **kw}, TurnBudget())


@pytest.mark.parametrize("area, count", [("adugodi", 1), ("koramangala", 9), ("Koramangala", 9),
                                         ("6 block", 3), ("KORAMANGALA!", 9), ("whitefield", 0),
                                         ("ram", 0)])
def test_area_filter_on_fixture(offline, area, count):
    result = _search(area)
    assert result.ok and len(result.rooms) == count
    assert result.api_count == 10 and result.fetch_limit == config.ROOMS_AREA_FETCH_LIMIT
    assert result.area_matches == count


def test_area_matches_landmark_not_only_location():
    room = {"location": "Elsewhere", "landmark": "Near 6 Block, Koramangala"}
    assert rooms_client.room_matches_area(room, "6 block")
    assert not rooms_client.room_matches_area({"location": None, "landmark": 5}, "6 block")


def test_no_area_search_is_unchanged(offline):
    result = _search()
    assert len(result.rooms) == 10 and result.area_matches is None and result.areas == []
    assert result.fetch_limit == config.ROOMS_LIMIT


def test_unmatched_area_reports_real_areas_of_fetched_rooms(offline):
    result = _search("whitefield")
    assert result.rooms == [] and result.areas == ["Koramangala", "Adugodi"]


def test_display_limit_applies_after_the_area_filter(monkeypatch, tmp_path):
    many = [{"id": str(i), "city": "bangalore", "location": "Koramangala", "pricePerMonth": 9000,
             "roomType": "Single"} for i in range(25)]
    path = tmp_path / "rooms.json"
    path.write_text(json.dumps({"rooms": many}), encoding="utf-8")
    monkeypatch.setenv(config.OFFLINE_ROOMS_ENV, str(path))
    result = _search("koramangala")
    assert len(result.rooms) == config.ROOMS_LIMIT and result.api_count == 25 and result.truncated


def test_fetch_limit_env_override(monkeypatch):
    monkeypatch.setenv("X_LIMIT", "45")
    assert config._int_env("X_LIMIT", 30) == 45
    monkeypatch.setenv("X_LIMIT", "nope")
    assert config._int_env("X_LIMIT", 30) == 30
    monkeypatch.setenv("X_LIMIT", "-3")
    assert config._int_env("X_LIMIT", 30) == 30
    assert config.ROOMS_AREA_FETCH_LIMIT >= config.ROOMS_LIMIT


# ---- area is never sent to the API ---------------------------------------------------


class _Resp:
    status_code = 200

    def __init__(self, rooms):
        self._rooms = rooms

    def json(self):
        return {"success": True, "data": self._rooms}


def _live(monkeypatch, rooms):
    """Live path with a spy on the HTTP call. Returns the list of captured params."""
    monkeypatch.delenv(config.OFFLINE_ROOMS_ENV, raising=False)
    monkeypatch.setenv("KANGAROO_TOKEN", "not-a-real-token")
    calls: list[dict] = []

    def fake_get(url, params, headers, timeout):
        calls.append(dict(params))
        return _Resp(rooms[: params.get("limit", len(rooms))])

    monkeypatch.setattr(rooms_client, "_httpx_get", fake_get)
    monkeypatch.setenv("LLM_APPROACH_SEARCH_TRANSPORT", "direct")  # v2 mcp: this fixture fakes the direct httpx client
    return calls


def _rooms(n, location="Koramangala"):
    return [{"id": str(i), "city": "bangalore", "location": location, "landmark": "",
             "pricePerMonth": 9000} for i in range(n)]


def test_area_is_never_an_api_param_and_fetch_limit_is_larger(monkeypatch):
    calls = _live(monkeypatch, _rooms(4) + _rooms(2, "Adugodi"))
    result = _search("adugodi", max_price=20000)
    assert len(calls) == 1
    assert "area" not in calls[0] and "location" not in calls[0] and "landmark" not in calls[0]
    assert calls[0]["city"] == "Bangalore" and calls[0]["maxPrice"] == 20000
    assert calls[0]["limit"] == config.ROOMS_AREA_FETCH_LIMIT
    assert len(result.rooms) == 2 and result.api_count == 6


def test_no_area_uses_normal_limit(monkeypatch):
    calls = _live(monkeypatch, _rooms(3))
    _search()
    assert calls[0]["limit"] == config.ROOMS_LIMIT and "area" not in calls[0]


# ---- SHOW_MORE honesty ---------------------------------------------------------------


def _show_more(state):
    return search.handle_show_more(Decision(intent=Intent.SHOW_MORE), {}, state, TurnBudget())


def _area_state(monkeypatch, fetched, area="Koramangala"):
    calls = _live(monkeypatch, fetched)
    state = LLMConversationState()
    result = search._run_search({"city": "Bangalore", "area": area}, state, TurnBudget(), "refinement")
    assert calls and result.rooms
    return state


def test_show_more_area_search_under_fetch_limit_says_all(monkeypatch):
    # 10 rooms fetched (the display limit) but the fetch limit is 30: nothing is hidden.
    state = _area_state(monkeypatch, _rooms(10))
    assert state.api_room_count == 10 and not state.rooms_truncated
    assert _show_more(state).action == "show_more_all"


def test_show_more_area_search_at_fetch_limit_does_not_claim_all(monkeypatch):
    state = _area_state(monkeypatch, _rooms(config.ROOMS_AREA_FETCH_LIMIT + 5))
    assert state.api_room_count == config.ROOMS_AREA_FETCH_LIMIT
    assert _show_more(state).action == "show_more_limit"


def test_show_more_area_search_with_hidden_matches_does_not_claim_all(monkeypatch):
    state = _area_state(monkeypatch, _rooms(15))  # 15 match, 10 shown, fetch returned < 30
    assert len(state.rooms_on_screen) == config.ROOMS_LIMIT and state.rooms_truncated
    assert _show_more(state).action == "show_more_limit"


def test_show_more_without_area_uses_page_limit(monkeypatch):
    _live(monkeypatch, _rooms(config.ROOMS_LIMIT))
    state = LLMConversationState()
    search._run_search({"city": "Bangalore"}, state, TurnBudget(), "new_search")
    assert _show_more(state).action == "show_more_limit"


# ---- state rules and actions (through the pipeline) -----------------------------------


def test_area_reply_runs_the_search_instead_of_repeating_the_question(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    r = pipe.turn("something in a different area", dec("REFINEMENT", conf=0.8, nc=True))
    assert r.metrics["action"] == "ask_area"
    r = pipe.turn("koramangala", dec("REFINEMENT", area="Koramangala"))
    assert r.metrics["action"] == "refinement"
    assert r.reply.startswith("I found 9 rooms in Koramangala, Bangalore:")
    assert pipe.searches[-1]["area"] == "Koramangala" and pipe.searches[-1]["city"] == "Bangalore"
    assert pipe.state.current_search["area"] == "Koramangala"


def test_budget_refinement_keeps_area_and_city_then_new_city_clears_area(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    pipe.turn("koramangala", dec("REFINEMENT", area="Koramangala"))
    pipe.turn("under 15k", dec("REFINEMENT", max_price=15000))
    last = pipe.searches[-1]
    assert (last["city"], last["area"], last["max_price"]) == ("Bangalore", "Koramangala", 15000)
    pipe.turn("2bhk only", dec("REFINEMENT", bhk=2))
    assert pipe.searches[-1]["area"] == "Koramangala" and pipe.searches[-1]["max_price"] == 15000
    pipe.turn("in pune", dec("REFINEMENT", city="Pune"))
    assert pipe.searches[-1]["city"] == "Pune" and pipe.searches[-1]["area"] is None
    assert pipe.state.current_search["area"] is None


def test_new_search_in_a_different_city_has_no_area(pipe):
    pipe.turn("koramangala rooms", dec("NEW_SEARCH", city="Bangalore", area="Koramangala"))
    pipe.turn("rooms in pune", dec("NEW_SEARCH", city="Pune"))
    assert pipe.searches[-1]["area"] is None


def test_refinement_with_a_new_area_replaces_the_old_one(pipe):
    pipe.turn("a", dec("NEW_SEARCH", city="Bangalore", area="Koramangala"))
    pipe.turn("b", dec("REFINEMENT", area="Adugodi"))
    assert pipe.searches[-1]["area"] == "Adugodi" and pipe.searches[-1]["city"] == "Bangalore"
    assert len(pipe.state.rooms_on_screen) == 1


def test_area_with_no_city_asks_for_the_city_and_keeps_the_area(pipe):
    r = pipe.turn("rooms in koramangala", dec("NEW_SEARCH", area="Koramangala"))
    assert r.reply == "Which city is that area in?" and r.metrics["tool_called"] is False
    assert pipe.state.current_search["area"] == "Koramangala"
    assert pipe.state.pending_question["type"] == "city" and not pipe.searches
    r = pipe.turn("bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert pipe.searches[-1]["area"] == "Koramangala" and pipe.searches[-1]["city"] == "Bangalore"
    assert r.reply.startswith("I found 9 rooms in Koramangala, Bangalore:")


def test_area_without_city_uses_the_current_city(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    pipe.turn("rooms in adugodi", dec("NEW_SEARCH", area="Adugodi"))
    assert pipe.searches[-1]["city"] == "Bangalore" and pipe.searches[-1]["area"] == "Adugodi"


def test_no_match_is_honest_and_lists_only_real_areas(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    r = pipe.turn("whitefield", dec("REFINEMENT", area="Whitefield"))
    assert r.reply == "I couldn't find rooms in Whitefield, Bangalore. Areas with rooms: Koramangala, Adugodi."
    assert pipe.state.rooms_on_screen == [] and r.rooms == []


def test_no_area_reply_lists_at_most_five():
    reply = search.no_area_reply({"city": "Pune", "area": "Baner"}, [f"Area{i}" for i in range(8)])
    assert reply == "I couldn't find rooms in Baner, Pune. Areas with rooms: Area0, Area1, Area2, Area3, Area4."
    assert search.no_area_reply({"city": "Pune", "area": "Baner"}, []) == "I couldn't find rooms in Baner, Pune."


def test_repeating_the_city_keeps_the_existing_wording(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    n = len(pipe.searches)
    r = pipe.turn("in bangalore", dec("REFINEMENT", city="Bangalore"))
    assert r.reply.startswith("You're already searching in Bangalore. Which area would you like?")
    assert len(pipe.searches) == n


def test_a_city_in_the_area_slot_is_treated_as_the_city(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    n = len(pipe.searches)
    r = pipe.turn("bangalore", dec("REFINEMENT", area="Bangalore"))
    assert r.reply.startswith("You're already searching in Bangalore")
    assert len(pipe.searches) == n


def test_responder_header_mentions_area():
    rooms = [{"title": "R", "pricePerMonth": 5000}]
    assert responder.format_search_reply(rooms, {"city": "bangalore", "area": "Koramangala"}).startswith(
        "I found 1 room in Koramangala, Bangalore:")
    assert responder.format_search_reply(rooms, {"city": "bangalore"}).startswith("I found 1 room in Bangalore:")
    assert "in Koramangala, Bangalore" in responder.format_no_rooms_reply(
        {"city": "bangalore", "area": "koramangala"})


# ---- prompt ------------------------------------------------------------------------------


def test_prompt_documents_area_and_examples_parse():
    text = prompts.system_prompt()
    assert '"area": string|null' in text and "never a city" in text
    area_examples = [e for e in prompts.FEW_SHOT_EXAMPLES if e[2]["slots"].get("area")]
    assert 1 <= len(area_examples) <= 3
    for _, _, decision in area_examples:
        assert parse_decision(decision).slots.area


def test_state_summary_with_area_stays_within_the_context_limit():
    state = LLMConversationState()
    state.current_search = {"city": "Bangalore", "area": "Koramangala", "max_price": 15000}
    state.rooms_on_screen = [{"id": i, "title": "T" * 80, "location": "Koramangala", "pricePerMonth": 9000}
                             for i in range(10)]
    state.history = [{"role": "user", "content": "x" * 200}] * 12
    summary = prompts.build_state_summary(state.to_prompt_state())
    assert summary["current_search"]["area"] == "Koramangala"
    assert len(json.dumps(summary, ensure_ascii=False)) <= config.MAX_LLM_CONTEXT_CHARS
