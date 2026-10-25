"""The sort preference lives in the session's search state (not a global). Offline, fakes only."""

from __future__ import annotations

import fakeredis
import pytest

from llm_approach import config, rooms_client, session_store as ss, sorting
from llm_approach.actions import city_change
from llm_approach.state import LLMConversationState
from llm_approach.tests.conftest import dec


def sort_of(pipe):
    return pipe.state.current_search["sort"]


def start(pipe):
    pipe.turn("2 bhk in bangalore under 20k", dec("NEW_SEARCH", city="Bangalore", bhk=2, max_price=20000))


# ---- where it comes from ----------------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("show the cheaper ones", "price_asc"), ("cheapest first please", "price_asc"), ("sabse sasta dikhao", "price_asc"),
    ("most expensive ones", "price_desc"), ("highest rated only", "rating_desc"), ("newest first", sorting.RESET),
    ("2bhk under 15k", None), ("cheap rooms in pune", None), (None, None),
])
def test_detect(text, expected):
    assert sorting.detect(text) == expected


def test_the_sort_is_not_a_global_and_the_api_still_gets_latest(pipe):
    assert config.ROOMS_SORT == "latest"
    start(pipe)
    pipe.turn("show the cheaper ones", dec("REFINEMENT"))
    assert sort_of(pipe) == "price_asc"
    assert config.ROOMS_SORT == "latest"
    assert rooms_client.build_params({"city": "Pune", "sort": "price_asc"})["sort"] == "latest"  # the API knows only latest
    assert rooms_client.build_params({"city": "Pune"})["sort"] == "latest"


# ---- refinement and new search ----------------------------------------------------------------------


def test_a_refinement_sets_the_sort_and_the_rooms_come_back_in_that_order(pipe):
    start(pipe)
    r = pipe.turn("show the cheaper ones", dec("REFINEMENT"))
    assert r.metrics["action"] == "refinement" and r.metrics["tool_called"] is True  # not "nothing changed"
    prices = [c["price_per_month"] for c in r.rooms]
    assert prices == sorted(prices) and len(prices) > 1
    assert pipe.searches[-1]["max_price"] == 20000 and pipe.searches[-1]["sort"] == "price_asc"


def test_the_sort_is_kept_by_later_refinements(pipe):
    start(pipe)
    pipe.turn("cheapest first", dec("REFINEMENT"))
    pipe.turn("under 15k", dec("REFINEMENT", max_price=15000))
    assert sort_of(pipe) == "price_asc" and pipe.searches[-1]["sort"] == "price_asc"


def test_a_refinement_can_clear_the_sort(pipe):
    start(pipe)
    pipe.turn("cheapest first", dec("REFINEMENT"))
    pipe.turn("newest first", dec("REFINEMENT"))
    assert sort_of(pipe) is None


def test_a_new_search_resets_the_sort(pipe):
    start(pipe)
    pipe.turn("cheapest first", dec("REFINEMENT"))
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert sort_of(pipe) is None and pipe.searches[-1]["sort"] is None


def test_a_new_search_can_ask_for_its_own_sort(pipe):
    start(pipe)
    pipe.turn("cheapest rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert sort_of(pipe) == "price_asc"


def test_without_a_sort_the_order_is_untouched(pipe):
    start(pipe)
    assert sort_of(pipe) is None and pipe.searches[-1]["sort"] is None


# ---- the D4 question --------------------------------------------------------------------------------


def test_the_summary_shows_the_sort_in_plain_words():
    s = city_change.filters_summary({"city": "Bangalore", "bhk": 2, "max_price": 20000, "sort": "price_asc"})
    assert s == "2 BHK rooms in Bangalore under ₹20,000, cheapest first"


def test_a_sort_alone_is_enough_to_ask(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    pipe.turn("cheapest first", dec("REFINEMENT"))
    r = pipe.turn("in jaipur", dec("REFINEMENT", city="Jaipur"))
    assert r.metrics["action"] == "ask_city_change" and "rooms in Bangalore, cheapest first" in r.reply


def test_same_filters_carries_the_sort_and_a_new_search_drops_it(pipe):
    for reply, kept in (("same filters", "price_asc"), ("new search", None)):
        start(pipe)
        pipe.turn("cheapest first", dec("REFINEMENT"))
        r = pipe.turn("in jaipur", dec("REFINEMENT", city="Jaipur"))
        assert "cheapest first" in r.reply
        pipe.turn(reply, dec("UNCLEAR"))
        assert pipe.searches[-1]["city"] == "Jaipur" and pipe.searches[-1]["sort"] == kept
        assert sort_of(pipe) == kept


def test_a_sort_in_the_original_message_survives_a_fresh_choice(pipe):
    start(pipe)
    pipe.turn("cheapest in jaipur", dec("REFINEMENT", city="Jaipur"))
    pipe.turn("new search", dec("UNCLEAR"))
    assert pipe.searches[-1]["sort"] == "price_asc"


# ---- sessions and storage ---------------------------------------------------------------------------


def test_two_sessions_do_not_share_a_sort(pipe):
    import uuid

    start(pipe)
    pipe.turn("cheapest first", dec("REFINEMENT"))
    first = pipe.sid
    pipe.sid = str(uuid.uuid4())
    start(pipe)
    assert sort_of(pipe) is None and pipe.store.peek(first).current_search["sort"] == "price_asc"


def test_the_sort_survives_a_redis_round_trip_and_a_restart(pipe):
    server = fakeredis.FakeServer()

    def store():
        return ss.RedisSessionStore(redis_client=fakeredis.FakeRedis(server=server, decode_responses=True),
                                    lock_wait_s=2.0, lock_ttl_ms=5000, lock_max_hold_s=30.0)

    pipe.store = store()
    start(pipe)
    pipe.turn("cheapest first", dec("REFINEMENT"))
    pipe.store = store()
    assert pipe.store.peek(pipe.sid).current_search["sort"] == "price_asc"
    pipe.turn("under 15k", dec("REFINEMENT", max_price=15000))
    assert pipe.searches[-1]["sort"] == "price_asc"


def test_state_round_trip_keeps_the_sort_and_schema_version_stays_1():
    st = LLMConversationState()
    st.current_search.update(city="Pune", sort="price_desc")
    blob = st.to_dict()
    assert blob["schema_version"] == 1
    assert LLMConversationState.from_dict(blob).current_search["sort"] == "price_desc"


def test_an_old_state_without_a_sort_loads_with_none():
    blob = LLMConversationState().to_dict()
    del blob["current_search"]["sort"]
    assert LLMConversationState.from_dict(blob).current_search["sort"] is None
    blob["current_search"] = {"city": "Pune", "bhk": 2}  # a blob from before the field existed
    assert LLMConversationState.from_dict(blob).current_search["sort"] is None


def test_an_unknown_stored_sort_is_dropped():
    blob = LLMConversationState().to_dict()
    blob["current_search"]["sort"] = "drop table"
    assert LLMConversationState.from_dict(blob).current_search["sort"] is None


def test_the_model_never_sees_the_sort():
    st = LLMConversationState()
    st.current_search.update(city="Pune", sort="price_asc")
    assert "sort" not in st.to_prompt_state()["current_search"]
