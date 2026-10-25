"""D4: a refinement that names a different city while other filters are remembered asks first. Offline, fakes only."""

from __future__ import annotations

import fakeredis
import pytest

from llm_approach import observability, session_store as ss
from llm_approach.actions import city_change
from llm_approach.schema import Intent
from llm_approach.state import LLMConversationState, StateDecodeError
from llm_approach.tests.conftest import dec

FILTERED = dict(city="Bangalore", bhk=2, max_price=20000, amenity="wifi")


def counts() -> dict[str, int]:
    return dict(observability.get_metrics_snapshot()["city_change"])


def delta(before: dict[str, int], name: str) -> int:
    return counts().get(name, 0) - before.get(name, 0)


def filtered_search(pipe):
    pipe.turn("2 bhk in bangalore under 20k with wifi", dec("NEW_SEARCH", **FILTERED))
    assert pipe.searches[-1]["max_price"] == 20000


def ask(pipe, city="Jaipur", **more):
    return pipe.turn(f"in {city.lower()}", dec("REFINEMENT", city=city, **more))


def pending_type(pipe):
    return (pipe.state.pending_question or {}).get("type")


# ---- when the question is asked ---------------------------------------------------------------------


def test_question_asked_for_a_filtered_search_with_the_summary_and_two_choices(pipe):
    filtered_search(pipe)
    before, calls = counts(), len(pipe.searches)
    r = ask(pipe)
    assert r.reply == (
        "You were looking at 2 BHK rooms in Bangalore under ₹20,000 with wifi. Do you want to search in "
        "Jaipur with the same filters, or start a new search for Jaipur? You can also tell me your new requirements."
    )
    assert r.metrics["action"] == "ask_city_change" and r.metrics["tool_called"] is False
    assert len(pipe.searches) == calls  # no search yet
    assert pending_type(pipe) == "city_change"
    assert pipe.state.current_search["city"] == "Bangalore"  # nothing merged silently
    assert delta(before, "city_change_question_asked") == 1


def test_summary_shows_budget_range_ideal_for_and_formats_rupees():
    s = city_change.filters_summary(
        {"city": "bangalore", "bhk": 3, "min_price": 10000, "max_price": 25000, "ideal_for": "Family", "amenity": "hot-water"}
    )
    assert s == "3 BHK rooms for family in Bangalore between ₹10,000 and ₹25,000 with hot water"
    assert city_change.filters_summary({"city": "Pune", "min_price": 8000}) == "rooms in Pune above ₹8,000"


def test_no_question_when_the_search_has_only_a_city(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    r = ask(pipe)
    assert r.metrics["action"] == "refinement" and pipe.searches[-1]["city"] == "Jaipur"
    assert pending_type(pipe) is None


def test_the_area_alone_is_not_a_filter_to_carry(pipe):
    pipe.turn("koramangala", dec("NEW_SEARCH", city="Bangalore", area="Koramangala"))
    assert ask(pipe).metrics["action"] == "refinement"


@pytest.mark.parametrize("alias", ["Bengaluru", "blr", "BANGALORE"])
def test_no_question_for_the_same_city_by_alias(pipe, alias):
    filtered_search(pipe)
    r = pipe.turn("max 15k", dec("REFINEMENT", city=alias, max_price=15000))
    assert r.metrics["action"] == "refinement" and pipe.searches[-1]["max_price"] == 15000
    assert pipe.searches[-1]["city"] == "Bangalore" and pipe.searches[-1]["amenity"] == "WiFi"


def test_no_question_when_only_the_area_changes(pipe):
    filtered_search(pipe)
    r = pipe.turn("koramangala", dec("REFINEMENT", area="Koramangala"))
    assert r.metrics["action"] == "refinement" and pipe.searches[-1]["max_price"] == 20000


def test_no_question_for_a_new_search_and_it_starts_without_old_filters(pipe):
    filtered_search(pipe)
    r = pipe.turn("rooms in jaipur", dec("NEW_SEARCH", city="Jaipur"))
    assert r.metrics["action"] == "new_search"
    assert pipe.searches[-1]["city"] == "Jaipur" and pipe.searches[-1]["max_price"] is None
    assert pipe.searches[-1]["bhk"] is None


# ---- the reply --------------------------------------------------------------------------------------


@pytest.mark.parametrize("reply", ["same", "keep", "yes", "same filters", "Yes, same filters please!", "haan", "wahi", "thik hai"])
def test_keep_replies_search_the_new_city_with_the_old_filters(pipe, reply):
    filtered_search(pipe)
    ask(pipe)
    before = counts()
    r = pipe.turn(reply, dec("UNCLEAR"))
    assert r.metrics["action"] == "refinement" and r.metrics["tool_called"] is True
    assert pipe.searches[-1] == {
        "city": "Jaipur", "area": None, "bhk": 2, "min_price": None, "max_price": 20000, "ideal_for": None,
        "amenity": "WiFi", "sort": None,
    }
    assert pending_type(pipe) is None and delta(before, "kept") == 1


@pytest.mark.parametrize("reply", ["new", "fresh", "start over", "no", "nahi", "naya search", "start a new search"])
def test_fresh_replies_search_the_new_city_with_only_the_original_message(pipe, reply):
    filtered_search(pipe)
    ask(pipe, bhk=3)  # the original message also said 3 BHK
    before = counts()
    r = pipe.turn(reply, dec("GENERAL_CHAT"))
    assert r.metrics["action"] == "new_search"
    assert pipe.searches[-1]["city"] == "Jaipur" and pipe.searches[-1]["bhk"] == 3
    assert pipe.searches[-1]["max_price"] is None and pipe.searches[-1]["amenity"] is None  # no old budget leaks
    assert pending_type(pipe) is None and delta(before, "fresh") == 1


def test_keep_adds_what_the_original_message_said_over_the_old_filters(pipe):
    filtered_search(pipe)
    ask(pipe, bhk=3)
    pipe.turn("same", dec("UNCLEAR"))
    assert (pipe.searches[-1]["bhk"], pipe.searches[-1]["max_price"], pipe.searches[-1]["amenity"]) == (3, 20000, "WiFi")


def test_new_requirements_are_a_fresh_search_for_the_new_city(pipe):
    filtered_search(pipe)
    ask(pipe)
    before = counts()
    pipe.turn("2bhk under 15k", dec("REFINEMENT", bhk=2, max_price=15000))
    assert pipe.searches[-1]["city"] == "Jaipur" and pipe.searches[-1]["bhk"] == 2
    assert pipe.searches[-1]["max_price"] == 15000 and pipe.searches[-1]["amenity"] is None  # wifi is not carried
    assert pending_type(pipe) is None and delta(before, "fresh") == 1


def test_requirements_with_a_budget_only_do_not_leak_the_old_bhk(pipe):
    filtered_search(pipe)
    ask(pipe)
    pipe.turn("under 9k", dec("NEW_SEARCH", max_price=9000))
    assert pipe.searches[-1]["max_price"] == 9000 and pipe.searches[-1]["bhk"] is None


def test_an_unclear_reply_asks_again_once_then_searches_fresh(pipe):
    filtered_search(pipe)
    ask(pipe)
    before, calls = counts(), len(pipe.searches)
    first = pipe.turn("hmm not sure", dec("UNCLEAR"))
    assert first.metrics["action"] == "ask_city_change" and "same filters" in first.reply
    assert len(pipe.searches) == calls and pending_type(pipe) == "city_change"
    assert pipe.state.pending_question["context"]["asked"] == 2
    second = pipe.turn("what", dec("UNCLEAR"))
    assert second.metrics["action"] == "new_search"
    assert pipe.searches[-1]["city"] == "Jaipur" and pipe.searches[-1]["max_price"] is None
    assert pending_type(pipe) is None
    assert delta(before, "unclear") == 2 and delta(before, "fresh") == 1 and delta(before, "city_change_question_asked") == 0


def test_conflicting_words_are_unclear(pipe):
    filtered_search(pipe)
    ask(pipe)
    r = pipe.turn("same but start new", dec("UNCLEAR"))
    assert r.metrics["action"] == "ask_city_change"


def test_a_different_intent_drops_the_question_and_is_handled_normally(pipe):
    filtered_search(pipe)
    ask(pipe)
    before = counts()
    r = pipe.turn("what is the weather", dec("OUT_OF_SCOPE"))
    assert r.metrics["action"] != "ask_city_change" and pending_type(pipe) is None
    assert delta(before, "abandoned") == 1
    assert pipe.state.current_search["city"] == "Bangalore"  # the old search is untouched


def test_each_turn_makes_exactly_one_llm_call(pipe):
    filtered_search(pipe)
    start = pipe.llm_calls
    ask(pipe)
    pipe.turn("same", dec("UNCLEAR"))
    assert pipe.llm_calls - start == 2


def test_the_model_sees_a_plain_question_not_the_remembered_filters(pipe):
    filtered_search(pipe)
    ask(pipe)
    assert pipe.state.to_prompt_state()["pending_question"] == "clarify"


# ---- classifier -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message, intent, slots, expected",
    [
        ("same", Intent.UNCLEAR, {}, "keep"),
        ("ok", Intent.GENERAL_CHAT, {}, "keep"),
        ("no", Intent.GENERAL_CHAT, {}, "fresh"),
        ("yes no", Intent.UNCLEAR, {}, "unclear"),
        ("same with 3bhk", Intent.REFINEMENT, {"bhk": 3}, "keep"),
        ("3bhk under 12k", Intent.REFINEMENT, {"bhk": 3}, "requirements"),
        ("no wifi needed", Intent.REFINEMENT, {"amenity": "wifi"}, "requirements"),
        ("what is the weather", Intent.OUT_OF_SCOPE, {}, "other"),
        ("sdkjfh", Intent.UNCLEAR, {}, "unclear"),
        ("show me more", Intent.SHOW_MORE, {}, "other"),
    ],
)
def test_classify_reply(message, intent, slots, expected):
    assert city_change.classify_reply(message, intent, slots) == expected


# ---- state: memory, Redis, reset ----------------------------------------------------------------------


def test_pending_state_is_kept_in_the_in_memory_session(pipe):
    filtered_search(pipe)
    ask(pipe)
    live = pipe.store.peek(pipe.sid)
    assert live.pending_question["type"] == "city_change"
    assert live.pending_question["context"] == {"new": {"city": "Jaipur"}, "asked": 1}


def test_pending_state_survives_a_redis_round_trip_and_a_restart(pipe):
    server = fakeredis.FakeServer()

    def store():
        return ss.RedisSessionStore(
            redis_client=fakeredis.FakeRedis(server=server, decode_responses=True),
            lock_wait_s=2.0, lock_ttl_ms=5000, lock_max_hold_s=30.0,
        )

    pipe.store = store()
    filtered_search(pipe)
    ask(pipe, bhk=3)
    pipe.store = store()  # a restart: the in-process state is gone
    assert pipe.store.peek(pipe.sid).pending_question["context"]["new"] == {"city": "Jaipur", "bhk": 3}
    r = pipe.turn("same", dec("UNCLEAR"))
    assert r.metrics["action"] == "refinement"
    assert (pipe.searches[-1]["city"], pipe.searches[-1]["bhk"], pipe.searches[-1]["max_price"]) == ("Jaipur", 3, 20000)
    assert pipe.store.peek(pipe.sid).pending_question is None


def test_state_round_trip_keeps_the_context():
    st = LLMConversationState()
    st.current_search.update(city="Bangalore", max_price=20000)
    st.set_pending("city_change", city_change.pending_context({"city": "Jaipur", "bhk": 2, "max_price": None}))
    back = LLMConversationState.from_dict(st.to_dict())
    assert back.pending_question == st.pending_question


@pytest.mark.parametrize("context", [None, {}, {"new": {}, "asked": 1}, {"new": {"city": "X"}, "asked": 3},
                                     {"new": {"city": "X"}, "asked": True}, {"new": "Pune", "asked": 1}])
def test_a_stored_city_change_question_without_a_valid_context_is_refused(context):
    blob = LLMConversationState().to_dict()
    blob["pending_question"] = {"type": "city_change", "context": context}
    with pytest.raises(StateDecodeError):
        LLMConversationState.from_dict(blob)


def test_old_stored_sessions_without_the_new_type_still_load():
    st = LLMConversationState()
    st.set_pending("clarify", "refinement")
    blob = st.to_dict()
    assert blob["schema_version"] == 1
    assert LLMConversationState.from_dict(blob).pending_question == {"type": "clarify", "context": "refinement"}


def test_reset_clears_the_pending_question(pipe):
    filtered_search(pipe)
    ask(pipe)
    assert pipe.store.reset(pipe.sid) is True
    assert pipe.store.peek(pipe.sid) is None
    calls = len(pipe.searches)
    r = pipe.turn("same", dec("UNCLEAR"))  # a fresh session: nothing pending, nothing remembered
    assert r.metrics["action"] == "clarify" and len(pipe.searches) == calls
    assert pending_type(pipe) != "city_change"


# ---- observability ----------------------------------------------------------------------------------


def test_counters_hold_only_fixed_words():
    before = counts()
    observability.record_city_change("kept")
    assert set(counts()) <= {"city_change_question_asked", "kept", "fresh", "unclear", "abandoned"}
    assert counts()["kept"] == before.get("kept", 0) + 1
