"""Batch B: no knowledge reply without letters, blocked non-search replies keep the rooms, plan-name price questions on
PLATFORM_INFO, the owner-listing FAQ entry and the 0.40 floor.

Offline: mocked decisions, a fake retriever, a fake pricing API. No network, no LLM (the FAQ test uses the local embedding model)."""

from __future__ import annotations

import pytest

from llm_approach import pipeline, pricing_client
from llm_approach.actions import TurnBudget, owner_contact, platform
from llm_approach.rag import answer as answer_module
from llm_approach.rag import store
from llm_approach.schema import Decision, Intent, Slots
from llm_approach.state import LLMConversationState
from llm_approach.tests.conftest import dec
from llm_approach.tests.test_owner_contact import PLANS, FakeApi
from llm_approach.tests.test_price_routing import REFUND, fake_retrieve, on_screen, scored


@pytest.fixture
def api(monkeypatch):
    monkeypatch.delenv("LLM_APPROACH_OFFLINE_ROOMS", raising=False)
    fake = FakeApi({"jaipur": [{"data": PLANS["pune"]}] * 4})  # the pune plans stand in for jaipur's
    monkeypatch.setattr(pricing_client, "_httpx_get", fake)
    monkeypatch.setattr(pricing_client, "PRICING_BACKOFF_S", 0.0)
    return fake


# ---- 1. no letters, no knowledge reply ------------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["🏠💰❓", "???", "123", "a ?", "💰 5"])
def test_a_message_without_letters_never_reaches_the_knowledge_base(monkeypatch, raw):
    retrieve = fake_retrieve({"What are the Homilivo plan prices?": [scored("faq", REFUND, 0.9)], raw: [scored("faq", REFUND, 0.9)]})
    monkeypatch.setattr(answer_module, "retrieve", retrieve)
    d = Decision(intent=Intent.PLATFORM_INFO, slots=Slots(), standalone_query="What are the Homilivo plan prices?", confidence=0.9)
    r = platform.handle_platform_info(d, {}, LLMConversationState(), TurnBudget(), user_message=raw)
    assert r.reply == platform.NOT_GROUNDED and r.error_class == "no_letters"
    assert retrieve.calls == []  # neither the raw message nor the rewrite was retrieved


@pytest.mark.parametrize("raw", ["ok", "refund?", "रिफंड", "返金"])
def test_a_message_with_letters_still_reaches_the_knowledge_base(monkeypatch, raw):
    retrieve = fake_retrieve({raw: [scored("faq", REFUND, 0.9)]})
    monkeypatch.setattr(answer_module, "retrieve", retrieve)
    d = Decision(intent=Intent.PLATFORM_INFO, slots=Slots(), confidence=0.9)
    r = platform.handle_platform_info(d, {}, LLMConversationState(), TurnBudget(), user_message=raw)
    assert r.reply == REFUND and retrieve.calls == [raw]


def test_an_emoji_message_through_the_pipeline_gets_the_not_available_reply(pipe, api):
    r = pipe.turn("🏠💰❓", dec("PLATFORM_INFO", sq="What are the Homilivo plan prices?"))
    assert r.reply == platform.NOT_GROUNDED and r.metrics["action"] == "platform_info" and api.calls == []


# ---- 3. a blocked reply that shows no new rooms keeps the rooms --------------------------------------------------------


def _rooms_then_block(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    rooms = list(pipe.state.rooms_on_screen)
    assert rooms
    pipe.output_ok = False
    return rooms


def _kept(pipe, r, rooms):
    assert r.reply == pipeline.OUTPUT_BLOCKED_REPLY and r.metrics["action"] == "output_blocked"
    assert r.rooms is None and pipe.state.rooms_on_screen == rooms


def test_a_blocked_followup_keeps_the_rooms(pipe):
    rooms = _rooms_then_block(pipe)
    r = pipe.turn("price of the second one", dec("ROOM_FOLLOWUP", op="ROOM_PRICE", ref=2))
    _kept(pipe, r, rooms)


def test_a_blocked_owner_contact_reply_keeps_the_rooms(pipe):
    rooms = _rooms_then_block(pipe)
    r = pipe.turn("how to contact the owner", dec("UNSUPPORTED_ACTION"))
    _kept(pipe, r, rooms)


def test_a_blocked_fallback_reply_keeps_the_rooms(pipe):
    rooms = _rooms_then_block(pipe)
    r = pipe.turn("hmm what now", None)  # no decision: the rule-based fallback asks a question
    _kept(pipe, r, rooms)


@pytest.mark.parametrize("llm", [dec("GENERAL_CHAT", reply="Hello!"), dec("OUT_OF_SCOPE"), dec("UNCLEAR", conf=0.3)])
def test_other_blocked_replies_keep_the_rooms(pipe, llm):
    rooms = _rooms_then_block(pipe)
    _kept(pipe, pipe.turn("something", llm), rooms)


@pytest.mark.parametrize("llm", [dec("NEW_SEARCH", city="Pune"), dec("REFINEMENT", max_price=9000), dec("SHOW_MORE")])
def test_a_blocked_search_refinement_or_show_more_still_clears_the_rooms(pipe, llm):
    _rooms_then_block(pipe)
    r = pipe.turn("rooms", llm)
    assert r.reply == pipeline.OUTPUT_BLOCKED_REPLY and pipe.state.rooms_on_screen == []
    assert pipe.state.selected_room_index is None


def test_a_blocked_fallback_search_still_clears_the_rooms(pipe):
    _rooms_then_block(pipe)
    r = pipe.turn("rooms in pune under 9000", None)  # no decision: the fallback runs a search
    assert r.reply == pipeline.OUTPUT_BLOCKED_REPLY and pipe.state.rooms_on_screen == []


# ---- 4. plan-name price questions on PLATFORM_INFO ---------------------------------------------------------------------


@pytest.mark.parametrize("message", ["how much does platinum cost", "platinum kitne ka hai", "gold ka daam kya hai",
                                     "free kitna hai", "platinum ki keemat", "gold rate kitni hai"])
def test_platform_info_with_a_plan_name_and_a_price_cue_takes_the_price_route(pipe, api, message):
    on_screen(pipe, "jaipur")
    r = pipe.turn(message, dec("PLATFORM_INFO", sq="What is the price of the Platinum plan?"))
    assert r.metrics["action"] == "price_question" and api.cities() == ["jaipur"]
    assert r.reply.startswith("Here are the Homilivo plans for **Jaipur**:")
    assert all(message not in repr(c) for c in api.calls)  # the raw text never reaches the pricing client


@pytest.mark.parametrize("intent", ["UNSUPPORTED_ACTION", "GENERAL_CHAT", "OUT_OF_SCOPE"])
def test_a_plan_name_alone_is_not_enough_on_other_intents(pipe, api, intent):
    on_screen(pipe, "jaipur")
    r = pipe.turn("gold price", dec(intent, reply="ok"))
    assert api.calls == [] and r.metrics["action"] != "price_question"


@pytest.mark.parametrize("message", ["refund for platinum, how much", "cancel gold, what is the cost", "platinum cancellation fee",
                                     "money deducted for gold, how much", "failed payment for platinum, kitna"])
def test_refund_and_cancellation_messages_with_a_plan_name_stay_off_the_price_route(pipe, api, message):
    on_screen(pipe, "jaipur")
    r = pipe.turn(message, dec("PLATFORM_INFO"))
    assert api.calls == [] and r.metrics["action"] == "platform_info"


def test_the_named_city_still_comes_first(pipe, api):
    on_screen(pipe, "jaipur")
    r = pipe.turn("how much does platinum cost in pune", dec("PLATFORM_INFO", city="Pune"))
    assert api.cities() == ["pune"] and r.metrics["action"] == "price_question"


def test_no_city_still_goes_to_the_knowledge_base(pipe, api):
    r = pipe.turn("how much does platinum cost", dec("PLATFORM_INFO"))
    assert api.calls == [] and r.metrics["action"] == "platform_info"


@pytest.mark.parametrize("text", ["gold plan kitne ka hai", "subscription ka daam", "membership ki keemat"])
def test_hinglish_cues_with_a_plan_word_are_price_questions(text):
    assert owner_contact.is_price_question(text)


def test_the_strict_rule_is_unchanged_for_plan_names():
    assert not owner_contact.is_price_question("platinum kitne ka hai")  # a plan name alone needs PLATFORM_INFO
    assert owner_contact.is_plan_price_question("platinum kitne ka hai", Intent.PLATFORM_INFO)
    assert not owner_contact.is_plan_price_question("platinum kitne ka hai", Intent.UNSUPPORTED_ACTION)
    assert not owner_contact.is_plan_price_question("gold jewellery", Intent.PLATFORM_INFO)  # no price cue


# ---- 2. the owner-listing FAQ entry, 5. the floor --------------------------------------------------------------------


def test_the_floor_is_040():
    assert store.MIN_RELEVANCE_SCORE == 0.40


@pytest.fixture
def real_index(tmp_path, monkeypatch):
    """The shipped documents in a temporary index; skipped when the embedding model cannot be loaded."""
    from llm_approach.rag.embeddings import get_shared_model

    try:
        get_shared_model()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"embedding model not available: {type(exc).__name__}")
    monkeypatch.setenv(store.PERSIST_DIR_ENV, str(tmp_path))
    store._reset_caches()
    yield
    store._reset_caches()


def test_owners_listing_question_gets_the_new_faq_entry(real_index):
    out = answer_module.answer_platform_question("how do owners list a property")
    assert out["grounded"] is True
    assert out["message"].startswith("Owners list a property by signing up (create an account, or use Become an agent in the website footer)")
