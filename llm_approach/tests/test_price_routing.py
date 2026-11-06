"""Price and plan-cost questions go to the pricing API; knowledge retrieval uses the raw message and the rewrite.

Offline: a fake pricing API, mocked decisions, a fake retriever. No network, no LLM."""

from __future__ import annotations

import httpx
import pytest

from llm_approach import pricing_client, reuse
from llm_approach.actions import TurnBudget, owner_contact, platform
from llm_approach.rag import answer as answer_module
from llm_approach.rag.store import KnowledgeChunk, RetrievedChunk
from llm_approach.schema import Intent, Slots, Decision
from llm_approach.state import LLMConversationState
from llm_approach.tests.conftest import dec
from llm_approach.tests.test_owner_contact import FAILED, FakeApi, room

KB_REPLY = "Homilivo earns from owner subscriptions."  # the pipe fixture's knowledge base answer


@pytest.fixture
def api(monkeypatch):
    monkeypatch.delenv("LLM_APPROACH_OFFLINE_ROOMS", raising=False)
    fake = FakeApi()
    monkeypatch.setattr(pricing_client, "_httpx_get", fake)
    monkeypatch.setattr(pricing_client, "PRICING_BACKOFF_S", 0.0)
    return fake


def on_screen(pipe, city):
    st = pipe.store.get_or_create(pipe.sid)
    st.apply_search_result({"city": city}, [room(city, 1)])
    pipe.store.save(pipe.sid, st)


# ---- detection -------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "how much does gold plan cost in jaipur", "platinum plan price in Bangalore", "gold plan price", "platinum plan cost",
    "how much do the plans cost", "What is the subscription fee?", "membership charges", "pricing of plans", "gold plan rate",
    "GOLD PLAN PRICE?",
])
def test_price_questions_are_detected(text):
    assert owner_contact.is_price_question(text)


@pytest.mark.parametrize("text", [
    "Do I need a separate plan for each city", "what are the plans", "what is the refund policy", "gold",
    "gold price in jaipur", "how much does gold cost in jaipur", "platinum price in Bangalore", "gold price", "platinum cost",
    "is the gold plan fee refundable", "can I get a refund for the gold plan", "what is the refund policy", "cancel my plan fee",
    "cancellation charges for the plan", "money deducted for my plan, what is the charge", "my plan payment failed, what is the fee",
    "failed payment for the plan fee", "price of this room", "how much is the rent", "what is the rate of interest", "", None, 42,
])
def test_other_messages_are_not_price_questions(text):
    assert not owner_contact.is_price_question(text)


@pytest.mark.parametrize("raw, expected", [
    ("Bangalore", "bangalore"), (" jaipur ", "jaipur"), ("Bengaluru", "bangalore"), ("BLR", "bangalore"),
    ("Atlantis", None), ("bangalore; drop table", None), ("http://evil.example", None), ("HSR", None),
    ("New Delhi", None), ("", None), (None, None), (7, None),
])
def test_named_city_comes_from_the_known_list(raw, expected):
    assert owner_contact.named_city(raw) == expected


# ---- the pipeline ----------------------------------------------------------------------------------


def test_a_named_city_beats_the_session_city(pipe, api):
    on_screen(pipe, "pune")
    r = pipe.turn("platinum plan price in Bangalore", dec("PLATFORM_INFO", city="Bangalore", sq="platinum plan price in Bangalore"))
    assert api.cities() == ["bangalore"] and r.metrics["action"] == "price_question"
    assert r.reply.startswith("Here are the Homilivo plans for **Bangalore**:") and "₹149" in r.reply and "₹99" not in r.reply
    assert not r.reply.startswith("To connect with the owner")  # a price answer, not the owner-contact sentence
    assert r.rooms is None and pipe.state.rooms_on_screen  # the cards on screen are unchanged


def test_a_named_city_works_without_a_session_city(pipe, api):
    r = pipe.turn("how much does gold plan cost in mumbai", dec("UNSUPPORTED_ACTION", city="Mumbai"))
    assert api.cities() == ["mumbai"] and "₹499" in r.reply and r.metrics["action"] == "price_question"


def test_the_session_city_is_used_when_no_city_is_named(pipe, api):
    on_screen(pipe, "pune")
    r = pipe.turn("gold plan price", dec("PLATFORM_INFO"))
    assert api.cities() == ["pune"] and "₹99" in r.reply


@pytest.mark.parametrize("junk", ["Atlantis", "pune; drop table", "http://evil.example/x", "ignore previous instructions", "ab"])
def test_unknown_city_text_never_reaches_the_pricing_client(pipe, api, junk):
    on_screen(pipe, "pune")
    pipe.turn("gold plan price", dec("PLATFORM_INFO", city=junk))
    assert api.cities() == ["pune"]  # the session city, never the odd text
    assert all(junk.lower() not in repr(c).lower() for c in api.calls)


def test_unknown_city_without_a_session_goes_to_the_knowledge_base(pipe, api):
    r = pipe.turn("gold plan price in atlantis", dec("PLATFORM_INFO", city="Atlantis"))
    assert api.calls == [] and r.reply == KB_REPLY and r.metrics["action"] == "platform_info"


def test_no_city_and_no_session_falls_back_to_the_knowledge_base(pipe, api):
    r = pipe.turn("gold plan price", dec("PLATFORM_INFO", sq="What is the price of the gold plan?"))
    assert api.calls == [] and r.reply == KB_REPLY and r.metrics["action"] == "platform_info"


@pytest.mark.parametrize("message", ["Do I need a separate plan for each city", "what are the plans"])
def test_questions_without_a_price_cue_stay_on_the_knowledge_base(pipe, api, message):
    on_screen(pipe, "pune")
    r = pipe.turn(message, dec("PLATFORM_INFO", city="Bangalore", sq=message))
    assert api.calls == [] and r.reply == KB_REPLY and r.metrics["action"] == "platform_info"


def test_the_rewrite_alone_does_not_make_a_price_question(pipe, api):
    on_screen(pipe, "pune")
    r = pipe.turn("what are the plans", dec("PLATFORM_INFO", sq="What is the price of the plans?"))
    assert api.calls == [] and r.metrics["action"] == "platform_info"


def test_only_the_two_intents_take_the_price_route(pipe, api):
    on_screen(pipe, "pune")
    r = pipe.turn("gold plan price", dec("GENERAL_CHAT", reply="ok"))
    assert api.calls == [] and r.metrics["action"] != "price_question"


def test_owner_contact_wins_when_both_match(pipe, api):
    on_screen(pipe, "pune")
    r = pipe.turn("price to contact the owner of the gold plan", dec("PLATFORM_INFO", city="Bangalore"))
    assert r.metrics["action"] == "owner_contact" and api.cities() == ["pune"]  # owner contact keeps the session city


def test_a_failing_api_gives_the_safe_reply(pipe, monkeypatch):
    on_screen(pipe, "pune")
    monkeypatch.delenv("LLM_APPROACH_OFFLINE_ROOMS", raising=False)
    monkeypatch.setattr(pricing_client, "_httpx_get", FakeApi({"bangalore": [httpx.ReadTimeout("t"), httpx.ReadTimeout("t")]}))
    monkeypatch.setattr(pricing_client, "PRICING_BACKOFF_S", 0.0)
    r = pipe.turn("platinum plan price in Bangalore", dec("PLATFORM_INFO", city="Bangalore"))
    assert r.reply == FAILED and r.metrics["error_class"] == "pricing_failed" and pipe.state.last_bot_message == FAILED


def test_an_empty_plan_list_gives_the_existing_message(pipe, api):
    on_screen(pipe, "pune")
    api.script["bangalore"] = [{"data": []}]
    r = pipe.turn("platinum plan price in Bangalore", dec("PLATFORM_INFO", city="Bangalore"))
    assert "wasn't able to find subscription plans for Bangalore" in r.reply


# ---- knowledge retrieval: the raw message and the rewrite ----------------------------------------------


def chunk(doc, text):
    return RetrievedChunk(KnowledgeChunk(doc, doc, "test", True, text), 0.0)


def scored(doc, text, score):
    return RetrievedChunk(KnowledgeChunk(doc, doc, "test", True, text), score)


REFUND = "Refunds are possible within 7 days of subscribing."
CONTACT = "Email support@example.com for help."


def fake_retrieve(table):
    calls = []

    def retrieve(query):
        calls.append(query)
        return list(table.get(query, []))

    retrieve.calls = calls
    return retrieve


FAQ_REFUND = "Failed or duplicate payments are refunded in 5-7 business days. Once a plan is used the fee is non-refundable."
RAW = "what is the refund policy"


def ask(monkeypatch, table, raw, rewrite):
    retrieve = fake_retrieve(table)
    monkeypatch.setattr(answer_module, "retrieve", retrieve)
    d = Decision(intent=Intent.PLATFORM_INFO, slots=Slots(), standalone_query=rewrite, confidence=0.9)
    return platform.handle_platform_info(d, {}, LLMConversationState(), TurnBudget(), user_message=raw), retrieve


@pytest.mark.parametrize("rewrite", ["Homilivo refund policy contact", "contact Homilivo support", "Homilivo refund policy"])
def test_a_drifted_rewrite_cannot_replace_the_refund_entry(monkeypatch, rewrite):
    r, retrieve = ask(monkeypatch, {
        RAW: [scored("homilivo_faq", FAQ_REFUND, 0.70), scored("terms", CONTACT, 0.60)],
        rewrite: [scored("terms", CONTACT, 0.79), scored("homilivo_faq", FAQ_REFUND, 0.65)],  # better, but by less than 0.1
    }, RAW, rewrite)
    assert r.reply == FAQ_REFUND and r.action == "platform_info"
    assert retrieve.calls == [RAW, rewrite]  # the raw message first


def test_a_vague_message_with_nothing_above_the_floor_uses_the_rewrite(monkeypatch):
    r, _ = ask(monkeypatch, {"kuch bhi": [], "What is the refund policy?": [scored("faq", REFUND, 0.8)]}, "kuch bhi", "What is the refund policy?")
    assert r.reply == REFUND


def test_a_rewrite_better_by_the_margin_wins(monkeypatch):
    assert answer_module.REWRITE_WIN_MARGIN == 0.1
    r, _ = ask(monkeypatch, {"raw q": [scored("faq", CONTACT, 0.5)], "good q": [scored("faq", REFUND, 0.6)]}, "raw q", "good q")  # exactly 0.1
    assert r.reply == REFUND
    r, _ = ask(monkeypatch, {"raw q": [scored("faq", CONTACT, 0.5)], "good q": [scored("faq", REFUND, 0.9)]}, "raw q", "good q")
    assert r.reply == REFUND


def test_a_rewrite_better_by_less_than_the_margin_does_not_win(monkeypatch):
    r, _ = ask(monkeypatch, {"raw q": [scored("faq", CONTACT, 0.5)], "good q": [scored("faq", REFUND, 0.599)]}, "raw q", "good q")
    assert r.reply == CONTACT
    r, _ = ask(monkeypatch, {"raw q": [scored("faq", CONTACT, 0.5)], "good q": [scored("faq", REFUND, 0.3 + 0.1)]}, "raw q", "good q")  # rewrite lower
    assert r.reply == CONTACT


def test_an_empty_rewrite_result_keeps_the_raw_results(monkeypatch):
    r, _ = ask(monkeypatch, {"raw q": [scored("faq", REFUND, 0.5)], "good q": []}, "raw q", "good q")
    assert r.reply == REFUND


def test_a_contact_question_still_returns_the_faq_contact_answer(monkeypatch):
    faq = "How can I contact support? Email support@homilivo.com."  # the FAQ entry as indexed
    r, _ = ask(monkeypatch, {"how do I contact support": [scored("homilivo_faq", faq, 0.75)]}, "how do I contact support", None)
    assert r.reply == "Email support@homilivo.com."


def test_the_floor_still_applies_when_neither_query_clears_it(monkeypatch):
    monkeypatch.setattr(answer_module, "retrieve", fake_retrieve({}))  # the retriever applies the 0.40 floor: nothing comes back
    d = Decision(intent=Intent.PLATFORM_INFO, slots=Slots(), standalone_query="other words", confidence=0.9)
    r = platform.handle_platform_info(d, {}, LLMConversationState(), TurnBudget(), user_message="some words")
    assert r.reply == platform.NOT_GROUNDED and r.error_class == "ungrounded"


def test_the_same_query_is_retrieved_once(monkeypatch):
    retrieve = fake_retrieve({"what is the refund policy": [scored("faq", REFUND, 0.8)]})
    monkeypatch.setattr(answer_module, "retrieve", retrieve)
    d = Decision(intent=Intent.PLATFORM_INFO, slots=Slots(), standalone_query="what is the refund policy", confidence=0.9)
    platform.handle_platform_info(d, {}, LLMConversationState(), TurnBudget(), user_message="what is the refund policy")
    assert retrieve.calls == ["what is the refund policy"]


def test_without_a_rewrite_only_the_message_is_used(monkeypatch):
    retrieve = fake_retrieve({"hello": [scored("faq", REFUND, 0.8)]})
    monkeypatch.setattr(answer_module, "retrieve", retrieve)
    d = Decision(intent=Intent.PLATFORM_INFO, slots=Slots(), confidence=0.9)
    assert platform.handle_platform_info(d, {}, LLMConversationState(), TurnBudget(), user_message="hello").reply == REFUND
    assert retrieve.calls == ["hello"]


def test_the_debug_log_shows_both_queries_their_top3_and_which_is_used(monkeypatch, caplog):
    import logging

    from llm_approach.rag import store

    monkeypatch.setattr(store, "_load_documents", lambda: [KnowledgeChunk("faq", "faq", "t", True, REFUND)])
    monkeypatch.setattr(store, "retrieve", fake_retrieve({"raw": [scored("faq", REFUND, 0.6)], "rewrite": [scored("faq", REFUND, 0.7)]}))
    caplog.set_level(logging.DEBUG, logger=platform.logger.name)
    platform._log_retrieval("raw", "rewrite")
    assert "query='raw'" in caplog.text and "rewritten='rewrite'" in caplog.text and "raw_top3=['faq#0=0.600']" in caplog.text
    assert "rewrite_top3=['faq#0=0.700']" in caplog.text and "used=rewrite" in caplog.text


def test_the_reuse_getter_still_returns_the_real_answer_function():
    reuse.get_answer_platform_question.cache_clear()
    fn = reuse.get_answer_platform_question()
    assert fn.__module__ == "llm_approach.rag.answer" and "extra_query" in fn.__code__.co_varnames


# ---- city from the message, and the reply ------------------------------------------------------------


def test_city_from_the_message_when_the_model_left_the_slot_empty(pipe, api):
    api.script["jaipur"] = [{"data": [{"plan": "gold", "price": 59, "features": ["Contact 2 owners"]}]}]
    r = pipe.turn("how much does gold plan cost in jaipur", dec("PLATFORM_INFO", sq="How much does the gold plan cost?"))
    assert api.cities() == ["jaipur"] and r.metrics["action"] == "price_question" and "₹59" in r.reply


def test_the_message_city_beats_the_session_city(pipe, api):
    on_screen(pipe, "pune")
    pipe.turn("platinum plan price in Bangalore", dec("PLATFORM_INFO"))
    assert api.cities() == ["bangalore"]


def test_the_model_slot_beats_the_message_city(pipe, api):
    pipe.turn("gold plan price in Mumbai", dec("PLATFORM_INFO", city="Pune"))
    assert api.cities() == ["pune"]


def test_an_unknown_city_in_the_message_is_ignored(pipe, api):
    on_screen(pipe, "pune")
    pipe.turn("gold plan price in Atlantis", dec("PLATFORM_INFO"))
    assert api.cities() == ["pune"]
    assert owner_contact.message_city("gold plan price in Atlantis") is None
    assert owner_contact.message_city("gold plan price in the city") is None
    assert owner_contact.message_city("how much do plans cost in blr") == "bangalore"


def test_no_city_anywhere_still_falls_back_to_the_knowledge_base(pipe, api):
    r = pipe.turn("gold plan price", dec("PLATFORM_INFO"))
    assert api.calls == [] and r.metrics["action"] == "platform_info"


@pytest.mark.parametrize("message", ["gold price in jaipur", "platinum price in Bangalore", "how much does gold cost in jaipur"])
def test_a_metal_word_without_a_plan_word_takes_the_price_route_only_on_platform_info(pipe, api, message):
    r = pipe.turn(message, dec("PLATFORM_INFO", city="Jaipur"))  # Batch B item 4: PLATFORM_INFO has ruled out jewellery
    assert api.cities() == ["jaipur"] and r.metrics["action"] == "price_question"
    for intent in ("OUT_OF_SCOPE", "GENERAL_CHAT"):
        calls = len(api.calls)
        r = pipe.turn(message, dec(intent, city="Jaipur", reply="ok"))
        assert len(api.calls) == calls and r.metrics["action"] != "price_question"


@pytest.mark.parametrize("message", ["is the gold plan fee refundable", "can I get a refund for the gold plan", "cancel the plan charge in jaipur"])
def test_refund_and_cancel_questions_stay_on_the_knowledge_base(pipe, api, message):
    on_screen(pipe, "pune")
    r = pipe.turn(message, dec("PLATFORM_INFO", city="Jaipur"))
    assert api.calls == [] and r.metrics["action"] == "platform_info"


def test_the_price_reply_has_the_new_first_line(pipe, api):
    api.script["jaipur"] = [{"data": [{"plan": "free", "price": 0, "features": ["Browse listings"]}, {"plan": "gold", "price": 59, "features": []}]}]
    r = pipe.turn("how much does gold plan cost in jaipur", dec("PLATFORM_INFO"))
    assert r.reply == ("Here are the Homilivo plans for **Jaipur**:\n\n**FREE** — ₹0\n• Browse listings\n\n**GOLD** — ₹59\n\n"
                       "Visit [homilivo.com](https://homilivo.com) to subscribe.")
    assert "To connect with the owner" not in r.reply


def test_the_owner_contact_reply_keeps_its_first_line(pipe, api):
    on_screen(pipe, "bangalore")
    r = pipe.turn("how to contact the owner", dec("UNSUPPORTED_ACTION"))
    assert r.reply.startswith("To connect with the owner, you'll need a Homilivo subscription for **Bangalore**:")


# ---- time budget: no retry past the turn deadline ------------------------------------------------


def test_retry_is_skipped_when_the_whole_call_budget_is_low():
    import httpx

    from llm_approach import pricing_client

    calls = []

    def timeout_get(url, params, headers, timeout):
        calls.append(timeout)
        raise httpx.ReadTimeout("slow")

    slept = []
    result = pricing_client.fetch_plans(["mumbai"], http_get=timeout_get, sleep=slept.append, total_timeout=2.0)
    assert len(calls) == 1 and slept == []  # 2 s < backoff + minimum retry: one attempt, no sleep
    assert result["mumbai"].ok is False and result["mumbai"].error_class == "timeout"


def test_retry_still_happens_with_enough_budget():
    import httpx

    from llm_approach import pricing_client

    calls = []

    def timeout_get(url, params, headers, timeout):
        calls.append(timeout)
        raise httpx.ReadTimeout("slow")

    pricing_client.fetch_plans(["mumbai"], http_get=timeout_get, sleep=lambda s: None, total_timeout=15.0)
    assert len(calls) == pricing_client.PRICING_MAX_ATTEMPTS


def test_low_turn_budget_caps_the_pricing_wait_and_keeps_the_failure_reply(monkeypatch):
    from llm_approach import pricing_client, turn_deadline
    from llm_approach.actions import owner_contact

    seen = {}

    def fake_fetch(cities, **kw):
        seen.update(kw)
        return {c: pricing_client.PricingResult(ok=False, error_class="timeout") for c in cities}

    monkeypatch.setattr(pricing_client, "fetch_plans", fake_fetch)
    monkeypatch.setattr(turn_deadline, "remaining", lambda: 8.0)
    result = owner_contact.handle_price_question(["mumbai"])
    assert seen["total_timeout"] == pytest.approx(8.0 - owner_contact.TURN_RESERVE_S)
    assert result.reply == owner_contact.FAILED


def test_no_turn_budget_left_skips_the_request_and_returns_the_failure_reply(monkeypatch):
    from llm_approach import pricing_client, turn_deadline
    from llm_approach.actions import owner_contact

    monkeypatch.setattr(pricing_client, "fetch_plans", lambda *a, **k: pytest.fail("no request expected"))
    monkeypatch.setattr(turn_deadline, "remaining", lambda: 2.0)
    result = owner_contact.handle_price_question(["mumbai"])
    assert result.reply == owner_contact.FAILED and result.error_class == "pricing_failed"


def test_no_deadline_leaves_the_client_default(monkeypatch):
    from llm_approach import pricing_client
    from llm_approach.actions import owner_contact

    seen = {}
    monkeypatch.setattr(pricing_client, "fetch_plans", lambda cities, **kw: seen.update(kw) or {
        c: pricing_client.PricingResult(ok=True, plans=[]) for c in cities})
    owner_contact.handle_price_question(["mumbai"])
    assert "total_timeout" not in seen


def test_remaining_is_none_outside_a_deadline_and_counts_down_inside():
    import asyncio

    from llm_approach import turn_deadline

    assert turn_deadline.remaining() is None

    async def inner():
        return turn_deadline.remaining()

    left = asyncio.run(turn_deadline.run(inner(), 40.0))
    assert 39.0 < left <= 40.0
