"""Owner-contact requests show the subscription plans of the city. Offline: a fake pricing API, no network, no LLM."""

from __future__ import annotations

import logging

import fakeredis
import httpx
import pytest

from llm_approach import observability, pipeline, pricing_client, session_store as ss, validators
from llm_approach.actions import TurnBudget, conversation, owner_contact
from llm_approach.schema import Decision, Intent, Slots
from llm_approach.state import LLMConversationState
from llm_approach.tests.conftest import dec

PLANS = {
    "bangalore": [
        {"plan": "free", "price": 0, "features": ["Browse listings"]},
        {"plan": "gold", "price": 149, "features": ["Contact 5 owners", "Priority support"]},
    ],
    "pune": [{"plan": "free", "price": 0, "features": []}, {"plan": "gold", "price": 99, "features": ["Contact 3 owners"]}],
    "mumbai": [{"plan": "platinum", "price": 499, "features": ["Unlimited contacts"]}],
    "chennai": [{"plan": "gold", "price": 79, "features": []}],
    "empty-city": [],
}


class Resp:
    def __init__(self, status=200, body=None, bad_json=False):
        self.status_code, self._body, self._bad = status, body, bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json: SECRET-BODY")
        return self._body


class FakeApi:
    """Stands in for the pricing API. ``script[city]`` is a list of outcomes used in order: an exception is raised,
    an int is a bare status, a dict or list is a 200 body, a Resp is returned. Unscripted: the city's PLANS."""

    def __init__(self, script=None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls = []

    def __call__(self, url, params, headers, timeout):
        self.calls.append({"url": url, "params": dict(params), "headers": dict(headers), "timeout": timeout})
        queue = self.script.get(params["city"])
        item = queue.pop(0) if queue else {"data": PLANS.get(params["city"], [])}
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Resp):
            return item
        if isinstance(item, int):
            return Resp(item, {"message": "SECRET-BODY"})
        return Resp(200, item)

    def cities(self):
        return [c["params"]["city"] for c in self.calls]


@pytest.fixture
def api(monkeypatch):
    """Online (not the offline fixture) with a fake API and no backoff wait."""
    monkeypatch.delenv("LLM_APPROACH_OFFLINE_ROOMS", raising=False)
    fake = FakeApi()
    monkeypatch.setattr(pricing_client, "_httpx_get", fake)
    monkeypatch.setattr(pricing_client, "PRICING_BACKOFF_S", 0.0)
    return fake


def room(city, n=1, **extra):
    return {"id": f"r{n}", "title": f"Room {n}", "city": city, "pricePerMonth": 5000 + n, "roomType": "Single", **extra}


def state_with(*cities, selected=None, search_city=None):
    s = LLMConversationState()
    s.apply_search_result({"city": search_city or (cities[0] if cities else None)}, [room(c, i + 1) for i, c in enumerate(cities)])
    if selected:
        s.select_room(selected)
    return s


def decision(intent="UNSUPPORTED_ACTION", ref=None, query=None):
    return Decision(intent=Intent(intent), slots=Slots(), room_reference=ref, standalone_query=query, confidence=0.9)


def ask(state, ref=None):
    d = decision(ref=ref)
    clean, _ = validators.validate_slots(
        d.slots, rooms_count=len(state.rooms_on_screen), has_selected=state.selected_room_index is not None,
        room_reference=ref)
    return owner_contact.handle_owner_contact(d, clean, state, TurnBudget())


def pricing_counts():
    return dict(observability.get_metrics_snapshot()["pricing"])


# ---- detection -------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "how to contact the owner", "I want to contact the owner", "contact owner", "owner number", "what is the owner's phone number",
    "owner contact details", "owner details please", "can I get the contact details of the owner", "get in touch with the landlord",
    "talk to the property owner", "how can I reach the owner of this room", "connect me with the owner",
    "owner ka number chahiye", "owner ka contact number do", "owner se baat karni hai", "owner ko kaise contact karun",
    "mujhe owner se contact karna hai", "contact karna hai owner ko", "How To Contact The OWNER?",
])
def test_owner_contact_phrases_are_detected(text):
    assert owner_contact.is_owner_contact(text)


@pytest.mark.parametrize("text", [
    "contact support", "how do I list my property", "call the owner", "book this room", "what are the subscription plans",
    "how does the subscription work", "who is the owner of homilivo", "I am a property owner, how do I contact support",
    "how can I contact support for my property owner account", "does the owner allow pets", "show me rooms in pune",
    "which is cheapest", "", None, 42,
])
def test_other_messages_are_not_detected(text):
    assert not owner_contact.is_owner_contact(text)


def test_the_standalone_query_counts_too():
    assert owner_contact.wants_owner_contact("kuch bhi", decision(query="how can I contact the owner"))
    assert owner_contact.wants_owner_contact("how can I contact the owner", decision())
    assert not owner_contact.wants_owner_contact("call the owner", decision(query="call the owner"))


# ---- which city -------------------------------------------------------------------------------------


def test_a_pointed_at_room_gives_its_city(api):
    r = ask(state_with("bangalore", "pune"), ref=2)
    assert api.cities() == ["pune"] and "**Pune**" in r.reply and "₹99" in r.reply and "₹149" not in r.reply


def test_the_selected_room_gives_its_city_and_beats_the_first_room(api):
    r = ask(state_with("pune", "bangalore", selected=2))
    assert api.cities() == ["bangalore"] and "**Bangalore**" in r.reply and "₹149" in r.reply


def test_the_word_selected_uses_the_selection(api):
    s = state_with("pune", "bangalore", selected=2)
    ask(s, ref="selected")
    assert api.cities() == ["bangalore"]


def test_rooms_in_one_city_fetch_that_city_once(api):
    r = ask(state_with("bangalore", "Bangalore", " BANGALORE "))
    assert api.cities() == ["bangalore"] and "**Bangalore**" in r.reply and r.error_class is None
    assert api.calls[0]["params"] == {"city": "bangalore"}


def test_rooms_in_several_cities_show_each_city_in_order_up_to_three(api):
    r = ask(state_with("bangalore", "pune", "bangalore", "mumbai", "chennai"))
    assert api.cities() == ["bangalore", "pune", "mumbai"]  # the fourth city is not asked for
    text = r.reply
    assert text.index("**Bangalore**") < text.index("**Pune**") < text.index("**Mumbai**") and "Chennai" not in text
    assert "₹149" in text and "₹99" in text and "₹499" in text
    assert text.index("₹99") > text.index("**Pune**") and text.index("₹499") > text.index("**Mumbai**")
    assert r.rooms is None


def test_no_pending_question_is_set_for_several_cities(api):
    s = state_with("bangalore", "pune")
    ask(s)
    assert s.pending_question is None


def test_without_rooms_the_current_search_city_is_used(api):
    s = LLMConversationState()
    s.current_search["city"] = "pune"
    r = ask(s)
    assert api.cities() == ["pune"] and "**Pune**" in r.reply and "₹99" in r.reply


def test_nothing_at_all_asks_for_a_search_first(api):
    r = ask(LLMConversationState())
    assert r.reply == "Please search for a room first so I can help you with the available subscription plans."
    assert api.calls == []


def test_rooms_without_a_usable_city_give_the_old_sentence_and_support(api):
    s = LLMConversationState()
    s.apply_search_result({"city": "bangalore"}, [{"id": "r1", "title": "A", "pricePerMonth": 5000, "roomType": "Single"}])
    r = ask(s)
    assert r.reply == ("To connect with the owner, you'll need a Homilivo subscription. "
                       "Please contact the Homilivo support team for assistance.")
    assert api.calls == []  # the search city is not used when rooms are on screen


def test_a_pointed_at_room_without_a_city_does_not_fall_back_to_another_room(api):
    s = LLMConversationState()
    s.apply_search_result({"city": "bangalore"}, [room("pune", 1), {"id": "r2", "title": "B", "pricePerMonth": 1, "roomType": "x"}])
    r = ask(s, ref=2)
    assert api.calls == [] and "support team" in r.reply


def test_a_nested_location_city_is_used(api):
    s = LLMConversationState()
    s.apply_search_result({}, [{"id": "r1", "title": "A", "pricePerMonth": 5000, "roomType": "x", "location": {"city": "Pune"}}])
    ask(s)
    assert api.cities() == ["pune"]


@pytest.mark.parametrize("bad", ["bangalore&x=1", "../etc/passwd", "pune\nHost: evil", "<script>", "", "a", "12345", None, 7,
                                 "bangalore?city=pune", "x" * 80, "pune#frag", "ban%67alore", "a=b"])
def test_an_odd_city_never_reaches_the_request(api, bad):
    s = LLMConversationState()
    s.apply_search_result({"city": "bangalore"}, [{"id": "r1", "title": "A", "pricePerMonth": 5000, "roomType": "x", "city": bad}])
    ask(s)
    t = LLMConversationState()
    t.current_search["city"] = bad
    r = ask(t)
    assert api.calls == [] and r.reply.startswith(("Please search", "To connect"))


def test_the_city_cleaner():
    assert pricing_client.clean_city("  New   Delhi ") == "new delhi"
    assert pricing_client.clean_city("St. John's-Town") == "st. john's-town"
    assert pricing_client.clean_city("bangalore; drop") is None and pricing_client.clean_city(None) is None


def test_two_sessions_with_different_cities_get_different_plans(api):
    a, b = ask(state_with("bangalore")), ask(state_with("pune"))
    assert "₹149" in a.reply and "₹99" not in a.reply
    assert "₹99" in b.reply and "₹149" not in b.reply


def test_nothing_is_cached_between_calls(api):
    s = state_with("bangalore")
    ask(s)
    api.script["bangalore"] = [{"data": [{"plan": "gold", "price": 5, "features": []}]}]
    assert "₹5" in ask(s).reply and len(api.calls) == 2


# ---- the API call ----------------------------------------------------------------------------------


def test_request_shape_and_no_token(api, monkeypatch):
    monkeypatch.setenv("KANGAROO_TOKEN", "tok-123-secret")
    ask(state_with("bangalore"))
    call = api.calls[0]
    assert call["url"] == pricing_client.PRICING_API_URL and call["timeout"] == 10.0
    assert set(call["headers"]) == {"Accept"} and "tok-123" not in repr(call)


def test_the_city_is_sent_as_an_encoded_query_parameter(monkeypatch):
    monkeypatch.delenv("LLM_APPROACH_OFFLINE_ROOMS", raising=False)
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params, timeout=timeout)
        return Resp(200, {"data": []})

    monkeypatch.setattr(pricing_client.httpx, "get", fake_get)
    pricing_client._httpx_get.original(pricing_client.PRICING_API_URL, params={"city": "new delhi"}, headers={}, timeout=10.0)
    assert seen["params"] == {"city": "new delhi"} and "?" not in seen["url"] and "new" not in seen["url"]
    assert httpx.Request("GET", seen["url"], params=seen["params"]).url.query == b"city=new+delhi"


def test_empty_plan_list_has_its_own_message(api):
    r = ask(state_with("empty-city"))
    assert r.reply == ("I wasn't able to find subscription plans for Empty-City right now. Please visit "
                       "[homilivo.com](https://homilivo.com) to view plans.")
    assert r.error_class is None


def test_the_format_matches_the_old_app(api):
    r = ask(state_with("bangalore"))
    assert r.reply == (
        "To connect with the owner, you'll need a Homilivo subscription for **Bangalore**:\n\n"
        "**FREE** — ₹0\n• Browse listings\n\n**GOLD** — ₹149\n• Contact 5 owners\n• Priority support\n\n"
        "Visit [homilivo.com](https://homilivo.com) to subscribe.")


def test_a_bare_list_is_accepted(api):
    api.script["pune"] = [[{"plan": "gold", "price": 99, "features": []}]]
    assert "₹99" in ask(state_with("pune")).reply


FAILED = "I wasn't able to retrieve subscription plans right now. Please visit homilivo.com to view plans."


def test_timeout_is_tried_twice_then_fails_honestly(api):
    api.script["pune"] = [httpx.ReadTimeout("t SECRET"), httpx.ConnectTimeout("t SECRET")]
    r = ask(state_with("pune"))
    assert r.reply == FAILED and len(api.calls) == 2 and r.error_class == "pricing_failed" and r.rooms is None


def test_a_5xx_is_retried_and_then_succeeds(api):
    api.script["pune"] = [503]
    r = ask(state_with("pune"))
    assert len(api.calls) == 2 and "₹99" in r.reply


def test_two_5xx_fail(api):
    api.script["pune"] = [500, 502]
    assert ask(state_with("pune")).reply == FAILED and len(api.calls) == 2


def test_a_connection_error_is_retried(api):
    api.script["pune"] = [httpx.ConnectError("down"), {"data": PLANS["pune"]}]
    assert "₹99" in ask(state_with("pune")).reply and len(api.calls) == 2


@pytest.mark.parametrize("status", [400, 401, 404, 429])
def test_a_4xx_is_not_retried(api, status):
    api.script["pune"] = [status]
    assert ask(state_with("pune")).reply == FAILED and len(api.calls) == 1


def test_an_unexpected_exception_is_a_failed_reply_without_a_trace(api):
    api.script["pune"] = [RuntimeError("Traceback SECRET")]
    r = ask(state_with("pune"))
    assert r.reply == FAILED and "Traceback" not in r.reply and len(api.calls) == 1


BAD_BODIES = {
    "not a list": {"data": {"plan": "x"}},
    "string body": "oops",
    "item not a dict": {"data": ["gold"]},
    "plan missing": {"data": [{"price": 1, "features": []}]},
    "plan not str": {"data": [{"plan": 5, "price": 1}]},
    "plan only markup": {"data": [{"plan": "<b></b>", "price": 1}]},
    "price missing": {"data": [{"plan": "gold"}]},
    "price str": {"data": [{"plan": "gold", "price": "149"}]},
    "price bool": {"data": [{"plan": "gold", "price": True}]},
    "price negative": {"data": [{"plan": "gold", "price": -1}]},
    "price huge": {"data": [{"plan": "gold", "price": 10**9}]},
    "price nan": {"data": [{"plan": "gold", "price": float("nan")}]},
    "features not list": {"data": [{"plan": "gold", "price": 1, "features": "all"}]},
    "feature not str": {"data": [{"plan": "gold", "price": 1, "features": [1]}]},
    "too many features": {"data": [{"plan": "gold", "price": 1, "features": ["a"] * 11}]},
    "too many plans": {"data": [{"plan": "p", "price": 1}] * 11},
}


@pytest.mark.parametrize("body", BAD_BODIES.values(), ids=BAD_BODIES.keys())
def test_bad_data_rejects_the_whole_response(api, body):
    api.script["pune"] = [body]
    r = ask(state_with("pune"))
    assert r.reply == FAILED and len(api.calls) == 1  # no fake plans, no retry


def test_one_bad_plan_rejects_the_good_ones_too(api):
    api.script["pune"] = [{"data": [{"plan": "gold", "price": 99, "features": []}, {"plan": "bad", "price": "x"}]}]
    r = ask(state_with("pune"))
    assert r.reply == FAILED and "GOLD" not in r.reply


def test_a_body_that_is_not_json_fails(api):
    api.script["pune"] = [Resp(200, bad_json=True)]
    assert ask(state_with("pune")).reply == FAILED


def test_one_of_several_cities_failing_shows_the_others(api):
    api.script["pune"] = [500, 500]
    r = ask(state_with("bangalore", "pune", "mumbai"))
    assert "₹149" in r.reply and "₹499" in r.reply and "₹99" not in r.reply
    assert f"**Pune**: {FAILED}" in r.reply and r.reply.count(FAILED) == 1
    assert r.error_class == "pricing_partial"


def test_all_cities_failing_gives_the_single_failure_line(api):
    api.script["bangalore"], api.script["pune"] = [400], [400]
    assert ask(state_with("bangalore", "pune")).reply == FAILED


def test_the_whole_call_is_bounded(monkeypatch):
    import time

    monkeypatch.delenv("LLM_APPROACH_OFFLINE_ROOMS", raising=False)

    def slow(url, params, headers, timeout):
        time.sleep(0.5)
        return Resp(200, {"data": []})

    started = time.monotonic()
    out = pricing_client.fetch_plans(["pune"], http_get=slow, total_timeout=0.05)
    assert time.monotonic() - started < 0.4 and out["pune"].ok is False and out["pune"].error_class == "timeout"


# ---- text from the API ----------------------------------------------------------------------------


def test_plan_text_is_cleaned_and_capped(api):
    api.script["pune"] = [{"data": [{
        "plan": "<b>gold</b> [click](http://evil.example/x) www.evil.example",
        "price": 99,
        "features": ["**bold** <img src=x onerror=alert(1)> `code` | table", "x" * 400, "line\nbreak\x00"],
    }]}]
    reply = ask(state_with("pune")).reply
    assert "<" not in reply and ">" not in reply
    assert "evil" not in reply and "<img" not in reply and "[click]" not in reply and "`" not in reply and "\x00" not in reply
    assert "x" * 121 not in reply and "• line break" in reply
    name_line = next(ln for ln in reply.splitlines() if ln.startswith("**GOLD"))
    assert name_line == "**GOLD CLICK** — ₹99"  # the link text stays, the link and tags go


def test_a_long_plan_name_is_cut(api):
    api.script["pune"] = [{"data": [{"plan": "a" * 100, "price": 1, "features": []}]}]
    assert "A" * 41 not in ask(state_with("pune")).reply


# ---- logs ---------------------------------------------------------------------------------------------


def test_logs_hold_no_city_url_body_exception_text_or_token(api, caplog, monkeypatch):
    monkeypatch.setenv("KANGAROO_TOKEN", "tok-123-secret")
    caplog.set_level(logging.DEBUG)
    api.script["pune"] = [httpx.ReadTimeout("https://x.example/p?city=pune SECRET"), 500]
    api.script["mumbai"] = [{"data": [{"plan": "LEAKNAME", "price": "LEAKPRICE"}]}]
    api.script["chennai"] = [Resp(200, bad_json=True)]
    ask(state_with("pune", "mumbai", "chennai"))
    text = caplog.text
    assert "pricing api failure class=timeout" in text and "pricing data rejected field=price" in text
    for banned in ("pune", "mumbai", "chennai", "?city", "://", "x.example", "SECRET", "LEAK", "tok-123", "Traceback"):
        assert banned not in text, banned


# ---- the pipeline -------------------------------------------------------------------------------------


def search_turn(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert pipe.state.rooms_on_screen


def test_the_message_that_used_to_get_the_fixed_reply_now_shows_plans(pipe):
    search_turn(pipe)
    calls = pipe.llm_calls
    r = pipe.turn("how to contact the owner", dec("UNSUPPORTED_ACTION", reply="Sure, call 99999"))
    assert "**Bangalore**" in r.reply and "GOLD" in r.reply and "₹149" in r.reply and "99999" not in r.reply
    assert r.metrics["action"] == "owner_contact" and r.metrics["tool_called"] is False
    assert pipe.llm_calls == calls + 1 and len(pipe.searches) == 1  # one decision call, no second search
    assert r.rooms is None and pipe.state.rooms_on_screen  # the cards on screen are unchanged
    assert pipe.state.last_bot_message == r.reply
    assert pipe.state.history[-1] == {"role": "assistant", "content": r.reply}


@pytest.mark.parametrize("message", ["owner ka number chahiye", "owner se baat karni hai", "owner ko kaise contact karun"])
def test_hinglish_requests_show_plans(pipe, message):
    search_turn(pipe)
    assert "GOLD" in pipe.turn(message, dec("PLATFORM_INFO")).reply


def test_platform_info_with_an_owner_contact_request_shows_plans_not_the_knowledge_base(pipe):
    search_turn(pipe)
    r = pipe.turn("how can I contact the owner of this room", dec("PLATFORM_INFO", sq="How do I contact an owner?"))
    assert "GOLD" in r.reply and r.metrics["action"] == "owner_contact"


def test_the_models_rewrite_alone_can_trigger_it(pipe):
    search_turn(pipe)
    r = pipe.turn("mujhe unse baat karni hai", dec("UNSUPPORTED_ACTION", sq="I want to contact the owner"))
    assert r.metrics["action"] == "owner_contact"


def test_general_platform_questions_still_use_the_knowledge_base(pipe):
    search_turn(pipe)
    r = pipe.turn("how does the subscription work", dec("PLATFORM_INFO", sq="How does the subscription work?"))
    assert r.reply == "Homilivo earns from owner subscriptions." and r.metrics["action"] == "platform_info"


def test_call_the_owner_keeps_the_fixed_reply(pipe):
    search_turn(pipe)
    r = pipe.turn("call the owner", dec("UNSUPPORTED_ACTION", sq="call the owner"))
    assert r.reply.startswith("Sorry, I can't complete that action yet") and r.metrics["action"] == "unsupported_action"


def test_only_the_two_intents_are_checked(pipe):
    search_turn(pipe)
    r = pipe.turn("owner number", dec("GENERAL_CHAT", reply="ok"))
    assert r.metrics["action"] != "owner_contact"


def test_no_rooms_yet_asks_for_a_search_first(pipe):
    before = pricing_counts()
    r = pipe.turn("how to contact the owner", dec("UNSUPPORTED_ACTION"))
    assert r.reply == "Please search for a room first so I can help you with the available subscription plans."
    assert pricing_counts().get("pricing_no_city", 0) - before.get("pricing_no_city", 0) == 1


def test_counters_use_fixed_words(pipe):
    search_turn(pipe)
    before = pricing_counts()
    pipe.turn("owner number", dec("UNSUPPORTED_ACTION"))
    after = pricing_counts()
    moved = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    assert moved["pricing_requested"] == 1 and moved["pricing_ok"] == 1
    assert set(after) <= {"pricing_requested", "pricing_ok", "pricing_empty", "pricing_failed", "pricing_no_city"}


def test_the_output_guardrail_applies(pipe):
    search_turn(pipe)
    pipe.output_ok = False
    r = pipe.turn("how to contact the owner", dec("UNSUPPORTED_ACTION"))
    assert r.reply == pipeline.OUTPUT_BLOCKED_REPLY and r.metrics["error_class"] == "output_blocked"
    assert any("GOLD" in text for text in pipe.output_calls)  # the plan text was what the guardrail saw


def test_a_failing_api_through_the_pipeline_is_honest(pipe, monkeypatch):
    search_turn(pipe)
    monkeypatch.delenv("LLM_APPROACH_OFFLINE_ROOMS", raising=False)
    monkeypatch.setattr(pricing_client, "_httpx_get", FakeApi({"bangalore": [httpx.ReadTimeout("t"), httpx.ReadTimeout("t")]}))
    monkeypatch.setattr(pricing_client, "PRICING_BACKOFF_S", 0.0)
    before = pricing_counts()
    r = pipe.turn("how to contact the owner", dec("UNSUPPORTED_ACTION"))
    assert r.reply == FAILED and pipe.state.last_bot_message == FAILED
    assert pricing_counts().get("pricing_failed", 0) - before.get("pricing_failed", 0) == 1


def test_a_pointed_at_room_through_the_pipeline(pipe, monkeypatch):
    st = pipe.store.get_or_create(pipe.sid)
    st.apply_search_result({"city": "bangalore"}, [room("bangalore", 1), room("pune", 2)])
    pipe.store.save(pipe.sid, st)
    r = pipe.turn("how to contact the owner of the second one", dec("UNSUPPORTED_ACTION", ref=2))
    assert "**Pune**" in r.reply and "₹99" in r.reply and "Bangalore" not in r.reply


def test_after_a_new_search_the_plans_follow_the_new_city(pipe):
    st = pipe.store.get_or_create(pipe.sid)
    st.apply_search_result({"city": "bangalore"}, [room("bangalore", 1)])
    pipe.store.save(pipe.sid, st)
    assert "**Bangalore**" in pipe.turn("owner number", dec("UNSUPPORTED_ACTION")).reply
    st = pipe.store.get_or_create(pipe.sid)
    st.apply_search_result({"city": "pune"}, [room("pune", 1)])  # what the D4 answer ends in
    pipe.store.save(pipe.sid, st)
    r = pipe.turn("owner number", dec("UNSUPPORTED_ACTION"))
    assert "**Pune**" in r.reply and "Bangalore" not in r.reply


def test_state_survives_a_redis_round_trip(pipe):
    server = fakeredis.FakeServer()

    def store():
        return ss.RedisSessionStore(
            redis_client=fakeredis.FakeRedis(server=server, decode_responses=True),
            lock_wait_s=2.0, lock_ttl_ms=5000, lock_max_hold_s=30.0)

    pipe.store = store()
    search_turn(pipe)
    r = pipe.turn("how to contact the owner", dec("UNSUPPORTED_ACTION"))
    pipe.store = store()  # a restart
    back = pipe.store.peek(pipe.sid)
    assert back.last_bot_message == r.reply and "GOLD" in back.last_bot_message and back.rooms_on_screen
    assert "GOLD" in pipe.turn("owner number", dec("UNSUPPORTED_ACTION")).reply


def test_handle_unsupported_alone_is_fixed_wording():
    r = conversation.handle_unsupported(decision(query="contact the owner"), {}, LLMConversationState(), TurnBudget())
    assert r.action == "unsupported_action" and r.reply.startswith("Sorry, I can't complete that action yet")
