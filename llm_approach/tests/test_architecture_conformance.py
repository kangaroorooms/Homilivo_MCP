"""Does llm_approach follow the flow drawn in docs/architecture/architecture.svg?

Each test turn runs through the real entry point (``pipeline.handle_turn``, or ``server_app.run_turn`` and the
real WebSocket / POST routes for the entry and deadline turns). Nothing in the source is changed: the functions
that correspond to the diagram's boxes are wrapped with spies (``monkeypatch``) that only record that they were
called, in order, and then call the real function. The recorded order is compared with the path the diagram draws.
If the code is re-ordered, merged or a box is skipped, a test here fails.

Offline: the decision LLM is the scripted stand-in from ``conftest.pipe`` (the "stub provider"; one test, turn S,
runs the real ``llm_client.decide`` against a fake Gemini client), the rooms are the offline fixture (turn A uses
the real in-process MCP tool against a fake ``urlopen``), the pricing API and the FAQ answer are fakes, sessions
are in memory (turn U uses fakeredis). No Gemini, Groq, rooms, pricing or Redis-server call; no key is read or printed.

Run with ``-s`` to see every trace:  pytest llm_approach/tests/test_architecture_conformance.py -s
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import json
import uuid
from collections import defaultdict

import fakeredis
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket

from llm_approach import (
    config, conversation_answerer, extra_security, fallback, llm_client, metrics as metrics_mod, observability,
    pipeline, pricing_client, prompts, providers, reply_safety, reuse, rooms_client, semantic_gate, server_app,
    switches, textclean, turn_deadline, validators,
)
from llm_approach.actions import clarify, conversation, followup, owner_contact, platform, recall, search
from llm_approach.schema import Intent
from llm_approach.session_store import RedisSessionStore
from llm_approach.state import LLMConversationState
from llm_approach.tests.conftest import dec

from .test_mcp_transport import FakeUrlopen, good_room
from .test_owner_contact import FakeApi

# turn id -> trace, and label -> the turn ids that reached it (read by the coverage test at the end)
TRACES: dict[str, list[str]] = {}
REACHED: dict[str, set[str]] = defaultdict(set)


class Trace:
    def __init__(self) -> None:
        self.events: list[str] = []

    def mark(self, label: str) -> None:
        self.events.append(label)

    def take(self) -> list[str]:
        events, self.events = self.events, []
        return events


def spy(monkeypatch, trace: Trace, obj, name: str, label: str) -> None:
    """Replace ``obj.name`` with a wrapper that records ``label`` and calls the real function."""
    original = getattr(obj, name)
    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapper(*a, **k):
            trace.mark(label)
            return await original(*a, **k)
    else:
        @functools.wraps(original)
        def wrapper(*a, **k):
            trace.mark(label)
            return original(*a, **k)
    monkeypatch.setattr(obj, name, wrapper)


WHERE: contextvars.ContextVar[str] = contextvars.ContextVar("architecture_trace_where", default="")


def scoped(monkeypatch, trace: Trace, obj, name: str, scope: str) -> None:
    """Replace ``obj.name`` with a wrapper that marks the call as inside ``scope`` while it runs (sync or async)."""
    original = getattr(obj, name)
    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapper(*a, **k):
            token = WHERE.set(scope)
            try:
                return await original(*a, **k)
            finally:
                WHERE.reset(token)
    else:
        @functools.wraps(original)
        def wrapper(*a, **k):
            token = WHERE.set(scope)
            try:
                return original(*a, **k)
            finally:
                WHERE.reset(token)
    monkeypatch.setattr(obj, name, wrapper)


def spy_by_scope(monkeypatch, trace: Trace, obj, name: str, base: str) -> None:
    """Label ``base`` for a call from the pipeline, ``<scope>:<base>`` for a call made inside a scoped box."""
    original = getattr(obj, name)

    @functools.wraps(original)
    def wrapper(*a, **k):
        scope = WHERE.get()
        trace.mark(f"{scope}:{base}" if scope else base)
        return original(*a, **k)

    monkeypatch.setattr(obj, name, wrapper)


def spy_returned(monkeypatch, trace: Trace, obj, name: str, label: str) -> None:
    """``obj.name()`` returns a function (the reuse getters): the returned function is what records."""
    getter = getattr(obj, name)

    def new_getter():
        fn = getter()

        def inner(*a, **k):
            trace.mark(label)
            return fn(*a, **k)

        return inner

    monkeypatch.setattr(obj, name, new_getter)


def install(monkeypatch, trace: Trace, store) -> None:
    """Spies on the function behind every box of the diagram. Order of the labels is the order of the calls."""
    spy(monkeypatch, trace, textclean, "normalize_input", "normalise")
    spy_by_scope(monkeypatch, trace, extra_security, "is_extra_blocked", "security:extra")
    spy_returned(monkeypatch, trace, reuse, "get_is_security_blocked", "security:app")
    spy(monkeypatch, trace, pipeline, "_semantic_gate", "gate")
    spy(monkeypatch, trace, semantic_gate, "evaluate", "gate:classify")
    spy(monkeypatch, trace, prompts, "build_state_summary", "llm:prompt")
    spy(monkeypatch, trace, llm_client, "decide", "llm:decide")
    spy(monkeypatch, trace, fallback, "run_fallback", "fallback")
    spy(monkeypatch, trace, validators, "validate_slots", "validate")
    spy(monkeypatch, trace, pipeline, "_decide_action", "router")
    # the action modules. Direct calls inside _decide_action go through the module attributes; the dispatch goes through
    # _HANDLERS, which holds the original functions: both are wrapped, with the same label for the same function.
    box = {
        "handle_new_search": "Room search", "handle_refinement": "Room search", "handle_show_more": "Room search",
        "handle_followup": "State-based answers", "handle_recall": "State-based answers",
        "handle_general_chat": "State-based answers", "handle_unsupported": "State-based answers",
        "handle_out_of_scope": "State-based answers", "handle_reject": "Clarify", "handle_unclear": "Clarify",
        "handle_platform_info": "Platform FAQ",
    }
    for intent, fn in list(pipeline._HANDLERS.items()):
        label = f"path:{box[fn.__name__]}:{fn.__name__}"

        def make(fn=fn, label=label):
            @functools.wraps(fn)
            def wrapper(*a, **k):
                trace.mark(label)
                token = WHERE.set("inner")  # the handler's own self-checks (is_safe_text and so on) are not diagram boxes
                try:
                    return fn(*a, **k)
                finally:
                    WHERE.reset(token)
            return wrapper

        monkeypatch.setitem(pipeline._HANDLERS, intent, make())
    spy(monkeypatch, trace, recall, "handle_recall", "path:State-based answers:handle_recall")
    spy(monkeypatch, trace, clarify, "handle_unclear", "path:Clarify:handle_unclear")
    scoped(monkeypatch, trace, conversation_answerer, "answer_question", "answerer")
    spy(monkeypatch, trace, conversation_answerer, "answer_question", "path:Conversation answerer")
    spy(monkeypatch, trace, llm_client, "generate_json", "answerer:llm-call-2")
    spy(monkeypatch, trace, validators, "validate_conversation_answer", "answerer:check")
    spy(monkeypatch, trace, owner_contact, "handle_owner_contact", "path:Owner contact")
    spy(monkeypatch, trace, pricing_client, "validate_plans", "owner:plans-check")
    spy(monkeypatch, trace, search, "handle_city_change_answer", "path:City-change answer")
    spy(monkeypatch, trace, search, "_ask_city_change", "path:City-change question")
    spy(monkeypatch, trace, search, "_check_arguments", "search:parameter-guardrail")
    spy(monkeypatch, trace, rooms_client, "_mcp_request", "search:mcp-tool")
    spy(monkeypatch, trace, rooms_client, "validate_room_data", "search:api-data-guardrail")
    spy_returned(monkeypatch, trace, reuse, "get_answer_platform_question", "platform:answer")
    spy_by_scope(monkeypatch, trace, reply_safety, "find_unsafe", "links-html")
    scoped(monkeypatch, trace, pipeline, "_output_ok", "response")  # find_unsafe inside it is labelled response:links-html
    spy(monkeypatch, trace, pipeline, "_output_ok", "response")
    spy(monkeypatch, trace, LLMConversationState, "record_turn", "save:record_turn")
    spy(monkeypatch, trace, metrics_mod, "record", "obs:metrics")
    spy(monkeypatch, trace, pipeline, "_record_observability", "obs")

    real_turn = store.turn

    @contextlib.asynccontextmanager
    async def turn(*a, **k):
        async with real_turn(*a, **k) as state:
            trace.mark("state_load")
            try:
                yield state
            finally:
                trace.mark("state_save")  # the lock is released, and a shared store saves, right after this

    monkeypatch.setattr(store, "turn", turn)


class Rig:
    def __init__(self, pipe, trace):
        self.pipe, self.trace = pipe, trace

    def setup(self, message, llm=None, answer=None):
        """A turn that only prepares state: not part of any asserted trace."""
        result = self.pipe.turn(message, llm, answer)
        self.trace.take()
        return result

    def run(self, turn_id, message, llm=None, answer=None, expect=None, collapse=()):
        """One traced turn through ``pipeline.handle_turn``. ``expect`` is the order the diagram draws."""
        self.trace.take()
        result = self.pipe.turn(message, llm, answer)
        events = [e for e in self.trace.take() if not e.startswith("inner:")]
        record(turn_id, events)
        if expect is not None:
            assert events == expect, f"turn {turn_id}: the code no longer follows the diagram\n  drawn: {expect}\n  ran:   {events}"
        return result, events


def record(turn_id: str, events: list[str]) -> None:
    TRACES[turn_id] = events
    for label in events:
        REACHED[label].add(turn_id)
    print(f"\n[{turn_id}] " + " -> ".join(events))


# the path the diagram draws, as labels
PRE = ["normalise", "state_load", "security:extra", "security:app", "gate"]
# validate_slots is the first statement of _decide_action: the router function is entered first, the validators run inside it
DECIDE = ["llm:prompt", "llm:decide", "router", "validate"]
# "response:links-html" is reply_safety.find_unsafe, the first statement inside _output_ok (labelled by scope; other
# boxes that call find_unsafe or extra_security, such as the chat handler and the answerer, get their own prefix)
POST = ["response", "response:links-html", "save:record_turn", "state_save", "obs:metrics", "obs"]


def path(*middle: str) -> list[str]:
    return PRE + DECIDE + list(middle) + POST


@pytest.fixture(autouse=True)
def _restore_switches(monkeypatch):
    """The turns here switch the search checks and the gate on (as configure_production does); nothing may leak to other files."""
    for name in ("_room_data_check", "_search_arg_check", "_semantic_gate"):
        monkeypatch.setattr(switches, name, getattr(switches, name))
    yield


@pytest.fixture
def rig(pipe, monkeypatch):
    trace = Trace()
    install(monkeypatch, trace, pipe.store)
    return Rig(pipe, trace)


@pytest.fixture
def pricing(monkeypatch):
    """A fake pricing API. Pricing reads its offline fixture while the offline-rooms variable is set, so a test calls
    ``pricing.online()`` after its offline room search and before the turn it traces."""
    fake = FakeApi()
    monkeypatch.setattr(pricing_client, "_httpx_get", fake)
    monkeypatch.setattr(pricing_client, "PRICING_BACKOFF_S", 0.0)
    fake.online = lambda: monkeypatch.delenv("LLM_APPROACH_OFFLINE_ROOMS", raising=False)
    return fake


FILTERED = dict(city="Bangalore", bhk=2, max_price=20000, amenity="wifi")


# ======================================================================= A  new room search (live path: MCP tool)


def test_A_new_search_goes_through_mcp_tool_rooms_api_and_api_data_guardrail(rig, monkeypatch):
    from llm_approach.tools import mcp_search

    monkeypatch.delenv(config.OFFLINE_ROOMS_ENV, raising=False)
    monkeypatch.delenv(config.SEARCH_TRANSPORT_ENV, raising=False)
    monkeypatch.setenv("KANGAROO_TOKEN", "fake-rooms-credential")
    monkeypatch.setattr(mcp_search, "HOMILIVO_API_RETRY_BACKOFF_SECONDS", 0.0)
    switches.configure(room_data_check=True, search_arg_check=True)  # what configure_production sets
    fake = FakeUrlopen()
    fake.script = [{"data": [good_room(1), good_room(2)], "meta": {"total": 2, "hasNextPage": False}}]

    def urlopen(request, timeout=None):
        rig.trace.mark("search:rooms-api")  # the fake stands where the HTTP GET leaves the process
        return fake(request, timeout)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    result, _ = rig.run(
        "A", "rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"),
        expect=path("path:Room search:handle_new_search", "search:parameter-guardrail", "search:mcp-tool",
                    "search:rooms-api", "search:api-data-guardrail"),
    )
    assert result.metrics["tool_called"] and result.rooms


# ======================================================================= B  refinement (budget)


def test_B_refinement_by_budget(rig):
    rig.setup("2 bhk in bangalore", dec("NEW_SEARCH", city="Bangalore", bhk=2))
    result, _ = rig.run(
        "B", "under 15k", dec("REFINEMENT", op="CHANGE_PRICE", max_price=15000),
        expect=path("path:Room search:handle_refinement", "search:parameter-guardrail"),
    )
    assert result.metrics["action"] == "refinement" and result.metrics["tool_called"]


# ======================================================================= C  city change question and its two answers


def _filtered(rig):
    rig.setup("2 bhk in bangalore under 20k with wifi", dec("NEW_SEARCH", **FILTERED))


def test_C1_a_different_city_after_a_filtered_search_asks_and_does_not_search(rig):
    _filtered(rig)
    result, events = rig.run(
        "C1", "in jaipur", dec("REFINEMENT", city="Jaipur"),
        expect=path("path:Room search:handle_refinement", "path:City-change question"),
    )
    assert result.metrics["action"] == "ask_city_change" and result.metrics["tool_called"] is False
    assert "search:parameter-guardrail" not in events and "path:Clarify:handle_unclear" not in events  # not the Clarify box
    assert rig.pipe.state.pending_question["type"] == "city_change"


def test_C2_reply_same_filters_runs_the_answer_check_first_then_a_refinement_search(rig):
    _filtered(rig)
    rig.setup("in jaipur", dec("REFINEMENT", city="Jaipur"))
    result, events = rig.run(
        "C2", "same filters", dec("UNCLEAR"),
        expect=PRE + DECIDE + ["path:City-change answer", "search:parameter-guardrail"] + POST,
    )
    assert result.metrics["action"] == "refinement" and result.metrics["tool_called"]
    assert "path:Clarify:handle_unclear" not in events  # the UNCLEAR intent never reached the clarifier
    assert rig.pipe.searches[-1]["city"] == "Jaipur" and rig.pipe.searches[-1]["max_price"] == 20000


def test_C3_reply_new_search_runs_a_fresh_search(rig):
    _filtered(rig)
    rig.setup("in jaipur", dec("REFINEMENT", city="Jaipur"))
    result, _ = rig.run(
        "C3", "new search", dec("GENERAL_CHAT"),
        expect=PRE + DECIDE + ["path:City-change answer", "search:parameter-guardrail"] + POST,
    )
    assert result.metrics["action"] == "new_search"
    assert rig.pipe.searches[-1]["city"] == "Jaipur" and rig.pipe.searches[-1]["max_price"] is None


# ======================================================================= D  follow-up on a displayed room


def test_D_follow_up_on_a_displayed_room(rig):
    rig.setup("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    result, _ = rig.run(
        "D", "price of the first one", dec("ROOM_FOLLOWUP", op="ROOM_PRICE", ref=1),
        expect=path("path:State-based answers:handle_followup"),
    )
    assert result.success and result.metrics["tool_called"] is False


# ======================================================================= E  recall


def test_E_recall(rig):
    d = dec("GENERAL_CHAT", reply="Nice to meet you!")
    d["user_facts"] = {"name": "ziyad"}
    rig.setup("my name is ziyad", d)
    result, _ = rig.run(
        "E", "what is my name?", {"intent": "CONVERSATION_RECALL", "confidence": 0.95, "recall_target": "name"},
        expect=path("path:State-based answers:handle_recall"),
    )
    assert "Ziyad" in result.reply


# ======================================================================= F  chat question: the second LLM call


def test_F_chat_question_makes_a_second_llm_call_and_checks_the_answer(rig):
    rig.setup("my college is VIT", dec("GENERAL_CHAT", reply="Nice."))
    chat = {"intent": "CONVERSATION_QUESTION", "confidence": 0.95}
    answer = {"found": True, "answer": "Your college is VIT.", "evidence_turns": [1]}
    result, _ = rig.run(
        "F", "what's my college?", chat, answer,
        expect=path("path:Conversation answerer", "answerer:llm-call-2", "answerer:check",
                    "answerer:security:extra", "answerer:links-html"),
    )
    assert result.reply == "Your college is VIT." and rig.pipe.llm_calls == 2 and rig.pipe.answer_calls == 1  # one decision call per turn + the answerer's call


# ======================================================================= G  platform FAQ


def test_G_platform_faq_is_answered_from_the_knowledge_base_not_the_model(rig):
    result, _ = rig.run(
        "G", "how does homilivo earn money", dec("PLATFORM_INFO", sq="how does homilivo earn money"),
        expect=path("path:Platform FAQ:handle_platform_info", "platform:answer"),
    )
    assert result.reply == "Homilivo earns from owner subscriptions."


# ======================================================================= H  owner contact (English and Hinglish)


@pytest.mark.parametrize("turn_id,message", [("H1", "how do I contact the owner"), ("H2", "owner ka number chahiye")])
def test_H_owner_contact_is_a_post_llm_check_not_an_intent(rig, pricing, turn_id, message):
    rig.setup("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    pricing.online()
    result, events = rig.run(
        turn_id, message, dec("UNSUPPORTED_ACTION", reply="Sure, call 99999"),
        expect=path("path:Owner contact", "owner:plans-check"),
    )
    assert result.metrics["action"] == "owner_contact" and pricing.cities() == ["bangalore"]
    assert not any(e.startswith("path:State-based") for e in events)  # the intent's own handler never ran


def test_H3_the_same_check_applies_to_platform_info_and_skips_the_faq(rig, pricing):
    rig.setup("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    pricing.online()
    result, events = rig.run(
        "H3", "how can I reach the owner", dec("PLATFORM_INFO"),
        expect=path("path:Owner contact", "owner:plans-check"),
    )
    assert result.metrics["action"] == "owner_contact" and "platform:answer" not in events


def test_H4_the_check_is_only_for_those_two_intents(rig, pricing):
    rig.setup("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    pricing.online()
    result, events = rig.run(
        "H4", "how do I contact the owner", dec("GENERAL_CHAT", reply="Ok."),
        expect=path("path:State-based answers:handle_general_chat"),
    )
    assert result.metrics["action"] == "general_chat" and pricing.calls == []


def test_H5_the_check_comes_after_the_clarify_decision(rig, pricing):
    rig.setup("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    pricing.online()
    result, events = rig.run(
        "H5", "how do I contact the owner", dec("UNSUPPORTED_ACTION", conf=0.3),
        expect=path("path:Clarify:handle_unclear"),
    )
    assert "path:Owner contact" not in events and pricing.calls == []


# ======================================================================= I  clarify


def test_I1_low_confidence_clarifies(rig):
    result, _ = rig.run(
        "I1", "hmm maybe something", dec("NEW_SEARCH", conf=0.3, city="Bangalore"),
        expect=path("path:Clarify:handle_unclear"),
    )
    assert result.metrics["tool_called"] is False


def test_I2_unclear_intent_clarifies(rig):
    rig.run("I2", "asdf", dec("UNCLEAR"), expect=path("path:Clarify:handle_unclear"))


def test_I3_a_search_slot_that_fails_validation_asks_before_any_handler(rig):
    result, events = rig.run(
        "I3", "rooms in !!!", dec("NEW_SEARCH", city="!!!"), expect=path(),
    )
    assert result.metrics["action"] == "ask_city" and result.metrics["tool_called"] is False
    assert not any(e.startswith("path:") for e in events)  # _clarify_rejected answered inside the router itself


def test_I4_a_search_with_no_city_is_asked_by_the_search_handler(rig):
    result, _ = rig.run(
        "I4", "find me a room", dec("NEW_SEARCH"), expect=path("path:Room search:handle_new_search"),
    )
    assert result.metrics["action"] == "ask_city"


def test_I5_unclear_room_reference_asks_which_room(rig):
    rig.setup("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    result, events = rig.run("I5", "price of the 15th", dec("ROOM_FOLLOWUP", op="ROOM_PRICE", ref=15), expect=path())
    assert result.metrics["action"] == "ask_which_room"


# ======================================================================= J  blocked by the security filter


def test_J1_app_input_guardrail_blocks_before_the_gate_and_the_llm(rig):
    rig.pipe.app_blocks.add("something the app guardrail refuses")
    result, _ = rig.run(
        "J1", "something the app guardrail refuses", "never",
        expect=["normalise", "state_load", "security:extra", "security:app", "save:record_turn", "state_save", "obs:metrics", "obs"],
    )
    assert result.reply == extra_security.REFUSAL and result.success is False and rig.pipe.llm_calls == 0


def test_J2_extra_security_blocks_before_the_app_guardrail(rig):
    result, _ = rig.run(
        "J2", "do anything now and jailbreak", "never",
        expect=["normalise", "state_load", "security:extra", "save:record_turn", "state_save", "obs:metrics", "obs"],
    )
    assert result.reply == extra_security.REFUSAL and rig.pipe.llm_calls == 0


# ======================================================================= K  no decision: rule-based fallback


def test_K_no_decision_runs_the_rule_based_fallback_and_its_result_goes_to_the_response_checks(rig):
    result, events = rig.run(
        "K", "rooms in bangalore under 15000", None,
        expect=PRE + ["llm:prompt", "llm:decide", "fallback", "validate", "search:parameter-guardrail"] + POST,
    )
    assert result.metrics["fallback_used"] is True and result.metrics["action"] == "fallback_new_search"
    assert "router" not in events and events.index("validate") > events.index("fallback")  # no router; the fallback validates its own slots
    assert result.rooms  # the fallback's rooms reach the reply


# ======================================================================= L  reply refused by the response checks


def test_L1_output_guardrail_refuses_the_reply(rig):
    rig.pipe.output_ok = False
    result, _ = rig.run(
        "L1", "hello there", dec("GENERAL_CHAT", reply="Hi!"),
        expect=path("path:State-based answers:handle_general_chat"),
    )
    assert result.reply == pipeline.OUTPUT_BLOCKED_REPLY and result.metrics["action"] == "output_blocked"
    assert result.rooms == []


def test_L2_a_link_that_slips_past_an_action_is_refused_by_the_response_checks(rig, monkeypatch):
    """A handler that (wrongly) returns a link: simulated by replacing the dispatch entry of GENERAL_CHAT."""
    from llm_approach.actions import ActionResult

    monkeypatch.setitem(pipeline._HANDLERS, Intent.GENERAL_CHAT,
                        lambda *a, **k: ActionResult("Visit http://evil.example/now for rooms", None, "general_chat"))
    result, _ = rig.run(
        "L2", "hello there", dec("GENERAL_CHAT", reply="Hi"), expect=path(),
    )
    assert result.reply == pipeline.OUTPUT_BLOCKED_REPLY and result.metrics["action"] == "output_blocked"
    assert rig.pipe.output_calls == []  # refused by reply_safety before the copied output guardrail was asked


# ======================================================================= M  message over 500 characters


def test_M1_over_500_characters_stops_in_the_pipeline_after_normalising_with_only_a_log_write(rig):
    result, events = rig.run(
        "M1", "a" * 501, "never",
        expect=["normalise", "state_load", "state_save", "obs:metrics", "obs"],
    )
    assert result.reply == pipeline.TOO_LONG_REPLY and result.success is False
    assert "security:extra" not in events and rig.pipe.llm_calls == 0 and any(e.get("text") == pipeline.TOO_LONG_REPLY for e in rig.pipe.state.transcript)


def test_M2_the_pipeline_measures_the_normalised_text_NFKC_can_make_it_longer(rig):
    text = "ﬁ" * 300  # 300 ligature characters become 600 after NFKC
    assert len(text) <= 500 < len(textclean.normalize_input(text))
    result, _ = rig.run("M2", text, "never", expect=["normalise", "state_load", "state_save", "obs:metrics", "obs"])
    assert result.reply == pipeline.TOO_LONG_REPLY


def test_M3_the_pipeline_check_is_on_the_normalised_text_invisible_characters_do_not_count(rig):
    text = "a" * 499 + "​​"  # 501 raw characters, 499 after the invisible ones are removed
    assert len(text) == 501 and len(textclean.normalize_input(text)) == 499
    result, _ = rig.run("M3", text, dec("GENERAL_CHAT", reply="Ok."), expect=path("path:State-based answers:handle_general_chat"))
    assert result.reply != pipeline.TOO_LONG_REPLY


# ======================================================================= N  turn over the 40 s limit


def test_N_a_turn_over_the_deadline_is_cancelled_saved_and_never_reaches_the_response_checks(rig, monkeypatch):
    pipe, trace = rig.pipe, rig.trace
    monkeypatch.setattr(pipeline, "_store", pipe.store)
    spy(monkeypatch, trace, observability, "record_turn_event", "obs:turn-event")
    seconds = []
    real_run = turn_deadline.run

    async def run_spy(awaitable, secs, **k):
        trace.mark("server:turn-deadline")
        seconds.append(secs)
        return await real_run(awaitable, secs, **k)

    monkeypatch.setattr(turn_deadline, "run", run_spy)

    async def main():
        started, never = asyncio.Event(), asyncio.Event()

        async def block():
            started.set()
            await never.wait()

        async def fake_sleep(secs):
            await started.wait()  # fires as soon as the slow stage has started: no real 40 s

        monkeypatch.setattr(server_app, "_deadline_sleep", fake_sleep)
        pipe.script.append(block)
        trace.take()
        return await server_app.run_turn(pipe.sid, "rooms in bangalore")

    turn = asyncio.run(main())
    events = trace.take()
    record("N", events)
    assert seconds == [40.0] == [config.turn_deadline_s()]
    assert turn.reply == pipeline.DEADLINE_REPLY and turn.success is False
    assert events == ["server:turn-deadline"] + PRE + ["llm:prompt", "llm:decide", "save:record_turn", "state_save", "obs:metrics", "obs", "obs:turn-event"]
    assert "response" not in events and "validate" not in events  # cancelled mid-LLM: nothing after it ran


# ======================================================================= O  semantic gate: the exact conditions


class FakeClassifier:
    def __init__(self, label):
        self.label, self.calls = label, []

    def __call__(self, text):
        self.calls.append(text)
        return self.label


@pytest.fixture
def gate_mode(monkeypatch):
    def set_mode(mode, label="OUT_OF_SCOPE"):
        clf = FakeClassifier(label)
        monkeypatch.setattr(reuse, "get_semantic_classifier", lambda: clf)
        switches.configure(semantic_gate=mode)
        return clf

    yield set_mode
    switches.configure(semantic_gate="off")


def test_O1_enforce_blocks_an_off_topic_first_message_with_no_llm_call(rig, gate_mode):
    gate_mode("enforce")
    result, _ = rig.run(
        "O1", "tell me a joke", "never",
        expect=PRE + ["gate:classify", "response", "response:links-html", "save:record_turn", "state_save", "obs:metrics", "obs"],
    )
    assert result.metrics["action"] == "semantic_block" and rig.pipe.llm_calls == 0


def test_O2_shadow_records_but_never_blocks(rig, gate_mode):
    clf = gate_mode("shadow")
    rig.run("O2", "tell me a joke", dec("GENERAL_CHAT", reply="Ha."),
            expect=PRE + ["gate:classify"] + DECIDE + ["path:State-based answers:handle_general_chat"] + POST)
    assert clf.calls and rig.pipe.llm_calls == 1


def test_O3_off_skips_the_classifier(rig, gate_mode):
    clf = gate_mode("off")
    rig.run("O3", "tell me a joke", dec("GENERAL_CHAT", reply="Ha."), expect=path("path:State-based answers:handle_general_chat"))
    assert clf.calls == []


def test_O4_with_rooms_on_screen_the_gate_does_not_classify(rig, gate_mode):
    rig.setup("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    clf = gate_mode("enforce")
    rig.run("O4", "does it have wifi", dec("GENERAL_CHAT", reply="Ok."), expect=path("path:State-based answers:handle_general_chat"))
    assert clf.calls == []


def test_O5_with_a_pending_question_the_gate_does_not_classify(rig, gate_mode):
    rig.setup("find me a room", dec("NEW_SEARCH"))  # asks for the city: a pending question, no rooms
    assert rig.pipe.state.pending_question and not rig.pipe.state.rooms_on_screen
    clf = gate_mode("enforce")
    rig.run("O5", "bangalore", dec("NEW_SEARCH", city="Bangalore"), expect=path("path:Room search:handle_new_search", "search:parameter-guardrail"))
    assert clf.calls == []


def test_O6_without_the_full_guardrails_the_gate_does_not_classify(rig, gate_mode):
    clf = gate_mode("enforce")
    pipeline.configure(full_guardrails=False)
    try:
        rig.pipe.turn("tell me a joke", dec("GENERAL_CHAT", reply="Ha."))
    finally:
        pipeline.configure(full_guardrails=True)
    assert clf.calls == []
    record("O6", rig.trace.take())


# ======================================================================= R  degraded-memory notice (omitted from the diagram)


def test_R_degraded_notice_is_added_after_the_response_checks_and_before_the_save(rig, monkeypatch):
    monkeypatch.setattr(LLMConversationState, "degraded_notice", True, raising=False)
    result, _ = rig.run(
        "R", "hello there", dec("GENERAL_CHAT", reply="Hi!"),
        expect=path("path:State-based answers:handle_general_chat"),
    )
    assert result.reply.endswith(pipeline.DEGRADED_NOTICE)


# ======================================================================= S  the real decision call: retry on bad JSON


def test_S_real_decide_retries_once_on_bad_json_and_parses_with_the_pydantic_model(monkeypatch, fake_gemini):
    """Not the scripted stand-in: the real ``llm_client.decide`` against a fake Gemini client."""
    from llm_approach.state import SessionStore

    trace = Trace()
    store = SessionStore()
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    pipeline.configure(full_guardrails=True)
    install(monkeypatch, trace, store)
    spy(monkeypatch, trace, llm_client, "_complete_with_5xx_retry", "llm:provider-call")
    spy(monkeypatch, trace, llm_client, "parse_decision", "llm:pydantic-parse")
    spy(monkeypatch, trace, providers, "_complete_gemini", "llm:gemini-sdk")
    client = fake_gemini(["this is not json", json.dumps(dec("NEW_SEARCH", city="Bangalore"))])
    result = asyncio.run(pipeline.handle_turn(str(uuid.uuid4()), "rooms in bangalore", store=store))
    events = trace.take()
    record("S", events)
    assert len(client.calls) == 2  # one retry, no third call
    assert events == PRE + [
        "llm:prompt", "llm:decide", "llm:provider-call", "llm:gemini-sdk", "llm:pydantic-parse",
        "llm:provider-call", "llm:gemini-sdk", "llm:pydantic-parse",
        "router", "validate", "path:Room search:handle_new_search", "search:parameter-guardrail",
    ] + POST
    assert result.metrics["action"] == "new_search"


# ======================================================================= T  the server side: entry checks and reply streaming


@pytest.fixture
def served(pipe, monkeypatch):
    from llm_approach.security.rate_limit import InMemoryRateLimiter

    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")
    monkeypatch.setenv("STREAM_CHUNK_DELAY_SECONDS", "0")
    monkeypatch.setenv("KANGAROO_TOKEN", "fake-rooms-credential")
    limiter = InMemoryRateLimiter()
    monkeypatch.setattr(server_app._rl, "get_rate_limiter", lambda: limiter)
    monkeypatch.setattr(pipeline, "_store", pipe.store)
    trace = Trace()
    install(monkeypatch, trace, pipe.store)

    real_hit = limiter.hit

    def hit(key, *a, **k):
        if str(key).startswith("newsession:"):
            trace.mark("entry:new-session-limit")
        return real_hit(key, *a, **k)

    monkeypatch.setattr(limiter, "hit", hit)
    spy(monkeypatch, trace, server_app._rl, "check_rate_limit", "entry:ip-rate-limit")
    spy(monkeypatch, trace, server_app._rl, "check_session_rate_limit", "entry:session-limit")
    spy(monkeypatch, trace, server_app, "normalize_session_id", "entry:session-id-format")
    spy(monkeypatch, trace, server_app, "run_turn", "server:run_turn")
    spy(monkeypatch, trace, turn_deadline, "run", "server:turn-deadline")
    spy(monkeypatch, trace, server_app, "build_done", "server:build_done")
    spy(monkeypatch, trace, pipeline, "handle_turn", "pipeline:handle_turn")

    real_send = WebSocket.send_json

    async def send_json(self, data, *a, **k):
        if isinstance(data, dict) and data.get("event") in ("typing", "done", "error"):
            trace.mark(f"ws:{data['event']}")
        elif isinstance(data, dict) and data.get("event") == "token":
            if "ws:token" not in trace.events[-1:]:
                trace.mark("ws:token")
        return await real_send(self, data, *a, **k)

    monkeypatch.setattr(WebSocket, "send_json", send_json)
    with TestClient(server_app.create_app()) as client:
        yield client, trace, pipe


def ws_turn(ws, **payload):
    ws.send_json(payload)
    events = []
    while True:
        events.append(ws.receive_json())
        if events[-1]["event"] in ("done", "error"):
            return events


def test_T1_a_websocket_turn_in_the_drawn_order_entry_checks_to_reply_streaming(served):
    client, trace, pipe = served
    pipe.script.append(dec("NEW_SEARCH", city="Bangalore"))
    with client.websocket_connect("/ws/chat?token=demo-token") as ws:
        trace.take()
        events = ws_turn(ws, message="rooms in bangalore", session_id=str(uuid.uuid4()), mode="fast")
    seen = trace.take()
    record("T1", seen)
    assert events[0]["event"] == "typing" and events[-1]["event"] == "done" and events[-1]["room_count"] > 0
    assert [e["event"] for e in events[1:-1]] and all(e["event"] == "token" for e in events[1:-1])
    assert seen == [
        "entry:session-id-format", "entry:ip-rate-limit", "entry:session-limit", "entry:new-session-limit",
        "ws:typing", "server:run_turn", "server:turn-deadline", "pipeline:handle_turn",
    ] + path("path:Room search:handle_new_search", "search:parameter-guardrail") + [
        "server:build_done", "ws:token", "ws:done",
    ]


def test_T2_the_ip_limit_is_checked_on_connect_and_again_per_message(served):
    client, trace, pipe = served
    pipe.script.append(dec("GENERAL_CHAT", reply="Hi"))
    trace.take()
    with client.websocket_connect("/ws/chat?token=demo-token") as ws:
        ws_turn(ws, message="hi", session_id=str(uuid.uuid4()), mode="fast")
    seen = trace.take()
    record("T2", seen)
    assert seen[0] == "entry:ip-rate-limit" and seen.count("entry:ip-rate-limit") == 2


def test_T3_entry_refusals_never_reach_the_pipeline(served, monkeypatch):
    client, trace, pipe = served
    # missing token: closed with 4401 before anything else
    trace.take()
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()
    assert trace.take() == []
    # over 500 characters, blank, or a bad mode: an error event, no pipeline
    with client.websocket_connect("/ws/chat?token=demo-token") as ws:
        trace.take()
        assert ws_turn(ws, message="a" * 501, session_id=None, mode="fast")[-1]["event"] == "error"
        assert ws_turn(ws, message="   ", session_id=None, mode="fast")[-1]["event"] == "error"
        assert ws_turn(ws, message="hi", session_id=None, mode="nope")[-1]["event"] == "error"
        seen = trace.take()
    record("T3", seen)
    assert "server:run_turn" not in seen and "pipeline:handle_turn" not in seen
    # server still loading: the user is told, the pipeline is not entered
    monkeypatch.setitem(server_app._startup, "loading", True)
    with client.websocket_connect("/ws/chat?token=demo-token") as ws:
        trace.take()
        events = ws_turn(ws, message="hi", session_id=str(uuid.uuid4()), mode="fast")
        seen = trace.take()
    record("T3-loading", seen)
    assert events[-1]["event"] == "error" and "server:run_turn" not in seen


def test_T3b_a_session_over_its_limit_is_refused_without_the_pipeline(served, monkeypatch):
    from llm_approach.security.rate_limit import RateLimitResult

    client, trace, pipe = served
    monkeypatch.setattr(server_app._rl, "check_session_rate_limit", lambda sid: RateLimitResult(False, 0, 5.0))
    with client.websocket_connect("/ws/chat?token=demo-token") as ws:
        trace.take()
        events = ws_turn(ws, message="hi", session_id=str(uuid.uuid4()), mode="fast")
    seen = trace.take()
    record("T3b", seen)
    assert events[-1]["event"] == "error" and "server:run_turn" not in seen and "pipeline:handle_turn" not in seen


def test_T4_post_chat_is_the_same_turn(served):
    client, trace, pipe = served
    pipe.script.append(dec("GENERAL_CHAT", reply="Hi there"))
    trace.take()
    r = client.post("/chat", json={"message": "hello", "session_id": str(uuid.uuid4())}, headers={"Authorization": "Bearer demo-token"})
    seen = trace.take()
    record("T4", seen)
    assert r.status_code == 200
    assert seen[seen.index("server:run_turn"):seen.index("server:run_turn") + 3] == ["server:run_turn", "server:turn-deadline", "pipeline:handle_turn"]
    assert "ws:typing" not in seen  # no streaming on POST


# ======================================================================= U  the lock is real: state is loaded and saved under it


def test_U_the_state_is_loaded_and_saved_by_the_session_store_and_a_redis_store_works_the_same(pipe, monkeypatch):
    trace = Trace()
    store = RedisSessionStore(redis_client=fakeredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True))
    monkeypatch.setattr(pipeline, "_store", store)
    install(monkeypatch, trace, store)
    saves = []
    real_save = store._save_after_turn if hasattr(store, "_save_after_turn") else None
    if real_save is not None:
        def save_spy(*a, **k):
            trace.mark("redis:save")
            return real_save(*a, **k)
        monkeypatch.setattr(store, "_save_after_turn", save_spy)
    pipe.script.append(dec("GENERAL_CHAT", reply="Hi"))
    asyncio.run(pipeline.handle_turn(str(uuid.uuid4()), "hello", store=store))
    events = trace.take()
    record("U", events)
    assert events[:2] == ["normalise", "state_load"]
    assert events.index("save:record_turn") < events.index("state_save")
    if real_save is not None:
        assert "redis:save" in events and events.index("redis:save") > events.index("save:record_turn")


# ======================================================================= the coverage map: which turn proves which box


# diagram box -> (labels that must have been reached, ...). The turn ids come from REACHED.
BOXES = {
    "Entry checks: IP rate limit": ["entry:ip-rate-limit"],
    "Entry checks: session-id format": ["entry:session-id-format"],
    "Entry checks: session limit": ["entry:session-limit"],
    "Entry checks: new-session limit": ["entry:new-session-limit"],
    "Entry checks: 40 s turn limit": ["server:turn-deadline"],
    "FastAPI route run_turn": ["server:run_turn", "pipeline:handle_turn"],
    "Input normalisation": ["normalise"],
    "Session state load": ["state_load"],
    "Security filters": ["security:extra", "security:app"],
    "Semantic gate": ["gate", "gate:classify"],
    "Prompt build + LLM decision": ["llm:prompt", "llm:decide"],
    "  retry once on bad JSON / Pydantic parse": ["llm:pydantic-parse", "llm:gemini-sdk"],
    "Validators": ["validate"],
    "Intent router": ["router"],
    "Platform FAQ": ["path:Platform FAQ:handle_platform_info"],
    "  Grounded answer (KB call)": ["platform:answer"],
    "State-based answers": ["path:State-based answers:handle_followup", "path:State-based answers:handle_recall",
                            "path:State-based answers:handle_general_chat"],
    "Conversation answerer": ["path:Conversation answerer", "answerer:llm-call-2"],
    "Answer check": ["answerer:check"],
    "Room search": ["path:Room search:handle_new_search", "path:Room search:handle_refinement"],
    "Parameter guardrail": ["search:parameter-guardrail"],
    "MCP tool": ["search:mcp-tool"],
    "Homilivo rooms API (fake urlopen)": ["search:rooms-api"],
    "API-data guardrail": ["search:api-data-guardrail"],
    "Owner contact": ["path:Owner contact"],
    "Plans check": ["owner:plans-check"],
    "Clarify": ["path:Clarify:handle_unclear"],
    "City-change question": ["path:City-change question"],
    "City-change answer": ["path:City-change answer"],
    "Rule-based fallback": ["fallback"],
    "Response checks": ["response", "response:links-html"],
    "State save": ["save:record_turn", "state_save"],
    "Observability": ["obs", "obs:metrics"],
    "WebSocket reply (typing/token/done)": ["ws:typing", "ws:token", "ws:done"],
}
NOT_REACHABLE_HERE = {  # drawn boxes no test turn can reach offline, and why
    "Chroma FAQ index": "the knowledge answer is a fake here; the embedding model and Chroma are not loaded (test_rag_* cover the store alone)",
    "Pricing API (HTTP)": "pricing is a fake at the httpx boundary; the real GET is not made",
    "Gemini API (real network)": "turn S reaches the SDK boundary with a fake client only",
    "Session store: Redis server": "fakeredis stands in for a Redis server (turn U)",
    "React chatbot / User": "not in this repo's backend; see CROSS_CHECK.md",
}


def test_zz_coverage_report():
    """Prints which turn proves which box, and fails if a box that was proven stops being reached. Needs the whole file."""
    if len(TRACES) < 30:
        pytest.skip("coverage needs the whole file to have run")
    print("\n\nBOX -> TURNS THAT REACH IT")
    missing = []
    for box_name, labels in BOXES.items():
        turns = sorted(set().union(*(REACHED.get(label, set()) for label in labels)))
        unreached = [label for label in labels if not REACHED.get(label)]
        print(f"  {box_name:48s} {', '.join(turns) if turns else '-'}" + (f"   (labels not reached: {unreached})" if unreached else ""))
        if unreached:
            missing.append((box_name, unreached))
    for box_name, why in NOT_REACHABLE_HERE.items():
        print(f"  {box_name:48s} UNVERIFIED: {why}")
    assert not missing, f"boxes with a label no turn reached: {missing}"
