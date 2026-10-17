"""Phase 4 pipeline tests. Offline: scripted LLM, fake guardrails, fixture rooms, no torch."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

from llm_approach import config, pipeline, prompts
from llm_approach.eval import run_scenarios
from llm_approach.schema import Intent
from llm_approach.state import ERROR_REPLY
from llm_approach.tests.conftest import ROOT, dec

SCENARIOS = run_scenarios.load_scenarios()


# ---- S1 to S23 in stub mode ----------------------------------------------------


def test_all_scenarios_exist():
    assert [s["id"] for s in SCENARIOS] == [f"S{n}" for n in range(1, 24)]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_scenario_stub(scenario):
    reports = asyncio.run(run_scenarios.run_scenario(scenario, "stub"))
    assert len(reports) == len(scenario["turns"])
    failures = [run_scenarios.format_report(r) for r in reports if not r.passed]
    assert not failures, "\n".join(failures)


def test_scenario_checks_can_fail():
    """A wrong expectation must be reported, otherwise the runner proves nothing."""
    scenario = json.loads(json.dumps(SCENARIOS[3]))  # S4
    scenario["turns"][0]["expect"]["action"] = "new_search"
    scenario["turns"][0]["expect"]["reply_contains"] = ["welcome"]
    reports = asyncio.run(run_scenarios.run_scenario(scenario, "stub"))
    assert not reports[0].passed
    assert {c.name for c in reports[0].checks if not c.ok} >= {"action", "reply has 'welcome'"}


def test_live_mode_skips_stub_only_scenarios():
    s16 = next(s for s in SCENARIOS if s["id"] == "S16")
    reports = asyncio.run(run_scenarios.run_scenario(s16, "live"))
    assert all(r.skipped for r in reports)


# ---- clarify / validation gates --------------------------------------------------


def test_low_confidence_clarifies_and_does_not_search(pipe):
    r = pipe.turn("rooms in pune", dec("NEW_SEARCH", conf=0.3, city="Pune"))
    assert r.metrics["action"] == "clarify" and not pipe.searches
    assert r.reply == "Which city and budget are you looking for?"


def test_just_above_threshold_acts(pipe):
    r = pipe.turn("rooms in bangalore", dec("NEW_SEARCH", conf=config.CONFIDENCE_THRESHOLD, city="Bangalore"))
    assert r.metrics["action"] == "new_search" and len(pipe.searches) == 1


def test_needs_clarification_on_followup_asks_instead_of_acting(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    r = pipe.turn("is it ok", dec("ROOM_FOLLOWUP", op="ROOM_PRICE", nc=True, q="Which room do you mean?"))
    assert r.metrics["action"] == "clarify"


@pytest.mark.parametrize("city", ["pls urgent", "a different area", "I Dont Know", "x", "12345", "<script>"])
def test_invalid_city_never_reaches_rooms_client(pipe, city):
    for intent in ("NEW_SEARCH", "REFINEMENT"):
        r = pipe.turn("rooms please", dec(intent, conf=0.9, city=city))
        assert r.metrics["tool_called"] is False
        assert r.metrics["action"] in ("ask_city", "ask_area")
    assert pipe.searches == []


def test_missing_city_never_reaches_rooms_client(pipe):
    r = pipe.turn("find me a room", dec("NEW_SEARCH", conf=0.9, bhk=2))
    assert r.metrics["action"] == "ask_city" and pipe.searches == []
    assert pipe.state.pending_question["type"] == "city"


def test_rejected_budget_is_asked_not_dropped(pipe):
    r = pipe.turn("2bhk in bangalore under 50", dec("NEW_SEARCH", city="Bangalore", bhk=2, max_price=50))
    assert r.metrics["action"] == "ask_slot" and pipe.searches == []
    assert "budget" in r.reply and "max_price" in r.metrics["slots_rejected"]


def test_reference_to_missing_room_is_asked_not_guessed(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    r = pipe.turn("price of the 15th", dec("ROOM_FOLLOWUP", op="ROOM_PRICE", ref=15))
    assert r.metrics["action"] == "ask_which_room" and "10 rooms" in r.reply
    assert pipe.state.selected_room_index is None


def test_only_one_rooms_call_per_turn(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    pipe.turn("in pune", dec("REFINEMENT", city="Pune"))
    assert len(pipe.searches) == 2  # one per turn


# ---- guardrails and limits ---------------------------------------------------------


def test_output_guardrail_block_gives_safe_reply(pipe):
    pipe.output_ok = False
    r = pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert r.reply == pipeline.OUTPUT_BLOCKED_REPLY
    assert r.rooms is None and r.metrics["error_class"] == "output_blocked"
    assert pipe.state.last_bot_message == pipeline.OUTPUT_BLOCKED_REPLY
    assert pipe.state.history[-1] == {"role": "assistant", "content": pipeline.OUTPUT_BLOCKED_REPLY}


def test_output_guardrail_sees_the_final_reply(pipe):
    r = pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert pipe.output_calls == [r.reply]


def test_501_characters_gets_short_reply_without_llm(pipe):
    r = pipe.turn("a" * 501)
    assert r.reply == pipeline.TOO_LONG_REPLY and r.success is False
    assert pipe.llm_calls == 0 and pipe.security_calls == [] and pipe.state is None


def test_500_characters_is_accepted(pipe):
    r = pipe.turn("rooms " + "x" * 494, dec("UNCLEAR", conf=0.3))
    assert pipe.llm_calls == 1 and r.reply != pipeline.TOO_LONG_REPLY


def test_blank_message_gets_prompt_without_llm(pipe):
    r = pipe.turn("   ")
    assert r.reply == pipeline.EMPTY_REPLY and pipe.llm_calls == 0


# ---- failures -------------------------------------------------------------------------


def test_exception_in_action_gives_friendly_reply_and_consistent_history(pipe, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("database password is hunter2")

    original = pipeline._HANDLERS[Intent.NEW_SEARCH]
    monkeypatch.setitem(pipeline._HANDLERS, Intent.NEW_SEARCH, boom)
    r = pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert r.reply == ERROR_REPLY and r.success is False
    assert "hunter2" not in r.reply and "Traceback" not in r.reply
    assert r.metrics["error_class"] == "RuntimeError"
    assert pipe.state.history == [
        {"role": "user", "content": "rooms in bangalore"},
        {"role": "assistant", "content": ERROR_REPLY},
    ]
    # the session still works afterwards
    monkeypatch.setitem(pipeline._HANDLERS, Intent.NEW_SEARCH, original)
    assert pipe.turn("hi", dec("GENERAL_CHAT", reply="Hi!")).metrics["action"] == "general_chat"
    assert len(pipe.state.history) == 4


def test_exception_in_llm_call_gives_friendly_reply(pipe):
    r = pipe.turn("hello", ValueError("x"))
    assert r.reply == ERROR_REPLY and len(pipe.state.history) == 2


def test_invalid_session_id_gives_friendly_reply(pipe):
    pipe.sid = "not-a-uuid"
    r = pipe.turn("hello", dec("GENERAL_CHAT", reply="Hi!"))
    assert r.reply == ERROR_REPLY and pipe.llm_calls == 0


def test_two_simultaneous_turns_on_one_session_are_serialised(pipe):
    order: list[str] = []

    def scripted(name, decision, delay):
        async def run():
            order.append(f"start {name}")
            await asyncio.sleep(delay)
            order.append(f"end {name}")
            return decision

        return run

    async def both():
        first = asyncio.create_task(
            pipe.turn_async("first message", scripted("one", dec("GENERAL_CHAT", reply="One."), 0.05))
        )
        await asyncio.sleep(0)  # let the first turn take the lock
        second = asyncio.create_task(
            pipe.turn_async("second message", scripted("two", dec("GENERAL_CHAT", reply="Two."), 0))
        )
        return await asyncio.gather(first, second)

    asyncio.run(both())
    # script is popped in call order; the second LLM call must not start before the first ends
    assert order == ["start one", "end one", "start two", "end two"]
    assert [m["content"] for m in pipe.state.history if m["role"] == "user"] == ["first message", "second message"]
    assert [m["role"] for m in pipe.state.history] == ["user", "assistant", "user", "assistant"]


# ---- S8 prompt check: "does it have wifi" after "which one is cheapest" ---------------


def test_prompt_has_selected_room_wifi_example():
    matches = [
        (state, decision)
        for state, message, decision in prompts.FEW_SHOT_EXAMPLES
        if message == "does it have wifi"
    ]
    assert len(matches) == 1
    state, decision = matches[0]
    assert state["selected_room_index"] == 1 and state["rooms_on_screen"]
    assert decision["intent"] == "ROOM_FOLLOWUP" and decision["operation"] == "AMENITY"
    assert decision["room_reference"] == "selected"
    assert '"does it have wifi"' not in prompts.system_prompt()  # rendered as <user_message>, not quoted
    assert "<user_message>does it have wifi</user_message>" in prompts.system_prompt()
    assert 15 <= len(prompts.FEW_SHOT_EXAMPLES) <= 25


def test_s8_through_the_real_llm_client_with_mocked_groq(monkeypatch, fake_groq):
    """Real llm_client + prompts + validators + actions; only Groq and guardrails are fake."""
    import uuid

    from llm_approach import reuse
    from llm_approach.state import SessionStore

    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    groq = fake_groq(
        [
            json.dumps(dec("NEW_SEARCH", city="Bangalore")),
            json.dumps(dec("ROOM_FOLLOWUP", op="LOWEST_PRICE")),
            json.dumps(dec("ROOM_FOLLOWUP", op="AMENITY", ref="selected", amenity="WiFi")),
        ]
    )
    store, sid = SessionStore(), str(uuid.uuid4())

    async def chat():
        out = []
        for text in ("rooms in bangalore", "which one is cheapest", "does it have wifi"):
            out.append(await pipeline.handle_turn(sid, text, store=store))
        return out

    first, cheapest, wifi = asyncio.run(chat())
    assert "Room 6" in cheapest.reply and store.peek(sid).selected_room_index == 6
    assert "Room 6" in wifi.reply and "does not list WiFi" in wifi.reply
    assert wifi.metrics["operation"] == "AMENITY" and wifi.metrics["tool_called"] is False
    # the third prompt told the model which room is selected
    last_user = groq.calls[2]["messages"][-1]["content"]
    assert '"selected_room_index": 6' in last_user and "does it have wifi" in last_user
    assert len(groq.calls) == 3  # one LLM call per turn


# ---- misc ------------------------------------------------------------------------------


def test_debug_info_is_returned_but_not_in_metrics(pipe):
    r = pipe.turn("rooms in Pune", dec("NEW_SEARCH", city="Pune", max_price="15k"))
    assert r.debug["intent"] == "NEW_SEARCH" and r.debug["validated"]["max_price"] == 15000
    assert "Pune" not in json.dumps(r.metrics) and "rooms in" not in json.dumps(r.metrics)


def test_platform_question_uses_knowledge_base_text(pipe):
    r = pipe.turn("how does homilivo make money", dec("PLATFORM_INFO", sq="How does Homilivo make money?"))
    assert r.reply == "Homilivo earns from owner subscriptions." and r.metrics["tool_called"] is False


def test_modules_import_without_torch():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import llm_approach.pipeline, llm_approach.fallback, llm_approach.metrics,"
        "llm_approach.extra_security, llm_approach.cli, llm_approach.eval.run_scenarios;"
        "bad = [m for m in ('torch', 'sentence_transformers', 'ollama', 'mcp', 'app.experiments.agent_modes', 'app.guardrails')"
        " if m in sys.modules]; print('BAD:' + ','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT)], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "BAD:", out.stdout
