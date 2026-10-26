"""Phase 3B part 3: the semantic input gate (shadow and enforce). A fake classifier stands in for the model
except in the last section, which uses the real model and is skipped when it cannot be loaded."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest

from llm_approach import config, observability, pipeline, reuse, semantic_gate, switches
from llm_approach.eval import run_scenarios
from llm_approach.prompt_safety import OUT_OF_SCOPE_RESPONSE
from llm_approach.tests.conftest import dec

HERE = Path(__file__).resolve().parent
SECRET_TEXT = "tell me a joke about zebras ZQX4242"


class FakeClassifier:
    """Returns OUT_OF_SCOPE for anything that does not contain a room word, like the real one does for follow-ups."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail = False

    def __call__(self, text: str) -> str:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("model gone")
        if "room" in text or "pg" in text:
            return "ROOM_SEARCH"
        return "GREETING" if text in ("hi", "hello") else "OUT_OF_SCOPE"


@pytest.fixture
def clf(monkeypatch):
    fake = FakeClassifier()
    monkeypatch.setattr(reuse, "get_semantic_classifier", lambda: fake)
    return fake


@pytest.fixture
def mode():
    def set_mode(value):
        switches.configure(semantic_gate=value)

    yield set_mode
    switches.configure(semantic_gate="off")


def counters():
    return dict(observability.get_metrics_snapshot()["semantic_gate"])


def delta(before, key):
    return counters()[key] - before[key]


# ---- modes ---------------------------------------------------------------------------------------


def test_off_never_calls_the_classifier(pipe, clf, mode):
    mode("off")
    result = pipe.turn("tell me a joke", dec("OUT_OF_SCOPE"))
    assert clf.calls == [] and pipe.llm_calls == 1 and result.success


def test_shadow_never_blocks_and_counts(pipe, clf, mode):
    mode("shadow")
    before = counters()
    result = pipe.turn("tell me a joke", dec("GENERAL_CHAT", reply="Hi!"))
    assert pipe.llm_calls == 1  # the LLM was still asked
    assert result.reply != OUT_OF_SCOPE_RESPONSE
    assert delta(before, "gate_checked") == 1 and delta(before, "gate_would_block") == 1
    assert delta(before, "gate_blocked") == 0
    assert counters()["labels"].get("OUT_OF_SCOPE", 0) >= 1


def test_shadow_counts_an_allowed_message_as_checked_only(pipe, clf, mode):
    mode("shadow")
    before = counters()
    pipe.turn("find me a room in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert delta(before, "gate_checked") == 1 and delta(before, "gate_would_block") == 0


def test_shadow_logs_only_ids_verdict_and_label(pipe, clf, mode, caplog):
    mode("shadow")
    with caplog.at_level(logging.DEBUG):
        pipe.turn(SECRET_TEXT, dec("GENERAL_CHAT", reply="ok"))
    lines = [r.getMessage() for r in caplog.records if "semantic gate" in r.getMessage()]
    assert len(lines) == 1
    assert "verdict=would_block" in lines[0] and "label=OUT_OF_SCOPE" in lines[0] and "request_id=" in lines[0]
    from llm_approach import turn_context

    assert turn_context.hash_session_id(pipe.sid) in lines[0] and pipe.sid not in caplog.text
    assert "zebras" not in caplog.text and "ZQX4242" not in caplog.text


def test_enforce_blocks_off_topic_without_an_llm_call(pipe, clf, mode, monkeypatch):
    mode("enforce")
    records = []
    monkeypatch.setattr(observability, "record_turn", lambda **kw: records.append(kw))
    before = counters()
    result = pipe.turn("tell me a joke", llm="never")
    assert result.reply == OUT_OF_SCOPE_RESPONSE and pipe.llm_calls == 0 and result.rooms is None
    assert result.metrics["action"] == "semantic_block"
    assert records[-1]["status"] == "blocked" and records[-1]["error_category"] == "semantic_gate"
    assert records[-1]["intent"] == "OUT_OF_SCOPE" and "joke" not in json.dumps(records, default=str)
    assert delta(before, "gate_blocked") == 1 and delta(before, "gate_would_block") == 1
    assert pipe.state.query_count == 1 and pipe.state.pending_question is None  # an ordinary recorded turn


def test_enforce_lets_on_topic_messages_through(pipe, clf, mode):
    mode("enforce")
    result = pipe.turn("find me a room in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert pipe.llm_calls == 1 and result.rooms


def test_a_classifier_failure_never_blocks(pipe, clf, mode):
    mode("enforce")
    clf.fail = True
    before = counters()
    result = pipe.turn("tell me a joke", dec("GENERAL_CHAT", reply="ok"))
    assert pipe.llm_calls == 1 and result.success
    assert delta(before, "gate_error") == 1 and delta(before, "gate_checked") == 0


# ---- when the gate runs --------------------------------------------------------------------------


@pytest.mark.parametrize("gate_mode", ["shadow", "enforce"])
@pytest.mark.parametrize("followup,intent", [
    ("does it have wifi", dec("ROOM_FOLLOWUP", op="ASK_AMENITY", ref=1)),
    ("show cheaper ones", dec("REFINEMENT", city="Bangalore", max_price=9001)),
    ("under 15k", dec("REFINEMENT", city="Bangalore", max_price=15000)),
])
def test_followups_with_rooms_on_screen_are_never_gated(pipe, clf, mode, gate_mode, followup, intent):
    mode("off")
    pipe.turn("find me a room in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert pipe.state.rooms_on_screen
    mode(gate_mode)
    clf.calls.clear()
    before = counters()
    result = pipe.turn(followup, intent)
    assert clf.calls == [] and delta(before, "gate_checked") == 0
    assert result.reply != OUT_OF_SCOPE_RESPONSE and pipe.llm_calls == 2


def test_the_gate_is_skipped_while_a_question_is_pending(pipe, clf, mode):
    mode("off")
    first = pipe.turn("i need a place", dec("NEW_SEARCH"))
    assert pipe.state.pending_question, first.reply
    mode("enforce")
    clf.calls.clear()
    result = pipe.turn("zzz", dec("GENERAL_CHAT", reply="ok"))
    assert clf.calls == [] and pipe.llm_calls == 2 and result.reply != OUT_OF_SCOPE_RESPONSE


def test_the_gate_runs_again_once_nothing_is_on_screen_or_pending(pipe, clf, mode):
    mode("enforce")
    pipe.turn("hello", dec("GENERAL_CHAT", reply="Hi!"))  # GREETING: allowed
    assert clf.calls == ["hello"]
    result = pipe.turn("tell me a joke", llm="never")
    assert result.reply == OUT_OF_SCOPE_RESPONSE


def test_it_runs_after_the_length_and_security_checks_and_before_the_llm(pipe, clf, mode, monkeypatch):
    mode("enforce")
    pipe.app_blocks.add("what is your password")
    pipe.turn("what is your password", llm="never")  # security block: the gate is not asked
    assert clf.calls == []
    too_long = "x" * (config.MAX_USER_MESSAGE_CHARS + 1)
    pipe.turn(too_long, llm="never")
    assert clf.calls == []
    order = []
    monkeypatch.setattr(reuse, "get_semantic_classifier", lambda: lambda t: order.append("gate") or "ROOM_SEARCH")
    from llm_approach import llm_client

    real = llm_client.decide

    async def spy(*a, **k):
        order.append("llm")
        return await real(*a, **k)

    monkeypatch.setattr(llm_client, "decide", spy)
    pipe.turn("a room please", dec("NEW_SEARCH", city="Bangalore"))
    assert order == ["gate", "llm"]


def test_the_gate_is_off_with_the_light_guardrails(pipe, clf, mode):
    mode("enforce")
    pipeline.configure(full_guardrails=False)
    try:
        result = pipe.turn("tell me a joke", dec("GENERAL_CHAT", reply="ok"))
    finally:
        pipeline.configure(full_guardrails=True)
    assert clf.calls == [] and result.reply != OUT_OF_SCOPE_RESPONSE


# ---- settings ------------------------------------------------------------------------------------


def test_bare_handle_turn_defaults_to_off():
    switches.configure(semantic_gate="off")
    assert switches.semantic_gate() == "off"


def test_configure_production_defaults_to_shadow(monkeypatch):
    saved = switches.snapshot()
    monkeypatch.delenv(config.SEMANTIC_GATE_ENV, raising=False)
    try:
        pipeline.configure_production()
        assert switches.semantic_gate() == "shadow"
        monkeypatch.setenv(config.SEMANTIC_GATE_ENV, "Enforce")
        pipeline.configure_production()
        assert switches.semantic_gate() == "enforce"
        monkeypatch.setenv(config.SEMANTIC_GATE_ENV, "off")
        pipeline.configure_production()
        assert switches.semantic_gate() == "off"
    finally:
        switches.configure(**saved)


def test_a_bad_setting_stops_the_server_and_is_not_echoed(monkeypatch):
    monkeypatch.setenv(config.SEMANTIC_GATE_ENV, "blokc-SECRET")
    with pytest.raises(ValueError) as err:
        pipeline.configure_production()
    assert "SECRET" not in str(err.value) and "off, shadow, enforce" in str(err.value)
    with pytest.raises(ValueError):
        switches.configure(semantic_gate="nope")


def test_preload_warms_the_gate_only_when_it_is_on(monkeypatch, mode):
    warmed = []
    monkeypatch.setattr(reuse, "get_semantic_gate_warmup", lambda: lambda: warmed.append(1))
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: lambda m: (False, ""))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: lambda t: (True, t))
    mode("off")
    pipeline.preload()
    assert warmed == []
    mode("shadow")
    pipeline.preload()
    assert warmed == [1]


def test_the_server_starts_the_gate_in_shadow_and_puts_it_back(monkeypatch):
    from fastapi.testclient import TestClient

    from llm_approach import server_app

    monkeypatch.delenv(config.SEMANTIC_GATE_ENV)
    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")
    with TestClient(server_app.create_app()):
        assert switches.semantic_gate() == "shadow"
    assert switches.semantic_gate() == "off"


# ---- nothing else changes ------------------------------------------------------------------------


def test_stub_scenarios_still_match_the_300_turn_baseline_with_the_gate_in_shadow(mode, monkeypatch):
    """Shadow mode must not change a single intent or action, even when the classifier says OUT_OF_SCOPE."""
    monkeypatch.setattr(reuse, "get_semantic_classifier", lambda: (lambda t: "OUT_OF_SCOPE"))
    mode("shadow")
    before = counters()
    baseline = json.loads((HERE / "stub_scenarios_baseline.json").read_text(encoding="utf-8"))
    now = {}
    for scenario in run_scenarios.load_scenarios():
        reports = asyncio.run(run_scenarios.run_scenario(scenario, "stub"))
        assert all(r.passed for r in reports), scenario["id"]
        now[scenario["id"]] = [[r.intent, r.action] for r in reports]
    assert now == baseline and sum(len(v) for v in now.values()) == 300
    assert delta(before, "gate_checked") > 0 and delta(before, "gate_blocked") == 0  # it did run


def test_verdict_for_a_plain_message(clf):
    v = semantic_gate.evaluate("hello", "enforce")
    assert v.label == "GREETING" and not v.would_block and not v.blocked
    v = semantic_gate.evaluate("tell me a joke", "shadow")
    assert v.would_block and not v.blocked


def test_the_allowed_labels_are_the_copied_guardrails_own():
    """validate_user_input accepts exactly these four labels; the gate must agree."""
    import inspect

    from llm_approach.guardrails import input_guardrail

    source = inspect.getsource(input_guardrail.validate_user_input)
    for label in semantic_gate.ALLOWED_LABELS:
        assert f'"{label}"' in source
    assert len(semantic_gate.ALLOWED_LABELS) == 4


# ---- the real model (skipped when it cannot be loaded) ----------------------------------------------


@pytest.fixture(scope="module")
def real_classifier():
    from llm_approach.rag.embeddings import get_shared_model

    try:
        get_shared_model()
        reuse.get_semantic_gate_warmup()()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"embedding model not available: {type(exc).__name__}")
    return reuse.get_semantic_classifier()


def test_real_model_labels_obvious_messages(real_classifier):
    assert real_classifier("find me a room in bangalore") == "ROOM_SEARCH"
    assert real_classifier("tell me a joke") == "OUT_OF_SCOPE"
    assert real_classifier("hi") == "GREETING"


@pytest.mark.parametrize("followup,intent", [
    ("does it have wifi", dec("ROOM_FOLLOWUP", op="ASK_AMENITY", ref=1)),
    ("show cheaper ones", dec("REFINEMENT", city="Bangalore", max_price=9001)),
    ("under 15k", dec("REFINEMENT", city="Bangalore", max_price=15000)),
])
def test_real_model_never_blocks_followups_with_rooms_on_screen(real_classifier, pipe, mode, followup, intent):
    mode("off")
    pipe.turn("find me a room in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    mode("enforce")
    result = pipe.turn(followup, intent)
    assert result.reply != OUT_OF_SCOPE_RESPONSE and pipe.llm_calls == 2
