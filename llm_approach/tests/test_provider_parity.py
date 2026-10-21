"""Provider parity: every stub scenario through the real client with a fake Groq and with a fake Gemini
returning the same JSON. Replies, rooms, state and metrics must be identical; only timings and the
provider label may differ. Also pins the Groq request parameters.

No network, no SDK, no key, no torch (fake guardrails, offline rooms).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import types
import uuid
from types import SimpleNamespace
from unittest import mock

import pytest

from llm_approach import config, conversation_answerer, llm_client, pipeline, prompts, providers
from llm_approach.eval import run_scenarios
from llm_approach.state import SessionStore

REAL_DECIDE = llm_client.decide
REAL_GENERATE = llm_client.generate_json
TIMING = ("llm_ms", "llm_call_ms", "api_ms", "total_ms")  # wall-clock, differs from run to run
SCENARIOS = run_scenarios.load_scenarios()


class RateLimitError(Exception):  # Groq's error: recognised by its class name
    status_code = 429


class ClientError(Exception):  # Gemini's error: recognised by code / status
    code, status = 429, "RESOURCE_EXHAUSTED"


class Served:
    """Serves the scenario's scripted replies to whichever fake client asks."""

    def __init__(self, script, answers, spy, kind):
        self.script, self.answers, self.spy, self.kind = script, answers, spy, kind
        self.exhausted = 0
        self.retry_429 = False

    def reply(self, messages) -> str:
        if self.retry_429:  # Gemini only: the provider retries a 429 once, and the retry is throttled too
            self.retry_429 = False
            raise ClientError("429")
        is_answer = messages[0]["content"] == conversation_answerer.SYSTEM_PROMPT
        queue = self.answers if is_answer else self.script
        if is_answer:
            self.spy.answer_calls += 1
        else:
            self.spy.llm_calls += 1
        if not queue:
            self.exhausted += 1
            raise RuntimeError("script exhausted")
        item = queue.pop(0)
        if item == "rate_limit":
            self.retry_429 = self.kind == "gemini"
            raise (RateLimitError if self.kind == "groq" else ClientError)("429")
        if item in (None, "timeout"):
            raise asyncio.TimeoutError()
        return json.dumps(item) if isinstance(item, dict) else item


def fake_client(kind, served):
    usage_groq = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
    usage_gemini = SimpleNamespace(prompt_token_count=100, candidates_token_count=20,
                                   cached_content_token_count=None, thoughts_token_count=None)

    async def create(**kwargs):  # Groq
        text = served.reply(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=usage_groq)

    async def generate_content(**kwargs):  # Gemini
        messages = [{"role": "system", "content": kwargs["config"]["system_instruction"]}] + [
            {"role": c["role"], "content": c["parts"][0]["text"]} for c in kwargs["contents"]]
        return SimpleNamespace(text=served.reply(messages), usage_metadata=usage_gemini)

    if kind == "groq":
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))


@contextlib.contextmanager
def provider_runtime(scenario, kind, script, spy, answers):
    """The stub scenario environment (guardrails, rooms, KB, spies), but with the REAL ``decide`` and
    ``generate_json`` calling a fake provider client."""
    served = Served(script, answers, spy, kind)
    client = fake_client(kind, served)
    with run_scenarios.runtime(scenario, "stub", script, spy, answers), contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(llm_client, "decide", REAL_DECIDE))
        stack.enter_context(mock.patch.object(llm_client, "generate_json", REAL_GENERATE))
        stack.enter_context(mock.patch.dict("os.environ", {config.PROVIDER_ENV: kind}))
        stack.enter_context(mock.patch.object(providers, "_get_groq_client", lambda: client))
        stack.enter_context(mock.patch.object(providers, "_get_gemini_client", lambda: client))
        yield served


def snapshot(result, state):
    metrics = {k: v for k, v in result.metrics.items() if k not in TIMING}
    return {
        "reply": result.reply, "rooms": result.rooms, "success": result.success, "metrics": metrics,
        "state": state.to_prompt_state() if state else None,
        "queries": state.query_count if state else None,
        "user_queries": state.user_queries if state else None,
        "transcript": state.transcript if state else None,
    }


async def play(scenario, kind):
    script = [t["llm"] for t in scenario["turns"] if t["llm"] != "never"]
    answers = []
    for t in scenario["turns"]:
        given = t.get("answer")
        answers += given if isinstance(given, list) else ([] if given is None else [given])
    spy, store, sid = run_scenarios._Spy(), SessionStore(), str(uuid.uuid4())
    snaps, failed = [], []
    with provider_runtime(scenario, kind, script, spy, answers) as served:
        for i, turn in enumerate(scenario["turns"], 1):
            if turn.get("new_session"):
                sid = str(uuid.uuid4())
            calls, searches, answered = spy.llm_calls, len(spy.searches), spy.answer_calls
            result = await pipeline.handle_turn(sid, turn["user"], store=store)
            spy.llm_calls_this_turn, spy.answer_calls_this_turn = spy.llm_calls - calls, spy.answer_calls - answered
            snaps.append(snapshot(result, store.peek(sid)))
            checks = run_scenarios._check_turn(turn.get("expect", {}), result, store.peek(sid), spy, searches, "stub")
            if turn["llm"] == "never" and spy.llm_calls_this_turn:
                failed.append((scenario["id"], i, "the LLM was called for a blocked turn"))
            failed += [(scenario["id"], i, c.name) for c in checks if not c.ok]
    return snaps, failed, served.exhausted


def test_parity_covers_every_scenario_file():
    ids = {p.stem for p in run_scenarios.SCENARIO_DIR.glob("S*.json")}
    assert {s["id"] for s in SCENARIOS} == ids and len(ids) >= 23
    assert sum(len(s["turns"]) for s in SCENARIOS) >= 100


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_every_scenario_is_identical_through_fake_groq_and_fake_gemini(scenario, monkeypatch):
    monkeypatch.delenv("LLM_APPROACH_METRICS", raising=False)
    pipeline.configure(full_guardrails=True)
    groq_snaps, groq_failed, groq_exhausted = asyncio.run(play(scenario, "groq"))
    gem_snaps, gem_failed, gem_exhausted = asyncio.run(play(scenario, "gemini"))
    assert len(groq_snaps) == len(gem_snaps) == len(scenario["turns"])
    for i, (a, b) in enumerate(zip(groq_snaps, gem_snaps), 1):
        assert a == b, f"{scenario['id']} turn {i} differs between Groq and Gemini"
    assert groq_failed == gem_failed == []  # the scenario's own expectations hold through the real client
    assert groq_exhausted == gem_exhausted == 0


# ---- the Groq request is unchanged ---------------------------------------------------------------------


def test_groq_request_parameters_are_unchanged(fake_groq, monkeypatch):
    client = fake_groq(['{"intent": "NEW_SEARCH", "confidence": 0.9, "slots": {"city": "Pune"}}'])
    asyncio.run(llm_client.decide("rooms in pune", {}, llm_client.LLMBudget()))
    (kwargs,) = client.calls
    assert kwargs == {
        "model": "openai/gpt-oss-20b",
        "messages": prompts.build_messages("rooms in pune", {}),
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "reasoning_effort": "low",
    }
    assert config.LLM_TIMEOUT_S == 8 and config.MAX_LLM_CALLS_PER_TURN == 2


def test_groq_answerer_request_uses_the_answerer_token_limit(fake_groq):
    client = fake_groq(["{}"])
    asyncio.run(llm_client.generate_json([{"role": "user", "content": "x"}], llm_client.LLMBudget()))
    assert client.calls[0]["max_tokens"] == config.ANSWERER_MAX_TOKENS == 500


def test_groq_client_is_built_with_no_sdk_retries(monkeypatch):
    seen = {}

    class AsyncGroq:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    fake = types.ModuleType("groq")
    fake.AsyncGroq = AsyncGroq
    monkeypatch.setitem(sys.modules, "groq", fake)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-not-real")
    monkeypatch.setattr(providers, "_groq_client", None)
    providers._get_groq_client()
    assert seen == {"api_key": "gsk-test-not-real", "max_retries": 0}


def test_groq_is_still_selectable_and_gemini_is_the_default(monkeypatch):
    assert config.provider() == "gemini" and config.model() == "gemini-3.5-flash-lite"
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    assert config.provider() == "groq" and config.model() == "openai/gpt-oss-20b"
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "nonsense")
    with pytest.raises(ValueError) as err:
        config.provider()
    assert "nonsense" not in str(err.value)
    assert config.label() == ("invalid", "") and config.setup_problem()
