import asyncio
import json
import logging

import pytest

from llm_approach import config, llm_client
from llm_approach.llm_client import LLMBudget, decide
from llm_approach.schema import Intent

GOOD = json.dumps({"intent": "NEW_SEARCH", "slots": {"city": "Bangalore"}, "confidence": 0.9})
SECRET_TEXT = "my-secret-user-text-42"
RAW_MARKER = "RAW-MODEL-OUTPUT-MARKER"


def run(coro):
    return asyncio.run(coro)


def test_valid_first_try(fake_groq):
    client = fake_groq([GOOD])
    budget = LLMBudget()
    d = run(decide("find me a room in bangalore", {}, budget))
    assert d.intent is Intent.NEW_SEARCH and d.slots.city == "Bangalore"
    assert budget.calls == 1 and budget.tokens_in == 100 and budget.tokens_out == 20
    kwargs = client.calls[0]
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == config.TEMPERATURE
    assert kwargs["max_tokens"] == config.LLM_MAX_TOKENS


def test_invalid_then_valid_retries(fake_groq):
    client = fake_groq(["not json", GOOD])
    budget = LLMBudget()
    d = run(decide("hi", {}, budget))
    assert d is not None and budget.calls == 2
    assert client.calls[1]["messages"][-1]["content"] == llm_client.RETRY_REMINDER
    assert len(client.calls[1]["messages"]) == len(client.calls[0]["messages"]) + 1


def test_two_invalid_returns_none(fake_groq):
    fake_groq(["nope", '{"intent": "DANCE"}'])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is None
    assert budget.calls == 2


def test_timeout_returns_none_without_retry(fake_groq, monkeypatch):
    monkeypatch.setattr(config, "LLM_TIMEOUT_S", 0.05)

    async def slow():
        await asyncio.sleep(1)

    client = fake_groq([slow, GOOD])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is None
    assert budget.calls == 1 and len(client.calls) == 1


def test_api_error_returns_none(fake_groq):
    fake_groq([RuntimeError("boom")])
    assert run(decide("hi", {}, LLMBudget())) is None


def test_budget_caps_calls(fake_groq):
    client = fake_groq(["bad", "bad", GOOD])
    budget = LLMBudget(max_calls=1)
    assert run(decide("hi", {}, budget)) is None
    assert budget.calls == 1 and len(client.calls) == 1


def test_exhausted_budget_makes_no_call(fake_groq):
    client = fake_groq([GOOD])
    budget = LLMBudget(max_calls=2, calls=2)
    assert run(decide("hi", {}, budget)) is None
    assert client.calls == []


def test_default_budget_is_config():
    assert LLMBudget().max_calls == config.MAX_LLM_CALLS_PER_TURN == 2


def test_logs_have_no_user_text_or_raw_output(fake_groq, caplog):
    fake_groq([f"{RAW_MARKER} not json", GOOD])
    with caplog.at_level(logging.DEBUG):
        run(decide(SECRET_TEXT, {}, LLMBudget()))
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "llm_decision" in logged and "NEW_SEARCH" in logged
    assert SECRET_TEXT not in logged and RAW_MARKER not in logged


def test_error_log_has_class_name_only(fake_groq, caplog):
    fake_groq([RuntimeError(f"token {SECRET_TEXT}")])
    with caplog.at_level(logging.DEBUG):
        run(decide("hi", {}, LLMBudget()))
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "RuntimeError" in logged and SECRET_TEXT not in logged


def test_client_is_lazy(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_client, "_client", None)
    with pytest.raises(KeyError):
        llm_client._get_client()  # needs the key only when a call is made
