"""Gemini thinking config per model family, the output cap, MAX_TOKENS handling, thinking-token accounting
and the Groq request parameters. Fake clients only: no SDK, no network, no key."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from llm_approach import config, providers, smoke_gemini
from llm_approach.llm_client import LLMBudget, decide

GOOD = json.dumps({"intent": "NEW_SEARCH", "slots": {"city": "Bangalore"}, "confidence": 0.9})
KEY = "AIza" + "FAKEKEY-not-a-real-key-1234567"
RAW_MARK = "RAWREPLYMARKzulu"


def run(coro):
    return asyncio.run(coro)


def reply(text, finish="STOP", usage=None):
    """A fake generate_content result with a candidate that carries a finish reason."""
    usage = usage or SimpleNamespace(prompt_token_count=100, candidates_token_count=20,
                                     cached_content_token_count=None, thoughts_token_count=None)

    async def make():
        return SimpleNamespace(text=text, usage_metadata=usage,
                               candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))])

    return make


# ---- thinking per model family -------------------------------------------------------------------


@pytest.mark.parametrize("model, expected", [
    ("gemini-3.5-flash-lite", {"thinking_level": "LOW"}),
    ("gemini-3-pro-preview", {"thinking_level": "LOW"}),
    ("gemini-2.5-flash-lite", {"thinking_budget": 0}),
    ("gemini-2.5-pro", {"thinking_budget": 0}),
    ("gemini-2.0-flash", None),
    ("some-other-model", None),
])
def test_thinking_by_model_family_in_the_request(fake_gemini, monkeypatch, model, expected):
    monkeypatch.setenv("LLM_APPROACH_GEMINI_MODEL", model)
    client = fake_gemini([GOOD])
    run(decide("hi", {}, LLMBudget()))
    cfg = client.calls[0]["config"]
    if expected is None:
        assert "thinking_config" not in cfg
    else:
        assert cfg["thinking_config"] == expected


@pytest.mark.parametrize("value, expected", [
    ("level:LOW", {"thinking_level": "LOW"}),
    ("level:minimal", {"thinking_level": "MINIMAL"}),
    ("level:MEDIUM", {"thinking_level": "MEDIUM"}),
    ("level:HIGH", {"thinking_level": "HIGH"}),
    ("budget:0", {"thinking_budget": 0}),
    ("budget:2048", {"thinking_budget": 2048}),
    ("none", None),
])
def test_override_wins_over_the_family(monkeypatch, value, expected):
    monkeypatch.setenv("LLM_APPROACH_GEMINI_THINKING", value)
    for model in ("gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "other"):
        assert config.gemini_thinking(model) == expected


@pytest.mark.parametrize("value", ["level:ULTRA", "level:", "budget:abc", "budget:", "high", "off", "level LOW"])
def test_invalid_override_stops_with_the_allowed_values(monkeypatch, value):
    monkeypatch.setenv("LLM_APPROACH_GEMINI_THINKING", value)
    with pytest.raises(ValueError) as err:
        config.gemini_thinking()
    msg = str(err.value)
    assert "LLM_APPROACH_GEMINI_THINKING" in msg and "level:LOW" in msg and "budget:<int>" in msg and "none" in msg
    assert len(value) < 8 or value not in msg  # the bad value is not echoed (short ones may sit inside the allowed list)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    assert config.setup_problem() == msg and KEY not in msg


def test_invalid_override_through_decide_never_reaches_the_client(fake_gemini, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_GEMINI_THINKING", "bogus")
    client = fake_gemini([GOOD])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is None
    assert budget.error_class == "ValueError" and client.calls == []


def test_thinking_label():
    assert config.gemini_thinking_label("gemini-3.5-flash-lite") == "level LOW"
    assert config.gemini_thinking_label("gemini-2.5-flash") == "budget 0"
    assert config.gemini_thinking_label("x").startswith("none")


# ---- output cap ------------------------------------------------------------------------------------


def test_gemini_cap_default_is_1024_and_groq_cap_is_unchanged():
    assert config.gemini_max_output_tokens() == 1024 and config.LLM_MAX_TOKENS == 500


def test_gemini_cap_env_override(fake_gemini, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_GEMINI_MAX_OUTPUT_TOKENS", "2048")
    client = fake_gemini([GOOD])
    run(decide("hi", {}, LLMBudget()))
    assert client.calls[0]["config"]["max_output_tokens"] == 2048


@pytest.mark.parametrize("value", ["abc", "0", "-5", " "])
def test_bad_cap_falls_back_to_the_default(monkeypatch, value):
    monkeypatch.setenv("LLM_APPROACH_GEMINI_MAX_OUTPUT_TOKENS", value)
    assert config.gemini_max_output_tokens() == 1024


# ---- MAX_TOKENS ------------------------------------------------------------------------------------


def test_max_tokens_then_retry_succeeds(fake_gemini):
    client = fake_gemini([reply('{"intent": "NEW_SE', "MAX_TOKENS"), reply(GOOD)])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is not None and budget.calls == 2 and len(client.calls) == 2


def test_max_tokens_twice_returns_none_for_the_fallback(fake_gemini):
    fake_gemini([reply(GOOD, "MAX_TOKENS"), reply(GOOD, "MAX_TOKENS")])  # even text that parses is not trusted
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is None and budget.calls == 2 and budget.finish_reason == "MAX_TOKENS"


def test_blocked_finish_reason_counts_as_empty(fake_gemini):
    client = fake_gemini([reply("x", "SAFETY"), reply(GOOD)])
    assert run(decide("hi", {}, LLMBudget())) is not None and len(client.calls) == 2


def test_stop_finish_reason_is_used_normally(fake_gemini):
    fake_gemini([reply(GOOD, "STOP")])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is not None and budget.calls == 1 and budget.finish_reason == "STOP"


def test_enum_style_finish_reason_string_is_understood():
    r = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="FinishReason.MAX_TOKENS")])
    assert providers._finish_reason(r) == "MAX_TOKENS"
    assert providers._finish_reason(SimpleNamespace(candidates=[])) is None


# ---- thinking tokens in the budget --------------------------------------------------------------------


def test_thinking_tokens_are_counted_as_output(fake_gemini):
    usage = SimpleNamespace(prompt_token_count=300, candidates_token_count=40, cached_content_token_count=None,
                            thoughts_token_count=60)
    fake_gemini([reply("nope", usage=usage), reply(GOOD, usage=usage)])
    budget = LLMBudget()
    run(decide("hi", {}, budget))
    assert (budget.tokens_in, budget.tokens_out, budget.thinking_tokens) == (600, 200, 120)


# ---- Groq unchanged ----------------------------------------------------------------------------------------


def test_groq_request_parameters_are_unchanged(fake_groq, monkeypatch):
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=GOOD))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    fake_groq([])
    monkeypatch.setattr(providers, "_get_groq_client",
                        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    run(decide("hi", {}, LLMBudget()))
    assert seen["max_tokens"] == 500 and seen["temperature"] == 0
    assert seen["response_format"] == {"type": "json_object"}
    assert set(seen) == {"model", "messages", "temperature", "max_tokens", "response_format"} | (
        {"reasoning_effort"} if config.REASONING_EFFORT else set())


# ---- log audit and smoke ---------------------------------------------------------------------------------------


def test_logs_hold_no_key_prompt_or_reply_text_with_thinking(fake_gemini, monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    fake_gemini([reply(f"{RAW_MARK} cut", "MAX_TOKENS"), reply(f"{RAW_MARK} again", "MAX_TOKENS")])
    with caplog.at_level(logging.DEBUG):
        run(decide("UMARKquartz find a room", {}, LLMBudget()))
    assert RAW_MARK not in caplog.text and "UMARKquartz" not in caplog.text and KEY not in caplog.text


def test_smoke_prints_thinking_finish_and_thinking_tokens(fake_gemini, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=20, cached_content_token_count=None,
                            thoughts_token_count=7)
    fake_gemini([reply(GOOD, usage=usage)])
    lines: list[str] = []
    assert run(smoke_gemini.run(out=lines.append)) == 0
    out = "\n".join(lines)
    assert "thinking:  level LOW" in out and "finish:    STOP" in out and "in 100, out 27, thinking 7" in out
    assert KEY not in out and GOOD not in out and smoke_gemini.MESSAGE not in out


def test_smoke_with_invalid_thinking_override_calls_nothing(fake_gemini, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("LLM_APPROACH_GEMINI_THINKING", "bogus")
    client = fake_gemini([GOOD])
    lines: list[str] = []
    assert run(smoke_gemini.run(out=lines.append)) == 2 and client.calls == []
    assert "level:LOW" in "\n".join(lines) and KEY not in "\n".join(lines)
