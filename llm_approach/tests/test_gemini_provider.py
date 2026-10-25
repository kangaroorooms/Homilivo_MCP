"""The Gemini provider, offline: a fake google-genai client behind the real ``providers`` and
``llm_client``. No SDK, no network, no key."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import types
import uuid
from types import SimpleNamespace

import pytest

from llm_approach import config, llm_client, pipeline, prompts, providers, reuse, smoke_gemini
from llm_approach.llm_client import LLMBudget, decide
from llm_approach.schema import Intent
from llm_approach.state import SessionStore

from .conftest import ROOT

GOOD = json.dumps({"intent": "NEW_SEARCH", "slots": {"city": "Bangalore"}, "confidence": 0.9})
KEY = "AIza" + "FAKEKEY-not-a-real-key-1234567"  # fake; split so secret scanners stay quiet
USER_TEXT = "UMARKquartz find me a room"
RAW_MARK = "RAWREPLYMARKzulu"


def run(coro):
    return asyncio.run(coro)


class ClientError(Exception):
    """Same shape as google.genai.errors.ClientError: ``code`` and ``status``, no ``status_code``."""

    def __init__(self, code: int, status: str, message: str = "") -> None:
        super().__init__(message)
        self.code, self.status = code, status


class Blocked:
    """A blocked reply: reading ``.text`` raises (the SDK does this for some blocked replies)."""

    usage_metadata = None

    @property
    def text(self):
        raise ValueError("blocked")


# ---- success and request shape -------------------------------------------------------


def test_success_parses_and_counts_tokens(fake_gemini):
    client = fake_gemini([GOOD])
    budget = LLMBudget()
    d = run(decide("find me a room in bangalore", {}, budget))
    assert d.intent is Intent.NEW_SEARCH and d.slots.city == "Bangalore"
    assert budget.calls == 1 and (budget.tokens_in, budget.tokens_out) == (100, 20)
    assert budget.cached_tokens is None and len(budget.call_ms) == 1 and budget.error_class is None
    assert len(client.calls) == 1


def test_request_parameters(fake_gemini):
    client = fake_gemini([GOOD])
    run(decide("hi", {}, LLMBudget()))
    call = client.calls[0]
    cfg = call["config"]
    assert call["model"] == "gemini-3.5-flash-lite"
    assert cfg["temperature"] == config.TEMPERATURE == 0
    assert cfg["max_output_tokens"] == 1024 != config.LLM_MAX_TOKENS  # Gemini has its own cap; Groq stays 500
    assert cfg["response_mime_type"] == "application/json"
    assert cfg["thinking_config"] == {"thinking_level": "LOW"}  # gemini-3.x default
    assert cfg["system_instruction"] == prompts.build_messages("hi", {})[0]["content"]
    assert [c["role"] for c in call["contents"]] == ["user"]


def test_model_comes_from_the_environment(fake_gemini, monkeypatch):
    client = fake_gemini([GOOD])
    monkeypatch.setenv("LLM_APPROACH_GEMINI_MODEL", "gemini-test-model")
    run(decide("hi", {}, LLMBudget()))
    assert client.calls[0]["model"] == "gemini-test-model"


def test_thinking_setting_default_is_level_low_for_the_default_model():
    assert providers.gemini_config("s", 10)["thinking_config"] == {"thinking_level": "LOW"}


def test_message_mapping_keeps_order_and_maps_roles():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "now"},
    ]
    system, contents = providers.to_gemini_request(messages)
    assert system == "SYS"
    assert [(c["role"], c["parts"][0]["text"]) for c in contents] == [
        ("user", "q1"), ("model", "a1"), ("user", "q2"), ("model", "a2"), ("user", "now")]


def test_real_prompt_maps_system_to_instruction_and_user_to_user():
    messages = prompts.build_messages("hello", {})
    system, contents = providers.to_gemini_request(messages)
    assert system == messages[0]["content"]
    assert len(contents) == len(messages) - 1 and contents[-1]["parts"][0]["text"] == messages[-1]["content"]
    assert all(c["role"] in ("user", "model") for c in contents)


def test_no_system_message_means_no_system_instruction():
    system, _ = providers.to_gemini_request([{"role": "user", "content": "x"}])
    assert system is None and "system_instruction" not in providers.gemini_config(system, 5)


def test_answerer_call_uses_the_gemini_output_cap(fake_gemini):
    client = fake_gemini(['{"x": 1}'])
    raw = run(llm_client.generate_json([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], LLMBudget()))
    assert raw == '{"x": 1}'
    assert client.calls[0]["config"]["max_output_tokens"] == config.DEFAULT_GEMINI_MAX_OUTPUT_TOKENS == 1024


# ---- invalid, empty, blocked ------------------------------------------------------------


def test_invalid_json_then_one_retry(fake_gemini):
    client = fake_gemini(["not json", GOOD])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is not None and budget.calls == 2
    first, second = client.calls
    assert len(second["contents"]) == len(first["contents"]) + 1
    assert second["contents"][-1]["parts"][0]["text"] == llm_client.RETRY_REMINDER


def test_two_invalid_replies_return_none(fake_gemini):
    client = fake_gemini(["nope", "{still not"])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is None and budget.calls == 2 and len(client.calls) == 2


def test_empty_reply_is_retried_like_invalid_json(fake_gemini):
    client = fake_gemini([None, GOOD])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is not None and budget.calls == 2 and len(client.calls) == 2


def test_empty_twice_returns_none_for_the_fallback(fake_gemini):
    fake_gemini([None, None])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is None and budget.calls == 2 and budget.error_class is None


def test_blocked_reply_counts_as_empty(fake_gemini):
    async def blocked():
        return Blocked()

    client = fake_gemini([blocked, GOOD])
    assert run(decide("hi", {}, LLMBudget())) is not None and len(client.calls) == 2


# ---- errors ---------------------------------------------------------------------------------


def test_quota_error_maps_to_rate_limit_without_retry(fake_gemini):
    # the provider layer retries a 429 once after the wait; the decision loop itself never retries it
    err = ClientError(429, "RESOURCE_EXHAUSTED", "quota exceeded for key")
    client = fake_gemini([err, err])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is None
    assert budget.error_class == "RateLimitError" and budget.calls == 1 and len(client.calls) == 2


def test_resource_exhausted_status_alone_counts_as_rate_limit():
    assert llm_client.is_rate_limit_error(ClientError(0, "RESOURCE_EXHAUSTED"))
    assert llm_client.is_rate_limit_error(ClientError(429, ""))
    assert not llm_client.is_rate_limit_error(ClientError(500, "INTERNAL"))
    assert not llm_client.is_rate_limit_error(RuntimeError("boom"))


def test_timeout_returns_none_without_retry(fake_gemini, monkeypatch):
    monkeypatch.setattr(config, "gemini_timeout_s", lambda: 0.05)

    async def slow():
        await asyncio.sleep(1)

    client = fake_gemini([slow, GOOD])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is None
    assert budget.error_class == "TimeoutError" and budget.calls == 1 and len(client.calls) == 1


def test_other_api_error_returns_none_with_its_class_name(fake_gemini):
    client = fake_gemini([ClientError(500, "INTERNAL", "server blew up"), GOOD])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is None
    assert budget.error_class == "ClientError" and len(client.calls) == 1


def test_exhausted_budget_makes_no_call(fake_gemini):
    client = fake_gemini([GOOD])
    assert run(decide("hi", {}, LLMBudget(max_calls=2, calls=2))) is None and client.calls == []


# ---- keys ------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_or_blank_key_raises_before_any_request(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GEMINI_API_KEY", value)
    monkeypatch.setattr(providers, "_gemini_client", None)
    with pytest.raises(KeyError) as err:
        providers._get_gemini_client()  # raised before the SDK is imported
    assert err.value.args == ("GEMINI_API_KEY",)  # the name, never a value
    assert providers._gemini_client is None


def test_missing_key_through_decide_falls_back_without_secrets(monkeypatch, caplog):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(providers, "_gemini_client", None)
    budget = LLMBudget()
    with caplog.at_level(logging.DEBUG):
        assert run(decide("hi", {}, budget)) is None
    assert budget.error_class == "KeyError" and budget.calls == 1


def test_client_is_built_lazily_with_no_sdk_retries(monkeypatch):
    seen = {}

    class Client:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = Client
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(providers, "_gemini_client", None)
    client = providers._get_gemini_client()
    assert isinstance(client, Client) and providers._get_gemini_client() is client  # cached
    assert seen["api_key"] == KEY
    assert seen["http_options"] == {"timeout": 20000, "retry_options": {"attempts": 1}}  # 20 s, one attempt


# ---- token counting ----------------------------------------------------------------------------


def test_usage_prompt_output_cached_and_thinking_tokens(fake_gemini):
    usage = SimpleNamespace(prompt_token_count=4000, candidates_token_count=80, cached_content_token_count=1500,
                            thoughts_token_count=12)
    fake_gemini([GOOD, GOOD], usage)
    budget = LLMBudget()
    run(decide("hi", {}, budget))
    assert (budget.tokens_in, budget.tokens_out, budget.cached_tokens) == (4000, 92, 1500)


@pytest.mark.parametrize("usage", [None, SimpleNamespace(), SimpleNamespace(prompt_token_count="many",
                                                                          candidates_token_count=True)])
def test_missing_or_odd_usage_counts_zero(fake_gemini, usage):
    client = fake_gemini([GOOD])
    client.usage = usage
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is not None
    assert (budget.tokens_in, budget.tokens_out, budget.cached_tokens) == (0, 0, None)


# ---- lazy import --------------------------------------------------------------------------------


def test_sdks_are_not_imported_by_the_package_or_the_groq_path():
    code = (
        "import sys, asyncio; sys.path.insert(0, %r)\n"
        "from types import SimpleNamespace\n"
        "from llm_approach import config, llm_client, pipeline, providers, smoke_gemini\n"
        "assert 'google.genai' not in sys.modules and 'groq' not in sys.modules\n"
        "import os; os.environ['LLM_APPROACH_PROVIDER'] = 'groq'\n"
        "async def create(**kw):\n"
        "    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))], usage=None)\n"
        "providers._groq_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))\n"
        "asyncio.run(llm_client.decide('hi', {}, llm_client.LLMBudget()))\n"
        "assert 'google.genai' not in sys.modules, 'the Groq path imported the Gemini SDK'\n"
        "print('ok')\n"
    ) % str(ROOT)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "ok"


# ---- through the pipeline ---------------------------------------------------------------------------


@pytest.fixture
def real_pipeline(monkeypatch):
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    monkeypatch.setenv(config.OFFLINE_ROOMS_ENV, "1")
    monkeypatch.delenv("LLM_APPROACH_METRICS", raising=False)
    pipeline.configure(full_guardrails=True)

    def turn(message, store=None, sid=None):
        return run(pipeline.handle_turn(sid or str(uuid.uuid4()), message, store=store or SessionStore()))

    return turn


def test_quota_error_runs_the_rule_based_fallback(fake_gemini, real_pipeline):
    fake_gemini([ClientError(429, "RESOURCE_EXHAUSTED", "Quota exceeded")] * 2)  # the call and its one retry
    result = real_pipeline("rooms in pune")
    assert result.reply != pipeline.RATE_LIMIT_REPLY
    assert result.metrics["action"].startswith("fallback_") and result.metrics["error_class"] == "RateLimitError"
    assert result.metrics["fallback_used"] is True


def test_invalid_twice_uses_the_existing_fallback(fake_gemini, real_pipeline):
    fake_gemini([None, "{nope"])
    result = real_pipeline("rooms in bangalore")
    assert result.metrics["fallback_used"] is True and result.metrics["llm_calls"] == 2


def test_a_search_turn_end_to_end(fake_gemini, real_pipeline):
    fake_gemini([GOOD])
    result = real_pipeline("rooms in bangalore")
    assert result.metrics["intent"] == "NEW_SEARCH" and result.rooms
    assert (result.metrics["tokens_in"], result.metrics["tokens_out"]) == (100, 20)


# ---- log audit: no key, prompt text or reply text ------------------------------------------------------


def test_gemini_path_logs_class_names_only(fake_gemini, monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    fake_gemini([f"{RAW_MARK} not json", ClientError(500, "INTERNAL", f"echo {KEY} {USER_TEXT} {RAW_MARK}")])
    with caplog.at_level(logging.DEBUG):
        run(decide(USER_TEXT, {}, LLMBudget()))
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "llm_decision" in logged and "ClientError" in logged
    system = prompts.build_messages("x", {})[0]["content"]
    for secret in (KEY, USER_TEXT, RAW_MARK, "UMARKquartz", system[:60]):
        assert secret not in logged


def test_gemini_answerer_logs_no_text(fake_gemini, monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    fake_gemini([f"{RAW_MARK} answer"])
    with caplog.at_level(logging.DEBUG):
        run(llm_client.generate_json([{"role": "user", "content": USER_TEXT}], LLMBudget()))
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "llm_answer" in logged and RAW_MARK not in logged and USER_TEXT not in logged and KEY not in logged


# ---- smoke_gemini ----------------------------------------------------------------------------------------


def _smoke():
    lines: list[str] = []
    code = run(smoke_gemini.run(out=lines.append))
    return code, "\n".join(lines)


def test_smoke_refuses_without_a_key_and_calls_nothing(fake_gemini, monkeypatch):
    client = fake_gemini([GOOD])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    code, out = _smoke()
    assert code == 2 and "GEMINI_API_KEY is not set" in out and client.calls == []


def test_smoke_prints_only_the_allowed_fields(fake_gemini, monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    client = fake_gemini([GOOD])
    with caplog.at_level(logging.DEBUG):
        code, out = _smoke()
    assert code == 0 and len(client.calls) == 1
    assert "gemini" in out and "gemini-3.5-flash-lite" in out and "OK" in out
    assert "valid decision: yes" in out and "in 100, out 20, thinking 0" in out and "ms" in out
    assert "thinking:  level LOW" in out and "finish:" in out
    system = prompts.build_messages("x", {})[0]["content"]
    for text in (KEY, smoke_gemini.MESSAGE, system[:60], "Bangalore", GOOD):
        assert text not in out and text not in caplog.text


def test_smoke_with_provider_set_to_groq_still_tests_gemini(fake_gemini, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    client = fake_gemini([GOOD])
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    code, out = _smoke()
    assert code == 0 and len(client.calls) == 1 and "provider:  gemini" in out


@pytest.mark.parametrize("reply, expected", [
    (ClientError(429, "RESOURCE_EXHAUSTED", "secret detail"), "RateLimitError"),
    (ClientError(500, "INTERNAL", "secret detail"), "ClientError"),
    (None, "InvalidOrEmptyReply"),
])
def test_smoke_reports_the_error_class_and_details(fake_gemini, monkeypatch, reply, expected):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    fake_gemini([reply, reply])  # a 429 is retried once
    code, out = _smoke()
    assert code == 1 and f"result:    {expected}" in out and KEY not in out  # the message is now shown (redacted)
    assert ("secret detail" in out) == (reply is not None)
    assert "valid decision: no" in out
