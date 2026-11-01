"""Gemini API error details (HTTP code, status), redaction, the smoke request shape line and the automatic
narrowing. Fake client only: no SDK, no network, no key."""

from __future__ import annotations

import asyncio
import logging
import re

import pytest

from llm_approach import config, llm_client, prompts, providers, smoke_gemini
from llm_approach.llm_client import LLMBudget, decide

from .test_gemini_provider import GOOD, KEY, RAW_MARK, USER_TEXT, ClientError, run

TOKEN = "ya29." + "a0AfH6SMBxyzABCDEFGHIJKLMNOP"  # fake token-like string, split for scanners
LABELS = ["without the thinking config", "without JSON response mode", "without the system instruction",
          "without the output cap", "with only the last user message", "without the timeout option"]


def _smoke():
    lines: list[str] = []
    return run(smoke_gemini.run(out=lines.append)), "\n".join(lines)


@pytest.fixture
def narrowing(fake_gemini, monkeypatch):
    """Fake client for the first call and the narrowing; the no-timeout client is the same fake."""
    monkeypatch.setenv("GEMINI_API_KEY", KEY)

    def install(replies):
        client = fake_gemini(replies)
        monkeypatch.setattr(providers, "_new_gemini_client", lambda timeout_ms: client)
        return client

    return install


# ---- error capture ----------------------------------------------------------------------


def test_error_details_code_and_status():
    assert providers.error_details(ClientError(400, "INVALID_ARGUMENT", "m")) == (400, "INVALID_ARGUMENT")


def test_error_details_without_message_and_without_attributes():
    assert providers.error_details(ClientError(404, "NOT_FOUND")) == (404, "NOT_FOUND")
    assert providers.error_details(RuntimeError("boom")) == (None, None)
    assert providers.error_details(ClientError("x", "free text with spaces")) == (None, None)  # validated


def test_decide_records_code_status_and_redacted_message(fake_gemini, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    fake_gemini([ClientError(400, "INVALID_ARGUMENT", f"bad {KEY} {TOKEN} field")])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is None
    assert (budget.error_class, budget.error_code, budget.error_status) == ("ClientError", 400, "INVALID_ARGUMENT")
    assert KEY not in budget.error_message and TOKEN not in budget.error_message and "***" in budget.error_message


def test_decide_without_a_message(fake_gemini):
    fake_gemini([ClientError(400, "INVALID_ARGUMENT")])
    budget = LLMBudget()
    run(decide("hi", {}, budget))
    assert budget.error_code == 400 and budget.error_message == ""


def test_429_stays_rate_limit_with_code_and_status(fake_gemini):
    # a 429 is retried once after the wait (see test_gemini_limits); a second one is the rate-limit error
    client = fake_gemini([ClientError(429, "RESOURCE_EXHAUSTED", "quota"), ClientError(429, "RESOURCE_EXHAUSTED", "quota")])
    budget = LLMBudget()
    assert run(decide("hi", {}, budget)) is None
    assert budget.error_class == "RateLimitError" and budget.error_code == 429 and len(client.calls) == 2


def test_log_line_has_class_code_status_and_no_message(fake_gemini, monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    fake_gemini([ClientError(400, "INVALID_ARGUMENT", f"{KEY} {USER_TEXT} {RAW_MARK} {TOKEN}")])
    with caplog.at_level(logging.DEBUG):
        run(decide(USER_TEXT, {}, LLMBudget()))
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "class=ClientError code=400 status=INVALID_ARGUMENT" in logged
    for secret in (KEY, TOKEN, USER_TEXT, RAW_MARK):
        assert secret not in logged


def test_answerer_log_line_has_code_and_status(fake_gemini, caplog):
    fake_gemini([ClientError(403, "PERMISSION_DENIED", "hidden message")])
    with caplog.at_level(logging.DEBUG):
        run(llm_client.generate_json([{"role": "user", "content": "u"}], LLMBudget()))
    assert "code=403 status=PERMISSION_DENIED" in caplog.text and "hidden message" not in caplog.text


# ---- redaction ----------------------------------------------------------------------------


def test_redact_key_token_and_limit(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "shortkey")
    out = providers.redact(f"a shortkey b {TOKEN} c ABCDEFGHIJKLMNOPQRST d")
    assert out == "a *** b *** c *** d"
    assert len(providers.redact("word " * 100)) == 200


# ---- request shape ---------------------------------------------------------------------------


def test_request_shape_line_has_structure_only():
    messages = [{"role": "system", "content": "SYSTEMTEXTzulu"}, {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"}, {"role": "user", "content": "now"}]
    contents, cfg = providers.build_gemini_request(messages, 1024)
    line = providers.request_shape(contents, cfg)
    assert "messages 3 (roles: user, model, user)" in line and "system instruction yes, 14 chars" in line
    assert "total 21 chars" in line and "'temperature': 0" in line and "'max_output_tokens': 1024" in line
    assert "'response_mime_type': 'application/json'" in line and "{'thinking_level': 'LOW'}" in line
    assert "timeout 20000 ms" in line
    assert "SYSTEMTEXTzulu" not in line and "q1" not in line and "now" not in line


# ---- narrowing --------------------------------------------------------------------------------


@pytest.mark.parametrize("first", [GOOD, ClientError(429, "RESOURCE_EXHAUSTED", "q"), ClientError(500, "INTERNAL", "x")])
def test_no_narrowing_on_success_429_or_5xx(narrowing, first):
    is_429 = getattr(first, "code", None) == 429
    client = narrowing([first, first] if is_429 else [first])  # a 429 is retried once, so it is sent twice
    _, out = _smoke()
    assert len(client.calls) == (2 if is_429 else 1) and "narrowing" not in out


def test_narrowing_order_count_and_one_change_each(narrowing):
    err = ClientError(400, "INVALID_ARGUMENT", "bad request")
    client = narrowing([err] * 5 + [GOOD, err])  # real call, a-d fail, e) last user message works, f) fails
    code, out = _smoke()
    assert code == 1 and len(client.calls) == 7  # the real call + 6 variations
    first, a, b, c, d, e, f = client.calls
    cfg = first["config"]
    assert "thinking_config" in cfg and set(cfg) - set(a["config"]) == {"thinking_config"}
    assert set(cfg) - set(b["config"]) == {"response_mime_type"}
    assert set(cfg) - set(c["config"]) == {"system_instruction"}
    assert c["contents"][0]["parts"][0]["text"].startswith(cfg["system_instruction"])
    assert set(cfg) - set(d["config"]) == {"max_output_tokens"}
    assert e["contents"] == [first["contents"][-1]] and e["config"] == cfg
    assert f["contents"] == first["contents"] and f["config"] == cfg
    results = re.findall(r"^  (.+?): (OK|FAILED[^\n]*)$", out, re.M)
    assert [label for label, _ in results] == LABELS
    assert [r.split()[0] for _, r in results] == ["FAILED"] * 4 + ["OK", "FAILED"]
    assert "FAILED ClientError code 400 status INVALID_ARGUMENT" in out
    assert "first variation that succeeded: with only the last user message" in out


def test_narrowing_none_succeeded(narrowing):
    client = narrowing([ClientError(400, "INVALID_ARGUMENT", "bad")] * 7)
    _, out = _smoke()
    assert len(client.calls) == 7 and "first variation that succeeded: none succeeded" in out


def test_narrowing_reports_other_errors_per_variation(narrowing):
    err = ClientError(404, "NOT_FOUND", "m")
    narrowing([err, GOOD, RuntimeError("boom"), err, err, err, err])
    _, out = _smoke()
    assert "without the thinking config: OK" in out and "without JSON response mode: FAILED RuntimeError" in out
    assert "first variation that succeeded: without the thinking config" in out


def test_smoke_prints_code_status_and_redacted_message(narrowing):
    narrowing([ClientError(400, "INVALID_ARGUMENT", f"bad {KEY} {TOKEN}")] + [GOOD] * 6)
    _, out = _smoke()
    assert "error:     code 400, status INVALID_ARGUMENT, message 'bad *** ***'" in out
    assert "request:   messages 1 (roles: user)" in out and "system instruction yes," in out
    assert KEY not in out and TOKEN not in out


def test_smoke_message_is_cut_to_200_chars(narrowing):
    narrowing([ClientError(400, "INVALID_ARGUMENT", "x y " * 200)] + [GOOD] * 6)
    _, out = _smoke()
    line = next(row for row in out.splitlines() if row.startswith("error:"))
    assert len(line.split("message ", 1)[1]) <= 200 + 2  # plus the repr quotes


def test_no_prompt_or_reply_text_in_smoke_output_or_logs(narrowing, caplog):
    narrowing([ClientError(400, "INVALID_ARGUMENT", "m")] + [f"{RAW_MARK} reply"] * 6)
    with caplog.at_level(logging.DEBUG):
        _, out = _smoke()
    system = prompts.build_messages("x", {})[0]["content"]
    for text in (smoke_gemini.MESSAGE, system[:60], RAW_MARK, KEY):
        assert text not in out and text not in caplog.text


# ---- request-building checks ---------------------------------------------------------------------


def test_timeout_is_passed_in_milliseconds(monkeypatch):
    monkeypatch.delenv(config.GEMINI_TIMEOUT_ENV, raising=False)
    assert providers.gemini_timeout_ms() == 20000


def test_gemini_timeout_env_override(monkeypatch):
    monkeypatch.setenv(config.GEMINI_TIMEOUT_ENV, "30")
    assert providers.gemini_timeout_ms() == 30000


def test_gemini_timeout_below_minimum_is_clamped_with_one_warning(monkeypatch, caplog):
    monkeypatch.setenv(config.GEMINI_TIMEOUT_ENV, "8")
    with caplog.at_level("WARNING", logger="llm_approach.config"):
        assert providers.gemini_timeout_ms() == 10000
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and "8" not in warnings[0].getMessage().replace("10", "")


def test_groq_timeout_is_still_8_seconds():
    assert config.LLM_TIMEOUT_S == 8


def test_request_shape_line_shows_the_gemini_timeout(monkeypatch):
    monkeypatch.delenv(config.GEMINI_TIMEOUT_ENV, raising=False)
    contents, cfg = providers.build_gemini_request([{"role": "user", "content": "q"}], 500)
    assert "timeout 20000 ms" in providers.request_shape(contents, cfg)


def test_empty_text_parts_are_dropped_and_no_system_role_reaches_contents():
    system, contents = providers.to_gemini_request([
        {"role": "system", "content": "S"}, {"role": "user", "content": ""}, {"role": "user", "content": "  "},
        {"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}])
    assert system == "S" and [c["role"] for c in contents] == ["model", "user"]
    assert all(p["text"].strip() for c in contents for p in c["parts"])


def test_real_request_has_only_supported_config_keys_and_roles():
    contents, cfg = providers.build_gemini_request(prompts.build_messages("hi", {}), 1024)
    assert set(cfg) == {"temperature", "max_output_tokens", "response_mime_type", "thinking_config",
                        "system_instruction"}
    assert [c["role"] for c in contents] == ["user"] and all(c["parts"][0]["text"] for c in contents)


def test_gemini_turn_stops_calling_after_the_time_budget(monkeypatch):
    monkeypatch.setenv(config.PROVIDER_ENV, "gemini")
    budget = llm_client.LLMBudget()
    assert budget.try_consume()
    budget.record_latency(config.GEMINI_TURN_LLM_BUDGET_S * 1000)
    assert not budget.try_consume()  # worst case per turn: 25 s + one 20 s call
    monkeypatch.setenv(config.PROVIDER_ENV, "groq")
    assert budget.try_consume()  # Groq: call count only, unchanged
