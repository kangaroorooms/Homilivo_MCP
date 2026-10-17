"""Phase 7, offline: the CLI and the server with a missing or empty key. The message must be
clear and must not print any secret. No network, no real ``.env`` (``load_dotenv`` is stubbed)."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from llm_approach import cli, llm_client, pipeline, reuse, server_app
from llm_approach.state import SessionStore

OTHER_SECRET = "FAKEROOMSTOKEN-should-never-print"


@pytest.fixture
def no_dotenv(monkeypatch):
    """Stop the CLI from reading the developer's real .env during the test."""
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)


@pytest.mark.parametrize("value", [None, ""])
def test_cli_without_groq_key_says_so_and_exits(no_dotenv, monkeypatch, capsys, value):
    if value is None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GROQ_API_KEY", value)
    monkeypatch.setenv("KANGAROO_TOKEN", OTHER_SECRET)
    monkeypatch.setattr(cli, "run_chat", lambda **k: pytest.fail("the chat must not start without a key"))
    code = cli.main(["--offline-rooms", "--skip-guardrail-load"])
    out = capsys.readouterr()
    text = out.out + out.err
    assert code == 2
    assert "GROQ_API_KEY is not set" in text and ".env" in text
    assert OTHER_SECRET not in text


def test_cli_without_rooms_token_warns_without_printing_the_groq_key(no_dotenv, monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_FAKE_not_real_key_123")
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)
    monkeypatch.setattr(cli, "run_chat", lambda **k: None)
    try:
        assert cli.main(["--skip-guardrail-load"]) == 0
    finally:
        pipeline.configure(full_guardrails=True)
    text = capsys.readouterr().out
    assert "KANGAROO_TOKEN is not set" in text and "--offline-rooms" in text
    assert "gsk_FAKE" not in text


@pytest.fixture
def server_without_keys(monkeypatch):
    from app.rate_limit import InMemoryRateLimiter

    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)
    limiter = InMemoryRateLimiter()
    monkeypatch.setattr(server_app._rl, "get_rate_limiter", lambda: limiter)
    monkeypatch.setattr(pipeline, "_store", SessionStore())
    monkeypatch.setattr(llm_client, "_client", None)  # build the real client, which needs the key
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    pipeline.configure(full_guardrails=True)
    yield
    pipeline.configure(full_guardrails=True)


def test_server_starts_without_keys_and_reports_them_as_missing(server_without_keys):
    with TestClient(server_app.create_app()) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["checks"]["groq_key_present"] is False
        assert client.get("/chat-ui/demo-token").status_code == 404  # no KANGAROO_TOKEN to issue from


@pytest.mark.parametrize("empty", [False, True])
def test_chat_without_groq_key_gives_a_friendly_reply_not_a_crash(server_without_keys, monkeypatch, empty):
    if empty:
        monkeypatch.setenv("GROQ_API_KEY", "")
    with TestClient(server_app.create_app()) as client:
        r = client.post("/chat", json={"message": "hello", "session_id": str(uuid.uuid4())},
                        headers={"Authorization": "Bearer anything"})
    assert r.status_code == 200
    for bad in ("Traceback", "KeyError", "GROQ_API_KEY", "GroqError", "api_key"):
        assert bad not in r.text, f"{bad} leaked into the reply"
    assert r.json()["message"]  # the user still gets a reply (the fallback question)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_llm_client_refuses_a_blank_key_before_any_request(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GROQ_API_KEY", value)
    monkeypatch.setattr(llm_client, "_client", None)
    with pytest.raises(KeyError):
        llm_client._get_client()
    assert llm_client._client is None
