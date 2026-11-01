# Copied from tests/test_p3_13_secret_management.py (startup validation cases) at commit 7dbcba3 (copied, not imported).
# Changes: validate_secrets now comes from llm_approach.security.secrets_check and also requires the
# provider's key (GEMINI_API_KEY by default), so the tests set LLM_APPROACH_PROVIDER/keys accordingly.
# Every changed line is marked '# v2:'.
# Not copied: the server-log token cases (need app.server) and the .env.example case (not about this module).

import logging

import pytest

from llm_approach.security.secrets_check import validate_secrets  # v2: was: from app.api import validate_secrets

FAKE_TOKEN = "test-token-123"


@pytest.fixture(autouse=True)
def _gemini_provider(monkeypatch):  # v2: pins the provider so each test is independent of the machine's env
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "gemini")  # v2


def test_required_secret_missing_raises(monkeypatch):
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")  # v2: provider key present so only KANGAROO_TOKEN is missing

    with pytest.raises(RuntimeError, match="KANGAROO_TOKEN"):
        validate_secrets()


def test_required_secret_empty_raises(monkeypatch):
    monkeypatch.setenv("KANGAROO_TOKEN", "")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")  # v2

    with pytest.raises(RuntimeError, match="KANGAROO_TOKEN"):
        validate_secrets()


def test_required_secret_present_no_error(monkeypatch):
    monkeypatch.setenv("KANGAROO_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")  # v2: was: GROQ_API_KEY / fake-groq-key

    validate_secrets()  # must not raise


def test_provider_key_missing_raises(monkeypatch, caplog):  # v2: new case; replaces the optional-key warning case
    monkeypatch.setenv("KANGAROO_TOKEN", FAKE_TOKEN)  # v2
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # v2

    with caplog.at_level(logging.WARNING):  # v2
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):  # v2
            validate_secrets()  # v2

    assert FAKE_TOKEN not in caplog.text  # v2


def test_groq_provider_requires_groq_key(monkeypatch):  # v2: new case
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")  # v2
    monkeypatch.setenv("KANGAROO_TOKEN", FAKE_TOKEN)  # v2
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")  # v2
    monkeypatch.delenv("GROQ_API_KEY", raising=False)  # v2

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):  # v2
        validate_secrets()  # v2

    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")  # v2
    validate_secrets()  # v2: must not raise


def test_startup_error_does_not_contain_secret_value(monkeypatch, caplog):
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-xyz")  # v2: was: GROQ_API_KEY / fake-groq-key-xyz

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError) as excinfo:
            validate_secrets()

    assert "fake-gemini-key-xyz" not in str(excinfo.value)  # v2
    assert "fake-gemini-key-xyz" not in caplog.text  # v2
