# P3-13: Secret Management Tests
# Verifies startup validation of required secrets
# and that credentials never appear in logs.

import io
import logging
import re
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from app import server
from app.api import validate_secrets

FAKE_TOKEN = "test-token-123"
ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body

    def read(self):
        return self._body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def bound_token(monkeypatch):
    """Bind FAKE_TOKEN as the per-request token; skip retry sleeps."""
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)
    reset = server.set_kangaroo_token(FAKE_TOKEN)
    yield FAKE_TOKEN
    server._kangaroo_token_var.reset(reset)


# ------------------------------------------------------------
# Startup validation
# ------------------------------------------------------------

def test_required_secret_missing_raises(monkeypatch):
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="KANGAROO_TOKEN"):
        validate_secrets()


def test_required_secret_empty_raises(monkeypatch):
    monkeypatch.setenv("KANGAROO_TOKEN", "")

    with pytest.raises(RuntimeError, match="KANGAROO_TOKEN"):
        validate_secrets()


def test_required_secret_present_no_error(monkeypatch):
    monkeypatch.setenv("KANGAROO_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")

    validate_secrets()  # must not raise


def test_optional_secret_missing_warns(monkeypatch, caplog):
    monkeypatch.setenv("KANGAROO_TOKEN", FAKE_TOKEN)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with caplog.at_level(logging.WARNING):
        validate_secrets()  # must not raise

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("GROQ_API_KEY" in r.getMessage() for r in warnings)
    assert FAKE_TOKEN not in caplog.text


def test_startup_error_does_not_contain_secret_value(monkeypatch, caplog):
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key-xyz")

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError) as excinfo:
            validate_secrets()

    assert "fake-groq-key-xyz" not in str(excinfo.value)
    assert "fake-groq-key-xyz" not in caplog.text


# ------------------------------------------------------------
# Token never appears in logs / returned errors
# ------------------------------------------------------------

def test_token_not_in_server_logs(bound_token, caplog):
    body = '{"data": [{"id": 1, "title": "Room", "city": "Mumbai"}]}'

    with caplog.at_level(logging.DEBUG):
        with patch(
            "app.server.urllib.request.urlopen",
            lambda request, timeout=None: _FakeResponse(body),
        ):
            result = server.search_rooms(city="Mumbai")

    assert result["success"] is True
    assert FAKE_TOKEN not in caplog.text
    assert FAKE_TOKEN not in str(result)


def test_token_not_in_error_logs(bound_token, caplog):
    def raise_401(request, timeout=None):
        raise urllib.error.HTTPError(
            server.API_URL, 401, "Unauthorized", {}, io.BytesIO(b"")
        )

    with caplog.at_level(logging.DEBUG):
        with patch("app.server.urllib.request.urlopen", raise_401):
            result = server.search_rooms(city="Mumbai")

    assert result["success"] is False
    assert any("401" in r.getMessage() for r in caplog.records)  # it did log
    assert FAKE_TOKEN not in caplog.text
    assert FAKE_TOKEN not in str(result)


def test_token_not_in_malformed_response_logs(bound_token, caplog):
    with caplog.at_level(logging.DEBUG):
        with patch(
            "app.server.urllib.request.urlopen",
            lambda request, timeout=None: _FakeResponse("not json"),
        ):
            result = server.search_rooms(city="Mumbai")

    assert result["success"] is False
    assert FAKE_TOKEN not in caplog.text
    assert FAKE_TOKEN not in str(result)


# ------------------------------------------------------------
# .env.example
# ------------------------------------------------------------

def _env_example_values() -> dict[str, str]:
    values = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def test_env_example_contains_required_secrets():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")

    assert "KANGAROO_TOKEN" in text
    assert "GROQ_API_KEY" in text
    assert "DEEPGRAM_API_KEY" in text

    jwt_like = re.compile(r"^[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}$")
    secret_keys = ("KANGAROO_TOKEN", "GROQ_API_KEY", "DEEPGRAM_API_KEY")
    values = _env_example_values()

    for key in secret_keys:
        value = values[key]
        assert "your_" in value and "_here" in value, f"{key} is not a placeholder"
        assert len(value) <= 50, f"{key} value looks too long to be a placeholder"
        assert not jwt_like.match(value), f"{key} looks like a JWT"

    # No variable at all may look like a real credential.
    for key, value in values.items():
        assert not jwt_like.match(value), f"{key} looks like a JWT"
        assert not value.startswith(("gsk_", "sk-")), f"{key} looks like an API key"
