# P3-18: Observability and Secret-Redaction Tests
# Verifies structured logging contains required
# fields and never leaks credentials or PII.

from __future__ import annotations

import logging
import re
import urllib.error
import uuid

import httpx
import pytest

from app.experiments.agent_modes import fetch_subscription_plans
from app.server import search_rooms, set_kangaroo_token

USER_MESSAGE = "find rooms in Jaipur under 15000 zebra-marker-7431"
REQUIRED_TURN_FIELDS = (
    "request_id",
    "session_id",
    "intent",
    "tool",
    "mode",
    "status",
    "latency_ms",
    "error_category",
)


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        self.lines.append(f"{record.levelname} {record.getMessage()}")


@pytest.fixture()
def captured_logs():
    """
    The "chatbot" logger (app/observability.py) has propagate=False, so
    pytest's caplog (a root-logger handler) never sees the structured
    turn line. Attach a handler directly to it AND to the root logger so
    every logger in the app is captured.
    """
    handler = _ListHandler()
    chatbot_logger = logging.getLogger("chatbot")
    root_logger = logging.getLogger()
    old_levels = (chatbot_logger.level, root_logger.level)

    chatbot_logger.addHandler(handler)
    root_logger.addHandler(handler)
    chatbot_logger.setLevel(logging.DEBUG)
    root_logger.setLevel(logging.DEBUG)

    yield handler

    chatbot_logger.removeHandler(handler)
    root_logger.removeHandler(handler)
    chatbot_logger.setLevel(old_levels[0])
    root_logger.setLevel(old_levels[1])


def _all_log_text(handler: _ListHandler) -> str:
    return "\n".join(handler.lines)


def _turn_lines(handler: _ListHandler) -> list[str]:
    return [line for line in handler.lines if " turn request_id=" in line]


def _chat(client, headers, message=USER_MESSAGE, **extra):
    return client.post(
        "/chat",
        json={"message": message, "mode": "fast", **extra},
        headers=headers,
    )


# ============================================================
# TEST 1
# ============================================================

def test_turn_log_contains_required_fields(
    client, auth_headers, mock_homilivo_success, captured_logs
):
    response = _chat(client, auth_headers)
    assert response.status_code == 200

    turns = _turn_lines(captured_logs)
    assert len(turns) == 1, captured_logs.lines

    for field in REQUIRED_TURN_FIELDS:
        assert f"{field}=" in turns[0], f"{field} missing from turn log"

    body = response.json()
    assert f"request_id={body['request_id']}" in turns[0]
    assert f"session_id={body['session_id']}" in turns[0]
    assert re.search(r"latency_ms=\d+\.\d", turns[0])


# ============================================================
# TEST 2
# ============================================================

def test_turn_log_does_not_contain_user_message(
    client, auth_headers, mock_homilivo_success, captured_logs
):
    _chat(client, auth_headers)

    log_text = _all_log_text(captured_logs)
    assert "zebra-marker-7431" not in log_text
    assert USER_MESSAGE not in log_text


# ============================================================
# TEST 3
# ============================================================

def test_turn_log_does_not_contain_bearer_token(
    client, mock_homilivo_success, captured_logs
):
    secret_token = "p3-18-secret-token-9f8e7d6c5b4a"
    headers = {"Authorization": f"Bearer {secret_token}"}

    _chat(client, headers)

    log_text = _all_log_text(captured_logs)
    assert secret_token not in log_text
    assert "Bearer" not in log_text


# ============================================================
# TEST 4
# ============================================================

def test_session_id_in_logs_is_uuid_not_pii(
    client, auth_headers, mock_homilivo_success, captured_logs
):
    _chat(client, auth_headers)

    turns = _turn_lines(captured_logs)
    assert turns

    match = re.search(r"session_id=(\S+)", turns[0])
    assert match, turns[0]
    logged_session_id = match.group(1)

    # Raises ValueError if it is not a UUID.
    parsed = uuid.UUID(logged_session_id)
    assert str(parsed) == logged_session_id

    assert re.fullmatch(r"[0-9a-f-]{36}", logged_session_id)  # UUID pattern
    assert "@" not in logged_session_id  # no email
    assert " " not in logged_session_id  # no free-text name


# ============================================================
# TEST 5
# ============================================================

def test_api_error_log_does_not_contain_token(
    client, monkeypatch, captured_logs
):
    secret_token = "p3-18-401-token-abcdef123456"

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized", {}, None
        )

    monkeypatch.setattr("app.server.urllib.request.urlopen", fake_urlopen)

    # Direct tool call: the 401 error log must exist and be token-free.
    reset = set_kangaroo_token(secret_token)
    try:
        result = search_rooms(city="Jaipur")
    finally:
        from app.server import _kangaroo_token_var

        _kangaroo_token_var.reset(reset)

    assert result["success"] is False
    log_text = _all_log_text(captured_logs)
    assert "unauthorized (401)" in log_text
    assert secret_token not in log_text

    # Same failure through the HTTP layer.
    headers = {"Authorization": f"Bearer {secret_token}"}
    _chat(client, headers)
    log_text = _all_log_text(captured_logs)
    assert secret_token not in log_text
    assert "Bearer" not in log_text


# ============================================================
# TEST 6
# ============================================================

class _ExplodingClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def get(self, *args, **kwargs):
        raise RuntimeError("simulated pricing outage")


def test_pricing_api_error_logged_as_warning(monkeypatch, captured_logs):
    secret_value = "p3-18-pricing-secret-zzz999"
    monkeypatch.setenv("KANGAROO_TOKEN", secret_value)
    monkeypatch.setenv("GROQ_API_KEY", secret_value)
    monkeypatch.setattr(httpx, "Client", _ExplodingClient)

    assert fetch_subscription_plans("bangalore") is None

    warnings = [
        r for r in captured_logs.records
        if r.levelno == logging.WARNING and "fetch_subscription_plans" in r.getMessage()
    ]
    assert warnings, captured_logs.lines
    assert "simulated pricing outage" in warnings[0].getMessage()
    assert secret_value not in _all_log_text(captured_logs)


def test_pricing_malformed_json_logs_length_not_body(monkeypatch, captured_logs):
    body_marker = "SENSITIVE-BODY-MARKER-5521"

    class _BadJsonResponse:
        text = body_marker * 5

        def raise_for_status(self) -> None:
            return None

        def json(self):
            raise ValueError("bad json")

    class _BadJsonClient(_ExplodingClient):
        def get(self, *args, **kwargs):
            return _BadJsonResponse()

    monkeypatch.setattr(httpx, "Client", _BadJsonClient)

    assert fetch_subscription_plans("bangalore") is None

    log_text = _all_log_text(captured_logs)
    assert "malformed JSON" in log_text
    assert body_marker not in log_text


# ============================================================
# TEST 8
# ============================================================

def test_invalid_session_id_replaced_with_uuid(
    client, auth_headers, mock_homilivo_success, captured_logs
):
    bad_session_id = "user@email.com"

    response = _chat(client, auth_headers, session_id=bad_session_id)
    assert response.status_code == 200

    log_text = _all_log_text(captured_logs)
    assert bad_session_id not in log_text
    assert "user@email" not in log_text
    assert "Invalid session_id format received" in log_text

    used_session_id = response.json()["session_id"]
    assert used_session_id != bad_session_id
    assert str(uuid.UUID(used_session_id)) == used_session_id

    turns = _turn_lines(captured_logs)
    assert f"session_id={used_session_id}" in turns[0]


@pytest.mark.parametrize("case", [str.lower, str.upper])
def test_valid_uuid_session_id_is_accepted_and_lowercased(
    client, auth_headers, mock_homilivo_success, captured_logs, case
):
    canonical = str(uuid.uuid4())

    response = _chat(client, auth_headers, session_id=case(canonical))

    assert response.json()["session_id"] == canonical
    assert "Invalid session_id" not in _all_log_text(captured_logs)


def test_upper_and_lower_case_uuid_share_one_session(
    client, auth_headers, mock_homilivo_success
):
    from app.api import SESSIONS

    canonical = str(uuid.uuid4())

    _chat(client, auth_headers, session_id=canonical.upper())
    second = _chat(client, auth_headers, session_id=canonical.lower())

    assert second.json()["session_id"] == canonical
    history = [m["content"] for m in SESSIONS.get(canonical).conversation_history]
    assert history.count(USER_MESSAGE) == 2  # both turns landed in one session


def test_invalid_websocket_session_id_replaced(
    client, mock_homilivo_success, captured_logs
):
    bad_session_id = "jane.doe@example.com"

    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        ws.send_json(
            {"message": "find rooms in Jaipur", "mode": "fast", "session_id": bad_session_id}
        )
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event.get("event") in ("done", "error"):
                break

    assert bad_session_id not in _all_log_text(captured_logs)
    assert bad_session_id not in str(events)

    done = next(e for e in events if e.get("event") == "done")
    assert str(uuid.UUID(done["session_id"])) == done["session_id"]
