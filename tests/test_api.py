"""
FastAPI-layer test suite.

Exercises app/api.py end-to-end (through the real conversational engine
in app/experiments/agent_modes.py - this suite does not duplicate or
bypass the pipeline) with the outbound Homilivo HTTP call mocked via
the fixtures in conftest.py, so it is fast and deterministic. Real
end-to-end verification against the live API stays in tests/test_mcp.py
and tests/real_api_test.py - this file does not replace those.
"""

from __future__ import annotations

import asyncio
import types
from collections import Counter

import httpx
import pytest

from app.api import app
from app.experiments.agent_modes import GemmaError


# ============================================================
# ROOT / HEALTH
# ============================================================

def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "active_sessions" in body


def test_chat_ui_is_served_same_origin(client):
    response = client.get("/chat-ui")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "/chat" in body
    assert "fetch(" in body


def test_chat_ui_never_hardcodes_a_secret(client):
    """
    Demo-safety requirement: the page must never contain a real token or
    key baked into the HTML/JS - only the client-side field name/label
    text and runtime fetch() calls that send whatever the user typed.
    """
    body = client.get("/chat-ui").text.lower()

    for leaked in ("kangaroo_token=", "groq_api_key=", "gsk_", "bearer ey"):
        assert leaked not in body


# ============================================================
# AUTHENTICATION
# ============================================================

def test_chat_missing_authorization_returns_401(client):
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 401


def test_chat_wrong_auth_scheme_returns_401(client):
    response = client.post(
        "/chat",
        json={"message": "hello"},
        headers={"Authorization": "Basic not-a-bearer-token"},
    )

    assert response.status_code == 401


def test_chat_valid_authorization_is_accepted(client, auth_headers, mock_homilivo_success):
    response = client.post(
        "/chat",
        json={"message": "hello"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["success"] is True


# ============================================================
# SESSION HANDLING
# ============================================================

def test_chat_creates_session_id_when_omitted(client, auth_headers, mock_homilivo_success):
    response = client.post("/chat", json={"message": "hello"}, headers=auth_headers)

    body = response.json()
    assert body["session_id"]


def test_chat_session_persists_across_calls(client, auth_headers, mock_homilivo_success):
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    session_id = search["session_id"]
    assert search["room_count"] == 3

    followup = client.post(
        "/chat",
        json={
            "message": "which one is cheapest?",
            "session_id": session_id,
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert followup["session_id"] == session_id
    assert followup["selected_room"] is not None
    assert followup["selected_room"]["id"] == "room-2"
    assert followup["selected_room"]["price_per_month"] == 8500


def test_chat_session_isolation(client, auth_headers, mock_homilivo_success, monkeypatch):
    # Session A: search, then select the cheapest room.
    search_a = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()
    session_a = search_a["session_id"]

    client.post(
        "/chat",
        json={"message": "which one is cheapest?", "session_id": session_a, "mode": "fast"},
        headers=auth_headers,
    )

    # Session B: a brand new session asking the same follow-up cold, with
    # no prior search. The deterministic router requires previous_rooms /
    # selected_room to treat this as a follow-up; with neither, it falls
    # through to the local Gemma intent classifier - mocked here to be
    # unavailable so the test is deterministic regardless of whether
    # Ollama happens to be running locally (this also exercises the
    # "local model unavailable" degradation path).
    monkeypatch.setattr(
        "app.experiments.agent_modes.classify_intent",
        lambda query: (_ for _ in ()).throw(GemmaError("no local model available")),
    )

    response_b = client.post(
        "/chat",
        json={"message": "which one is cheapest?", "mode": "fast"},
        headers=auth_headers,
    ).json()

    # Session B must NOT see session A's selected room or search results.
    assert response_b["session_id"] != session_a
    assert response_b["selected_room"] is None
    assert response_b["room_count"] == 0


def test_chat_reset_clears_session(client, auth_headers, mock_homilivo_success):
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()
    session_id = search["session_id"]
    assert search["room_count"] == 3

    reset = client.post("/chat/reset", params={"session_id": session_id}).json()
    assert reset["existed"] is True

    after_reset = client.get("/health").json()
    assert after_reset["status"] == "healthy"


def test_chat_reset_nonexistent_session_returns_existed_false(client):
    reset = client.post("/chat/reset", params={"session_id": "does-not-exist"}).json()

    assert reset["existed"] is False


# ============================================================
# MODE VALIDATION
# ============================================================

def test_chat_invalid_mode_returns_422(client, auth_headers):
    response = client.post(
        "/chat",
        json={"message": "hello", "mode": "not-a-real-mode"},
        headers=auth_headers,
    )

    assert response.status_code == 422


@pytest.mark.parametrize("mode", ["fast", "qwen", "gemma", "groq"])
def test_chat_all_valid_modes_are_accepted_by_validation(client, auth_headers, mode):
    """
    Every documented mode must pass Pydantic validation (not be rejected
    at the request-validation layer). The actual response-generation
    behaviour per mode is covered by the dedicated mode tests below.
    """
    response = client.post(
        "/chat",
        json={"message": "hello", "mode": mode},
        headers=auth_headers,
    )

    assert response.status_code != 422


def test_chat_qwen_mode_uses_local_model(client, auth_headers, mock_homilivo_success, monkeypatch):
    fake_reply = types.SimpleNamespace(message=types.SimpleNamespace(content="Qwen says hi."))
    monkeypatch.setattr(
        "app.experiments.agent_modes.chat",
        lambda model, messages: fake_reply,
    )

    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "session_id": None, "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "tell me more about this one",
            "session_id": search["session_id"],
            "mode": "qwen",
        },
        headers=auth_headers,
    ).json()

    assert response["mode"] == "qwen"
    assert response["success"] is True


def test_chat_gemma_mode_uses_local_model(client, auth_headers, mock_homilivo_success, monkeypatch):
    fake_reply = types.SimpleNamespace(message=types.SimpleNamespace(content="Gemma says hi."))
    monkeypatch.setattr(
        "app.experiments.agent_modes.chat",
        lambda model, messages: fake_reply,
    )

    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "tell me more about this one",
            "session_id": search["session_id"],
            "mode": "gemma",
        },
        headers=auth_headers,
    ).json()

    assert response["mode"] == "gemma"
    assert response["success"] is True


# ============================================================
# GROQ MODE
# ============================================================

def _fake_groq_client(content: str | None = None, fail: bool = False):
    def create(model, messages):
        if fail:
            raise RuntimeError("simulated Groq outage")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        )

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )


def test_chat_groq_mode_returns_llm_generated_text(client, auth_headers, mock_homilivo_success, monkeypatch):
    monkeypatch.setattr(
        "app.experiments.agent_modes._get_groq_client",
        lambda: _fake_groq_client(content="Groq-generated answer."),
    )

    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "tell me more about this one",
            "session_id": search["session_id"],
            "mode": "groq",
        },
        headers=auth_headers,
    ).json()

    assert response["mode"] == "groq"
    assert response["message"] == "Groq-generated answer."


def test_chat_groq_failure_falls_back_gracefully(client, auth_headers, mock_homilivo_success, monkeypatch):
    monkeypatch.setattr(
        "app.experiments.agent_modes._get_groq_client",
        lambda: _fake_groq_client(fail=True),
    )

    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "which one is the most expensive?",
            "session_id": search["session_id"],
            "mode": "groq",
        },
        headers=auth_headers,
    )

    # A Groq outage must degrade to the fast-mode fallback, not a 500.
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "18,000" in body["message"] or "18000" in body["message"]


# ============================================================
# MALFORMED REQUESTS / PYDANTIC VALIDATION
# ============================================================

def test_chat_missing_message_field_returns_422(client, auth_headers):
    response = client.post("/chat", json={"mode": "fast"}, headers=auth_headers)

    assert response.status_code == 422


def test_chat_empty_message_returns_422(client, auth_headers):
    response = client.post("/chat", json={"message": ""}, headers=auth_headers)

    assert response.status_code == 422


def test_chat_overlong_message_returns_422(client, auth_headers):
    response = client.post(
        "/chat",
        json={"message": "x" * 501},
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_chat_non_json_body_returns_422(client, auth_headers):
    response = client.post(
        "/chat",
        content=b"this is not json",
        headers={**auth_headers, "Content-Type": "application/json"},
    )

    assert response.status_code == 422


# ============================================================
# GRACEFUL DEGRADATION
# ============================================================

def test_chat_invalid_price_range_degrades_gracefully(client, auth_headers, mock_homilivo_success):
    """
    A query whose extracted parameters fail the parameter guardrail
    (min_price > max_price) must not crash the request - it should
    come back as a normal 200 with an apologetic fallback message.
    """
    response = client.post(
        "/chat",
        json={"message": "rooms in Jaipur above 20000 under 10000", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["success"] is False


def test_chat_homilivo_api_failure_degrades_gracefully(client, auth_headers, mock_homilivo_failure):
    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["room_count"] == 0


def test_chat_out_of_scope_contact_owner_does_not_rerun_search(client, auth_headers, mock_homilivo_success):
    """
    Regression for Issue 1: "how to contact with the owner?" must not be
    silently treated as a search refinement (the bare "with " substring
    in refinement_patterns used to cause this) and must not come back
    looking like a fresh room search.
    """
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "how to contact with the owner?",
            "session_id": search["session_id"],
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert "Here are the" not in response["message"]
    assert "matching options" not in response["message"]


def test_chat_set_wide_amenity_question_is_answered_from_session_data(client, auth_headers, mock_homilivo_success):
    """
    Regression for Issue 2: "does any room have ac?" after a search must
    be answered from the already-verified room set, not the generic
    capability fallback message.
    """
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()

    response = client.post(
        "/chat",
        json={
            "message": "does any room have ac?",
            "session_id": search["session_id"],
            "mode": "fast",
        },
        headers=auth_headers,
    ).json()

    assert "Sunrise 2BHK" in response["message"]
    assert "I can answer questions about" not in response["message"]


def test_chat_empty_search_results(client, auth_headers, mock_homilivo_empty):
    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["room_count"] == 0
    assert "jaipur" in body["message"].lower()


# ============================================================
# CONCURRENT REQUEST TOKEN ISOLATION (integration-level)
# ============================================================
# app/server.py/test_token_context.py already proves the ContextVar
# mechanism itself in isolation; this proves it holds through a real
# concurrent HTTP request lifecycle (guardrails -> parser -> MCP call).

def test_concurrent_chat_requests_do_not_leak_tokens(mock_homilivo_token_recorder):
    get_recorded_tokens = mock_homilivo_token_recorder
    token_count = 15

    async def post_one(async_client: httpx.AsyncClient, index: int) -> httpx.Response:
        return await async_client.post(
            "/chat",
            json={"message": "find rooms in Jaipur", "mode": "fast"},
            headers={"Authorization": f"Bearer token-{index}"},
        )

    async def run() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
            return await asyncio.gather(*(post_one(async_client, i) for i in range(token_count)))

    responses = asyncio.run(run())

    for response in responses:
        assert response.status_code == 200

    expected = Counter(f"Bearer token-{i}" for i in range(token_count))
    actual = Counter(get_recorded_tokens())

    assert actual == expected
