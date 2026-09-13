"""
Genuine-branch-divergence tests.

Confirms each of the four LangGraph branches (app/graph/nodes/execute.py)
now does something different, not just that routing picks the right
branch name (that is already covered by tests/test_graph_router.py):

  - FAQ branch calls answer_platform_question() directly and skips
    process_message() entirely on a grounded hit, falling back to
    process_message() only when nothing grounded is found.
  - Conversation branch returns the fixed greeting response directly
    for a detected greeting, skipping process_message() entirely, and
    falls back to process_message() for anything else routed there
    (CHATBOT_INFO / EXIT / UNSUPPORTED_ACTION).
  - API-request branch is unchanged: it always calls process_message().

All assertions go through the real HTTP /chat endpoint (same pattern as
tests/test_graph_router.py) so they also exercise app/api.py's response
shape, not just the node functions in isolation.
"""

from __future__ import annotations

import pytest

from app.experiments.agent_modes import GemmaError
from app.graph.nodes import execute as execute_module


def _fake_process_message_factory(called: dict, response: dict):
    async def fake_process_message(**kwargs):
        called["process_message"] = True
        return response

    return fake_process_message


def _empty_process_message_result(**overrides) -> dict:
    result = {
        "response": None,
        "rooms": [],
        "search": {},
        "selected_room": None,
        "selected_room_id": None,
        "pending_search": {},
        "sources": [],
        "search_performed": False,
        "intent": None,
        "operation": None,
        "tool": None,
        "displayed_rooms": None,
    }
    result.update(overrides)
    return result


# ============================================================
# 1. FAQ branch calls answer_platform_question() directly
# ============================================================

def test_faq_branch_calls_answer_platform_question_directly(
    client, auth_headers, monkeypatch
):
    called = {"answer_platform_question": False, "process_message": False}

    def fake_answer(query):
        called["answer_platform_question"] = True
        return {
            "grounded": True,
            "message": "KNOWN_GROUNDED_ANSWER",
            "sources": ["Doc A"],
        }

    monkeypatch.setattr(execute_module.agent_modes, "answer_platform_question", fake_answer)
    monkeypatch.setattr(
        execute_module,
        "process_message",
        _fake_process_message_factory(called, _empty_process_message_result()),
    )

    response = client.post(
        "/chat",
        json={"message": "how does verification work?", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert called["answer_platform_question"] is True
    assert called["process_message"] is False
    assert "KNOWN_GROUNDED_ANSWER" in response.json()["message"]
    assert response.json()["sources"] == ["Doc A"]


def test_faq_branch_falls_back_to_process_message_when_no_answer(
    client, auth_headers, monkeypatch
):
    called = {"process_message": False}

    def fake_answer(query):
        return {"grounded": False, "message": "not available", "sources": []}

    monkeypatch.setattr(execute_module.agent_modes, "answer_platform_question", fake_answer)
    monkeypatch.setattr(
        execute_module,
        "process_message",
        _fake_process_message_factory(
            called, _empty_process_message_result(response="FALLBACK_RESPONSE_TEXT")
        ),
    )

    response = client.post(
        "/chat",
        json={"message": "how does verification work?", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert called["process_message"] is True
    assert response.json()["message"] == "FALLBACK_RESPONSE_TEXT"


# ============================================================
# 2. Conversation branch returns greeting directly
# ============================================================

@pytest.mark.parametrize("greeting", ["hi", "hello", "hey"])
def test_conversation_branch_returns_greeting_directly(
    client, auth_headers, monkeypatch, greeting
):
    called = {"process_message": False}

    monkeypatch.setattr(
        execute_module,
        "process_message",
        _fake_process_message_factory(called, _empty_process_message_result()),
    )

    response = client.post(
        "/chat", json={"message": greeting, "mode": "fast"}, headers=auth_headers
    )

    assert response.status_code == 200
    assert called["process_message"] is False
    assert "Homilivo room-search assistant" in response.json()["message"]


def test_conversation_branch_falls_back_for_ambiguous_message(
    client, auth_headers, monkeypatch
):
    """
    "book this room" routes to the conversation branch (it is a
    deterministic UNSUPPORTED_ACTION match, not a greeting), so
    conversation_node must fall back to process_message() rather than
    return the greeting response.
    """
    called = {"process_message": False}

    monkeypatch.setattr(
        execute_module,
        "process_message",
        _fake_process_message_factory(
            called, _empty_process_message_result(response="KNOWN_FALLBACK_RESPONSE")
        ),
    )

    response = client.post(
        "/chat", json={"message": "book this room", "mode": "fast"}, headers=auth_headers
    )

    assert response.status_code == 200
    assert called["process_message"] is True
    assert response.json()["message"] == "KNOWN_FALLBACK_RESPONSE"


# ============================================================
# 3. API-request branch is unchanged
# ============================================================

def test_api_request_branch_unchanged(client, auth_headers, monkeypatch):
    called = {"process_message": False}

    monkeypatch.setattr(
        execute_module,
        "process_message",
        _fake_process_message_factory(
            called,
            _empty_process_message_result(
                response="ROOM_SEARCH_RESPONSE", search_performed=True
            ),
        ),
    )

    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert called["process_message"] is True
    assert response.json()["message"] == "ROOM_SEARCH_RESPONSE"


# ============================================================
# 4. Every branch returns a valid, non-empty response shape
# ============================================================

def test_all_branches_return_valid_response_shape(
    client, auth_headers, mock_homilivo_success, monkeypatch
):
    # "any cheap rooms???" reaches the LLM-fallback branch, which calls
    # process_message() -> fallback_llm_intent() -> classify_intent()
    # (Gemma/Ollama). Mocked unavailable so this stays fast and
    # deterministic in CI, same pattern as
    # tests/test_graph_router.py's ambiguous-message test.
    monkeypatch.setattr(
        "app.experiments.agent_modes.classify_intent",
        lambda query: (_ for _ in ()).throw(GemmaError("no local model available")),
    )

    representative_messages = [
        "how does verification work?",  # FAQ
        "hello",  # Conversation (greeting)
        "find rooms in Jaipur",  # API request
        "any cheap rooms???",  # LLM fallback (ambiguous, no LLM available in CI)
    ]

    for message in representative_messages:
        response = client.post(
            "/chat", json={"message": message, "mode": "fast"}, headers=auth_headers
        )

        assert response.status_code == 200
        body = response.json()

        assert "message" in body and body["message"]
        assert "session_id" in body and body["session_id"]
        assert "rooms" in body
