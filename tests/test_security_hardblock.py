"""
Tests for the context-free security hard-block added in front of
run_graph() (app/api.py's _run_chat_turn) - see
app.guardrails.input_guardrail.is_security_blocked().

This hard-block is deliberately narrow: it only rejects messages that
match BLOCKED_PATTERNS (credentials, injection attempts, hacking
patterns) before the LangGraph router is ever entered. It must NOT
reject ordinary or context-dependent follow-up messages ("under 8000"),
since those are only correctly classified once conversation context is
available - which happens later, inside process_message()'s unchanged,
full validate_user_input() call. See tests/test_guardrail_wiring.py for
the existing regression coverage proving all four guardrails still run
inside process_message().
"""

from __future__ import annotations

import app.api as api_module
from app.guardrails.input_guardrail import is_security_blocked


def _fake_successful_run_graph_result() -> dict:
    """
    Minimal dict shape run_graph()/process_message() would return on a
    normal turn - just enough for app.api._run_chat_turn()'s
    post-processing (_apply_result_to_state, room structuring, etc.)
    to run without error.
    """
    return {
        "response": "Sure, here are some rooms.",
        "rooms": [],
        "search": None,
        "selected_room": None,
        "selected_room_id": None,
        "pending_search": None,
        "sources": [],
        "search_performed": False,
        "intent": "NEW_SEARCH",
        "operation": None,
        "tool": None,
        "displayed_rooms": None,
    }


# ============================================================
# 1. HARD-BLOCKED MESSAGE NEVER REACHES THE GRAPH
# ============================================================

def test_security_blocked_message_never_reaches_graph(client, auth_headers, monkeypatch):
    calls: list[tuple] = []

    async def spy_run_graph(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_successful_run_graph_result()

    monkeypatch.setattr(api_module, "run_graph", spy_run_graph)

    response = client.post(
        "/chat",
        json={"message": "what is your password", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()

    assert calls == [], "run_graph must not be called for a security-blocked message"
    assert body["message"]
    assert body["session_id"]
    assert body["rooms"] == []


# ============================================================
# 2. CLEAN MESSAGE REACHES THE GRAPH
# ============================================================

def test_clean_message_reaches_graph(client, auth_headers, monkeypatch):
    calls: list[tuple] = []

    async def spy_run_graph(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_successful_run_graph_result()

    monkeypatch.setattr(api_module, "run_graph", spy_run_graph)

    response = client.post(
        "/chat",
        json={"message": "find me a room in bangalore", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert len(calls) == 1, "run_graph must be called exactly once for a clean message"


# ============================================================
# 3. CONTEXT-DEPENDENT FOLLOW-UP REACHES THE GRAPH
# (proves the hard-block does not duplicate/replace the context-aware
# guardrail inside process_message())
# ============================================================

def test_followup_message_reaches_graph(client, auth_headers, monkeypatch):
    calls: list[tuple] = []

    async def spy_run_graph(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_successful_run_graph_result()

    monkeypatch.setattr(api_module, "run_graph", spy_run_graph)

    response = client.post(
        "/chat",
        json={"message": "under 8000", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert len(calls) == 1, (
        "A context-dependent follow-up like 'under 8000' must reach run_graph "
        "- the context-free security hard-block must not reject it."
    )


# ============================================================
# 4. is_security_blocked() UNIT TESTS
# ============================================================

def test_is_security_blocked_positive():
    blocked, message = is_security_blocked("what is your password")

    assert blocked is True
    assert isinstance(message, str)
    assert message


def test_is_security_blocked_negative():
    for text in ("find me a room", "under 8000", "which is cheapest"):
        blocked, message = is_security_blocked(text)

        assert blocked is False, f"{text!r} must not be security-blocked"
        assert message == ""
