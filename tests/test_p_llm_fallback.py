# Tests that Gemma/Qwen LLM failures degrade
# gracefully to FAST mode instead of raising
# a 500 error - matching Groq's existing behavior.
#
# NOTE ON FIX LOCATION: the fallback try/except lives at the
# generate_response() call sites for MODE_GEMMA/MODE_QWEN (mirroring
# the existing MODE_GROQ call site), not inside
# generate_gemma_response()/generate_qwen_response() themselves -
# those functions have no try/except, same as generate_groq_response().
# So these tests exercise the fallback through generate_response(),
# which is the only place the fallback actually happens.

from __future__ import annotations

import asyncio

from app.experiments.agent_modes import (
    MODE_GEMMA,
    MODE_GROQ,
    MODE_QWEN,
    generate_response,
)

ROOMS = [
    {
        "id": "room-1",
        "title": "Budget Single Room",
        "city": "Bangalore",
        "pricePerMonth": 7000,
        "roomType": "Single",
        "idealFor": ["Students"],
        "location": "HSR Layout, Bangalore",
        "availabilityStatus": "Available",
        "rating": 4.2,
        "reviewsCount": 10,
    },
]


def _call_generate_response(mode: str) -> str:
    return generate_response(
        mode=mode,
        user_query="tell me about this room",
        rooms=ROOMS,
        search_request={},
        conversation_history=[],
        is_followup=False,
        selected_room=None,
        selected_room_id=None,
    )


def test_gemma_failure_falls_back_gracefully(monkeypatch):
    def _raise(*args, **kwargs):
        raise Exception("Ollama crashed")

    monkeypatch.setattr("app.experiments.agent_modes.chat", _raise)

    result = _call_generate_response(MODE_GEMMA)

    assert isinstance(result, str)
    assert result != ""


def test_qwen_failure_falls_back_gracefully(monkeypatch):
    def _raise(*args, **kwargs):
        raise Exception("Ollama crashed")

    monkeypatch.setattr("app.experiments.agent_modes.chat", _raise)

    result = _call_generate_response(MODE_QWEN)

    assert isinstance(result, str)
    assert result != ""


def test_gemma_timeout_falls_back_gracefully(monkeypatch):
    def _raise(*args, **kwargs):
        raise asyncio.TimeoutError("Ollama timed out")

    monkeypatch.setattr("app.experiments.agent_modes.chat", _raise)

    result = _call_generate_response(MODE_GEMMA)

    assert isinstance(result, str)
    assert result != ""


def test_groq_fallback_still_works(monkeypatch):
    def _raise():
        raise Exception("Groq API unreachable")

    monkeypatch.setattr(
        "app.experiments.agent_modes._get_groq_client", _raise
    )

    result = _call_generate_response(MODE_GROQ)

    assert isinstance(result, str)
    assert result != ""
