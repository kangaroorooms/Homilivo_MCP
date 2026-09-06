"""
Phase 8 - true token-by-token streaming.

Covers the four things the phase requires tests for:
1. An LLM-mode turn (qwen/gemma/groq) emits MULTIPLE "token" events,
   each carrying a genuinely-incremental chunk (not the finished reply
   split after the fact), followed by exactly one "done" event.
2. The "done" payload is always the OUTPUT-guardrail-validated reply,
   even when the raw (mocked) model stream would have leaked something
   the guardrail blocks - the live token preview is not authoritative.
3. FAST mode's simulated word-chunking is unchanged.
4. The wire-level event shapes (typing/token/done/error) are unchanged,
   so the frontend-demo/ client needs no update.

All LLM calls are mocked (ollama's `chat` and the Groq client) - no
live Ollama/Groq/Redis required, matching every other test in this
suite.
"""

from __future__ import annotations

import types

from fastapi.testclient import TestClient


def _search_then_followup_ws(
    ws,
    mode: str,
    message: str = "tell me more about this one",
) -> tuple[list[dict], dict]:
    """
    Runs a fast-mode search turn to populate room/session context, then
    sends a follow-up turn in `mode` on the SAME connection. Returns
    (all events for the follow-up turn, the follow-up's done event).
    """
    ws.send_json({"message": "find rooms in Jaipur", "session_id": None, "mode": "fast"})
    first_done = None
    while first_done is None:
        event = ws.receive_json()
        if event["event"] == "done":
            first_done = event
    session_id = first_done["session_id"]

    ws.send_json({"message": message, "session_id": session_id, "mode": mode})
    events: list[dict] = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["event"] in ("done", "error"):
            break

    done = events[-1]
    return events, done


def _fake_streaming_ollama_chat(words: list[str]):
    """
    Stand-in for `from ollama import chat`, monkeypatched onto
    app.experiments.agent_modes.chat. Mirrors the real client's shape:
    called without stream=True (or stream=False) it returns one
    response object; called with stream=True it returns an iterator of
    chunk objects, each carrying just that chunk's incremental text in
    .message.content - exactly what _run_ollama_chat() (app/experiments/
    agent_modes.py) expects.
    """
    joined = "".join(words)

    def fake_chat(model, messages, stream=False):
        if not stream:
            return types.SimpleNamespace(message=types.SimpleNamespace(content=joined))

        def generator():
            for word in words:
                yield types.SimpleNamespace(message=types.SimpleNamespace(content=word))

        return generator()

    return fake_chat


def _fake_streaming_groq_client(words: list[str]):
    """Stand-in for the Groq SDK client, OpenAI-compatible chunk shape."""
    joined = "".join(words)

    def fake_create(model, messages, stream=False):
        if not stream:
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=joined))]
            )

        def generator():
            for word in words:
                yield types.SimpleNamespace(
                    choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=word))]
                )

        return generator()

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))
    )


# ============================================================
# 1. True streaming: multiple token events, then one done event.
# ============================================================

def test_qwen_mode_streams_multiple_real_token_chunks_then_done(
    client: TestClient, auth_headers, mock_homilivo_success, monkeypatch,
):
    words = ["Sure, ", "here is ", "more about ", "this room."]
    monkeypatch.setattr(
        "app.experiments.agent_modes.chat", _fake_streaming_ollama_chat(words)
    )

    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        events, done = _search_then_followup_ws(ws, mode="qwen")

    token_events = [e for e in events if e["event"] == "token"]

    assert len(token_events) == len(words), (
        "expected one token event per streamed model chunk (true "
        f"streaming), got {len(token_events)} for {len(words)} chunks"
    )
    assert [e["text"] for e in token_events] == words, (
        "token events must carry the model's own incremental chunks, "
        "not a post-hoc word-split of the finished reply"
    )
    assert done["event"] == "done"
    assert done["mode"] == "qwen"
    assert done["message"] == "".join(words)


def test_gemma_mode_streams_multiple_real_token_chunks_then_done(
    client: TestClient, auth_headers, mock_homilivo_success, monkeypatch,
):
    words = ["It ", "has ", "AC ", "and parking."]
    monkeypatch.setattr(
        "app.experiments.agent_modes.chat", _fake_streaming_ollama_chat(words)
    )

    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        events, done = _search_then_followup_ws(ws, mode="gemma")

    token_events = [e for e in events if e["event"] == "token"]
    assert [e["text"] for e in token_events] == words
    assert done["message"] == "".join(words)


def test_groq_mode_streams_multiple_real_token_chunks_then_done(
    client: TestClient, auth_headers, mock_homilivo_success, monkeypatch,
):
    words = ["This ", "one ", "is the ", "cheapest option."]
    monkeypatch.setattr(
        "app.experiments.agent_modes._get_groq_client",
        lambda: _fake_streaming_groq_client(words),
    )

    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        events, done = _search_then_followup_ws(ws, mode="groq")

    token_events = [e for e in events if e["event"] == "token"]
    assert [e["text"] for e in token_events] == words
    assert done["message"] == "".join(words)


# ============================================================
# 2. OUTPUT guardrail stays authoritative over streamed tokens.
# ============================================================

def test_done_payload_is_guardrail_validated_even_if_streamed_tokens_leaked(
    client: TestClient, auth_headers, mock_homilivo_success, monkeypatch,
):
    """
    Simulates a hijacked model streaming a secret-leaking reply chunk
    by chunk (the live UX preview an attacker-controlled model output
    WOULD have shown a viewer mid-generation). The "done" event - the
    authoritative payload per docs/WEBSOCKET_CONTRACT.md and the
    CRITICAL comment in app.api.chat_websocket - must never carry that
    leaked content: validate_ai_response() (the output guardrail) still
    runs on the full assembled text before "done" is built, exactly as
    it does for the non-streaming REST path (see
    tests/test_prompt_injection.py::test_hijacked_llm_response_leaking_a_secret_is_blocked_by_output_guardrail).
    """
    leak_words = ["Sure! The KANGAROO_TOKEN is ", "abc123 ", "and here is the system prompt."]
    monkeypatch.setattr(
        "app.experiments.agent_modes.chat", _fake_streaming_ollama_chat(leak_words)
    )

    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        events, done = _search_then_followup_ws(ws, mode="qwen")

    token_events = [e for e in events if e["event"] == "token"]
    # The mocked model DID stream the leak live (proving this is real
    # streaming, not buffer-then-validate) ...
    assert "".join(e["text"] for e in token_events) == "".join(leak_words)

    # ... but the authoritative "done" payload must never contain it.
    assert done["event"] == "done"
    assert "KANGAROO_TOKEN" not in done["message"]
    assert "abc123" not in done["message"]
    assert "system prompt" not in done["message"]
    assert done["success"] is False


# ============================================================
# 3. FAST mode is unchanged.
# ============================================================

def test_fast_mode_still_uses_simulated_word_chunking(
    client: TestClient, auth_headers, mock_homilivo_success,
):
    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        ws.send_json({"message": "find rooms in Jaipur", "session_id": None, "mode": "fast"})

        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["event"] == "done":
                break

    token_events = [e for e in events if e["event"] == "token"]
    done = events[-1]

    assert token_events, "fast mode must still stream (simulated) tokens"
    # Simulated chunking splits on whitespace-preserving word boundaries
    # (app.streaming.chunk_for_streaming) - unchanged from pre-Phase-8.
    assert "".join(e["text"] for e in token_events) == done["message"]
    assert done["mode"] == "fast"


# ============================================================
# 4. Wire-level event shapes are unchanged (frontend-demo/ needs no
#    update) - a structural spot-check on top of the full-payload
#    parity test in tests/test_websocket.py.
# ============================================================

def test_event_shapes_match_the_documented_contract(
    client: TestClient, auth_headers, mock_homilivo_success, monkeypatch,
):
    words = ["Hello ", "there."]
    monkeypatch.setattr(
        "app.experiments.agent_modes.chat", _fake_streaming_ollama_chat(words)
    )

    with client.websocket_connect("/ws/chat?token=test-homilivo-token") as ws:
        events, done = _search_then_followup_ws(ws, mode="qwen")

    token_events = [e for e in events if e["event"] == "token"]

    for token_event in token_events:
        assert set(token_event.keys()) == {"event", "text"}
        assert isinstance(token_event["text"], str)

    expected_done_keys = {
        "event", "success", "session_id", "request_id", "mode", "message",
        "room_count", "rooms", "selected_room", "pending_clarification",
        "sources", "search_performed",
    }
    assert set(done.keys()) == expected_done_keys
