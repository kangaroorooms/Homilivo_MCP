"""
WebSocket response-streaming helpers.

True token-level streaming straight from an LLM is not implemented for
every mode: FAST-mode answers (and the deterministic structured
answers every mode tries first - see generate_response() in
app/experiments/agent_modes.py) are plain Python strings assembled
before any model is involved, so there is no token stream to relay in
the first place. Unifying that with QWEN/GEMMA/GROQ's real per-token
stream would mean threading a fundamentally different return contract
through generate_response() and every caller of it.

Instead, /ws/chat in app/api.py calls the SAME process_message() REST
uses (no forked/duplicated chatbot logic) and streams the resulting
final text here, chunked into small windows with a short delay between
them - giving a WebSocket client progressive output for every mode
uniformly. This is a presentation-layer concern only.
"""

from __future__ import annotations

import re


def chunk_for_streaming(text: str) -> list[str]:
    """
    Split text into whitespace-preserving word-ish chunks so a client
    can render them progressively and reconstruct the exact original
    string by concatenating every chunk in order.
    """
    if not text:
        return []

    return re.findall(r"\S+\s*", text)
