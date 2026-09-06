"""
WebSocket response-streaming helpers.

Two streaming strategies coexist here, chosen per turn by app/api.py's
chat_websocket handler based on whether anything was actually relayed
through a LiveTokenRelay:

1. TRUE token streaming (Phase 8), for qwen/gemma/groq turns that
   reach a real model call. generate_qwen_response() /
   generate_gemma_response() / generate_groq_response() (see
   app/experiments/agent_modes.py) call the provider's streaming API
   directly and invoke an `on_token(text)` callback per chunk AS THE
   MODEL GENERATES IT. LiveTokenRelay below bridges that synchronous,
   possibly-off-thread callback into an async iterator app/api.py can
   `async for` over, sending each chunk to the client the moment it
   arrives.

2. Simulated chunking (unchanged from pre-Phase-8), for everything
   that never calls on_token: FAST mode, and the deterministic
   structured answers every mode tries first (comparison/fact/set-wide
   short-circuits in generate_response()) - these are plain Python
   strings assembled before any model is involved, so there is no
   token stream to relay in the first place. chunk_for_streaming()
   splits the finished string into small windows so the client still
   sees progressive output.

Either way, the WS wire contract (docs/WEBSOCKET_CONTRACT.md) is
unchanged: a run of `{"event":"token",...}` frames followed by one
`{"event":"done",...}` frame whose `message` is the OUTPUT-guardrail-
validated final text - see app/api.py's chat_websocket for exactly how
the two strategies are chosen and why the "done" payload stays
authoritative even when live tokens were relayed before validation ran.
"""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator, Callable

# Type of the callback generate_qwen/gemma/groq_response() call per
# streamed chunk. Called synchronously, and not necessarily from the
# event-loop thread (see LiveTokenRelay.on_token below) - it must never
# block or raise.
TokenSink = Callable[[str], None]


def chunk_for_streaming(text: str) -> list[str]:
    """
    Split text into whitespace-preserving word-ish chunks so a client
    can render them progressively and reconstruct the exact original
    string by concatenating every chunk in order.
    """
    if not text:
        return []

    return re.findall(r"\S+\s*", text)


class LiveTokenRelay:
    """
    Bridges a synchronous, possibly off-event-loop-thread `on_token`
    callback to an async consumer.

    generate_qwen_response() / generate_gemma_response() /
    generate_groq_response() call the ollama/Groq SDKs synchronously,
    so a live turn runs them via `asyncio.to_thread` (see
    agent_modes._generate_response_maybe_streamed) to keep the event
    loop free to send WebSocket frames while generation is still in
    progress. `on_token` is therefore called from that worker thread;
    `call_soon_threadsafe` is the correct way to hand a value back to
    the event loop from another thread. When no threading is involved
    (e.g. a direct unit test calling `on_token` from the main thread),
    `call_soon_threadsafe` still works - it is safe to use whether or
    not the caller is actually a different thread.

    Usage:
        relay = LiveTokenRelay()
        task = asyncio.create_task(some_coro(on_token=relay.on_token))
        task.add_done_callback(lambda _t: relay.close())
        async for chunk in relay:
            await websocket.send_json({"event": "token", "text": chunk})
        result = await task  # re-raises if some_coro() failed
    """

    _DONE = object()

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._loop = asyncio.get_running_loop()

    def on_token(self, text: str) -> None:
        if not text:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, text)

    def close(self) -> None:
        """Signal that no more tokens are coming (call exactly once)."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, self._DONE)

    async def __aiter__(self) -> AsyncIterator[str]:
        while True:
            item = await self._queue.get()
            if item is self._DONE:
                return
            yield item
