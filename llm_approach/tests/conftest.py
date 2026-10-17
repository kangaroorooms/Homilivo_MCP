"""Shared fixtures. Everything here is offline: no network, no torch."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "llm_approach" / "fixtures" / "rooms_bangalore.json"


@pytest.fixture
def rooms() -> list[dict]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["rooms"] if isinstance(data, dict) else data


class FakeGroq:
    """Stands in for AsyncGroq. Each reply is a str (message content), an
    exception to raise, or an async callable."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return await reply()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


@pytest.fixture
def fake_groq(monkeypatch):
    from llm_approach import llm_client

    def install(replies):
        client = FakeGroq(replies)
        monkeypatch.setattr(llm_client, "_get_client", lambda: client)
        return client

    return install


# ---- Phase 4: pipeline environment --------------------------------------------

import asyncio  # noqa: E402
import uuid  # noqa: E402


def dec(intent, op=None, conf=0.95, ref=None, cmp=None, nc=False, q=None, reply=None, sq=None, **slots):
    """Compact decision dict, as the model would return it."""
    d = {"intent": intent, "confidence": conf}
    if op:
        d["operation"] = op
    s = {k: v for k, v in slots.items() if v is not None}
    if s:
        d["slots"] = s
    if ref is not None:
        d["room_reference"] = ref
    if cmp:
        d["compare_references"] = cmp
    if nc:
        d["needs_clarification"] = True
    if q:
        d["clarifying_question"] = q
    if reply:
        d["reply"] = reply
    if sq:
        d["standalone_query"] = sq
    return d


class PipeEnv:
    """Pipeline with a scripted LLM, fake guardrails, offline rooms and a spy on searches."""

    def __init__(self) -> None:
        from llm_approach.state import SessionStore

        self.script: list = []
        self.llm_calls = 0
        self.searches: list[dict] = []
        self.security_calls: list[str] = []
        self.output_calls: list[str] = []
        self.app_blocks: set[str] = set()  # messages the (fake) app guardrail blocks
        self.output_ok = True
        self.store = SessionStore()
        self.sid = str(uuid.uuid4())

    async def turn_async(self, message: str, llm=None):
        from llm_approach import pipeline

        if llm != "never":  # "never": the LLM must not be asked, so nothing is scripted
            self.script.append(llm)
        return await pipeline.handle_turn(self.sid, message, store=self.store)

    def turn(self, message: str, llm=None):
        return asyncio.run(self.turn_async(message, llm))

    @property
    def state(self):
        return self.store.peek(self.sid)


@pytest.fixture
def pipe(monkeypatch):
    from llm_approach import llm_client, pipeline, reuse, rooms_client
    from llm_approach.schema import parse_decision

    env = PipeEnv()

    async def fake_decide(user_message, state_summary, budget):
        env.llm_calls += 1
        item = env.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = await item()
        return parse_decision(item) if isinstance(item, dict) else None

    def security(message):
        env.security_calls.append(message)
        return (message in env.app_blocks, "refused" if message in env.app_blocks else "")

    def output(text):
        env.output_calls.append(text)
        return env.output_ok, text

    real_search = rooms_client.search_rooms

    def spy_search(search, budget, **kw):
        env.searches.append(dict(search))
        return real_search(search, budget, **kw)

    monkeypatch.setattr(llm_client, "decide", fake_decide)
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: security)
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: output)
    monkeypatch.setattr(reuse, "get_answer_platform_question",
                        lambda: (lambda q: {"grounded": True, "message": "Homilivo earns from owner subscriptions.", "sources": []}))
    monkeypatch.setattr(rooms_client, "search_rooms", spy_search)
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)
    monkeypatch.delenv("LLM_APPROACH_METRICS", raising=False)
    pipeline.configure(full_guardrails=True)
    yield env
    pipeline.configure(full_guardrails=True)
