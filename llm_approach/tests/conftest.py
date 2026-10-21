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


@pytest.fixture(autouse=True)
def _provider_env(monkeypatch):
    """The developer's own provider settings must not change a test: start from the defaults (Gemini)."""
    for name in ("LLM_APPROACH_PROVIDER", "LLM_APPROACH_GEMINI_MODEL", "LLM_APPROACH_GEMINI_SPACING_S",
                 "LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET", "LLM_APPROACH_GEMINI_THINKING",
                 "LLM_APPROACH_GEMINI_MAX_OUTPUT_TOKENS", "LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET"):
        monkeypatch.delenv(name, raising=False)


class FakeClock:
    """Monotonic clock whose ``sleep`` just moves time forward: limiter waits cost no real time."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def _gemini_limiter(tmp_path, monkeypatch):
    """Every test gets its own Gemini limiter with a temp counter file and a fake clock: no test touches the
    real request counter or waits for real."""
    from llm_approach import gemini_limits

    clock = FakeClock()
    limiter = gemini_limits.GeminiLimiter(
        gemini_limits.DailyCounter(tmp_path / "gemini_requests.json"), clock=clock, sleep=clock.sleep)
    gemini_limits.set_limiter(limiter)
    limiter.fake_clock = clock
    from llm_approach.eval import run_eval

    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path / "eval_results")  # run state and results of main()
    yield limiter
    gemini_limits.set_limiter(None)


@pytest.fixture
def fake_groq(monkeypatch):
    """Selects the Groq provider and installs a fake AsyncGroq."""
    from llm_approach import providers

    def install(replies):
        client = FakeGroq(replies)
        monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
        monkeypatch.setattr(providers, "_get_groq_client", lambda: client)
        return client

    return install


class FakeGemini:
    """Stands in for google.genai.Client: ``client.aio.models.generate_content(model=, contents=, config=)``.
    Each reply is a str (the reply text), None (an empty reply: no text), an exception to raise, or an
    async callable."""

    def __init__(self, replies, usage=None):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.usage = usage if usage is not None else SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20, cached_content_token_count=None,
            thoughts_token_count=None,
        )
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            return await reply()
        return SimpleNamespace(text=reply, usage_metadata=self.usage)


@pytest.fixture
def fake_gemini(monkeypatch):
    """Selects the Gemini provider (the default) and installs a fake google-genai client."""
    from llm_approach import providers

    def install(replies, usage=None):
        client = FakeGemini(replies, usage)
        monkeypatch.setenv("LLM_APPROACH_PROVIDER", "gemini")
        monkeypatch.setattr(providers, "_get_gemini_client", lambda: client)
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
        self.answers: list = []  # scripted replies of the conversation answerer (second call)
        self.answer_calls = 0
        self.answer_messages: list = []  # the messages each answer call was given
        self.searches: list[dict] = []
        self.security_calls: list[str] = []
        self.output_calls: list[str] = []
        self.app_blocks: set[str] = set()  # messages the (fake) app guardrail blocks
        self.output_ok = True
        self.store = SessionStore()
        self.sid = str(uuid.uuid4())

    async def turn_async(self, message: str, llm=None, answer=None):
        from llm_approach import pipeline

        if llm != "never":  # "never": the LLM must not be asked, so nothing is scripted
            self.script.append(llm)
        if answer is not None:
            self.answers.append(answer)
        return await pipeline.handle_turn(self.sid, message, store=self.store)

    def turn(self, message: str, llm=None, answer=None):
        return asyncio.run(self.turn_async(message, llm, answer))

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

    async def fake_generate_json(messages, budget):
        """The answerer's call: a str is the raw model text, a dict is dumped to JSON, "rate_limit" / "timeout"
        set the budget's error class like the real client, an exception is raised."""
        env.answer_calls += 1
        env.answer_messages.append(messages)
        if not budget.try_consume():
            return None
        budget.record_latency(2.0)
        item = env.answers.pop(0)
        if isinstance(item, BaseException):
            raise item
        if item in ("rate_limit", "timeout"):
            budget.error_class = "RateLimitError" if item == "rate_limit" else "TimeoutError"
            return None
        budget.tokens_in += 300
        budget.tokens_out += 40
        return json.dumps(item) if isinstance(item, dict) else item

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
    monkeypatch.setattr(llm_client, "generate_json", fake_generate_json)
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
