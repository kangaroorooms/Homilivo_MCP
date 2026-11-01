"""Fixes from the release audit (limiter settings, ignore file, counter I/O, session floods, degraded notice, production
rate-limit check, placeholder flag). Fakes only: no Gemini, Groq, rooms, pricing or Redis server call."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
import uuid

import fakeredis
import pytest
from fastapi.testclient import TestClient

from llm_approach import config, gemini_limits, knowledge_status, observability, pipeline, server_app
from llm_approach.gemini_limits import GeminiLimiter, RequestCapReached, UsageCounter
from llm_approach.security.rate_limit import RateLimitResult
from llm_approach.tests.conftest import ROOT, dec

from .test_hardening import _ready
from .test_prod_hardening import AUTH, base, production  # noqa: F401 - fixtures
from .test_server import stub_turn, ws_turn
from .test_session_redis import make

LIMIT_ENV_NAMES = (
    "LLM_APPROACH_GEMINI_WINDOW_SECONDS", "LLM_APPROACH_GEMINI_WINDOW_MAX_REQUESTS", "LLM_APPROACH_GEMINI_WINDOW_MAX_TOKENS",
    "LLM_APPROACH_GEMINI_SPACING_S", "LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET", "LLM_APPROACH_GEMINI_DAILY_TOKEN_BUDGET",
)


@pytest.fixture
def clean_limit_env(monkeypatch):
    for name in LIMIT_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


# ---- 1. limiter settings from the environment -----------------------------------------------------------------


def test_limiter_defaults_are_the_free_tier_values(clean_limit_env):
    assert (gemini_limits.window_seconds(), gemini_limits.window_max_requests(), gemini_limits.window_max_tokens()) == (
        60.0, 12, 200_000)
    assert config.daily_request_budget() == 450 and config.spacing_s("gemini") == 5.0
    assert config.daily_token_budget("gemini") is None
    gemini_limits.validate_env()  # nothing set: fine


def test_limit_overrides_are_honoured(clean_limit_env, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_GEMINI_WINDOW_SECONDS", "30")
    monkeypatch.setenv("LLM_APPROACH_GEMINI_WINDOW_MAX_REQUESTS", "2")
    monkeypatch.setenv("LLM_APPROACH_GEMINI_WINDOW_MAX_TOKENS", "500")
    monkeypatch.setenv("LLM_APPROACH_GEMINI_DAILY_REQUEST_BUDGET", "7")
    assert (gemini_limits.window_seconds(), gemini_limits.window_max_requests(), gemini_limits.window_max_tokens()) == (30.0, 2, 500)
    assert config.daily_request_budget() == 7
    limiter = GeminiLimiter(counter=UsageCounter(path=None), clock=lambda: 100.0)
    limiter._window.extend([[90.0, 10.0], [95.0, 10.0]])  # two requests in the window: the third waits for the first
    assert limiter._wait_needed(10) == pytest.approx(20.0)  # 90 + 30 - 100
    monkeypatch.delenv("LLM_APPROACH_GEMINI_WINDOW_MAX_REQUESTS")
    assert limiter._wait_needed(10) == 0.0  # the default of 12 fits


@pytest.mark.parametrize("name", LIMIT_ENV_NAMES)
@pytest.mark.parametrize("bad", ["abc", "0", "-3", "nan", "inf", "1.5x"])
def test_a_bad_limit_value_is_rejected_by_name_without_the_value(clean_limit_env, monkeypatch, name, bad):
    monkeypatch.setenv(name, bad)
    with pytest.raises(ValueError) as caught:
        gemini_limits.validate_env()
    assert name in str(caught.value) and bad not in str(caught.value).replace(name, "")


def test_whole_number_settings_reject_a_fraction(clean_limit_env, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_GEMINI_WINDOW_MAX_REQUESTS", "2.5")
    with pytest.raises(ValueError):
        gemini_limits.validate_env()
    monkeypatch.setenv("LLM_APPROACH_GEMINI_WINDOW_MAX_REQUESTS", "2")
    monkeypatch.setenv("LLM_APPROACH_GEMINI_WINDOW_SECONDS", "0.5")  # a fractional window length is fine
    gemini_limits.validate_env()


def test_the_server_refuses_to_start_on_a_bad_limit_setting(base, clean_limit_env, monkeypatch, caplog):  # noqa: F811
    monkeypatch.setenv("LLM_APPROACH_GEMINI_WINDOW_MAX_TOKENS", "secret-looking-value")
    with pytest.raises(ValueError) as caught, TestClient(server_app.create_app()):
        pass
    assert "LLM_APPROACH_GEMINI_WINDOW_MAX_TOKENS" in str(caught.value)
    assert "secret-looking-value" not in str(caught.value) + caplog.text


# ---- 2. the ignore file ---------------------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
@pytest.mark.parametrize("path", [
    "llm_approach/data/chroma_v2/chroma.sqlite3",
    "llm_approach/data/chroma_v2/doc_fingerprint.txt",
    "llm_approach/.ingest.lock",
    "llm_approach/doc_fingerprint.txt",
    "llm_approach/eval/results/gemini_requests.json",
    "llm_approach/eval/results/gemini_requests.json.123.ab12cd34.tmp",
    "llm_approach/eval/results/gemini_requests.json.123.probe",
    "llm_approach/eval/results/metrics.jsonl",
])
def test_generated_files_are_git_ignored(path):
    done = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True)
    assert done.returncode == 0, path


# ---- 3. counter file I/O off the event loop -------------------------------------------------------------------


def test_a_slow_counter_write_does_not_block_the_event_loop(tmp_path, monkeypatch):
    counter = UsageCounter(path=tmp_path / "counter.json")
    real = counter._write

    def slow(entries):
        time.sleep(0.2)  # a slow disk: add() and set_tokens() each write once
        real(entries)

    monkeypatch.setattr(counter, "_write", slow)
    limiter = GeminiLimiter(counter=counter)

    async def send():
        return object()

    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        await limiter.call(send, budget=10, prompt_estimate=100, tokens_of=lambda r: 50)
        task.cancel()
        return ticks

    assert asyncio.run(main()) >= 15  # the loop kept running during about 0.4 s of writes
    assert counter.used_requests() == 1 and counter.used_tokens() == 50
    assert [p.name for p in tmp_path.iterdir()] == ["counter.json"]  # atomic write left no temp file


def test_two_concurrent_calls_cannot_both_take_the_last_request(tmp_path):
    limiter = GeminiLimiter(counter=UsageCounter(path=tmp_path / "counter.json"))

    async def send():
        return object()

    async def one():
        try:
            await limiter.call(send, budget=1, prompt_estimate=10)
            return "ok"
        except RequestCapReached:
            return "cap"

    async def main():
        return sorted(await asyncio.gather(one(), one()))

    assert asyncio.run(main()) == ["cap", "ok"]
    assert limiter.counter.used_requests() == 1


# ---- 4. new-session flood and /chat/reset ---------------------------------------------------------------------


def _chat(client, **body):
    return client.post("/chat", json={"message": "hi", **body}, headers=AUTH)


def test_a_new_session_id_every_request_hits_the_new_session_limit(base, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LLM_APPROACH_NEW_SESSION_RATE_LIMIT", "3")
    stub_turn(monkeypatch)
    with TestClient(server_app.create_app()) as client:
        codes = [_chat(client, session_id=str(uuid.uuid4())).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]  # the per-session limit (20/min) never saw a repeat


def test_requests_without_or_with_bad_session_ids_count_as_new_sessions(base, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LLM_APPROACH_NEW_SESSION_RATE_LIMIT", "2")
    stub_turn(monkeypatch)
    with TestClient(server_app.create_app()) as client:
        codes = [_chat(client).status_code, _chat(client, session_id="not-a-uuid").status_code,
                 _chat(client, session_id="x@y.com").status_code]
    assert codes == [200, 200, 429]


def test_an_existing_session_is_not_counted_again(base, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LLM_APPROACH_NEW_SESSION_RATE_LIMIT", "1")
    stub_turn(monkeypatch)
    session = str(uuid.uuid4())
    pipeline.get_store().get_or_create(session)  # a turn would have created it
    with TestClient(server_app.create_app()) as client:
        assert [_chat(client, session_id=session).status_code for _ in range(5)] == [200] * 5


def test_the_new_session_limit_fails_open_when_the_limiter_errors(base, monkeypatch):  # noqa: F811
    class Broken:
        def hit(self, key, limit, window_seconds, now=None):
            if key.startswith("newsession:"):
                raise RuntimeError("redis timeout")
            return RateLimitResult(allowed=True, remaining=1, retry_after_seconds=0.0)

    monkeypatch.setattr(server_app._rl, "get_rate_limiter", lambda: Broken())
    stub_turn(monkeypatch)
    with TestClient(server_app.create_app()) as client:
        assert [_chat(client, session_id=str(uuid.uuid4())).status_code for _ in range(3)] == [200, 200, 200]


def test_the_new_session_limit_is_off_with_the_rate_limits(base, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LLM_APPROACH_NEW_SESSION_RATE_LIMIT", "1")
    monkeypatch.setattr(server_app._rl, "RATE_LIMIT_ENABLED", False)
    stub_turn(monkeypatch)
    with TestClient(server_app.create_app()) as client:
        assert [_chat(client, session_id=str(uuid.uuid4())).status_code for _ in range(3)] == [200, 200, 200]


def test_the_websocket_counts_new_sessions_too(base, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LLM_APPROACH_NEW_SESSION_RATE_LIMIT", "2")
    stub_turn(monkeypatch)
    with TestClient(server_app.create_app()) as client, client.websocket_connect("/ws/chat?token=abc") as ws:
        last = [ws_turn(ws, message="hi", session_id=str(uuid.uuid4()))[-1]["event"] for _ in range(3)]
    assert last == ["done", "done", "error"]


def test_chat_reset_has_a_per_ip_limit(base, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LLM_APPROACH_RESET_RATE_LIMIT", "2")
    with TestClient(server_app.create_app()) as client:
        codes = [client.post(f"/chat/reset?session_id={uuid.uuid4()}").status_code for _ in range(4)]
    assert codes == [200, 200, 429, 429]


def test_chat_reset_limit_fails_open(base, monkeypatch):  # noqa: F811
    class Broken:
        def hit(self, *a, **k):
            raise RuntimeError("down")

    monkeypatch.setattr(server_app._rl, "get_rate_limiter", lambda: Broken())
    with TestClient(server_app.create_app()) as client:
        assert [client.post(f"/chat/reset?session_id={uuid.uuid4()}").status_code for _ in range(3)] == [200] * 3


# ---- 5. the degraded-store notice -----------------------------------------------------------------------------


def _count(name):
    return observability.get_metrics_snapshot()["session_store_events"].get(name, 0)


def test_degraded_notice_once_per_degraded_period_and_again_after_recovery(pipe):
    server = fakeredis.FakeServer()
    pipe.store = make(server)
    notice = pipeline.DEGRADED_NOTICE
    assert notice == "(I'm having trouble remembering our conversation right now, so you may need to repeat details.)"
    before = _count("degraded_notice")

    ok = pipe.turn("hello", dec("GENERAL_CHAT"))
    assert notice not in ok.reply and pipe.state is not None  # Redis is up

    server.connected = False
    first = pipe.turn("hello again", dec("GENERAL_CHAT"))
    assert first.reply.endswith(notice) and first.success
    second = pipe.turn("still there?", dec("GENERAL_CHAT"))
    assert notice not in second.reply  # same degraded period: not repeated
    assert _count("degraded_notice") - before == 1

    server.connected = True
    back = pipe.turn("and now?", dec("GENERAL_CHAT"))
    assert notice not in back.reply

    server.connected = False
    again = pipe.turn("down again", dec("GENERAL_CHAT"))
    assert again.reply.endswith(notice)  # a new degraded period
    assert _count("degraded_notice") - before == 2
    snapshot = str(observability.get_metrics_snapshot())
    assert "down again" not in snapshot and "hello again" not in snapshot  # counters only, no user text


def test_no_notice_with_the_memory_store_or_a_healthy_redis(pipe):
    assert pipeline.DEGRADED_NOTICE not in pipe.turn("hello", dec("GENERAL_CHAT")).reply
    pipe.store = make(fakeredis.FakeServer())
    assert pipeline.DEGRADED_NOTICE not in pipe.turn("hello", dec("GENERAL_CHAT")).reply


# ---- 6. production refuses to run with the rate limits off ----------------------------------------------------


def test_production_refuses_rate_limit_enabled_false(production, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server_app._rl, "RATE_LIMIT_ENABLED", False)
    with pytest.raises(RuntimeError) as caught, TestClient(server_app.create_app()):
        pass
    assert "RATE_LIMIT_ENABLED" in str(caught.value)


def test_development_still_starts_with_the_rate_limits_off(base, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server_app._rl, "RATE_LIMIT_ENABLED", False)
    with TestClient(server_app.create_app()) as client:
        assert client.get("/health").status_code == 200


# ---- 7. the placeholder flag ----------------------------------------------------------------------------------


def test_ready_reports_placeholder_true_but_stays_200(monkeypatch):
    knowledge_status.placeholder.cache_clear()
    r = _ready(monkeypatch, {"documents_loaded": 2, "chunks_indexed": 7, "embedding_model_available": True})
    assert r.status_code == 200 and r.json()["knowledge"]["placeholder"] is True  # the shipped FAQ carries the banner


def test_placeholder_is_false_once_no_document_carries_the_banner(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("---\ntitle: A\nverified: true\n---\nApproved text.", encoding="utf-8")
    monkeypatch.setattr(knowledge_status, "DOCUMENTS", tmp_path)
    knowledge_status.placeholder.cache_clear()
    try:
        assert knowledge_status.placeholder() is False
        knowledge_status.placeholder.cache_clear()
        (tmp_path / "b.md").write_text("> **PLACEHOLDER DOCUMENT - NOT OFFICIAL CONTENT.**", encoding="utf-8")
        assert knowledge_status.placeholder() is True
    finally:
        knowledge_status.placeholder.cache_clear()


def test_metrics_report_the_placeholder_flag(base, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LLM_APPROACH_METRICS_TOKEN", "metrics-test-token")
    knowledge_status.placeholder.cache_clear()
    with TestClient(server_app.create_app()) as client:
        body = client.get("/metrics", headers={"Authorization": "Bearer metrics-test-token"}).json()
    assert body["knowledge_placeholder"] is True
