"""Production hardening of the server (audit items A3, B1, B2, B3, B7, B15, C and the silent dev defaults).

Fakes only: no Gemini, Groq, rooms, pricing or Redis server call. Redis is fakeredis.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
import uuid

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient

from llm_approach import config, pipeline, reuse, server_app
from llm_approach.security import demo_auth, fail_open
from llm_approach.security.rate_limit import InMemoryRateLimiter, RateLimitResult
from llm_approach.session_store import RedisSessionStore
from llm_approach.state import SessionStore
from llm_approach.tests.conftest import ROOT

from .test_server import stub_turn, ws_turn

ENV_NAMES = (
    "LLM_APPROACH_ENV", "LLM_APPROACH_TRUSTED_PROXY_HOPS", "LLM_APPROACH_DEMO_TOKEN_ENABLED",
    "LLM_APPROACH_DEMO_TOKEN_RATE_LIMIT", "LLM_APPROACH_FALLBACK_PROVIDER", "LLM_APPROACH_SEARCH_TRANSPORT",
    "LLM_APPROACH_OFFLINE_ROOMS", "LLM_APPROACH_LOG_LEVEL", "LLM_APPROACH_GEMINI_COUNTER_PATH",
)
AUTH = {"Authorization": "Bearer some-user-token"}


@pytest.fixture
def base(monkeypatch):
    """A server with no development shortcuts: offline rooms off, development mode, fresh in-memory limits."""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(server_app.SKIP_PRELOAD_ENV, "1")
    monkeypatch.setenv("STREAM_CHUNK_DELAY_SECONDS", "0")
    monkeypatch.setenv("KANGAROO_TOKEN", "fake-rooms-credential")
    limiter = InMemoryRateLimiter()
    monkeypatch.setattr(server_app._rl, "get_rate_limiter", lambda: limiter)
    monkeypatch.setattr(pipeline, "_store", SessionStore())
    monkeypatch.setattr(reuse, "get_is_security_blocked", lambda: (lambda m: (False, "")))
    monkeypatch.setattr(reuse, "get_output_guardrail", lambda: (lambda t: (True, t)))
    pipeline.configure(full_guardrails=True)
    yield limiter
    pipeline.configure(full_guardrails=True)


@pytest.fixture
def production(base, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_ENV", "production")
    return base


def sid() -> str:
    return str(uuid.uuid4())


# ---- 1. A3: the Redis-backed calls run off the event loop --------------------------------------------------


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def test_rate_limit_and_demo_token_calls_run_off_the_event_loop(base, monkeypatch):
    seen: dict[str, list[bool]] = {}

    def spy(name, fn):
        def inner(*args, **kwargs):
            seen.setdefault(name, []).append(_on_event_loop())
            return fn(*args, **kwargs)

        return inner

    for attr in ("check_rate_limit", "check_session_rate_limit"):
        monkeypatch.setattr(server_app._rl, attr, spy(attr, getattr(server_app._rl, attr)))
    monkeypatch.setattr(server_app._demo, "redeem_demo_token", spy("redeem", server_app._demo.redeem_demo_token))
    stub_turn(monkeypatch)
    with TestClient(server_app.create_app()) as client:
        assert client.post("/chat", json={"message": "hi", "session_id": sid()}, headers=AUTH).status_code == 200
        with client.websocket_connect("/ws/chat?token=abc") as ws:
            assert ws_turn(ws, message="hi", session_id=sid())[-1]["event"] == "done"
    assert set(seen) == {"check_rate_limit", "check_session_rate_limit", "redeem"}
    assert all(flag is False for flags in seen.values() for flag in flags)  # none of them ran on the loop


class SlowBackend:
    def hit(self, key, limit, window_seconds, now=None):
        time.sleep(0.3)  # a slow Redis
        return RateLimitResult(allowed=True, remaining=1, retry_after_seconds=0.0)


def test_a_slow_redis_does_not_block_another_coroutine(base, monkeypatch):
    monkeypatch.setattr(server_app._rl, "get_rate_limiter", lambda: SlowBackend())
    stub_turn(monkeypatch)

    async def main():
        gaps: list[float] = []

        async def ticker():
            last = time.perf_counter()
            for _ in range(60):
                await asyncio.sleep(0.02)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        transport = httpx.ASGITransport(app=server_app.create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            task = asyncio.create_task(ticker())
            response = await client.post("/chat", json={"message": "hi", "session_id": sid()}, headers=AUTH)
            await task
        return response, max(gaps)

    response, worst_gap = asyncio.run(main())
    assert response.status_code == 200
    assert worst_gap < 0.2  # two 0.3 s Redis calls happened and the loop never stood still for one of them


def test_fail_open_is_unchanged_when_the_backend_raises(base, monkeypatch):
    class Down:
        def hit(self, *args, **kwargs):
            raise ConnectionError("redis down")

    class DownTokens:
        def issue(self, *a):
            raise ConnectionError

        def get_issued_at(self, *a):
            raise ConnectionError

        def delete(self, *a):
            raise ConnectionError

    wrapped = fail_open.FailOpenRateLimiter(Down())
    tokens = fail_open.FailOpenDemoAuthBackend(DownTokens(), demo_auth.InMemoryDemoAuthBackend())
    monkeypatch.setattr(server_app._rl, "get_rate_limiter", lambda: wrapped)
    monkeypatch.setattr(demo_auth, "get_demo_auth_backend", lambda: tokens)
    stub_turn(monkeypatch)
    with TestClient(server_app.create_app()) as client:
        token = client.get("/chat-ui/demo-token").json()["token"]  # minted into this process's memory
        assert client.post("/chat", json={"message": "hi", "session_id": sid()},
                           headers={"Authorization": f"Bearer {token}"}).status_code == 200
        with client.websocket_connect(f"/ws/chat?token={token}") as ws:
            assert ws_turn(ws, message="hi", session_id=sid())[-1]["event"] == "done"


# ---- 7. B3: no API docs in production -----------------------------------------------------------------------


def test_docs_stay_on_in_development(base):
    with TestClient(server_app.create_app()) as client:
        assert client.get("/docs").status_code == 200 and client.get("/openapi.json").status_code == 200


def test_docs_are_off_in_production(production):
    with TestClient(server_app.create_app()) as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404
        assert client.get("/health").status_code == 200


def test_an_unknown_environment_name_stops_the_start_without_echoing_it(base, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_ENV", "prodd-secretish")
    with pytest.raises(ValueError) as caught:
        server_app.create_app()
    assert "prodd-secretish" not in str(caught.value)


# ---- 6 and 8. startup checks ---------------------------------------------------------------------------------


def test_a_clean_production_start_logs_nothing_new(production, caplog):
    with caplog.at_level(logging.INFO, logger="llm_approach.server_app"), TestClient(server_app.create_app()):
        pass
    assert "startup checks" not in caplog.text and server_app.DEV_MODE_WARNING not in caplog.text
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == [
        "secret check skipped by LLM_APPROACH_SKIP_SECRET_CHECK: do not use this on a real server"]  # the test harness's own


def test_development_mode_logs_one_fixed_warning_at_startup(base, caplog):
    with caplog.at_level(logging.INFO, logger="llm_approach.server_app"), TestClient(server_app.create_app()):
        pass
    warnings = [r.getMessage() for r in caplog.records if r.getMessage() == server_app.DEV_MODE_WARNING]
    assert warnings == ["Running in development mode: set LLM_APPROACH_ENV=production on a real server."]


def test_production_refuses_offline_rooms_without_echoing_the_value(production, monkeypatch, caplog):
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "/secret/place/rooms.json")
    with pytest.raises(RuntimeError) as caught, TestClient(server_app.create_app()):
        pass
    assert "LLM_APPROACH_OFFLINE_ROOMS" in str(caught.value) and "/secret/place" not in str(caught.value) + caplog.text


def test_development_still_allows_offline_rooms(base, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_OFFLINE_ROOMS", "1")
    with TestClient(server_app.create_app()) as client:
        assert client.get("/health").status_code == 200


@pytest.mark.parametrize("provider, missing, setting", [
    ("gemini", "google.genai", None),
    ("groq", "groq", None),
])
def test_production_fails_without_the_provider_package(production, monkeypatch, provider, missing, setting):
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", provider)
    monkeypatch.setattr(server_app, "_package_available", lambda name: name != missing)
    with pytest.raises(RuntimeError) as caught, TestClient(server_app.create_app()):
        pass
    assert missing in str(caught.value)


def test_production_fails_without_the_mcp_package_but_not_on_the_direct_transport(production, monkeypatch):
    monkeypatch.setattr(server_app, "_package_available", lambda name: name != "mcp")
    with pytest.raises(RuntimeError) as caught, TestClient(server_app.create_app()):
        pass
    assert "mcp" in str(caught.value)
    monkeypatch.setenv("LLM_APPROACH_SEARCH_TRANSPORT", "direct")
    with TestClient(server_app.create_app()) as client:
        assert client.get("/health").status_code == 200


def test_production_checks_the_fallback_providers_key_by_name_only(production, monkeypatch, caplog):
    monkeypatch.setenv("LLM_APPROACH_FALLBACK_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as caught, TestClient(server_app.create_app()):
        pass
    assert "GROQ_API_KEY is not set" in str(caught.value)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_value_must_not_appear")
    with caplog.at_level(logging.INFO), TestClient(server_app.create_app()):
        pass
    assert "gsk_value_must_not_appear" not in caplog.text


def test_development_does_not_run_the_package_checks(base, monkeypatch):
    monkeypatch.setattr(server_app, "_package_available", lambda name: False)
    with TestClient(server_app.create_app()) as client:
        assert client.get("/health").status_code == 200


def test_rate_limit_off_logs_one_warning_in_any_mode(base, monkeypatch, caplog):
    monkeypatch.setattr(server_app._rl, "RATE_LIMIT_ENABLED", False)
    with caplog.at_level(logging.WARNING, logger="llm_approach.server_app"), TestClient(server_app.create_app()):
        pass
    assert server_app.RATE_LIMIT_OFF_WARNING in caplog.text


@pytest.fixture
def redis_production(production, monkeypatch):
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, decode_responses=True)
    monkeypatch.setattr(pipeline, "_store", RedisSessionStore(redis_client=client))
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": True, "knowledge_base": True})
    monkeypatch.setattr(server_app, "_package_available", lambda name: True)
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_value_must_not_appear")
    return server


def test_redis_down_at_startup_warns_but_starts_and_ready_says_down(redis_production, caplog):
    redis_production.connected = False
    with caplog.at_level(logging.WARNING, logger="llm_approach.server_app"), TestClient(server_app.create_app()) as client:
        assert server_app.REDIS_DOWN_WARNING in caplog.text
        assert client.get("/health").status_code == 200
        down = client.get("/ready")
        assert down.status_code == 200 and down.json()["redis"] == "down"  # reported, not required
        redis_production.connected = True
        up = client.get("/ready")
        assert up.status_code == 200 and up.json()["redis"] == "up"
    assert "gsk_value_must_not_appear" not in caplog.text + down.text + up.text


def test_redis_up_at_startup_does_not_warn(redis_production, caplog):
    with caplog.at_level(logging.WARNING, logger="llm_approach.server_app"), TestClient(server_app.create_app()):
        pass
    assert server_app.REDIS_DOWN_WARNING not in caplog.text


def test_ready_picks_up_a_knowledge_base_that_loaded_after_a_failed_start(base, monkeypatch):
    monkeypatch.setattr(server_app, "_readiness", {"guardrails": True, "knowledge_base": False})
    monkeypatch.setattr(server_app, "_knowledge", {"embedding_model": False, "chunks_indexed": 0})
    monkeypatch.setenv("LLM_APPROACH_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    with TestClient(server_app.create_app()) as client:
        assert client.get("/ready").status_code == 503
        monkeypatch.setattr(reuse, "get_knowledge_cached_stats",
                            lambda: (lambda: {"documents_loaded": 2, "chunks_indexed": 7, "embedding_model_available": True}))
        body = client.get("/ready")
        assert body.status_code == 200 and body.json()["knowledge"]["chunks_indexed"] == 7


# ---- 10. B1: demo token ------------------------------------------------------------------------------------


def test_demo_token_has_its_own_per_ip_limit_of_10_a_minute(base):
    with TestClient(server_app.create_app()) as client:
        assert [client.get("/chat-ui/demo-token").status_code for _ in range(10)] == [200] * 10
        before = len(demo_auth._tokens)
        refused = client.get("/chat-ui/demo-token")
        assert refused.status_code == 429 and int(refused.headers["Retry-After"]) >= 1
        assert len(demo_auth._tokens) == before  # no token was minted for the refused call


def test_demo_token_limit_is_configurable_and_per_ip(base, monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_DEMO_TOKEN_RATE_LIMIT", "2")
    monkeypatch.setenv("LLM_APPROACH_TRUSTED_PROXY_HOPS", "1")
    with TestClient(server_app.create_app()) as client:
        one, other = {"X-Forwarded-For": "7.7.7.7"}, {"X-Forwarded-For": "8.8.8.8"}
        assert [client.get("/chat-ui/demo-token", headers=one).status_code for _ in range(3)] == [200, 200, 429]
        assert client.get("/chat-ui/demo-token", headers=other).status_code == 200


def test_demo_token_endpoint_can_be_switched_off(base, monkeypatch):
    with TestClient(server_app.create_app()) as client:
        assert client.get("/chat-ui/demo-token").status_code == 200  # default: on
        for value in ("false", "0", "no", "off"):
            monkeypatch.setenv("LLM_APPROACH_DEMO_TOKEN_ENABLED", value)
            assert client.get("/chat-ui/demo-token").status_code == 404
        monkeypatch.setenv("LLM_APPROACH_DEMO_TOKEN_ENABLED", "true")
        assert client.get("/chat-ui/demo-token").status_code == 200


# ---- 11. B2: trusted proxy hops -----------------------------------------------------------------------------


def test_hops_zero_uses_the_tcp_peer_whatever_the_header_says():
    assert server_app.forwarded_client("10.0.0.5", ["1.2.3.4"], 0) == "10.0.0.5"
    assert server_app.forwarded_client(None, [], 0) == "unknown"


def test_hops_pick_the_nth_entry_from_the_right():
    header = ["1.1.1.1, 2.2.2.2, 3.3.3.3"]
    assert server_app.forwarded_client("10.0.0.5", header, 1) == "3.3.3.3"
    assert server_app.forwarded_client("10.0.0.5", header, 2) == "2.2.2.2"
    assert server_app.forwarded_client("10.0.0.5", header, 3) == "1.1.1.1"
    assert server_app.forwarded_client("10.0.0.5", ["1.1.1.1", "2.2.2.2"], 1) == "2.2.2.2"  # two header lines
    assert server_app.forwarded_client("10.0.0.5", ["2001:db8::1"], 1) == "2001:db8::1"


def test_a_forged_leftmost_entry_is_never_used():
    # The client sent "6.6.6.6"; the one trusted proxy appended the address it really saw.
    assert server_app.forwarded_client("10.0.0.5", ["6.6.6.6, 5.5.5.5"], 1) == "5.5.5.5"
    # Two proxies: forged, then the client as proxy 2 saw it, then proxy 1's address.
    assert server_app.forwarded_client("10.0.0.5", ["6.6.6.6, 5.5.5.5, 10.0.0.9"], 2) == "5.5.5.5"
    # A client that forges a long chain cannot move the entry the proxies appended.
    assert server_app.forwarded_client("10.0.0.5", ["1.1.1.1, 2.2.2.2, 3.3.3.3, 5.5.5.5"], 1) == "5.5.5.5"


@pytest.mark.parametrize("header", [[], [""], ["garbage"], ["1.2.3.4:8080"], ["unknown"], ["1.1.1.1,,"], ["999.1.1.1"]])
def test_a_missing_or_malformed_header_falls_back_to_the_peer(header):
    assert server_app.forwarded_client("10.0.0.5", header, 1) == "10.0.0.5"


def test_fewer_entries_than_hops_falls_back_to_the_peer():
    assert server_app.forwarded_client("10.0.0.5", ["1.1.1.1"], 2) == "10.0.0.5"


@pytest.mark.parametrize("value", ["abc", "-1", "1.5"])
def test_a_bad_hops_value_is_an_error_that_stops_the_start(base, monkeypatch, value):
    monkeypatch.setenv("LLM_APPROACH_TRUSTED_PROXY_HOPS", value)
    with pytest.raises(ValueError) as caught:
        config.trusted_proxy_hops()
    assert value not in str(caught.value)  # the bad value is not echoed
    with pytest.raises(ValueError), TestClient(server_app.create_app()):
        pass


class Recorder:
    def __init__(self):
        self.keys: list[str] = []

    def hit(self, key, limit, window_seconds, now=None):
        self.keys.append(key)
        return RateLimitResult(allowed=True, remaining=1, retry_after_seconds=0.0)


def test_the_chat_and_websocket_limits_use_the_forwarded_address_only_when_asked(base, monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(server_app._rl, "get_rate_limiter", lambda: recorder)
    stub_turn(monkeypatch)
    forged = {"X-Forwarded-For": "6.6.6.6, 5.5.5.5"}

    def run():
        recorder.keys.clear()
        with TestClient(server_app.create_app()) as client:
            client.post("/chat", json={"message": "hi"}, headers={**AUTH, **forged})
            with client.websocket_connect("/ws/chat?token=abc", headers=forged) as ws:
                ws_turn(ws, message="hi")
        return {k.removeprefix("newsession:") for k in recorder.keys if not k.startswith("session:")}  # v2: the new-session budget is keyed by the same address

    assert run() == {"testclient"}  # default: the TCP peer, the header is ignored
    monkeypatch.setenv("LLM_APPROACH_TRUSTED_PROXY_HOPS", "1")
    assert run() == {"5.5.5.5"}  # opt in: the entry the proxy appended, not the forged one


# ---- 9. B7: logging ------------------------------------------------------------------------------------------


def _record(name, level):
    return logging.LogRecord(name, level, __file__, 1, "message", (), None)


def test_configure_logging_shows_app_info_and_keeps_other_libraries_at_warning(monkeypatch):
    monkeypatch.delenv("LLM_APPROACH_LOG_LEVEL", raising=False)
    root = logging.Logger("test-root")
    assert server_app.configure_logging(root) is True
    assert root.level == logging.INFO and len(root.handlers) == 1
    gate = root.handlers[0].filter
    assert gate(_record("llm_approach.llm_client", logging.INFO)) and gate(_record("llm_approach", logging.INFO))
    assert not gate(_record("httpx", logging.INFO)) and not gate(_record("llm_approachX", logging.INFO))
    assert gate(_record("httpx", logging.WARNING))
    assert server_app.configure_logging(root) is False and len(root.handlers) == 1  # never twice


def test_configure_logging_leaves_a_host_configured_root_alone():
    root = logging.Logger("test-root")
    root.addHandler(logging.NullHandler())
    assert server_app.configure_logging(root) is False and len(root.handlers) == 1


def test_log_level_comes_from_the_environment_and_a_bad_value_means_info(monkeypatch):
    monkeypatch.setenv("LLM_APPROACH_LOG_LEVEL", "warning")
    root = logging.Logger("a")
    server_app.configure_logging(root)
    assert root.level == logging.WARNING
    monkeypatch.setenv("LLM_APPROACH_LOG_LEVEL", "loud")
    root = logging.Logger("b")
    server_app.configure_logging(root)
    assert root.level == logging.INFO


def test_the_lifespan_configures_logging(base, monkeypatch):
    calls = []
    monkeypatch.setattr(server_app, "configure_logging", lambda *a: calls.append(1))
    with TestClient(server_app.create_app()):
        pass
    assert calls == [1]


# ---- 4. B15: .env is read before config ---------------------------------------------------------------------


def _import_server_with(tmp_path, env_lines, extra_env=None):
    env_file = tmp_path / "test.env"
    env_file.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    names = {"SESSION_MAX_COUNT", "LLM_APPROACH_ROOMS_API_URL", "LLM_APPROACH_ROOMS_AREA_FETCH_LIMIT", "PRICING_API_URL"}
    env = {k: v for k, v in os.environ.items() if k not in names}
    env.update({"LLM_APPROACH_ENV_FILE": str(env_file), "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"})
    env.update(extra_env or {})
    code = ("from llm_approach import server_app, config, pricing_client\n"
            "print(config.SESSION_MAX_COUNT, config.ROOMS_API_URL, config.ROOMS_AREA_FETCH_LIMIT, pricing_client.PRICING_API_URL)")
    done = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-500:]
    return done.stdout.split()


def test_values_set_only_in_the_env_file_reach_config_and_pricing(tmp_path):
    lines = ["SESSION_MAX_COUNT=7", "LLM_APPROACH_ROOMS_API_URL=https://rooms.example.invalid/api",
             "LLM_APPROACH_ROOMS_AREA_FETCH_LIMIT=11", "PRICING_API_URL=https://pricing.example.invalid/p"]
    assert _import_server_with(tmp_path, lines) == [
        "7", "https://rooms.example.invalid/api", "11", "https://pricing.example.invalid/p"]


def test_a_real_environment_variable_still_wins_over_the_env_file(tmp_path):
    out = _import_server_with(tmp_path, ["SESSION_MAX_COUNT=7"], {"SESSION_MAX_COUNT": "3"})
    assert out[0] == "3"
