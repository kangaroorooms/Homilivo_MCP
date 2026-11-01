"""Stage 10: the rooms search goes through the copied in-process MCP tool. Fakes only: a fake ``urlopen`` for the tool's
HTTP and a fake ``_httpx_get`` for the direct client. No network, no keys."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
import time
import urllib.error

import pytest

from llm_approach import config, observability, rooms_client, turn_context
from llm_approach.actions import TurnBudget, search
from llm_approach.state import LLMConversationState
from llm_approach.tests.conftest import dec
from llm_approach.tools import mcp_search, mcp_timeout

TOKEN = "rooms-token-SECRET-777"


def good_room(n=1, **over):
    room = {"id": f"room-{n}", "title": f"Villa {n}", "city": "Bangalore", "pricePerMonth": 9000 + n,
            "roomType": "1BHK", "idealFor": ["Students"], "location": "Koramangala"}
    room.update(over)
    return room


class Body:
    def __init__(self, payload) -> None:
        self._raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeUrlopen:
    """Stands in for ``urllib.request.urlopen`` inside the copied tool. ``script`` items are consumed one per call:
    an exception is raised, anything else is returned as the body (the last item repeats)."""

    def __init__(self) -> None:
        self.script: list = [{"data": [good_room(1), good_room(2)], "meta": {"total": 2, "hasNextPage": False}}]
        self.requests: list = []
        self.block: threading.Event | None = None
        self.request_ids: list = []

    def __call__(self, request, timeout=None):
        self.requests.append((request, timeout))
        self.request_ids.append(turn_context.request_id())
        if self.block is not None:
            self.block.wait(10)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return Body(item)

    @property
    def calls(self) -> int:
        return len(self.requests)

    @property
    def auth(self) -> list[str]:
        return [r.headers["Authorization"] for r, _ in self.requests]


def http_error(code):
    return urllib.error.HTTPError(mcp_search.API_URL, code, f"error {code}", {}, io.BytesIO(b"Prisma internal detail"))


@pytest.fixture
def api(monkeypatch):
    monkeypatch.delenv(config.OFFLINE_ROOMS_ENV, raising=False)
    monkeypatch.delenv(config.SEARCH_TRANSPORT_ENV, raising=False)
    monkeypatch.setenv("KANGAROO_TOKEN", TOKEN)
    monkeypatch.setattr(mcp_search, "HOMILIVO_API_RETRY_BACKOFF_SECONDS", 0.0)
    fake = FakeUrlopen()
    monkeypatch.setattr("urllib.request.urlopen", fake)

    def no_direct(*a, **k):
        raise AssertionError("the direct httpx client must not be used on the MCP transport")

    monkeypatch.setattr(rooms_client, "_httpx_get", no_direct)
    yield fake
    if fake.block is not None:
        fake.block.set()


def run(search_dict=None, **kw):
    return rooms_client.search_rooms(search_dict or {"city": "Bangalore"}, TurnBudget(), **kw)


class OkResponse:
    status_code = 200

    def __init__(self, payload=None) -> None:
        self.payload = payload if payload is not None else {"data": [good_room(1)]}

    def json(self):
        return self.payload


# ---- the switch -----------------------------------------------------------------------------------


def test_mcp_is_the_default_and_a_bad_value_falls_back_to_it(monkeypatch):
    monkeypatch.delenv(config.SEARCH_TRANSPORT_ENV, raising=False)
    assert config.search_transport() == "mcp"
    for value, expected in (("mcp", "mcp"), ("MCP", "mcp"), ("direct", "direct"), (" Direct ", "direct"),
                            ("http", "mcp"), ("", "mcp")):
        monkeypatch.setenv(config.SEARCH_TRANSPORT_ENV, value)
        assert config.search_transport() == expected


def test_the_constants_match_the_original_server():
    assert mcp_search.MCP_TOOL_TIMEOUT_SECONDS == 15
    assert mcp_search.HOMILIVO_API_MAX_ATTEMPTS == 3
    assert mcp_search.HOMILIVO_API_TIMEOUT_SECONDS == 10


# ---- a normal search ------------------------------------------------------------------------------


def test_a_normal_search_through_mcp_returns_the_same_rooms_as_direct(api, monkeypatch):
    wanted = {"city": "Bangalore", "room_types": ["1BHK"], "max_price": 20000}
    through_mcp = run(wanted, page=2, limit=5)
    assert api.calls == 1
    request, timeout = api.requests[0]
    assert timeout == 10
    assert "roomTypes=1BHK" in request.full_url and "maxPrice=20000" in request.full_url
    assert "page=2" in request.full_url and "limit=5" in request.full_url and "sort=latest" in request.full_url
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"

    payload = {"data": [good_room(1), good_room(2)], "meta": {"total": 2, "hasNextPage": False}}
    monkeypatch.setenv(config.SEARCH_TRANSPORT_ENV, "direct")
    monkeypatch.setattr(rooms_client, "_httpx_get", lambda *a, **k: OkResponse(payload))
    direct = run(wanted, page=2, limit=5)
    assert through_mcp == direct and through_mcp.ok and len(through_mcp.rooms) == 2
    assert (through_mcp.total, through_mcp.has_next, through_mcp.page) == (2, False, 2)


def test_page_one_sends_no_page_and_the_bhk_and_ideal_for_filters_are_sent(api):
    run({"city": "Pune", "bhk": 3, "ideal_for": "Students"})
    url = api.requests[0][0].full_url
    assert "page=" not in url and "roomTypes=3BHK" in url and "idealFor=Students" in url


def test_a_response_without_meta_gives_none(api):
    api.script = [{"data": [good_room(1)]}]
    result = run()
    assert result.ok and (result.total, result.has_next) == (None, None)


def test_the_api_url_still_comes_from_config():
    assert mcp_search.API_URL == config.ROOMS_API_URL


# ---- failures give the same replies as today ------------------------------------------------------


def reply_for(state=None):
    state = state or LLMConversationState()
    return search._run_search({"city": "Bangalore"}, state, TurnBudget(), "new_search"), state


@pytest.mark.parametrize("failure,error_class,status", [
    (http_error(500), "http_500", 500),
    (http_error(503), "http_503", 503),
    (http_error(401), "http_401", 401),
    (http_error(429), "http_429", 429),
    (TimeoutError(), "timeout", None),
    (urllib.error.URLError("refused"), "connection_error", None),
    (b"", "bad_payload", None),
    (b"{not json", "bad_payload", None),
    ([], "bad_payload", None),
    ({"data": None}, "bad_payload", None),
])
def test_every_api_failure_gives_the_unavailable_reply_and_keeps_state(api, failure, error_class, status):
    api.script = [failure]
    previous = LLMConversationState()
    previous.apply_search_result({"city": "Pune", "bhk": 2}, [good_room(7, city="Pune")])
    before = previous.to_dict()
    result, state = reply_for(previous)
    assert result.reply == rooms_client.UNAVAILABLE_MESSAGE and result.rooms is None
    assert result.error_class == error_class
    assert state.to_dict() == before
    raw = run()
    assert not raw.ok and raw.rooms == [] and raw.error_class == error_class and raw.status == status
    assert "prisma" not in (raw.message or "").lower()


def test_empty_results_give_the_same_no_results_reply_as_direct(api, monkeypatch):
    empty = {"data": [], "meta": {"total": 0, "hasNextPage": False}}
    api.script = [empty]
    via_mcp, _ = reply_for()
    monkeypatch.setenv(config.SEARCH_TRANSPORT_ENV, "direct")
    monkeypatch.setattr(rooms_client, "_httpx_get", lambda *a, **k: OkResponse(empty))
    via_direct, _ = reply_for()
    assert via_mcp.reply == via_direct.reply and via_mcp.rooms == via_direct.rooms == []
    assert "couldn't find any rooms" in via_mcp.reply


def test_a_bad_room_is_rejected_after_extraction_and_never_stored(api, monkeypatch, caplog):
    monkeypatch.setattr("llm_approach.switches.room_data_check", lambda: True)
    api.script = [{"data": [good_room(1), good_room(2, pricePerMonth=-5)]}]
    before = observability.get_metrics_snapshot()["room_data_rejections"]
    with caplog.at_level(logging.WARNING):
        result, state = reply_for()
    assert result.reply == rooms_client.UNAVAILABLE_MESSAGE and result.error_class == "invalid_room_data"
    assert state.rooms_on_screen == []
    assert observability.get_metrics_snapshot()["room_data_rejections"] == before + 1
    assert "-5" not in caplog.text and TOKEN not in caplog.text


# ---- one retry layer ------------------------------------------------------------------------------


def test_only_the_tool_retries_a_5xx(api):
    api.script = [http_error(503)]
    slept = []
    result = run(sleep=slept.append, jitter=lambda a, b: 1.0)
    assert api.calls == 3 and slept == []  # the tool's 3 attempts, none from rooms_client
    assert result.error_class == "http_503"


def test_a_timeout_is_tried_three_times_by_the_tool_alone(api):
    api.script = [TimeoutError()]
    slept = []
    result = run(sleep=slept.append)
    assert api.calls == 3 and slept == [] and result.error_class == "timeout"


def test_a_5xx_then_success_recovers_inside_the_tool(api):
    api.script = [http_error(500), {"data": [good_room(1)]}]
    result = run()
    assert result.ok and api.calls == 2


def test_a_4xx_is_not_retried(api):
    api.script = [http_error(400)]
    assert run().error_class == "http_400" and api.calls == 1


# ---- the limits -----------------------------------------------------------------------------------


def test_the_whole_search_limit_cuts_a_slow_call(api, monkeypatch):
    monkeypatch.setenv(config.SEARCH_TIMEOUT_ENV, "0.3")
    api.block = threading.Event()
    before = observability.get_metrics_snapshot()
    started = time.monotonic()
    result = run()
    assert time.monotonic() - started < 3
    assert result.error_class == "search_timeout" and result.rooms == []
    after = observability.get_metrics_snapshot()
    assert after["search_timeouts"] == before["search_timeouts"] + 1
    assert after["tool_failures"] == before["tool_failures"] + 1


def test_the_copied_wrapper_limit_cuts_a_slow_call_and_counts_once(api, monkeypatch):
    monkeypatch.setattr(mcp_timeout, "MCP_TOOL_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setenv(config.SEARCH_TIMEOUT_ENV, "10")
    api.block = threading.Event()
    before = observability.get_metrics_snapshot()["search_timeouts"]
    started = time.monotonic()
    result, _ = reply_for()
    assert time.monotonic() - started < 3
    assert result.reply == rooms_client.UNAVAILABLE_MESSAGE and result.error_class == "search_timeout"
    assert observability.get_metrics_snapshot()["search_timeouts"] == before + 1


def test_a_timeout_keeps_the_session_usable(api, monkeypatch):
    monkeypatch.setenv(config.SEARCH_TIMEOUT_ENV, "0.2")
    api.block = threading.Event()
    assert run().error_class == "search_timeout"
    api.block.set()
    monkeypatch.setenv(config.SEARCH_TIMEOUT_ENV, "5")
    api.block = None
    assert run().ok


def test_latency_is_recorded_once_per_search(api):
    def count():
        return len(observability._metrics.api_latencies_ms)

    before = count()
    run()
    assert count() == before + 1
    api.script = [http_error(500)]
    run()
    assert count() == before + 2


# ---- the token ------------------------------------------------------------------------------------


def test_the_env_token_is_used_when_none_is_bound(api):
    run()
    assert api.auth == [f"Bearer {TOKEN}"]


def test_a_bound_token_wins_and_is_reset(api):
    handle = rooms_client.set_rooms_token("caller-own-token")
    try:
        run()
    finally:
        rooms_client.reset_rooms_token(handle)
    run()
    assert api.auth == ["Bearer caller-own-token", f"Bearer {TOKEN}"]
    assert mcp_search._kangaroo_token_var.get() is None  # the tool's own variable is not left bound


def test_concurrent_requests_keep_their_own_token_through_mcp(api):
    async def one(name):
        handle = rooms_client.set_rooms_token(f"token-{name}")
        try:
            await asyncio.sleep(0.01 * (3 - len(name)))
            await asyncio.to_thread(run)
        finally:
            rooms_client.reset_rooms_token(handle)

    async def main():
        await asyncio.gather(one("a"), one("bb"), one("ccc"))

    asyncio.run(main())
    assert sorted(api.auth) == ["Bearer token-a", "Bearer token-bb", "Bearer token-ccc"]


def test_no_token_anywhere_gives_no_token_and_sends_nothing(api, monkeypatch):
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)
    result = run()
    assert result.error_class == "no_token" and api.calls == 0


def test_the_token_never_reaches_the_result_or_the_logs(api, caplog):
    api.script = [urllib.error.URLError(f"failed Bearer {TOKEN} for Bangalore")]
    with caplog.at_level(logging.DEBUG):
        result = run()
    assert result.error_class == "connection_error"
    assert TOKEN not in caplog.text + repr(result) and "Bangalore" not in (result.message or "")


# ---- context, offline, direct ---------------------------------------------------------------------


def test_the_request_id_reaches_the_tool_thread(api):
    handles = turn_context.begin("req-123", "session-x")
    try:
        run()
    finally:
        turn_context.end(handles)
    assert api.request_ids == ["req-123"]


def test_the_offline_fixture_never_reaches_the_tool(api, monkeypatch):
    monkeypatch.setenv(config.OFFLINE_ROOMS_ENV, "1")
    before = threading.active_count()
    result = run()
    assert result.ok and result.rooms and api.calls == 0
    assert threading.active_count() == before


def test_switching_to_direct_uses_the_httpx_client_and_its_one_retry(api, monkeypatch):
    import httpx

    monkeypatch.setenv(config.SEARCH_TRANSPORT_ENV, "direct")
    calls = []

    def flaky(url, params, headers, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow")
        return OkResponse()

    monkeypatch.setattr(rooms_client, "_httpx_get", flaky)
    slept = []
    result = run(sleep=slept.append, jitter=lambda a, b: 2.5)
    assert result.ok and len(calls) == 2 and slept == [2.5] and api.calls == 0


def test_an_injected_http_get_always_means_direct(api):
    seen = []
    result = run(http_get=lambda *a, **k: seen.append(1) or OkResponse())
    assert result.ok and seen == [1] and api.calls == 0


def test_the_pipeline_uses_the_mcp_transport_by_default(pipe, api):
    reply = pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    assert reply.success and len(reply.rooms) == 2 and api.calls == 1


def test_importing_the_pipeline_does_not_load_the_mcp_package():
    import subprocess
    import sys

    code = ("import sys, llm_approach.pipeline, llm_approach.rooms_client;"
            "print('mcp' in sys.modules or 'llm_approach.tools.mcp_search' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
