# Copied from tests/test_p2_12_api_failure_handling.py at commit 609317a (copied, not imported).
# Changes: imports and the patched/log names point at the copied tool. Every changed line is marked '# v2:'.
# Not copied: the fetch_subscription_plans (pricing API) tests, which need agent_modes. Source under test is the copied tool llm_approach/tools/mcp_search.py (from app/server.py); app/ is not imported.

# P2-12: External API Failure Handling Tests
# Verifies explicit handling of all HTTP failure
# modes for Homilivo API and Pricing API.
# Never fabricate results after API failure.

import io
import logging
import urllib.error
from unittest.mock import patch  # v2: was: from unittest.mock import MagicMock, patch

import pytest

from llm_approach.guardrails.api_data_guardrail import validate_room_data  # v2: was: from app.guardrails.api_data_guardrail import validate_room_data
from llm_approach.tools import mcp_search as server  # v2: was: from app import server


# ------------------------------------------------------------
# Helpers - Homilivo API (urllib)
# ------------------------------------------------------------

class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _token_and_no_sleep(monkeypatch):
    monkeypatch.setenv("KANGAROO_TOKEN", "test-token")
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        server.API_URL, code, f"error {code}", {}, io.BytesIO(b"")
    )


def _search_with(side_effect=None, body: str | None = None):
    def fake_urlopen(request, timeout=None):
        if side_effect is not None:
            raise side_effect
        return _FakeResponse(body)

    with patch("llm_approach.tools.mcp_search.urllib.request.urlopen", fake_urlopen):  # v2: was: "app.server.urllib.request.urlopen"
        return server.search_rooms(city="Mumbai")


def _assert_safe_failure(result: dict):
    assert result["success"] is False
    assert result["count"] == 0
    assert result["rooms"] == []


# ------------------------------------------------------------
# search_rooms() - HTTP status codes
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "code, expected_log",
    [
        (400, "bad request (400)"),
        (401, "unauthorized (401)"),
        (403, "forbidden (403)"),
        (404, "endpoint not found (404)"),
        (429, "rate limited (429)"),
        (500, "server error (500)"),
    ],
)
def test_homilivo_api_http_errors_return_empty_and_log(
    code, expected_log, caplog
):
    with caplog.at_level(logging.ERROR, logger="llm_approach.tools.mcp_search"):  # v2: was: logger="app.server"
        result = _search_with(side_effect=_http_error(code))

    _assert_safe_failure(result)
    assert expected_log in caplog.text


def test_homilivo_api_400_returns_empty():
    _assert_safe_failure(_search_with(side_effect=_http_error(400)))


def test_homilivo_api_401_returns_empty():
    _assert_safe_failure(_search_with(side_effect=_http_error(401)))


def test_homilivo_api_403_returns_empty():
    _assert_safe_failure(_search_with(side_effect=_http_error(403)))


def test_homilivo_api_404_returns_empty():
    _assert_safe_failure(_search_with(side_effect=_http_error(404)))


def test_homilivo_api_429_returns_empty():
    _assert_safe_failure(_search_with(side_effect=_http_error(429)))


def test_homilivo_api_500_returns_empty():
    _assert_safe_failure(_search_with(side_effect=_http_error(500)))


def test_homilivo_api_other_status_logged_generically(caplog):
    with caplog.at_level(logging.ERROR, logger="llm_approach.tools.mcp_search"):  # v2: was: logger="app.server"
        result = _search_with(side_effect=_http_error(418))

    _assert_safe_failure(result)
    assert "Homilivo API error (418)" in caplog.text


def test_homilivo_api_4xx_is_not_retried():
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(1)
        raise _http_error(401)

    with patch("llm_approach.tools.mcp_search.urllib.request.urlopen", fake_urlopen):  # v2: was: "app.server.urllib.request.urlopen"
        server.search_rooms(city="Mumbai")

    assert len(calls) == 1


# ------------------------------------------------------------
# search_rooms() - timeout, bad bodies
# ------------------------------------------------------------

def test_homilivo_api_timeout_returns_empty():
    _assert_safe_failure(_search_with(side_effect=TimeoutError()))


def test_homilivo_api_malformed_json_returns_empty(caplog):
    with caplog.at_level(logging.WARNING, logger="llm_approach.tools.mcp_search"):  # v2: was: logger="app.server"
        result = _search_with(body="{not valid json" + "x" * 500)

    _assert_safe_failure(result)
    assert "malformed JSON" in caplog.text
    # raw body is truncated to 200 chars in the log
    assert "x" * 200 not in caplog.text


def test_homilivo_api_empty_body_returns_empty(caplog):
    with caplog.at_level(logging.WARNING, logger="llm_approach.tools.mcp_search"):  # v2: was: logger="app.server"
        result = _search_with(body="")

    _assert_safe_failure(result)
    assert "empty body" in caplog.text


def test_homilivo_api_whitespace_body_returns_empty():
    _assert_safe_failure(_search_with(body="   \n  "))


def test_homilivo_api_non_dict_json_returns_empty():
    _assert_safe_failure(_search_with(body="[]"))


def test_homilivo_api_non_list_data_returns_empty():
    _assert_safe_failure(_search_with(body='{"data": null}'))


def test_homilivo_api_empty_rooms_list_returns_empty():
    # 200 OK but data=[] - API worked, zero results: success=True.
    result = _search_with(body='{"data": []}')

    assert result["success"] is True
    assert result["count"] == 0
    assert result["rooms"] == []


# ------------------------------------------------------------
# api_data_guardrail
# ------------------------------------------------------------

def test_empty_rooms_list_passes_guardrail():
    # Empty is valid - just no results. The function's contract is
    # tuple[bool, str], so the message is a string, not a list.
    assert validate_room_data([]) == (True, "No matching rooms returned.")
