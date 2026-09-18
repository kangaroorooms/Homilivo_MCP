# P2-12: External API Failure Handling Tests
# Verifies explicit handling of all HTTP failure
# modes for Homilivo API and Pricing API.
# Never fabricate results after API failure.

import io
import logging
import urllib.error
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app import server
from app.experiments.agent_modes import fetch_subscription_plans
from app.guardrails.api_data_guardrail import validate_room_data


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

    with patch("app.server.urllib.request.urlopen", fake_urlopen):
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
    with caplog.at_level(logging.ERROR, logger="app.server"):
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
    with caplog.at_level(logging.ERROR, logger="app.server"):
        result = _search_with(side_effect=_http_error(418))

    _assert_safe_failure(result)
    assert "Homilivo API error (418)" in caplog.text


def test_homilivo_api_4xx_is_not_retried():
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(1)
        raise _http_error(401)

    with patch("app.server.urllib.request.urlopen", fake_urlopen):
        server.search_rooms(city="Mumbai")

    assert len(calls) == 1


# ------------------------------------------------------------
# search_rooms() - timeout, bad bodies
# ------------------------------------------------------------

def test_homilivo_api_timeout_returns_empty():
    _assert_safe_failure(_search_with(side_effect=TimeoutError()))


def test_homilivo_api_malformed_json_returns_empty(caplog):
    with caplog.at_level(logging.WARNING, logger="app.server"):
        result = _search_with(body="{not valid json" + "x" * 500)

    _assert_safe_failure(result)
    assert "malformed JSON" in caplog.text
    # raw body is truncated to 200 chars in the log
    assert "x" * 200 not in caplog.text


def test_homilivo_api_empty_body_returns_empty(caplog):
    with caplog.at_level(logging.WARNING, logger="app.server"):
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
# fetch_subscription_plans() (httpx)
# ------------------------------------------------------------

def _mock_httpx_client(response=None, get_side_effect=None):
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False

    if get_side_effect is not None:
        client.get.side_effect = get_side_effect
    else:
        client.get.return_value = response

    return patch("httpx.Client", return_value=client)


def _httpx_response(status: int, text: str = "{}") -> httpx.Response:
    return httpx.Response(
        status,
        text=text,
        request=httpx.Request("GET", "https://example.test/plans"),
    )


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 503])
def test_pricing_api_4xx_returns_none(status):
    with _mock_httpx_client(_httpx_response(status)):
        assert fetch_subscription_plans("Mumbai") is None


def test_pricing_api_timeout_returns_none():
    with _mock_httpx_client(get_side_effect=httpx.TimeoutException("t")):
        assert fetch_subscription_plans("Mumbai") is None


def test_pricing_api_malformed_json_returns_none(caplog):
    # P3-18: this is now a logger.warning() rather than a print().
    with caplog.at_level("WARNING"):
        with _mock_httpx_client(_httpx_response(200, "{not json")):
            assert fetch_subscription_plans("Mumbai") is None

    assert "malformed JSON" in caplog.text


def test_pricing_api_empty_plans_returns_empty_list():
    # 200 OK with no plans: [] (API worked), never None (API failed).
    with _mock_httpx_client(_httpx_response(200, '{"data": []}')):
        result = fetch_subscription_plans("Mumbai")

    assert result == []
    assert result is not None


# ------------------------------------------------------------
# api_data_guardrail
# ------------------------------------------------------------

def test_empty_rooms_list_passes_guardrail():
    # Empty is valid - just no results. The function's contract is
    # tuple[bool, str], so the message is a string, not a list.
    assert validate_room_data([]) == (True, "No matching rooms returned.")
