"""The copied rooms tool no longer logs any part of a malformed rooms response: only its length and an error word."""

from __future__ import annotations

import logging

import pytest

from llm_approach.tools import mcp_search

BODY_MARK = "OWNERPHONE9876501234"


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        return None


@pytest.mark.parametrize("body, error", [
    (("{not json " + BODY_MARK * 20).encode(), "JSONDecodeError"),
    (('{"data": "' + BODY_MARK * 20 + '"}').encode(), "UnexpectedShape"),
])
def test_a_malformed_response_logs_its_length_and_an_error_word_only(monkeypatch, caplog, body, error):
    monkeypatch.setenv("KANGAROO_TOKEN", "fake-token")
    monkeypatch.setattr("llm_approach.tools.mcp_search.urllib.request.urlopen", lambda request, timeout=None: FakeResponse(body))
    with caplog.at_level(logging.WARNING, logger="llm_approach.tools.mcp_search"):
        result = mcp_search.search_rooms(city="pune")
    assert result["success"] is False and result["rooms"] == []
    assert BODY_MARK not in caplog.text and BODY_MARK not in str(result)
    assert f"length={len(body)} error={error}" in caplog.text
