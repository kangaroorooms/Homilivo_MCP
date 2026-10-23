# Copied from tests/test_demo_auth.py at commit 7dbcba3 (copied, not imported).
# Changes: import now points to llm_approach.security.demo_auth; the /chat-ui/demo-token and /chat integration cases are omitted. Every changed line is marked '# v2:'.
# Not copied from this file: cases that need app.api, TestClient, agent_modes or the MCP server.

"""
Tests for app/demo_auth.py and its wiring into /chat-ui/demo-token and
/chat in app/api.py.

Covers the "opt-in server credential" design: the demo page can obtain
an opaque, meaningless-outside-the-server demo token instead of asking
a presenter to paste the real Homilivo credential, while /chat's own
BYOT Authorization contract (see tests/test_api.py's AUTHENTICATION
section) stays completely unchanged for any caller not using one.
"""

from __future__ import annotations

import time

import fakeredis
import pytest

import llm_approach.security.demo_auth as demo_auth  # v2: was: import app.demo_auth as demo_auth


@pytest.fixture(autouse=True)
def _clear_demo_tokens():
    demo_auth._tokens.clear()
    yield
    demo_auth._tokens.clear()


# ============================================================
# UNIT: app/demo_auth.py (memory backend, via the public functions -
# these exercise get_demo_auth_backend()'s default in-memory choice)
# ============================================================

def test_issue_returns_none_without_server_credential(monkeypatch):
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)

    assert demo_auth.server_credential_configured() is False
    assert demo_auth.issue_demo_token() is None


def test_issue_and_redeem_round_trip(monkeypatch):
    monkeypatch.setenv("KANGAROO_TOKEN", "real-server-secret")

    token = demo_auth.issue_demo_token()
    assert token is not None
    assert token != "real-server-secret"  # opaque, not the real value

    redeemed = demo_auth.redeem_demo_token(token)
    assert redeemed == "real-server-secret"


def test_redeem_rejects_unknown_token(monkeypatch):
    monkeypatch.setenv("KANGAROO_TOKEN", "real-server-secret")

    assert demo_auth.redeem_demo_token("not-a-real-demo-token") is None


def test_redeem_rejects_expired_token(monkeypatch):
    monkeypatch.setenv("KANGAROO_TOKEN", "real-server-secret")

    token = demo_auth.issue_demo_token()
    # Backdate the issue time past the TTL.
    demo_auth._tokens[token] = time.time() - demo_auth.DEMO_TOKEN_TTL_SECONDS - 1

    assert demo_auth.redeem_demo_token(token) is None


# ============================================================
# UNIT: DemoAuthBackend contract - parametrized over both backends
# ============================================================
#
# RedisDemoAuthBackend is exercised here against fakeredis (a pure-
# Python, in-memory Redis-protocol server), so its actual code path
# gets the same coverage as InMemoryDemoAuthBackend without requiring
# a live Redis server. These call the DemoAuthBackend directly (not
# the module-level issue_demo_token()/redeem_demo_token(), which are
# always wired to whichever backend get_demo_auth_backend() currently
# returns) so both backends can be tested side by side in one process.

@pytest.fixture(params=["memory", "redis"])
def backend_factory(request):
    if request.param == "redis":
        client = fakeredis.FakeStrictRedis(decode_responses=True)
        return lambda: demo_auth.RedisDemoAuthBackend(redis_client=client)
    return lambda: demo_auth.InMemoryDemoAuthBackend()


def test_backend_issue_and_get_issued_at_round_trip(backend_factory):
    backend = backend_factory()
    now = time.time()

    backend.issue("token-a", now)

    assert backend.get_issued_at("token-a") == pytest.approx(now, abs=1.0)


def test_backend_get_issued_at_returns_none_for_unknown_token(backend_factory):
    backend = backend_factory()

    assert backend.get_issued_at("never-issued") is None


def test_backend_delete_removes_the_token(backend_factory):
    backend = backend_factory()
    now = time.time()

    backend.issue("token-a", now)
    backend.delete("token-a")

    assert backend.get_issued_at("token-a") is None
