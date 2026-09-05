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

import app.demo_auth as demo_auth


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


# ============================================================
# INTEGRATION: /chat-ui/demo-token and /chat
# ============================================================

def test_demo_token_endpoint_404s_without_server_credential(client, monkeypatch):
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)

    response = client.get("/chat-ui/demo-token")

    assert response.status_code == 404


def test_demo_token_endpoint_returns_token_with_server_credential(client, monkeypatch):
    monkeypatch.setenv("KANGAROO_TOKEN", "real-server-secret")

    response = client.get("/chat-ui/demo-token")

    assert response.status_code == 200
    body = response.json()
    assert body["token"]
    assert body["token"] != "real-server-secret"


def test_chat_with_demo_token_uses_server_credential_outbound(
    client, monkeypatch, mock_homilivo_token_recorder,
):
    monkeypatch.setenv("KANGAROO_TOKEN", "real-server-secret")
    get_recorded_tokens = mock_homilivo_token_recorder

    demo_token = client.get("/chat-ui/demo-token").json()["token"]

    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers={"Authorization": f"Bearer {demo_token}"},
    )

    assert response.status_code == 200
    # The outbound Homilivo call must carry the REAL server credential,
    # never the opaque demo token itself.
    assert get_recorded_tokens() == ["Bearer real-server-secret"]
    # And the real credential must never be echoed back to the client.
    assert "real-server-secret" not in response.text


def test_chat_with_ordinary_bearer_is_unaffected_by_demo_auth(
    client, monkeypatch, mock_homilivo_token_recorder,
):
    """
    BYOT regression: even with a server credential configured, a normal
    caller-supplied token that isn't a recognized demo token must pass
    through to the outbound call completely unchanged.
    """
    monkeypatch.setenv("KANGAROO_TOKEN", "real-server-secret")
    get_recorded_tokens = mock_homilivo_token_recorder

    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers={"Authorization": "Bearer my-own-homilivo-token"},
    )

    assert response.status_code == 200
    assert get_recorded_tokens() == ["Bearer my-own-homilivo-token"]
