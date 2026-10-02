# P3-15: Production Health and Readiness Tests
# Verifies /health and /ready endpoints behave
# correctly under normal and degraded conditions.

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

import app.api as api


def _boom(*_args, **_kwargs):
    raise RuntimeError("simulated dependency failure")


def _healthy_collection(count: int = 5):
    collection = MagicMock()
    collection.count.return_value = count
    return collection


@pytest.fixture()
def all_deps_ok(monkeypatch):
    """Every required dependency check succeeds."""
    monkeypatch.setattr(api, "_check_embedding_model", lambda: "ok")
    monkeypatch.setattr(api, "_check_knowledge_base", lambda: "ok")

    async def _mcp_ok():
        return "ok"

    monkeypatch.setattr(api, "_check_mcp_server", _mcp_ok)


# ------------------------------------------------------------
# /health (liveness)
# ------------------------------------------------------------

def test_health_endpoint_returns_200(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "homilivo-chatbot"


def test_health_endpoint_always_up(client):
    with patch("app.embeddings.get_shared_model", side_effect=_boom), patch(
        "app.knowledge.store._get_collection", side_effect=_boom
    ):
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_response_has_no_session_count(client):
    # active_sessions does Redis I/O on the redis backend; liveness
    # must stay I/O-free (it is exposed by /metrics instead).
    assert "active_sessions" not in client.get("/health").json()


# ------------------------------------------------------------
# /ready (readiness)
# ------------------------------------------------------------

def test_ready_endpoint_all_ok(client, all_deps_ok):
    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"] == {
        "embedding_model": "ok",
        "knowledge_base": "ok",
        "mcp_server": "ok",
    }
    assert set(body["optional"]) == {"groq_api_key", "redis"}


def test_ready_endpoint_model_failed(client):
    with patch("app.embeddings.get_shared_model", side_effect=_boom):
        response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["embedding_model"] == "failed"


def test_ready_endpoint_knowledge_base_failed(client):
    with patch("app.knowledge.store._get_collection", side_effect=_boom):
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["knowledge_base"] == "failed"


def test_ready_endpoint_empty_knowledge_base_failed(client):
    with patch(
        "app.knowledge.store._get_collection",
        return_value=_healthy_collection(count=0),
    ):
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["knowledge_base"] == "failed"


def test_ready_endpoint_knowledge_base_unavailable_client(client):
    with patch("app.knowledge.store._get_collection", return_value=None):
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["knowledge_base"] == "failed"


def test_ready_endpoint_partial_failure(client, monkeypatch, all_deps_ok):
    monkeypatch.setattr(api, "_check_knowledge_base", lambda: "failed")

    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"] == {
        "embedding_model": "ok",
        "knowledge_base": "failed",
        "mcp_server": "ok",
    }


def test_ready_real_checks_pass_when_dependencies_mocked_healthy(client):
    # Exercises the real _check_* bodies (not the all_deps_ok stubs).
    with patch("app.embeddings.get_shared_model", return_value=object()), patch(
        "app.knowledge.store._get_collection",
        return_value=_healthy_collection(),
    ):
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"] == {
        "embedding_model": "ok",
        "knowledge_base": "ok",
        "mcp_server": "ok",
    }


def test_mcp_tool_missing_is_failed(monkeypatch):
    listing = MagicMock()
    listing.tools = []

    class _FakeClient:
        def __init__(self, *_a, **_k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def list_tools(self):
            return listing

    monkeypatch.setattr(api, "Client", _FakeClient)

    assert asyncio.run(api._check_mcp_server()) == "failed"


def test_mcp_real_server_reports_ok():
    assert asyncio.run(api._check_mcp_server()) == "ok"


def test_ready_mcp_probe_error_is_unknown_and_not_503(client, monkeypatch):
    class _BrokenClient:
        def __init__(self, *_a, **_k):
            raise RuntimeError("probe broke")

    monkeypatch.setattr(api, "Client", _BrokenClient)
    monkeypatch.setattr(api, "_check_embedding_model", lambda: "ok")
    monkeypatch.setattr(api, "_check_knowledge_base", lambda: "ok")

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["mcp_server"] == "unknown"


def test_ready_never_crashes(client):
    with patch("app.embeddings.get_shared_model", side_effect=_boom), patch(
        "app.knowledge.store._get_collection", side_effect=_boom
    ), patch.object(api, "Client", side_effect=_boom), patch.object(
        api, "_check_groq_api_key", side_effect=_boom
    ), patch.object(api, "_check_redis", side_effect=_boom):
        response = client.get("/ready")

    assert response.status_code in (200, 503)
    assert isinstance(response.json(), dict)


def test_ready_requires_no_authentication(client, all_deps_ok):
    assert client.get("/ready").status_code == 200
    assert client.get("/health").status_code == 200


# ------------------------------------------------------------
# Optional dependencies never affect the status code
# ------------------------------------------------------------

def test_missing_groq_key_reported_but_still_ready(client, monkeypatch, all_deps_ok):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["optional"]["groq_api_key"] == "missing"


def test_redis_not_configured_on_memory_backend(client, monkeypatch, all_deps_ok):
    monkeypatch.setenv("STATE_BACKEND", "memory")

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["optional"]["redis"] == "not_configured"


def test_redis_unreachable_is_warning_not_503(client, monkeypatch, all_deps_ok):
    monkeypatch.setenv("STATE_BACKEND", "redis")
    fake_redis = MagicMock()
    fake_redis.ping.side_effect = ConnectionError("redis down")
    monkeypatch.setattr(api.SESSIONS, "_redis", fake_redis, raising=False)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["optional"]["redis"] == "failed"


def test_redis_reachable_reports_ok(client, monkeypatch, all_deps_ok):
    monkeypatch.setenv("STATE_BACKEND", "redis")
    fake_redis = MagicMock()
    fake_redis.ping.return_value = True
    monkeypatch.setattr(api.SESSIONS, "_redis", fake_redis, raising=False)

    response = client.get("/ready")

    assert response.json()["optional"]["redis"] == "ok"


def test_redis_backend_without_client_never_crashes(client, monkeypatch, all_deps_ok):
    monkeypatch.setenv("STATE_BACKEND", "redis")
    monkeypatch.delattr(api.SESSIONS, "_redis", raising=False)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["optional"]["redis"] == "failed"
