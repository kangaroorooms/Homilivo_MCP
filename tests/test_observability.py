"""
Tests for app/observability.py (structured logging + in-memory metrics)
and its wiring: GET /metrics, and the guarantee that no request logs a
credential (Authorization header, KANGAROO_TOKEN, demo token).
"""

from __future__ import annotations

import logging

import app.observability as observability_module
from app.observability import (
    _sanitize,
    get_metrics_snapshot,
    record_api_latency,
    record_llm_latency,
    record_rate_limit_event,
    record_turn,
)
import app.rate_limit as rate_limit_module
from app.rate_limit import InMemoryRateLimiter


def _fresh_registry():
    """Give each test an isolated metrics registry so counts don't leak across tests."""
    fresh = observability_module._MetricsRegistry()
    observability_module._metrics = fresh
    return fresh


# ============================================================
# REDACTION
# ============================================================

def test_sanitize_redacts_secret_shaped_keys():
    payload = {
        "Authorization": "Bearer real-secret",
        "kangaroo_token": "abc123",
        "api_key": "xyz",
        "password": "hunter2",
        "city": "Jaipur",
    }

    cleaned = _sanitize(payload)

    assert cleaned["Authorization"] == "[redacted]"
    assert cleaned["kangaroo_token"] == "[redacted]"
    assert cleaned["api_key"] == "[redacted]"
    assert cleaned["password"] == "[redacted]"
    assert cleaned["city"] == "Jaipur"


def test_sanitize_recurses_into_nested_structures():
    payload = {"outer": {"nested_secret_token": "xyz", "fine": "ok"}, "list": [{"password": "x"}]}

    cleaned = _sanitize(payload)

    assert cleaned["outer"]["nested_secret_token"] == "[redacted]"
    assert cleaned["outer"]["fine"] == "ok"
    assert cleaned["list"][0]["password"] == "[redacted]"


# ============================================================
# METRICS REGISTRY
# ============================================================

def test_record_turn_updates_counts_and_status():
    _fresh_registry()

    record_turn(
        request_id="r1", session_id="s1", intent="NEW_SEARCH", operation=None,
        tool="search_rooms", mode="fast", status="success", error_category=None,
        latency_ms=42.0,
    )
    record_turn(
        request_id="r2", session_id="s2", intent=None, operation=None,
        tool=None, mode="fast", status="blocked", error_category="blocked_or_unclassified",
        latency_ms=5.0,
    )

    snapshot = get_metrics_snapshot()
    assert snapshot["request_count"] == 2
    assert snapshot["status_counts"] == {"success": 1, "blocked": 1}
    assert snapshot["turn_latency"]["count"] == 2


def test_record_turn_error_status_increments_error_count():
    _fresh_registry()

    record_turn(
        request_id="r1", session_id="s1", intent=None, operation=None, tool=None,
        mode="fast", status="error", error_category="unhandled_exception", latency_ms=10.0,
    )

    snapshot = get_metrics_snapshot()
    assert snapshot["error_count"] == 1


def test_record_api_latency_tracks_tool_failures():
    _fresh_registry()

    record_api_latency(100.0, success=True)
    record_api_latency(200.0, success=False)

    snapshot = get_metrics_snapshot()
    assert snapshot["tool_failures"] == 1
    assert snapshot["api_latency"]["count"] == 2


def test_record_llm_latency_is_tracked_separately_from_api_latency():
    _fresh_registry()

    record_llm_latency(300.0, "qwen3:4b")

    snapshot = get_metrics_snapshot()
    assert snapshot["llm_latency"]["count"] == 1
    assert snapshot["api_latency"]["count"] == 0


def test_record_rate_limit_event_increments_counter():
    _fresh_registry()

    record_rate_limit_event()
    record_rate_limit_event()

    snapshot = get_metrics_snapshot()
    assert snapshot["rate_limit_events"] == 2


def test_latency_stats_shape_when_empty():
    _fresh_registry()

    snapshot = get_metrics_snapshot()
    assert snapshot["api_latency"] == {"count": 0, "avg_ms": None, "p95_ms": None, "max_ms": None}


def test_latency_p95_is_computed_reasonably():
    _fresh_registry()

    for value in range(1, 101):  # 1..100
        record_api_latency(float(value), success=True)

    snapshot = get_metrics_snapshot()
    stats = snapshot["api_latency"]
    assert stats["count"] == 100
    assert 90 <= stats["p95_ms"] <= 100
    assert stats["max_ms"] == 100.0


# ============================================================
# /metrics ENDPOINT
# ============================================================

def test_metrics_endpoint_returns_expected_shape(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    for key in (
        "active_sessions", "request_count", "error_count", "tool_failures",
        "rate_limit_events", "status_counts", "api_latency", "llm_latency", "turn_latency",
    ):
        assert key in body


def test_metrics_reflect_a_real_chat_request(client, auth_headers, mock_homilivo_success):
    _fresh_registry()

    client.post("/chat", json={"message": "find rooms in Jaipur", "mode": "fast"}, headers=auth_headers)

    body = client.get("/metrics").json()
    assert body["request_count"] >= 1
    assert body["status_counts"].get("success", 0) >= 1


def test_metrics_reflect_a_rate_limit_event(client, auth_headers, mock_homilivo_success, monkeypatch):
    _fresh_registry()

    fresh_limiter = InMemoryRateLimiter()
    monkeypatch.setattr(rate_limit_module, "_backend", fresh_limiter)
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_MAX_REQUESTS", 1)
    monkeypatch.setattr(rate_limit_module, "RATE_LIMIT_WINDOW_SECONDS", 60.0)

    client.post("/chat", json={"message": "hello"}, headers=auth_headers)
    limited = client.post("/chat", json={"message": "hello"}, headers=auth_headers)
    assert limited.status_code == 429

    body = client.get("/metrics").json()
    assert body["rate_limit_events"] >= 1


# ============================================================
# NO SECRETS IN LOGS
# ============================================================

def test_chat_request_never_logs_the_bearer_token(client, auth_headers, mock_homilivo_success, caplog):
    caplog.set_level(logging.DEBUG)

    secret_token = auth_headers["Authorization"].split(" ")[1]

    client.post("/chat", json={"message": "find rooms in Jaipur", "mode": "fast"}, headers=auth_headers)

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_token not in log_text
