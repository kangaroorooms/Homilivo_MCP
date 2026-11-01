# Copied from tests/test_observability.py at commit 7dbcba3 (copied, not imported).
# Changes: imports now point to llm_approach; the /metrics endpoint and log-leak cases (need the app client) are omitted. Every changed line is marked '# v2:'.
# Not copied from this file: cases that need app.api, TestClient, agent_modes or the MCP server.

"""
Tests for app/observability.py (structured logging + in-memory metrics)
and its wiring: GET /metrics, and the guarantee that no request logs a
credential (Authorization header, KANGAROO_TOKEN, demo token).
"""

from __future__ import annotations

import logging

import pytest

import llm_approach.observability as observability_module  # v2: was: import app.observability as observability_module
from llm_approach.observability import (  # v2: was: from app.observability import (
    get_metrics_snapshot,
    record_api_latency,
    record_llm_latency,
    record_llm_usage,
    record_rate_limit_event,
    record_turn,
)
import llm_approach.security.rate_limit as rate_limit_module  # v2: was: import app.rate_limit as rate_limit_module
from llm_approach.security.rate_limit import InMemoryRateLimiter  # v2: was: from app.rate_limit import InMemoryRateLimiter


def _fresh_registry():
    """Give each test an isolated metrics registry so counts don't leak across tests."""
    fresh = observability_module._MetricsRegistry()
    observability_module._metrics = fresh
    return fresh


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


# ============================================================
# PHASE 8: COST / MODEL-TIER CONTROLS
# ============================================================

def test_record_llm_usage_tracks_calls_and_estimated_tokens_per_mode():
    _fresh_registry()

    record_llm_usage("qwen", "qwen3:4b", prompt_text="x" * 400, completion_text="y" * 400)
    record_llm_usage("qwen", "qwen3:4b", prompt_text="x" * 40, completion_text="y" * 40)
    record_llm_usage("groq", "openai/gpt-oss-20b", prompt_text="x" * 400, completion_text="y" * 400)

    snapshot = get_metrics_snapshot()

    assert snapshot["llm_usage"]["qwen"]["calls"] == 2
    # ~4 chars/token estimate: (400+400)//4 + (40+40)//4 = 200 + 20
    assert snapshot["llm_usage"]["qwen"]["estimated_tokens"] == 220
    assert snapshot["llm_usage"]["groq"]["calls"] == 1
    # fast mode never appears - it makes no model call.
    assert "fast" not in snapshot["llm_usage"]


def test_record_llm_usage_estimated_cost_uses_the_configured_rate(monkeypatch):
    _fresh_registry()
    monkeypatch.setenv("LLM_COST_PER_1K_TOKENS_GROQ", "2.0")

    record_llm_usage("groq", "openai/gpt-oss-20b", prompt_text="x" * 400, completion_text="y" * 400)

    snapshot = get_metrics_snapshot()
    # 200 estimated tokens * $2.0/1000 = $0.40
    assert snapshot["llm_usage"]["groq"]["estimated_cost_usd"] == pytest.approx(0.4)


def test_record_llm_usage_cost_defaults_to_zero_for_locally_hosted_modes():
    _fresh_registry()

    record_llm_usage("qwen", "qwen3:4b", prompt_text="x" * 400, completion_text="y" * 400)
    record_llm_usage("gemma", "gemma3:4b", prompt_text="x" * 400, completion_text="y" * 400)

    snapshot = get_metrics_snapshot()
    assert snapshot["llm_usage"]["qwen"]["estimated_cost_usd"] == 0.0
    assert snapshot["llm_usage"]["gemma"]["estimated_cost_usd"] == 0.0


def test_tier_summary_splits_llm_calls_from_fast_or_deterministic_turns():
    _fresh_registry()

    record_turn(
        request_id="r1", session_id="s1", intent="NEW_SEARCH", operation=None,
        tool="search_rooms", mode="fast", status="success", error_category=None,
        latency_ms=1.0,
    )
    record_turn(
        request_id="r2", session_id="s2", intent="ROOM_FOLLOWUP", operation=None,
        tool=None, mode="qwen", status="success", error_category=None, latency_ms=1.0,
    )
    record_llm_usage("qwen", "qwen3:4b", prompt_text="x", completion_text="y")

    snapshot = get_metrics_snapshot()
    assert snapshot["tier_summary"] == {
        "llm_tier_calls": 1,
        "fast_or_deterministic_tier_turns": 1,
    }


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
