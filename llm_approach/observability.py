# Copied from app/observability.py at commit 7dbcba3 (copied, not imported).
# Changes: none
# Drift is checked by llm_approach/tests/test_copy_drift.py.
# ==== end of v2 provenance header ====
"""
Lightweight structured logging + in-memory metrics.

Not a monitoring stack. logging.getLogger("chatbot") writes one
structured line per event (safe metadata only - fixed scalar fields),
and a small in-memory registry backs GET /metrics. This is enough for
a demo/small deployment; swapping the registry for
OpenTelemetry/Prometheus/a real log shipper later would only mean
changing this one module, since every call site funnels through
record_turn() / record_api_latency() / record_llm_latency() /
record_rate_limit_event() rather than logging directly.

NEVER pass a token, Authorization header, password, or other secret
into any function here. Every function takes fixed scalar fields only
(no free-form payloads), so safety comes from what callers pass; see
tests/test_p3_18_observability.py for the no-credentials-in-logs checks.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from typing import Any

logger = logging.getLogger("chatbot")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _latency_stats(values: "deque[float]") -> dict[str, Any]:
    if not values:
        return {"count": 0, "avg_ms": None, "p95_ms": None, "max_ms": None}

    ordered = sorted(values)
    count = len(ordered)
    p95_index = min(count - 1, int(round(0.95 * (count - 1))))

    return {
        "count": count,
        "avg_ms": round(sum(ordered) / count, 2),
        "p95_ms": round(ordered[p95_index], 2),
        "max_ms": round(ordered[-1], 2),
    }


class _MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.request_count = 0
        self.error_count = 0
        self.tool_failures = 0
        self.rate_limit_events = 0
        self.status_counts: dict[str, int] = {}
        self.api_latencies_ms: deque[float] = deque(maxlen=500)
        self.llm_latencies_ms: deque[float] = deque(maxlen=500)
        self.turn_latencies_ms: deque[float] = deque(maxlen=500)
        # Phase 8 cost/model-tier controls: which tier (free deterministic
        # fast-path vs. a real model call) actually served each turn, plus
        # a rough token/cost estimate per mode - see record_llm_usage().
        self.llm_usage_by_mode: dict[str, dict[str, float]] = {}

    def record_turn(self, status: str, latency_ms: float) -> None:
        with self._lock:
            self.request_count += 1
            self.status_counts[status] = self.status_counts.get(status, 0) + 1
            self.turn_latencies_ms.append(latency_ms)
            if status == "error":
                self.error_count += 1

    def record_api_latency(self, latency_ms: float, success: bool) -> None:
        with self._lock:
            self.api_latencies_ms.append(latency_ms)
            if not success:
                self.tool_failures += 1

    def record_llm_latency(self, latency_ms: float) -> None:
        with self._lock:
            self.llm_latencies_ms.append(latency_ms)

    def record_rate_limit_event(self) -> None:
        with self._lock:
            self.rate_limit_events += 1

    def record_llm_usage(self, mode: str, estimated_tokens: int) -> None:
        with self._lock:
            bucket = self.llm_usage_by_mode.setdefault(
                mode, {"calls": 0, "estimated_tokens": 0, "estimated_cost_usd": 0.0}
            )
            bucket["calls"] += 1
            bucket["estimated_tokens"] += estimated_tokens
            bucket["estimated_cost_usd"] += (
                estimated_tokens / 1000.0
            ) * _cost_per_1k_tokens(mode)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            llm_calls = sum(
                int(bucket["calls"]) for bucket in self.llm_usage_by_mode.values()
            )
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "tool_failures": self.tool_failures,
                "rate_limit_events": self.rate_limit_events,
                "status_counts": dict(self.status_counts),
                "api_latency": _latency_stats(self.api_latencies_ms),
                "llm_latency": _latency_stats(self.llm_latencies_ms),
                "turn_latency": _latency_stats(self.turn_latencies_ms),
                "llm_usage": {
                    mode: {
                        "calls": int(bucket["calls"]),
                        "estimated_tokens": int(bucket["estimated_tokens"]),
                        "estimated_cost_usd": round(bucket["estimated_cost_usd"], 6),
                    }
                    for mode, bucket in self.llm_usage_by_mode.items()
                },
                "tier_summary": {
                    "llm_tier_calls": llm_calls,
                    "fast_or_deterministic_tier_turns": max(
                        0, self.request_count - llm_calls
                    ),
                },
            }


_metrics = _MetricsRegistry()

# Per-mode $/1K-token rate used only to produce an ESTIMATED cost figure
# at /metrics - never a stand-in for a provider's real invoice. Locally
# hosted modes (qwen/gemma via Ollama) have no real per-token cost, so
# they default to 0; a hosted mode's real price is an external contract
# this project must not guess at (see docs/Homilivo_Project_Plan.md
# guardrail #10), so it is only ever read from an operator-supplied env
# var, never hardcoded.
_LLM_COST_PER_1K_TOKENS_ENV = {
    "qwen": "LLM_COST_PER_1K_TOKENS_QWEN",
    "gemma": "LLM_COST_PER_1K_TOKENS_GEMMA",
    "groq": "LLM_COST_PER_1K_TOKENS_GROQ",
}


def _cost_per_1k_tokens(mode: str) -> float:
    env_name = _LLM_COST_PER_1K_TOKENS_ENV.get(mode)
    if not env_name:
        return 0.0
    try:
        return float(os.getenv(env_name, "0") or "0")
    except ValueError:
        return 0.0


def _estimate_tokens(text: str) -> int:
    """
    ~4 characters/token is the standard rough heuristic for English
    text. Used instead of a provider's reported usage field because
    that field's shape differs per SDK/mode and is simply absent on the
    mocked clients this project's tests use - an estimate computed the
    same way for every mode is honest about being an estimate, whereas
    reading a provider-specific field would look precise while still
    being an external contract this project has not verified (guardrail
    #10: don't guess external contracts).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def record_turn(
    *,
    request_id: str,
    session_id: str,
    intent: str | None,
    operation: str | None,
    tool: str | None,
    mode: str,
    status: str,
    error_category: str | None,
    latency_ms: float,
) -> None:
    """
    One call per chat turn (REST or WebSocket). `status` is one of
    "success" | "blocked" | "error". Arguments are metadata only - no
    user message text or credentials are ever logged here.
    """
    _metrics.record_turn(status, latency_ms)

    logger.info(
        "turn request_id=%s session_id=%s intent=%s operation=%s tool=%s "
        "mode=%s status=%s error_category=%s latency_ms=%.1f",
        request_id, session_id, intent, operation, tool,
        mode, status, error_category, latency_ms,
    )


def record_api_latency(latency_ms: float, success: bool) -> None:
    """One call per outbound Homilivo API attempt sequence (app/server.py)."""
    _metrics.record_api_latency(latency_ms, success)


def record_llm_latency(latency_ms: float, model: str) -> None:
    """One call per LLM generation call (qwen/gemma/groq)."""
    _metrics.record_llm_latency(latency_ms)
    logger.info("llm_call model=%s latency_ms=%.1f", model, latency_ms)


def record_rate_limit_event() -> None:
    """One call whenever a request or WebSocket connection is rate-limited."""
    _metrics.record_rate_limit_event()
    logger.info("rate_limit_exceeded")


def record_llm_usage(mode: str, model: str, prompt_text: str, completion_text: str) -> None:
    """
    One call per actual qwen/gemma/groq generation (never for fast mode,
    which makes no model call) - see generate_qwen_response() /
    generate_gemma_response() / generate_groq_response() in
    app/experiments/agent_modes.py. Feeds /metrics's "llm_usage" and
    "tier_summary" - the Phase 8 cost/model-tier controls: how many
    turns actually needed the LLM tier vs. the free deterministic/fast
    tier, and a rough token/cost estimate per mode. See
    _estimate_tokens()/_cost_per_1k_tokens() above for why these are
    estimates, not billing figures.
    """
    estimated_tokens = _estimate_tokens(prompt_text) + _estimate_tokens(completion_text)
    _metrics.record_llm_usage(mode, estimated_tokens)
    logger.info(
        "llm_usage mode=%s model=%s estimated_tokens=%d",
        mode, model, estimated_tokens,
    )


def get_metrics_snapshot() -> dict[str, Any]:
    return _metrics.snapshot()
