"""
Lightweight structured logging + in-memory metrics.

Not a monitoring stack. logging.getLogger("chatbot") writes one
structured line per event (safe metadata only - see _sanitize() below),
and a small in-memory registry backs GET /metrics. This is enough for
a demo/small deployment; swapping the registry for
OpenTelemetry/Prometheus/a real log shipper later would only mean
changing this one module, since every call site funnels through
record_turn() / record_api_latency() / record_llm_latency() /
record_rate_limit_event() rather than logging directly.

NEVER pass a token, Authorization header, password, or other secret
into any function here - _sanitize() redacts anything whose key looks
secret-shaped, but callers must still not log raw credentials in the
first place (redaction is defense in depth, not a substitute for care).
"""

from __future__ import annotations

import logging
import re
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


_SECRET_KEY_PATTERN = re.compile(
    r"(token|secret|password|authorization|api[_-]?key|credential)", re.IGNORECASE
)


def _sanitize(value: Any) -> Any:
    """Redact anything whose dict key looks secret-shaped before it is logged."""
    if isinstance(value, dict):
        return {
            key: ("[redacted]" if _SECRET_KEY_PATTERN.search(str(key)) else _sanitize(val))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


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

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "tool_failures": self.tool_failures,
                "rate_limit_events": self.rate_limit_events,
                "status_counts": dict(self.status_counts),
                "api_latency": _latency_stats(self.api_latencies_ms),
                "llm_latency": _latency_stats(self.llm_latencies_ms),
                "turn_latency": _latency_stats(self.turn_latencies_ms),
            }


_metrics = _MetricsRegistry()


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


def get_metrics_snapshot() -> dict[str, Any]:
    return _metrics.snapshot()
