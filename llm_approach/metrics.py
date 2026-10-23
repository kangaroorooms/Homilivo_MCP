"""Per-turn metrics. Numbers, flags and names only: never message or reply text.

``record`` appends one JSON line to ``eval/results/metrics.jsonl`` when
``LLM_APPROACH_METRICS=1``. The line is built from the fixed field list below,
so nothing else can leak into the file.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENABLE_ENV = "LLM_APPROACH_METRICS"
PATH_ENV = "LLM_APPROACH_METRICS_PATH"  # tests only
DEFAULT_PATH = Path(__file__).resolve().parent / "eval" / "results" / "metrics.jsonl"


@dataclass
class TurnMetrics:
    intent: str | None = None
    operation: str | None = None
    confidence: float | None = None
    slots_valid: list[str] = field(default_factory=list)  # slot names only
    slots_rejected: list[str] = field(default_factory=list)  # slot names only
    action: str | None = None
    tool_called: bool = False
    llm_ms: float = 0.0  # all LLM calls of the turn together
    llm_calls: int = 0
    llm_call_ms: list[float] = field(default_factory=list)  # each LLM call, in order
    api_ms: float = 0.0
    total_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    fallback_used: bool = False
    error_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("llm_ms", "api_ms", "total_ms"):
            data[key] = round(data[key], 1)
        data["llm_call_ms"] = [round(v, 1) for v in data["llm_call_ms"]]
        if data["confidence"] is not None:
            data["confidence"] = round(data["confidence"], 3)
        return data


def enabled() -> bool:
    return os.getenv(ENABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def metrics_path() -> Path:
    override = os.getenv(PATH_ENV, "").strip()
    return Path(override) if override else DEFAULT_PATH


def record(metrics: TurnMetrics) -> None:
    """Append one JSON line when enabled. Never raises."""
    if not enabled():
        return
    from llm_approach import config  # provider and model names only, never a key

    provider, model = config.label()
    line = {"ts": round(time.time(), 3), "provider": provider, "model": model, **metrics.to_dict()}
    try:
        path = metrics_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("metrics write failed: %s", type(exc).__name__)


# ---- summary (p50 / p95 from the JSONL) ---------------------------------------


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (pct in 0..100); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * pct // 100))  # ceil
    return ordered[int(rank) - 1]


def _stats(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95)}


def summarize(path: Path | None = None) -> dict[str, Any]:
    """p50 and p95 of the latency fields in the metrics file (numbers only, no text).

    ``per_llm_call_ms`` (also kept as ``per_groq_call_ms``): every individual LLM call. ``llm_ms``: the
    LLM time of a turn (retries included). ``providers``: the (provider, model) pairs found; rows
    written before the provider switch count as Groq. ``total_ms``: the whole turn. Unreadable lines are skipped."""
    path = path or metrics_path()
    turns: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    turns.append(row)
    except OSError:
        pass

    def numbers(key: str, rows: list[dict[str, Any]]) -> list[float]:
        return [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]

    llm_rows = [r for r in turns if numbers("llm_ms", [r]) and r["llm_ms"] > 0]
    calls = [float(v) for r in turns for v in (r.get("llm_call_ms") or []) if isinstance(v, (int, float))]
    from llm_approach import config

    pairs = sorted({(str(r.get("provider") or config.LEGACY_PROVIDER), str(r.get("model") or config.LEGACY_MODEL))
                    for r in turns})
    return {
        "providers": [{"provider": p, "model": m} for p, m in pairs],
        "turns": len(turns),
        "total_ms": _stats(numbers("total_ms", turns)),
        "llm_ms": _stats(numbers("llm_ms", llm_rows)),
        "per_llm_call_ms": _stats(calls),
        "per_groq_call_ms": _stats(calls),  # old name, same numbers
        "fallback_turns": sum(1 for r in turns if r.get("fallback_used") is True),
        "error_turns": sum(1 for r in turns if r.get("error_class")),
    }


if __name__ == "__main__":  # python -m llm_approach.metrics [path]
    import sys

    print(json.dumps(summarize(Path(sys.argv[1]) if len(sys.argv) > 1 else None), indent=2))
