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
    llm_ms: float = 0.0
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
    line = {"ts": round(time.time(), 3), **metrics.to_dict()}
    try:
        path = metrics_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("metrics write failed: %s", type(exc).__name__)
