"""
LangGraph orchestration layer (Phase 5 - target architecture diagram's
"LangGraph Router" box).

This package adds explicit graph structure around the existing,
unchanged conversational engine (app/experiments/agent_modes.py). It
does not reimplement routing, guardrails, extraction, or response
generation - see app/graph/router.py for the wrap.
"""

from __future__ import annotations

from app.graph.router import run_graph

__all__ = ["run_graph"]
