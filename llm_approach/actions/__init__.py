"""Actions: one handler per intent. Handlers take the *validated* slots.

Every handler has the shape ``handler(decision, validated, state, budget, ...)``
where ``validated`` is the ``clean`` dict from ``validators.validate_slots``.
Handlers never read ``decision.slots`` for values they act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_approach import config


class ToolBudgetExceeded(RuntimeError):
    """A second rooms API call was attempted in one turn."""


@dataclass
class TurnBudget:
    """Per-turn tool-call budget (the rooms API counts as one call)."""

    max_tool_calls: int = config.MAX_TOOL_CALLS_PER_TURN
    tool_calls: int = 0

    def use_tool_call(self) -> None:
        if self.tool_calls >= self.max_tool_calls:
            raise ToolBudgetExceeded(
                f"at most {self.max_tool_calls} tool call(s) per turn"
            )
        self.tool_calls += 1


@dataclass
class ActionResult:
    reply: str
    rooms: list[dict[str, Any]] | None = None  # frontend cards, None = unchanged
    action: str = ""
    tool_called: bool = False
    error_class: str | None = None
