"""On/off switches for the copied guardrails that run inside the search path.

The two checks are OFF by default so unit tests that feed hand-made rooms or slots keep working, and are
turned ON together by ``pipeline.configure_production()`` (which ``server_app`` calls at startup).
The semantic gate is "off" by default and "shadow" (or the ``LLM_APPROACH_SEMANTIC_GATE`` setting) in production.
Never rely on the defaults in a server.
"""

from __future__ import annotations

_room_data_check = False  # validate_room_data on every rooms API response
_search_arg_check = False  # validate_search_arguments before the rooms API call
GATE_MODES = ("off", "shadow", "enforce")
_semantic_gate = "off"  # off | shadow (record only) | enforce (block off-topic first messages)


def configure(
    *, room_data_check: bool | None = None, search_arg_check: bool | None = None, semantic_gate: str | None = None
) -> None:
    global _room_data_check, _search_arg_check, _semantic_gate
    if semantic_gate is not None:
        if semantic_gate not in GATE_MODES:
            raise ValueError(f"semantic_gate must be one of: {', '.join(GATE_MODES)}")
        _semantic_gate = semantic_gate
    if room_data_check is not None:
        _room_data_check = bool(room_data_check)
    if search_arg_check is not None:
        _search_arg_check = bool(search_arg_check)


def room_data_check() -> bool:
    return _room_data_check


def search_arg_check() -> bool:
    return _search_arg_check


def semantic_gate() -> str:
    return _semantic_gate


def snapshot() -> dict[str, bool | str]:
    return {"room_data_check": _room_data_check, "search_arg_check": _search_arg_check, "semantic_gate": _semantic_gate}
