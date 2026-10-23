"""On/off switches for the copied guardrails that run inside the search path.

Both are OFF by default so unit tests that feed hand-made rooms or slots keep working, and are
turned ON together by ``pipeline.configure_production()`` (which ``server_app`` calls at startup).
Never rely on the default in a server.
"""

from __future__ import annotations

_room_data_check = False  # validate_room_data on every rooms API response
_search_arg_check = False  # validate_search_arguments before the rooms API call


def configure(*, room_data_check: bool | None = None, search_arg_check: bool | None = None) -> None:
    global _room_data_check, _search_arg_check
    if room_data_check is not None:
        _room_data_check = bool(room_data_check)
    if search_arg_check is not None:
        _search_arg_check = bool(search_arg_check)


def room_data_check() -> bool:
    return _room_data_check


def search_arg_check() -> bool:
    return _search_arg_check


def snapshot() -> dict[str, bool]:
    return {"room_data_check": _room_data_check, "search_arg_check": _search_arg_check}
