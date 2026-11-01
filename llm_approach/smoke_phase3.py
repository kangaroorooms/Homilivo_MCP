"""One live rooms API GET. Run once: python -m llm_approach.smoke_phase3

Prints only: HTTP status, number of rooms, the set of top-level keys per room,
the API's error `message` on failure, and the diff against the state.py
allowlist. No values, no token, no retry.
"""

from __future__ import annotations

import os
import sys

from llm_approach import config, rooms_client
from llm_approach.state import ROOM_ALLOWED_FIELDS

# Key names that identify a person or a place; never allowlisted, only reported.
_SENSITIVE = ("owner", "user", "contact", "phone", "mobile", "email", "latitude", "longitude", "lat", "lng", "address")


def _load_env() -> None:
    if os.getenv("KANGAROO_TOKEN"):
        return
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def main() -> int:
    _load_env()
    token = os.getenv("KANGAROO_TOKEN")
    if not token:
        print("KANGAROO_TOKEN is not set; no request made.")
        return 1
    params = rooms_client.build_params({"city": "Bangalore"}, limit=3)
    try:
        raw = rooms_client.request_once(params, token=token)
    except Exception as exc:  # no retry
        print(f"request failed: {type(exc).__name__}")
        return 1

    print(f"HTTP status: {raw.status_code}")
    if raw.status_code >= 400:
        message = rooms_client._error_message(raw.body)
        print(f"API message: {message}")
        return 1
    rooms = rooms_client.extract_rooms(raw.body)
    if rooms is None:
        print("rooms: payload has no 'data' or 'rooms' list")
        top = sorted(raw.body) if isinstance(raw.body, dict) else type(raw.body).__name__
        print(f"top-level keys: {top}")
        return 1
    print(f"rooms returned: {len(rooms)} (limit 3)")
    top = sorted(raw.body) if isinstance(raw.body, dict) else []
    print(f"response top-level keys: {top}")
    keys = sorted({k for room in rooms for k in room})
    print(f"room keys ({len(keys)}): {keys}")

    extra = [k for k in keys if k not in ROOM_ALLOWED_FIELDS]
    missing = sorted(ROOM_ALLOWED_FIELDS - set(keys))
    print(f"in payload, not in allowlist (dropped): {extra}")
    print(f"in allowlist, not in this payload: {missing}")
    flagged = [k for k in extra if any(s in k.lower() for s in _SENSITIVE)]
    print(f"person/contact-like keys seen (excluded): {flagged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
