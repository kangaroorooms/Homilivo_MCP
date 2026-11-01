"""One-off, read-only check: ONE real rooms API response through validate_room_data.

    python llm_approach/scripts/check_room_data.py [City]        (default city: Bangalore)

Sends one GET with the existing rooms client (same URL, headers, timeout and KANGAROO_TOKEN handling as the
chatbot), then checks each returned room on its own and prints ONLY: the HTTP status, how many rooms came
back, how many pass, and the names of the failing fields with a count. It never prints a room, a value, the
API's message, the token or any header. It writes nothing and makes no LLM call.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_approach import config, rooms_client  # noqa: E402
from llm_approach.guardrails.api_data_guardrail import validate_room_data  # noqa: E402


def main(argv: list[str]) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except Exception:  # noqa: BLE001 - python-dotenv missing or .env unreadable: use the environment
        pass
    token = os.getenv("KANGAROO_TOKEN", "").strip()
    if not token:
        print("KANGAROO_TOKEN is not set (put it in .env at the repo root). Nothing was sent.")
        return 2
    city = argv[0] if argv else "Bangalore"
    params = rooms_client.build_params({"city": city}, limit=config.ROOMS_LIMIT)
    try:
        raw = rooms_client.request_once(params, token=token)
    except Exception as exc:  # noqa: BLE001 - class name only: the message could hold the URL or token
        print(f"Request failed: {type(exc).__name__}")
        return 1
    print(f"HTTP status: {raw.status_code}")
    rooms = rooms_client.extract_rooms(raw.body)
    if rooms is None:
        print("No room list found in the response (bad payload).")
        return 1
    failing: Counter[str] = Counter()
    passed = 0
    for room in rooms:
        ok, reason = validate_room_data([room])
        if ok:
            passed += 1
            continue
        name, kind = rooms_client.failing_field(reason)
        failing[f"{name} ({kind})"] += 1
    print(f"Rooms returned: {len(rooms)}; pass validate_room_data: {passed}; fail: {len(rooms) - passed}")
    for field_name, count in failing.most_common():
        print(f"  failing field: {field_name}: {count} room(s)")
    if not failing:
        print("  no failing fields")
    print("Whole-response verdict (what the chatbot does):", "accepted" if len(rooms) == passed else "REJECTED")
    return 0 if len(rooms) == passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
