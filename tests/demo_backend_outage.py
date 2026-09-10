"""
Manual, live-demo script: simulates the Homilivo backend being down and
shows that /chat degrades gracefully - a real HTTP 200 response with a
clean, human-readable message - instead of crashing or hanging.

Not a pytest test (no test_* functions, not auto-collected) - run it
directly, same convention as tests/hard_tests.py and
tests/real_api_test.py:

    python tests/demo_backend_outage.py

This does NOT touch the real Homilivo API or the real network: it
patches the same single choke point app/server.py's search_rooms()
calls (urllib.request.urlopen) that tests/conftest.py's
mock_homilivo_failure fixture patches for the automated suite, so this
script and that fixture can never disagree about what "the backend is
down" looks like.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import app


def main() -> None:
    client = TestClient(app)

    with patch(
        "app.server.urllib.request.urlopen",
        side_effect=ConnectionError("simulated Homilivo outage"),
    ):
        response = client.post(
            "/chat",
            json={"message": "find rooms in Jaipur", "mode": "fast"},
            headers={"Authorization": "Bearer demo-token-not-checked-here"},
        )

    print(f"HTTP status: {response.status_code}")
    print()
    print(json.dumps(response.json(), indent=2))

    body = response.json()
    ok = (
        response.status_code == 200
        and body.get("success") is False
        and "temporarily unavailable" in body.get("message", "").lower()
    )

    print()
    if ok:
        print("OK: clean HTTP 200 with an honest 'temporarily unavailable' "
              "message - no crash, no stack trace, no invented rooms.")
    else:
        print("UNEXPECTED: response did not match the known graceful-"
              "failure shape - investigate before presenting this.")


if __name__ == "__main__":
    main()
