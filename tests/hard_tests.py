"""
Manual, human-readable report for the operation-detection/routing core.

Run from the PROJECT ROOT (the folder that contains the `app/` package):

    python tests/hard_tests.py

This is a presentation layer only - the actual pass/fail assertions
that CI runs live in tests/test_operations.py (pytest-collected). This
script imports the SAME case data from there so the two never drift
out of sync, and prints the same PASS/FAIL report the original version
of this file printed.

Sections:
  STRICT   - deterministic Tier-A cases. Model-independent. Hard assertions.
             If any of these fail, the change broke the deterministic core.
  RANKED   - "second cheapest" / "third most expensive" / "last one" -
             the compound ordinal+price-ranking cases.
  ROUTING  - multi-turn routing via route_message(). Reports PASS/FAIL.
  SOFT     - Tier-B semantic cases. Require sentence-transformers + the
             model. Reported as expected-vs-actual, never hard-failed,
             because the exact mapping depends on ACCEPT_THRESHOLD /
             SLOT_MARGIN tuning (see tests/test_operations.py for the
             one entry currently xfailed there).
  NOTES    - cases whose behaviour intentionally changed with a past
             refactor. Printed so you can confirm the new behaviour is
             still acceptable.
"""

import sys

try:
    from app.experiments.agent_modes import (
        detect_room_operation,
        resolve_room_reference,
        route_message,
    )
    from tests.test_operations import (
        RANKED_OPERATION_CASES,
        ROUTING_CASES,
        SOFT_OPERATION_CASES,
        STRICT_OPERATION_CASES,
        _priced_rooms,
    )
except Exception as error:  # noqa: BLE001
    print("Could not import app.experiments.agent_modes / tests.test_operations.")
    print("Run this from the project root, inside the same env as the app.")
    print("Import error:", error)
    sys.exit(2)


def _model_available() -> bool:
    try:
        from app.experiments.interpreter import warmup
        warmup()
        from app.experiments import interpreter
        return not interpreter._model_failed and interpreter._model is not None
    except Exception:  # noqa: BLE001
        return False


MODEL = _model_available()


def run_strict() -> int:
    print("\n=== STRICT (deterministic, must pass) ===")
    failures = 0
    for query, expected in STRICT_OPERATION_CASES:
        got = detect_room_operation(query)
        ok = got == expected
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {query!r:45} "
              f"expected {expected:24} got {got}")
    print(f"  -> {len(STRICT_OPERATION_CASES) - failures}/{len(STRICT_OPERATION_CASES)} passed")
    return failures


def run_ranked() -> int:
    print("\n=== RANKED (second cheapest / last one, must pass) ===")
    failures = 0
    for query, expected in RANKED_OPERATION_CASES:
        got = detect_room_operation(query)
        ok = got == expected
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {query!r:35} "
              f"expected {expected:22} got {got}")

    room = resolve_room_reference("second cheapest one", _priced_rooms(), None)
    ok = bool(room and room["id"] == "r2")
    if not ok:
        failures += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] resolve('second cheapest one') "
          f"-> {room['id'] if room else None} (expected r2)")

    print(f"  -> {len(RANKED_OPERATION_CASES) + 1 - failures}/{len(RANKED_OPERATION_CASES) + 1} passed")
    return failures


def run_routing() -> None:
    print("\n=== ROUTING (multi-turn; reports PASS/FAIL) ===")
    for query, make_state, expected in ROUTING_CASES:
        route = route_message(query, make_state())
        got = route.message_type
        ok = got == expected
        print(f"  [{'PASS' if ok else 'FAIL'}] {query!r:32} "
              f"expected {expected:22} got {got:22}")


def run_soft() -> None:
    print("\n=== SOFT (Tier-B semantic; report only, tune if wrong) ===")
    if not MODEL:
        print("  Embedding model NOT available -> every Tier-B case returns None.")
        print("  Install `sentence-transformers` to exercise this section.")
    for case in SOFT_OPERATION_CASES:
        # pytest.param(...) wraps the xfailed entry; unwrap if needed.
        query, expected = case.values if hasattr(case, "values") else case
        got = detect_room_operation(query)
        flag = "ok " if got == expected else "??"
        print(f"  [{flag}] {query!r:48} want {str(expected):16} got {got}")


def run_notes() -> None:
    print("\n=== NOTES (behaviour changed on purpose; confirm acceptable) ===")
    samples = [
        ("cheapest of the first two",
         "Ordinals win when the superlative does not directly precede a "
         "rank word -> expect ORDINAL_1. 'second cheapest' (ordinal "
         "directly before the superlative) is RANKED_LOWEST_PRICE instead."),
        ("is this one more expensive than that one",
         "Pairwise 'than' comparison -> resolves to ROOM_PRICE of the selected "
         "room (not a set-wide max). Was left to fallback before."),
    ]
    for query, note in samples:
        print(f"  {query!r} -> {detect_room_operation(query)}")
        print(f"      {note}")


if __name__ == "__main__":
    print(f"Embedding model available: {MODEL}")
    failures = run_strict()
    failures += run_ranked()
    run_routing()
    run_soft()
    run_notes()
    print("\n" + ("FAILURES PRESENT" if failures else "ALL STRICT/RANKED CASES PASS"))
    sys.exit(1 if failures else 0)
