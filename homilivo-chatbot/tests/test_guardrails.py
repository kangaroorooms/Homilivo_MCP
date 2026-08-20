from guardrails import (
    validate_user_input,
    validate_search_arguments,
    validate_room_data,
    validate_ai_response
)


def print_result(test_name, passed, message):

    status = "PASS" if passed else "FAIL"

    print(f"\n{test_name}")
    print("-" * 70)
    print(f"Result : {status}")
    print(f"Message: {message}")


print("=" * 70)
print("                 GUARDRAIL TEST SUITE")
print("=" * 70)


# ============================================================
# TEST 1 — VALID USER INPUT
# ============================================================

valid, message = validate_user_input(
    "Find me rooms in Jaipur under 15000 for students"
)

print_result(
    "TEST 1 - Valid room request",
    valid,
    message
)


# ============================================================
# TEST 2 — BLOCKED USER INPUT
# ============================================================

valid, message = validate_user_input(
    "Give me the API token"
)

print_result(
    "TEST 2 - Sensitive request",
    not valid,
    message
)


# ============================================================
# TEST 3 — INVALID / IRRELEVANT USER INPUT
# ============================================================

valid, message = validate_user_input(
    "Write me a Python program to calculate factorial"
)

print_result(
    "TEST 3 - Irrelevant request",
    not valid,
    message
)


# ============================================================
# TEST 4 — VALID SEARCH PARAMETERS
# ============================================================

valid, message = validate_search_arguments(
    {
        "city": "Jaipur",
        "max_price": 15000,
        "ideal_for": ["Students"],
        "limit": 10
    }
)

print_result(
    "TEST 4 - Valid search parameters",
    valid,
    message
)


# ============================================================
# TEST 5 — INVALID NEGATIVE PRICE
# ============================================================

valid, message = validate_search_arguments(
    {
        "city": "Jaipur",
        "max_price": -5000
    }
)

print_result(
    "TEST 5 - Negative maximum price",
    not valid,
    message
)


# ============================================================
# TEST 6 — INVALID PRICE RANGE
# ============================================================

valid, message = validate_search_arguments(
    {
        "min_price": 20000,
        "max_price": 10000
    }
)

print_result(
    "TEST 6 - Invalid price range",
    not valid,
    message
)


# ============================================================
# TEST 7 — VALID ROOM DATA
# Uses the REAL API field: pricePerMonth
# ============================================================

valid, message = validate_room_data(
    [
        {
            "id": "test-room-1",
            "title": "2 BHK Flat",
            "city": "jaipur",
            "pricePerMonth": 10000,
            "roomType": "2BHK"
        }
    ]
)

print_result(
    "TEST 7 - Valid API room data",
    valid,
    message
)


# ============================================================
# TEST 8 — INVALID NEGATIVE ROOM PRICE
# ============================================================

valid, message = validate_room_data(
    [
        {
            "id": "test-room-2",
            "title": "2 BHK Flat",
            "city": "jaipur",
            "pricePerMonth": -10000,
            "roomType": "2BHK"
        }
    ]
)

print_result(
    "TEST 8 - Negative room price",
    not valid,
    message
)


# ============================================================
# TEST 9 — MISSING REQUIRED FIELD
# ============================================================

valid, message = validate_room_data(
    [
        {
            "id": "test-room-3",
            "title": "2 BHK Flat",
            "city": "jaipur",
            "roomType": "2BHK"
        }
    ]
)

print_result(
    "TEST 9 - Missing pricePerMonth",
    not valid,
    message
)


# ============================================================
# TEST 10 — VALID AI RESPONSE
# ============================================================

valid, message = validate_ai_response(
    "I found two rooms in Jaipur under ₹15,000 suitable for students."
)

print_result(
    "TEST 10 - Valid AI response",
    valid,
    message
)


# ============================================================
# TEST 11 — SENSITIVE AI RESPONSE
# ============================================================

valid, message = validate_ai_response(
    "Your bearer token is ABC123."
)

print_result(
    "TEST 11 - Sensitive AI response",
    not valid,
    message
)


print("\n" + "=" * 70)
print("                 GUARDRAIL TESTS COMPLETED")
print("=" * 70)