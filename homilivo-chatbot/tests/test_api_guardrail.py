from guardrails import validate_room_data


print("=" * 70)
print("  API DATA GUARDRAIL TEST")
print("=" * 70)

# Simulated malicious/invalid API response
# Negative room price should never be accepted.

test_room_data = [
    {
        "id": "test-room-001",
        "title": "Test Room",
        "city": "jaipur",
        "price": -10000
    }
]

print("\nSimulated API Data:")
print(test_room_data)

print("\nRunning API data guardrail...")

valid, message = validate_room_data(
    test_room_data
)

if valid:

    print("\n✗ TEST FAILED")
    print("Invalid room data was accepted.")

else:

    print("\n✓ TEST PASSED")
    print("Invalid room data was blocked.")

print(f"Reason: {message}")

print("\n" + "=" * 70)