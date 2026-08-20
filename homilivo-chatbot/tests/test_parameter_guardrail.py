from guardrails import validate_search_arguments


print("=" * 70)
print("  PARAMETER GUARDRAIL TEST")
print("=" * 70)

# Invalid case:
# Minimum price is greater than maximum price

test_input = {
    "min_price": 20000,
    "max_price": 10000
}

print("\nTest Input:")
print(test_input)

print("\nRunning parameter guardrail...")

valid, message = validate_search_arguments(test_input)

if valid:
    print("\n✗ TEST FAILED")
    print("Invalid parameters were accepted.")

else:
    print("\n✓ TEST PASSED")
    print("Invalid parameters were blocked.")

print(f"Reason: {message}")

print("\n" + "=" * 70)