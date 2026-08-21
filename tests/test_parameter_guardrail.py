from app.guardrails import validate_search_arguments


def test_minimum_price_greater_than_maximum_is_rejected():
    """
    The parameter guardrail must reject an invalid price range.
    """

    test_input = {
        "min_price": 20000,
        "max_price": 10000,
    }

    valid, message = validate_search_arguments(
        test_input
    )

    assert valid is False, message