from app.guardrails import validate_room_data


def test_negative_room_price_is_rejected():
    """
    The API data guardrail must reject a room with a negative price.
    """

    test_room_data = [
        {
            "id": "test-room-001",
            "title": "Test Room",
            "city": "jaipur",
            "pricePerMonth": -10000,
            "roomType": "2BHK",
        }
    ]

    valid, message = validate_room_data(
        test_room_data
    )

    assert valid is False, message