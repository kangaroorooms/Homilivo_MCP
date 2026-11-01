# Copied from tests/test_api_guardrail.py at commit 7dbcba3 (copied, not imported).
# Changes: import now points to llm_approach.guardrails. Every changed line is marked '# v2:'.
# Not copied from this file: cases that need app.api, TestClient, agent_modes or the MCP server.

from llm_approach.guardrails.api_data_guardrail import validate_room_data  # v2: was: from app.guardrails import validate_room_data


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
