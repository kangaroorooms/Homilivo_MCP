# Copied from tests/test_parameter_guardrail.py at commit 7dbcba3 (copied, not imported).
# Changes: import now points to llm_approach.guardrails. Every changed line is marked '# v2:'.
# Not copied from this file: cases that need app.api, TestClient, agent_modes or the MCP server.

from llm_approach.guardrails.parameter_guardrail import validate_search_arguments  # v2: was: from app.guardrails import validate_search_arguments


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
