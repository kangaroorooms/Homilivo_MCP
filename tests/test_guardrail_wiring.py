"""
Regression test proving all four guardrails (input, parameter, api-data,
output - see docs/Homilivo_Target_Architecture.md) are actually invoked
by process_message() (app/experiments/agent_modes.py) on a normal
room-search turn, in the correct order.

This test exists solely to fail loudly if a future change ever drops one
of the four calls from the pipeline - it does not re-test guardrail
LOGIC (that's tests/test_guardrails.py, tests/test_parameter_guardrail.py,
tests/test_api_guardrail.py). It spies on the guardrail functions as
agent_modes imports and calls them (module-level names bound at import
time), rather than patching app.guardrails directly, since that is
exactly how process_message() reaches them.
"""

from __future__ import annotations

import app.experiments.agent_modes as agent_modes


def test_all_four_guardrails_invoked_in_order_on_room_search(
    client, auth_headers, mock_homilivo_success, monkeypatch
):
    calls: list[str] = []

    real_validate_user_input = agent_modes.validate_user_input
    real_validate_search_arguments = agent_modes.validate_search_arguments
    real_validate_room_data = agent_modes.validate_room_data
    real_validate_ai_response = agent_modes.validate_ai_response

    def spy_input(*args, **kwargs):
        calls.append("input")
        return real_validate_user_input(*args, **kwargs)

    def spy_parameter(*args, **kwargs):
        calls.append("parameter")
        return real_validate_search_arguments(*args, **kwargs)

    def spy_api_data(*args, **kwargs):
        calls.append("api_data")
        return real_validate_room_data(*args, **kwargs)

    def spy_output(*args, **kwargs):
        calls.append("output")
        return real_validate_ai_response(*args, **kwargs)

    monkeypatch.setattr(agent_modes, "validate_user_input", spy_input)
    monkeypatch.setattr(agent_modes, "validate_search_arguments", spy_parameter)
    monkeypatch.setattr(agent_modes, "validate_room_data", spy_api_data)
    monkeypatch.setattr(agent_modes, "validate_ai_response", spy_output)

    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["search_performed"] is True

    # Strict, ordered equality (not e.g. "input" in calls) is what makes
    # this fail loudly the moment any one of the four calls is dropped
    # from process_message(), duplicated, or reordered - a subset/superset
    # check would silently tolerate a removed guardrail as long as the
    # others still ran.
    assert calls == ["input", "parameter", "api_data", "output"], (
        "Expected all four guardrails to run exactly once, in order "
        f"input -> parameter -> api_data -> output on a normal room-search "
        f"turn. Got: {calls}. If a guardrail was intentionally removed "
        "from the pipeline, update this test AND "
        "docs/Homilivo_Target_Architecture.md; otherwise this is a "
        "regression - a trust boundary in the architecture is now "
        "unguarded."
    )


def test_all_four_guardrails_invoked_in_order_through_the_graph(
    client, auth_headers, mock_homilivo_success, monkeypatch
):
    """
    Phase 5 (LangGraph router, app/graph/router.py): every /chat turn
    is now orchestrated by the graph rather than calling
    process_message() directly - see app/api.py's _run_chat_turn().
    This is the same spy as the test above, proving the wrap changed
    nothing about guardrail wiring: the graph's api_request branch
    node still reaches process_message() unchanged, so all four
    guardrails still fire exactly once, in order, on a normal
    room-search turn - regardless of which node in the graph called
    process_message().
    """
    import app.graph.router as router_module

    calls: list[str] = []

    real_validate_user_input = agent_modes.validate_user_input
    real_validate_search_arguments = agent_modes.validate_search_arguments
    real_validate_room_data = agent_modes.validate_room_data
    real_validate_ai_response = agent_modes.validate_ai_response

    def spy_input(*args, **kwargs):
        calls.append("input")
        return real_validate_user_input(*args, **kwargs)

    def spy_parameter(*args, **kwargs):
        calls.append("parameter")
        return real_validate_search_arguments(*args, **kwargs)

    def spy_api_data(*args, **kwargs):
        calls.append("api_data")
        return real_validate_room_data(*args, **kwargs)

    def spy_output(*args, **kwargs):
        calls.append("output")
        return real_validate_ai_response(*args, **kwargs)

    monkeypatch.setattr(agent_modes, "validate_user_input", spy_input)
    monkeypatch.setattr(agent_modes, "validate_search_arguments", spy_parameter)
    monkeypatch.setattr(agent_modes, "validate_room_data", spy_api_data)
    monkeypatch.setattr(agent_modes, "validate_ai_response", spy_output)

    # Also confirm the graph actually routed this turn to its
    # api_request branch node, not just that process_message() itself
    # behaved correctly if called directly.
    branches_executed: list[str] = []
    real_api_request_node = router_module.api_request_node

    async def spy_api_request_node(state):
        branches_executed.append("api_request")
        return await real_api_request_node(state)

    monkeypatch.setattr(router_module, "api_request_node", spy_api_request_node)
    router_module._COMPILED_GRAPH = None  # force a rebuild against the spy

    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    )

    router_module._COMPILED_GRAPH = None  # don't leak the spy graph

    assert response.status_code == 200
    assert response.json()["search_performed"] is True
    assert branches_executed == ["api_request"]
    assert calls == ["input", "parameter", "api_data", "output"], (
        "All four guardrails must still fire exactly once, in order, "
        f"when a turn is routed through the LangGraph router. Got: {calls}."
    )
