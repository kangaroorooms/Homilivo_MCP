"""
Phase 5 - LangGraph router tests.

Covers three things the DoD requires:
  1. classify_branch() maps every route_message() outcome onto the
     correct one of the four graph branches (FAQ/RAG, Conversation,
     API-request, LLM-fallback).
  2. fast_path_node() - which WRAPS route_message() unchanged - picks
     the correct branch for real intents: FAQ, conversation
     (greeting/info/unsupported-action), new-search, refinement,
     follow-up, and clarification.
  3. The compiled graph actually reaches the branch node the
     classifier selected (not just that classify_branch() is correct
     in isolation), and that a follow-up still resolves via
     conversation memory when run through the graph end to end.

Guardrail-ordering-through-the-graph is covered separately in
tests/test_guardrail_wiring.py (this file does not duplicate that).

CI-SAFE: no test here requires a live Ollama/Groq/Redis service -
the one case that would reach the Gemma fallback (an UNKNOWN message)
mocks classify_intent() unavailable, exactly like
tests/test_golden_eval.py's out-of-scope eval does.
"""

from __future__ import annotations

import pytest

from app.experiments.agent_modes import (
    ConversationState,
    GemmaError,
    MESSAGE_CLARIFICATION,
    MESSAGE_EXIT,
    MESSAGE_FOLLOWUP,
    MESSAGE_GREETING,
    MESSAGE_INFO,
    MESSAGE_NEW_SEARCH,
    MESSAGE_PLATFORM_KNOWLEDGE,
    MESSAGE_REFINEMENT,
    MESSAGE_UNKNOWN,
    MESSAGE_UNSUPPORTED_ACTION,
    MessageRoute,
)
from app.graph import router as router_module
from app.graph.nodes import execute as execute_module
from app.graph.nodes import fast_path as fast_path_module
from app.graph.nodes.fast_path import (
    BRANCH_API_REQUEST,
    BRANCH_CONVERSATION,
    BRANCH_FAQ,
    BRANCH_LLM_FALLBACK,
    classify_branch,
    fast_path_node,
)


# ============================================================
# 1. classify_branch() - every MessageRoute.message_type
# ============================================================

BRANCH_MAPPING_CASES = [
    (MESSAGE_GREETING, BRANCH_CONVERSATION),
    (MESSAGE_INFO, BRANCH_CONVERSATION),
    (MESSAGE_EXIT, BRANCH_CONVERSATION),
    (MESSAGE_UNSUPPORTED_ACTION, BRANCH_CONVERSATION),
    (MESSAGE_PLATFORM_KNOWLEDGE, BRANCH_FAQ),
    (MESSAGE_NEW_SEARCH, BRANCH_API_REQUEST),
    (MESSAGE_REFINEMENT, BRANCH_API_REQUEST),
    (MESSAGE_FOLLOWUP, BRANCH_API_REQUEST),
    (MESSAGE_CLARIFICATION, BRANCH_API_REQUEST),
    (MESSAGE_UNKNOWN, BRANCH_LLM_FALLBACK),
]


@pytest.mark.parametrize("message_type, expected_branch", BRANCH_MAPPING_CASES)
def test_classify_branch_maps_every_message_type(message_type, expected_branch):
    route = MessageRoute(message_type=message_type)
    assert classify_branch(route) == expected_branch


# ============================================================
# 2. fast_path_node() - real route_message(), real queries
# ============================================================

def _fresh_state() -> ConversationState:
    return ConversationState()


def _state_after_search() -> ConversationState:
    """Mirrors tests/test_operations.py's helper of the same shape:
    a session that has already run one search and has rooms/search
    context to follow up on or refine."""
    state = ConversationState()
    state.previous_search = {"city": "Jaipur", "sort": "latest", "limit": 10}
    state.previous_rooms = [
        {
            "id": "r1",
            "title": "Room A",
            "pricePerMonth": 8000,
            "roomType": "1BHK",
            "city": "Jaipur",
            "idealFor": ["Students"],
        },
        {
            "id": "r2",
            "title": "Room B",
            "pricePerMonth": 12000,
            "roomType": "2BHK",
            "city": "Jaipur",
            "idealFor": ["Family"],
        },
    ]
    return state


def _state_pending_city() -> ConversationState:
    state = ConversationState()
    state.pending_search = {
        "city": None,
        "room_types": ["2BHK"],
        "sort": "latest",
        "limit": 10,
    }
    return state


FAST_PATH_CASES = [
    # FAQ / RAG
    ("what are the platform fees?", _fresh_state, BRANCH_FAQ),
    ("how does verification work?", _fresh_state, BRANCH_FAQ),
    # Conversation - greeting / info / unsupported action
    ("hello", _fresh_state, BRANCH_CONVERSATION),
    ("what do you do", _fresh_state, BRANCH_CONVERSATION),
    ("book this room", _fresh_state, BRANCH_CONVERSATION),
    # API request - new search
    ("find rooms in Jaipur", _fresh_state, BRANCH_API_REQUEST),
    # API request - refinement (needs prior search context)
    ("only 2BHK", _state_after_search, BRANCH_API_REQUEST),
    ("under 8000", _state_after_search, BRANCH_API_REQUEST),
    # API request - follow-up (needs prior rooms)
    ("which one is cheapest?", _state_after_search, BRANCH_API_REQUEST),
    ("tell me about the first one", _state_after_search, BRANCH_API_REQUEST),
    # API request - clarification response (needs a pending search)
    ("Jaipur", _state_pending_city, BRANCH_API_REQUEST),
    # Ambiguous - deterministic router genuinely cannot decide
    ("any cheap rooms???", _fresh_state, BRANCH_LLM_FALLBACK),
]


@pytest.mark.parametrize("query, make_state, expected_branch", FAST_PATH_CASES)
def test_fast_path_node_routes_real_queries_to_the_correct_branch(
    query, make_state, expected_branch
):
    state = {"user_query": query, "conversation_state": make_state()}
    assert fast_path_node(state)["branch"] == expected_branch

# ============================================================
# 2.5. Single routing - route is created once and reused
# ============================================================

@pytest.mark.anyio
async def test_graph_reuses_route_created_by_fast_path(monkeypatch):

    call_count = 0

    original_route_message = fast_path_module.route_message

    def spy_route_message(*args, **kwargs):

        nonlocal call_count

        call_count += 1

        return original_route_message(*args, **kwargs)

    monkeypatch.setattr(
        fast_path_module,
        "route_message",
        spy_route_message,
    )

    captured_route = None

    async def fake_process_message(**kwargs):

        nonlocal captured_route

        captured_route = kwargs.get("route")

        return {"response": "test"}

    monkeypatch.setattr(
        execute_module,
        "process_message",
        fake_process_message,
    )

    conversation_state = ConversationState()

    graph_state = {
        "user_query": "hello",
        "mode": "fast",
        "client": None,
        "conversation_state": conversation_state,
        "branch": None,
        "route": None,
        "result": None,
        "on_token": None,
    }

    fast_result = fast_path_node(graph_state)

    graph_state.update(fast_result)

    await execute_module._run_turn(graph_state)

    assert call_count == 1
    assert captured_route is fast_result["route"]
# ============================================================
# 3. The compiled graph reaches the branch the classifier picked,
#    and a follow-up resolves via conversation memory end to end.
# ============================================================

def _spy_on_branch_nodes(monkeypatch) -> list[str]:
    """
    Wraps the four node callables app.graph.router.build_graph() binds
    onto its branch edges (module-global names looked up at call time,
    same technique tests/test_guardrail_wiring.py uses for the
    guardrail functions) so a test can assert which single branch node
    the graph actually executed - not just that classify_branch()
    would have picked the right one in isolation.
    """
    executed: list[str] = []

    for branch in (BRANCH_FAQ, BRANCH_CONVERSATION, BRANCH_API_REQUEST, BRANCH_LLM_FALLBACK):
        real = getattr(router_module, f"{branch}_node")

        def make_spy(name, real_fn):
            async def spy(state):
                executed.append(name)
                return await real_fn(state)

            return spy

        monkeypatch.setattr(router_module, f"{branch}_node", make_spy(branch, real))

    return executed


def test_faq_query_reaches_the_faq_branch_node(client, auth_headers, monkeypatch):
    executed = _spy_on_branch_nodes(monkeypatch)
    router_module._COMPILED_GRAPH = None  # force a rebuild against the spies

    response = client.post(
        "/chat",
        json={"message": "how does verification work?", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert executed == [BRANCH_FAQ]
    assert "About this assistant" in response.json()["sources"]

    router_module._COMPILED_GRAPH = None  # don't leak the spy into later tests


def test_greeting_reaches_the_conversation_branch_node(client, auth_headers, monkeypatch):
    executed = _spy_on_branch_nodes(monkeypatch)
    router_module._COMPILED_GRAPH = None

    response = client.post(
        "/chat", json={"message": "hello", "mode": "fast"}, headers=auth_headers,
    )

    assert response.status_code == 200
    assert executed == [BRANCH_CONVERSATION]

    router_module._COMPILED_GRAPH = None


def test_unsupported_action_reaches_the_conversation_branch_node(
    client, auth_headers, monkeypatch
):
    executed = _spy_on_branch_nodes(monkeypatch)
    router_module._COMPILED_GRAPH = None

    response = client.post(
        "/chat", json={"message": "book this room", "mode": "fast"}, headers=auth_headers,
    )

    assert response.status_code == 200
    assert executed == [BRANCH_CONVERSATION]
    assert "can't complete that action" in response.json()["message"]

    router_module._COMPILED_GRAPH = None


def test_new_search_reaches_the_api_request_branch_node(
    client, auth_headers, mock_homilivo_success, monkeypatch
):
    executed = _spy_on_branch_nodes(monkeypatch)
    router_module._COMPILED_GRAPH = None

    response = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert executed == [BRANCH_API_REQUEST]
    assert response.json()["search_performed"] is True

    router_module._COMPILED_GRAPH = None


def test_refinement_reaches_the_api_request_branch_node(
    client, auth_headers, mock_homilivo_success, monkeypatch
):
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()
    session_id = search["session_id"]

    executed = _spy_on_branch_nodes(monkeypatch)
    router_module._COMPILED_GRAPH = None

    refinement = client.post(
        "/chat",
        json={"message": "only 2BHK", "session_id": session_id, "mode": "fast"},
        headers=auth_headers,
    )

    assert refinement.status_code == 200
    assert executed == [BRANCH_API_REQUEST]

    router_module._COMPILED_GRAPH = None


def test_followup_reaches_api_request_branch_and_resolves_via_conversation_memory(
    client, auth_headers, mock_homilivo_success, monkeypatch
):
    """
    Proves follow-ups still resolve via session/conversation memory
    when routed through the graph: the second turn ("which one is
    cheapest?") carries no room information of its own - only the
    session's remembered previous_rooms (set by the first turn) lets
    resolve_room_reference() (called inside process_message(), reached
    via the graph's api_request branch) identify the right room.
    """
    search = client.post(
        "/chat",
        json={"message": "find rooms in Jaipur", "mode": "fast"},
        headers=auth_headers,
    ).json()
    session_id = search["session_id"]
    assert search["room_count"] == 3

    executed = _spy_on_branch_nodes(monkeypatch)
    router_module._COMPILED_GRAPH = None

    followup = client.post(
        "/chat",
        json={
            "message": "which one is cheapest?",
            "session_id": session_id,
            "mode": "fast",
        },
        headers=auth_headers,
    )

    assert followup.status_code == 200
    body = followup.json()

    assert executed == [BRANCH_API_REQUEST]
    assert body["session_id"] == session_id
    assert body["selected_room"] is not None
    assert body["selected_room"]["id"] == "room-2"
    assert body["selected_room"]["price_per_month"] == 8500

    router_module._COMPILED_GRAPH = None


def test_ambiguous_message_reaches_llm_fallback_branch_node(
    client, auth_headers, monkeypatch
):
    """
    "any cheap rooms???" has no city/price/type token, so
    route_message() genuinely cannot classify it (MESSAGE_UNKNOWN) -
    this is the one case the graph routes to the LLM-fallback branch.
    classify_intent() is mocked unavailable so the test is fast and
    deterministic regardless of whether Ollama happens to be running -
    same pattern as tests/test_golden_eval.py's out-of-scope eval.
    """
    monkeypatch.setattr(
        "app.experiments.agent_modes.classify_intent",
        lambda query: (_ for _ in ()).throw(GemmaError("no local model available")),
    )
    executed = _spy_on_branch_nodes(monkeypatch)
    router_module._COMPILED_GRAPH = None

    response = client.post(
        "/chat", json={"message": "any cheap rooms???", "mode": "fast"}, headers=auth_headers,
    )

    assert response.status_code == 200
    assert executed == [BRANCH_LLM_FALLBACK]

    router_module._COMPILED_GRAPH = None
