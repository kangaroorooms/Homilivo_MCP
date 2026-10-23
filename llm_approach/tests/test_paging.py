"""Phase 5c: rooms API paging info (meta.total, meta.hasNextPage) and SHOW_MORE."""

from __future__ import annotations

import json

import pytest

from llm_approach import config, rooms_client
from llm_approach.actions import TurnBudget, search
from llm_approach.schema import Decision, Intent
from llm_approach.state import LLMConversationState

BANGALORE = {"city": "Bangalore"}
AREA = {"city": "Bangalore", "area": "Koramangala"}


def _rooms(n, location="Koramangala"):
    return [{"id": str(i), "title": f"Room {i}", "city": "bangalore", "location": location,
             "landmark": "", "pricePerMonth": 9000, "roomType": "Single"} for i in range(n)]


# ---- meta parsing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"data": [], "meta": {"total": 24, "hasNextPage": True}}, (24, True)),
        ({"data": [], "meta": {"total": 0, "hasNextPage": False}}, (0, False)),
        ({"data": []}, (None, None)),
        ({"data": [], "meta": None}, (None, None)),
        ({"data": [], "meta": "x"}, (None, None)),
        ({"data": [], "meta": {}}, (None, None)),
        ({"data": [], "meta": {"total": "24", "hasNextPage": "yes"}}, (None, None)),
        ({"data": [], "meta": {"total": True, "hasNextPage": 1}}, (None, None)),
        ({"data": [], "meta": {"total": -3, "hasNextPage": False}}, (None, False)),
        ({"data": [], "meta": {"total": 7}}, (7, None)),
        (None, (None, None)),
        ([1, 2], (None, None)),
    ],
)
def test_parse_meta(body, expected):
    assert rooms_client.parse_meta(body) == expected


# ---- live path with a spy -------------------------------------------------------


class _Resp:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _live(monkeypatch, make_body):
    monkeypatch.delenv(config.OFFLINE_ROOMS_ENV, raising=False)
    monkeypatch.setenv("KANGAROO_TOKEN", "not-a-real-token")
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append(dict(params))
        return _Resp(make_body(params))

    monkeypatch.setattr(rooms_client, "_httpx_get", fake_get)
    return calls


def _body(rooms, total, has_next):
    return {"success": True, "data": rooms, "meta": {"total": total, "hasNextPage": has_next}}


def test_page_param_only_for_page_above_one(monkeypatch):
    calls = _live(monkeypatch, lambda p: _body(_rooms(3), 3, False))
    rooms_client.search_rooms(BANGALORE, TurnBudget())
    rooms_client.search_rooms(BANGALORE, TurnBudget(), page=1)
    rooms_client.search_rooms(BANGALORE, TurnBudget(), page=0)
    rooms_client.search_rooms(BANGALORE, TurnBudget(), page=3)
    assert ["page" in c for c in calls] == [False, False, False, True]
    assert calls[3]["page"] == 3


def test_live_result_carries_meta(monkeypatch):
    _live(monkeypatch, lambda p: _body(_rooms(10), 24, True))
    result = rooms_client.search_rooms(BANGALORE, TurnBudget())
    assert (result.total, result.has_next, result.page) == (24, True, 1)


def test_live_result_without_meta_is_none(monkeypatch):
    _live(monkeypatch, lambda p: {"success": True, "data": _rooms(2)})
    result = rooms_client.search_rooms(BANGALORE, TurnBudget())
    assert result.ok and result.total is None and result.has_next is None


# ---- offline fixture mode -------------------------------------------------------


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    path = tmp_path / "rooms25.json"
    path.write_text(json.dumps(_rooms(25)), encoding="utf-8")
    monkeypatch.setenv(config.OFFLINE_ROOMS_ENV, str(path))
    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)


def test_offline_meta_and_paging(synthetic):
    seen = []
    for page in (1, 2, 3, 4):
        r = rooms_client.search_rooms(BANGALORE, TurnBudget(), page=page)
        seen.append((len(r.rooms), r.total, r.has_next, r.rooms[0]["id"] if r.rooms else None))
    assert seen == [(10, 25, True, "0"), (10, 25, True, "10"), (5, 25, False, "20"), (0, 25, False, None)]


def test_offline_exact_multiple_has_no_next(monkeypatch, tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_rooms(20)), encoding="utf-8")
    monkeypatch.setenv(config.OFFLINE_ROOMS_ENV, str(path))
    assert rooms_client.search_rooms(BANGALORE, TurnBudget(), page=2).has_next is False


# ---- actions --------------------------------------------------------------------


def _run(state, **kw):
    return search._run_search(dict(BANGALORE), state, TurnBudget(), "new_search", **kw)


def _show_more(state, budget=None):
    return search.handle_show_more(Decision(intent=Intent.SHOW_MORE), {}, state, budget or TurnBudget())


def test_first_page_mentions_total_and_does_not_claim_all(synthetic):
    state = LLMConversationState()
    r = _run(state)
    assert "showing 10 of 25" in r.reply and "show more" in r.reply
    assert "all the matching" not in r.reply
    assert (state.search_page, state.has_next, state.total_rooms) == (1, True, 25)


def test_show_more_walks_the_pages(synthetic):
    state = LLMConversationState()
    _run(state)
    r2 = _show_more(state)
    assert r2.action == "show_more" and r2.tool_called
    assert r2.reply.startswith("Here are more rooms (page 2)")
    assert [x["id"] for x in state.rooms_on_screen] == [str(i) for i in range(10, 20)]
    assert r2.reply.splitlines()[1].startswith("1. Room 10")  # numbering matches the screen
    assert state.search_page == 2 and state.has_next is True
    r3 = _show_more(state)
    assert r3.reply.startswith("Here are more rooms (page 3)")
    assert len(state.rooms_on_screen) == 5
    assert state.search_page == 3 and state.has_next is False
    assert "There are more rooms" not in r3.reply
    r4 = _show_more(state)  # no more: honest, no tool call
    assert r4.action == "show_more_all" and not r4.tool_called
    assert r4.reply.startswith("These are all the matching rooms")


def test_all_matching_never_said_while_has_next(synthetic):
    state = LLMConversationState()
    replies = [_run(state).reply]
    while state.has_next:
        replies.append(_show_more(state).reply)
        assert state.has_next is not True or "all the matching" not in replies[-1]
    assert not any("all the matching" in r for r in replies)


def test_show_more_is_one_tool_call(synthetic):
    state = LLMConversationState()
    _run(state)
    budget = TurnBudget()
    _show_more(state, budget)
    assert budget.tool_calls == 1


def test_new_search_resets_the_page(synthetic):
    state = LLMConversationState()
    _run(state)
    _show_more(state)
    assert state.search_page == 2
    _run(state)
    assert state.search_page == 1 and state.has_next is True


def test_show_more_api_failure_keeps_previous_rooms(synthetic, monkeypatch):
    state = LLMConversationState()
    _run(state)
    before = list(state.rooms_on_screen)

    def failing(search_, budget, **kw):
        budget.use_tool_call()
        return rooms_client.RoomsResult(ok=False, error_class="timeout")

    monkeypatch.setattr(rooms_client, "search_rooms", failing)
    r = _show_more(state)
    assert r.reply == rooms_client.UNAVAILABLE_MESSAGE and r.rooms is None
    assert state.rooms_on_screen == before
    assert (state.search_page, state.has_next) == (1, True)  # a retry asks for page 2 again


def test_missing_meta_keeps_the_older_rule(monkeypatch):
    _live(monkeypatch, lambda p: {"success": True, "data": _rooms(4)})
    state = LLMConversationState()
    _run(state)
    assert state.has_next is None
    assert _show_more(state).action == "show_more_all"  # 4 < fetch limit, nothing cut


def test_has_next_false_with_cut_matches_is_not_complete(monkeypatch):
    # Area search: 15 match, 10 shown, the API says there is no next page.
    _live(monkeypatch, lambda p: _body(_rooms(15), 15, False))
    state = LLMConversationState()
    search._run_search(dict(AREA), state, TurnBudget(), "refinement")
    assert state.rooms_truncated and _show_more(state).action == "show_more_limit"


def test_area_search_with_next_page_is_honest(monkeypatch):
    calls = _live(monkeypatch, lambda p: _body(_rooms(30, "Adugodi"), 80, True))
    state = LLMConversationState()
    budget = TurnBudget()
    r = search._run_search(dict(AREA), state, budget, "refinement")
    assert budget.tool_calls == 1 and len(calls) == 1  # no extra calls
    assert "couldn't find" not in r.reply
    assert "first 30 rooms of Bangalore" in r.reply and "budget" in r.reply and "room type" in r.reply
    assert state.has_next is True


def test_area_search_with_matches_and_next_page_adds_a_note(monkeypatch):
    _live(monkeypatch, lambda p: _body(_rooms(30), 80, True))
    state = LLMConversationState()
    r = search._run_search(dict(AREA), state, TurnBudget(), "refinement")
    assert "only searched the first 30 rooms of Bangalore" in r.reply
    assert "of 80" not in r.reply  # the total is the city's, not the area's


def test_area_search_without_next_page_keeps_no_area_wording(monkeypatch):
    _live(monkeypatch, lambda p: _body(_rooms(5, "Adugodi"), 5, False))
    state = LLMConversationState()
    r = search._run_search(dict(AREA), state, TurnBudget(), "refinement")
    assert r.reply.startswith("I couldn't find rooms in Koramangala, Bangalore")


def test_show_more_empty_page_keeps_rooms_on_screen(monkeypatch):
    pages = {1: _body(_rooms(30), 90, True), 2: _body(_rooms(30, "Adugodi"), 90, True)}
    _live(monkeypatch, lambda p: pages[p.get("page", 1)])
    state = LLMConversationState()
    search._run_search(dict(AREA), state, TurnBudget(), "refinement")
    before = list(state.rooms_on_screen)
    r = _show_more(state)
    assert state.rooms_on_screen == before and r.rooms is None
    assert state.search_page == 2 and state.has_next is True
    assert "none matched Koramangala" in r.reply
