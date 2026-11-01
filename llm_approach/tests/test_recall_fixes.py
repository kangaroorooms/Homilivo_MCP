"""Recall bugs from the live test: two facts in one question (P1), the clarifier loop (P2),
the facts acknowledgement nudge (P3), and the narrow recall safety net.

Offline, scripted LLM. The transcript scripts below are the decisions the live model could
plausibly have returned for "what's my name and age ?" (it was answered with the generic clarifier).
"""

from __future__ import annotations

import pytest

from llm_approach.actions import clarify, recall
from llm_approach.eval import datasets
from llm_approach.eval.run_safety import GENERAL_DATASETS
from llm_approach.schema import parse_decision

from .conftest import dec


def rc(**extra):
    d = {"intent": "CONVERSATION_RECALL", "confidence": 0.95}
    d.update(extra)
    return d


def start(pipe):
    """Rooms on screen, then the first live message: name and age given."""
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    d = dec("GENERAL_CHAT", reply="Nice to meet you!")
    d["user_facts"] = {"name": "ziyad", "age": 20}
    return pipe.turn("actually my name is ziyad and i'm 20 years old", d)


# ---- P3: acknowledgement ----------------------------------------------------------------------


def test_p3_acknowledgement_has_no_rooms_nudge(pipe):
    r = start(pipe)
    assert r.reply == "Got it, Ziyad!"
    assert "rooms" not in r.reply and "search" not in r.reply
    assert pipe.state.get_fact("name") == "Ziyad" and pipe.state.get_fact("age") == 20


def test_p3_other_acknowledgements_are_short(pipe):
    d = dec("GENERAL_CHAT", reply="Hi")
    d["user_facts"] = {"name": "Faizal"}
    assert pipe.turn("my name is Faizal", d).reply == "Got it, Faizal!"
    d["user_facts"] = {"name": "Ravi"}
    assert pipe.turn("actually my name is Ravi", d).reply == "Okay, I'll call you Ravi from now on."
    d = dec("GENERAL_CHAT", reply="Hi")
    d["user_facts"] = {"occupation": "teacher"}
    assert pipe.turn("I'm a teacher", d).reply == "Got it, I'll remember that."


def test_p3_mixed_message_still_searches(pipe):
    d = dec("NEW_SEARCH", city="Pune")
    d["user_facts"] = {"name": "Ziyad"}
    r = pipe.turn("I'm Ziyad, rooms in pune", d)
    assert r.metrics["tool_called"] is True and pipe.state.get_fact("name") == "Ziyad"


# ---- P1: several facts in one question --------------------------------------------------------


def test_schema_parses_recall_targets():
    d = parse_decision(rc(recall_targets=["Name", "age", "bogus", "first_query", "nth_query"]))
    assert d.recall_targets == ["name", "age", "first_query"]  # nth_query needs an index: single only
    assert parse_decision(rc(recall_target=["name", "age"])).recall_targets == ["name", "age"]
    assert parse_decision(rc(recall_target="fact", recall_key="occupation", recall_targets=["fact"])
                          ).recall_targets == ["occupation"]
    assert parse_decision(rc(recall_target="name")).recall_targets == []  # single form still works


@pytest.mark.parametrize("script", [
    rc(recall_targets=["name", "age"]),
    rc(recall_target="name", recall_targets=["name", "age"]),
    rc(recall_target=["name", "age"]),
    rc(recall_targets=["name", "age"], confidence=0.4),  # even a low confidence recall is answered
    rc(recall_targets=["name", "age"], needs_clarification=True, clarifying_question="Which one?"),
])
def test_p1_name_and_age_in_one_sentence(pipe, script):
    start(pipe)
    r = pipe.turn("what's my name and age ?", script)
    assert r.reply == "Your name is Ziyad and you're 20."
    assert r.metrics["tool_called"] is False and r.metrics["action"].startswith("recall")


def test_p1_partly_unknown_and_none_known(pipe):
    d = dec("GENERAL_CHAT", reply="Hi")
    d["user_facts"] = {"name": "Ziyad"}
    pipe.turn("my name is Ziyad", d)
    assert pipe.turn("what's my name and age?", rc(recall_targets=["name", "age"])).reply == \
        "Your name is Ziyad, but you haven't told me your age in this chat."
    pipe.sid = __import__("uuid").uuid4().hex and str(__import__("uuid").uuid4())
    assert pipe.turn("what's my name and age?", rc(recall_targets=["name", "age"])).reply == \
        "You haven't told me that in this chat."


def test_p1_first_query_and_name_together(pipe):
    d = dec("NEW_SEARCH", city="Pune")
    d["user_facts"] = {"name": "Ziyad"}
    pipe.turn("I'm Ziyad, rooms in pune", d)
    r = pipe.turn("what was my first query and my name?", rc(recall_targets=["first_query", "name"]))
    assert r.reply == 'Your name is Ziyad. Your first question was: "I\'m Ziyad, rooms in pune".'


def test_p1_three_facts_use_commas(pipe):
    d = dec("GENERAL_CHAT", reply="Hi")
    d["user_facts"] = {"name": "Ziyad", "age": 20, "occupation": "student"}
    pipe.turn("I'm Ziyad, 20, a student", d)
    r = pipe.turn("tell me my name, age and job", rc(recall_targets=["name", "age", "occupation"]))
    assert r.reply == "Your name is Ziyad, you're 20 and your occupation is student."


# ---- narrow recall safety net -------------------------------------------------------------------


@pytest.mark.parametrize("script", [
    {"intent": "UNCLEAR", "confidence": 0.3, "needs_clarification": True, "clarifying_question": "Which?"},
    {"intent": "UNCLEAR", "confidence": 0.9},
    {"intent": "OUT_OF_SCOPE", "confidence": 0.3, "reply": "No."},
    {"intent": "GENERAL_CHAT", "confidence": 0.2, "reply": "Hello"},
    rc(confidence=0.3),  # a recall decision without any target
])
def test_safety_net_routes_to_recall_by_message_cue(pipe, script):
    start(pipe)
    r = pipe.turn("what's my name and age ?", script)
    assert r.reply == "Your name is Ziyad and you're 20."
    assert pipe.state.user_queries[-1]["intent"] == "CONVERSATION_RECALL"


def test_safety_net_does_not_touch_a_confident_non_recall_decision(pipe):
    start(pipe)
    r = pipe.turn("what's my name and age ?", dec("GENERAL_CHAT", reply="Hello there"))
    assert "Your name is" not in r.reply


@pytest.mark.parametrize("message,expected", [
    ("what's my name and age ?", ["name", "age"]),
    ("what is my name", ["name"]),
    ("how old am i?", ["age"]),
    ("what was my first query?", ["first_query"]),
    ("what was my first message regarding rooms", ["first_room_query"]),
    ("what was my last question", ["last_query"]),
    ("do you remember my name and my first question?", ["name", "first_query"]),
    ("what did i say?", []),
    ("i told you my name", ["name"]),
])
def test_cue_detection(message, expected):
    assert recall.detect_recall_cue(message) == expected


@pytest.mark.parametrize("message", [
    "find me a room in bangalore", "my name is Faizal", "my age is 23", "I'm 23 and need a 2 BHK",
    "2bhk under 15k", "what is the speed of light", "what's my bank balance", "my budget is 15k",
    "which one is cheapest", "tell me about the first one", "what is the first room", "how old is the building",
    "I told you I want pune", "what did the owner say", "my first choice is koramangala",
])
def test_cue_detection_ignores_other_messages(message):
    assert recall.detect_recall_cue(message) is None


def test_no_benign_or_eval_message_routes_to_recall():
    """The 144 benign safety messages and all 186 eval items: none may hit the recall cue."""
    seen, hits = 0, []
    for name in GENERAL_DATASETS + ["golden", "golden_slots"]:
        for item in datasets.load_dataset(name):
            text = item.get("text") or item.get("message") or item.get("query") or ""
            seen += 1
            if text and recall.detect_recall_cue(text) is not None:
                hits.append((name, item.get("item") or item.get("id"), text))
    assert seen >= 186
    assert hits == []


# ---- P2: the clarifier loop ---------------------------------------------------------------------


def test_p2_yes_loop_ends_in_help(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    first = pipe.turn("hmm", {"intent": "UNCLEAR", "confidence": 0.3})
    assert first.reply == clarify.ASK_AFTER_RESULTS
    replies = [pipe.turn("yes", {"intent": "UNCLEAR", "confidence": 0.3}).reply for _ in range(3)]
    assert replies[0] == clarify.HELP_WITH_ROOMS
    assert all(replies[i] != replies[i - 1] for i in range(1, 3)) or replies[1] == clarify.HELP_SHORT
    assert clarify.ASK_AFTER_RESULTS not in replies
    chain = [first.reply] + replies
    # never the identical question twice in a row
    assert all(chain[i] != chain[i + 1] or chain[i] in (clarify.HELP_SHORT,) for i in range(len(chain) - 1))


@pytest.mark.parametrize("word", ["yes", "no", "ok", "okay", "yeah", "sure", "haan", "nahi", "Yes!", " ok. "])
def test_p2_bare_reply_to_generic_question_gets_help(pipe, word):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    pipe.turn("hmm", {"intent": "UNCLEAR", "confidence": 0.3})
    r = pipe.turn(word, dec("GENERAL_CHAT", reply="Okay!"))  # the model may call a bare yes small talk
    assert r.reply == clarify.HELP_WITH_ROOMS


def test_p2_identical_question_never_repeats(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    a = pipe.turn("umm", {"intent": "UNCLEAR", "confidence": 0.3}).reply
    b = pipe.turn("errr", {"intent": "UNCLEAR", "confidence": 0.3}).reply
    assert a == clarify.ASK_AFTER_RESULTS and b != a and "remember" in b


def test_p2_help_only_mentions_rooms_shown_when_on_screen(pipe):
    r = pipe.turn("hmm", {"intent": "UNCLEAR", "confidence": 0.3})
    assert r.reply == clarify.ASK_CITY_BUDGET
    r = pipe.turn("yes", {"intent": "UNCLEAR", "confidence": 0.3})
    assert r.reply == clarify.HELP_NO_ROOMS and "shown" not in r.reply and "these rooms" not in r.reply
    assert "remember" in r.reply


def test_p2_model_question_about_rooms_is_replaced_when_none_on_screen(pipe):
    r = pipe.turn("hmm", {"intent": "UNCLEAR", "confidence": 0.4, "needs_clarification": True,
                          "clarifying_question": "Would you like to ask about these rooms?"})
    assert "these rooms" not in r.reply


def test_p2_pending_city_question_is_still_asked_again(pipe):
    pipe.turn("find me a room", dec("NEW_SEARCH", nc=True, conf=0.85, q="Which city are you looking for?"))
    r = pipe.turn("i dont know", {"intent": "UNCLEAR", "confidence": 0.4})
    assert r.reply == clarify.ASK_CITY


def test_p2_help_makes_no_search_and_keeps_state(pipe):
    pipe.turn("rooms in bangalore", dec("NEW_SEARCH", city="Bangalore"))
    searches = len(pipe.searches)
    pipe.turn("hmm", {"intent": "UNCLEAR", "confidence": 0.3})
    r = pipe.turn("yes", {"intent": "UNCLEAR", "confidence": 0.3})
    assert len(pipe.searches) == searches and len(pipe.state.rooms_on_screen) == 10 and r.rooms is None
