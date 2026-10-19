"""CONVERSATION_RECALL and fact capture: answers come from the session state, never from the model.

* ``capture_facts`` runs as a side effect on every intent (before the main action): the facts the
  model extracted are validated, must appear in the user's message, and are stored in
  ``state.user_facts``. A new value overwrites the old one (a correction).
* ``handle_recall`` answers a recall question from ``state.user_facts`` / ``state.user_queries``
  with fixed templates. One question may ask for several things ("what's my name and age"):
  they are answered in one reply. The only free text in a reply is a past query, which goes
  through ``sanitize.neutralize`` and the injection filter, and the pipeline's output guard like
  any reply.
* ``acknowledge`` is the fixed reply for a chat message that only gave facts ("my name is Faizal").
* ``detect_recall_cue`` is the narrow safety net: it finds personal-recall questions by wording,
  for the case the model labels one UNCLEAR or is unsure about it.

The current message is already the last entry of ``user_queries`` when these run, so "earlier"
means every entry with a smaller turn number.
"""

from __future__ import annotations

import re
from typing import Any

from llm_approach import extra_security, validators
from llm_approach.actions import ActionResult, TurnBudget
from llm_approach.sanitize import neutralize
from llm_approach.schema import Decision, RecallTarget
from llm_approach.state import QUERY_BLOCKED, LLMConversationState

ROOM_INTENTS = frozenset({"NEW_SEARCH", "REFINEMENT"})
RECALL_INTENT = "CONVERSATION_RECALL"
QUOTE_MAX_CHARS = 200
ALL_QUERIES_SHOWN = 5

ASK_WHAT_TO_RECALL = "What would you like me to recall: your name, your age, or your earlier questions?"
ASK_WHICH_DETAIL = "Which detail do you mean: your name, age, occupation, gender, preferred city or budget?"
NOTHING_EARLIER = "This is your first message in this chat, so there's nothing earlier yet."
NO_ROOM_SEARCH = "You haven't searched for a room in this chat yet."
NO_OTHER_QUERY = "You haven't asked me anything else yet."
NOT_THAT_MANY = "You haven't sent that many messages yet."
FORGOTTEN = "I no longer have that message."
EARLIEST_FORGOTTEN = "I no longer have your earliest messages."
GOT_IT = "Got it, I'll remember that."
NOTHING_KNOWN = "You haven't told me that in this chat."

_LABELS = {
    "name": "name",
    "age": "age",
    "occupation": "occupation",
    "gender": "gender",
    "preferred_city": "preferred city",
    "budget": "budget",
}
_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def ordinal(n: int) -> str:
    return _ORDINALS[n - 1] if 1 <= n <= len(_ORDINALS) else f"{n}th"


# ---- fact capture ----------------------------------------------------------------------


def capture_facts(raw_facts: Any, message: str, state: LLMConversationState) -> dict[str, Any]:
    """Validate and store the facts from the decision. Returns ``{key: previous value or None}``
    for every fact stored (an unchanged value counts as stored)."""
    clean, _ = validators.validate_user_facts(raw_facts, message)
    stored: dict[str, Any] = {}
    for key, value in clean.items():
        previous = state.get_fact(key)
        if state.set_fact(key, value):
            stored[key] = previous
    return stored


def acknowledge(stored: dict[str, Any], state: LLMConversationState) -> ActionResult:
    """Fixed, short reply for a chat message that only told us facts. No question, no nudge."""
    name = state.get_fact("name")
    if "name" in stored and name:
        if stored["name"] is not None and stored["name"] != name:
            text = f"Okay, I'll call you {name} from now on."
        else:
            text = f"Got it, {name}!"
    else:
        text = GOT_IT
    return ActionResult(text, None, "facts_saved")


# ---- recall ----------------------------------------------------------------------------


def _quote(entry: dict[str, Any]) -> str | None:
    """A past message as a quoted string, or None when it must not be repeated."""
    if entry["intent"] == QUERY_BLOCKED:
        return None
    text = re.sub(r"\s+", " ", neutralize(str(entry["text"]))).strip().replace('"', "'")
    if not text or extra_security.is_extra_blocked(text)[0]:
        return None
    if len(text) > QUOTE_MAX_CHARS:
        text = text[: QUOTE_MAX_CHARS - 3].rstrip() + "..."
    return f'"{text}"'


def _say(label: str, entry: dict[str, Any], noun: str = "question") -> str:
    quoted = _quote(entry)
    if quoted is None:
        return f"Your {label} message couldn't be processed, so I can't repeat it."
    return f"Your {label} {noun} was: {quoted}."


def _format_fact(key: str, value: Any) -> str:
    value = neutralize(value) if isinstance(value, str) else value
    if key == "name":
        return f"Your name is {value}."
    if key == "age":
        return f"You told me you're {value}."
    if key == "budget":
        return f"Your budget is ₹{int(value):,}."
    return f"Your {_LABELS[key]} is {value}."


def _fact_answer(key: str, state: LLMConversationState) -> str:
    value = state.get_fact(key)
    if value is None:
        return f"You haven't told me your {_LABELS[key]} in this chat."
    return _format_fact(key, value)


def _earlier(state: LLMConversationState) -> list[dict[str, Any]]:
    return [q for q in state.user_queries if q["turn"] < state.query_count]


def _first_query(earlier: list[dict[str, Any]]) -> str:
    if not earlier:
        return NOTHING_EARLIER
    if earlier[0]["turn"] != 1:
        return EARLIEST_FORGOTTEN
    return _say("first", earlier[0])


def _first_room_query(earlier: list[dict[str, Any]]) -> str:
    found = next((q for q in earlier if q["intent"] in ROOM_INTENTS), None)
    return _say("first room", found, "search") if found else NO_ROOM_SEARCH


def _last_query(earlier: list[dict[str, Any]]) -> str:
    found = next((q for q in reversed(earlier) if q["intent"] != RECALL_INTENT), None)
    return _say("last", found) if found else NO_OTHER_QUERY


_QUERY_TOKENS = {"first_query": _first_query, "first_room_query": _first_room_query, "last_query": _last_query}


def _fact_phrase(key: str, value: Any) -> str:
    """A fragment that follows "Your name is Ziyad and ..." (lower case, no full stop)."""
    value = neutralize(value) if isinstance(value, str) else value
    if key == "name":
        return f"your name is {value}"
    if key == "age":
        return f"you're {value}"
    if key == "budget":
        return f"your budget is ₹{int(value):,}"
    return f"your {_LABELS[key]} is {value}"


def _join(parts: list[str], last: str = "and") -> str:
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} {last} {parts[-1]}"


def _answer_several(tokens: list[str], state: LLMConversationState) -> str:
    """One reply for several asks. Facts share one sentence; each query gets its own."""
    earlier = _earlier(state)
    keys = [t for t in tokens if t in _LABELS]
    known = [k for k in keys if state.get_fact(k) is not None]
    unknown = [_LABELS[k] for k in keys if state.get_fact(k) is None]
    sentences: list[str] = []
    if known:
        text = _join([_fact_phrase(k, state.get_fact(k)) for k in known])
        text = text[0].upper() + text[1:]
        if unknown:
            text += f", but you haven't told me your {_join(unknown, 'or')} in this chat"
        sentences.append(text + ".")
    elif keys:
        sentences.append(NOTHING_KNOWN)
    sentences += [_QUERY_TOKENS[t](earlier) for t in tokens if t in _QUERY_TOKENS]
    return " ".join(sentences)


def _single(decision: Decision, state: LLMConversationState) -> tuple[str, str]:
    """(reply, action suffix) for the one-target form."""
    target = decision.recall_target
    earlier = _earlier(state)
    if target is RecallTarget.NAME:
        return _fact_answer("name", state), "name"
    if target is RecallTarget.AGE:
        return _fact_answer("age", state), "age"
    if target is RecallTarget.FACT:
        key = decision.recall_key
        if key not in _LABELS:
            return ASK_WHICH_DETAIL, "fact_ask"
        return _fact_answer(key, state), "fact"
    if target is RecallTarget.FIRST_QUERY:
        return _first_query(earlier), "first_query"
    if target is RecallTarget.FIRST_ROOM_QUERY:
        return _first_room_query(earlier), "first_room_query"
    if target is RecallTarget.LAST_QUERY:
        return _last_query(earlier), "last_query"
    if target is RecallTarget.NTH_QUERY:
        n = decision.recall_index
        if n is None:
            return ASK_WHAT_TO_RECALL, "nth_query_ask"
        if n >= state.query_count:
            return NOT_THAT_MANY, "nth_query"
        found = next((q for q in earlier if q["turn"] == n), None)
        return (_say(ordinal(n), found) if found else FORGOTTEN), "nth_query"
    if target is RecallTarget.ALL_QUERIES:
        asked = [q for q in earlier if q["intent"] != RECALL_INTENT]
        if not asked:
            return NO_OTHER_QUERY, "all_queries"
        shown = asked[-ALL_QUERIES_SHOWN:]
        parts = [f"{i}) {_quote(q) or '(a message I could not process)'}" for i, q in enumerate(shown, 1)]
        lead = "So far you asked:" if len(shown) == len(asked) else f"Your last {len(shown)} questions:"
        return f"{lead} {' '.join(parts)}", "all_queries"
    return ASK_WHAT_TO_RECALL, "ask"


def _token_as_target(token: str) -> tuple[RecallTarget, str | None]:
    if token in ("name", "age", "first_query", "last_query", "first_room_query"):
        return RecallTarget(token), None
    return RecallTarget.FACT, token


def handle_recall(
    decision: Decision,
    validated: dict[str, Any],
    state: LLMConversationState,
    budget: TurnBudget,
    *,
    user_message: str | None = None,
) -> ActionResult:
    """Answer from state. Uses ``recall_targets`` (several asks) if the model gave them, else
    ``recall_target``; with neither, the targets are read from the message wording."""
    tokens = list(decision.recall_targets)
    if not tokens and decision.recall_target is None and user_message:
        tokens = detect_recall_cue(user_message) or []
    if len(tokens) >= 2:
        return ActionResult(_answer_several(tokens, state), None, "recall_several")
    if len(tokens) == 1:
        target, key = _token_as_target(tokens[0])
        single = decision.model_copy(update={"recall_target": target, "recall_key": key})
        reply, kind = _single(single, state)
        return ActionResult(reply, None, f"recall_{kind}")
    reply, kind = _single(decision, state)
    return ActionResult(reply, None, f"recall_{kind}")


# ---- safety net: personal-recall wording -----------------------------------------------

# The cue list is deliberately small. It only runs when the model said UNCLEAR or was unsure,
# and it needs BOTH a recall cue ("my name", "my age", "how old am I", "my first/last
# query|message|question", "what did I say|tell|ask", "I told you already|before|earlier")
# AND question wording (a "?", a wh-word, "tell me", "remember", "recall", "did I", or "told you").
# A statement ("my name is Faizal") or a room request never matches.
_ASKING = re.compile(
    r"\?|\b(?:what|what's|whats|which|who|how|tell me|remember|recall|did i|do you know|told you)\b",
    re.IGNORECASE,
)
_NAME = re.compile(r"\bmy name\b", re.IGNORECASE)
_AGE = re.compile(r"\bmy age\b|\bhow old am i\b", re.IGNORECASE)
_FIRST = re.compile(
    r"\bmy first\b(?P<mid>[^?.]{0,25}?)\b(?:quer(?:y|ies)|messages?|questions?|search(?:es)?)\b(?P<tail>[^?.]*)",
    re.IGNORECASE,
)
_LAST = re.compile(r"\bmy (?:last|previous|latest) (?:quer(?:y|ies)|messages?|questions?)\b", re.IGNORECASE)
_SAID = re.compile(
    r"\bwhat did i (?:say|tell|ask)\b|\bi (?:already )?told you (?:already|before|earlier|just now)\b",
    re.IGNORECASE,
)


def detect_recall_cue(message: Any) -> list[str] | None:
    """``None`` when the message is not a personal-recall question. Otherwise the targets it
    asks for (RECALL_TOKENS, in a fixed order); ``[]`` for "what did I say" with no target."""
    if not isinstance(message, str) or not _ASKING.search(message):
        return None
    targets: list[str] = []
    if _NAME.search(message) or _AGE.search(message):
        # "my name and age": once one of them is cued, the other word counts as asked too
        if re.search(r"\bname\b", message, re.IGNORECASE):
            targets.append("name")
        if re.search(r"\bage\b|\bhow old am i\b", message, re.IGNORECASE):
            targets.append("age")
    first = _FIRST.search(message)
    if first:
        about_rooms = re.search(r"\b(?:rooms?|search(?:es)?|accommodation|flats?)\b", first.group("mid") + first.group("tail"), re.I)
        targets.append("first_room_query" if about_rooms or "search" in first.group(0).lower() else "first_query")
    if _LAST.search(message):
        targets.append("last_query")
    if targets or _SAID.search(message):
        return targets
    return None
