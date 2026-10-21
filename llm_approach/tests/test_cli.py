"""CLI tests with a mocked pipeline. No Groq, no guardrails, no input() blocking."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_approach import cli, config, pipeline


def make_turn_fn(replies):
    calls = []

    async def turn(session_id, message):
        calls.append((session_id, message))
        reply, rooms, debug = replies.pop(0)
        return SimpleNamespace(
            reply=reply, rooms=rooms, success=True, debug=debug,
            metrics={"action": "x", "tool_called": False, "fallback_used": False},
        )

    turn.calls = calls
    return turn


def chat(inputs, turn_fn=None, **kw):
    lines = iter(inputs)
    out: list[str] = []

    def fake_input(prompt):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError

    cli.run_chat(turn_fn=turn_fn or make_turn_fn([]), input_fn=fake_input, print_fn=lambda *a: out.append(" ".join(map(str, a))), **kw)
    return out


CARDS = [
    {"id": "1", "title": "Budget Single Room", "room_type": "Single", "price_per_month": 7000, "location": "Koramangala"},
    {"id": "2", "title": "Cozy 2BHK", "room_type": "2BHK", "price_per_month": 15000, "location": None},
]


def test_quit_stops_before_any_turn():
    turn = make_turn_fn([])
    out = chat(["/quit", "never reached"], turn)
    assert turn.calls == [] and out == [cli.HELP]


def test_message_goes_to_the_pipeline_and_reply_is_printed():
    turn = make_turn_fn([("Hello there", None, None)])
    out = chat(["hi", "/quit"], turn)
    assert turn.calls[0][1] == "hi" and "bot> Hello there" in out


def test_same_session_id_for_the_whole_chat():
    turn = make_turn_fn([("a", None, None), ("b", None, None)])
    chat(["one", "two"], turn)
    assert turn.calls[0][0] == turn.calls[1][0]


def test_state_command_prints_state_json():
    state = {"current_search": {"city": "Pune"}, "rooms_on_screen": [], "selected_room_index": None}
    seen = []
    out = chat(["/state", "/quit"], state_fn=lambda sid: (seen.append(sid), state)[1])
    assert json.loads(out[1]) == state and len(seen) == 1


def test_state_command_without_a_session():
    out = chat(["/state"], state_fn=lambda sid: None)
    assert "(no session yet)" in out


def test_reset_command_resets_and_starts_a_new_session():
    turn = make_turn_fn([("a", None, None), ("b", None, None)])
    reset = []
    out = chat(["one", "/reset", "two"], turn, reset_fn=reset.append)
    assert "Conversation reset." in out
    assert reset == [turn.calls[0][0]]
    assert turn.calls[0][0] != turn.calls[1][0]


def test_debug_toggle_and_output():
    debug = {"intent": "NEW_SEARCH", "operation": None, "slots": {"city": "Pune", "bhk": None},
             "validated": {"city": "Pune"}, "rejected": [], "room_reference": None, "confidence": 0.9,
             "needs_clarification": False}
    turn = make_turn_fn([("r1", None, debug), ("r2", None, debug), ("r3", None, debug)])
    out = chat(["one", "/debug on", "two", "/debug off", "three"], turn)
    debug_lines = [line for chunk in out for line in chunk.splitlines() if line.startswith("[debug]")]
    assert len(debug_lines) == 2  # two lines, printed for the middle turn only
    assert "intent=NEW_SEARCH" in debug_lines[0] and "confidence=0.90" in debug_lines[0]
    assert '"city": "Pune"' in debug_lines[1] and "bhk" not in debug_lines[1]


def test_debug_flag_starts_debug_on():
    turn = make_turn_fn([("r", None, None)])
    out = chat(["hi"], turn, debug=True)
    assert any(line.startswith("[debug] no decision") for line in out)


def test_unknown_command_shows_help_and_does_not_call_the_pipeline():
    turn = make_turn_fn([])
    out = chat(["/bogus", "/debug maybe"], turn)
    assert out.count(cli.HELP) == 3 and turn.calls == []


def test_blank_lines_are_ignored_and_eof_ends_the_chat():
    turn = make_turn_fn([])
    chat(["", "   "], turn)
    assert turn.calls == []


def test_room_cards_print_as_a_table():
    turn = make_turn_fn([("Found 2", CARDS, None)])
    out = [line for chunk in chat(["rooms"], turn) for line in chunk.splitlines()]
    table = next(line for line in out if "Budget Single Room" in line and "Title" not in line)
    assert "Rs 7,000" in table and "Koramangala" in table
    header = next(line for line in out if line.startswith("# "))
    assert "Title" in header and "Price/month" in header
    assert any("Cozy 2BHK" in line and "Rs 15,000" in line and "-" in line for line in out)


def test_format_rooms_empty():
    assert cli.format_rooms([]) == ""


def test_parse_args_flags():
    args = cli.parse_args(["--offline-rooms", "--debug", "--skip-guardrail-load"])
    assert args.offline_rooms and args.debug and args.skip_guardrail_load
    args = cli.parse_args([])
    assert not (args.offline_rooms or args.debug or args.skip_guardrail_load)


@pytest.fixture
def main_env(monkeypatch):
    """main() with .env loading, the chat loop and the guardrail load mocked."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test-secret-value-123")  # Gemini is the default provider
    monkeypatch.setenv(config.OFFLINE_ROOMS_ENV, "")  # so the test's env is restored afterwards
    ran = SimpleNamespace(preload=0, chat=None, configured=None)
    monkeypatch.setattr(pipeline, "preload", lambda: setattr(ran, "preload", ran.preload + 1))
    monkeypatch.setattr(pipeline, "configure", lambda **kw: setattr(ran, "configured", kw))
    monkeypatch.setattr(cli, "run_chat", lambda **kw: setattr(ran, "chat", kw))
    yield ran
    pipeline.configure(full_guardrails=True)


def test_main_preloads_guardrails_by_default(main_env, capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "Loading guardrails (can take ~40 s)..." in out
    assert main_env.preload == 1 and main_env.configured is None
    assert "sk-test-secret-value-123" not in out


def test_main_skip_flag_warns_and_does_not_preload(main_env, capsys):
    assert cli.main(["--skip-guardrail-load", "--offline-rooms", "--debug"]) == 0
    out = capsys.readouterr().out
    assert "NOT the full safety stack" in out and "evaluation" in out
    assert main_env.preload == 0 and main_env.configured == {"full_guardrails": False}
    assert main_env.chat == {"debug": True}
    assert "Loading guardrails" not in out
    import os

    assert os.environ[config.OFFLINE_ROOMS_ENV] == "1"


def test_main_without_a_key_exits_and_prints_no_values(monkeypatch, capsys):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert cli.main([]) == 2
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().out
