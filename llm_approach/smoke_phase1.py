"""Live smoke test for the Phase 1 decision layer: exactly 3 LLM calls (the active provider).

Run from the repo root:  python -m llm_approach.smoke_phase1
Needs the active provider's key (GEMINI_API_KEY, or GROQ_API_KEY with LLM_APPROACH_PROVIDER=groq)
in the environment or in .env, which is loaded here.
Prints input, intent, operation, city, confidence, latency and tokens only.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from llm_approach import config, prompts
from llm_approach.llm_client import LLMBudget, decide
from llm_approach.validators import validate_city

SPACING_S = 60
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rooms_bangalore.json"

# (message, rooms on screen?, current search, expected intent, expected city)
CASES = [
    ("find me a room in bangalore", False, None, "NEW_SEARCH", "Bangalore"),
    ("in udupi", True, {"city": "Bangalore"}, "REFINEMENT", "Udupi"),
    ("none of these are good", True, {"city": "Bangalore"}, "REJECT_CURRENT", None),
]


def _load_rooms() -> list[dict]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["rooms"] if isinstance(data, dict) else data


async def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    problem = config.setup_problem()
    if problem:
        print(f"{problem} Nothing was called.")
        return 2
    spacing = SPACING_S if config.provider() == "groq" else config.spacing_s()

    rooms = _load_rooms()
    failures = 0
    print(f"{'input':<30} {'intent':<15} {'operation':<10} {'city':<10} {'conf':<5} {'ms':<6} {'tokens'}")
    for n, (message, with_rooms, search, want_intent, want_city) in enumerate(CASES):
        if n:
            await asyncio.sleep(spacing)  # Groq: stay under its tokens-per-minute limit; Gemini: from the environment
        state = {
            "current_search": search,
            "last_bot_message": "Here are the rooms I found." if with_rooms else None,
            "rooms_on_screen": rooms if with_rooms else [],
        }
        budget = LLMBudget(max_calls=1)  # one live call per case, no retry
        decision = await decide(message, prompts.build_state_summary(state), budget)
        if decision is None:
            print(f"{message:<30} INVALID/NONE")
            failures += 1
            continue
        city = validate_city(decision.slots.city)
        ok = decision.intent.value == want_intent and city == want_city
        failures += not ok
        print(
            f"{message:<30} {decision.intent.value:<15} {str(decision.operation and decision.operation.value):<10} "
            f"{str(city):<10} {decision.confidence:<5.2f} {budget.latency_ms:<6.0f} "
            f"{budget.tokens_in}+{budget.tokens_out}  {'OK' if ok else 'MISMATCH'}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
