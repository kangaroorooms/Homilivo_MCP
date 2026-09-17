import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FILE_SEARCH_STORE = os.getenv("GEMINI_FILE_SEARCH_STORE")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not FILE_SEARCH_STORE:
    raise RuntimeError("GEMINI_FILE_SEARCH_STORE is missing")


client = genai.Client(api_key=GEMINI_API_KEY)


SYSTEM_INSTRUCTION = """
You are the Homilivo customer support chatbot.

Your job is to answer user questions using the Homilivo knowledge base.

Rules:
- Give simple, natural, customer-friendly answers.
- Answer only with information supported by the Homilivo knowledge base.
- Do not invent Homilivo policies, pricing, features, permissions, or workflows.
- If the information is not available in the knowledge base, clearly say that
  the information is not currently available.
- Do not mention File Search, embeddings, retrieval, internal systems, or
  implementation details to the user.
- Keep normal answers concise.
- Give more detail when the user asks for an explanation.
- Do not repeat the same information unnecessarily.
""".strip()


def chat_with_homilivo(
    message: str,
    previous_interaction_id: str | None = None,
) -> dict[str, Any]:

    interaction_kwargs = {
    "model": "gemini-3.5-flash-lite",
    "system_instruction": SYSTEM_INSTRUCTION,
    "input": message,
    "tools": [
        {
            "type": "file_search",
            "file_search_store_names": [FILE_SEARCH_STORE],
        }
    ],
    "extra_body": {
        "generation_config": {
            "thinking_level": "low"
        }
    },
}

    if previous_interaction_id:
        interaction_kwargs["previous_interaction_id"] = previous_interaction_id

    # Measure Gemini + File Search response time
    start_time = time.perf_counter()

    interaction = client.interactions.create(**interaction_kwargs)

    elapsed_time = time.perf_counter() - start_time

    print(f"Gemini response time: {elapsed_time:.2f} seconds")

    sources: list[dict[str, str]] = []

    for step in interaction.steps:
        if step.type != "model_output":
            continue

        for content in step.content:
            annotations = getattr(content, "annotations", None)

            if not annotations:
                continue

            for annotation in annotations:
                if annotation.type == "file_citation":
                    source = {
                        "file": annotation.file_name,
                    }

                    if source not in sources:
                        sources.append(source)

    return {
        "answer": interaction.output_text,
        "interaction_id": interaction.id,
        "sources": sources,
    }