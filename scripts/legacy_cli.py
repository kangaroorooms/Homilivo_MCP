"""
Standalone manual-debug CLI - NOT used by the FastAPI server.

Run directly with `python -m app.agent` for an interactive, single-turn
terminal walkthrough of the guardrails -> fast_parser -> MCP ->
Homilivo API -> guardrails pipeline (prompts for one query, one response
mode, and prints each pipeline stage as it runs). Nothing in app/api.py
or app/experiments/agent_modes.py (the real conversational engine the
server uses) imports this module.

Kept for manual/local debugging of the pipeline in isolation; it has no
conversational memory, follow-up handling, or session support like
app/experiments/agent_modes.process_message() does. Do not add new
server-facing logic here - extend agent_modes.py instead.
"""

import asyncio
import json
import sys

# See app/api.py for why: Windows stdout defaults to cp1252 whenever it
# isn't an interactive UTF-8 console, which crashes on the Unicode
# symbols (arrows, checkmarks, rupee sign) this module prints.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from ollama import chat
from mcp import Client

from app.server import mcp
from app.fast_parser import extract_search_parameters

from app.guardrails import (
    validate_user_input,
    validate_search_arguments,
    validate_room_data,
    validate_ai_response,
)


# ============================================================
# DISPLAY HELPER
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================
# NON-SEARCH INTENT RESPONSES
# ============================================================

GREETING_RESPONSE = (
    "Hello! I'm the Homilivo room-search assistant. "
    "I can help you find rooms and accommodation based "
    "on city, budget, room type and other preferences."
)


CHATBOT_INFO_RESPONSE = (
    "I'm the Homilivo room-search assistant. "
    "I can help you find rooms and accommodation "
    "based on criteria such as city, budget, room type "
    "and who the accommodation is suitable for."
)


# ============================================================
# FAST PYTHON RESPONSE
# ============================================================

def generate_fast_response(
    user_query: str,
    rooms: list,
    search_request: dict,
) -> str:

    if not rooms:

        city = search_request.get("city")
        max_price = search_request.get("max_price")

        if city and max_price:
            return (
                f"I couldn't find any matching rooms "
                f"in {city} under ₹{max_price:,}."
            )

        if city:
            return (
                f"I couldn't find any matching rooms "
                f"in {city}."
            )

        return "I couldn't find any matching rooms."

    city = search_request.get("city")
    max_price = search_request.get("max_price")
    ideal_for = search_request.get("ideal_for", [])

    response_lines = []

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if city and max_price:

        response_lines.append(
            f"I found {len(rooms)} room"
            f"{'s' if len(rooms) != 1 else ''} "
            f"in {city} under ₹{max_price:,}."
        )

    elif city:

        response_lines.append(
            f"I found {len(rooms)} room"
            f"{'s' if len(rooms) != 1 else ''} "
            f"in {city}."
        )

    else:

        response_lines.append(
            f"I found {len(rooms)} matching room"
            f"{'s' if len(rooms) != 1 else ''}."
        )

    if ideal_for:

        response_lines.append(
            f"Suitable for: {', '.join(ideal_for)}."
        )

    response_lines.append("")

    # --------------------------------------------------------
    # ROOM DETAILS
    # --------------------------------------------------------

    for index, room in enumerate(
        rooms,
        start=1,
    ):

        title = room.get(
            "title",
            "Untitled Room",
        )

        room_city = room.get(
            "city",
            "Unknown",
        )

        price = room.get(
            "pricePerMonth"
        )

        if price is None:
            price = room.get("price")

        room_type = room.get(
            "roomType",
            "Not specified",
        )

        room_ideal_for = room.get(
            "idealFor",
            [],
        )

        response_lines.append(
            f"{index}. {title}"
        )

        if price is not None:

            try:

                response_lines.append(
                    f"   ₹{float(price):,.0f}/month"
                )

            except (
                TypeError,
                ValueError,
            ):

                response_lines.append(
                    f"   Monthly Rent: {price}"
                )

        response_lines.append(
            f"   City: {str(room_city).title()}"
        )

        response_lines.append(
            f"   Type: {room_type}"
        )

        if room_ideal_for:

            response_lines.append(
                "   Ideal for: "
                + ", ".join(
                    str(x)
                    for x in room_ideal_for
                )
            )

        response_lines.append("")

    return "\n".join(
        response_lines
    ).strip()


# ============================================================
# OPTIONAL QWEN3 RESPONSE
# ============================================================

def generate_ai_response(
    user_query: str,
    rooms: list,
) -> str:

    room_context = json.dumps(
        rooms,
        indent=2,
        ensure_ascii=False,
    )

    system_prompt = """
You are a concise room-search assistant.

The room data below comes directly from
the MCP room-search tool.

Use ONLY the supplied room data.

Never invent:
- prices
- locations
- amenities
- room types
- availability
- facilities
- ratings
- descriptions

Keep the response concise.

Mention relevant room title,
city, monthly price, room type,
and ideal-for information when available.
"""

    user_prompt = f"""
USER REQUEST:
{user_query}

VERIFIED ROOM DATA:
{room_context}

Give a concise natural-language answer
based only on the verified room data.
"""

    response = chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.message.content


# ============================================================
# EXTRACT ROOMS FROM MCP RESPONSE
# ============================================================

def extract_rooms_from_response(
    api_data: dict,
) -> list:

    """
    Handles response structures returned by MCP.

    Supported:

    1. {"success": true, "rooms": [...]}

    2. {"success": true, "data": [...]}

    3. {"success": true, "data": {"rooms": [...]}}

    4. {"success": true, "data": {"data": [...]}}
    """

    # --------------------------------------------------------
    # FORMAT 1
    # --------------------------------------------------------

    if isinstance(
        api_data.get("rooms"),
        list,
    ):

        return api_data["rooms"]

    # --------------------------------------------------------
    # FORMAT 2
    # --------------------------------------------------------

    if isinstance(
        api_data.get("data"),
        list,
    ):

        return api_data["data"]

    # --------------------------------------------------------
    # FORMAT 3 / 4
    # --------------------------------------------------------

    data = api_data.get("data")

    if isinstance(data, dict):

        if isinstance(
            data.get("rooms"),
            list,
        ):

            return data["rooms"]

        if isinstance(
            data.get("data"),
            list,
        ):

            return data["data"]

    # --------------------------------------------------------
    # NOTHING FOUND
    # --------------------------------------------------------

    return []


# ============================================================
# HANDLE NON-SEARCH INTENT
# ============================================================

def handle_non_search_intent(
    intent: str,
) -> bool:

    """
    Handle intents that do not require room searching.

    Returns:
        True  -> intent was handled
        False -> continue with room-search pipeline
    """

    normalized_intent = (
        str(intent)
        .strip()
        .upper()
    )

    if normalized_intent == "GREETING":

        print_header(
            "FINAL RESPONSE"
        )

        print(
            GREETING_RESPONSE
        )

        return True

    if normalized_intent == "CHATBOT_INFO":

        print_header(
            "FINAL RESPONSE"
        )

        print(
            CHATBOT_INFO_RESPONSE
        )

        return True

    return False


# ============================================================
# MAIN AGENT
# ============================================================

async def main():

    print_header(
        "HOMILIVO ROOM SEARCH AGENT"
    )

    print(
        "Architecture: Guardrails + "
        "Fast Parser + MCP + Homilivo API"
    )

    print(
        "Default response mode: FAST"
    )

    # ========================================================
    # CONNECT TO MCP
    # ========================================================

    print(
        "\nConnecting to MCP Server..."
    )

    async with Client(mcp) as client:

        print(
            "✓ MCP Server Connected"
        )

        # ====================================================
        # DISCOVER MCP TOOLS
        # ====================================================

        tools_response = (
            await client.list_tools()
        )

        if hasattr(
            tools_response,
            "tools",
        ):

            tools = tools_response.tools

        elif isinstance(
            tools_response,
            tuple,
        ):

            tools = tools_response[1]

        else:

            tools = tools_response

        print(
            "\nAvailable MCP Tools:"
        )

        for tool in tools:

            if hasattr(
                tool,
                "name",
            ):

                print(
                    f"  • {tool.name}"
                )

            else:

                print(
                    f"  • {tool}"
                )

        # ====================================================
        # USER INPUT
        # ====================================================

        print(
            "\n" + "-" * 70
        )

        user_query = input(
            "What kind of room are you looking for?\n> "
        ).strip()

        # ====================================================
        # RESPONSE MODE
        # ====================================================

        print(
            "\nResponse Mode:"
        )

        print(
            "  1. FAST  - Python response"
        )

        print(
            "  2. AI    - Qwen3 response"
        )

        mode = input(
            "\nChoose mode [1/2, default=1]: "
        ).strip()

        if mode not in ["1", "2"]:
            mode = "1"

        if mode == "1":

            print(
                "\n✓ FAST MODE selected"
            )

        else:

            print(
                "\n✓ AI MODE selected"
            )

        # ====================================================
        # INPUT GUARDRAIL
        # ====================================================

        print(
            "\n→ Running input guardrail..."
        )

        input_allowed, input_message = (
            validate_user_input(
                user_query
            )
        )

        if not input_allowed:

            print(
                "\n✗ INPUT GUARDRAIL "
                "BLOCKED REQUEST"
            )

            print(
                f"Reason: {input_message}"
            )

            return

        print(
            "✓ Input guardrail passed"
        )

        # ====================================================
        # NON-SEARCH INTENTS
        # ====================================================

        if handle_non_search_intent(
            input_message
        ):

            return

        # ====================================================
        # FAST PARAMETER EXTRACTION
        # ====================================================

        print(
            "\n→ Extracting search parameters..."
        )

        search_request = (
            extract_search_parameters(
                user_query
            )
        )

        print(
            "✓ Parameters extracted locally"
        )

        print(
            "\nExtracted Parameters:"
        )

        print(
            f"  City        : "
            f"{search_request.get('city')}"
        )

        print(
            f"  Min Price   : "
            f"{search_request.get('min_price')}"
        )

        print(
            f"  Max Price   : "
            f"{search_request.get('max_price')}"
        )

        print(
            f"  Room Types  : "
            f"{search_request.get('room_types')}"
        )

        print(
            f"  Ideal For   : "
            f"{search_request.get('ideal_for')}"
        )

        # ====================================================
        # CHECK EXTRACTION
        # ====================================================

        if not any([
            search_request.get("city"),
            search_request.get("min_price"),
            search_request.get("max_price"),
            search_request.get("room_types"),
            search_request.get("ideal_for"),
        ]):

            print(
                "\n✗ Could not determine useful "
                "search filters."
            )

            print(
                "Please provide a city, price, "
                "room type, or target user."
            )

            return

        # ====================================================
        # PARAMETER GUARDRAIL
        # ====================================================

        print(
            "\n→ Validating search parameters..."
        )

        parameters_valid, parameter_message = (
            validate_search_arguments(
                search_request
            )
        )

        if not parameters_valid:

            print(
                "\n✗ PARAMETER GUARDRAIL "
                "BLOCKED REQUEST"
            )

            print(
                f"Reason: {parameter_message}"
            )

            return

        print(
            "✓ Search parameters validated"
        )

        # ====================================================
        # EXECUTE MCP TOOL
        # ====================================================

        print_header(
            "EXECUTING MCP TOOL"
        )

        print(
            "Calling: search_rooms(...)"
        )

        print(
            "✓ Validated parameters sent to MCP"
        )

        try:

            result = await client.call_tool(
                "search_rooms",
                search_request,
            )

        except Exception as error:

            print(
                "\n✗ MCP TOOL EXECUTION FAILED"
            )

            print(
                f"Error: {error}"
            )

            return

        print(
            "✓ MCP tool executed"
        )

        # ====================================================
        # EXTRACT MCP RESULT
        # ====================================================

        if not result.content:

            print(
                "\n✗ MCP returned no content"
            )

            return

        try:

            result_text = (
                result.content[0].text
            )

            api_data = json.loads(
                result_text
            )

        except (
            AttributeError,
            IndexError,
            json.JSONDecodeError,
        ) as error:

            print(
                "\n✗ Invalid MCP response"
            )

            print(
                f"Error: {error}"
            )

            return

        # ====================================================
        # MCP RESPONSE INFORMATION
        # ====================================================

        print(
            "\n→ MCP response received"
        )

        print(
            f"✓ Response keys: "
            f"{list(api_data.keys())}"
        )

        # ====================================================
        # CHECK API ERROR
        # ====================================================

        if api_data.get("success") is False:

            print(
                "\n✗ MCP/API returned an error"
            )

            print(
                f"Error: "
                f"{api_data.get('error')}"
            )

            print(
                "\nFull MCP Response:"
            )

            print(
                json.dumps(
                    api_data,
                    indent=2,
                    ensure_ascii=False,
                )
            )

            return

        # ====================================================
        # EXTRACT ROOMS
        # ====================================================

        rooms = extract_rooms_from_response(
            api_data
        )

        print(
            f"✓ MCP returned "
            f"{len(rooms)} room record(s)"
        )

        # ====================================================
        # DATA GUARDRAIL
        # ====================================================

        print(
            "\n→ Validating room data..."
        )

        data_valid, data_message = (
            validate_room_data(
                rooms
            )
        )

        if not data_valid:

            print(
                "\n✗ DATA GUARDRAIL "
                "BLOCKED API DATA"
            )

            print(
                f"Reason: {data_message}"
            )

            print(
                "\nMCP RESPONSE:"
            )

            print(
                json.dumps(
                    api_data,
                    indent=2,
                    ensure_ascii=False,
                )
            )

            return

        print(
            "✓ API room data validated"
        )

        print(
            f"✓ Verified rooms: "
            f"{len(rooms)}"
        )

        # ====================================================
        # GENERATE RESPONSE
        # ====================================================

        print_header(
            "GENERATING RESPONSE"
        )

        # ----------------------------------------------------
        # FAST MODE
        # ----------------------------------------------------

        if mode == "1":

            print(
                "Mode: FAST"
            )

            print(
                "Generating response locally..."
            )

            final_text = (
                generate_fast_response(
                    user_query,
                    rooms,
                    search_request,
                )
            )

        # ----------------------------------------------------
        # AI MODE
        # ----------------------------------------------------

        else:

            print(
                "Mode: QWEN3"
            )

            print(
                "Sending verified data "
                "to local Qwen3..."
            )

            try:

                final_text = (
                    generate_ai_response(
                        user_query,
                        rooms,
                    )
                )

            except Exception as error:

                print(
                    "\n✗ Qwen3 failed"
                )

                print(
                    f"Error: {error}"
                )

                return

        # ====================================================
        # OUTPUT GUARDRAIL
        # ====================================================

        print(
            "\n→ Running output guardrail..."
        )

        output_valid, output_message = (
            validate_ai_response(
                final_text
            )
        )

        if not output_valid:

            print(
                "\n✗ OUTPUT GUARDRAIL "
                "BLOCKED RESPONSE"
            )

            print(
                f"Reason: {output_message}"
            )

            return

        print(
            "✓ Output guardrail passed"
        )

        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        print_header(
            "FINAL RESPONSE"
        )

        print(
            final_text
        )

        # ====================================================
        # FINAL STATUS
        # ====================================================

        print_header(
            "AGENT TEST COMPLETED"
        )

        print(
            "✓ Input Guardrail      : PASSED"
        )

        print(
            "✓ Fast Parser          : PASSED"
        )

        print(
            "✓ Parameter Guardrail  : PASSED"
        )

        print(
            "✓ MCP Connection       : SUCCESS"
        )

        print(
            "✓ MCP Tool Execution   : SUCCESS"
        )

        print(
            "✓ API Data Guardrail   : PASSED"
        )

        if mode == "1":

            print(
                "✓ Response Generation  : FAST MODE"
            )

        else:

            print(
                "✓ Local Qwen3          : SUCCESS"
            )

        print(
            "✓ Output Guardrail     : PASSED"
        )

        print(
            "✓ Overall Result       : PASSED"
        )

        print(
            "=" * 70
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())
'''
"""

'''