import json
from app.fast_parser import extract_search_parameters

from app.guardrails import (
    validate_user_input,
    validate_search_arguments,
    validate_room_data,
    validate_ai_response
)


# ============================================================
# EXTRACT ROOMS FROM MCP RESPONSE
# ============================================================

def extract_rooms_from_response(api_data: dict) -> list:

    if isinstance(api_data.get("rooms"), list):
        return api_data["rooms"]

    if isinstance(api_data.get("data"), list):
        return api_data["data"]

    data = api_data.get("data")

    if isinstance(data, dict):

        if isinstance(data.get("rooms"), list):
            return data["rooms"]

        if isinstance(data.get("data"), list):
            return data["data"]

    return []


# ============================================================
# FAST RESPONSE GENERATOR
# ============================================================

def generate_fast_response(
    rooms: list,
    search_request: dict
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

    for index, room in enumerate(rooms, start=1):

        title = room.get(
            "title",
            "Untitled Room"
        )

        room_city = room.get(
            "city",
            "Unknown"
        )

        price = room.get("pricePerMonth")

        if price is None:
            price = room.get("price")

        room_type = room.get(
            "roomType",
            "Not specified"
        )

        room_ideal_for = room.get(
            "idealFor",
            []
        )

        response_lines.append(
            f"{index}. {title}"
        )

        if price is not None:

            try:
                response_lines.append(
                    f"   ₹{float(price):,.0f}/month"
                )

            except (TypeError, ValueError):

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

    return "\n".join(response_lines).strip()


# ============================================================
# MAIN CHATBOT FUNCTION
# ============================================================

async def process_chat(
    user_query: str,
    client
) -> dict:

    # --------------------------------------------------------
    # INPUT GUARDRAIL
    # --------------------------------------------------------

    input_allowed, input_message = (
        validate_user_input(user_query)
    )

    if not input_allowed:

        return {
            "success": False,
            "stage": "input_guardrail",
            "message": input_message
        }

    # --------------------------------------------------------
    # FAST PARAMETER EXTRACTION
    # --------------------------------------------------------

    search_request = extract_search_parameters(
        user_query
    )

    # --------------------------------------------------------
    # PARAMETER GUARDRAIL
    # --------------------------------------------------------

    parameters_valid, parameter_message = (
        validate_search_arguments(
            search_request
        )
    )

    if not parameters_valid:

        return {
            "success": False,
            "stage": "parameter_guardrail",
            "message": parameter_message
        }

    # --------------------------------------------------------
    # MCP TOOL EXECUTION
    # --------------------------------------------------------

    try:

        result = await client.call_tool(
            "search_rooms",
            search_request
        )

    except Exception as error:

        return {
            "success": False,
            "stage": "mcp",
            "message": f"MCP tool execution failed: {error}"
        }

    # --------------------------------------------------------
    # MCP RESPONSE
    # --------------------------------------------------------

    if not result.content:

        return {
            "success": False,
            "stage": "mcp",
            "message": "MCP returned no content."
        }

    try:

        result_text = result.content[0].text

        api_data = json.loads(result_text)

    except (
        AttributeError,
        IndexError,
        json.JSONDecodeError
    ) as error:

        return {
            "success": False,
            "stage": "mcp_response",
            "message": f"Invalid MCP response: {error}"
        }

    # --------------------------------------------------------
    # API ERROR
    # --------------------------------------------------------

    if api_data.get("success") is False:

        return {
            "success": False,
            "stage": "api",
            "message": api_data.get(
                "error",
                "Homilivo API returned an error."
            )
        }

    # --------------------------------------------------------
    # EXTRACT ROOMS
    # --------------------------------------------------------

    rooms = extract_rooms_from_response(
        api_data
    )

    # --------------------------------------------------------
    # API DATA GUARDRAIL
    # --------------------------------------------------------

    data_valid, data_message = (
        validate_room_data(rooms)
    )

    if not data_valid:

        return {
            "success": False,
            "stage": "api_data_guardrail",
            "message": data_message
        }

    # --------------------------------------------------------
    # FAST RESPONSE
    # --------------------------------------------------------

    final_response = generate_fast_response(
        rooms,
        search_request
    )

    # --------------------------------------------------------
    # OUTPUT GUARDRAIL
    # --------------------------------------------------------

    output_valid, output_message = (
        validate_ai_response(final_response)
    )

    if not output_valid:

        return {
            "success": False,
            "stage": "output_guardrail",
            "message": output_message
        }

    # --------------------------------------------------------
    # STRUCTURED ROOM RESPONSE
    # --------------------------------------------------------

    structured_rooms = []

    for room in rooms:

        price = room.get("pricePerMonth")

        if price is None:
            price = room.get("price")

        structured_rooms.append(
            {
                "id": str(room.get("id")),
                "title": room.get(
                    "title",
                    "Untitled Room"
                ),
                "city": room.get(
                    "city",
                    "Unknown"
                ),
                "location": room.get(
                    "location"
                ),
                "price_per_month": price,
                "room_type": room.get(
                    "roomType"
                ),
                "ideal_for": room.get(
                    "idealFor",
                    []
                )
            }
        )

    # --------------------------------------------------------
    # SUCCESS RESPONSE
    # --------------------------------------------------------

    return {
        "success": True,
        "message": final_response,
        "room_count": len(structured_rooms),
        "rooms": structured_rooms
    }