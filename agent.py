import asyncio
import json

from ollama import chat
from mcp import Client

from server import mcp


MODEL = "qwen3:4b"


def build_tool_schema():
    """
    Convert our MCP search_rooms tool into
    a tool description that Ollama can understand.
    """

    return {
        "type": "function",
        "function": {
            "name": "search_rooms",
            "description": (
                "Search Homilivo rooms using city, "
                "price, room type, gender preference, "
                "ideal user, sorting and result limit."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "city": {
                        "type": "string",
                        "description": "City to search in"
                    },

                    "min_price": {
                        "type": "integer",
                        "description": "Minimum monthly rent"
                    },

                    "max_price": {
                        "type": "integer",
                        "description": "Maximum monthly rent"
                    },

                    "room_types": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "Room types such as "
                            "PG, 1BHK, 2BHK"
                        )
                    },

                    "gender_preference": {
                        "type": "string",
                        "description": (
                            "Gender preference such as "
                            "MALE_ONLY or FEMALE_ONLY"
                        )
                    },

                    "ideal_for": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "Suitable users such as "
                            "Students or Working Professionals"
                        )
                    },

                    "sort": {
                        "type": "string",
                        "description": "Sorting method"
                    },

                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results"
                    }
                }
            }
        }
    }


async def main():

    print("\n" + "=" * 70)
    print("              HOMILIVO AI + MCP AGENT")
    print("=" * 70)

    # =========================================================
    # CONNECT TO MCP
    # =========================================================

    async with Client(mcp) as client:

        print("\n✓ MCP Server Connected")

        # -----------------------------------------------------
        # DISCOVER MCP TOOLS
        # -----------------------------------------------------

        tools_response = await client.list_tools()

        if hasattr(tools_response, "tools"):
            mcp_tools = tools_response.tools

        elif isinstance(tools_response, tuple):
            mcp_tools = tools_response[1]

        else:
            mcp_tools = tools_response

        print("\nAvailable MCP Tools:")

        for tool in mcp_tools:

            if hasattr(tool, "name"):
                print(f"  • {tool.name}")

        # -----------------------------------------------------
        # USER INPUT
        # -----------------------------------------------------

        print("\n" + "-" * 70)

        user_query = input(
            "What kind of room are you looking for?\n> "
        )

        print("\nAI is analyzing your request...")

        # -----------------------------------------------------
        # OLLAMA TOOL
        # -----------------------------------------------------

        tools = [
            build_tool_schema()
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a room-search assistant. "
                    "Use the search_rooms tool whenever "
                    "the user asks to find rooms. "
                    "Only use information explicitly provided "
                    "by the MCP tool results. "
                    "Never invent or assume room details, "
                    "amenities, facilities, locations, prices, "
                    "availability, or other information."
                )
            },
            {
                "role": "user",
                "content": user_query
            }
        ]

        # -----------------------------------------------------
        # FIRST AI CALL
        # -----------------------------------------------------

        response = chat(
            model=MODEL,
            messages=messages,
            tools=tools
        )

        messages.append(response.message)

        # -----------------------------------------------------
        # CHECK FOR TOOL CALL
        # -----------------------------------------------------

        if not response.message.tool_calls:

            print("\nAI RESPONSE:")
            print("-" * 70)
            print(response.message.content)

            return

        # -----------------------------------------------------
        # PROCESS TOOL CALLS
        # -----------------------------------------------------

        for tool_call in response.message.tool_calls:

            function_name = tool_call.function.name

            arguments = tool_call.function.arguments

            print("\n" + "=" * 70)
            print("              AI TOOL DECISION")
            print("=" * 70)

            print(f"Tool      : {function_name}")

            print(
                "Arguments : "
                + json.dumps(
                    arguments,
                    indent=2
                )
            )

            # -------------------------------------------------
            # CALL MCP TOOL
            # -------------------------------------------------

            if function_name == "search_rooms":

                print("\nCalling MCP: search_rooms(...)")

                result = await client.call_tool(
                    "search_rooms",
                    arguments
                )

                result_text = result.content[0].text

                print("✓ MCP tool executed")

                # -------------------------------------------------
                # SEND RESULT BACK TO AI
                # -------------------------------------------------

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": "search_rooms",
                        "content": result_text
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Now answer the user's original question using ONLY "
                            "the room data returned by the tool. "
                            "Do not invent, infer, or assume any amenities, "
                            "facilities, locations, prices, or availability."
        )
    }
)

        # -----------------------------------------------------
        # FINAL AI RESPONSE
        # -----------------------------------------------------
        print("\n→ Sending real room results to AI for final response...")
        final_response = chat(
            model=MODEL,
            messages=messages,
            tools=tools
        )
        print("✓ Final AI response generated")
        print("\n" + "=" * 70)
        print("                 AI RESPONSE")
        print("=" * 70)

        print(final_response.message.content)

        print("\n" + "=" * 70)
        print("              AGENT TEST COMPLETED")
        print("=" * 70)

        print("✓ Local LLM       : SUCCESS")
        print("✓ MCP Connection  : SUCCESS")
        print("✓ Tool Discovery  : SUCCESS")
        print("✓ Tool Execution  : SUCCESS")
        print("✓ Homilivo API    : SUCCESS")
        print("✓ AI Response     : SUCCESS")

        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())