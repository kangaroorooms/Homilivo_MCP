import asyncio
import json

from mcp import Client
from server import mcp


async def main():

    print("\n" + "=" * 70)
    print("              HOMILIVO MCP DEMONSTRATION")
    print("=" * 70)

    async with Client(mcp) as client:

        print("\n✓ MCP Server Connected")

        # -----------------------------------------------------
        # DISCOVER TOOLS
        # -----------------------------------------------------

        tools_response = await client.list_tools()

        if hasattr(tools_response, "tools"):
            tools = tools_response.tools

        elif isinstance(tools_response, tuple):
            tools = tools_response[1]

        else:
            tools = tools_response

        print("\nAvailable MCP Tools:")

        for tool in tools:

            if hasattr(tool, "name"):
                print(f"  • {tool.name}")

        # -----------------------------------------------------
        # REQUEST
        # -----------------------------------------------------

        search_request = {
            "sort": "latest",
            "limit": 10
        }

        print("\n" + "=" * 70)
        print("              MCP TOOL REQUEST")
        print("=" * 70)

        print("Tool        : search_rooms")
        print("Sort        : latest")
        print("Result Limit: 10")

        print("\nCalling real Homilivo API...")
        print("https://homilivo.onrender.com/api/rooms")

        # -----------------------------------------------------
        # CALL MCP TOOL
        # -----------------------------------------------------

        result = await client.call_tool(
            "search_rooms",
            search_request
        )

        # -----------------------------------------------------
        # READ RESULT
        # -----------------------------------------------------

        response_text = result.content[0].text

        data = json.loads(response_text)

        print("\n" + "=" * 70)
        print("              API RESPONSE")
        print("=" * 70)

        if data.get("success"):

            rooms = data.get("rooms", [])

            print("Status      : SUCCESS")
            print(f"Rooms Found : {len(rooms)}")

            for i, room in enumerate(rooms, 1):

                print("\n" + "-" * 70)
                print(f"ROOM #{i}")
                print("-" * 70)

                print(
                    f"ID          : "
                    f"{room.get('id', 'N/A')}"
                )

                print(
                    f"Title       : "
                    f"{room.get('title', 'N/A')}"
                )

                print(
                    f"City        : "
                    f"{room.get('city', 'N/A')}"
                )

                price = room.get("pricePerMonth")

                if price is not None:
                    print(
                        f"Monthly Rent: "
                        f"₹{price:,}"
                    )
                else:
                    print("Monthly Rent: N/A")

                print(
                    f"Room Type   : "
                    f"{room.get('roomType', 'N/A')}"
                )

                ideal_for = room.get("idealFor", [])

                if ideal_for:

                    print(
                        f"Ideal For   : "
                        f"{', '.join(ideal_for)}"
                    )

        else:

            print("Status : FAILED")

            print(
                f"Error  : "
                f"{data.get('error', 'Unknown error')}"
            )

        # -----------------------------------------------------
        # FINAL STATUS
        # -----------------------------------------------------

        print("\n" + "=" * 70)
        print("              MCP TEST COMPLETED")
        print("=" * 70)

        if data.get("success"):

            print("✓ MCP Connection : SUCCESS")
            print("✓ Tool Discovery : SUCCESS")
            print("✓ Tool Execution : SUCCESS")
            print("✓ Real API       : SUCCESS")
            print("✓ Overall Result : PASSED")

        else:

            print("✗ Overall Result : FAILED")

        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())