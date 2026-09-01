import json

import pytest
from mcp import Client

from app.server import mcp


@pytest.mark.anyio
async def test_mcp_server_connection_and_tool_discovery():
    """
    Verify that the MCP server can be connected to and exposes
    the expected room-search tool.
    """

    async with Client(mcp) as client:

        tools_response = await client.list_tools()

        if hasattr(tools_response, "tools"):
            tools = tools_response.tools

        elif isinstance(tools_response, tuple):
            tools = tools_response[1]

        else:
            tools = tools_response

        tool_names = []

        for tool in tools:

            if hasattr(tool, "name"):
                tool_names.append(tool.name)

        assert "search_rooms" in tool_names


@pytest.mark.anyio
async def test_mcp_search_rooms_tool_execution():
    """
    Verify that the MCP search_rooms tool can be executed and
    returns a valid JSON response.
    """

    async with Client(mcp) as client:

        search_request = {
            "sort": "latest",
            "limit": 10,
        }

        result = await client.call_tool(
            "search_rooms",
            search_request,
        )

        assert result.content
        assert hasattr(
            result.content[0],
            "text",
        )

        response_text = result.content[0].text

        data = json.loads(
            response_text
        )

        assert isinstance(
            data,
            dict,
        )

        assert "success" in data

        if data.get("success"):

            assert "rooms" in data
            assert isinstance(
                data["rooms"],
                list,
            )

        else:

            assert (
                "error" in data
                or "message" in data
            )