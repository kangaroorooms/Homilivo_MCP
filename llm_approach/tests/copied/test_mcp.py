# Copied from tests/test_mcp.py at commit f085b0c (copied, not imported).
# Changes: the import points at the copied tool; the execution test removes KANGAROO_TOKEN so it can never call the real API. Every changed line is marked '# v2:'.
# Source under test is the copied tool llm_approach/tools/mcp_search.py (from app/server.py); app/ is not imported.

import json

import pytest
from mcp import Client

from llm_approach.tools.mcp_search import mcp  # v2: was: from app.server import mcp


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
async def test_mcp_search_rooms_tool_execution(monkeypatch):
    """
    Verify that the MCP search_rooms tool can be executed and
    returns a valid JSON response.
    """

    monkeypatch.delenv("KANGAROO_TOKEN", raising=False)  # v2: never reach the real Rooms API; the tool fails fast without a token
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