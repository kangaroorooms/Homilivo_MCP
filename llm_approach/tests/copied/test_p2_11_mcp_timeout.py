# Copied from tests/test_p2_11_cost_controls.py at commit 0221d80 (copied, not imported).
# Changes: only the MCP timeout constant and the _call_search_rooms_with_timeout wrapper test; the wrapper comes from tools/mcp_timeout.py. Every changed line is marked '# v2:'.
# Not copied: the end-to-end process_message() part and the LLM budget tests, which need agent_modes. Source under test is the copied tool llm_approach/tools/mcp_search.py (from app/server.py); app/ is not imported.

import asyncio
import logging
from unittest.mock import patch

from llm_approach.tools import mcp_timeout  # v2: was: from app.experiments import agent_modes


def test_mcp_timeout_constant_exists():
    from llm_approach.tools.mcp_search import MCP_TOOL_TIMEOUT_SECONDS  # v2: was: from app.server import MCP_TOOL_TIMEOUT_SECONDS

    assert MCP_TOOL_TIMEOUT_SECONDS == 15


class _HangingClient:
    async def call_tool(self, name, arguments):
        await asyncio.sleep(60)


class _TimeoutClient:
    async def call_tool(self, name, arguments):
        raise asyncio.TimeoutError()


def test_mcp_tool_times_out_gracefully(caplog):
    # The wait_for wrapper itself: a hung call is cut off and logged.
    with patch.object(mcp_timeout, "MCP_TOOL_TIMEOUT_SECONDS", 0.05):  # v2: was: patch.object(agent_modes, ...)
        with caplog.at_level(logging.WARNING):
            try:
                asyncio.run(
                    mcp_timeout._call_search_rooms_with_timeout(  # v2: was: agent_modes._call_search_rooms_with_timeout(
                        _HangingClient(), {"city": "Bangalore"}
                    )
                )
                raised = None
            except asyncio.TimeoutError as error:
                raised = error

    assert raised is not None
    assert "timed out" in caplog.text
