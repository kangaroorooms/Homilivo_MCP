# Copied from app/experiments/agent_modes.py at commit fca91ac (the last commit that changed it; the other copies cite 7dbcba3) (copied, not imported).
# Changes: extracted only _call_search_rooms_with_timeout, not the whole file; the two imports below and the module logger are added
# because the originals live at the top of agent_modes.py; MCP_TOOL_TIMEOUT_SECONDS comes from the copied tools/mcp_search.py.
# Drift is checked by llm_approach/tests/test_copy_drift.py.
# ==== end of v2 provenance header ====
"""The 15 s bound on one search_rooms MCP call, copied next to the copied tool."""

from __future__ import annotations  # v2: header for the extracted file

import asyncio  # v2: the original imports it at the top of agent_modes.py
import logging  # v2: same

from llm_approach.tools.mcp_search import MCP_TOOL_TIMEOUT_SECONDS  # v2: was: from app.server import MCP_TOOL_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)  # v2: module logger, as in agent_modes.py


async def _call_search_rooms_with_timeout(client, search_request):
    """
    Await the search_rooms MCP tool, bounded by MCP_TOOL_TIMEOUT_SECONDS
    so a hung MCP process can't stall the turn forever. On timeout it
    logs and re-raises asyncio.TimeoutError, which the call site's
    existing `except Exception` turns into the standard "Homilivo is
    temporarily unavailable" reply (never a fake "0 rooms found").
    """
    try:

        return await asyncio.wait_for(
            client.call_tool("search_rooms", search_request),
            timeout=MCP_TOOL_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:

        logger.warning(
            "search_rooms MCP tool timed out after %ss",
            MCP_TOOL_TIMEOUT_SECONDS,
        )

        raise
