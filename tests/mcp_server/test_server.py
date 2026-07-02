"""FastMCP integration (spec §5, §7): tool registration + call round-trip.

Async (pytest-asyncio strict). Drives FastMCP in-process via ``list_tools`` /
``call_tool`` — no stdio, no network. Service is injected (FakeCounter + tmp store).
"""

from __future__ import annotations

import json

import pytest

from contextmanager.durable import DurableStore
from contextmanager.mcp.config import McpConfig
from contextmanager.mcp.service import GovernorService
from contextmanager.mcp.server import build_server
from conftest import FakeCounter

EXPECTED_TOOLS = {
    "store_save",
    "store_search",
    "state_snapshot",
    "state_load",
    "context_checkpoint",
    "context_rehydrate",
}


def _build(tmp_path):
    config = McpConfig(store_root=str(tmp_path / "store"), preview_chars=20)
    svc = GovernorService(DurableStore(config.store_root), FakeCounter(), config)
    return build_server(config, service=svc), svc


def _payload(result):
    """Normalize FastMCP call_tool result -> the tool's returned dict.

    Across mcp versions call_tool returns either ``list[Content]`` or a
    ``(list[Content], structured)`` tuple. Parse the first TextContent's JSON.
    """
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


@pytest.mark.asyncio
async def test_list_tools_has_the_six(tmp_path):
    server, svc = _build(tmp_path)
    try:
        tools = await server.list_tools()
        assert {t.name for t in tools} == EXPECTED_TOOLS
        # each tool carries a description (from the wrapper docstring)
        assert all(t.description for t in tools)
    finally:
        svc.close()


@pytest.mark.asyncio
async def test_call_store_save_then_search(tmp_path):
    server, svc = _build(tmp_path)
    try:
        saved = _payload(
            await server.call_tool("store_save", {"content": "a beacon by the harbor wall"})
        )
        assert "handle" in saved
        found = _payload(await server.call_tool("store_search", {"query": "beacon"}))
        assert found["count"] >= 1
        assert found["results"][0]["handle"] == saved["handle"]
    finally:
        svc.close()


@pytest.mark.asyncio
async def test_call_state_roundtrip(tmp_path):
    server, svc = _build(tmp_path)
    try:
        await server.call_tool("state_snapshot", {"state": {"hp": 10}})
        loaded = _payload(await server.call_tool("state_load", {}))
        assert loaded["state"] == {"hp": 10}
    finally:
        svc.close()
