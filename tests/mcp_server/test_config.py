"""McpConfig validation (spec §2, §7)."""

from __future__ import annotations

import pytest

from contextmanager.mcp.config import McpConfig


def test_valid_defaults(tmp_path):
    cfg = McpConfig(store_root=str(tmp_path))
    assert cfg.transport == "stdio"
    assert cfg.default_search_k == 5
    assert cfg.upstream_base_url is None


@pytest.mark.parametrize(
    "over",
    [
        {"default_search_k": 0},
        {"default_search_k": -1},
        {"preview_chars": -1},
        {"rehydrate_budget_tokens": -1},
        {"transport": "carrier-pigeon"},
    ],
)
def test_invalid_fields_raise(over):
    with pytest.raises(ValueError):
        McpConfig(**over)


def test_valid_transports():
    for t in ("stdio", "streamable-http", "sse"):
        assert McpConfig(transport=t).transport == t
