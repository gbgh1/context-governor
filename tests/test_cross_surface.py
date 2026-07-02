"""Cross-surface integration: the proxy (Surface A) and the MCP server (Surface B)
share ONE on-disk store, so a handle minted by either resolves in both.

This pins the "one store, two surfaces" contract asserted in wiki/context-governor.md
and the surface notes. Two *separate* DurableStore instances at the SAME root model the
two real processes sharing the filesystem. No network: FakeCounter throughout.
"""

from __future__ import annotations

from conftest import FakeCounter

from contextmanager.durable import DurableStore
from contextmanager.proxy.config import ProxyConfig
from contextmanager.proxy.rewriter import PromptRewriter
from contextmanager.mcp.config import McpConfig
from contextmanager.mcp.service import GovernorService


def _bulky(words: int = 30) -> str:
    return " ".join(f"lighthouse{i}" for i in range(words))


def test_proxy_handle_resolves_in_mcp(tmp_path):
    """Surface A pages a message out; Surface B (same root) recalls it by handle."""
    root = str(tmp_path / "contextstore")
    proxy_store = DurableStore(root)
    mcp_store = DurableStore(root)
    try:
        # Surface A: handle-ize a bulky message -> persisted to disk.
        rewriter = PromptRewriter(
            ProxyConfig(upstream_base_url="http://x", store_root=root,
                        handle_threshold_tokens=10, stub_preview_chars=10),
            FakeCounter(), proxy_store,
        )
        content = _bulky()
        result = rewriter.rewrite_outgoing([{"role": "user", "content": content}])
        stub = result.messages[0]["content"]
        handle = PromptRewriter.parse_handles(stub)[0]

        # Surface B: the same handle resolves; rehydrate returns the full content.
        svc = GovernorService(mcp_store, FakeCounter(),
                              McpConfig(store_root=root, rehydrate_budget_tokens=1000))
        out = svc.context_rehydrate(handle=handle, budget_tokens=1000)
        assert out["found"] is True
        assert content in out["text"]
        # and it is searchable from Surface B too
        assert any(r["handle"] == handle for r in svc.store_search("lighthouse0")["results"])
    finally:
        proxy_store.close()
        mcp_store.close()


def test_mcp_handle_resolves_in_proxy(tmp_path):
    """Surface B saves content; Surface A (same root) auto-rehydrates a reference to it."""
    root = str(tmp_path / "contextstore")
    proxy_store = DurableStore(root)
    mcp_store = DurableStore(root)
    try:
        # Surface B: save content -> handle.
        svc = GovernorService(mcp_store, FakeCounter(), McpConfig(store_root=root))
        saved = svc.store_save("the keeper logs the tide each dawn and dusk", role="note")
        handle = saved["handle"]

        # Surface A: a message explicitly referencing that handle auto-rehydrates it
        # from the shared store (proxy reads notes/<handle>.md written by Surface B).
        rewriter = PromptRewriter(
            ProxyConfig(upstream_base_url="http://x", store_root=root,
                        handle_threshold_tokens=10_000,  # don't handle-ize the small ref msg
                        rehydrate_budget_tokens=1000),
            FakeCounter(), proxy_store,
        )
        ref = f"please recall [[cm:stored handle={handle} role=note tokens=9]]"
        result = rewriter.rewrite_outgoing([{"role": "user", "content": ref}])
        assert handle in result.rehydrated_handles
        rehydrated = [m for m in result.messages
                      if PromptRewriter.is_rehydrated(m.get("content", ""))]
        assert rehydrated and "keeper logs the tide" in rehydrated[0]["content"]
    finally:
        proxy_store.close()
        mcp_store.close()
