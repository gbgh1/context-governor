"""FastMCP adapter for Surface B (NORMATIVE: tasks/phase4-spec.md §5).

Thin: each tool is a typed wrapper that delegates to ``GovernorService`` and returns
its plain dict. All correctness lives in the service; this file only wires names,
schemas (from type hints), and descriptions (from docstrings).
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from ..durable import DurableStore
from ..tokenizer import LlamaServerTokenCounter
from ..types import TokenCounter
from .config import McpConfig
from .counter import HeuristicTokenCounter
from .service import GovernorService


def build_service(config: McpConfig) -> GovernorService:
    """Construct a GovernorService with a real store and a config-selected counter."""
    store = DurableStore(config.store_root)
    counter: TokenCounter
    if config.upstream_base_url:
        counter = LlamaServerTokenCounter(
            config.upstream_base_url, api_key=config.upstream_api_key
        )
    else:
        counter = HeuristicTokenCounter()
    return GovernorService(store, counter, config)


def build_server(config: McpConfig, *, service: Optional[GovernorService] = None) -> FastMCP:
    """Build a FastMCP server exposing the six governor tools.

    Pass ``service`` to inject a test/double; otherwise one is built from ``config``.
    The service is stashed on the server as ``_governor_service`` so the entrypoint
    can close it on shutdown.
    """
    svc = service if service is not None else build_service(config)
    mcp = FastMCP(config.server_name)

    @mcp.tool(name="store_save")
    def store_save(content: str, role: str = "note") -> dict:
        """Your long-term notepad. Call this WHENEVER you read a file or produce output
        longer than ~10 lines that you might reference later — keep only the returned
        handle and a one-line summary, not the full text. Returns {handle, id, tokens}."""
        return svc.store_save(content, role=role)

    @mcp.tool(name="store_search")
    def store_search(query: str, k: Optional[int] = None) -> dict:
        """Find something you saved earlier by keyword (before asking the user to repeat
        it). Returns matching handles, scores, and short previews."""
        return svc.store_search(query, k=k)

    @mcp.tool(name="state_snapshot")
    def state_snapshot(state: dict, merge: bool = True) -> dict:
        """Your save point. Call AFTER any real change (files edited, a decision made, a
        task finished) with the FULL current state as a JSON object — this is the one
        thing that survives across turns, so keep it accurate. merge=True for partial
        updates, merge=False to replace."""
        return svc.state_snapshot(state, merge=merge)

    @mcp.tool(name="state_load")
    def state_load() -> dict:
        """Call this at the START of a task to recover what you already know — your only
        memory of earlier turns. Returns the parsed state plus rendered text."""
        return svc.state_load()

    @mcp.tool(name="context_checkpoint")
    def context_checkpoint(
        label: str, content: str, state: Optional[dict] = None
    ) -> dict:
        """Save a labeled milestone (what you did + what's next, plus optional state) so you
        can resume exactly. Re-using a label overwrites it — no duplicates."""
        return svc.context_checkpoint(label, content, state=state)

    @mcp.tool(name="context_rehydrate")
    def context_rehydrate(
        query: Optional[str] = None,
        handle: Optional[str] = None,
        budget_tokens: Optional[int] = None,
        k: Optional[int] = None,
    ) -> dict:
        """Need a detail you offloaded earlier? Page it back into context here (by query or
        by handle) under a token budget — instead of asking the user to re-paste it."""
        return svc.context_rehydrate(
            query=query, handle=handle, budget_tokens=budget_tokens, k=k
        )

    setattr(mcp, "_governor_service", svc)
    return mcp
