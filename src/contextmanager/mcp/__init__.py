"""Phase 4 — Surface B: the cooperative MCP server.

Exposes the shared Phase 1/2 engine + ``DurableStore`` as six explicit MCP tools
(store_save/search, state_snapshot/load, context_checkpoint/rehydrate) so an
MCP-capable agent can deliberately externalize and retrieve durable state.
"""

from __future__ import annotations

from .config import McpConfig
from .counter import HeuristicTokenCounter
from .service import GovernorService
from .server import build_server, build_service

__all__ = [
    "McpConfig",
    "HeuristicTokenCounter",
    "GovernorService",
    "build_server",
    "build_service",
]
