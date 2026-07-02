"""Phase 4 — Surface B configuration (NORMATIVE: tasks/phase4-spec.md §2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_TRANSPORTS = {"stdio", "streamable-http", "sse"}


@dataclass(frozen=True)
class McpConfig:
    """Immutable configuration for the cooperative MCP server.

    ``upstream_base_url`` selects the token counter: when set, the server uses the
    real ``LlamaServerTokenCounter`` (HTTP to llama-server); when ``None`` it uses
    the offline ``HeuristicTokenCounter`` so the server is self-contained.
    """

    store_root: str = "./contextstore"
    upstream_base_url: Optional[str] = None
    upstream_api_key: Optional[str] = None
    server_name: str = "context-governor"
    transport: str = "stdio"
    default_search_k: int = 5
    preview_chars: int = 200
    rehydrate_budget_tokens: int = 4000

    def __post_init__(self) -> None:
        if self.default_search_k <= 0:
            raise ValueError("default_search_k must be > 0")
        if self.preview_chars < 0:
            raise ValueError("preview_chars must be >= 0")
        if self.rehydrate_budget_tokens < 0:
            raise ValueError("rehydrate_budget_tokens must be >= 0")
        if self.transport not in _TRANSPORTS:
            raise ValueError(
                f"transport must be one of {sorted(_TRANSPORTS)}, got {self.transport!r}"
            )
