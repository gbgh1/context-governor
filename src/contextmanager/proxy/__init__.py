"""Phase 3 — Surface A: the Endpoint Proxy (universal OpenAI-compatible reverse proxy).

Package marker. Exports the configuration and the pure message-list rewriter (the core).
"""

from __future__ import annotations

from .config import ProxyConfig
from .rewriter import PromptRewriter, RewriteResult

__all__ = [
    "ProxyConfig",
    "PromptRewriter",
    "RewriteResult",
]
