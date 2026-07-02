"""Offline ``TokenCounter`` for the MCP server (NORMATIVE: tasks/phase4-spec.md §3).

A deterministic ~1-token-per-4-chars approximation. It exists so the server's
budgeted page-in (``DurableStore.page_in``) works WITHOUT a running llama-server.
The only hard requirement is measurability: truncating to ``n`` tokens must yield
text that counts as ``<= n`` tokens, so the page_in clamp never overshoots.
"""

from __future__ import annotations

from ..types import Message


class HeuristicTokenCounter:
    """Char-based ``TokenCounter`` implementation (no network)."""

    CHARS_PER_TOKEN = 4
    PER_MESSAGE_OVERHEAD = 4

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        # ceil(len / CHARS_PER_TOKEN)
        return (len(text) + self.CHARS_PER_TOKEN - 1) // self.CHARS_PER_TOKEN

    def count_messages(self, messages: list[Message]) -> int:
        return sum(self.count_text(m.content) + self.PER_MESSAGE_OVERHEAD for m in messages)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        # max_tokens * CHARS_PER_TOKEN chars -> at most max_tokens tokens by count_text:
        #   (min(len, 4n) + 3) // 4 <= n. Measurable, never overshoots.
        return text[: max_tokens * self.CHARS_PER_TOKEN]
