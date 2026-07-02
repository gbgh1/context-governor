"""GovernorService — the pure logic core of Surface B (NORMATIVE: tasks/phase4-spec.md §4).

No MCP types here. Every method takes/returns plain JSON-serializable values so the
correctness can be tested without an MCP runtime; ``server.py`` is the thin adapter.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from ..durable import DurableStore
from ..note_store import StoreError
from ..types import Message, TokenCounter
from .config import McpConfig

_UNSAFE = re.compile(r"[^a-z0-9._-]+")


def _slug(label: str) -> str:
    """Filesystem/handle-safe slug from a label; stable for the same label."""
    s = _UNSAFE.sub("-", label.lower()).strip("-")
    if not s or set(s) <= {"."}:
        return hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]
    return s


def _preview(content: str, n: int) -> str:
    if n <= 0 or len(content) <= n:
        return content if n > 0 else ""
    return content[:n] + f"…(+{len(content) - n} more chars)"


class GovernorService:
    """Cooperative externalize/retrieve operations over the shared DurableStore."""

    def __init__(self, store: DurableStore, counter: TokenCounter, config: McpConfig) -> None:
        self.store = store
        self.counter = counter
        self.config = config

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def stable_id(prefix: str, role: str, content: str) -> str:
        digest = hashlib.sha1((role + "\x00" + content).encode("utf-8")).hexdigest()[:16]
        return f"{prefix}-{digest}"

    # ------------------------------------------------------------------ store.*
    def store_save(
        self,
        content: str,
        *,
        role: str = "note",
        id: Optional[str] = None,
        links: Optional[list[str]] = None,
    ) -> dict:
        note_id = id or self.stable_id("note", role, content)
        msg = Message(role=role, content=content, id=note_id)
        handle = self.store.page_out(msg)  # also indexes for search
        return {
            "handle": handle,
            "id": note_id,
            "role": role,
            "tokens": self.counter.count_text(content),
        }

    def store_search(self, query: str, *, k: Optional[int] = None) -> dict:
        slices = self.store.search(query, k if k is not None else self.config.default_search_k)
        return {
            "query": query,
            "count": len(slices),
            "results": [
                {
                    "handle": s.handle,
                    "score": s.score,
                    "tokens": self.counter.count_text(s.content),
                    "preview": _preview(s.content, self.config.preview_chars),
                }
                for s in slices
            ],
        }

    # ------------------------------------------------------------------ state.*
    def state_snapshot(self, state: dict, *, merge: bool = True) -> dict:
        if not isinstance(state, dict):
            raise ValueError("state must be a JSON object")
        if merge:
            result = self.store.state.update(state)
        else:
            self.store.state.save(state)
            result = self.store.state.load()
        return {
            "ok": True,
            "merge": merge,
            "keys": sorted(result.keys()),
            "tokens": self.counter.count_text(self.store.state.render()),
        }

    def state_load(self) -> dict:
        data = self.store.state.load()
        return {
            "state": data,
            "rendered": self.store.state.render(),
            "empty": not data,
        }

    # ------------------------------------------------------------------ context.*
    def context_checkpoint(
        self,
        label: str,
        content: str,
        *,
        state: Optional[dict] = None,
    ) -> dict:
        # label-stable id so re-checkpointing the SAME label overwrites (no duplicate).
        note_id = "checkpoint-" + _slug(label)
        save = self.store_save(content, role="checkpoint", id=note_id)
        if state is not None:
            self.state_snapshot(state, merge=True)
        return {
            "handle": save["handle"],
            "label": label,
            "id": note_id,
            "tokens": save["tokens"],
            "state_updated": state is not None,
        }

    def context_rehydrate(
        self,
        *,
        query: Optional[str] = None,
        handle: Optional[str] = None,
        budget_tokens: Optional[int] = None,
        k: Optional[int] = None,
    ) -> dict:
        if (query is None) == (handle is None):
            raise ValueError("exactly one of query / handle is required")
        budget = (
            self.config.rehydrate_budget_tokens if budget_tokens is None else budget_tokens
        )
        pairs: list[tuple[str, str]] = []
        if handle is not None:
            try:
                content = self.store.get(handle)
            except StoreError:
                content = None
            if content is not None:
                if self.counter.count_text(content) > budget:
                    content = self.counter.truncate_to_tokens(content, budget)
                if content:
                    pairs = [(handle, content)]
        else:
            assert query is not None
            slices = self.store.page_in(
                query, budget, self.counter, k if k is not None else self.config.default_search_k
            )
            pairs = [(s.handle, s.content) for s in slices]

        text = "\n\n".join(f"[[cm:slice handle={h}]]\n{c}" for h, c in pairs)
        # `tokens` reports the budgeted quantity: the retrieved CONTENT tokens
        # (which page_in / the handle-path truncation hold <= budget). The
        # `[[cm:slice …]]` markers are presentation scaffolding and are NOT
        # counted against the budget.
        content_tokens = sum(self.counter.count_text(c) for _, c in pairs)
        return {
            "found": bool(pairs),
            "budget_tokens": budget,
            "count": len(pairs),
            "handles": [h for h, _ in pairs],
            "tokens": content_tokens,
            "text": text,
        }

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self.store.close()
        close = getattr(self.counter, "close", None)
        if callable(close):
            close()
