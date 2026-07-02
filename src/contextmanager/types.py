from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional


@dataclass
class Message:
    role: str                 # "system" | "user" | "assistant" | "tool"
    content: str
    id: str                   # stable unique id
    pinned: bool = False      # part of the protected head; never paged out
    sealed: bool = False      # already represented inside distilled memory


@dataclass
class ContextState:
    """The reconstructable context for one turn, by tier."""
    head: list[Message]            # pinned spec: system prompt + protect_first_n
    state_snapshot: str            # authoritative state.json rendered to text ("" if none)
    distilled_memory: str          # sealed rolling summary ("" if none)
    window: list[Message]          # recent messages + retrieved slices (compactable)


class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...
    def count_messages(self, messages: list[Message]) -> int: ...
    # Truncate text to AT MOST max_tokens tokens (by real tokenization), returning text.
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str: ...


class Summarizer(Protocol):
    # Produce a handoff summary of `messages`, optionally folding in prior_summary.
    # The RETURNED length is NOT trusted; the caller measures+truncates. target_tokens
    # is only a hint passed to the model.
    def summarize(self, messages: list[Message], prior_summary: Optional[str],
                  target_tokens: int) -> str: ...


class Store(Protocol):
    # Page a message's full content out to durable storage; return a short handle string.
    def page_out(self, message: Message) -> str: ...
    def get(self, handle: str) -> str: ...
