"""Shared fixtures and deterministic fakes for the Phase 1 test suite.

All names bind to `tasks/phase1-spec.md` (NORMATIVE). The fakes here are deliberately
deterministic and measurable so the no-re-fire invariant can be asserted exactly.

`FakeCounter` token accounting:
  - count_text(text)  = number of whitespace-split words (overridable per-string via
    `cost_overrides`).
  - count_messages    = sum(count_text(content)) + PER_MESSAGE_OVERHEAD for each message
    (mimics role/template markers the real server adds).
  - truncate_to_tokens(text, n) = first `n` words rejoined with single spaces, so the
    result measures EXACTLY min(n, words_in_text) tokens. Deterministic and measurable.

Consequently the "cost" of a `msg_of_cost(id, n)` message, as observed by the
compactor's `current_load` (which uses count_messages for window/head), is `n +
PER_MESSAGE_OVERHEAD`. Tests that need a message "cost" refer to that measured value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from contextmanager.types import Message, ContextState, TokenCounter, Summarizer, Store
from contextmanager.budget import BudgetConfig
from contextmanager.compactor import HysteresisCompactor


# ---------------------------------------------------------------------------
# FakeCounter
# ---------------------------------------------------------------------------

PER_MESSAGE_OVERHEAD = 4


class FakeCounter:
    """Deterministic TokenCounter implementation.

    Token cost of a string = number of whitespace-split words, unless the exact string
    is present in `cost_overrides`, in which case the override is used. This lets tests
    pin a specific cost to a specific content string when convenient.
    """

    def __init__(self, cost_overrides: Optional[dict[str, int]] = None) -> None:
        self.cost_overrides: dict[str, int] = dict(cost_overrides or {})

    # -- TokenCounter protocol ------------------------------------------------

    def count_text(self, text: str) -> int:
        if text in self.cost_overrides:
            return self.cost_overrides[text]
        return len(text.split())

    def count_messages(self, messages: list[Message]) -> int:
        total = 0
        for m in messages:
            total += self.count_text(m.content)
            total += PER_MESSAGE_OVERHEAD
        return total

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])


# ---------------------------------------------------------------------------
# FakeSummarizer
# ---------------------------------------------------------------------------

# Fixed summary body: 500 distinct words so it always exceeds any reasonable cap and
# forces seal_summary's measure+truncate path.
SUMMARY_WORDS = 500
FIXED_SUMMARY = " ".join(f"sumword{i}" for i in range(SUMMARY_WORDS))


class FakeSummarizer:
    """Summarizer that always returns the same long string regardless of input.

    The returned length (500 words) is deliberately large so `seal_summary` is
    guaranteed to measure it as over cap and truncate — exercising the invariant
    "never trust the model's self-reported length".
    """

    def summarize(
        self,
        messages: list[Message],
        prior_summary: Optional[str],
        target_tokens: int,
    ) -> str:
        return FIXED_SUMMARY


# ---------------------------------------------------------------------------
# FakeStore
# ---------------------------------------------------------------------------


class FakeStore:
    """In-memory Store: page_out stores content under a handle `h{N}`."""

    def __init__(self) -> None:
        self._records: dict[str, str] = {}
        self._counter = 0
        self.paged_in_ids: list[str] = []  # message ids paged out, in order

    def page_out(self, message: Message) -> str:
        self._counter += 1
        handle = f"h{self._counter}"
        self._records[handle] = message.content
        self.paged_in_ids.append(message.id)
        return handle

    def get(self, handle: str) -> str:
        return self._records[handle]


# ---------------------------------------------------------------------------
# Message builder helpers
# ---------------------------------------------------------------------------


def msg_of_cost(
    id: str,
    n: int,
    role: str = "user",
    pinned: bool = False,
) -> Message:
    """Build a Message whose content has EXACTLY `n` whitespace-split words.

    `count_text(content) == n` (content is `n` distinct words). The compactor-measured
    cost via count_messages is `n + PER_MESSAGE_OVERHEAD`.
    """
    if n <= 0:
        content = ""
    else:
        content = " ".join(f"{id}_w{i}" for i in range(n))
    return Message(role=role, content=content, id=id, pinned=pinned)


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------


def make_budget(
    n_ctx: int = 10000,
    reserved_headroom_tokens: int = 1000,
    state_cap_tokens: int = 500,
    distilled_cap_tokens: int = 500,
    trigger_ratio: float = 0.75,
    target_ratio: float = 0.50,
    protect_first_n: int = 3,
    protect_last_n: int = 8,
) -> BudgetConfig:
    return BudgetConfig(
        n_ctx=n_ctx,
        reserved_headroom_tokens=reserved_headroom_tokens,
        state_cap_tokens=state_cap_tokens,
        distilled_cap_tokens=distilled_cap_tokens,
        trigger_ratio=trigger_ratio,
        target_ratio=target_ratio,
        protect_first_n=protect_first_n,
        protect_last_n=protect_last_n,
    )


def build_scenario(
    *,
    config: BudgetConfig,
    state: ContextState,
    counter: Optional[FakeCounter] = None,
    summarizer: Optional[FakeSummarizer] = None,
    store: Optional[FakeStore] = None,
) -> tuple[FakeCounter, FakeSummarizer, FakeStore, HysteresisCompactor]:
    """Wire fakes + compactor for a given config + state and return them.

    Does NOT run assert_floor_fits beyond what HysteresisCompactor.__init__ does.
    """
    counter = counter if counter is not None else FakeCounter()
    summarizer = summarizer if summarizer is not None else FakeSummarizer()
    store = store if store is not None else FakeStore()
    compactor = HysteresisCompactor(
        config=config,
        counter=counter,
        summarizer=summarizer,
        store=store,
    )
    return counter, summarizer, store, compactor


# ---------------------------------------------------------------------------
# pytest fixtures (thin wrappers around the builders above)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_counter() -> FakeCounter:
    return FakeCounter()


@pytest.fixture
def fake_summarizer() -> FakeSummarizer:
    return FakeSummarizer()


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def default_budget() -> BudgetConfig:
    return make_budget()
