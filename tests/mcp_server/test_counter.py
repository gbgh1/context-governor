"""HeuristicTokenCounter measurability invariants (spec §3, §7)."""

from __future__ import annotations

import pytest

from contextmanager.mcp.counter import HeuristicTokenCounter
from contextmanager.types import Message


def test_count_text_empty_and_ceil():
    c = HeuristicTokenCounter()
    assert c.count_text("") == 0
    assert c.count_text("a") == 1          # ceil(1/4)
    assert c.count_text("abcd") == 1       # ceil(4/4)
    assert c.count_text("abcde") == 2      # ceil(5/4)


def test_count_messages_includes_overhead():
    c = HeuristicTokenCounter()
    msgs = [Message(role="user", content="abcd", id="1")]
    assert c.count_messages(msgs) == 1 + c.PER_MESSAGE_OVERHEAD


@pytest.mark.parametrize("n", [0, 1, 2, 5, 13, 100])
@pytest.mark.parametrize(
    "text",
    ["", "a", "abcd", "the quick brown fox", "x" * 257, "unicode: café ☕ " * 9],
)
def test_truncate_is_measurable(text, n):
    """count_text(truncate_to_tokens(text, n)) <= n for all text and n>=0."""
    c = HeuristicTokenCounter()
    out = c.truncate_to_tokens(text, n)
    assert c.count_text(out) <= n
    if n <= 0:
        assert out == ""
