"""seal_summary tests — binds to `tasks/phase1-spec.md` §5 (NORMATIVE).

The contract:
  1. text = summarizer.summarize(messages, prior_summary, target_tokens=cap_tokens)
  2. MEASURE: counter.count_text(text)
  3. If over cap_tokens -> counter.truncate_to_tokens(text, cap_tokens)
  4. Return text guaranteed counter.count_text(result) <= cap_tokens.
"""

from __future__ import annotations

from contextmanager.sealing import seal_summary
from contextmanager.types import Message

from conftest import (
    FakeCounter,
    FakeSummarizer,
    FIXED_SUMMARY,
    SUMMARY_WORDS,
    msg_of_cost,
)


# ---------------------------------------------------------------------------
# Over cap -> measured and truncated to <= cap
# ---------------------------------------------------------------------------


def test_seal_summary_truncates_when_over_cap() -> None:
    counter = FakeCounter()
    summarizer = FakeSummarizer()
    cap = 100

    # FakeSummarizer returns FIXED_SUMMARY (500 words), well over cap=100.
    assert SUMMARY_WORDS == 500 > cap

    messages = [msg_of_cost("m1", 10), msg_of_cost("m2", 10)]
    result = seal_summary(messages, prior_summary=None, counter=counter,
                         summarizer=summarizer, cap_tokens=cap)

    # Postcondition: measured result MUST be <= cap.
    assert counter.count_text(result) <= cap
    # And the truncation actually happened (result is shorter than the raw summary).
    assert counter.count_text(result) < counter.count_text(FIXED_SUMMARY)
    # FakeCounter.truncate_to_tokens rejoins the first `cap` words -> exactly `cap`.
    assert counter.count_text(result) == cap


def test_seal_summary_does_not_trust_summarizer_reported_length() -> None:
    """Even though target_tokens=cap is passed as a hint, the result is MEASURED, not
    assumed to be at/under cap just because the summarizer was asked nicely."""
    counter = FakeCounter()
    summarizer = FakeSummarizer()
    cap = 50

    result = seal_summary(
        [msg_of_cost("m1", 3)],
        prior_summary="",
        counter=counter,
        summarizer=summarizer,
        cap_tokens=cap,
    )
    # The measured result must satisfy the cap regardless of what the summarizer did.
    assert counter.count_text(result) <= cap


# ---------------------------------------------------------------------------
# Under cap -> returned unchanged (no truncation)
# ---------------------------------------------------------------------------


class ShortSummarizer:
    """A summarizer that returns a short string under cap, to verify the under-cap
    path returns text unchanged (no truncation, no mutation)."""

    def summarize(self, messages, prior_summary, target_tokens) -> str:  # type: ignore[no-untyped-def]
        return "tiny summary"


def test_seal_summary_returns_unchanged_when_under_cap() -> None:
    counter = FakeCounter()
    short = ShortSummarizer()
    cap = 10_000  # far above the 2-word summary

    result = seal_summary(
        [msg_of_cost("m1", 5)],
        prior_summary=None,
        counter=counter,
        summarizer=short,
        cap_tokens=cap,
    )

    assert result == "tiny summary"
    assert counter.count_text(result) == 2  # "tiny summary" -> 2 words
    assert counter.count_text(result) <= cap


def test_seal_summary_at_exactly_cap_is_unchanged() -> None:
    """Boundary: measured == cap means NOT over -> returned unchanged."""

    class ExactSummarizer:
        def summarize(self, messages, prior_summary, target_tokens) -> str:  # type: ignore[no-untyped-def]
            return " ".join(f"w{i}" for i in range(20))  # exactly 20 words

    counter = FakeCounter()
    cap = 20
    result = seal_summary(
        [msg_of_cost("m1", 5)],
        prior_summary=None,
        counter=counter,
        summarizer=ExactSummarizer(),
        cap_tokens=cap,
    )
    assert counter.count_text(result) == cap
    # Unchanged: still the 20 generated words.
    assert result == " ".join(f"w{i}" for i in range(20))


# ---------------------------------------------------------------------------
# Empty input edge cases
# ---------------------------------------------------------------------------


def test_seal_summary_empty_messages_still_measures_result() -> None:
    counter = FakeCounter()
    summarizer = FakeSummarizer()
    cap = 30

    result = seal_summary(
        [],
        prior_summary=None,
        counter=counter,
        summarizer=summarizer,
        cap_tokens=cap,
    )
    assert counter.count_text(result) <= cap


# ---------------------------------------------------------------------------
# prior_summary is forwarded to summarizer (smoke check the wiring)
# ---------------------------------------------------------------------------


def test_seal_summary_forwards_prior_summary_to_summarizer() -> None:
    seen: dict = {}

    class CapturingSummarizer:
        def summarize(self, messages, prior_summary, target_tokens):  # type: ignore[no-untyped-def]
            seen["prior"] = prior_summary
            seen["target"] = target_tokens
            return "ok"

    result = seal_summary(
        [msg_of_cost("m1", 1)],
        prior_summary="PREVIOUS SUMMARY",
        counter=FakeCounter(),
        summarizer=CapturingSummarizer(),
        cap_tokens=100,
    )
    assert seen["prior"] == "PREVIOUS SUMMARY"
    assert seen["target"] == 100
    assert result == "ok"


# ===========================================================================
# Round-2 correction §10.4 / §10.7 — re-measure/shrink loop (fixes H1)
# ===========================================================================
#
# seal_summary must use the same _enforce_cap semantics (or an inline
# equivalent): after truncation, RE-MEASURE and shrink the target until
# count_text(result) <= cap_tokens. A single truncate call is NOT trusted,
# because real tokenizers can OVERSHOOT the requested token count.
# ===========================================================================


class OvershootCounter:
    """A TokenCounter whose truncate_to_tokens OVERSHOOTS the requested token
    count by a FIXED amount, mimicking real-tokenizer truncation overshoot.

    truncate_to_tokens(text, max_tokens) returns the first
    (max_tokens + OVERSHOOT) words (when the text is longer than that). So a
    request for `max_tokens` tokens yields a result measuring
    `max_tokens + OVERSHOOT` tokens — strictly over the request.

    The overshoot is FIXED (does not grow), so as the _enforce_cap loop shrinks
    its target (cap, cap-1, cap-2, ...), the measured result shrinks with it
    and converges: at target = cap - OVERSHOOT, the result measures exactly
    `cap` tokens, which is `<= cap` => converges.
    """

    OVERSHOOT = 5

    def count_text(self, text: str) -> int:
        return len(text.split())

    def count_messages(self, messages: list[Message]) -> int:  # unused here
        return sum(self.count_text(m.content) for m in messages) + 4 * len(messages)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        words = text.split()
        if len(words) <= max_tokens:
            return text
        # Overshoot: take more words than requested.
        take = max_tokens + self.OVERSHOOT
        return " ".join(words[:take])


def test_seal_truncation_reconfirmed() -> None:
    """A fake counter whose truncate_to_tokens OVERSHOOTS by a fixed amount
    must still yield a seal_summary result measuring <= cap_tokens — proving
    the re-measure/shrink loop in seal_summary (§10.4), not a single truncate
    call that trusts the truncator.
    """
    counter = OvershootCounter()
    summarizer = FakeSummarizer()  # returns FIXED_SUMMARY = 500 words
    cap = 100

    assert counter.count_text(FIXED_SUMMARY) == SUMMARY_WORDS  # 500
    assert SUMMARY_WORDS > cap

    result = seal_summary(
        [msg_of_cost("m1", 10)],
        prior_summary=None,
        counter=counter,
        summarizer=summarizer,
        cap_tokens=cap,
    )

    # The result MUST be measured <= cap, despite the truncator overshooting.
    measured = counter.count_text(result)
    assert measured <= cap, (
        f"seal_summary returned text measuring {measured} > cap={cap}; "
        f"the re-measure/shrink loop did not converge."
    )
    # And it is non-trivially truncated (well below the raw 500-word summary).
    assert measured < SUMMARY_WORDS

