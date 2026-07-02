"""THE no-re-fire invariant suite — binds to `tasks/phase1-spec.md` §6 + §8.

All nine tests from spec §8 are implemented here with the EXACT names required:

  1. test_floor_fits_ok
  2. test_floor_exceeds_raises
  3. test_post_compaction_below_low_water          (hypothesis property-based)
  4. test_no_immediate_refire
  5. test_giant_tail_message_paged_out
  6. test_summary_measured_and_truncated
  7. test_idempotent
  8. test_head_never_paged_out
  9. test_forward_progress

The compactor mutates the passed ContextState in place (documented in spec §9) and
returns a CompactionResult. Fakes are deterministic (see conftest.py).

Assumption note: the spec says assert_floor_fits is "called in __init__-time check AND
at the start of compact()". The implementation performs a minimal (ctx-independent)
floor check in __init__ and the full ctx-dependent assert_floor_fits(ctx) at compact()
entry. Tests here exercise assert_floor_fits(ctx) directly AND via compact(), which is
the behaviour the spec's invariant suite is really probing.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from contextmanager import ContractError, InvariantViolationError
from contextmanager.budget import BudgetConfig
from contextmanager.compactor import (
    HysteresisCompactor,
    FloorExceedsTargetError,
)
from contextmanager.types import ContextState, Message

from conftest import (
    FakeCounter,
    FakeSummarizer,
    FakeStore,
    PER_MESSAGE_OVERHEAD,
    build_scenario,
    make_budget,
    msg_of_cost,
)


# ---------------------------------------------------------------------------
# Small shared config used by most tests.
#   n_ctx=10_000, reserved=1_000  -> budget B = 9_000
#   trigger_ratio=0.75 -> high_water = 6_750
#   target_ratio=0.50 -> low_water  = 4_500
#   state_cap=500, distilled_cap=500 -> minimal floor (caps only) = 1_000 <= 4_500  OK
# ---------------------------------------------------------------------------

def standard_budget(
    *,
    state_cap_tokens: int = 500,
    distilled_cap_tokens: int = 500,
    protect_last_n: int = 8,
) -> BudgetConfig:
    return make_budget(
        n_ctx=10_000,
        reserved_headroom_tokens=1_000,
        state_cap_tokens=state_cap_tokens,
        distilled_cap_tokens=distilled_cap_tokens,
        trigger_ratio=0.75,
        target_ratio=0.50,
        protect_first_n=3,
        protect_last_n=protect_last_n,
    )


def small_head() -> list[Message]:
    """A tiny pinned head: P = count_messages(head) = 3*(10+4) = 42 tokens."""
    return [
        msg_of_cost("head0", 10, role="system", pinned=True),
        msg_of_cost("head1", 10, role="system", pinned=True),
        msg_of_cost("head2", 10, role="system", pinned=True),
    ]


def make_state(
    window: list[Message] | None = None,
    head: list[Message] | None = None,
    state_snapshot: str = "",
    distilled_memory: str = "",
) -> ContextState:
    return ContextState(
        head=head if head is not None else small_head(),
        state_snapshot=state_snapshot,
        distilled_memory=distilled_memory,
        window=list(window) if window is not None else [],
    )


# ===========================================================================
# 1. floor_fits_ok
# ===========================================================================


def test_floor_fits_ok() -> None:
    cfg = standard_budget()
    counter, summarizer, store, compactor = build_scenario(
        config=cfg, state=make_state(window=[])
    )
    # F = P(head=42) + state_cap(500) + distilled_cap(500) = 1042 <= low_water(4500).
    compactor.assert_floor_fits(make_state(window=[]))  # must not raise.
    # And compact() on an already-small load does not raise on the floor check.
    ctx = make_state(window=[])
    result = compactor.compact(ctx)
    assert result.fired is False


# ===========================================================================
# 2. floor_exceeds_raises
# ===========================================================================


def test_floor_exceeds_raises() -> None:
    cfg = standard_budget()  # low_water = 4500, caps = 500 + 500 = 1000
    counter, summarizer, store, compactor = build_scenario(
        config=cfg, state=make_state(window=[])
    )
    # Build a head so large that F = P + 1000 > 4500.
    # P = count_messages(head) = 1*(4000+4) = 4004. F = 4004 + 1000 = 5004 > 4500.
    big_head = [msg_of_cost("big0", 4000, role="system", pinned=True)]
    ctx = make_state(head=big_head, window=[])

    # assert_floor_fits raises FloorExceedsTargetError (a ValueError subclass).
    with pytest.raises(FloorExceedsTargetError):
        compactor.assert_floor_fits(ctx)
    assert issubclass(FloorExceedsTargetError, ValueError)

    # compact() must also raise (it calls assert_floor_fits at entry).
    with pytest.raises(FloorExceedsTargetError):
        compactor.compact(ctx)


# ===========================================================================
# 3. post_compaction_below_low_water  (hypothesis property-based)
# ===========================================================================


@given(
    num_messages=st.integers(min_value=1, max_value=30),
    word_counts=st.lists(
        st.integers(min_value=0, max_value=200),
        min_size=1,
        max_size=30,
    ),
)
@settings(max_examples=200, deadline=None)
def test_post_compaction_below_low_water(num_messages: int, word_counts: list[int]) -> None:
    cfg = standard_budget()  # high=6750, low=4500, caps=1000, head P=42 -> F=1042
    counter = FakeCounter()
    summarizer = FakeSummarizer()
    store = FakeStore()
    compactor = HysteresisCompactor(config=cfg, counter=counter, summarizer=summarizer,
                                    store=store)

    # Build window from the generated word counts (truncate/extend to num_messages).
    words = list(word_counts[:num_messages])
    if len(words) < num_messages:
        words += [0] * (num_messages - len(words))
    window = [msg_of_cost(f"m{i}", w) for i, w in enumerate(words)]

    ctx = make_state(window=window)

    # Ensure compaction FIRES: if load < high_water, append one padding message sized
    # so the resulting load exceeds high_water. (Keeps the test meaningful: we only
    # claim load_after <= low_water when compaction actually runs.)
    load_before_pad = compactor.current_load(ctx)
    if load_before_pad < cfg.high_water + 1:
        need = cfg.high_water + 100 - load_before_pad  # extra W needed
        pad_words = max(0, need - PER_MESSAGE_OVERHEAD)
        ctx.window.append(msg_of_cost("pad", pad_words))

    assert compactor.needs_compaction(ctx) is True  # precondition: it fires

    result = compactor.compact(ctx)
    assert result.fired is True
    load_after = compactor.current_load(ctx)
    # The invariant: post-compaction load is at or below low_water, ALWAYS.
    assert load_after <= cfg.low_water, (
        f"load_after={load_after} > low_water={cfg.low_water}"
    )
    assert result.load_after == load_after


# ===========================================================================
# 4. no_immediate_refire
# ===========================================================================


def test_no_immediate_refire() -> None:
    cfg = standard_budget()
    counter, summarizer, store, compactor = build_scenario(
        config=cfg, state=make_state(window=[])
    )
    # Build a window that triggers compaction: 10 messages of 1000 words each.
    # W = 10 * 1004 = 10040; load = 42 + 10040 = 10082 >= 6750 -> fires.
    window = [msg_of_cost(f"m{i}", 1000) for i in range(10)]
    ctx = make_state(window=window)

    assert compactor.needs_compaction(ctx) is True
    result = compactor.compact(ctx)
    assert result.fired is True

    # Immediately after: load <= low_water < high_water => no re-fire.
    assert compactor.current_load(ctx) <= cfg.low_water
    assert compactor.needs_compaction(ctx) is False


# ===========================================================================
# 5. giant_tail_message_paged_out  (THE livelock-killer)
# ===========================================================================


def test_giant_tail_message_paged_out() -> None:
    cfg = standard_budget()  # low_water=4500, high_water=6750
    counter, summarizer, store, compactor = build_scenario(
        config=cfg, state=make_state(window=[])
    )

    # A SINGLE window message whose measured cost alone exceeds low_water (and even
    # exceeds high_water, so compaction fires).
    # measured cost (via count_messages) = 8000 + 4 = 8004 > 4500 and > 6750.
    giant = msg_of_cost("giant", 8000, role="tool", pinned=False)
    ctx = make_state(window=[giant])

    assert compactor.current_load(ctx) >= cfg.high_water  # fires
    result = compactor.compact(ctx)

    assert result.fired is True
    # The giant is paged out (the tail is NOT untouchable).
    assert "giant" in result.paged_out_message_ids
    assert result.used_mechanical_fallback is True
    # After page-out the window is empty -> load drops to P + S + D = F <= low_water.
    assert compactor.current_load(ctx) <= cfg.low_water
    assert result.load_after <= cfg.low_water
    assert ctx.window == []


# ===========================================================================
# 6. summary_measured_and_truncated
# ===========================================================================


def test_summary_measured_and_truncated() -> None:
    # Use a distilled_cap SMALLER than FakeSummarizer's fixed 500-word output, so the
    # measure+truncate path is always exercised.
    cfg = standard_budget(distilled_cap_tokens=100)  # cap=100 < 500 -> truncates
    counter, summarizer, store, compactor = build_scenario(
        config=cfg, state=make_state(window=[])
    )

    # 10 messages; protect_last_n=8 -> the first 2 are sealable.
    window = [msg_of_cost(f"m{i}", 1000) for i in range(10)]
    ctx = make_state(window=window)
    assert compactor.current_load(ctx) >= cfg.high_water  # fires

    result = compactor.compact(ctx)
    assert result.fired is True
    # The sealed summary is stored into distilled_memory and MUST be measured <= cap.
    assert result.summary is not None
    assert counter.count_text(result.summary) <= cfg.distilled_cap_tokens
    assert counter.count_text(ctx.distilled_memory) <= cfg.distilled_cap_tokens
    # The sealed message ids are recorded.
    assert result.sealed_message_ids == ["m0", "m1"]


# ===========================================================================
# 7. idempotent
# ===========================================================================


def test_idempotent() -> None:
    cfg = standard_budget()
    counter, summarizer, store, compactor = build_scenario(
        config=cfg, state=make_state(window=[])
    )
    window = [msg_of_cost(f"m{i}", 1000) for i in range(10)]
    ctx = make_state(window=window)

    first = compactor.compact(ctx)
    assert first.fired is True

    # Second call immediately after: load already <= low_water -> no-op.
    second = compactor.compact(ctx)
    assert second.fired is False
    assert second.sealed_message_ids == []
    assert second.paged_out_message_ids == []
    assert second.summary is None
    assert second.used_mechanical_fallback is False
    # No further state change: load unchanged across the no-op call.
    load_before_second = compactor.current_load(ctx)
    assert second.load_before == load_before_second
    assert second.load_after == load_before_second


# ===========================================================================
# 8. head_never_paged_out
# ===========================================================================


def test_head_never_paged_out() -> None:
    cfg = standard_budget()
    head = small_head()  # 3 pinned head messages
    counter, summarizer, store, compactor = build_scenario(
        config=cfg, state=make_state(head=head, window=[])
    )
    head_ids = {m.id for m in head}

    # Big window to force both sealing and page-out.
    window = [msg_of_cost(f"m{i}", 1000, role="user", pinned=False)
              for i in range(12)]
    ctx = make_state(head=head, window=window)
    assert compactor.current_load(ctx) >= cfg.high_water

    result = compactor.compact(ctx)
    assert result.fired is True

    # Head ids never appear in paged_out.
    for hid in head_ids:
        assert hid not in result.paged_out_message_ids
    # Head messages remain present and intact.
    assert [m.id for m in ctx.head] == [m.id for m in head]
    assert all(m.pinned for m in ctx.head)


# ===========================================================================
# 9. forward_progress
# ===========================================================================


def test_forward_progress() -> None:
    cfg = standard_budget()
    counter, summarizer, store, compactor = build_scenario(
        config=cfg, state=make_state(window=[])
    )
    window = [msg_of_cost(f"m{i}", 800) for i in range(10)]
    ctx = make_state(window=window)

    result = compactor.compact(ctx)
    assert result.fired is True
    # Whenever compaction fires, load strictly decreases.
    assert result.load_after < result.load_before
    # And the post-state is below low_water (forward progress to a safe point).
    assert result.load_after <= cfg.low_water


# ---------------------------------------------------------------------------
# Extra: forward_progress holds for the giant-tail case too.
# ---------------------------------------------------------------------------


def test_forward_progress_giant_tail() -> None:
    cfg = standard_budget()
    counter, summarizer, store, compactor = build_scenario(
        config=cfg, state=make_state(window=[])
    )
    ctx = make_state(window=[msg_of_cost("giant", 8000, role="tool")])
    result = compactor.compact(ctx)
    assert result.fired is True
    assert result.load_after < result.load_before


# ===========================================================================
# Round-2 corrections (spec §10) — stress / boundary tests
# ===========================================================================
#
# These tests bind to the NORMATIVE §10 corrections that make the no-re-fire
# invariant universal. They cover the two critical holes found in code review:
#   C1 — oversized state/distilled tiers blowing the floor formula
#   H2 — pinned messages sitting in the window (must live in head)
#   M4 — bare `assert` postcondition vanishing under `python -O`
# Each test asserts the corrected behaviour: caps are defensively enforced at
# compact() entry, pinned-in-window is a contract error, and the postcondition
# is checked with a typed InvariantViolationError.
# ===========================================================================


# ---------------------------------------------------------------------------
# §10.7 test_oversized_state_snapshot_enforced (C1)
# ---------------------------------------------------------------------------


def test_oversized_state_snapshot_enforced() -> None:
    """A state_snapshot FAR over state_cap_tokens must be defensively truncated
    at compact() entry so the floor formula is a true upper bound on P+S+D.

    After compact(): state_snapshot measures <= state_cap_tokens AND
    current_load(ctx) <= config.low_water, with NO exception raised.
    """
    cfg = standard_budget(state_cap_tokens=100, distilled_cap_tokens=100)
    counter, summarizer, store, compactor = build_scenario(
        config=cfg, state=make_state(window=[])
    )

    # state_snapshot = 7000 distinct words >> state_cap=100. Tiny window.
    state_snapshot = " ".join(f"state_w{i}" for i in range(7000))
    window = [msg_of_cost("w0", 10)]
    ctx = make_state(window=window, state_snapshot=state_snapshot)

    # Sanity: load_before is over high_water purely from the snapshot.
    assert compactor.current_load(ctx) >= cfg.high_water
    assert compactor.needs_compaction(ctx) is True

    # Must not raise — caps are enforced, then the floor formula holds.
    result = compactor.compact(ctx)
    assert result.fired is True

    # The snapshot was truncated to within its cap.
    assert counter.count_text(ctx.state_snapshot) <= cfg.state_cap_tokens
    # Post-compaction load is at or below low_water (no re-fire).
    assert compactor.current_load(ctx) <= cfg.low_water
    assert result.load_after <= cfg.low_water


# ---------------------------------------------------------------------------
# §10.7 test_stale_oversized_distilled_no_sealable (C1)
# ---------------------------------------------------------------------------


def test_stale_oversized_distilled_no_sealable() -> None:
    """distilled_memory FAR over distilled_cap_tokens, with a window that contains
    ONLY protected-tail messages (count <= protect_last_n, nothing sealable).

    After compact(): distilled_memory measures <= distilled_cap_tokens AND
    current_load(ctx) <= config.low_water. The cap is enforced even though no
    sealing occurs (the oversize is stale from a prior turn).
    """
    cfg = standard_budget(state_cap_tokens=100, distilled_cap_tokens=100,
                          protect_last_n=8)
    counter, summarizer, store, compactor = build_scenario(
        config=cfg, state=make_state(window=[])
    )

    # distilled_memory = 7000 distinct words >> distilled_cap=100.
    distilled_memory = " ".join(f"dist_w{i}" for i in range(7000))
    # Window with 5 messages — all within protect_last_n=8, so NOTHING sealable.
    window = [msg_of_cost(f"m{i}", 10) for i in range(5)]
    ctx = make_state(window=window, distilled_memory=distilled_memory)

    # Sanity: load_before is over high_water purely from distilled tier.
    assert compactor.current_load(ctx) >= cfg.high_water
    assert compactor.needs_compaction(ctx) is True

    # Must not raise.
    result = compactor.compact(ctx)
    assert result.fired is True

    # Distilled tier truncated to within its cap.
    assert counter.count_text(ctx.distilled_memory) <= cfg.distilled_cap_tokens
    # No sealing happened (nothing sealable) — window messages all retained.
    assert result.sealed_message_ids == []
    # Post-compaction load <= low_water.
    assert compactor.current_load(ctx) <= cfg.low_water
    assert result.load_after <= cfg.low_water


# ---------------------------------------------------------------------------
# §10.7 test_pinned_in_window_rejected (H2)
# ---------------------------------------------------------------------------


def test_pinned_in_window_rejected() -> None:
    """A pinned=True message placed in window is a contract violation: pinned
    content must live in ctx.head. compact() must raise ContractError naming
    the offending id(s).
    """
    cfg = standard_budget()
    counter, summarizer, store, compactor = build_scenario(
        config=cfg, state=make_state(window=[])
    )

    # A pinned message in the window, sized so load >= high_water (compaction
    # fires; the pinned check runs after the no-op check per §10.3).
    pinned_in_window = msg_of_cost("pinned_in_win", 8000,
                                   role="system", pinned=True)
    ctx = make_state(window=[pinned_in_window])

    assert compactor.needs_compaction(ctx) is True  # would fire

    with pytest.raises(ContractError) as excinfo:
        compactor.compact(ctx)
    # The offending id is named in the message.
    assert "pinned_in_win" in str(excinfo.value)


# ---------------------------------------------------------------------------
# §10.3 / §10.1 InvariantViolationError is the postcondition guard type
# ---------------------------------------------------------------------------
#
# The corrected postcondition (§10.3 #3) raises InvariantViolationError when
# load_after > low_water. After the §10 fixes, this is NOT reachable with any
# well-behaved TokenCounter: assert_floor_fits at entry + defensive cap
# enforcement + the page-out loop together guarantee load_after <= F <=
# low_water by construction. The guard is defense-in-depth against an
# internal implementation bug or a TokenCounter that violates its monotonicity
# contract.
#
# We construct the ONLY reachable breach: a TokenCounter whose
# count_messages(head) is non-monotonic — it reports a small value on the first
# call (so assert_floor_fits passes) and a large value on every subsequent
# call (so the postcondition measure sees load > low_water). This deliberately
# violates the TokenCounter contract; the test proves the guard raises the
# SPECIFIED typed exception (InvariantViolationError) rather than silently
# returning an over-budget state or dying on a bare `assert` under `python -O`.
# ---------------------------------------------------------------------------


class _NonMonotonicCounter:
    """Hostile TokenCounter for the InvariantViolationError reachability test.

    count_messages returns `small` on the FIRST call (the assert_floor_fits
    invocation inside compact()) and `large` on every subsequent call. This
    breaks the monotonicity contract a real TokenCounter upholds. count_text
    and truncate_to_tokens remain honest so cap enforcement still works.
    """

    def __init__(self, small: int, large: int) -> None:
        self.small = small
        self.large = large
        self._calls = 0

    def count_text(self, text: str) -> int:
        return len(text.split())

    def count_messages(self, messages: list[Message]) -> int:
        self._calls += 1
        return self.small if self._calls == 1 else self.large

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])


def test_postcondition_guard_raises_invariant_violation() -> None:
    """If the postcondition load_after <= low_water is breached (here by a
    contract-violating non-monotonic counter), compact() must raise
    InvariantViolationError — the typed guard from §10.1, NOT a bare assert.
    """
    cfg = standard_budget(state_cap_tokens=10, distilled_cap_tokens=10)
    # low_water = 4500, high_water = 6750.
    # First-call small: 0 (so assert_floor_fits sees F = 0 + 10 + 10 = 20 <= 4500).
    # Subsequent large: 10000 (so every later current_load sees load >> 4500).
    counter = _NonMonotonicCounter(small=0, large=10000)
    summarizer = FakeSummarizer()
    store = FakeStore()
    compactor = HysteresisCompactor(config=cfg, counter=counter,
                                   summarizer=summarizer, store=store)

    # 10 messages so Step A has 2 sealable (protect_last_n=8); after sealing the
    # window still has 8 messages for Step B to drain. Each current_load call
    # after the first returns large=10000, so load stays >> low_water and the
    # page-out loop drains the window to empty; the final measure then sees
    # load = 10000 (head) + S + D + 0 > 4500 -> InvariantViolationError.
    window = [msg_of_cost(f"m{i}", 50) for i in range(10)]
    ctx = make_state(window=window)

    with pytest.raises(InvariantViolationError):
        compactor.compact(ctx)

    # Confirm the guard is the spec-defined type (RuntimeError subclass per §10.1).
    assert issubclass(InvariantViolationError, RuntimeError)

