from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import ContextState, Message, TokenCounter, Summarizer, Store
from .budget import BudgetConfig
from .sealing import seal_summary


class FloorExceedsTargetError(ValueError):
    """Raised when the floor F = P + state_cap + distilled_cap exceeds low_water."""


class ContractError(ValueError):
    """Raised when a caller violates an input contract (e.g. pinned msg in window)."""


class InvariantViolationError(RuntimeError):
    """Raised when an internal guarantee failed (e.g. post-compaction load > low_water)."""


def _enforce_cap(text: str, cap: int, counter: TokenCounter) -> str:
    """
    Measure `text`; if over `cap`, call `counter.truncate_to_tokens(text, cap)`, then
    RE-MEASURE and repeat with a shrinking target (cap, cap-1, ...) until the measured
    length is <= cap OR the text is empty. Returns the capped text.

    This tolerates real-tokenizer truncation overshoot (where truncate_to_tokens
    returns slightly more than the requested target). If convergence is impossible
    (cap <= 0 with non-empty text, or truncation keeps overshooting down to target=1),
    returns "" which trivially measures 0 <= cap.
    """
    measured = counter.count_text(text)
    if measured <= cap:
        return text
    target = cap
    while target > 0 and text:
        text = counter.truncate_to_tokens(text, target)
        measured = counter.count_text(text)
        if measured <= cap:
            return text
        target -= 1
    # Could not get under cap with a positive target, or text is empty.
    return ""


@dataclass
class CompactionResult:
    fired: bool
    load_before: int
    load_after: int
    sealed_message_ids: list[str]
    paged_out_message_ids: list[str]
    summary: Optional[str]
    used_mechanical_fallback: bool


class HysteresisCompactor:
    def __init__(
        self,
        config: BudgetConfig,
        counter: TokenCounter,
        summarizer: Summarizer,
        store: Store,
    ) -> None:
        self.config = config
        self.counter = counter
        self.summarizer = summarizer
        self.store = store
        # Construction-time floor check (the part of the floor that does not depend on
        # ctx.head): the minimal floor with P=0 is state_cap_tokens + distilled_cap_tokens.
        # If even that exceeds low_water, the full floor F = P + caps can never fit for any
        # head, so the compactor is unusable -> reject eagerly. The head-dependent part is
        # re-checked via assert_floor_fits(ctx) at compact-time.
        min_floor = config.state_cap_tokens + config.distilled_cap_tokens
        if min_floor > config.low_water:
            raise FloorExceedsTargetError(
                f"Floor exceeds low_water by construction: minimal floor "
                f"(state_cap_tokens + distilled_cap_tokens)={min_floor} > "
                f"low_water={config.low_water}. Reduce the caps or raise target_ratio."
            )

    def current_load(self, ctx: ContextState) -> int:
        """P + S + D + W, all measured via counter."""
        p = self.counter.count_messages(ctx.head)
        s = self.counter.count_text(ctx.state_snapshot)
        d = self.counter.count_text(ctx.distilled_memory)
        w = self.counter.count_messages(ctx.window)
        return p + s + d + w

    def needs_compaction(self, ctx: ContextState) -> bool:
        """current_load(ctx) >= config.high_water"""
        return self.current_load(ctx) >= self.config.high_water

    def assert_floor_fits(self, ctx: ContextState) -> None:
        """
        Compute F = count_messages(head) + state_cap_tokens + distilled_cap_tokens.
        If F > config.low_water: raise FloorExceedsTargetError (a ValueError subclass)
        with a message naming F and low_water.
        """
        p = self.counter.count_messages(ctx.head)
        f = p + self.config.state_cap_tokens + self.config.distilled_cap_tokens
        if f > self.config.low_water:
            raise FloorExceedsTargetError(
                f"Floor F={f} exceeds low_water={self.config.low_water}; "
                f"P(head)={p}, state_cap_tokens={self.config.state_cap_tokens}, "
                f"distilled_cap_tokens={self.config.distilled_cap_tokens}. "
                f"Compaction cannot reach the target by construction."
            )

    def compact(self, ctx: ContextState) -> CompactionResult:
        """
        Precondition check: assert_floor_fits(ctx).
        If not needs_compaction(ctx): return CompactionResult(fired=False, ... no-op).

        Step A — SEAL the middle.
        Step B — MECHANICAL FALLBACK.
        Postcondition asserted.
        """
        # By-construction floor check, at compact-time as well as __init__-time.
        # (Spec calls assert_floor_fits at construction time; here we also have ctx
        # in hand, which is what the floor actually depends on.)
        self.assert_floor_fits(ctx)

        load_before = self.current_load(ctx)

        if not self.needs_compaction(ctx):
            return CompactionResult(
                fired=False,
                load_before=load_before,
                load_after=load_before,
                sealed_message_ids=[],
                paged_out_message_ids=[],
                summary=None,
                used_mechanical_fallback=False,
            )

        # §10.3.1 — Reject pinned window messages (H2). Pinned content must live in
        # ctx.head; a pinned message in ctx.window would be both protected from
        # page-out AND counted in W, breaking the floor math.
        pinned_in_window = [m.id for m in ctx.window if m.pinned]
        if pinned_in_window:
            raise ContractError(
                f"Pinned messages must live in ctx.head, not ctx.window; "
                f"offending ids: {pinned_in_window}"
            )

        # §10.3.2 — Enforce the caps defensively (C1, M4). The floor formula
        # F = P + state_cap_tokens + distilled_cap_tokens is only a true upper
        # bound on P + S + D if S <= state_cap_tokens and D <= distilled_cap_tokens
        # at compact() entry. A stale/oversized snapshot or distilled memory could
        # otherwise violate the floor and break the no-re-fire invariant.
        ctx.state_snapshot = _enforce_cap(
            ctx.state_snapshot, self.config.state_cap_tokens, self.counter
        )
        ctx.distilled_memory = _enforce_cap(
            ctx.distilled_memory, self.config.distilled_cap_tokens, self.counter
        )

        sealed_ids: list[str] = []
        paged_out_ids: list[str] = []
        used_mechanical_fallback = False
        summary: Optional[str] = None

        # Step A — SEAL the middle.
        protect_last_n = self.config.protect_last_n
        window = ctx.window
        # Sealable = all window messages except the last protect_last_n and any pinned.
        # Determine the index boundary: the last protect_last_n messages are protected.
        n = len(window)
        # The "tail" protected set = the last protect_last_n messages.
        tail_start = max(0, n - protect_last_n)
        sealable_indices = [
            i for i in range(n)
            if i < tail_start and not window[i].pinned
        ]
        if sealable_indices:
            sealable_msgs = [window[i] for i in sealable_indices]
            summary = seal_summary(
                sealable_msgs,
                ctx.distilled_memory,
                self.counter,
                self.summarizer,
                cap_tokens=self.config.distilled_cap_tokens,
            )
            ctx.distilled_memory = summary
            sealed_ids = [m.id for m in sealable_msgs]
            # Remove sealed messages from window (preserving order of the rest).
            sealable_id_set = set(sealed_ids)
            ctx.window = [m for m in window if m.id not in sealable_id_set]

        # Step B — MECHANICAL FALLBACK (the Hermes-killer).
        while self.current_load(ctx) > self.config.low_water and ctx.window:
            # Victim = OLDEST non-pinned window message.
            victim_idx: Optional[int] = None
            for i, m in enumerate(ctx.window):
                if not m.pinned:
                    victim_idx = i
                    break
            if victim_idx is None:
                # No non-pinned window message remains; cannot reduce further via
                # page-out. The floor guarantee ensures load already <= low_water in
                # this case (W=0 effectively). Break to avoid an infinite loop.
                break
            victim = ctx.window.pop(victim_idx)
            self.store.page_out(victim)
            paged_out_ids.append(victim.id)
            used_mechanical_fallback = True

        load_after = self.current_load(ctx)

        # §10.3.3 — Postcondition without bare assert (M4). A bare `assert` is
        # stripped under `python -O`; use a real exception so the invariant
        # survives optimization. Combined with §10.2 (low_water < high_water
        # guaranteed for every constructible config), load_after <= low_water
        # implies load_after < high_water => needs_compaction is False => no re-fire.
        if load_after > self.config.low_water:
            raise InvariantViolationError(
                f"post-compaction load_after={load_after} > "
                f"low_water={self.config.low_water}"
            )

        return CompactionResult(
            fired=True,
            load_before=load_before,
            load_after=load_after,
            sealed_message_ids=sealed_ids,
            paged_out_message_ids=paged_out_ids,
            summary=summary,
            used_mechanical_fallback=used_mechanical_fallback,
        )
