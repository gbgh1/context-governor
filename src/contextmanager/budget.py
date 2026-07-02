from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetConfig:
    n_ctx: int                          # from /props default_generation_settings.n_ctx
    reserved_headroom_tokens: int       # H: reserved for generation + safety margin
    state_cap_tokens: int               # S_max: hard cap for state_snapshot
    distilled_cap_tokens: int           # D_max: hard cap for distilled_memory
    trigger_ratio: float = 0.75         # high-water, fraction of usable budget B
    target_ratio: float = 0.50          # low-water,  fraction of usable budget B
    protect_first_n: int = 3
    protect_last_n: int = 8

    def __post_init__(self) -> None:
        # validate: 0 < target_ratio < trigger_ratio < 1
        if not (0.0 < self.target_ratio < self.trigger_ratio < 1.0):
            raise ValueError(
                f"Require 0 < target_ratio < trigger_ratio < 1; got "
                f"target_ratio={self.target_ratio}, trigger_ratio={self.trigger_ratio}"
            )
        # validate: 0 < reserved_headroom_tokens < n_ctx
        if not (0 < self.reserved_headroom_tokens < self.n_ctx):
            raise ValueError(
                f"Require 0 < reserved_headroom_tokens < n_ctx; got "
                f"reserved_headroom_tokens={self.reserved_headroom_tokens}, "
                f"n_ctx={self.n_ctx}"
            )
        # validate: state_cap_tokens >= 0, distilled_cap_tokens >= 0
        if self.state_cap_tokens < 0:
            raise ValueError(
                f"state_cap_tokens must be >= 0; got {self.state_cap_tokens}"
            )
        if self.distilled_cap_tokens < 0:
            raise ValueError(
                f"distilled_cap_tokens must be >= 0; got {self.distilled_cap_tokens}"
            )
        # §10.2: enforce integer water separation (fixes C2). The ratio check above
        # guarantees target_ratio < trigger_ratio, but after int() truncation the
        # integer waters can collapse (e.g. budget=10, target=0.10, trigger=0.11 ->
        # low=1, high=1). A collapsed water pair breaks the no-re-fire invariant
        # (load <= low_water would NOT be < high_water), so reject it here.
        if not (self.low_water < self.high_water):
            raise ValueError(
                f"integer waters collapsed: low_water={self.low_water} not < "
                f"high_water={self.high_water} (budget={self.budget}); "
                f"increase n_ctx/budget or widen the ratio gap."
            )

    @property
    def budget(self) -> int:        # B = usable prompt budget
        return self.n_ctx - self.reserved_headroom_tokens

    @property
    def high_water(self) -> int:    # T_hi
        return int(self.trigger_ratio * self.budget)

    @property
    def low_water(self) -> int:     # T_lo
        return int(self.target_ratio * self.budget)
