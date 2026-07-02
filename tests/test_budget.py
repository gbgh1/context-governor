"""BudgetConfig validation + tier math tests.

Binds to `tasks/phase1-spec.md` §4 (NORMATIVE):
  budget      = n_ctx - reserved_headroom_tokens
  high_water  = int(trigger_ratio * budget)
  low_water   = int(target_ratio  * budget)
"""

from __future__ import annotations

import pytest

from contextmanager.budget import BudgetConfig


# ---------------------------------------------------------------------------
# Tier math
# ---------------------------------------------------------------------------


def test_budget_is_n_ctx_minus_reserved() -> None:
    cfg = BudgetConfig(
        n_ctx=10_000,
        reserved_headroom_tokens=1_000,
        state_cap_tokens=500,
        distilled_cap_tokens=500,
    )
    assert cfg.budget == 10_000 - 1_000
    assert cfg.budget == 9_000


def test_high_water_is_int_trigger_ratio_times_budget() -> None:
    cfg = BudgetConfig(
        n_ctx=10_000,
        reserved_headroom_tokens=1_000,
        state_cap_tokens=500,
        distilled_cap_tokens=500,
        trigger_ratio=0.75,
        target_ratio=0.50,
    )
    assert cfg.high_water == int(0.75 * 9_000)
    assert cfg.high_water == 6_750


def test_low_water_is_int_target_ratio_times_budget() -> None:
    cfg = BudgetConfig(
        n_ctx=10_000,
        reserved_headroom_tokens=1_000,
        state_cap_tokens=500,
        distilled_cap_tokens=500,
        trigger_ratio=0.75,
        target_ratio=0.50,
    )
    assert cfg.low_water == int(0.50 * 9_000)
    assert cfg.low_water == 4_500


def test_high_water_strictly_greater_than_low_water() -> None:
    cfg = BudgetConfig(
        n_ctx=10_000,
        reserved_headroom_tokens=1_000,
        state_cap_tokens=500,
        distilled_cap_tokens=500,
        trigger_ratio=0.75,
        target_ratio=0.50,
    )
    assert cfg.low_water < cfg.high_water
    assert cfg.low_water < cfg.budget
    assert cfg.high_water < cfg.budget


def test_int_truncation_matches_spec_formula() -> None:
    # Pick values where int() truncation is non-trivial (not already integers).
    cfg = BudgetConfig(
        n_ctx=10_000,
        reserved_headroom_tokens=3,
        state_cap_tokens=10,
        distilled_cap_tokens=10,
        trigger_ratio=0.75,
        target_ratio=0.50,
    )
    budget = 10_000 - 3  # 9997
    assert cfg.high_water == int(0.75 * 9997)  # int(7497.75) == 7497
    assert cfg.low_water == int(0.50 * 9997)  # int(4998.5) == 4998
    # Sanity: int() truncates toward zero for positive floats.
    assert cfg.high_water == 7497
    assert cfg.low_water == 4998


# ---------------------------------------------------------------------------
# Validation: ratios
# ---------------------------------------------------------------------------


def _base_kwargs() -> dict:
    return dict(
        n_ctx=10_000,
        reserved_headroom_tokens=1_000,
        state_cap_tokens=500,
        distilled_cap_tokens=500,
        trigger_ratio=0.75,
        target_ratio=0.50,
    )


def test_post_init_rejects_target_ratio_ge_trigger_ratio() -> None:
    # Equal
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "trigger_ratio": 0.60, "target_ratio": 0.60})
    # target > trigger
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "trigger_ratio": 0.50, "target_ratio": 0.60})


def test_post_init_rejects_ratios_outside_zero_one() -> None:
    # trigger_ratio out of (0,1)
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "trigger_ratio": 1.0, "target_ratio": 0.5})
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "trigger_ratio": 0.0, "target_ratio": 0.0})
    # target_ratio <= 0
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "trigger_ratio": 0.75, "target_ratio": 0.0})
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "trigger_ratio": 0.75, "target_ratio": -0.1})
    # trigger_ratio >= 1
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "trigger_ratio": 1.5, "target_ratio": 0.5})
    # target_ratio >= 1 (also breaks target < trigger if trigger < 1)
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "trigger_ratio": 0.75, "target_ratio": 1.0})


# ---------------------------------------------------------------------------
# Validation: reserved_headroom_tokens
# ---------------------------------------------------------------------------


def test_post_init_rejects_reserved_headroom_ge_n_ctx() -> None:
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "reserved_headroom_tokens": 10_000})
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "reserved_headroom_tokens": 10_001})


def test_post_init_rejects_reserved_headroom_le_zero() -> None:
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "reserved_headroom_tokens": 0})
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "reserved_headroom_tokens": -5})


# ---------------------------------------------------------------------------
# Validation: caps non-negative
# ---------------------------------------------------------------------------


def test_post_init_rejects_negative_state_cap() -> None:
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "state_cap_tokens": -1})


def test_post_init_rejects_negative_distilled_cap() -> None:
    with pytest.raises(ValueError):
        BudgetConfig(**{**_base_kwargs(), "distilled_cap_tokens": -1})


def test_post_init_allows_zero_caps() -> None:
    # Caps may be zero (e.g. no state snapshot in this turn).
    cfg = BudgetConfig(
        n_ctx=10_000,
        reserved_headroom_tokens=1_000,
        state_cap_tokens=0,
        distilled_cap_tokens=0,
    )
    assert cfg.state_cap_tokens == 0
    assert cfg.distilled_cap_tokens == 0


# ---------------------------------------------------------------------------
# Frozen dataclass sanity
# ---------------------------------------------------------------------------


def test_budget_config_is_frozen() -> None:
    cfg = BudgetConfig(
        n_ctx=10_000,
        reserved_headroom_tokens=1_000,
        state_cap_tokens=500,
        distilled_cap_tokens=500,
    )
    with pytest.raises(Exception):
        cfg.n_ctx = 20_000  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Round-2 correction §10.2 — reject integer water collapse (fixes C2)
# ---------------------------------------------------------------------------


def test_water_collapse_rejected() -> None:
    """When int() truncation collapses low_water and high_water to the same
    integer, BudgetConfig.__post_init__ MUST raise ValueError so that
    `low_water < high_water` holds for every constructible config — which is
    what makes a post-compaction load `<= low_water` strictly `< high_water`
    => needs_compaction False => no re-fire.

    budget = 110 - 100 = 10
    low_water  = int(0.10 * 10) = 1
    high_water = int(0.11 * 10) = 1
    => low == high => collapsed => ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        BudgetConfig(
            n_ctx=110,
            reserved_headroom_tokens=100,
            state_cap_tokens=1,
            distilled_cap_tokens=1,
            target_ratio=0.10,
            trigger_ratio=0.11,
        )
    # The message should surface the collapsed integer waters.
    msg = str(excinfo.value)
    assert "low_water" in msg
    assert "high_water" in msg


def test_water_separation_accepted_when_ints_differ() -> None:
    """Sanity counterpoint: a config whose integer waters are strictly ordered
    is accepted (the §10.2 check only rejects the collapsed case)."""
    cfg = BudgetConfig(
        n_ctx=110,
        reserved_headroom_tokens=100,
        state_cap_tokens=1,
        distilled_cap_tokens=1,
        target_ratio=0.10,
        trigger_ratio=0.20,  # high = int(0.20*10) = 2 > low = 1
    )
    assert cfg.low_water == 1
    assert cfg.high_water == 2
    assert cfg.low_water < cfg.high_water
