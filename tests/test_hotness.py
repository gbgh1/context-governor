"""Phase 7 Stage 3 — HotnessTracker (decay-on-read working-set score).

The clock is injected via ``now=`` so the exponential-decay math is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextmanager.hotness import HotnessTracker


def test_first_bump_is_one() -> None:
    h = HotnessTracker(":memory:")
    assert h.bump("a", now=0.0) == 1.0
    assert h.score("a", now=0.0) == 1.0


def test_repeated_bumps_accumulate_at_same_instant() -> None:
    h = HotnessTracker(":memory:")
    h.bump("a", now=0.0)
    assert h.bump("a", now=0.0) == 2.0  # no decay at the same instant
    assert h.bump("a", now=0.0) == 3.0


def test_decays_by_half_over_one_half_life() -> None:
    h = HotnessTracker(":memory:", half_life_seconds=100.0)
    h.bump("a", now=0.0)
    assert h.score("a", now=100.0) == pytest.approx(0.5)   # one half-life
    assert h.score("a", now=200.0) == pytest.approx(0.25)  # two half-lives


def test_bump_decays_then_increments() -> None:
    h = HotnessTracker(":memory:", half_life_seconds=100.0)
    h.bump("a", now=0.0)                  # 1.0
    assert h.bump("a", now=100.0) == pytest.approx(1.5)  # decay to 0.5, then +1


def test_unseen_handle_scores_zero() -> None:
    h = HotnessTracker(":memory:")
    assert h.score("never", now=0.0) == 0.0
    assert h.scores(["never", "x"], now=0.0) == {"never": 0.0, "x": 0.0}


def test_scores_batch_matches_individual() -> None:
    h = HotnessTracker(":memory:", half_life_seconds=100.0)
    h.bump("a", now=0.0)
    h.bump("b", now=0.0)
    h.bump("b", now=0.0)
    s = h.scores(["a", "b", "c"], now=50.0)
    assert s["a"] == pytest.approx(h.score("a", now=50.0))
    assert s["b"] == pytest.approx(h.score("b", now=50.0))
    assert s["c"] == 0.0


def test_coldest_orders_by_decayed_score() -> None:
    h = HotnessTracker(":memory:", half_life_seconds=100.0)
    for _ in range(3):
        h.bump("hot", now=0.0)   # 3.0
    h.bump("warm", now=0.0)      # 1.0
    h.bump("cold", now=0.0)      # 1.0 -> ties warm, handle-asc
    assert h.coldest(limit=2, now=0.0) == ["cold", "warm"]  # hot (3.0) excluded


def test_remove_drops_handle() -> None:
    h = HotnessTracker(":memory:")
    h.bump("a", now=0.0)
    h.remove("a")
    assert h.score("a", now=0.0) == 0.0
    assert "a" not in h.handles()


def test_invalid_half_life_rejected() -> None:
    with pytest.raises(ValueError):
        HotnessTracker(":memory:", half_life_seconds=0)


def test_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "hot.db"
    h1 = HotnessTracker(db, half_life_seconds=100.0)
    h1.bump("a", now=0.0)
    h1.bump("a", now=0.0)  # 2.0 at t=0
    h1.close()
    h2 = HotnessTracker(db, half_life_seconds=100.0)
    try:
        assert h2.score("a", now=0.0) == pytest.approx(2.0)
    finally:
        h2.close()
