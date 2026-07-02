"""HotnessTracker — per-handle decay-on-read access score (Phase 7 Stage 3).

A cheap, persistent working-set signal. On each access the stored score is first
DECAYED to the access time (exponential half-life), then incremented by 1:

    score(t) = score(last) * 0.5 ** ((t - last) / half_life) + 1

Decay is computed lazily from the timestamp delta — there is NO background sweep, so
updating is O(1) per access and ~a dozen bytes per live handle. This is the Denning
"working set" made concrete: recency-weighted frequency, which (by temporal locality)
is the workhorse predictor of the near-future working set. Used to (a) re-rank search
results toward the hot set and (b) pick the COLDEST handles for generational eviction.

State lives in its own sqlite db (WAL) so the proxy and MCP surfaces share one signal.
The clock is injectable (`now=`) so the decay math is deterministically testable.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Iterable, Optional


class HotnessTracker:
    def __init__(self, path: str | Any = ":memory:", *,
                 half_life_seconds: float = 86_400.0) -> None:
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be > 0")
        self._path = str(path)
        self._half_life = float(half_life_seconds)
        # check_same_thread=False: the proxy mutates hotness from a worker thread (the
        # off-loaded rewrite); the proxy's single rewrite-lock serializes access so this
        # connection is never used concurrently.
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
            self._path, check_same_thread=False
        )
        if self._path != ":memory:":
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute("PRAGMA busy_timeout=5000")
            except sqlite3.Error:
                pass
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS hotness "
            "(handle TEXT PRIMARY KEY, score REAL NOT NULL, last_seen REAL NOT NULL)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------ internals
    def _decay(self, score: float, last: float, now: float) -> float:
        dt = now - last
        if dt <= 0:
            return score
        return score * (0.5 ** (dt / self._half_life))

    @staticmethod
    def _now(now: Optional[float]) -> float:
        return time.time() if now is None else now

    # ------------------------------------------------------------------ mutation
    def bump(self, handle: str, *, now: Optional[float] = None) -> float:
        """Record an access: decay the existing score to `now`, then +1. Returns it."""
        t = self._now(now)
        row = self._conn.execute(
            "SELECT score, last_seen FROM hotness WHERE handle=?", (handle,)
        ).fetchone()
        new = (self._decay(float(row[0]), float(row[1]), t) + 1.0) if row else 1.0
        self._conn.execute(
            "INSERT INTO hotness(handle, score, last_seen) VALUES(?,?,?) "
            "ON CONFLICT(handle) DO UPDATE SET score=excluded.score, "
            "last_seen=excluded.last_seen",
            (handle, new, t),
        )
        self._conn.commit()
        return new

    def bump_many(self, handles: Iterable[str], *, now: Optional[float] = None) -> None:
        t = self._now(now)
        for h in handles:
            self.bump(h, now=t)

    def remove(self, handle: str) -> None:
        self._conn.execute("DELETE FROM hotness WHERE handle=?", (handle,))
        self._conn.commit()

    # ------------------------------------------------------------------ queries
    def score(self, handle: str, *, now: Optional[float] = None) -> float:
        """Current decayed score (read-only; does NOT bump). 0.0 if unseen."""
        t = self._now(now)
        row = self._conn.execute(
            "SELECT score, last_seen FROM hotness WHERE handle=?", (handle,)
        ).fetchone()
        return self._decay(float(row[0]), float(row[1]), t) if row else 0.0

    def scores(self, handles: Iterable[str], *,
               now: Optional[float] = None) -> dict[str, float]:
        """Batch decayed scores for `handles` (one query) — for search re-ranking.
        Unseen handles map to 0.0."""
        t = self._now(now)
        want = list(dict.fromkeys(handles))
        out: dict[str, float] = {h: 0.0 for h in want}
        if not want:
            return out
        qmarks = ",".join("?" * len(want))
        rows = self._conn.execute(
            f"SELECT handle, score, last_seen FROM hotness WHERE handle IN ({qmarks})",
            want,
        ).fetchall()
        for h, s, last in rows:
            out[h] = self._decay(float(s), float(last), t)
        return out

    def coldest(self, *, limit: int, now: Optional[float] = None) -> list[str]:
        """The `limit` handles with the lowest current (decayed) score, coldest first;
        ties broken by handle ascending for determinism."""
        if limit <= 0:
            return []
        t = self._now(now)
        rows = self._conn.execute(
            "SELECT handle, score, last_seen FROM hotness"
        ).fetchall()
        decayed = sorted(
            ((self._decay(float(s), float(last), t), h) for h, s, last in rows),
            key=lambda p: (p[0], p[1]),
        )
        return [h for _, h in decayed[:limit]]

    def handles(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT handle FROM hotness").fetchall()]

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        conn = self._conn
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> "HotnessTracker":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
