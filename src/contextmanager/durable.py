from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

from .hotness import HotnessTracker
from .note_store import NoteStore, StoreError
from .retriever import Retriever, make_retriever
from .state_store import StateStore
from .types import Message, TokenCounter


@dataclass
class RetrievedSlice:
    handle: str
    content: str
    score: float


class DurableStore:
    """Composes StateStore + NoteStore + Retriever. Implements the engine's Store Protocol
    (page_out, get) AND the page-in/retrieval API the proxy/MCP will use."""

    def __init__(self, root: str | Path, *, retriever: Optional[Retriever] = None,
                 hotness_half_life: float = 86_400.0, rerank_weight: float = 0.3,
                 rerank_pool: int = 50, ghost_maxlen: int = 512) -> None:
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        self.notes = NoteStore(root_path)
        self.state = StateStore(root_path / "state.json")
        self.retriever = retriever or make_retriever(str(root_path / "index.db"))
        # Decay-on-read working-set signal (Stage 3) — re-ranks recall toward the hot set
        # and drives generational eviction. Its own WAL db so both surfaces share it.
        self.hotness = HotnessTracker(str(root_path / "hotness.db"),
                                      half_life_seconds=hotness_half_life)
        # 0.0 = pure relevance; >0 blends hotness into the rank over a candidate pool.
        self._rerank_weight = max(0.0, float(rerank_weight))
        self._rerank_pool = max(1, int(rerank_pool))
        # Ghost ring: recently-evicted handle ids (no content). A later request for one
        # is a "cascade miss" — the eviction-feedback signal that we cut too deep.
        # In-memory + bounded (not persisted across restart — it is a tuning signal).
        self._ghosts: deque = deque(maxlen=max(1, int(ghost_maxlen)))
        # Retrieval-path counters (zero tokenizer cost) — the Stage-1 instrumentation
        # that tells us whether recall is actually exercised and how search cost grows
        # with the corpus. Thread-safe; surfaced via stats() for /metrics.
        self._stats_lock = Lock()
        self._search_calls = 0
        self._search_hits = 0
        self._results_returned = 0
        self._search_seconds = 0.0
        self._page_in_calls = 0
        self._page_in_slices = 0
        self._ghost_hits = 0
        self._resurrections = 0
        # Make the index reflect the canonical notes at open time (backward compat).
        self._reconcile_index()

    def _reconcile_index(self) -> None:
        """Reindex any note MISSING from the search index at open time. Covers an index
        schema/backend change (e.g. upgrading to the FTS5 backend over a store the
        pure-Python backend built) and an externally-deleted index.db — so the user can
        just run the governor and continue. Cheap when already consistent (a set diff, no
        file reads); the notes are the source of truth, so the index is always rebuildable.
        Also keeps gc() safe: every note is indexed, so none is mistaken for an
        eviction-orphan. Best-effort and non-fatal.
        """
        handles_fn = getattr(self.retriever, "handles", None)
        if not callable(handles_fn):
            return
        try:
            indexed = set(handles_fn())
        except Exception:
            return
        for h in self.notes.list_handles():
            if h in indexed:
                continue
            try:
                # notes.get (not self.get): a migration is not an access -> no hotness bump.
                self.retriever.index(h, self.notes.get(h))
            except Exception:
                continue

    # --- engine Store Protocol ---
    def page_out(self, message: Message) -> str:
        handle = self.notes.page_out(message)
        self.retriever.index(handle, message.content, {"id": message.id, "role": message.role})
        return handle

    def get(self, handle: str) -> str:
        try:
            content = self.notes.get(handle)
        except StoreError:
            resurrected = self._resurrect(handle)
            if resurrected is not None:
                return resurrected
            # A request for a truly vanished note: if it was recently evicted, that is
            # a cascade-miss (we cut too deep) — record the feedback signal, then
            # re-raise (callers like the proxy rewriter already treat a missing handle
            # gracefully). With the archive tier this only happens after a hard delete
            # (archive=False) or for a handle that never existed.
            if self.is_ghost(handle):
                with self._stats_lock:
                    self._ghost_hits += 1
            raise
        self.hotness.bump(handle)  # strong access signal (the rehydration path)
        return content

    def _resurrect(self, handle: str) -> Optional[str]:
        """Bring an archived (evicted) note back into the live set: restore the body,
        re-index it, and warm it — being requested again is the proof the eviction cut
        too deep. Returns the body, or None when the handle is not archived. A
        ghost-ring hit still counts as a cascade-miss so the eviction-feedback signal
        survives the lossless upgrade."""
        if not self.notes.restore(handle):
            return None
        content = self.notes.get(handle)
        try:
            meta = self.notes.read_meta(handle)
            self.retriever.index(handle, content, {"id": meta.id, "role": meta.role})
        except Exception:
            # Best-effort: the body is already back (lossless comes first);
            # _reconcile_index() repairs the index on the next open if this failed.
            pass
        self.hotness.bump(handle)
        with self._stats_lock:
            self._resurrections += 1
            if self.is_ghost(handle):
                self._ghost_hits += 1
        return content

    # --- page-IN / retrieval ---
    def page_in_by_id(self, message_id: str) -> Optional[str]:
        handle = NoteStore.handle_for(message_id)
        try:
            return self.get(handle)  # resurrects from the archive tier if evicted
        except StoreError:
            return None

    def search(self, query: str, k: int = 5) -> list[RetrievedSlice]:
        t0 = time.perf_counter()
        # Pull a larger candidate POOL from the index, then re-rank just that small pool
        # by a relevance×hotness blend and keep the top k. Re-ranking is O(pool), never
        # O(corpus) — the index's sublinear win stays intact.
        pool = max(k, self._rerank_pool)
        present = [(h, rel) for (h, rel) in self.retriever.search(query, pool)
                   if self.notes.has(h)]
        ranked = self._rerank(present)[:k]

        results: list[RetrievedSlice] = []
        for handle, _rel, blended in ranked:
            try:
                content = self.notes.get(handle)  # notes.get: no bump (search bumps below)
            except StoreError:
                continue
            results.append(RetrievedSlice(handle=handle, content=content, score=blended))

        elapsed = time.perf_counter() - t0
        with self._stats_lock:
            self._search_calls += 1
            self._search_seconds += elapsed
            if results:
                self._search_hits += 1
            self._results_returned += len(results)
        # The returned set IS the recall signal — warm it (decay-on-read).
        if results:
            self.hotness.bump_many([r.handle for r in results])
        return results

    def _rerank(self, items: list[tuple[str, float]]) -> list[tuple[str, float, float]]:
        """Blend relevance with decayed hotness over the candidate pool. Returns
        ``[(handle, relevance, blended)]`` sorted by blended desc, handle asc. The
        blend normalizes each signal to [0,1] over the pool, so it only RE-ORDERS within
        the candidates the index already deemed relevant — it never invents matches.
        Falls back to pure relevance when ``rerank_weight==0`` or no hotness exists yet.
        """
        if not items:
            return []
        w = self._rerank_weight
        max_rel = max((rel for _, rel in items), default=0.0)
        hot = self.hotness.scores([h for h, _ in items]) if w > 0.0 else {}
        max_hot = max(hot.values(), default=0.0) if hot else 0.0
        out: list[tuple[str, float, float]] = []
        for h, rel in items:
            rel_n = (rel / max_rel) if max_rel > 0 else 0.0
            hot_n = (hot.get(h, 0.0) / max_hot) if max_hot > 0 else 0.0
            blended = (1.0 - w) * rel_n + w * hot_n
            out.append((h, rel, blended))
        out.sort(key=lambda p: (-p[2], p[0]))
        return out

    def page_in(self, query: str, budget_tokens: int,
                counter: TokenCounter, k: int = 5) -> list[RetrievedSlice]:
        with self._stats_lock:
            self._page_in_calls += 1
        if budget_tokens <= 0:
            return []
        slices = self.search(query, k)
        out: list[RetrievedSlice] = []
        running = 0
        for sl in slices:
            remaining = budget_tokens - running
            if remaining <= 0:
                break
            count = counter.count_text(sl.content)
            if 0 < count <= remaining:
                # fits within the remaining budget -> include verbatim.
                out.append(sl)
                running += count
                continue
            # This slice would overflow (or is empty) -> truncate to the
            # remaining budget. Clamp against overshoot: only include the
            # truncated slice if its recount fits in `remaining` (spec §8.2).
            truncated = counter.truncate_to_tokens(sl.content, remaining)
            recount = counter.count_text(truncated)
            if 0 < recount <= remaining:
                out.append(RetrievedSlice(handle=sl.handle, content=truncated, score=sl.score))
                running += recount
            # else: overshoot or empty -> drop this slice.
            break
        with self._stats_lock:
            self._page_in_slices += len(out)
        return out

    # --- observability ---
    def corpus_size(self) -> int:
        """Number of notes currently persisted (authoritative content count)."""
        return len(self.notes.list_handles())

    def stats(self) -> dict:
        """JSON-able retrieval-path counters for /metrics — zero tokenizer cost.

        `search_calls` counts every search() (including those issued by page_in);
        `recall_hit_rate` = searches returning >=1 slice / search_calls; `avg_search_ms`
        is cumulative wall time / calls. These are exactly the signals that tell us
        whether the heavier FTS5/hot-scope work is warranted (Phase 7 Stage 1).
        """
        with self._stats_lock:
            calls = self._search_calls
            hits = self._search_hits
            results = self._results_returned
            seconds = self._search_seconds
            pin_calls = self._page_in_calls
            pin_slices = self._page_in_slices
            ghost_hits = self._ghost_hits
            resurrections = self._resurrections
        avg_ms = (seconds / calls * 1000.0) if calls else 0.0
        return {
            "backend": type(self.retriever).__name__,
            "corpus_size": self.corpus_size(),
            "search_calls": calls,
            "search_hits": hits,
            "search_empty": calls - hits,
            "recall_hit_rate": round(hits / calls, 4) if calls else 0.0,
            "results_returned": results,
            "avg_search_ms": round(avg_ms, 3),
            "page_in_calls": pin_calls,
            "page_in_slices": pin_slices,
            "ghost_hits": ghost_hits,
            "archived_count": len(self.notes.list_archived()),
            "resurrections": resurrections,
        }

    # --- maintenance (Phase 7 Stage 2) ---
    def gc(self, *, min_age_seconds: float = 0.0,
           protect: Optional[set] = None, archive: bool = True) -> dict:
        """Reconcile notes/ with the search index (the liveness root). Two SAFE sweeps:

          1. drop index entries whose note file is gone (broken references);
          2. take note files whose handle is NOT in the index (i.e. evicted from
             search) AND not in ``protect``, not referenced by persisted state, and
             older than ``min_age_seconds`` OUT of the live tier — by default
             (``archive=True``) demoting them to the archive tier (LOSSLESS, Phase 9:
             a later ``get()`` resurrects); ``archive=False`` hard-deletes instead.

        A note still in the index is never touched, so anything search-reachable — and
        anything a live stub could rehydrate while still indexed — stays safe. The
        eviction *decision* (what to drop from the index) belongs to the caller / Stage
        3; this is the tax-free mechanism that keeps the two sides consistent and
        reclaims the file-system side of an eviction. Idempotent and re-entrant.
        Returns ``{"dropped_index": n, "removed_notes": m}``.
        """
        handles_fn = getattr(self.retriever, "handles", None)
        if not callable(handles_fn):
            # An external retriever without handle enumeration -> cannot root the GC
            # safely; do nothing rather than guess.
            return {"dropped_index": 0, "removed_notes": 0}
        index_handles = set(handles_fn())
        note_handles = set(self.notes.list_handles())
        protect = set(protect or ())

        # Sweep 1: index entries pointing at vanished notes.
        dropped = 0
        for h in index_handles - note_handles:
            self.retriever.remove(h)
            self.hotness.remove(h)
            dropped += 1
        index_handles &= note_handles

        # Sweep 2: note files no longer in the index (evicted from search).
        removed = 0
        state_text = self.state.render()
        now = time.time()
        for h in note_handles - index_handles:
            # `h in state_text` is a deliberately CONSERVATIVE substring check: a false
            # positive only KEEPS a note (safe, just less reclaim) — it never deletes one.
            if h in protect or h in state_text:
                continue
            if min_age_seconds > 0.0:
                m = self.notes.mtime(h)
                if m is not None and (now - m) < min_age_seconds:
                    continue
            if archive:
                self.notes.archive(h)
            else:
                self.notes.remove(h)
            self.hotness.remove(h)
            removed += 1
        return {"dropped_index": dropped, "removed_notes": removed}

    def evict_cold(self, *, target_size: int, protect: Optional[set] = None,
                   now: Optional[float] = None, archive: bool = True) -> dict:
        """Bound the working set: while the corpus exceeds ``target_size``, evict the
        COLDEST handles (lowest decayed hotness; never-accessed notes score 0 and go
        first) — dropping them from the index and hotness, recording each as a ghost.

        By default (``archive=True``) eviction is LOSSLESS (Phase 9): the body is
        demoted to the archive tier and a later ``get()`` transparently resurrects it
        (restore + re-index + hotness bump). Pass ``archive=False`` to hard-delete the
        bodies when the disk itself must be reclaimed — only then is the content gone.
        Never evicts a ``protect``-ed or state-referenced handle.
        Returns ``{"evicted": n}``.
        """
        protect = set(protect or ())
        note_handles = self.notes.list_handles()
        n = len(note_handles)
        if n <= target_size:
            return {"evicted": 0}
        need = n - target_size
        state_text = self.state.render()
        scored = self.hotness.scores(note_handles, now=now)
        # coldest first; ties by handle ascending (deterministic).
        order = sorted(note_handles, key=lambda h: (scored.get(h, 0.0), h))

        evicted = 0
        for h in order:
            if evicted >= need:
                break
            if h in protect or h in state_text:
                continue
            self.retriever.remove(h)
            if archive:
                self.notes.archive(h)
            else:
                self.notes.remove(h)
            self.hotness.remove(h)
            self._record_ghost(h)
            evicted += 1
        return {"evicted": evicted}

    # --- eviction-feedback (ghost ring) ---
    def _record_ghost(self, handle: str) -> None:
        if handle not in self._ghosts:        # deque is bounded -> oldest auto-drops
            self._ghosts.append(handle)

    def is_ghost(self, handle: str) -> bool:
        """True iff `handle` was recently evicted (a request for it = a cascade-miss)."""
        return handle in self._ghosts

    # --- cold-tier compression ---
    def compress_cold(self, *, keep_hot: int, now: Optional[float] = None) -> dict:
        """Gzip-at-rest all but the ``keep_hot`` hottest notes. LOSSLESS — compressed
        notes stay searchable and GET transparently decompresses — so this is the SAFE
        disk-saver (contrast ``evict_cold``, which deletes). Returns {"compressed": n}.
        """
        handles = self.notes.list_handles()
        excess = len(handles) - max(0, int(keep_hot))
        if excess <= 0:
            return {"compressed": 0}
        scored = self.hotness.scores(handles, now=now)
        order = sorted(handles, key=lambda h: (scored.get(h, 0.0), h))  # coldest first
        n = 0
        for h in order[:excess]:
            if self.notes.compress(h):
                n += 1
        return {"compressed": n}

    # --- lifecycle ---
    def close(self) -> None:
        """Release the retriever's resources (e.g. sqlite connection), if any.

        Safe to call once; idempotent (spec §8.4).
        """
        close = getattr(self.retriever, "close", None)
        if callable(close):
            close()
        self.hotness.close()
