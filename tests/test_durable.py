"""Phase 2 DurableStore facade tests — binds to `tasks/phase2-spec.md` §6
(test_durable).

Covers:
  - DurableStore(tmp_path): page_out(Message) then get(handle) round-trips; AND
    search(query matching the content) finds that handle (page_out indexes it).
  - page_in_by_id returns exact content for a paged-out message id; None for an
    unknown id.
  - search returns list[RetrievedSlice] in score order; a handle whose note was
    removed is skipped (manually remove the note via .notes.remove then search
    must not raise).
  - page_in respects budget EXACTLY using the Phase 1 FakeCounter (word-count
    tokens): total measured tokens across returned slices never exceeds
    budget_tokens; the overflowing slice is truncated; budget_tokens=0 -> [].
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextmanager.durable import DurableStore, RetrievedSlice
from contextmanager.note_store import StoreError
from contextmanager.retriever import Fts5Retriever, LexicalRetriever, fts5_available
from contextmanager.types import Message

from conftest import FakeCounter, msg_of_cost


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_message(mid: str, content: str, role: str = "user") -> Message:
    return Message(role=role, content=content, id=mid)


# ---------------------------------------------------------------------------
# page_out + get + search round-trip
# ---------------------------------------------------------------------------


def test_page_out_then_get_round_trips(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    msg = _make_message("m1", "hello durable world")
    handle = ds.page_out(msg)
    assert isinstance(handle, str)
    assert ds.get(handle) == "hello durable world"


def test_page_out_indexes_content_so_search_finds_it(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    ds.page_out(_make_message("m1", "quantum field theory hamiltonian"))
    ds.page_out(_make_message("m2", "baking sourdough bread flour recipe"))

    res = ds.search("quantum hamiltonian field", k=5)
    handles = [s.handle for s in res]
    assert handles, "expected search to find indexed content"
    # The paged-out message id "m1" has a deterministic handle; search must
    # surface that handle (not a raw "m1" string, which is never a handle).
    assert ds.notes.handle_for("m1") in handles

    # the first hit must be the quantum doc, not the baking doc
    assert res[0].handle == ds.notes.handle_for("m1")


def test_search_returns_retrieved_slices_in_score_order(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    ds.page_out(_make_message("alpha", "cat dog animal pet mammal"))
    ds.page_out(_make_message("beta", "rocket orbit satellite aerospace"))

    res = ds.search("rocket orbit aerospace", k=5)
    assert all(isinstance(s, RetrievedSlice) for s in res)
    # scores non-increasing
    scores = [s.score for s in res]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# page_in_by_id
# ---------------------------------------------------------------------------


def test_page_in_by_id_returns_exact_content_for_paged_out_message(
    tmp_path: Path,
) -> None:
    ds = DurableStore(tmp_path)
    body = "line one\nline two\nline three"
    ds.page_out(_make_message("mid-42", body))
    assert ds.page_in_by_id("mid-42") == body


def test_page_in_by_id_returns_none_for_unknown_id(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    assert ds.page_in_by_id("never-paged-out") is None


# ---------------------------------------------------------------------------
# search skips missing notes (no raise)
# ---------------------------------------------------------------------------


def test_search_skips_handle_whose_note_was_removed(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    h1 = ds.page_out(_make_message("keep-1", "alpha shared token"))
    h2 = ds.page_out(_make_message("drop-2", "alpha shared token"))

    # both currently findable
    res_before = ds.search("alpha shared", k=5)
    assert {s.handle for s in res_before} >= {h1, h2}

    # manually remove one note via the underlying note store
    ds.notes.remove(h2)

    # search must not raise, and the removed handle must be skipped
    res_after = ds.search("alpha shared", k=5)
    handles_after = [s.handle for s in res_after]
    assert h2 not in handles_after
    assert h1 in handles_after


# ---------------------------------------------------------------------------
# page_in budget: FakeCounter (word-count tokens)
# ---------------------------------------------------------------------------


def _index_budget_corpus(ds: DurableStore) -> dict[str, str]:
    """Index 3 docs, each a distinct query term + filler to reach known word counts.

    Each doc content has EXACTLY 10 whitespace-split words so FakeCounter
    (word-count) measures it as 10 tokens. Each doc contains a unique query term
    ("termA"/"termB"/"termC") plus an irrelevant 4th doc to keep df low enough
    that BM25 idf is unambiguously positive (df=1, N=4 -> idf>0 under both the
    smoothed and unsmoothed Okapi formulas).
    """
    def make(mid: str, term: str, n_words: int = 10) -> str:
        # content = term + (n_words-1) filler words -> exactly n_words tokens
        filler = " ".join(f"{mid}f{i}" for i in range(n_words - 1))
        return f"{term} {filler}"

    ds.page_out(_make_message("d1", make("d1", "termA")))   # 10 words, has termA
    ds.page_out(_make_message("d2", make("d2", "termB")))   # 10 words, has termB
    ds.page_out(_make_message("d3", make("d3", "termC")))   # 10 words, has termC
    ds.page_out(_make_message("d4", make("d4", "filler")))  # 10 words, irrelevant
    return {
        ds.notes.handle_for("d1"): "termA",
        ds.notes.handle_for("d2"): "termB",
        ds.notes.handle_for("d3"): "termC",
    }


def test_page_in_total_tokens_never_exceeds_budget(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    _index_budget_corpus(ds)
    counter = FakeCounter()

    for budget in [0, 1, 5, 7, 10, 15, 25, 30]:
        slices = ds.page_in("termA termB termC", budget_tokens=budget, counter=counter)
        total = sum(counter.count_text(s.content) for s in slices)
        assert total <= budget, f"budget={budget}: total={total} exceeded budget"


def test_page_in_budget_zero_returns_empty(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    _index_budget_corpus(ds)
    assert ds.page_in("termA termB termC", budget_tokens=0, counter=FakeCounter()) == []


def test_page_in_overflowing_slice_is_truncated(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    _index_budget_corpus(ds)
    counter = FakeCounter()

    # Each doc = 10 words. budget=15 -> first slice (10) fits, second would
    # overflow by 5 -> truncated to the 5-word remaining budget and included,
    # then stop. Total == 15 exactly.
    slices = ds.page_in("termA termB termC", budget_tokens=15, counter=counter)
    total = sum(counter.count_text(s.content) for s in slices)
    assert total == 15

    # there should be exactly two slices (first full + second truncated)
    assert len(slices) == 2

    # the first slice is a full 10-word doc
    assert counter.count_text(slices[0].content) == 10
    # the second slice is the truncated one: 5 words
    assert counter.count_text(slices[1].content) == 5
    # and it is a prefix (first-N-words) of its source note
    source_handle = slices[1].handle
    full = ds.get(source_handle)
    assert slices[1].content == " ".join(full.split()[:5])


def test_page_in_first_slice_truncated_when_smaller_than_one_doc(
    tmp_path: Path,
) -> None:
    ds = DurableStore(tmp_path)
    _index_budget_corpus(ds)
    counter = FakeCounter()

    # budget=5 < 10 (one doc): the first slice overflows immediately; it is
    # truncated to 5 words and included (remaining 5 > 0), then stop. Total == 5.
    slices = ds.page_in("termA termB termC", budget_tokens=5, counter=counter)
    total = sum(counter.count_text(s.content) for s in slices)
    assert total == 5
    assert len(slices) == 1
    assert counter.count_text(slices[0].content) == 5


def test_page_in_exactly_one_doc_budget(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    _index_budget_corpus(ds)
    counter = FakeCounter()

    # budget == 10 -> exactly one full doc fits; second would overflow with
    # remaining 0 -> not included.
    slices = ds.page_in("termA termB termC", budget_tokens=10, counter=counter)
    total = sum(counter.count_text(s.content) for s in slices)
    assert total == 10
    assert len(slices) == 1
    assert counter.count_text(slices[0].content) == 10


def test_page_in_large_budget_includes_all_untruncated(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    _index_budget_corpus(ds)
    counter = FakeCounter()

    # budget >= total of all matching docs (3 docs x 10 = 30): all included full.
    slices = ds.page_in("termA termB termC", budget_tokens=100, counter=counter)
    total = sum(counter.count_text(s.content) for s in slices)
    assert total == 30
    assert len(slices) == 3
    for s in slices:
        assert counter.count_text(s.content) == 10


# ---------------------------------------------------------------------------
# Phase 7 Stage 1 — retrieval-path stats() + corpus_size()
# ---------------------------------------------------------------------------


def test_stats_initial_zero(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    s = ds.stats()
    assert s["corpus_size"] == 0
    assert s["search_calls"] == 0
    assert s["search_hits"] == 0
    assert s["recall_hit_rate"] == 0.0
    assert s["page_in_calls"] == 0
    assert s["avg_search_ms"] == 0.0


def test_corpus_size_counts_paged_out_notes(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    ds.page_out(_make_message("a", "first note body"))
    ds.page_out(_make_message("b", "second note body"))
    assert ds.corpus_size() == 2
    assert ds.stats()["corpus_size"] == 2


def test_stats_tracks_search_hits_and_misses(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    ds.page_out(_make_message("a", "quantum hamiltonian eigenstate operator"))
    ds.search("quantum hamiltonian")        # hit
    ds.search("zzzqqq nonexistent vocab")   # miss -> empty
    s = ds.stats()
    assert s["search_calls"] == 2
    assert s["search_hits"] == 1
    assert s["search_empty"] == 1
    assert s["recall_hit_rate"] == 0.5
    assert s["results_returned"] >= 1
    assert s["avg_search_ms"] >= 0.0


def test_stats_page_in_counts_include_internal_search(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    ds.page_out(_make_message("a", "alpha beta gamma delta epsilon zeta eta theta"))
    out = ds.page_in("alpha beta", budget_tokens=100, counter=FakeCounter())
    s = ds.stats()
    # page_in() issues exactly one internal search().
    assert s["page_in_calls"] == 1
    assert s["search_calls"] == 1
    assert s["page_in_slices"] == len(out)


def test_stats_page_in_zero_budget_counts_call_but_no_search(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    ds.page_out(_make_message("a", "alpha beta gamma"))
    assert ds.page_in("alpha", budget_tokens=0, counter=FakeCounter()) == []
    s = ds.stats()
    assert s["page_in_calls"] == 1
    assert s["search_calls"] == 0  # early-returns before searching


# ---------------------------------------------------------------------------
# Phase 7 Stage 2 — index-rooted mark-and-sweep GC
# ---------------------------------------------------------------------------


def test_gc_noop_on_consistent_store(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    ds.page_out(_make_message("a", "alpha beta gamma"))
    ds.page_out(_make_message("b", "delta epsilon zeta"))
    assert ds.gc() == {"dropped_index": 0, "removed_notes": 0}
    assert ds.corpus_size() == 2


def test_gc_drops_index_entry_for_missing_note(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    h = ds.page_out(_make_message("a", "alpha beta gamma"))
    # Note file vanishes but the index still references it (broken reference).
    ds.notes.remove(h)
    assert h in ds.retriever.handles()
    res = ds.gc()
    assert res["dropped_index"] == 1
    assert h not in ds.retriever.handles()


def test_gc_removes_note_evicted_from_index(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    h = ds.page_out(_make_message("a", "alpha beta gamma"))
    # Simulate an eviction decision: drop from the search index, keep the file.
    ds.retriever.remove(h)
    assert ds.notes.has(h)
    res = ds.gc()
    assert res["removed_notes"] == 1
    assert not ds.notes.has(h)


def test_gc_protect_keeps_note(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    h = ds.page_out(_make_message("a", "alpha beta gamma"))
    ds.retriever.remove(h)
    res = ds.gc(protect={h})
    assert res["removed_notes"] == 0
    assert ds.notes.has(h)


def test_gc_keeps_note_referenced_by_state(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    h = ds.page_out(_make_message("pinnednote", "alpha beta gamma"))
    ds.retriever.remove(h)
    ds.state.save({"ref": h})  # state references the handle -> must not be swept
    res = ds.gc()
    assert res["removed_notes"] == 0
    assert ds.notes.has(h)


def test_gc_min_age_guards_recent_note(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    h = ds.page_out(_make_message("a", "alpha beta gamma"))
    ds.retriever.remove(h)
    # A large min-age keeps the just-written note (guards write/index races).
    res = ds.gc(min_age_seconds=3600)
    assert res["removed_notes"] == 0
    assert ds.notes.has(h)


def test_gc_never_touches_indexed_note(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    h = ds.page_out(_make_message("a", "alpha beta gamma"))
    # Still in the index -> live -> GC must leave it completely alone.
    res = ds.gc()
    assert res == {"dropped_index": 0, "removed_notes": 0}
    assert ds.notes.has(h)
    assert h in ds.retriever.handles()


def test_gc_idempotent(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    h = ds.page_out(_make_message("a", "alpha beta gamma"))
    ds.retriever.remove(h)
    first = ds.gc()
    second = ds.gc()
    assert first["removed_notes"] == 1
    assert second == {"dropped_index": 0, "removed_notes": 0}


# ---------------------------------------------------------------------------
# Phase 7 Stage 3 — hotness re-rank, generational eviction, ghost feedback
# ---------------------------------------------------------------------------


def test_search_rerank_promotes_hot_doc(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path, rerank_weight=0.5)
    ds.page_out(_make_message("d1", "alpha shared keyword one"))
    ds.page_out(_make_message("d2", "alpha shared keyword two"))
    h2 = ds.notes.handle_for("d2")
    # Warm d2 heavily (get() bumps hotness); d1 stays cold.
    for _ in range(10):
        ds.get(h2)
    res = ds.search("alpha shared keyword", k=2)
    assert res[0].handle == h2  # hotness lifted d2 above the equal-relevance d1


def test_search_rerank_weight_zero_is_pure_relevance(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path, rerank_weight=0.0)
    ds.page_out(_make_message("d1", "quantum hamiltonian eigenstate"))
    ds.page_out(_make_message("d2", "baking sourdough flour recipe"))
    # Even if d2 is hot, weight 0 -> relevance decides; query matches d1.
    for _ in range(5):
        ds.get(ds.notes.handle_for("d2"))
    res = ds.search("quantum hamiltonian", k=2)
    assert res[0].handle == ds.notes.handle_for("d1")


def test_evict_cold_removes_coldest(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    for i in range(5):
        ds.page_out(_make_message(f"d{i}", f"document number {i} content"))
    ds.get(ds.notes.handle_for("d4"))  # warm d4 & d3
    ds.get(ds.notes.handle_for("d3"))
    res = ds.evict_cold(target_size=3)
    assert res["evicted"] == 2
    assert ds.corpus_size() == 3
    assert ds.notes.has(ds.notes.handle_for("d4"))
    assert ds.notes.has(ds.notes.handle_for("d3"))
    # The two coldest (all score 0 -> handle-asc) were d0, d1, recorded as ghosts.
    assert ds.is_ghost(ds.notes.handle_for("d0"))
    assert ds.is_ghost(ds.notes.handle_for("d1"))


def test_evict_cold_respects_protect_and_state(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    for i in range(4):
        ds.page_out(_make_message(f"d{i}", f"document {i}"))
    h0 = ds.notes.handle_for("d0")
    h1 = ds.notes.handle_for("d1")
    ds.state.save({"ref": h0})  # state-pinned
    # need=2; coldest order d0,d1,d2,d3 (all 0, handle-asc); skip d0 (state) & d1 (protect)
    res = ds.evict_cold(target_size=2, protect={h1})
    assert res["evicted"] == 2
    assert ds.notes.has(h0) and ds.notes.has(h1)
    assert not ds.notes.has(ds.notes.handle_for("d2"))
    assert not ds.notes.has(ds.notes.handle_for("d3"))


def test_evict_cold_noop_when_under_target(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    ds.page_out(_make_message("a", "alpha"))
    assert ds.evict_cold(target_size=5) == {"evicted": 0}
    assert ds.corpus_size() == 1


def test_ghost_hit_recorded_on_get_of_evicted(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    for i in range(3):
        ds.page_out(_make_message(f"d{i}", f"document {i}"))
    # archive=False = the explicit HARD-DELETE path (the default archives, and a
    # get() would resurrect instead of raising — covered by the Phase 9 tests).
    ds.evict_cold(target_size=1, archive=False)  # evicts 2 coldest -> ghosts
    ghost = next(
        ds.notes.handle_for(f"d{i}") for i in range(3)
        if ds.is_ghost(ds.notes.handle_for(f"d{i}"))
    )
    with pytest.raises(StoreError):
        ds.get(ghost)
    assert ds.stats()["ghost_hits"] == 1


def test_compress_cold_keeps_searchable_and_byte_exact(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    body = "alpha unicorn keyword\r\nwindows crlf\rbare cr\nlf trailing\n"
    ds.page_out(_make_message("d0", body))
    ds.page_out(_make_message("d1", "beta other unrelated content here"))
    ds.get(ds.notes.handle_for("d1"))  # warm d1 -> d0 is the cold one
    res = ds.compress_cold(keep_hot=1)
    assert res["compressed"] == 1

    h0 = ds.notes.handle_for("d0")
    assert ds.notes.is_compressed(h0)
    assert ds.get(h0) == body                       # byte-exact through gzip
    hits = [s.handle for s in ds.search("unicorn keyword", k=5)]
    assert h0 in hits                               # still findable (index untouched)


# ---------------------------------------------------------------------------
# Phase 9 — lossless saving: archive tier + resurrection
# ---------------------------------------------------------------------------


def test_evict_cold_archives_by_default(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    for i in range(3):
        ds.page_out(_make_message(f"d{i}", f"document {i}"))
    res = ds.evict_cold(target_size=1)
    assert res["evicted"] == 2
    assert ds.corpus_size() == 1  # the working-set bound is exactly as tight
    evicted = [h for h in (ds.notes.handle_for(f"d{i}") for i in range(3))
               if not ds.notes.has(h)]
    assert len(evicted) == 2
    for h in evicted:
        assert ds.notes.is_archived(h)  # demoted, not destroyed
        assert ds.is_ghost(h)           # eviction-feedback signal unchanged
    assert ds.stats()["archived_count"] == 2


def test_get_resurrects_archived_note(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    body = "alpha unicorn keyword\r\nwindows crlf\rbare cr\nlf trailing\n"
    ds.page_out(_make_message("d0", body))
    ds.page_out(_make_message("d1", "beta other unrelated content"))
    ds.get(ds.notes.handle_for("d1"))  # warm d1 -> d0 is the coldest
    ds.evict_cold(target_size=1)
    h0 = ds.notes.handle_for("d0")
    assert not ds.notes.has(h0)

    assert ds.get(h0) == body            # byte-exact resurrection
    assert ds.notes.has(h0)              # back in the live set
    assert not ds.notes.is_archived(h0)
    hits = [s.handle for s in ds.search("unicorn keyword", k=5)]
    assert h0 in hits                    # re-indexed -> searchable again
    st = ds.stats()
    assert st["resurrections"] == 1
    assert st["ghost_hits"] == 1         # a resurrection is still a cascade-miss
    assert st["archived_count"] == 0


def test_evict_cold_archive_false_hard_deletes(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    for i in range(2):
        ds.page_out(_make_message(f"d{i}", f"document {i}"))
    ds.evict_cold(target_size=1, archive=False)
    gone = next(h for h in (ds.notes.handle_for(f"d{i}") for i in range(2))
                if not ds.notes.has(h))
    assert not ds.notes.is_archived(gone)
    with pytest.raises(StoreError):
        ds.get(gone)


def test_resurrection_survives_reopen(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    ds.page_out(_make_message("d0", "resurrect me later"))
    ds.page_out(_make_message("d1", "the survivor"))
    ds.get(ds.notes.handle_for("d1"))  # warm d1 -> d0 is the coldest
    ds.evict_cold(target_size=1)
    ds.close()

    ds2 = DurableStore(tmp_path)  # fresh process: the ghost ring is NOT persisted
    h0 = ds2.notes.handle_for("d0")
    assert ds2.corpus_size() == 1                  # reconcile did NOT resurrect
    assert ds2.get(h0) == "resurrect me later"     # ...but a real request does
    st = ds2.stats()
    assert st["resurrections"] == 1
    assert st["ghost_hits"] == 0                   # not a recent ghost in this process
    ds2.close()


def test_page_in_by_id_resurrects(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    ds.page_out(_make_message("d0", "paged content"))
    ds.page_out(_make_message("d1", "warm content"))
    ds.get(ds.notes.handle_for("d1"))  # warm d1 -> d0 is the coldest
    ds.evict_cold(target_size=1)
    assert ds.page_in_by_id("d0") == "paged content"
    assert ds.notes.has(ds.notes.handle_for("d0"))
    assert ds.page_in_by_id("never-existed") is None


def test_gc_archives_sweep2_by_default(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    ds.page_out(_make_message("d0", "evicted from index"))
    h = ds.notes.handle_for("d0")
    ds.retriever.remove(h)  # un-index -> a sweep-2 candidate
    res = ds.gc()
    assert res["removed_notes"] == 1
    assert not ds.notes.has(h)
    assert ds.notes.is_archived(h)              # demoted, not destroyed
    assert ds.get(h) == "evicted from index"    # and still resurrectable


def test_gc_archive_false_hard_deletes(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    ds.page_out(_make_message("d0", "gone for good"))
    h = ds.notes.handle_for("d0")
    ds.retriever.remove(h)
    res = ds.gc(archive=False)
    assert res["removed_notes"] == 1
    assert not ds.notes.is_archived(h)
    with pytest.raises(StoreError):
        ds.get(h)


def test_archive_of_compressed_note_and_restore(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    body = "compressed then archived\r\ncrlf kept\n"
    ds.page_out(_make_message("d0", body))
    ds.page_out(_make_message("d1", "stays hot"))
    ds.get(ds.notes.handle_for("d1"))
    h0 = ds.notes.handle_for("d0")
    ds.compress_cold(keep_hot=1)               # d0 -> .md.gz (still live)
    assert ds.notes.is_compressed(h0)
    ds.evict_cold(target_size=1)               # gz note -> pure-rename archive path
    assert ds.notes.is_archived(h0)
    assert ds.get(h0) == body                  # byte-exact through gz -> archive -> back


# ---------------------------------------------------------------------------
# Phase 7 — backward compatibility: reindex-from-notes on open + backend visibility
# ---------------------------------------------------------------------------


def test_stats_reports_backend(tmp_path: Path) -> None:
    ds = DurableStore(tmp_path)
    assert ds.stats()["backend"] in ("Fts5Retriever", "LexicalRetriever")


def test_store_usable_from_worker_thread(tmp_path: Path) -> None:
    # Enables the proxy's off-event-loop rewrite (asyncio.to_thread): the store's sqlite
    # connections are opened check_same_thread=False, so page_out/search/get must work from
    # a thread other than the one that constructed the store. Without that flag, sqlite
    # raises "objects created in a thread can only be used in that same thread".
    import threading

    ds = DurableStore(tmp_path)
    errors: list = []

    def work() -> None:
        try:
            ds.page_out(_make_message("a", "alpha beta gamma worker thread content"))
            h = ds.notes.handle_for("a")
            assert ds.search("alpha beta", k=1)[0].handle == h
            ds.get(h)  # bumps hotness -> a cross-thread sqlite WRITE
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert not errors, errors
    assert ds.corpus_size() == 1


def test_open_reindexes_notes_missing_from_index(tmp_path: Path) -> None:
    root = tmp_path / "s"
    ds = DurableStore(root)
    ds.page_out(_make_message("a", "alpha beta gamma delta"))
    h = ds.notes.handle_for("a")
    ds.retriever.remove(h)        # simulate a lost/stale index entry, note kept
    ds.close()

    ds2 = DurableStore(root)      # reopen -> _reconcile_index re-indexes from notes/
    try:
        assert h in ds2.retriever.handles()
        assert ds2.search("alpha beta", k=1)[0].handle == h
    finally:
        ds2.close()


@pytest.mark.skipif(not fts5_available(), reason="FTS5 not available")
def test_open_reindexes_legacy_lexical_store_into_fts5(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir(parents=True)  # the legacy retriever opens index.db before DurableStore mkdir
    # Build a store the OLD way: pure-Python lexical index (a regular `docs` table).
    old = DurableStore(root, retriever=LexicalRetriever(str(root / "index.db")))
    old.page_out(_make_message("a", "quantum hamiltonian eigenstate operator"))
    old.page_out(_make_message("b", "baking sourdough bread flour recipe"))
    old.close()

    # Reopen with the DEFAULT backend (FTS5) over the SAME index.db (legacy table present):
    # no collision (FTS5 uses fts_idx), and reconcile reindexes the old notes.
    new = DurableStore(root)
    try:
        assert isinstance(new.retriever, Fts5Retriever)
        res = new.search("quantum hamiltonian", k=3)
        assert res and res[0].handle == new.notes.handle_for("a")  # old content searchable
    finally:
        new.close()
