"""Phase 7 Stage 2 — Fts5Retriever (contentless FTS5 inverted index) + factory.

Asserts the FTS5 backend honors the same ``Retriever`` contract as the pure-Python
``LexicalRetriever`` (ranking, remove, upsert, persistence, handles, close), that it
is genuinely CONTENTLESS (no second copy of the text), and that ``make_retriever`` /
``fts5_available`` select the fast path. Skipped wholesale if this sqlite build lacks
FTS5 (then the portable fallback is exercised by the LexicalRetriever suite instead).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from contextmanager.retriever import (
    Fts5Retriever,
    LexicalRetriever,
    Retriever,
    fts5_available,
    make_retriever,
)

pytestmark = pytest.mark.skipif(
    not fts5_available(), reason="FTS5/contentless_delete not compiled into this sqlite"
)


def _index_corpus(r) -> None:
    docs = {
        "cooking": "recipe bake flour sugar eggs butter oven whisk knead dough",
        "python": "python function module import class decorator async await loop",
        "gardening": "garden soil compost seedling mulch prune water sunlight roots",
    }
    for h, text in docs.items():
        r.index(h, text)


# --------------------------------------------------------------------------- ranking


def test_fts5_query_matching_one_doc_returns_that_handle_first() -> None:
    r = Fts5Retriever(":memory:")
    _index_corpus(r)
    res = r.search("python async await", k=3)
    assert res and res[0][0] == "python"
    scores = [s for _, s in res]
    assert all(s > 0 for s in scores)              # positive (bm25 negated)
    assert scores == sorted(scores, reverse=True)  # descending, best first


def test_fts5_unrelated_query_empty() -> None:
    r = Fts5Retriever(":memory:")
    _index_corpus(r)
    assert r.search("zzzqqq xxwwyy", k=3) == []


def test_fts5_search_at_most_k() -> None:
    r = Fts5Retriever(":memory:")
    _index_corpus(r)
    assert len(r.search("python", k=2)) <= 2


def test_fts5_empty_corpus_empty() -> None:
    r = Fts5Retriever(":memory:")
    assert r.search("anything", k=5) == []


def test_fts5_tie_break_handle_ascending() -> None:
    r = Fts5Retriever(":memory:")
    text = "shared common vocabulary tokens here"
    for h in ("zzz", "mmm", "aaa"):
        r.index(h, text)
    res = r.search("shared common vocabulary", k=3)
    assert [h for h, _ in res] == ["aaa", "mmm", "zzz"]


# --------------------------------------------------------------------------- mutation


def test_fts5_remove_drops_doc() -> None:
    r = Fts5Retriever(":memory:")
    _index_corpus(r)
    assert r.search("python", k=3)[0][0] == "python"
    r.remove("python")
    assert "python" not in [h for h, _ in r.search("python", k=3)]


def test_fts5_remove_absent_no_error() -> None:
    r = Fts5Retriever(":memory:")
    r.remove("never-indexed")  # must not raise


def test_fts5_upsert_replaces_text_no_duplicate() -> None:
    r = Fts5Retriever(":memory:")
    r.index("h", "alpha beta gamma")
    r.index("h", "delta epsilon zeta")  # overwrite
    assert r.search("delta", k=1)[0][0] == "h"
    assert r.search("alpha", k=1) == []          # old terms gone
    assert r.handles().count("h") == 1           # single row per handle


def test_fts5_handles_lists_indexed() -> None:
    r = Fts5Retriever(":memory:")
    _index_corpus(r)
    assert set(r.handles()) == {"cooking", "python", "gardening"}
    r.remove("python")
    assert set(r.handles()) == {"cooking", "gardening"}


# --------------------------------------------------------------------------- on-disk


def test_fts5_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "fts.db"
    r1 = Fts5Retriever(db)
    _index_corpus(r1)
    assert r1.search("python async", k=1)[0][0] == "python"
    r1.close()

    r2 = Fts5Retriever(db)
    try:
        assert r2.search("python async await", k=3)[0][0] == "python"
        r2.remove("python")
        assert "python" not in [h for h, _ in r2.search("python", k=3)]
    finally:
        r2.close()


def test_fts5_is_contentless_no_second_text_copy(tmp_path: Path) -> None:
    db = tmp_path / "cl.db"
    r = Fts5Retriever(db)
    secret = "supersecretuniquetoken alpha beta gamma"
    r.index("h", secret)
    r.close()
    con = sqlite3.connect(str(db))
    try:
        # Contentless: the indexed column is NOT stored -> reads back as NULL, and the
        # raw text is nowhere retrievable from the fts table (the body lives in notes/).
        rows = con.execute("SELECT text FROM fts_idx").fetchall()
        assert rows == [(None,)]
        assert all(secret not in str(v) for row in rows for v in row)
    finally:
        con.close()


def test_fts5_close_releases_file(tmp_path: Path) -> None:
    db = tmp_path / "lock.db"
    r = Fts5Retriever(db)
    r.index("h", "alpha beta")
    r.close()
    r.close()  # idempotent
    assert db.exists()
    os.remove(str(db))  # succeeds only once the sqlite lock is released
    assert not db.exists()


def test_fts5_context_manager(tmp_path: Path) -> None:
    db = tmp_path / "cm.db"
    with Fts5Retriever(db) as r:
        r.index("ctx", "context manager usage works")
        assert r.search("context", k=1)[0][0] == "ctx"
    os.remove(str(db))  # exiting the block closed it -> deletable


# --------------------------------------------------------------------------- factory


def test_make_retriever_prefers_fts5() -> None:
    r = make_retriever(":memory:")
    try:
        assert isinstance(r, Fts5Retriever)
        assert isinstance(r, Retriever)  # satisfies the runtime_checkable Protocol
    finally:
        r.close()


# --------------------------------------------------------------------------- parity


def test_parity_first_hit_agrees_with_lexical() -> None:
    lex = LexicalRetriever(":memory:")
    fts = Fts5Retriever(":memory:")
    _index_corpus(lex)
    _index_corpus(fts)
    for q in ["python async await", "recipe flour oven", "garden soil mulch"]:
        a = lex.search(q, k=3)
        b = fts.search(q, k=3)
        assert a and b
        assert a[0][0] == b[0][0], f"first-hit disagreement for {q!r}: {a[0]} vs {b[0]}"
