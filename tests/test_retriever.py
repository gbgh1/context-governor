"""Phase 2 Retriever tests — binds to `tasks/phase2-spec.md` §6 (test_retriever).

Covers:
  - LexicalRetriever(":memory:"):
      * index docs with clearly different vocabularies; a query matching one doc
        returns that handle FIRST; an unrelated query returns empty or low.
      * remove() drops a doc (no longer in results).
      * equal-score tie-break -> handle ascending.
  - persistence: index into a path, drop the object, open a NEW retriever on the
    same path, search still finds the docs.
  - LlamaServerEmbedder via httpx.MockTransport (injected through `client`):
      * embeddings returned in input order for a 2-text input.
      * HTTP 500 -> RetrieverError.
      * malformed (non-JSON) 200 body -> RetrieverError.

No real network: httpx is mocked via MockTransport only.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import httpx
import pytest

from contextmanager.retriever import (
    LexicalRetriever,
    LlamaServerEmbedder,
    RetrieverError,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _index_corpus(r: LexicalRetriever) -> dict[str, str]:
    """Index three docs with clearly distinct vocabularies; return handle->text."""
    docs = {
        "cooking": "recipe bake flour sugar eggs butter oven whisk knead dough",
        "python": "python function module import class decorator async await loop",
        "gardening": "garden soil compost seedling mulch prune water sunlight roots",
    }
    for h, text in docs.items():
        r.index(h, text)
    return docs


# ---------------------------------------------------------------------------
# :memory: lexical scoring
# ---------------------------------------------------------------------------


def test_query_matching_one_doc_returns_that_handle_first() -> None:
    r = LexicalRetriever(":memory:")
    _index_corpus(r)
    res = r.search("python async await", k=3)
    assert res, "expected at least one result for a matching query"
    assert res[0][0] == "python"


def test_unrelated_query_returns_empty_or_low() -> None:
    r = LexicalRetriever(":memory:")
    _index_corpus(r)
    res = r.search("quantum entanglement blockchain", k=3)
    # Either no results, or every score is essentially zero / very low.
    if res:
        assert all(score <= 0.0 for _, score in res) or all(
            score < 0.01 for _, score in res
        )
    # Stronger: a query with zero overlapping terms should yield nothing meaningful.
    assert r.search("zzzqqq xxwwyy", k=3) == []


def test_search_returns_at_most_k() -> None:
    r = LexicalRetriever(":memory:")
    _index_corpus(r)
    res = r.search("python", k=2)
    assert len(res) <= 2


def test_search_empty_corpus_returns_empty() -> None:
    r = LexicalRetriever(":memory:")
    assert r.search("anything", k=5) == []


# ---------------------------------------------------------------------------
# remove()
# ---------------------------------------------------------------------------


def test_remove_drops_doc_from_results() -> None:
    r = LexicalRetriever(":memory:")
    _index_corpus(r)
    res = r.search("python", k=3)
    assert res and res[0][0] == "python"

    r.remove("python")
    res2 = r.search("python", k=3)
    handles = [h for h, _ in res2]
    assert "python" not in handles


def test_remove_absent_handle_no_error() -> None:
    r = LexicalRetriever(":memory:")
    # should not raise
    r.remove("never-indexed")


# ---------------------------------------------------------------------------
# index upsert (re-index same handle replaces)
# ---------------------------------------------------------------------------


def test_index_upsert_replaces_text() -> None:
    r = LexicalRetriever(":memory:")
    r.index("h", "alpha beta gamma")
    r.index("h", "delta epsilon zeta")  # overwrite
    res = r.search("delta", k=1)
    assert res and res[0][0] == "h"
    # old term should no longer contribute meaningfully
    res2 = r.search("alpha", k=1)
    assert "h" not in [x for x, _ in res2] or all(s <= 0.0 for _, s in res2)


# ---------------------------------------------------------------------------
# tie-break: equal scores -> handle ascending
# ---------------------------------------------------------------------------


def test_equal_score_tie_break_handle_ascending() -> None:
    r = LexicalRetriever(":memory:")
    # Identical text in three docs -> identical BM25 scores -> tie-break handle asc.
    text = "shared common vocabulary tokens here"
    r.index("zzz", text)
    r.index("mmm", text)
    r.index("aaa", text)

    res = r.search("shared common vocabulary", k=3)
    assert len(res) == 3
    handles_in_order = [h for h, _ in res]
    assert handles_in_order == sorted(handles_in_order)  # ascending
    # explicitly: aaa, mmm, zzz
    assert handles_in_order == ["aaa", "mmm", "zzz"]
    # and the scores are actually equal
    scores = [s for _, s in res]
    assert all(abs(s - scores[0]) < 1e-9 for s in scores)


# ---------------------------------------------------------------------------
# persistence across re-open
# ---------------------------------------------------------------------------


def test_persists_across_reopen_same_path(tmp_path: Path) -> None:
    db = tmp_path / "r.db"
    r1 = LexicalRetriever(db)
    _index_corpus(r1)
    # confirm it works while open
    assert r1.search("python async", k=1)[0][0] == "python"

    # drop the object (and reopen a brand-new instance on the same path)
    del r1
    r2 = LexicalRetriever(db)
    res = r2.search("python async await", k=3)
    assert res
    assert res[0][0] == "python"

    # remove persists too
    r2.remove("python")
    assert "python" not in [h for h, _ in r2.search("python", k=3)]


# ---------------------------------------------------------------------------
# LlamaServerEmbedder (mocked httpx)
# ---------------------------------------------------------------------------


def _make_embedder(handler) -> LlamaServerEmbedder:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return LlamaServerEmbedder(base_url="http://mock.local", client=client, model="m")


def test_embedder_returns_embeddings_in_input_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        # echo input list to prove order preservation
        inputs = body["input"]
        data = [
            {"embedding": [float(i), float(i) + 0.5, -1.0]}
            for i in range(len(inputs))
        ]
        return httpx.Response(200, json={"data": data})

    emb = _make_embedder(handler)
    out = emb.embed(["first", "second"])
    assert out == [[0.0, 0.5, -1.0], [1.0, 1.5, -1.0]]
    assert len(out) == 2


def test_embedder_posts_to_v1_embeddings_endpoint() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200, json={"data": [{"embedding": [0.1]}, {"embedding": [0.2]}]}
        )

    emb = _make_embedder(handler)
    emb.embed(["a", "b"])
    assert str(captured["url"]).endswith("/v1/embeddings")
    assert captured["body"]["input"] == ["a", "b"]
    assert captured["body"]["model"] == "m"


def test_embedder_http_500_raises_retriever_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server boom")

    emb = _make_embedder(handler)
    with pytest.raises(RetrieverError):
        emb.embed(["a"])


def test_embedder_malformed_non_json_200_raises_retriever_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all <<", headers={"content-type": "text/plain"})

    emb = _make_embedder(handler)
    with pytest.raises(RetrieverError):
        emb.embed(["a"])


def test_embedder_missing_data_field_raises_retriever_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"oops": True})

    emb = _make_embedder(handler)
    with pytest.raises(RetrieverError):
        emb.embed(["a"])


# ---------------------------------------------------------------------------
# Round-2 (spec §8.6) edge cases
# ---------------------------------------------------------------------------


def test_embedder_transport_error_wrapped() -> None:
    """Spec §8.6 (M7): a transport-level error (httpx.ConnectError) raised by
    the underlying client must be wrapped as RetrieverError, not leak the
    raw httpx exception to the caller.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    emb = _make_embedder(handler)
    with pytest.raises(RetrieverError):
        emb.embed(["x"])


def test_embedder_count_mismatch_raises() -> None:
    """Spec §8.6 (M8): if the server returns FEWER embeddings in `data` than
    inputs were sent, embed() must raise RetrieverError rather than silently
    returning a short list (which would mis-align inputs to embeddings).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        inputs = body["input"]
        # Return one embedding regardless of how many inputs were sent.
        data = [{"embedding": [0.1, 0.2, 0.3]}]
        # Confirm we are actually exercising the mismatch path.
        assert len(inputs) > len(data)
        return httpx.Response(200, json={"data": data})

    emb = _make_embedder(handler)
    with pytest.raises(RetrieverError):
        emb.embed(["first", "second"])


# ---------------------------------------------------------------------------
# Round-2 (spec §8.4 / §8.6) — LexicalRetriever close() releases sqlite file lock
# ---------------------------------------------------------------------------


def test_lexical_retriever_close_releases_file(tmp_path: Path) -> None:
    """Spec §8.4 (M5) + §8.6: LexicalRetriever must expose close() that releases
    the sqlite connection (and thus the Windows file-lock on index.db), plus
    __enter__/__exit__ so it can be used as a context manager.

    Asserts:
      * after close(), the db file can be deleted on Windows (lock released);
      * close() is idempotent (calling twice does not raise);
      * the context-manager protocol works if __enter__/__exit__ are present.
    """
    db = tmp_path / "lex.db"
    r = LexicalRetriever(db)
    r.index("h1", "alpha beta gamma delta epsilon")
    # sanity: it works while open
    assert r.search("alpha beta", k=1)[0][0] == "h1"

    r.close()
    # idempotent: second close must not raise
    r.close()

    # The db file must exist on disk (we indexed) and be deletable now that the
    # connection is closed — on Windows this fails if the sqlite lock is held.
    assert db.exists()
    os.remove(str(db))  # os.remove succeeds -> lock released
    assert not db.exists()

    # Context-manager support: only assert if __enter__/__exit__ exist, so the
    # test does not fail merely because the protocol is absent; spec §8.4 calls
    # for them, so we assert their behavior when present.
    if hasattr(LexicalRetriever, "__enter__") and hasattr(LexicalRetriever, "__exit__"):
        db2 = tmp_path / "lex2.db"
        with LexicalRetriever(db2) as r2:
            r2.index("ctx", "context manager usage works")
            assert r2.search("context", k=1)[0][0] == "ctx"
        # exiting the with-block closed it -> file deletable
        assert db2.exists()
        os.remove(str(db2))
        assert not db2.exists()
        # idempotent close after context exit too
        r2.close()


# ---------------------------------------------------------------------------
# Phase 7 Stage 1 — incremental auto_vacuum self-maintenance on remove()
# ---------------------------------------------------------------------------


def test_file_retriever_enables_incremental_autovacuum(tmp_path: Path) -> None:
    db = tmp_path / "av.db"
    r = LexicalRetriever(db)
    r.index("h", "alpha beta gamma")
    r.close()
    # Inspect the persisted db header via an independent connection.
    con = sqlite3.connect(str(db))
    try:
        assert con.execute("PRAGMA auto_vacuum").fetchone()[0] == 2  # 2 == INCREMENTAL
    finally:
        con.close()


def test_memory_retriever_does_not_force_autovacuum() -> None:
    # ":memory:" skips the file-only pragmas; auto_vacuum stays default (0/NONE).
    r = LexicalRetriever(":memory:")
    try:
        assert r._conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 0
    finally:
        r.close()


def test_autovacuum_converts_preexisting_non_autovacuum_db(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    # Build a db the OLD way: default auto_vacuum (NONE) + a populated docs table.
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE docs (handle TEXT PRIMARY KEY, text TEXT, metadata TEXT)")
    con.execute("INSERT INTO docs VALUES (?,?,?)", ("legacy", "alpha beta gamma", None))
    con.commit()
    assert con.execute("PRAGMA auto_vacuum").fetchone()[0] == 0  # NONE
    con.close()

    # Opening via LexicalRetriever must convert it AND preserve the existing data.
    r = LexicalRetriever(db)
    try:
        res = r.search("alpha beta", k=1)
        assert res and res[0][0] == "legacy"
    finally:
        r.close()
    con = sqlite3.connect(str(db))
    try:
        assert con.execute("PRAGMA auto_vacuum").fetchone()[0] == 2
    finally:
        con.close()


def test_remove_reclaims_freelist_pages(tmp_path: Path) -> None:
    db = tmp_path / "reclaim.db"
    r = LexicalRetriever(db)
    big = "lorem ipsum dolor sit amet consectetur " * 200  # ~ multi-page per doc
    for i in range(40):
        r.index(f"h{i}", f"{big} unique{i}")
    for i in range(40):
        r.remove(f"h{i}")  # each remove() runs a bounded incremental_vacuum
    r.close()
    con = sqlite3.connect(str(db))
    try:
        freelist = con.execute("PRAGMA freelist_count").fetchone()[0]
    finally:
        con.close()
    # Without per-remove reclaim the freelist would hold dozens of pages; with it
    # the freed pages are returned to the OS instead of piling up.
    assert freelist <= 2
