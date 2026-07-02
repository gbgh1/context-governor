from __future__ import annotations

import json
import math
import re
import sqlite3
from typing import Any, Optional, Protocol, runtime_checkable

import httpx


class RetrieverError(Exception):
    """Raised on retriever/embedder failures (transport, HTTP, JSON, schema)."""


# --------------------------------------------------------------------------- #
# Retriever Protocol + LexicalRetriever (pure-Python BM25 over stdlib sqlite3)
# --------------------------------------------------------------------------- #


@runtime_checkable
class Retriever(Protocol):
    def index(self, handle: str, text: str, metadata: Optional[dict] = None) -> None: ...
    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]: ...
    def remove(self, handle: str) -> None: ...


_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop empties."""
    return [tok for tok in _TOKEN_SPLIT.split(text.lower()) if tok]


def _enable_incremental_autovacuum(conn: sqlite3.Connection) -> None:
    """Switch a file-backed db to INCREMENTAL auto_vacuum (mode 2) so deletes can later
    return free pages to the OS via a bounded `incremental_vacuum`. MUST run before any
    write (incl. `journal_mode=WAL`, which initializes the header): once a db is created
    under auto_vacuum=0 the pragma is silently ignored. A fresh db needs only the pragma;
    a pre-existing non-auto-vacuum db is converted by a one-time VACUUM. Best-effort —
    an old sqlite or a locked db may refuse, and the store still works (just no reclaim).
    """
    try:
        row = conn.execute("PRAGMA auto_vacuum").fetchone()
        if row is not None and int(row[0]) == 2:
            return  # already INCREMENTAL
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        has_schema = conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
        if has_schema is not None:
            conn.execute("VACUUM")
    except sqlite3.Error:
        pass


def fts5_available() -> bool:
    """True iff this sqlite build can create a contentless, deletable FTS5 table
    (`content=''`, `contentless_delete=1` — needs FTS5 + sqlite >= 3.43)."""
    con = sqlite3.connect(":memory:")
    try:
        con.execute(
            "CREATE VIRTUAL TABLE _probe USING fts5(x, content='', contentless_delete=1)"
        )
        return True
    except sqlite3.Error:
        return False
    finally:
        con.close()


def make_retriever(path: str | Any = ":memory:") -> "Retriever":
    """Prefer the C-speed FTS5 backend; fall back to the portable pure-Python BM25
    when FTS5 (or `contentless_delete`) isn't compiled into this sqlite. This keeps the
    published package working everywhere while using the fast path where available."""
    if fts5_available():
        try:
            return Fts5Retriever(path)
        except sqlite3.Error:
            pass
    return LexicalRetriever(path)


class LexicalRetriever:
    """Pure-Python Okapi BM25 retriever backed by a stdlib sqlite3 database.

    Schema: docs(handle TEXT PRIMARY KEY, text TEXT, metadata TEXT json).
    index() upserts (REPLACE). search() loads the corpus from sqlite at query
    time and scores with Okapi BM25 (k1=1.5, b=0.75) — fine for our scale.
    """

    _K1 = 1.5
    _B = 0.75

    def __init__(self, path: str | Any = ":memory:") -> None:
        self._path: str = str(path)
        # Keep a persistent connection open across calls. For ":memory:" this is
        # required for the in-memory db to survive between method invocations;
        # for a file path it is also fine (and durable via commits after writes).
        # check_same_thread=False: the proxy runs the rewrite (which touches this db) in a
        # worker thread via asyncio.to_thread; a single proxy-side lock serializes access so
        # the connection is never used concurrently. Safe for our sequential cross-thread use.
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
            self._path, check_same_thread=False
        )
        if self._path != ":memory:":
            # INCREMENTAL auto_vacuum: pages freed by remove() go on the freelist and
            # are returned to the OS by a bounded `incremental_vacuum` after each delete
            # — self-maintenance with NO full-db rewrite on the hot path. This MUST run
            # before any write (incl. `journal_mode=WAL`, which initializes the header):
            # once the db is initialized under auto_vacuum=0 the pragma is silently
            # ignored, so a fresh db needs it first and a pre-existing one needs a VACUUM.
            # (Pointless for ":memory:" -> skipped.)
            _enable_incremental_autovacuum(self._conn)
            # WAL: a writer doesn't block readers, so the proxy and the MCP server
            # can safely share one index.db concurrently. synchronous=NORMAL keeps
            # durability across app crashes (only at risk on OS/power loss) while
            # avoiding an fsync per write; busy_timeout waits out a peer's write
            # lock instead of erroring. Best-effort (older sqlite may refuse).
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute("PRAGMA busy_timeout=5000")
            except sqlite3.Error:
                pass
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS docs ("
            "handle TEXT PRIMARY KEY, "
            "text TEXT, "
            "metadata TEXT)"
        )
        self._conn.commit()

    # --------------------------------------------------------------- index / remove

    def index(self, handle: str, text: str, metadata: Optional[dict] = None) -> None:
        meta_json = json.dumps(metadata) if metadata is not None else None
        # UPSERT: handle is the PRIMARY KEY -> REPLACE inserts-or-overwrites.
        self._conn.execute(
            "INSERT OR REPLACE INTO docs (handle, text, metadata) VALUES (?, ?, ?)",
            (handle, text, meta_json),
        )
        self._conn.commit()

    def remove(self, handle: str) -> None:
        self._conn.execute("DELETE FROM docs WHERE handle = ?", (handle,))
        self._conn.commit()
        # Return the just-freed pages to the OS — bounded to at most 64 pages per call
        # so eviction never triggers an unbounded rewrite. No-op for ":memory:".
        # NB: a plain execute("PRAGMA incremental_vacuum(N)") only steps the statement
        # ONCE (reclaims a single page); executescript drives it to completion, so the
        # bound is the pragma's N, not the Python step count.
        if self._path != ":memory:":
            try:
                self._conn.executescript("PRAGMA incremental_vacuum(64);")
            except sqlite3.Error:
                pass

    def handles(self) -> list[str]:
        """Every indexed handle — the liveness root for the store's GC."""
        return [r[0] for r in self._conn.execute("SELECT handle FROM docs").fetchall()]

    # ----------------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Close the sqlite connection. Idempotent / safe to call twice (spec §8.4)."""
        conn = self._conn
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> "LexicalRetriever":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------------------------------------------------------------------- search

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q_terms = _tokenize(query)
        if not q_terms:
            return []

        # Load the corpus from sqlite.
        rows = self._conn.execute("SELECT handle, text FROM docs").fetchall()
        if not rows:
            return []

        # Tokenize each doc once; record handle + doc length + term freqs.
        docs: list[tuple[str, list[str]]] = []
        for handle, text in rows:
            docs.append((handle, _tokenize(text or "")))

        n_docs = len(docs)
        avgdl: float = sum(len(toks) for _, toks in docs) / n_docs if n_docs else 0.0

        # Document frequency per query term (count docs containing the term).
        df: dict[str, int] = {term: 0 for term in q_terms}
        for _, toks in docs:
            tok_set = set(toks)
            for term in q_terms:
                if term in tok_set:
                    df[term] += 1

        # IDF (Okapi BM25 variant): ln( (N - df + 0.5) / (df + 0.5) + 1 ).
        idf: dict[str, float] = {}
        for term in q_terms:
            df_t = df[term]
            idf[term] = math.log((n_docs - df_t + 0.5) / (df_t + 0.5) + 1.0)

        # Score each doc.
        scored: list[tuple[str, float]] = []
        for handle, toks in docs:
            dl = len(toks)
            if dl == 0:
                continue
            tf_counts: dict[str, int] = {}
            for t in toks:
                if t in idf:
                    tf_counts[t] = tf_counts.get(t, 0) + 1
            if not tf_counts:
                continue
            denom_norm = self._K1 * (1.0 - self._B + self._B * (dl / avgdl)) if avgdl > 0 else self._K1
            score = 0.0
            for term, tf in tf_counts.items():
                score += idf[term] * (tf * (self._K1 + 1.0)) / (tf + denom_norm)
            if score > 0.0:
                scored.append((handle, score))

        # Deterministic tie-break: score desc, then handle asc.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]


# --------------------------------------------------------------------------- #
# Fts5Retriever — SQLite FTS5 inverted index (C-speed, sublinear, contentless)
# --------------------------------------------------------------------------- #


class Fts5Retriever:
    """Retriever backed by SQLite FTS5 — a C-level inverted index (sublinear search).

    CONTENTLESS (`content=''`, `contentless_delete=1`): the FTS table stores ONLY the
    inverted index, never a second copy of the text — the canonical body lives in the
    NoteStore's `notes/*.md`, killing the duplication the pure-Python backend incurs. A
    `handle_map(rid, handle)` side table bridges FTS rowids to handles (a contentless
    table can't store retrievable columns). Ranking uses the built-in `bm25()`; results
    are returned POSITIVE and DESCENDING (best first), tie-broken by handle ascending,
    to match `LexicalRetriever`'s contract. Implements the same `Retriever` Protocol
    (index/search/remove) plus `handles`/`close`/context-manager. Chosen automatically
    by `make_retriever` when FTS5 is compiled in; otherwise `LexicalRetriever` is used.
    """

    def __init__(self, path: str | Any = ":memory:") -> None:
        self._path: str = str(path)
        # check_same_thread=False: the proxy runs the rewrite (which touches this db) in a
        # worker thread via asyncio.to_thread; a single proxy-side lock serializes access so
        # the connection is never used concurrently. Safe for our sequential cross-thread use.
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
            self._path, check_same_thread=False
        )
        if self._path != ":memory:":
            # auto_vacuum BEFORE any write (see LexicalRetriever), then WAL for safe
            # concurrent proxy+MCP sharing of one index.db.
            _enable_incremental_autovacuum(self._conn)
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute("PRAGMA busy_timeout=5000")
            except sqlite3.Error:
                pass
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS handle_map "
            "(rid INTEGER PRIMARY KEY, handle TEXT UNIQUE NOT NULL)"
        )
        # NB: a distinct table name ("fts_idx", not "docs") so opening an index.db that a
        # previous LexicalRetriever created (which owns a regular `docs` table) does NOT
        # collide — the legacy table is simply ignored, and DurableStore reindexes the
        # notes into fts_idx. Backward-compatible: "just run the governor and continue".
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_idx USING "
            "fts5(text, content='', contentless_delete=1)"
        )
        self._conn.commit()

    # --------------------------------------------------------------- index / remove

    def _rid(self, handle: str) -> Optional[int]:
        row = self._conn.execute(
            "SELECT rid FROM handle_map WHERE handle=?", (handle,)
        ).fetchone()
        return int(row[0]) if row is not None else None

    def index(self, handle: str, text: str, metadata: Optional[dict] = None) -> None:
        # metadata is intentionally ignored: a contentless index stores no columns,
        # and role/id already live in the note frontmatter.
        rid = self._rid(handle)
        if rid is None:
            cur = self._conn.execute(
                "INSERT INTO handle_map(handle) VALUES(?)", (handle,)
            )
            rid = int(cur.lastrowid)
        else:
            # UPSERT: contentless_delete=1 lets us delete the old indexed row by rowid.
            self._conn.execute("DELETE FROM fts_idx WHERE rowid=?", (rid,))
        self._conn.execute("INSERT INTO fts_idx(rowid, text) VALUES(?, ?)", (rid, text or ""))
        self._conn.commit()

    def remove(self, handle: str) -> None:
        rid = self._rid(handle)
        if rid is None:
            return
        self._conn.execute("DELETE FROM fts_idx WHERE rowid=?", (rid,))
        self._conn.execute("DELETE FROM handle_map WHERE rid=?", (rid,))
        self._conn.commit()
        if self._path != ":memory:":
            try:
                self._conn.executescript("PRAGMA incremental_vacuum(64);")
            except sqlite3.Error:
                pass

    def handles(self) -> list[str]:
        """Every indexed handle — the liveness root for the store's GC."""
        return [
            r[0] for r in self._conn.execute("SELECT handle FROM handle_map").fetchall()
        ]

    # ---------------------------------------------------------------------- search

    @staticmethod
    def _match_query(query: str) -> str:
        terms = _tokenize(query)
        if not terms:
            return ""
        # Quote each (already alnum-only) term as a literal and OR them, so any overlap
        # matches and bm25 ranks — mirrors the lexical-union semantics of the fallback.
        return " OR ".join(f'"{t}"' for t in terms)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        match = self._match_query(query)
        if not match:
            return []
        try:
            rows = self._conn.execute(
                "SELECT m.handle, bm25(fts_idx) AS s "
                "FROM fts_idx JOIN handle_map m ON m.rid = fts_idx.rowid "
                "WHERE fts_idx MATCH ? ORDER BY s, m.handle LIMIT ?",
                (match, k),
            ).fetchall()
        except sqlite3.Error:
            return []
        # bm25() is <= 0 with more-negative = better; negate -> positive, descending.
        return [(h, -float(s)) for h, s in rows]

    # ----------------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Close the sqlite connection. Idempotent / safe to call twice."""
        conn = self._conn
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> "Fts5Retriever":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Embedder Protocol + LlamaServerEmbedder (httpx /v1/embeddings)
# --------------------------------------------------------------------------- #


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LlamaServerEmbedder:
    """Embedder backed by llama-server's OpenAI-compatible /v1/embeddings route."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        client: Optional[httpx.Client] = None,
        model: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.model = model
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None

    # ------------------------------------------------------------------ helpers

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        url = f"{self.base_url}{path}"
        try:
            resp = self._client.post(url, json=body, headers=self._headers())
        except httpx.TransportError as e:
            raise RetrieverError(f"Transport error POST {url}: {e}") from e
        except httpx.HTTPError as e:
            raise RetrieverError(f"HTTP error POST {url}: {e}") from e
        return resp

    @staticmethod
    def _raise_for_status(resp: httpx.Response, path: str) -> None:
        if resp.status_code >= 400:
            raise RetrieverError(
                f"HTTP {resp.status_code} from {path}: {resp.text[:200]}"
            )

    @staticmethod
    def _json(resp: httpx.Response, path: str) -> Any:
        try:
            return resp.json()
        except Exception as e:
            raise RetrieverError(f"malformed JSON from {path}: {e}") from e

    # ------------------------------------------------------------------- embed

    def embed(self, texts: list[str]) -> list[list[float]]:
        path = "/v1/embeddings"
        resp = self._post(path, {"input": texts, "model": self.model})
        self._raise_for_status(resp, path)
        data = self._json(resp, path)

        if not isinstance(data, dict):
            raise RetrieverError(f"Unexpected {path} response (not an object): {data!r}")
        items = data.get("data")
        if not isinstance(items, list):
            raise RetrieverError(
                f"Unexpected {path} response (no data list): {data!r}"
            )
        if len(items) != len(texts):
            raise RetrieverError(
                f"Unexpected {path} response (got {len(items)} embeddings for "
                f"{len(texts)} inputs): {data!r}"
            )

        embeddings: list[list[float]] = []
        for item in items:
            if not isinstance(item, dict):
                raise RetrieverError(
                    f"Unexpected {path} response (embedding entry not an object): {item!r}"
                )
            vec = item.get("embedding")
            if not isinstance(vec, list):
                raise RetrieverError(
                    f"Unexpected {path} response (no embedding list): {item!r}"
                )
            try:
                embeddings.append([float(x) for x in vec])
            except (TypeError, ValueError) as e:
                raise RetrieverError(
                    f"Unexpected {path} response (non-float embedding component): {item!r}"
                ) from e
        return embeddings

    # ------------------------------------------------------------------- cleanup

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "LlamaServerEmbedder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
