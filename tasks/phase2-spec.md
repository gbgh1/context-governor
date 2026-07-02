# Phase 2 Spec — Store + Retriever (durable state & demand paging)

> Authoritative contract for Phase 2. Implementers and the tester bind to THIS document.
> Field names, signatures, file formats, and behavior are normative — do not rename or
> paraphrase. Python 3.11+, `src/` layout, stdlib + `httpx` only (NO new runtime deps:
> persistence uses stdlib `sqlite3` / `json` / `pathlib`; lexical scoring is pure Python).

## 0. What Phase 2 is

Phase 2 builds the durable **Store** that sits behind the engine's narrow `Store` Protocol
(`page_out`, `get`) and adds the **page-IN / retrieval** side (demand paging). It does NOT
modify the Phase 1 engine. Four new modules + a facade, all under `src/contextmanager/`:

```
src/contextmanager/
  state_store.py   # StateStore — authoritative state.json, atomic rewrite-in-place
  note_store.py    # NoteStore  — markdown+frontmatter notes; implements Store (page_out/get)
  retriever.py     # Retriever Protocol + LexicalRetriever (default) + Embedder + LlamaServerEmbedder
  durable.py       # DurableStore facade — composes the above; engine Store + page-in API
tests/
  test_state_store.py
  test_note_store.py
  test_retriever.py
  test_durable.py
```

**Decision (vector index):** Phase 2 ships the **pure-Python `LexicalRetriever`** (BM25-style
bag-of-words scoring, persisted via stdlib `sqlite3`) as the default, fully-tested retriever —
no native/heavy deps, fully offline. The `Embedder` Protocol + `LlamaServerEmbedder`
(`/v1/embeddings`) are defined and tested so a vector retriever can drop in later behind the
SAME `Retriever` interface. A vector index (sqlite-vec/faiss) is explicitly deferred.

**Store root:** all runtime artifacts live under a configurable root dir (default
`./contextstore`), NOT the project's own `wiki/`. Layout:
```
<root>/state.json          # StateStore
<root>/notes/<handle>.md   # NoteStore notes (markdown + frontmatter)
<root>/index.db            # LexicalRetriever sqlite db
```

## 1. Shared error type

Define `class StoreError(Exception)` in `note_store.py` and reuse it across store modules
(import where needed). `retriever.py` defines `class RetrieverError(Exception)`.

## 2. `state_store.py` — NORMATIVE

```python
class StateStore:
    """Authoritative world/project state as JSON, rewritten in place atomically."""
    def __init__(self, path: str | os.PathLike) -> None: ...
        # path = the state.json file path. Parent dirs are created on first save.

    def load(self) -> dict: ...
        # Return the parsed dict, or {} if the file does not exist.
        # Raise StoreError on malformed JSON (do not silently reset).

    def save(self, state: dict) -> None: ...
        # ATOMIC write: write to "<path>.tmp" (utf-8, json.dumps sort_keys=True, indent=2),
        # flush + os.fsync the temp file, then os.replace(tmp, path). os.replace is atomic
        # on Windows and POSIX for same-volume. Create parent dirs if missing.

    def update(self, patch: dict) -> dict: ...
        # Shallow-merge patch into current load(), save(), return the merged dict.

    def render(self) -> str: ...
        # Deterministic text rendering for the engine's state_snapshot tier:
        # json.dumps(load(), sort_keys=True, indent=2, ensure_ascii=False). "" if state is {}.
```

## 3. `note_store.py` — NORMATIVE

```python
class StoreError(Exception): ...

@dataclass
class NoteMeta:
    id: str            # source message id (or note id)
    role: str          # "user"|"assistant"|"tool"|"note"|...
    handle: str        # the storage handle (== filename stem)
    created: str       # ISO-8601 UTC timestamp
    tokens: Optional[int]   # token count if known, else None
    links: list[str]   # wikilink targets (handles), may be empty

class NoteStore:
    """Persists content as human-auditable markdown notes with frontmatter.
    Implements the engine's Store Protocol (page_out, get)."""
    def __init__(self, root: str | os.PathLike) -> None: ...
        # notes live under <root>/notes/ ; created on first write.

    # --- engine Store Protocol ---
    def page_out(self, message: Message) -> str: ...
        # handle = handle_for(message.id). Write the note (overwrite if exists -> idempotent).
        # role = message.role, content = message.content, links = []. Return handle.
    def get(self, handle: str) -> str: ...
        # Return the note BODY (content only, frontmatter stripped). Raise StoreError if absent.

    # --- general note API ---
    def write_note(self, *, id: str, role: str, content: str,
                   tokens: Optional[int] = None, links: Optional[list[str]] = None) -> str: ...
        # Write/overwrite a note. Return its handle. links default [].
    def read_meta(self, handle: str) -> NoteMeta: ...
        # Parse frontmatter -> NoteMeta. Raise StoreError if absent/malformed.
    def has(self, handle: str) -> bool: ...
    def list_handles(self) -> list[str]: ...   # sorted
    def remove(self, handle: str) -> None: ...  # delete note; no error if absent

    @staticmethod
    def handle_for(message_id: str) -> str: ...
        # Deterministic, filesystem-safe slug from message_id: keep [A-Za-z0-9._-],
        # replace any other run with "-"; if the result is empty or collides risk,
        # append a short sha1 hex of the original id. Stable: same id -> same handle.
```

### Note file format (`<root>/notes/<handle>.md`)
YAML-style frontmatter (write/parse manually — NO yaml dependency; only the scalar/list
fields below), then a blank line, then the content body verbatim:
```
---
id: <id>
role: <role>
handle: <handle>
created: <iso8601>
tokens: <int or empty>
links: [<h1>, <h2>]
---

<content body, verbatim, may contain blank lines>
```
- Writing must be atomic (temp file + os.replace), like StateStore.
- `get()` returns everything AFTER the closing `---` line and its following blank line,
  with no trailing modification (preserve the body exactly as written).
- The parser must handle content bodies that themselves contain `---` lines (only the
  FIRST frontmatter block is metadata; split on the first two `---` delimiters only).

## 4. `retriever.py` — NORMATIVE

```python
class RetrieverError(Exception): ...

class Retriever(Protocol):
    def index(self, handle: str, text: str, metadata: Optional[dict] = None) -> None: ...
    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]: ...
        # Return up to k (handle, score) pairs, HIGHEST score first. Empty list if no match.
    def remove(self, handle: str) -> None: ...

class LexicalRetriever:   # implements Retriever; pure-Python BM25-style over stdlib sqlite3
    def __init__(self, path: str | os.PathLike = ":memory:") -> None: ...
        # path = sqlite db file (or ":memory:"). Schema: a docs table (handle TEXT PRIMARY KEY,
        # text TEXT, metadata TEXT json). index() upserts (handle unique -> replace).
    # Tokenization: lowercase, split on non-alphanumeric, drop empties. Scoring: Okapi BM25
    # (k1=1.5, b=0.75) computed in Python over the corpus loaded from sqlite. Deterministic
    # tie-break: higher score first, then handle ascending. search() ignores docs with score<=0.

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class LlamaServerEmbedder:   # implements Embedder
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 30.0,
                 client: "httpx.Client | None" = None, model: str = "") -> None: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
        # POST {base_url}/v1/embeddings  body {"input": texts, "model": model}
        # -> response {"data": [{"embedding": [float,...]}, ...]} ; return embeddings in
        # the SAME order as input. Wrap transport/HTTP/JSON errors in RetrieverError.
```
Note: `LlamaServerEmbedder` is defined+tested but NOT wired into any retriever in Phase 2
(vector index deferred). It exists so a future vector retriever drops in behind `Retriever`.

## 5. `durable.py` — NORMATIVE (the facade)

```python
@dataclass
class RetrievedSlice:
    handle: str
    content: str
    score: float

class DurableStore:
    """Composes StateStore + NoteStore + Retriever. Implements the engine's Store Protocol
    (page_out, get) AND the page-in/retrieval API the proxy/MCP will use."""
    def __init__(self, root: str | os.PathLike, *,
                 retriever: Optional[Retriever] = None) -> None: ...
        # Build NoteStore(root), StateStore(root/"state.json"),
        # retriever = retriever or LexicalRetriever(root/"index.db").
        # Expose .notes, .state, .retriever attributes.

    # --- engine Store Protocol ---
    def page_out(self, message: Message) -> str: ...
        # handle = notes.page_out(message); retriever.index(handle, message.content,
        # {"id": message.id, "role": message.role}); return handle.
    def get(self, handle: str) -> str: ...   # notes.get(handle)

    # --- page-IN / retrieval ---
    def page_in_by_id(self, message_id: str) -> Optional[str]: ...
        # handle = NoteStore.handle_for(message_id); return notes.get(handle) if present else None.
    def search(self, query: str, k: int = 5) -> list[RetrievedSlice]: ...
        # retriever.search(query, k) -> for each (handle, score) load notes.get(handle) ->
        # RetrievedSlice. Skip handles whose note is missing. Preserve score order.
    def page_in(self, query: str, budget_tokens: int, counter: TokenCounter,
                k: int = 5) -> list[RetrievedSlice]: ...
        # search(query, k); accumulate slices while the running token total (measured via
        # counter.count_text on each content) stays <= budget_tokens. The slice that would
        # overflow is TRUNCATED via counter.truncate_to_tokens to the remaining budget and
        # included only if remaining budget > 0; then stop. Never exceed budget_tokens.
```

## 6. Tests — what MUST be proven

`tests/test_state_store.py`
- load() returns {} when file absent; save() then load() round-trips; save() is atomic
  (no ".tmp" left behind; existing file intact if we simulate failure — at minimum assert
  tmp cleaned up and content valid JSON). update() shallow-merges. render() is deterministic
  & sorted; "" when empty. Malformed JSON file -> StoreError on load.

`tests/test_note_store.py`
- page_out(message) writes a note and returns a stable handle; get(handle) returns the body
  EXACTLY (including a body that itself contains a `---` line). handle_for is deterministic
  and filesystem-safe (weird ids with spaces/slashes -> safe slug). read_meta round-trips
  NoteMeta (id, role, tokens, links). has/list_handles/remove behave. Missing handle -> StoreError.
  Idempotent: page_out same id twice -> same handle, single note.

`tests/test_retriever.py`
- LexicalRetriever: index several docs; search returns the most lexically relevant handle
  first; unrelated query -> low/empty; remove() drops a doc; persists across re-open of the
  same db path (index, close, reopen, search still works). :memory: works. Deterministic
  tie-break (equal scores -> handle ascending).
- LlamaServerEmbedder: against MOCKED httpx (MockTransport via client param) returns
  embeddings in input order; HTTP 500 / malformed JSON -> RetrieverError.

`tests/test_durable.py`
- page_out then get round-trips; page_out also makes the content searchable (search finds it).
- page_in_by_id returns exact content for a paged-out message, None for unknown id.
- search returns RetrievedSlice list in score order, skipping missing notes.
- page_in respects budget_tokens EXACTLY: total measured tokens never exceeds budget; the
  overflowing slice is truncated; budget 0 -> empty list. Use a deterministic fake
  TokenCounter (reuse the conftest FakeCounter from Phase 1 if importable, else a local one).

## 7. Constraints

- NO new runtime dependencies. stdlib `sqlite3`, `json`, `pathlib`, `hashlib`, `datetime`,
  `os`, `re` + existing `httpx`. Dev: existing `pytest`/`hypothesis`.
- All disk writes atomic (temp + os.replace). Full type hints. `from __future__ import annotations`.
- Do NOT modify the Phase 1 engine modules. Do NOT edit `__init__.py` (the orchestrator updates
  exports after integration) and do NOT edit `conftest.py` except the tester, additively.
- Every module passes `python -m py_compile`. No real network in tests (mock httpx only).

## 8. Round-2 corrections (NORMATIVE — supersede §2–§6 where they conflict)

A code review found correctness/durability bugs that the fake-based tests masked. Apply ALL.

### 8.1 note_store.py + state_store.py — newline-stable I/O (fixes H1, CRITICAL-ish)
Notes are currently written/read in text mode, so Windows newline translation CORRUPTS any
body containing `\r` and breaks notes the moment they cross an OS (the project syncs via
OneDrive/Obsidian). Fix BOTH the writer and reader to disable newline translation:
- Writing: `open(tmp, "w", encoding="utf-8", newline="")` (so `\n` is written as-is, no `\r\n`).
- Reading: `path.read_text(encoding="utf-8")` → change to `open(path, "r", encoding="utf-8",
  newline="").read()` (or `read_text(..., newline="")` on 3.13; use the explicit open form for
  3.11 compat). Apply to `NoteStore.get`, `NoteStore.read_meta`, and the writer.
- Apply the same `newline=""` to `StateStore.save` for byte-deterministic output.
- The frontmatter parser must split on lines equal to `---` (after `.split("\n")`), and the
  body returned by `get()` must be byte-identical to the `content` passed in for ANY content,
  including content containing `\r\n`, `\r`, internal `---`, a leading `---`, an empty body,
  and a body with no trailing newline.

### 8.2 durable.py — clamp page_in truncation overshoot (fixes H2, protects engine invariant)
`page_in` must NEVER let total tokens exceed `budget_tokens`, for ANY TokenCounter (the real
`truncate_to_tokens` can overshoot the requested target). Change the overflow branch from
trusting `recount` to clamping:
```python
truncated = counter.truncate_to_tokens(sl.content, remaining)
recount = counter.count_text(truncated)
if 0 < recount <= remaining:
    out.append(RetrievedSlice(sl.handle, truncated, sl.score))
    running += recount
# else: overshoot or empty -> drop this slice
break
```
Also guard the non-overflow branch against appending a zero-content slice: `if 0 < count <= remaining` (fixes L11).

### 8.3 note_store.py — clean up temp file on failure (fixes M4)
`NoteStore._atomic_write` must mirror `StateStore.save`: wrap the write/fsync/replace in
try/except and `os.unlink(tmp)` (ignore errors) on any failure, so a failed write never leaves
an orphan `<handle>.md.tmp`.

### 8.4 retriever.py + durable.py — release the sqlite connection (fixes M5)
`LexicalRetriever` must add `close()` (calls `self._conn.close()`), `__enter__`, and `__exit__`
(mirroring `LlamaServerEmbedder`). `DurableStore` must add `close()` that calls
`self.retriever.close()` if the retriever has a `close` attribute (and is safe to call once).
This prevents Windows file-locks on `index.db` and fd leaks.

### 8.5 note_store.py — dot-only ids (fixes L12)
`handle_for(".")`, `handle_for("..")`, or any id that slugs to only dots must NOT produce a
hidden/invalid filename. If the slug is empty OR consists only of `.` characters, fall back to
(or suffix with) the short sha1 hex so the handle is a visible, valid `.md` stem.

### 8.6 Additional REQUIRED tests (fixes H3, M6, M7, M8; clean up L13)
- `test_handle_for_no_collision_across_separator_collapse` (note_store): assert
  `len({NoteStore.handle_for(s) for s in ["a b","a-b","a/b","a.b","a_b","a b c","a-b-c"]}) == 7`.
- `test_note_body_verbatim_edge_cases` (note_store): round-trip bodies = `"---\nfoo"` (leading
  `---`), `""` (empty), `"abc"` (no trailing newline), and `"line1\r\nline2\r\n"` (CRLF) — each
  `get()` must equal the exact input bytes/string written.
- `test_state_store_atomic_on_simulated_failure` (state_store): pre-write a valid file;
  monkeypatch `os.replace` to raise; assert `save()` raises, the ORIGINAL file is intact, and no
  `.tmp` remains.
- `test_embedder_transport_error_wrapped` (retriever): MockTransport handler raises
  `httpx.ConnectError` → `embed()` raises `RetrieverError`.
- `test_embedder_count_mismatch_raises` (retriever): server returns fewer `data` items than
  inputs → `RetrieverError`.
- `test_lexical_retriever_close_releases_file` (retriever): open file-path retriever, index,
  `close()`, then the db file can be deleted (Windows lock released) — or at minimum `close()`
  is idempotent and search after close raises a clear error / reopen works.
- Fix the confusing `or` assertion in `test_durable.py` (L13) to assert
  `ds.notes.handle_for("m1") in handles`.
