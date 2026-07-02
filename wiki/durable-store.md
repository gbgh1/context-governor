---
title: The DurableStore — externalized state
updated: 2026-07-01
tags: [store, retriever, design]
---

# The DurableStore

The externalized, human-auditable home for everything paged out of context. Composes three
parts behind one facade (`src/contextmanager/durable.py`); shared by [[surface-a-proxy]] and
[[surface-b-mcp]]. Part of the [[context-governor]].

## Three components

- **`StateStore`** (`state.json`) — authoritative world/project state as a single JSON object,
  rewritten **atomically** (temp file + `os.fsync` + `os.replace`). `render()` gives a
  deterministic text snapshot for the engine's state tier. This is the source of truth that
  survives compaction.
- **`NoteStore`** (`notes/*.md`) — content persisted as markdown notes with frontmatter
  (`id, role, handle, created, tokens, links`). Bodies round-trip **byte-exact** (`newline=""`,
  first-`---`-block-only frontmatter split). `handle_for(id)` is a deterministic,
  filesystem-safe slug → same id, same handle, same file (idempotent overwrite).
- **Retriever** — behind a `Retriever` Protocol, picked by `make_retriever()`:
  **`Fts5Retriever`** (contentless SQLite FTS5, ~185× faster search, no second text copy)
  when FTS5 is compiled in, else the portable pure-Python **`LexicalRetriever`** (Okapi BM25
  over stdlib `sqlite3`). Both over `index.db`; a vector backend can swap in later.
- **`HotnessTracker`** (`hotness.db`) — decay-on-read working-set signal: every access bumps a
  per-handle score that half-lives away (lazy decay, O(1)). Blends into search ranking
  (`rerank_weight`) and picks the victims for the lifecycle tiers below.

## Page-out / page-in

- `page_out(message)` → writes the note AND indexes its content for search → returns a handle.
- `page_in(query, budget_tokens, counter, k)` → BM25 search, then assemble slices under an
  **exact token budget**: include whole slices that fit; truncate the last to the remaining
  budget via `counter.truncate_to_tokens`, clamping against overshoot. This budgeted recall is
  what `context_rehydrate` and the proxy's auto-rehydration build on.

## Lifecycle tiers — lossless by construction (Phase 9)

A note descends through tiers as it cools, and **no tier destroys content by default**:

1. **Live** (`notes/<h>.md`) — searchable, hot.
2. **Compressed** (`notes/<h>.md.gz`, via `compress_cold`) — gzipped at rest, still indexed
   and searchable; reads decompress transparently, byte-exact.
3. **Archived** (`archive/<h>.md.gz`, via `evict_cold` / `gc`) — OUT of the index, hotness,
   and `corpus_size` (the working set stays bounded), but the body survives. A later `get()`
   **resurrects** it: restore + re-index + hotness bump, byte-exact. Works across restarts
   and across both surfaces (the archive lives in the shared store root).
4. **Deleted** — only ever by the explicit opt-in `archive=False` on `evict_cold`/`gc`.

The in-memory **ghost ring** still records evictions; a request for one counts as a
`ghost_hit` (the "cut too deep" cascade-miss signal) even when the resurrection succeeds.
`stats()` reports `archived_count` + `resurrections` alongside the retrieval counters, and
the proxy `/metrics` surfaces the whole block. The archive grows unbounded by design —
disk is cheap, content is precious; prune it manually (they are plain `.md.gz` files) or
hard-delete via `archive=False` when disk truly matters.

## Why markdown notes

`notes/*.md` with `[[wikilink]]`-style cross-refs are human-auditable and Obsidian-compatible —
a clean surface for reviewing what was externalized. The runtime store (`contextstore/`) is
gitignored because it holds paged-out conversation content.
