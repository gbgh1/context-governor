---
title: Surface A — the endpoint proxy
updated: 2026-07-01
tags: [proxy, surface-a, design]
---

# Surface A — the endpoint proxy

A transparent OpenAI-compatible reverse proxy between any CLI and llama-server. Part of the
[[context-governor]]. Code: `src/contextmanager/proxy/`.

## What it does

Set the CLI's `model.base_url` to the proxy. For each `/v1/chat/completions`, the proxy
rewrites the `messages` so bulky, stable content leaves the wire — replaced by a short,
parseable **stub** — while the full content is saved to the [[durable-store]]. Because the
CLI measures the *API-reported* prompt tokens, a smaller wire means lower pressure, so the
host's native [[hermes-compaction]] rarely fires. The livelock is avoided structurally,
without any CLI cooperation.

## Handle-ization (the rewriter)

`PromptRewriter.rewrite_outgoing` is a **pure, deterministic** transform (no network):

- A message whose content tokenizes `>= handle_threshold_tokens` is paged out and replaced by
  `[[cm:stored handle=<h> role=<r> tokens=<n>]] …preview… [[/cm:stored]]`.
- The note id is `stable_id(role, content) = "msg-"+sha1(role\x00content)[:16]` — so the SAME
  message always maps to the SAME handle. This is what gives the [[no-re-fire-invariant]]'s
  cousin here: **prefix stability / idempotency** (KV-cache safe).
- **Auto-rehydration:** an explicit `[[cm:stored handle=H]]` reference in non-stub content is
  expanded into an appended `[[cm:rehydrated handle=H]]` user-role message, under
  `rehydrate_budget_tokens`. Stubs themselves do NOT auto-expand (keeps the common case bounded).
  (Role "user", not "system" — learned live: strict chat templates like Qwen's raise
  "System message must be at the beginning" for any mid-conversation system message.)
- **Diff-encoding (lossless delta compression):** when a bulky message is a near-duplicate of a
  recent same-role stored note (`difflib` ratio ≥ `diff_min_similarity`, scanning the last
  `diff_lookback` stubs) and a unified diff comes out smaller than a normal stub, it's replaced
  by `[[cm:diff handle=<new> base=<base> …]] <unified-diff> [[/cm:diff]]`. The full content is
  still paged out under `<new>` (fully recoverable), so it's lossless — and the diff *shows the
  model exactly what changed*, so it needn't rehydrate. Genuinely free: it compresses real
  redundancy (file re-reads, repeated state dumps), depends only on earlier messages (so the
  prefix-stability invariant holds under append), and is idempotent (a diff-stub re-enters as a
  no-op). For a 100-line file with one line changed: ~91% smaller than the re-sent content.
  (Similarity is measured with `autojunk=False` — the difflib default marks "popular" characters
  as junk on strings ≥ 200 chars, collapsing a true ~0.999 re-read similarity to ~0.51 and
  silently disabling the delta path. Measured and fixed in Phase 10.)
- **Auto-recall (Pass 4 — anticipatory demand paging, Phase 10):** the live run proved agents
  never ask for their memory back (`messages_rehydrated: 0`) — a model cannot page-fault on
  content it cannot see. So each request, the proxy derives an implicit query from the live
  tail (`proxy/recall.py`: recency-weighted salient terms, stopwords dropped, marker lines
  stripped), searches the [[durable-store]], filters to **off-wire** handles only (never
  duplicates a stub, a diff base, a rehydrated marker, or verbatim wire content), suppresses
  near-duplicate slices, and injects the top `auto_recall_k` under `recall_budget_tokens` as ONE
  `[[cm:recall]]` user-role message before the final message (template-safe: mid-wire "system"
  is rejected by strict templates). The block is **stripped on entry and
  recomputed fresh** every call — at most one exists, so `rewrite(rewrite(x))` cannot grow — and
  `[[cm:recalled` deliberately does not match the stored-reference regex, so Pass 2 never
  re-expands it. Recall flows through `store.search()`, so retrieval metrics AND hotness
  warming come free: recall feeds the working-set signal that drives eviction and the archive.

## The idempotency invariant (critical)

`rewrite_outgoing(rewrite_outgoing(M)) == rewrite_outgoing(M)` for ALL M, including M with
explicit references. A Round-2 review caught that rehydrated messages were re-expanded every
turn — *recreating the very livelock the proxy exists to kill*. Fixed by `is_rehydrated`
passthrough (Pass 1) + handle dedup (Pass 2). Pinned by `test_idempotency_with_explicit_reference`
and `test_multiturn_no_growth`. **If you touch the rewriter, keep these green.**

## App + observability

`create_app` (FastAPI): `POST /v1/chat/completions` (stream + non-stream SSE passthrough),
`/v1/models`, `/props`, `/healthz`, and `/metrics` (Phase 5 — cumulative `chars_saved`,
handle-izations; zero tokenizer cost). `UpstreamError` → 502 on both branches (the stream
branch primes the first chunk before committing headers). Run: `python -m contextmanager.proxy`.

Contract: `tasks/phase3-spec.md` (§9 = the Round-2 corrections).
