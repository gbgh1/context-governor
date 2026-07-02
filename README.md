# Context Governor

**Keep a local-LLM agent's context approximately constant over an arbitrarily long
session** — instead of recursively summarizing an ever-growing transcript until it
collapses. Drop-in for any OpenAI-compatible CLI (Hermes Agent, OpenCode, …) in front of
`llama-server`.

> Python package: `contextmanager`. License: MIT.

## The problem it solves

Long agentic sessions on a small local model fall into a **compaction livelock**: the
CLI's protected message tail (full of big tool outputs and file dumps) grows past the
compaction threshold, there's no mechanism to re-compress it, so compaction fires
fruitlessly *every turn* and the session grinds to a halt. Raising the context window only
delays it.

The Context Governor fixes this structurally by keeping bulky, stable content **off the
wire** and in a durable store, reconstructing a bounded prompt each turn.

## How it works — two surfaces over one engine

- **Core engine** — tiered token budget, hysteresis compaction with a *no-re-fire
  invariant* (a compaction can never trigger another), bounded summaries, exact token
  counts from `llama-server` (`/tokenize`, `/props`), and a durable store (authoritative
  `state.json` + human-auditable markdown notes + an FTS5/lexical retriever; a vector
  backend is pluggable behind the `Retriever` interface). The store is **lossless end to
  end**: content is persisted *before* it ever leaves the window, cold notes are gzipped
  (still searchable), and evicted notes are archived — never deleted — so a later request
  for any handle transparently *resurrects* it, byte-exact, even across restarts.
- **Surface A — endpoint proxy** (universal, no cooperation needed): a transparent
  OpenAI-compatible reverse proxy between *any* CLI and `llama-server`. It replaces bulky
  messages with short, parseable stubs (full content stored), so the CLI's
  API-reported prompt stays small and its native compaction rarely fires. Idempotent and
  prefix-stable, so KV-cache reuse is preserved. And it reads as well as it writes:
  **auto-recall** derives a query from the live conversation tail each turn and injects the
  most relevant off-wire memory back as one small, budgeted block — the model gets its
  memory back without ever knowing to ask.
- **Surface B — MCP server** (cooperative, precise): six MCP tools
  (`store_save`/`store_search`, `state_snapshot`/`state_load`,
  `context_checkpoint`/`context_rehydrate`) so an MCP-capable agent can *deliberately*
  externalize and recall state. Shares the same store as the proxy.

Both surfaces share one on-disk store, so a handle minted by either resolves in both.

## Proven result

First live run (Hermes Agent + a 35B-A3B local model, a 138-message / 73-tool-output
build session): the proxy cut the wire prompt by **~60%** (7.03M → 2.82M characters,
405 messages handle-ized over 72 requests), every chat completion returned 200, and the
long tool-heavy session completed with no livelock.

## Install

```bash
git clone https://github.com/gbgh1/context-governor.git
cd context-governor
python -m venv .venv
# Linux/macOS:           source .venv/bin/activate
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q          # 224 tests, no network / llama-server needed
```

## Quickstart

See it work end-to-end with **no model required** (uses the offline heuristic counter):

```bash
python examples/demo.py
```

Run the surfaces (point `CM_UPSTREAM_BASE_URL` at your `llama-server`):

```bash
# Surface A — OpenAI-compatible proxy (listens on :8900, forwards to llama-server)
CM_UPSTREAM_BASE_URL=http://127.0.0.1:8080 python -m contextmanager.proxy

# Surface B — MCP stdio server (usually spawned by your CLI; --store-root shares the proxy's store)
python -m contextmanager.mcp --store-root ./contextstore
```

Then set your CLI's model base URL to `http://127.0.0.1:8900/v1`. The proxy exposes
`GET /metrics` (cumulative `chars_saved`, handle-izations) so you can watch it work.

## Wire it to your agent CLI

[`integration/`](integration/README.md) has copy-paste setup for **Hermes Agent** and
**OpenCode** — provider entries, MCP registration, an optional system-prompt directive that
makes the agent use the cooperative tools, an immediate-relief Hermes config patch, and a
before/after measurement runbook.

## Configuration

Both entrypoints read `CM_*` env vars (and the MCP server also accepts `--store-root` /
`--upstream-base-url` / `--transport`). Key knobs: `CM_UPSTREAM_BASE_URL`, `CM_STORE_ROOT`,
`CM_LISTEN_PORT`, `CM_MODEL_ALIAS` (name the proxy advertises in `/v1/models`; default
`context-governor`, `""` to pass through), and `CM_DIFF_MIN_SIMILARITY` (lossless
delta-compression of near-duplicate content; `0` disables). See
`src/contextmanager/proxy/config.py` and `.../mcp/config.py`.

**llama-server is the source of truth for context size.** At startup the proxy reads the
real `n_ctx` from `/props` and:
- **(a) anchors** the per-message handle-ization threshold to it — `CM_HANDLE_THRESHOLD_RATIO`
  (default `0.02` = 2% of the true window), so the governor self-tunes to whatever `-c` you
  launch (fixed `CM_HANDLE_THRESHOLD_TOKENS` is the fallback when the server is unreachable);
- **(b) bounds the *total* wire** to `CM_CONTEXT_BUDGET_RATIO` (default `0.50` = 50% of the
  window) via **lossless budget-windowing** — paging out the oldest non-pinned middle messages
  to retrievable stubs (pinned head + recent tail kept verbatim). This pre-empts the CLI's own
  *lossy* compaction so it rarely needs to fire;
- **(c) propagates** the true `n_ctx` into `/v1/models`, so CLIs read the real window instead
  of guessing.

## Design docs

- [tasks/plan.md](tasks/plan.md) — architecture, milestones, and the root-cause analysis.
- [wiki/index.md](wiki/index.md) — interlinked concept notes (the surfaces, the store, the
  no-re-fire invariant, the Hermes compaction mechanics).
- `tasks/phase*-spec.md` — the normative module contracts each phase was built against.

## License

[MIT](LICENSE).
