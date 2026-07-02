---
title: Surface B — the cooperative MCP server
updated: 2026-06-23
tags: [mcp, surface-b, design]
---

# Surface B — the cooperative MCP server

A stdio MCP server that exposes the [[context-governor]] engine + [[durable-store]] as explicit
tools, so an MCP-capable agent can *deliberately* externalize and recall durable state. Code:
`src/contextmanager/mcp/`. Cooperative complement to [[surface-a-proxy]] — it does NOT intercept
the host's prompt and so cannot stop a host's native compaction; it makes the agent's own
context management explicit and durable.

## The six tools (underscore names, portable across clients)

| Tool | Backed by | Returns |
|------|-----------|---------|
| `store_save(content, role?)` | `DurableStore.page_out` | `{handle, id, role, tokens}` |
| `store_search(query, k?)` | `DurableStore.search` | matches: `{handle, score, tokens, preview}` |
| `state_snapshot(state, merge?)` | `StateStore.update`/`save` | `{ok, merge, keys, tokens}` |
| `state_load()` | `StateStore.load`/`render` | `{state, rendered, empty}` |
| `context_checkpoint(label, content, state?)` | label-stable note + state | `{handle, label, id, …}` |
| `context_rehydrate(query?/handle?, budget?)` | `DurableStore.page_in` / `get` | `{found, handles, tokens, text}` |

Checkpoints are **label-stable** (re-checkpointing a label overwrites, never duplicates).
`context_rehydrate` budgets the retrieved **content** tokens (the `[[cm:slice …]]` markers are
uncounted scaffolding).

## Design

- All logic is in a pure `GovernorService` (no MCP types) → tested without an MCP runtime.
  `server.py` is a thin `FastMCP` adapter (typed tool wrappers → service → JSON dict).
- Offline by default: a `HeuristicTokenCounter` (~1 tok/4 chars, measurable) lets the budgeted
  page-in work with **no llama-server**. Set `CM_UPSTREAM_BASE_URL` to use exact
  `LlamaServerTokenCounter` counts instead.
- Tests live in `tests/mcp_server/` — a test package literally named `mcp` would shadow the
  installed `mcp` SDK under pytest's prepend import.

Driven by the system-prompt protocol in `integration/surface-b-protocol.md`. Run:
`python -m contextmanager.mcp` (stdio). Contract: `tasks/phase4-spec.md`.
