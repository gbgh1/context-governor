---
title: The Context Governor — architecture
updated: 2026-06-23
tags: [architecture, design, moc]
---

# The Context Governor

A universal layer that keeps a low-context local model's *retained* context approximately
constant over an arbitrarily long agentic session, instead of recursively summarizing an
ever-growing transcript. Born to kill the [[hermes-compaction]] livelock.

## One core engine, two surfaces, one store

```
CLI (Hermes) ──/v1/chat/completions──▶ [ Surface A: proxy ] ──▶ llama-server (Qwen3.6)
                                              │  ▲                 /tokenize · /props
                                  externalize │  │ retrieve
                                              ▼  │
                                      [ DurableStore ]  ← [[durable-store]]
                                              ▲
       any MCP CLI ──MCP tools──▶ [ Surface B: mcp ] ──┘
```

- **Core engine** — exact token accounting, tiered budget, hysteresis compaction with the
  [[no-re-fire-invariant]] (sealed, measured, truncation-bounded summaries; mechanical
  drop-oldest fallback). `src/contextmanager/{tokenizer,budget,compactor,sealing}.py`.
- **[[surface-a-proxy]]** — transparent OpenAI-compatible reverse proxy. Universal: needs no
  CLI cooperation. Shrinks the wire prompt so Hermes' own compaction rarely fires.
- **[[surface-b-mcp]]** — cooperative MCP server. Gives an agent explicit tools to externalize
  and recall. Precise, but cannot override a host's native compaction (that's the proxy's job).
- **[[durable-store]]** — `state.json` + human-auditable `notes/*.md` + a lexical retriever.
  Both surfaces share ONE store; a handle minted by either resolves in both.

## Key insight

MCP alone *cannot* stop a host CLI from compacting — only the proxy, which sits on the wire
and controls the API-reported prompt size, can. So the proxy is the primary lever; the MCP
server is the cooperative complement.

## Status

Both surfaces implemented and tested; validated live in front of Hermes Agent (~60% wire
reduction on a long tool-heavy session). See [tasks/plan.md](../tasks/plan.md) for the full
milestone history and design rationale.
