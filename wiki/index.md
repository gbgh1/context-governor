---
title: Index
updated: 2026-06-23
---

# ContextManager - Knowledge Base

Entry point. Browse the graph from here.

## Map of content

**The problem**
- [[hermes-compaction]] — how Hermes Agent compacts context, and the loop it causes.

**The solution — [[context-governor]]** (architecture overview)
- [[no-re-fire-invariant]] — the core-engine guarantee that a compaction can't trigger another.
- [[surface-a-proxy]] — the universal endpoint proxy (handle-ization + idempotency).
- [[surface-b-mcp]] — the cooperative MCP server (six tools).
- [[durable-store]] — `state.json` + markdown notes + lexical retriever.

**Operational docs**
- [[configuration]] — every parameter, where to set it, and which way is safe to turn it.
- Plan + milestones: [tasks/plan.md](../tasks/plan.md).
- Wiring to a CLI: [integration/README.md](../integration/README.md).
