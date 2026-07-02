---
title: The no-re-fire invariant
updated: 2026-06-23
tags: [engine, compaction, invariant]
---

# The no-re-fire invariant

The mathematical guarantee at the heart of the [[context-governor]] core engine: a compaction
can **never** trigger another compaction. This is what structurally kills the
[[hermes-compaction]] livelock instead of merely delaying it. Code:
`src/contextmanager/{compactor,budget,sealing}.py`.

## Hysteresis, not a single threshold

- **High-water trigger** vs **low-water target** (distinct levels). Compaction runs only when
  load crosses the high-water mark, and compacts down to the low-water mark — leaving a gap so
  the next turn's growth does not immediately re-trigger.
- **Floor <= low-water by construction.** The non-compactable tiers (pinned head + state
  snapshot + a *capped* distilled-memory budget) are guaranteed to sit below the low-water
  target. The summary tier is bounded by **measurement + truncation**, never by trusting the
  model's self-reported length.
- **Mechanical fallback.** If even the bounded summary cannot get under target, the engine
  drops the oldest compactable messages (paging them to the [[durable-store]]) — guaranteeing
  forward progress every time.

## The consequence

Post-compaction load is `< trigger` by construction ⇒ the very next measurement does not
re-fire. Contrast with Hermes, whose protected tail can itself exceed the threshold with **no
mechanism to re-compress** → fruitless per-turn compaction.

## How it's proven

53 unit tests in Phase 1, including a `hypothesis` property test, floor-under-stress,
water-collapse, and giant-tail page-out cases. Contract: `tasks/phase1-spec.md`. The same
discipline (deterministic, measured, truncation-bounded) reappears as the idempotency
invariant in [[surface-a-proxy]].
