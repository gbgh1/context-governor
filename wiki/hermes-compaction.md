---
title: Hermes Agent — Context Compaction Mechanics
updated: 2026-06-22
tags: [hermes, compaction, context, reference]
---

# Hermes Agent — Context Compaction

How Nous Research's [Hermes Agent](https://github.com/NousResearch/hermes-agent) manages
context, and the design gap that causes the **compaction livelock** we are solving in
[[index]].

## Trigger

- Fires when `prompt_tokens >= threshold * context_length`.
- Default `threshold: 0.50` (50% of window).
- Measured from **actual API-reported token counts** in the tool loop, not estimates.

## Preservation zones (per compaction)

- **Head** — `protect_first_n: 3` (+ system prompt), pinned across all compactions.
- **Tail** — `max(protect_last_n msgs, threshold_tokens * target_ratio)`,
  *"whichever protects more"*. Defaults: `protect_last_n: 20`, `target_ratio: 0.20`.
  Boundary aligns so tool-call/result pairs are never split.
- **Middle** — summarized by the auxiliary LLM into a "handoff summary".
  Prior summaries are updated incrementally, not regenerated.

## Summary budget

- `content_tokens * 0.20` (`_SUMMARY_RATIO`), min 2,000,
  max `min(context_length * 0.05, 12,000)` tokens.

## The gap that causes the loop

> "If the tail itself exceeds the threshold after compression, there is **no mechanism
> to re-compress** — this is a potential issue not addressed."

Big tool outputs / file dumps land in the **protected tail**. Once that tail alone
exceeds `threshold * context_length`, compaction cannot reduce it (tail is protected;
middle already summarized) → it re-fires every turn = fruitless livelock. Raising the
window just relocates the 50% line.

## Silent-loss trap

- The summary (auxiliary) model **must** have context >= the main model's.
- If smaller, `_generate_summary()` logs a warning and returns `None`, and the middle
  is **dropped with no summary** (silent context loss). It does **not** auto-lower.

## Knobs (`config.yaml`, hot-reloadable)

- `compression.{enabled,threshold,target_ratio,protect_last_n,protect_first_n,hygiene_hard_message_limit}`
- `auxiliary.compression.{model,provider,base_url,api_key,timeout,fallback_chain}`
- `model.{context_length,base_url,api_key}` — `base_url` lets us insert a proxy.

## Sources

- Developer guide: <https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching/>
- Configuration: <https://hermes-agent.nousresearch.com/docs/user-guide/configuration/>
- Repo: <https://github.com/NousResearch/hermes-agent>
