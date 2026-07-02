# Context Governor — durable memory (read me every turn)

You have a durable memory via the `context-governor` MCP tools. Your visible context is
small and old bulk may be replaced by short stubs, so externalize aggressively and recall
on demand instead of asking the user to re-paste.

- **Snapshot state.** After any change to the authoritative world/project state (game state,
  file map, task list, key decisions), call `state_snapshot` with the FULL state as a JSON
  object (use `merge=true` for partial updates). This survives compaction; treat it as truth.
- **Offload bulk.** Before keeping a large artifact (a file you read, a long log, generated
  code you won't edit right now), call `store_save(content)` and keep only the returned
  `handle` + a one-line summary in context. Refer back by handle.
- **Checkpoint milestones.** At the end of a step, call
  `context_checkpoint(label, content, state)` so the step + state can be reloaded exactly.
- **Recall on demand.** When you need detail you offloaded, call
  `context_rehydrate(query="…")` (or `handle="…"`) with a `budget_tokens` you can afford —
  do not ask the user to re-send it.

Rule of thumb: if something is more than a few lines and you might need it later but not
right now, store it and carry the handle, not the text.
