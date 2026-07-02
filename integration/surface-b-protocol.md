# Surface B protocol — driving the MCP tools from a low-context agent

The MCP server (`python -m contextmanager.mcp`) only helps if the agent **calls** its tools.
This is the cooperative protocol: a short, mechanical instruction block to paste into the
agent's system prompt. Keep it terse — the local model is low-context.

## The six tools (underscore names; portable across MCP clients)

| Tool | Use it to… |
|------|-----------|
| `store_save(content, role?)` | persist a large artifact (file dump, log) → returns a `handle` |
| `store_search(query, k?)` | find previously stored notes → handles + previews |
| `state_snapshot(state, merge?)` | write authoritative world/project state to `state.json` |
| `state_load()` | read the current authoritative state |
| `context_checkpoint(label, content, state?)` | save a labeled checkpoint (overwrites same label) |
| `context_rehydrate(query?/handle?, budget_tokens?)` | page durable content back into context |

## System-prompt block (copy verbatim)

```
You have a durable memory via MCP tools. Your visible context is small, so externalize
aggressively and recall on demand. Each turn:

1. SNAPSHOT STATE. After any change to the world/project state (game state, file map,
   task list), call state_snapshot with the FULL current state as a JSON object. This is
   the single source of truth; it survives compaction. Use merge=true for partial updates.

2. OFFLOAD BULK. Before pasting a large artifact (a file you read, a long log, generated
   code you won't edit immediately) into your reasoning, call store_save(content) and keep
   only the returned handle + a one-line summary in context. Refer back by handle.

3. CHECKPOINT MILESTONES. At the end of a meaningful step, call
   context_checkpoint(label="<step>", content="<what you did + next step>", state=<state>).

4. RECALL ON DEMAND. When you need detail you offloaded, call context_rehydrate(query="...")
   (or handle="...") with a budget_tokens you can afford. Do NOT ask the user to re-paste.

Rule of thumb: if something is longer than a few lines and you might need it later but not
right now, store it and carry the handle — not the text.
```

## Why this is safe with the proxy

Surface A (the proxy) already keeps bulk off the wire automatically and without cooperation.
Surface B makes the agent's externalization *deliberate and structured* (e.g. authoritative
`state.json` it can reload exactly), which the proxy's opportunistic handle-ization cannot do.
They share one `./contextstore`, so a `handle` minted by either surface resolves in both.
