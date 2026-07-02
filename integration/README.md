# Integration — wiring the Context Governor to Hermes + llama-server

Copy-paste setup for putting the Context Governor in front of a running `llama-server`.
Paths below use `<CG>` for wherever you cloned this repo, and `<PY>` for its venv Python
(`<CG>/.venv/Scripts/python.exe` on Windows, `<CG>/.venv/bin/python` on Linux/macOS).

## Topology

```
Hermes Agent ──/v1/chat/completions──▶ [ contextmanager.proxy :8900 ] ──▶ llama-server :8080
   model.base_url = http://127.0.0.1:8900        CM_UPSTREAM_BASE_URL = http://127.0.0.1:8080
                                                          │
                                                          ▼
                                              ./contextstore  (state.json · notes/ · index.db)

   (optional, cooperative — same store)
Hermes MCP client ──stdio──▶ [ contextmanager.mcp ]
```

## Step-by-step

1. **Tier-0 relief first (no code).** Apply [`hermes-config.tier0.yaml`](hermes-config.tier0.yaml)
   to your active Hermes `config.yaml`. `compression.*` and `model.context_length` hot-reload.
   Verify the context-pressure bar stabilizes (no per-turn compaction). This alone may suffice
   for light workloads; continue for the structural fix.

2. **Start llama-server** as usual (note the `-c` context size; it must equal
   `model.context_length` in the Hermes config).

3. **Start the proxy (Surface A)** in front of llama-server:
   ```bash
   CM_UPSTREAM_BASE_URL=http://127.0.0.1:8080 \
   CM_STORE_ROOT=./contextstore \
   CM_HANDLE_THRESHOLD_TOKENS=2000 \
   CM_LISTEN_PORT=8900 \
   python -m contextmanager.proxy
   ```
   The proxy needs llama-server reachable for exact tokenization (`/tokenize`, `/props`).

4. **Point Hermes at the proxy:** set `model.base_url: http://127.0.0.1:8900`. Hermes now
   measures the *shrunken* wire prompt, so its native compaction rarely fires.

5. **(Optional) Surface B — the cooperative MCP server.** Registers the six governor tools
   (`store_save`, `store_search`, `state_snapshot`, `state_load`, `context_checkpoint`,
   `context_rehydrate`) so the agent can *deliberately* externalize/recall. It shares the SAME
   store as the proxy via `--store-root` (so a handle from either surface resolves in both).

   **Registration (CLI's spawn it over stdio — already wired into the live configs):**

   OpenCode `config.json` → `mcp` block (use absolute paths for `<PY>` and the store):
   ```json
   "context-governor": {
     "type": "local",
     "command": ["<PY>", "-m", "contextmanager.mcp", "--store-root", "<CG>/contextstore"],
     "enabled": true
   }
   ```
   Hermes `config.yaml` → `mcp_servers` block:
   ```yaml
   context-governor:
     command: <PY>
     args: [-m, contextmanager.mcp, --store-root, <CG>/contextstore]
   ```

   **IMPORTANT — registering only makes the tools *available*. The agent will not *use* them
   unless its system prompt says to.** Add [`surface-b-systemprompt.md`](surface-b-systemprompt.md)
   to the agent's instructions (OpenCode: append its path to the top-level `instructions` array;
   Hermes: include it via your agent/context prompt). Without it, expect `messages_rehydrated`
   to stay 0 — the proxy still works, but no deliberate recall happens.

## Measuring it worked

`GET http://127.0.0.1:8900/metrics` → cumulative `{requests, messages_handle_ized,
messages_rehydrated, chars_in, chars_out, chars_saved}`. `chars_saved` rising over a session =
the proxy is keeping bulk off the wire. Full before/after procedure:
[`measurement.md`](measurement.md).
