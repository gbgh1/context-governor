# Phase 5 Spec — Hermes integration + proving ground

> Authoritative contract for Phase 5. Unlike Phases 1–4 (pure, fully offline-faked), this
> phase is **integration + measurement** against the real stack (llama-server + Hermes Agent
> + Qwen3.6). It therefore splits into (a) artifacts + code that ARE buildable/testable
> offline now, and (b) a runbook the user executes on the live machine. Items marked
> **[offline]** are done in-repo now; **[live]** items need the real stack.

## 0. Goal

Prove the compaction livelock is gone when the governor sits in front of the real model, and
quantify it. Success = with the proxy in place, Hermes' per-turn compaction no longer fires
fruitlessly, and context-pressure stabilizes over a long Spaceshooter build session.

## 1. Topology (the wiring) — [live]

```
Hermes Agent ──/v1/chat/completions──▶ [ contextmanager.proxy :8900 ] ──▶ llama-server :8080
   (model.base_url = http://127.0.0.1:8900)          │  CM_UPSTREAM_BASE_URL=http://127.0.0.1:8080
                                                      ▼
                                          ./contextstore (state.json · notes · index.db)
   (optional, cooperative)
Hermes MCP client ──stdio──▶ [ contextmanager.mcp ]  (same ./contextstore)
```

Run order: llama-server → `python -m contextmanager.proxy` (env `CM_UPSTREAM_BASE_URL`,
`CM_STORE_ROOT`, `CM_HANDLE_THRESHOLD_TOKENS`) → point Hermes `model.base_url` at the proxy.

## 2. Tier-0 Hermes config patch — [offline artifact + live apply]

Ship the patch from `tasks/plan.md` §Tier 0 as a concrete, copy-pasteable artifact at
`integration/hermes-config.tier0.yaml` with inline comments. It raises `compression.threshold`
(0.50→0.72), lowers `protect_last_n` (20→8) and `target_ratio` (0.20→0.12), sets
`model.context_length` to match `llama-server -c`, and keeps the summarizer ctx == main
(never point `auxiliary.compression.model` at a smaller-context model — silent middle-drop).
This bounds the loop on its own; the proxy removes the residual (one giant tool dump in the
protected tail). Both `compression.*` and `model.context_length` hot-reload.

## 3. Proxy observability — `/metrics` — [offline, NORMATIVE]

To measure "before/after" we need a proxy-side counter (the open decision in plan.md). Add a
zero-tokenizer-cost stats surface to the Phase 3 proxy (additive; does not change the rewriter
contract).

`src/contextmanager/proxy/metrics.py`:
```python
@dataclass
class ProxyStats:
    requests: int = 0
    messages_in: int = 0
    messages_handle_ized: int = 0
    messages_rehydrated: int = 0
    chars_in: int = 0          # sum of len(content) over string messages, pre-rewrite
    chars_out: int = 0         # ... post-rewrite (what actually goes on the wire)
    # chars_saved property = chars_in - chars_out (can be negative when rehydration
    # adds more than handle-ization removed — that is real signal, surface it)

class StatsCollector:
    def record(self, *, messages_in, messages_handle_ized, messages_rehydrated,
               chars_in, chars_out) -> None: ...   # thread-safe accumulate
    def snapshot(self) -> dict: ...                # JSON-able incl. chars_saved
```
`app.py`: build `app.state.stats = StatsCollector()` in `create_app` (so it exists in both
injected-test mode and lifespan mode); in the chat handler, AFTER `rewrite_outgoing`, call
`stats.record(...)` (chars via a `_sum_content_chars(messages)` helper over string contents);
add `GET /metrics` → `JSONResponse(app.state.stats.snapshot())` (does NOT touch upstream).
Counts are recorded for both stream and non-stream (record right after rewrite, before
forwarding, so it measures the prompt transform regardless of upstream outcome).

Tests (`tests/proxy/test_app_metrics.py`): POST a bulky message → `/metrics` shows
`requests==1`, `messages_handle_ized>=1`, `chars_out < chars_in` (wire shrank); `/metrics`
does not call upstream; a small-only request shows `messages_handle_ized==0` and
`chars_saved==0`.

## 4. Surface B skill / system-prompt protocol — [offline artifact]

The MCP tools only help if the agent calls them. Ship a protocol doc at
`integration/surface-b-protocol.md` (and a copyable system-prompt block) instructing the
agent to, each turn: `state_snapshot` the authoritative game/project state JSON; `store_save`
large artifacts it will not need verbatim soon; `context_rehydrate` (by query) what it needs
back. Keep the protocol short and mechanical (the model is low-context). This was deferred
from Phase 4 because it only bites with a live agent driving the tools.

## 5. Measurement procedure — [live runbook]

Document in `integration/measurement.md`:
1. Baseline: run the Spaceshooter build under Hermes with NO governor; record compaction
   events/turn (from Hermes logs) and turns-to-livelock.
2. Tier-0 only: apply §2 patch; re-run; record.
3. Tier-0 + proxy: add the proxy; re-run; record `/metrics` (`chars_saved`, handle-izations)
   alongside Hermes compaction counts.
4. Compare: expect compaction-per-turn → ~0 and a long stable session. Capture the numbers in
   a results table appended to `tasks/plan.md`.

## 6. Constraints
- §3 is the only new src this phase and is ADDITIVE to the proxy (new `metrics.py`; small
  `app.py` edits). Full type hints; `from __future__ import annotations`; py_compile clean.
- Offline tests only for §3 (ASGITransport + fakes). §1/§2/§4/§5 are artifacts/runbooks.
- Do NOT regress the §3.5 / §9 proxy invariants — keep the existing proxy tests green.
