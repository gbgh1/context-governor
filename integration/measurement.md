# Measurement runbook — proving the loop is gone

Goal: quantify compaction behavior **before vs after** the governor on the Spaceshooter build
(the original failing workload). Run the same scripted build three ways and compare.

## What to record per run

| Metric | Source |
|--------|--------|
| compaction events / turn | Hermes logs (count compaction fires per assistant turn) |
| turns until livelock (or "none") | Hermes logs / observation |
| final context-pressure | Hermes pressure bar at session end |
| `chars_saved`, `messages_handle_ized` | proxy `GET /metrics` (runs C only) |
| session completed? (built the demo?) | observation |

## The three runs

**A — Baseline (no governor).** Stock Hermes `config.yaml` (threshold 0.50, protect_last_n 20),
no proxy. Run the Spaceshooter build. Expect: compaction fires repeatedly, livelock after N
turns. Record N and compaction/turn.

**B — Tier-0 only.** Apply [`hermes-config.tier0.yaml`](hermes-config.tier0.yaml) (hot-reload).
No proxy. Re-run the SAME build. Expect: the loop is bounded but a single large tool dump in
the protected tail can still trigger it. Record.

**C — Tier-0 + proxy.** Keep the Tier-0 config; start the proxy in front of llama-server and
point Hermes at it (see [README.md](README.md)). Re-run. Poll `GET /metrics` periodically.
Expect: compaction/turn → ~0, the session runs long and stable, `chars_saved` grows.

## Results

Aggregate metrics only (no conversation content) so the numbers can live in the repo safely.

### Actual runs

| # | date | config | requests | msgs handle-ized | chars in → out | chars_saved | ctx peak | completed | livelock |
|---|------|--------|----------|------------------|----------------|-------------|----------|-----------|----------|
| 1 | 2026-06-23 | proxy only (Hermes compaction left at default 0.5; no Tier-0) | 72 | 405 | 7,030,113 → 2,818,931 | 4,211,182 (~60%) | ~40K / 76K | yes (138-msg, 73-tool session) | none observed |

Run 1 notes: proxy worked end-to-end — all 66 chat completions HTTP 200, session finished a long
tool-heavy build with no lockup. `messages_rehydrated: 0` (agent had no recall directive yet →
motivated the Surface-B system prompt). **Compaction-events/turn NOT captured** in run 1 (only the
proxy + llama-server logs were kept; capture Hermes' own compaction log next time). Source:
`tests/live_tests/` (gitignored).

### Planned A/B/C comparison (pending)

| Run | compaction/turn | turns-to-livelock | completed? | chars_saved |
|-----|-----------------|-------------------|------------|-------------|
| A baseline (no governor) | _pending_ | _pending_ | _pending_ | n/a |
| B Tier-0 only            | _pending_ | _pending_ | _pending_ | n/a |
| C Tier-0 + proxy         | _pending_ | _pending_ | _pending_ | _pending_ |

## Notes

- Keep the build script / prompt identical across runs so the comparison is fair.
- `CM_HANDLE_THRESHOLD_TOKENS` is the main knob: lower = more aggressive offload (more
  `chars_saved`, smaller wire) at the cost of more rehydration round-trips. Start at 2000.
- If run C still compacts, lower the threshold and/or re-check that Hermes is actually pointed
  at the proxy (`/metrics` `requests` should climb as the agent talks).
