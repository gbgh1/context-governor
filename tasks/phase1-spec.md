# Phase 1 Spec — Core Engine (the Context Governor)

> Authoritative contract for Phase 1. Both the implementer and the tester bind to
> THIS document. Field names, signatures, and the invariant are normative — do not
> rename or paraphrase. Language: **Python 3.11+**, `src/` layout, `uv` + `pyproject.toml`,
> tests with `pytest` (+ `hypothesis` for the randomized invariant tests).

## 0. What Phase 1 is (and is NOT)

Phase 1 builds the **shared core engine** only: exact token accounting, the tiered
budget math, the hysteresis compactor, summary sealing, and the mechanical fallback —
plus unit tests that **prove the no-re-fire invariant**.

NOT in Phase 1: the real durable store/vector index (Phase 2), the proxy (Phase 3), the
MCP server (Phase 4). Those are reached only through small **Protocols** (interfaces)
here, with trivial in-memory fakes for testing.

## 1. The problem this math kills

Hermes livelocks because its protected tail can exceed the compaction threshold and
there is **no mechanism to re-compress it** → compaction fires every turn, fruitlessly.
Our fix is structural: **no message is unconditionally protected from page-out.** Recent
big tool outputs are a *preference* to keep verbatim, not an absolute. When even the tail
exceeds budget, the mechanical fallback pages the biggest/oldest items to the store,
leaving a handle. Combined with a floor that fits under the low-water mark *by
construction*, post-compaction load is always `<= low_water < high_water` ⇒ it cannot
immediately re-fire. Forward progress is guaranteed.

## 2. Module layout

```
pyproject.toml
src/contextmanager/
  __init__.py
  types.py        # Message, ContextState, dataclasses, Protocols
  tokenizer.py    # LlamaServerTokenCounter (HTTP) + TokenCounter Protocol
  budget.py       # BudgetConfig + tier math (high/low water, floor)
  sealing.py      # seal_summary(): generate -> MEASURE -> TRUNCATE to cap
  compactor.py    # HysteresisCompactor: needs_compaction / compact / invariant
tests/
  conftest.py
  test_budget.py
  test_sealing.py
  test_compactor.py   # THE no-re-fire invariant suite
  test_tokenizer.py   # HTTP client against mocked responses
```

## 3. Exact types (`types.py`) — NORMATIVE

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Optional

@dataclass
class Message:
    role: str                 # "system" | "user" | "assistant" | "tool"
    content: str
    id: str                   # stable unique id
    pinned: bool = False      # part of the protected head; never paged out
    sealed: bool = False      # already represented inside distilled memory

@dataclass
class ContextState:
    """The reconstructable context for one turn, by tier."""
    head: list[Message]            # pinned spec: system prompt + protect_first_n
    state_snapshot: str            # authoritative state.json rendered to text ("" if none)
    distilled_memory: str          # sealed rolling summary ("" if none)
    window: list[Message]          # recent messages + retrieved slices (compactable)

class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...
    def count_messages(self, messages: list[Message]) -> int: ...
    # Truncate text to AT MOST max_tokens tokens (by real tokenization), returning text.
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str: ...

class Summarizer(Protocol):
    # Produce a handoff summary of `messages`, optionally folding in prior_summary.
    # The RETURNED length is NOT trusted; the caller measures+truncates. target_tokens
    # is only a hint passed to the model.
    def summarize(self, messages: list[Message], prior_summary: Optional[str],
                  target_tokens: int) -> str: ...

class Store(Protocol):
    # Page a message's full content out to durable storage; return a short handle string.
    def page_out(self, message: Message) -> str: ...
    def get(self, handle: str) -> str: ...
```

## 4. Budget math (`budget.py`) — NORMATIVE

```python
@dataclass(frozen=True)
class BudgetConfig:
    n_ctx: int                          # from /props default_generation_settings.n_ctx
    reserved_headroom_tokens: int       # H: reserved for generation + safety margin
    state_cap_tokens: int               # S_max: hard cap for state_snapshot
    distilled_cap_tokens: int           # D_max: hard cap for distilled_memory
    trigger_ratio: float = 0.75         # high-water, fraction of usable budget B
    target_ratio: float = 0.50          # low-water,  fraction of usable budget B
    protect_first_n: int = 3
    protect_last_n: int = 8

    def __post_init__(self):
        # validate: 0 < target_ratio < trigger_ratio < 1
        # validate: 0 < reserved_headroom_tokens < n_ctx
        # validate: state_cap_tokens >= 0, distilled_cap_tokens >= 0
        # raise ValueError with a clear message on violation
        ...

    @property
    def budget(self) -> int:        # B = usable prompt budget
        return self.n_ctx - self.reserved_headroom_tokens
    @property
    def high_water(self) -> int:    # T_hi
        return int(self.trigger_ratio * self.budget)
    @property
    def low_water(self) -> int:     # T_lo
        return int(self.target_ratio * self.budget)
```

Definitions used by the compactor:
- `P` = measured tokens of `head` (via counter.count_messages(head)).
- `S` = measured tokens of `state_snapshot` (counter.count_text); enforced `<= state_cap_tokens`.
- `D` = measured tokens of `distilled_memory`; enforced `<= distilled_cap_tokens`.
- `W` = measured tokens of `window`.
- `load = P + S + D + W` (current prompt size, measured).
- **Floor** `F = P + state_cap_tokens + distilled_cap_tokens`.

## 5. Sealing (`sealing.py`) — NORMATIVE

```python
def seal_summary(messages, prior_summary, counter, summarizer, cap_tokens) -> str:
    """
    1. text = summarizer.summarize(messages, prior_summary, target_tokens=cap_tokens)
    2. MEASURE: counter.count_text(text)
    3. If over cap_tokens -> counter.truncate_to_tokens(text, cap_tokens)
    4. Return text guaranteed counter.count_text(result) <= cap_tokens.
    NEVER trust the model's self-reported / requested length. Always measure+truncate.
    """
```

## 6. Compactor (`compactor.py`) — NORMATIVE

```python
@dataclass
class CompactionResult:
    fired: bool
    load_before: int
    load_after: int
    sealed_message_ids: list[str]
    paged_out_message_ids: list[str]
    summary: Optional[str]
    used_mechanical_fallback: bool

class HysteresisCompactor:
    def __init__(self, config: BudgetConfig, counter: TokenCounter,
                 summarizer: Summarizer, store: Store): ...

    def current_load(self, ctx: ContextState) -> int:
        """P + S + D + W, all measured via counter."""

    def needs_compaction(self, ctx: ContextState) -> bool:
        """current_load(ctx) >= config.high_water"""

    def assert_floor_fits(self, ctx: ContextState) -> None:
        """
        Compute F = count_messages(head) + state_cap_tokens + distilled_cap_tokens.
        If F > config.low_water: raise FloorExceedsTargetError (a ValueError subclass)
        with a message naming F and low_water. This is the BY-CONSTRUCTION guarantee
        that compaction can always reach <= low_water. Call it in __init__-time check
        AND at the start of compact().
        """

    def compact(self, ctx: ContextState) -> CompactionResult:
        """
        Precondition check: assert_floor_fits(ctx).
        If not needs_compaction(ctx): return CompactionResult(fired=False, ... no-op).

        Step A — SEAL the middle:
          - Identify sealable window messages: all EXCEPT the last `protect_last_n`
            and any pinned. (head is never in window.)
          - If any sealable messages: summary = seal_summary(sealable, ctx.distilled_memory,
            counter, summarizer, cap_tokens=config.distilled_cap_tokens).
            Set ctx.distilled_memory = summary; remove sealed msgs from window; mark ids.
          - Recompute load.

        Step B — MECHANICAL FALLBACK (the Hermes-killer):
          - While current_load(ctx) > config.low_water AND window is non-empty:
              pick the page-out victim = the OLDEST non-pinned window message
              (oldest first; this naturally includes a big tool output once it is the
              oldest, and we never treat the tail as untouchable). store.page_out(msg);
              remove it from window; record its id; set used_mechanical_fallback=True.
          - This loop strictly reduces W each iteration and W can reach 0, while
            F <= low_water by construction => the loop ALWAYS terminates with
            current_load(ctx) <= low_water.

        Postcondition (MUST hold, assert it):
            current_load(ctx) <= config.low_water  < config.high_water
          => needs_compaction(ctx) is False immediately after. No re-fire.

        Return CompactionResult with load_before/after and the recorded ids.
        """
```

Notes:
- `head` (pinned) messages are NEVER selected for page-out. The floor assertion guarantees
  head + caps fit, so this is safe.
- Step B victim selection is **oldest-first among non-pinned window messages**. A giant
  recent tool output will be paged out once older messages are exhausted; if a SINGLE
  message is so large that even alone it exceeds low_water, it still gets paged out
  (window empties), so load drops to `P + S + D <= F <= low_water`. Livelock impossible.

## 7. Tokenizer client (`tokenizer.py`) — NORMATIVE

`LlamaServerTokenCounter(TokenCounter)` using `httpx` against llama-server:
- `base_url`, optional `api_key`, `timeout`, injected `httpx.Client` allowed for testing.
- `count_text(text)`: `POST {base_url}/tokenize` body `{"content": text, "add_special": False}`
  → response `{"tokens": [int,...]}` → return `len(tokens)`.
- `count_messages(messages)`: PRIMARY: `POST {base_url}/v1/chat/completions/input_tokens`
  body `{"messages": [{"role","content"}...]}` → `{"input_tokens": int}` → return it.
  FALLBACK (if that route 404s): `POST /apply-template` {messages} → `{"prompt": str}`,
  then count_text(prompt) with add_special=True. Detect 404 and cache which path works.
- `truncate_to_tokens(text, max_tokens)`: `POST /tokenize` with `{"content": text,
  "with_pieces": true}` → reassemble text from the first `max_tokens` pieces (decode byte
  arrays as needed). If text already <= max_tokens tokens, return unchanged.
- `n_ctx()` helper: `GET {base_url}/props` → `default_generation_settings.n_ctx`.
- Raise a typed `TokenizerError` on transport/HTTP errors; do not swallow.

## 8. The invariant test suite (`test_compactor.py`) — what MUST be proven

Use a deterministic fake counter (token cost derived from content, e.g. len of
whitespace-split words, or an explicit per-id cost map), a fake summarizer (returns a
fixed long string so truncation is exercised), and an in-memory fake store.

1. **floor_fits_ok**: F <= low_water → constructor / assert_floor_fits passes.
2. **floor_exceeds_raises**: head + caps > low_water → FloorExceedsTargetError raised.
3. **post_compaction_below_low_water** (property-based, hypothesis): for randomized
   window sizes and per-message token costs, after compact(), load <= low_water.
4. **no_immediate_refire**: after compact(), needs_compaction(ctx) is False (always).
5. **giant_tail_message_paged_out** (THE livelock-killer): a single window message whose
   cost alone exceeds low_water is paged out; load_after <= low_water;
   used_mechanical_fallback is True; the message id is in paged_out_message_ids.
6. **summary_measured_and_truncated**: summarizer returns text far over distilled_cap;
   resulting distilled_memory measures <= distilled_cap_tokens.
7. **idempotent**: calling compact() again immediately returns fired=False (no-op).
8. **head_never_paged_out**: pinned/head messages never appear in paged_out_message_ids
   and remain present after compaction.
9. **forward_progress**: load_after < load_before whenever fired is True.

## 9. Constraints

- Dependencies: `httpx`, `pytest`, `hypothesis`. Keep it minimal; stdlib otherwise.
- Pure functions where possible; the compactor mutates the passed ContextState in place
  (documented) and also returns a CompactionResult.
- Type-hint everything; target clean `python -m py_compile` and (if configured) mypy.
- No network in tests — inject fakes/mocked httpx only.
```

## 10. Round-2 corrections (NORMATIVE — supersede §4–§8 where they conflict)

A code review found the v1 implementation's invariant was only *conditionally* true. The
following corrections make the no-re-fire guarantee universal and are mandatory.

### 10.1 New typed exceptions (compactor.py; export from `__init__.py`)
- `class ContractError(ValueError)` — caller violated an input contract.
- `class InvariantViolationError(RuntimeError)` — an internal guarantee failed. Used INSTEAD
  of bare `assert` for the postcondition so it survives `python -O`.

### 10.2 budget.py — enforce integer water separation (fixes C2)
In `__post_init__`, AFTER the ratio checks, compute the integer waters and require strict
separation; raise `ValueError` if they collapse:
```python
if not (self.low_water < self.high_water):
    raise ValueError(
        f"integer waters collapsed: low_water={self.low_water} not < "
        f"high_water={self.high_water} (budget={self.budget}); "
        f"increase n_ctx/budget or widen the ratio gap."
    )
```
This guarantees `low_water < high_water` for every constructible config, so a post-compaction
load `<= low_water` is strictly `< high_water` ⇒ `needs_compaction` is False ⇒ no re-fire.

### 10.3 compactor.py — make the floor SOUND (fixes C1, H2, M4)
At the START of `compact()` (after the no-op check, before Step A):

1. **Reject pinned window messages (H2):** if any `m.pinned` for `m in ctx.window`, raise
   `ContractError` naming the offending id(s). Pinned content must live in `ctx.head`.
2. **Enforce the caps defensively (C1):** measure and, if over, truncate BOTH tiers to their
   caps so the floor formula is a true upper bound on `P + S + D`:
   ```python
   ctx.state_snapshot   = _enforce_cap(ctx.state_snapshot,   self.config.state_cap_tokens,   self.counter)
   ctx.distilled_memory = _enforce_cap(ctx.distilled_memory, self.config.distilled_cap_tokens, self.counter)
   ```
   where `_enforce_cap(text, cap, counter)` is a module helper that:
   measures; if `> cap`, calls `counter.truncate_to_tokens(text, cap)`, then RE-MEASURES and
   repeats with a shrinking target (`cap`, then `cap-1`, …) until measured `<= cap` or text is
   empty; returns the capped text. This tolerates real-tokenizer truncation overshoot.
3. **Postcondition without assert (M4):** replace the bare `assert load_after <= low_water`
   with:
   ```python
   if load_after > self.config.low_water:
       raise InvariantViolationError(
           f"post-compaction load_after={load_after} > low_water={self.config.low_water}")
   ```

Note: `assert_floor_fits` still uses caps for `F` — that is now correct *because* the caps are
enforced on S and D at compact() entry.

### 10.4 sealing.py — guarantee the cap by re-measuring (fixes H1)
`seal_summary` must use the same `_enforce_cap` semantics (or an inline equivalent): after
truncation, RE-MEASURE and shrink until `count_text(result) <= cap_tokens`. Do not return a
truncated string without confirming it measures within cap.

### 10.5 tokenizer.py — robustness (fixes H3, M1, M2)
- **H3:** wrap every `resp.json()` in try/except → raise `TokenizerError(f"malformed JSON from {path}: ...")`.
- **M1:** do NOT use a numeric sentinel for the 404 fallback. Detect 404 via a private
  `_RouteNotFound` exception raised inside the primary path and caught in `count_messages`.
  Only set the cached path to `"fallback"` AFTER the fallback call returns successfully.
- **M2:** in `truncate_to_tokens` piece reassembly, validate each byte element is an `int`
  before `bytes(...)`; on a malformed piece raise `TokenizerError`.

### 10.6 pyproject.toml — test path (fixes L1)
Add `pythonpath = ["src"]` to `[tool.pytest.ini_options]` so a clean `pytest` run resolves
`contextmanager` and `conftest` without an editable install.

### 10.7 Additional REQUIRED tests (fixes C3) — add to `tests/`
- `test_oversized_state_snapshot_enforced`: build ctx with `state_snapshot` whose measured
  tokens >> `state_cap_tokens` and a tiny window; after `compact()`, the snapshot measures
  `<= state_cap_tokens` AND `load_after <= low_water` (no crash, no InvariantViolationError).
- `test_stale_oversized_distilled_no_sealable`: `distilled_memory` >> `distilled_cap_tokens`,
  window contains only protected-tail messages (nothing sealable); after `compact()`,
  distilled_memory measures `<= distilled_cap_tokens` and `load_after <= low_water`.
- `test_water_collapse_rejected`: `BudgetConfig(n_ctx=110, reserved_headroom_tokens=100,
  state_cap_tokens=1, distilled_cap_tokens=1, target_ratio=0.10, trigger_ratio=0.11)` (budget=10
  ⇒ low=1, high=1) raises `ValueError`.
- `test_pinned_in_window_rejected`: a `pinned=True` message placed in `window` ⇒ `compact()`
  raises `ContractError`.
- `test_seal_truncation_reconfirmed`: a fake counter whose `truncate_to_tokens` deliberately
  OVERSHOOTS (returns slightly more than `max_tokens`) ⇒ `seal_summary` still returns text
  measuring `<= cap` (proving the re-measure loop), OR raises if it cannot converge.
- `test_tokenizer_malformed_json_raises`: a mocked 200 response with non-JSON body ⇒
  `TokenizerError`.
