# Phase 3 Spec — Surface A: the Endpoint Proxy (universal)

> Authoritative contract for Phase 3. Implementers and the tester bind to THIS document.
> Names, signatures, request/response shapes, and the invariants are normative.
> Python 3.11+, `src/` layout. NEW runtime deps allowed THIS phase: `fastapi`, `uvicorn`
> (server). `httpx` already present (upstream client, incl. streaming). Dev: `pytest`,
> `httpx` (ASGITransport for in-process testing — NO real network, NO real llama-server).

## 0. What Phase 3 is

A transparent **OpenAI-compatible reverse proxy** between any CLI (Hermes) and llama-server.
It applies the Phase 1/2 machinery to keep Hermes' *API-reported* prompt tokens low — by
replacing bulky, stable messages with short stubs + handles (full content stored in the
Phase 2 `DurableStore`) — so Hermes' native compaction rarely fires. Because the wire prompt
shrinks, the livelock is structurally avoided WITHOUT any CLI cooperation.

Decisions (locked):
- **Handle-ize only bulky messages** (token-threshold), small messages pass through verbatim.
- **Auto-rehydrate on handle reference**: if a later request references a stored handle, the
  proxy pages that content back into THAT request before forwarding.
- **Streaming + non-streaming** both supported (SSE passthrough).

NOT in Phase 3: the MCP server (Phase 4), Hermes wiring/proving ground (Phase 5).

## 1. Module layout

```
src/contextmanager/proxy/
  __init__.py
  config.py        # ProxyConfig
  rewriter.py      # PromptRewriter — pure, deterministic message-list transform (the core)
  upstream.py      # UpstreamClient — async httpx to llama-server (stream + non-stream)
  app.py           # FastAPI app factory create_app(config) -> ASGI app
  __main__.py      # `python -m contextmanager.proxy` -> uvicorn launch
tests/proxy/
  test_rewriter.py     # the prefix-stability + handle-ization invariants (pure, no I/O)
  test_app_nonstream.py
  test_app_stream.py
  test_upstream.py
```
Keep proxy code in its own subpackage; do NOT modify Phase 1/2 modules. The proxy DEPENDS on
them (`DurableStore`, `LlamaServerTokenCounter`/the `TokenCounter` Protocol, `Message`).

## 2. config.py — NORMATIVE

```python
@dataclass(frozen=True)
class ProxyConfig:
    upstream_base_url: str            # llama-server, e.g. "http://127.0.0.1:8080"
    store_root: str = "./contextstore"
    upstream_api_key: Optional[str] = None
    listen_host: str = "127.0.0.1"
    listen_port: int = 8900
    handle_threshold_tokens: int = 2000   # messages whose content >= this get handle-ized
    stub_preview_chars: int = 200         # chars of head/tail kept in the stub preview
    rehydrate_budget_tokens: int = 4000   # max tokens paged back in per request (auto-rehydrate)
    request_timeout: float = 300.0        # upstream timeout (generation can be long)
    # __post_init__: validate handle_threshold_tokens > 0, stub_preview_chars >= 0,
    # rehydrate_budget_tokens >= 0, port in 1..65535. ValueError on violation.
```

## 3. rewriter.py — THE CORE (pure, deterministic) — NORMATIVE

The rewriter is a **pure function over a messages list** (no network; it is given a
`TokenCounter` and a `Store`-like sink/source). This is where all the correctness lives and
where the invariant tests bite. Keep it free of FastAPI/httpx.

### 3.1 Handle marker format (stable, parseable)
A handle-ized message's content is replaced by a deterministic stub:
```
[[cm:stored handle=<handle> role=<role> tokens=<n>]]
<preview-head>
…(truncated <m> chars)…
<preview-tail>
[[/cm:stored]]
```
- `<handle>` is the `DurableStore` handle (== `NoteStore.handle_for(message_id)`); requires a
  stable per-message id. Since the proxy gets RAW OpenAI messages (no ids), derive a stable id
  deterministically: `id = "msg-" + sha1(role + "\x00" + content).hexdigest()[:16]`. SAME
  (role, content) ⇒ SAME id ⇒ SAME handle ⇒ idempotent (re-sending the same message re-uses
  the same stored note; no duplication). This determinism is what gives prefix stability.
- preview-head = first `stub_preview_chars` chars, preview-tail = last `stub_preview_chars`
  chars (omit the "…truncated…" line and tail if content length <= 2*stub_preview_chars).
- The marker must be unambiguously machine-detectable for rehydration (see 3.4): a regex like
  `r"\[\[cm:stored handle=(?P<handle>\S+) role=\S+ tokens=\d+\]\]"`.

### 3.2 RewriteResult
```python
@dataclass
class RewriteResult:
    messages: list[dict]          # the rewritten OpenAI messages to send upstream
    handle_ized_ids: list[str]    # ids of messages that were replaced this call
    rehydrated_handles: list[str] # handles whose content was paged back in this call
```

### 3.3 PromptRewriter
```python
class PromptRewriter:
    def __init__(self, config: ProxyConfig, counter: TokenCounter, store: DurableStore) -> None: ...

    @staticmethod
    def stable_id(role: str, content: str) -> str: ...     # "msg-"+sha1(...)[:16]

    @staticmethod
    def make_stub(handle: str, role: str, tokens: int, content: str,
                  preview_chars: int) -> str: ...           # the 3.1 marker text

    @staticmethod
    def parse_handles(text: str) -> list[str]: ...          # all cm:stored handles in text

    @staticmethod
    def is_stub(content: str) -> bool: ...

    def rewrite_outgoing(self, messages: list[dict]) -> RewriteResult: ...
```

### 3.4 `rewrite_outgoing` behavior (EXACT)
Input: the OpenAI `messages` array (list of dicts with at least `role`, `content`; `content`
may be a string OR the OpenAI content-parts list — if it is not a plain string, treat the
message as NON-handle-izable and pass it through untouched). For each message, IN ORDER:

1. **Already a stub?** If `is_stub(content)`: leave it as-is in the output (do NOT re-handle-ize,
   do NOT page out again). It MAY trigger rehydration in step 4. This is what makes repeated
   turns idempotent and prefix-stable: a message handle-ized on turn N stays the exact same
   stub bytes on turn N+1.
2. **Bulky & plain string?** Else if `content` is a str and
   `counter.count_text(content) >= config.handle_threshold_tokens`:
   - `mid = stable_id(role, content)`; `msg = Message(role=role, content=content, id=mid)`.
   - `handle = store.page_out(msg)` (idempotent: same id ⇒ same handle ⇒ overwrites identical note).
   - replace the message's content with `make_stub(handle, role, tokens, content, preview_chars)`;
     record `mid` in `handle_ized_ids`.
3. **Else** pass through unchanged.
4. **Auto-rehydration (after the pass above):** scan the FINAL message that is from the model's
   counterpart input — specifically, collect handles referenced by `parse_handles` in ANY
   NON-stub message content (i.e. the model or user explicitly wrote a `[[cm:stored handle=…]]`
   reference, or asked about a handle). For each referenced handle not already expanded, load
   `store.get(handle)` and APPEND a synthetic message
   `{"role": "user", "content": "[[cm:rehydrated handle=<h>]]\n<full content>"}` (amended
   2026-07-01: role was "system"; strict chat templates — Qwen — reject a system message
   anywhere but index 0, so ALL synthetic mid-wire messages use role "user") immediately
   AFTER the message that referenced it, subject to a running token budget of
   `config.rehydrate_budget_tokens` (use `counter.count_text`; truncate the last one via
   `counter.truncate_to_tokens` to fit; never exceed the budget). Record handles in
   `rehydrated_handles`. (Stubs themselves do NOT auto-rehydrate — only explicit references do;
   this keeps the common case bounded.)

### 3.5 Prefix-stability invariant (MUST hold; tested)
For any messages list `M`, let `R1 = rewrite_outgoing(M).messages`. Then
`rewrite_outgoing(R1).messages == R1` — i.e. **rewriting is idempotent** (a second pass is a
no-op on already-rewritten output, modulo rehydration of newly-present explicit references).
Additionally: if `M2` extends `M1` by APPENDING messages (same prefix), then the rewritten
prefix of `M2` equals the rewritten `M1` for all the shared leading messages that were plain
or already-stub (handle-ization of message i depends ONLY on message i, never on its
neighbors — proven by construction since stable_id/threshold are per-message). This per-message
determinism is the KV-cache prefix-stability guarantee.

## 4. upstream.py — NORMATIVE

```python
class UpstreamError(Exception): ...

class UpstreamClient:
    def __init__(self, config: ProxyConfig, client: "httpx.AsyncClient | None" = None) -> None: ...
        # owns an httpx.AsyncClient(base_url=upstream_base_url, timeout=request_timeout) unless injected.
    async def chat_completion(self, payload: dict) -> dict: ...
        # POST /v1/chat/completions (non-stream). Return parsed JSON. Wrap errors in UpstreamError.
    async def chat_completion_stream(self, payload: dict) -> "AsyncIterator[bytes]": ...
        # POST with stream=true; yield raw SSE chunks (bytes) as they arrive, unbuffered,
        # passthrough verbatim (do NOT parse/re-serialize the SSE; forward bytes). Wrap
        # connection errors in UpstreamError. Forward upstream non-200 as UpstreamError(status).
    async def passthrough_get(self, path: str) -> dict: ...    # for /v1/models, /props passthrough
    async def aclose(self) -> None: ...
```

## 5. app.py — NORMATIVE (FastAPI)

`create_app(config: ProxyConfig) -> FastAPI`. On startup build: `DurableStore(config.store_root)`,
`LlamaServerTokenCounter(config.upstream_base_url, api_key=config.upstream_api_key)` (as the
`TokenCounter`), `PromptRewriter(config, counter, store)`, `UpstreamClient(config)`. Store them
on `app.state`. Clean up (close store + upstream) on shutdown.

Routes:
- `POST /v1/chat/completions`:
  1. Parse JSON body. Extract `messages`. If absent/not a list → 400 JSON error
     `{"error": {"message": ..., "type": "invalid_request_error"}}`.
  2. `result = rewriter.rewrite_outgoing(messages)`; set `payload = {**body, "messages": result.messages}`.
  3. If `body.get("stream") is True`: return a `StreamingResponse` with
     `media_type="text/event-stream"` that yields from `upstream.chat_completion_stream(payload)`.
  4. Else: `data = await upstream.chat_completion(payload)`; return it as JSON (verbatim — the
     proxy does NOT rewrite responses in Phase 3; response-side handle-ization is out of scope).
  5. On `UpstreamError`: return JSON error with status 502 (or the upstream status if it was an
     HTTP status error), shape `{"error": {"message": ..., "type": "upstream_error"}}`.
- `GET /v1/models` and `GET /props`: passthrough to upstream (best-effort; 502 on failure).
- `GET /healthz`: return `{"status": "ok"}` (does NOT touch upstream).

Headers: forward `Authorization` only as configured `upstream_api_key` (do not blindly forward
client auth). Set `Content-Type: application/json` on JSON responses.

## 6. __main__.py
`python -m contextmanager.proxy` reads config from env vars (CM_UPSTREAM_BASE_URL,
CM_STORE_ROOT, CM_LISTEN_HOST, CM_LISTEN_PORT, CM_HANDLE_THRESHOLD_TOKENS, ...; sensible
defaults from ProxyConfig) and launches `uvicorn.run(create_app(config), host=..., port=...)`.

## 7. Tests — what MUST be proven (no real network; use httpx ASGITransport + a fake upstream)

Use a deterministic fake `TokenCounter` (reuse Phase 1 conftest `FakeCounter`: tokens = word
count; truncate = first-N-words) and a real `DurableStore(tmp_path)`. For the app tests, mount
a FAKE upstream: inject a stub `UpstreamClient` (or an httpx.MockTransport AsyncClient) so no
llama-server is needed.

`tests/proxy/test_rewriter.py` (pure, the heart):
- bulky message (>= threshold) → replaced by a stub; full content retrievable from store by the
  stub's handle; small message passes through unchanged.
- `is_stub`/`parse_handles`/`make_stub` round-trip: parse_handles(make_stub(...)) == [handle].
- **idempotency:** `rewrite_outgoing(rewrite_outgoing(M).messages).messages == that messages`
  (the prefix-stability invariant). Run on a mixed list (small + bulky + already-stub).
- **per-message determinism / prefix stability:** rewriting `M` then `M + [new_msgs]` yields the
  SAME rewritten content for every shared leading message.
- `content` as OpenAI content-parts list (not a str) → passed through untouched (not handle-ized).
- auto-rehydration: a message that explicitly contains `[[cm:stored handle=H]]` (for an H in the
  store) → a `[[cm:rehydrated handle=H]]` user-role message with the full content is appended after
  it; respects `rehydrate_budget_tokens` (truncates/stops); unknown handle → no crash, skipped.
- stable_id determinism: same (role, content) → same id; different → different.

`tests/proxy/test_app_nonstream.py`:
- POST /v1/chat/completions (stream not set) with a bulky message → the payload the fake upstream
  RECEIVES has the bulky message handle-ized (assert the fake upstream saw a stub, not the 40k
  blob); the proxy returns the fake upstream's JSON verbatim.
- missing/invalid `messages` → 400 with the error shape.
- upstream failure → 502 with the error shape.
- GET /healthz → {"status":"ok"} without calling upstream.

`tests/proxy/test_app_stream.py`:
- POST with `stream: true` → response media type text/event-stream; the streamed bytes equal the
  fake upstream's SSE chunks in order (passthrough, unmodified); the request the upstream saw was
  handle-ized.

`tests/proxy/test_upstream.py`:
- UpstreamClient against httpx.MockTransport (AsyncClient): non-stream returns parsed JSON;
  stream yields the chunk bytes in order; upstream 500 → UpstreamError; transport error → UpstreamError.

## 8. Constraints

- New deps THIS phase: `fastapi`, `uvicorn`. Add to pyproject `[project].dependencies`. httpx
  already present. Dev unchanged (pytest/hypothesis). Use `httpx.ASGITransport` for app tests.
- Do NOT modify Phase 1/2 src modules. The orchestrator updates root `__init__.py`/pyproject
  deps after integration; implementers may add `fastapi`/`uvicorn` to pyproject `dependencies`
  (single edit, the durable/proxy facade agent owns that edit) — coordinate via this note: the
  `app.py` implementer also edits pyproject to add the two deps.
- Async correctness: streaming must NOT buffer the whole response; yield chunks as they arrive.
- Full type hints; `from __future__ import annotations`. Every module passes `python -m py_compile`.
- No real network and no real llama-server in tests — fakes / ASGITransport / MockTransport only.

## 9. Round-2 corrections (NORMATIVE — supersede §3–§7 where they conflict)

A code review found a CRITICAL idempotency break (the proxy would recreate the very livelock
it exists to kill) plus a streaming error-path gap and an untested startup path. Apply ALL.

### 9.1 rewriter.py — rehydrated messages must be idempotent (fixes C1, CRITICAL)
The synthetic rehydrated message `"[[cm:rehydrated handle=H]]\n<full>"` is currently re-handle-
ized and re-rehydrated every turn, growing the prompt unboundedly. Fix BOTH passes:
1. Add `@staticmethod is_rehydrated(content) -> bool`: True iff `content` is a str whose lstrip
   starts with `"[[cm:rehydrated handle="`.
2. **Pass 1 (handle-ization):** a message that `is_stub(content)` OR `is_rehydrated(content)` is
   passed through UNCHANGED (never handle-ized, never paged out). A rehydrated message is
   already-rewritten output.
3. **Pass 2 (auto-rehydration):** before rehydrating a referenced handle `H`, scan the CURRENT
   working message list for an existing message that `is_rehydrated` AND whose marker handle ==
   `H` (parse the `[[cm:rehydrated handle=H]]` marker). If one already exists, SKIP `H` (the
   reference has already been expanded in this conversation — do not append a duplicate).
4. Result: `rewrite_outgoing(rewrite_outgoing(M).messages).messages == rewrite_outgoing(M).messages`
   for ALL M, INCLUDING M containing explicit `[[cm:stored handle=H]]` references. And over N
   turns where the output is fed back as input, the message count and per-message content are
   STABLE (no growth). This is the §3.5 invariant, now universal.
5. **L2 hardening:** in the rehydration budget path, truncate only the `<full>` BODY to the
   remaining budget (never truncate the `[[cm:rehydrated handle=H]]` marker line); always emit
   the complete marker so the message stays detectable by `is_rehydrated`/handle-scan next turn.
6. **M1:** compute `count_text(content)` ONCE per bulky message and reuse it for the stub
   `tokens=` field (no double tokenize round-trip).
7. **M2:** in Pass 2, when a handle's (truncated) content does not fit `remaining`, `continue`
   to try the next referenced handle rather than `break` (per-handle budget trial). Stop only
   when `remaining <= 0`.

### 9.2 app.py — streaming UpstreamError must return 502 (fixes H1)
The streaming branch must surface an upstream error as a proper JSON error response with the
correct status BEFORE the 200/event-stream headers are committed. Implement by priming the
generator's first chunk inside a try/except:
```python
agen = upstream.chat_completion_stream(payload)
try:
    first = await agen.__anext__()
except StopAsyncIteration:
    first = None
except UpstreamError as e:
    return _upstream_error_response(e)   # same mapping as non-stream: status_code or 502
async def _body():
    if first is not None:
        yield first
    async for chunk in agen:
        yield chunk
return StreamingResponse(_body(), media_type="text/event-stream")
```
`_upstream_error_response(e)` = JSONResponse({"error":{"message":str(e),"type":"upstream_error"}},
status_code = e.status_code if isinstance(e.status_code,int) and 400<=e.status_code<=599 else 502).
Use it for BOTH the stream and non-stream branches. (A mid-stream error after the first chunk
cannot change the already-sent status — that is acceptable; we only guarantee the pre-header
error maps correctly.)

### 9.3 Additional REQUIRED tests (fixes H2, H3, H4)
- `test_idempotency_with_explicit_reference` (test_rewriter.py): store note H; `M` has a user
  message containing `[[cm:stored handle=H role=user tokens=5]]`; `R1=rewrite_outgoing(M)`;
  `R2=rewrite_outgoing(R1.messages)`; assert `R2.messages == R1.messages` and
  `len(R2.messages) == len(R1.messages)`.
- `test_multiturn_no_growth` (test_rewriter.py): simulate 4 turns feeding each output back as the
  next input (optionally appending a new small user message each turn); assert the count of
  stub + rehydrated messages does NOT grow turn-over-turn for the unchanged prefix (bounded).
- `test_stream_upstream_error_returns_502` (test_app_stream.py): inject a FakeUpstream whose
  `chat_completion_stream` raises `UpstreamError("boom", status_code=500)` on first iteration;
  POST with `stream: true`; assert HTTP 502 (or 500 if status_code carried) and body
  `error.type == "upstream_error"` — NOT a 200 with an empty stream.
- `test_lifespan_startup_wiring` (new test, e.g. test_app_lifespan.py): drive the ASGI lifespan
  (use `asgi-lifespan`'s `LifespanManager` if available, else `starlette.testclient.TestClient`
  as a context manager, else manually enter `app.router.lifespan_context(app)`), passing an
  INJECTED stub upstream + counter via `create_app(..., upstream=..., counter=..., store=...)` so
  no network is needed; after startup assert `app.state.rewriter` is wired to `app.state.counter`
  and `app.state.store` (the resolved instances), and that ownership flags don't close injected
  objects. This closes the untested production-startup seam.
