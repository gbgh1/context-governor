# Phase 4 Spec — Surface B: the MCP Server (cooperative)

> Authoritative contract for Phase 4. Implementers and the tester bind to THIS document.
> Names, signatures, and the return shapes are normative. Python 3.11+, `src/` layout.
> NEW runtime dep THIS phase: `mcp` (official Python SDK, `from mcp.server.fastmcp import
> FastMCP`). Dev unchanged (`pytest`, `pytest-asyncio`, `hypothesis`). NO real network and
> NO real llama-server in tests — inject a fake `TokenCounter`; use a real `DurableStore(tmp_path)`.

## 0. What Phase 4 is

A **cooperative** surface: an MCP server that exposes the Phase 1/2 engine + store as
explicit tools, so an MCP-capable agent can *deliberately* externalize and retrieve durable
state (snapshot world/project state each turn, save facts, page them back in on demand).

It is the precise complement to Surface A (the proxy). **Reality check (project memory):**
MCP alone CANNOT override a host CLI's native compaction — that is the proxy's job. Surface
B makes the agent's own context management explicit and durable; it does not intercept the
host's prompt. The two surfaces share ONE store (`DurableStore`) and never fork it.

Decisions (locked):
- **stdio transport** by default (the standard for local agent CLIs). `streamable-http` is
  reachable via config but not required this phase.
- **All logic lives in a pure `GovernorService`** (no MCP types) so correctness is tested
  without an MCP runtime; `server.py` is a thin FastMCP adapter that (de)serializes.
- **Six tools**, named with underscores (portable across MCP clients that restrict names to
  `[A-Za-z0-9_-]`). Conceptual dotted names from the plan map 1:1:
  `store.save→store_save`, `store.search→store_search`, `state.snapshot→state_snapshot`,
  `state.load→state_load`, `context.checkpoint→context_checkpoint`,
  `context.rehydrate→context_rehydrate`.

NOT in Phase 4: Hermes wiring / proving ground (Phase 5), sterilization (Phase 6).

## 1. Module layout

```
src/contextmanager/mcp/
  __init__.py      # exports McpConfig, GovernorService, HeuristicTokenCounter, build_server
  config.py        # McpConfig (frozen dataclass, validated)
  counter.py       # HeuristicTokenCounter — offline TokenCounter (no llama-server needed)
  service.py       # GovernorService — pure logic over DurableStore + TokenCounter (the core)
  server.py        # build_server(config, *, service=None) -> FastMCP ; registers the 6 tools
  __main__.py      # `python -m contextmanager.mcp` -> build + run stdio (CM_* env config)
tests/mcp_server/      # NOT tests/mcp/ — a package named `mcp` shadows the installed
  __init__.py          #   `mcp` SDK under pytest prepend-import (tests/ on sys.path).
  conftest.py          # make_config, make_service fixtures; reuse Phase1 FakeCounter
  test_config.py       # McpConfig validation
  test_counter.py      # HeuristicTokenCounter measurability invariants
  test_service.py      # the 6 operations, pure (no MCP runtime) — the heart
  test_server.py       # FastMCP integration: list_tools == 6; call_tool round-trips (async)
```
Keep MCP code in its own subpackage; do NOT modify Phase 1/2/3 modules. The MCP server
DEPENDS on them (`DurableStore`, `Message`, the `TokenCounter` Protocol).

## 2. config.py — NORMATIVE

```python
@dataclass(frozen=True)
class McpConfig:
    store_root: str = "./contextstore"
    upstream_base_url: Optional[str] = None   # set -> LlamaServerTokenCounter; None -> HeuristicTokenCounter
    upstream_api_key: Optional[str] = None
    server_name: str = "context-governor"
    transport: str = "stdio"                  # "stdio" | "streamable-http" | "sse"
    default_search_k: int = 5
    preview_chars: int = 200
    rehydrate_budget_tokens: int = 4000
    # __post_init__: ValueError unless default_search_k > 0, preview_chars >= 0,
    #   rehydrate_budget_tokens >= 0, transport in {"stdio","streamable-http","sse"}.
```

## 3. counter.py — NORMATIVE

```python
class HeuristicTokenCounter:  # implements types.TokenCounter (no network)
    """~1 token / 4 chars approximation, deterministic and MEASURABLE so the
    page_in budget clamp (DurableStore.page_in) holds without a llama-server."""
    CHARS_PER_TOKEN = 4
    PER_MESSAGE_OVERHEAD = 4
    def count_text(self, text: str) -> int       # 0 if empty else (len(text)+3)//4
    def count_messages(self, messages: list[Message]) -> int  # sum(count_text(content)) + overhead each
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str  # "" if max<=0 else text[: max*4]
```
Invariant (tested): for all text and n>=0, `count_text(truncate_to_tokens(text, n)) <= n`.

## 4. service.py — THE CORE (pure, no MCP types) — NORMATIVE

```python
class GovernorService:
    def __init__(self, store: DurableStore, counter: TokenCounter, config: McpConfig) -> None: ...

    @staticmethod
    def stable_id(prefix: str, role: str, content: str) -> str:
        # f"{prefix}-" + sha1(role + "\x00" + content).hexdigest()[:16]   (deterministic)

    def store_save(self, content: str, *, role: str = "note",
                   id: Optional[str] = None, links: Optional[list[str]] = None) -> dict:
        # id = id or stable_id("note", role, content)
        # handle = store.page_out(Message(role=role, content=content, id=id))  (also indexes for search)
        # -> {"handle": handle, "id": id, "role": role, "tokens": counter.count_text(content)}

    def store_search(self, query: str, *, k: Optional[int] = None) -> dict:
        # slices = store.search(query, k or config.default_search_k)
        # -> {"query": query, "count": len(slices),
        #     "results": [{"handle": s.handle, "score": s.score,
        #                  "tokens": counter.count_text(s.content),
        #                  "preview": _preview(s.content, config.preview_chars)} ...]}

    def state_snapshot(self, state: dict, *, merge: bool = True) -> dict:
        # merge: store.state.update(state) ; else store.state.save(state); read-back result
        # -> {"ok": True, "merge": merge, "keys": sorted(result.keys()),
        #     "tokens": counter.count_text(store.state.render())}
        # TypeError/ValueError -> raise ValueError("state must be a JSON object") if not isinstance(state, dict)

    def state_load(self) -> dict:
        # data = store.state.load()
        # -> {"state": data, "rendered": store.state.render(), "empty": (not data)}

    def context_checkpoint(self, label: str, content: str, *, state: Optional[dict] = None) -> dict:
        # id = stable_id("checkpoint", label, content); but handle must be label-stable:
        #   use id = "checkpoint-" + _slug(label)   (so re-checkpointing a label OVERWRITES it)
        # save = store_save(content, role="checkpoint", id=id)
        # if state is not None: state_snapshot(state, merge=True)
        # -> {"handle": save["handle"], "label": label, "id": id,
        #     "tokens": save["tokens"], "state_updated": state is not None}

    def context_rehydrate(self, *, query: Optional[str] = None, handle: Optional[str] = None,
                          budget_tokens: Optional[int] = None, k: Optional[int] = None) -> dict:
        # Exactly one of query/handle required (else ValueError).
        # budget = rehydrate_budget_tokens if budget_tokens is None else budget_tokens
        # handle path: content = store.get(handle) (StoreError -> {"found": False, ...});
        #   if counter.count_text(content) > budget: content = counter.truncate_to_tokens(content, budget)
        #   slices = [(handle, content)]
        # query path: slices = store.page_in(query, budget, counter, k or config.default_search_k)
        # text = "\n\n".join(f"[[cm:slice handle={h}]]\n{c}" for h,c in slices)
        # -> {"found": bool(slices), "budget_tokens": budget, "count": len(slices),
        #     "handles": [h...], "tokens": sum(count_text(c) for slices), "text": text}
        # NOTE: `tokens` reports the budgeted RETRIEVED-CONTENT tokens (held <= budget by
        # page_in / handle-path truncation); the [[cm:slice …]] markers are presentation
        # scaffolding and are NOT counted against the budget.

    def close(self) -> None:   # store.close(); close counter if it has .close()
```
`_preview(content, n)`: first `n` chars; append `"…(+M more chars)"` when truncated (M = len-n).
`_slug(label)`: lowercase, `[^a-z0-9._-]+` -> `-`, strip leading/trailing `-`; fallback to a
sha1[:8] of label if empty. (Mirrors NoteStore.handle_for safety.)

All return values are JSON-serializable plain dicts/lists/strs/ints/floats/bools (FastMCP
serializes them to a JSON `TextContent`).

## 5. server.py — NORMATIVE

```python
def build_service(config: McpConfig) -> GovernorService:
    # store = DurableStore(config.store_root)
    # counter = LlamaServerTokenCounter(config.upstream_base_url, api_key=...) if upstream_base_url
    #           else HeuristicTokenCounter()
    # return GovernorService(store, counter, config)

def build_server(config: McpConfig, *, service: Optional[GovernorService] = None) -> FastMCP:
    # svc = service or build_service(config)
    # mcp = FastMCP(config.server_name)
    # register 6 @mcp.tool() wrappers (underscore names) each calling svc.* and returning its dict.
    # stash svc on mcp so __main__ can close it: setattr(mcp, "_governor_service", svc)
    # return mcp
```
Tool wrappers MUST have full type hints (FastMCP builds the input schema from them) and a
one-line docstring (becomes the tool description). `state`-typed params use `dict`.
`context_rehydrate` params are all optional (`query: str | None = None`, etc.).

## 6. __main__.py — NORMATIVE
`python -m contextmanager.mcp` builds `McpConfig` from `CM_*` env vars (`CM_STORE_ROOT`,
`CM_UPSTREAM_BASE_URL`, `CM_UPSTREAM_API_KEY`, `CM_MCP_SERVER_NAME`, `CM_MCP_TRANSPORT`,
`CM_DEFAULT_SEARCH_K`, `CM_PREVIEW_CHARS`, `CM_REHYDRATE_BUDGET_TOKENS`), builds the server,
and `mcp.run(transport=config.transport)`. `finally:` close the stashed service.

## 7. Tests — what MUST be proven (no real network / llama-server)

`tests/mcp/conftest.py`: `make_config()` (callable -> McpConfig with tmp_path store_root,
small previews), `make_service()` (callable -> GovernorService wired to a Phase-1 `FakeCounter`
and a real `DurableStore(tmp_path)`). Reuse `from conftest import FakeCounter`.

- `test_config.py`: valid build; each invalid field (k<=0, preview<0, budget<0, bad transport)
  -> ValueError.
- `test_counter.py`: `count_text("")==0`; non-empty -> ceil(len/4); the measurability invariant
  `count_text(truncate_to_tokens(text, n)) <= n` over several (text, n) incl. n=0.
- `test_service.py` (the heart):
  - store_save -> handle; `store.get(handle)` returns the exact content; tokens match counter;
    idempotent (same content+role -> same id/handle, no duplicate note).
  - store_search finds a saved note by a term in its content; preview respects preview_chars.
  - state_snapshot(merge=True) shallow-merges; merge=False replaces; state_load returns state +
    rendered + empty flag.
  - context_checkpoint saves a checkpoint note AND (when state given) updates state; same label
    -> same handle (overwrite, not duplicate).
  - context_rehydrate by handle returns that content (truncated to budget when over);
    by query returns slices with content `tokens <= budget`; budget 0 -> found False / empty;
    neither query nor handle -> ValueError; unknown handle -> found False (no crash).
- `test_server.py` (async, `@pytest.mark.asyncio`): build_server(config, service=injected);
  `await mcp.list_tools()` has exactly the 6 underscore names; `await mcp.call_tool("store_save",
  {"content": <bulky>})` -> parse the returned TextContent JSON, assert a "handle"; then
  `call_tool("store_search", {"query": ...})` finds it. (Handle FastMCP returning either a
  content list or a (content, structured) tuple — normalize in the test.)

## 8. Constraints
- New dep THIS phase: `mcp`. Add to pyproject `[project].dependencies`.
- Do NOT modify Phase 1/2/3 src. MCP code is additive under `src/contextmanager/mcp/`.
- Full type hints; `from __future__ import annotations`. Every module passes `python -m py_compile`.
- Pure `GovernorService` carries the correctness; `server.py` stays a thin adapter.
- No real network / llama-server in tests — `FakeCounter` + `DurableStore(tmp_path)` only.
```
