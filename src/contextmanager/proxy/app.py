from __future__ import annotations

import asyncio
import dataclasses
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..durable import DurableStore
from ..tokenizer import LlamaServerTokenCounter
from ..types import TokenCounter
from .config import ProxyConfig
from .metrics import StatsCollector
from .rewriter import PromptRewriter, RewriteResult
from .upstream import UpstreamClient, UpstreamError


# Floor for the n_ctx-anchored handle-ization threshold (don't stub trivially small msgs).
_MIN_HANDLE_THRESHOLD = 256


def resolve_handle_threshold(config: ProxyConfig, n_ctx: Optional[int]) -> int:
    """Effective per-message handle-ization threshold. When ``handle_threshold_ratio``
    > 0 AND the upstream's true context size ``n_ctx`` is known, anchor it to the real
    window (``ratio * n_ctx``, floored). Otherwise fall back to the fixed
    ``handle_threshold_tokens``. llama-server is the source of truth for context size."""
    if config.handle_threshold_ratio > 0.0 and n_ctx:
        return max(_MIN_HANDLE_THRESHOLD, int(n_ctx * config.handle_threshold_ratio))
    return config.handle_threshold_tokens


async def _probe_n_ctx(upstream: UpstreamClient) -> Optional[int]:
    """Best-effort read of llama-server's true context size from /props. None on any
    failure (server down at startup, unexpected shape) -> caller falls back."""
    try:
        data = await upstream.passthrough_get("/props")
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return int(data["default_generation_settings"]["n_ctx"])
    except (KeyError, TypeError, ValueError):
        try:
            return int(data["n_ctx"])  # some builds expose it at the top level
        except (KeyError, TypeError, ValueError):
            return None


def _inject_context_length(data: dict, n_ctx: Optional[int]) -> dict:
    """Advertise the upstream's true context size on each model entry (OpenAI `data`
    and Ollama `models` shapes), so clients read the real window from /v1/models."""
    if not n_ctx or not isinstance(data, dict):
        return data
    out = dict(data)
    for key in ("data", "models"):
        items = out.get(key)
        if isinstance(items, list) and items and isinstance(items[0], dict):
            out[key] = [{**it, "context_length": n_ctx} for it in items]
    return out


def _apply_model_alias(data: dict, alias: Optional[str]) -> dict:
    """Present the upstream model list under ``alias`` (e.g. "context-governor"),
    inheriting every other field from the real loaded model. Handles both the OpenAI
    shape (`{"data":[{"id":…}]}`) and the Ollama shape (`{"models":[{"name":…}]}`).
    No-op when alias is falsy or the payload has no recognizable model list.
    """
    if not alias or not isinstance(data, dict):
        return data
    out = dict(data)
    # llama-server can return BOTH keys (OpenAI `data` + Ollama `models`); alias each.
    items = out.get("data")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        out["data"] = [{**items[0], "id": alias}]
    items = out.get("models")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        out["models"] = [{**items[0], "name": alias, "model": alias}]
    return out


def _sum_content_chars(messages: list) -> int:
    """Total chars of string-valued message contents (free; no tokenization)."""
    total = 0
    for m in messages:
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, str):
                total += len(c)
    return total


def _upstream_error_response(exc: UpstreamError) -> JSONResponse:
    """Map an UpstreamError onto the outbound 502-ish JSON error shape.

    Status is the upstream HTTP status when it is a sane int in 400..599,
    otherwise 502 (bad gateway).
    """
    status = exc.status_code
    if not (isinstance(status, int) and 400 <= status <= 599):
        status = 502
    return JSONResponse(
        {"error": {"message": str(exc), "type": "upstream_error"}},
        status_code=status,
        media_type="application/json",
    )


def _invalid_request(message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error"}},
        status_code=400,
        media_type="application/json",
    )


def create_app(
    config: ProxyConfig,
    *,
    upstream: Optional[UpstreamClient] = None,
    store: Optional[DurableStore] = None,
    counter: Optional[TokenCounter] = None,
) -> FastAPI:
    """Build the FastAPI proxy app.

    The optional ``upstream``/``store``/``counter`` are testability hooks: when
    provided they are used as-is; when omitted, the real Phase 1/2 objects are
    constructed on startup. Tests may also overwrite ``app.state.upstream`` /
    ``app.state.store`` / ``app.state.counter`` / ``app.state.rewriter`` AFTER
    construction and BEFORE issuing requests (the lifespan will not clobber
    instances it did not create itself).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Build only what was NOT injected. Idempotent: if a value is already
        # present on app.state (injected by the caller), do not replace it.
        if app.state.store is None:
            app.state.store = DurableStore(config.store_root)
            app.state._owns_store = True
        else:
            app.state._owns_store = False

        if app.state.counter is None:
            app.state.counter = LlamaServerTokenCounter(
                config.upstream_base_url, api_key=config.upstream_api_key
            )
            app.state._owns_counter = True
        else:
            app.state._owns_counter = False

        if app.state.upstream is None:
            app.state.upstream = UpstreamClient(config)
            app.state._owns_upstream = True
        else:
            app.state._owns_upstream = False

        # Anchor on llama-server's TRUE context size (the source of truth, not the
        # CLI). Best-effort + short-timed: if the server isn't reachable at startup,
        # fall back to the fixed threshold. Cached for /v1/models propagation.
        try:
            app.state.n_ctx = await asyncio.wait_for(
                _probe_n_ctx(app.state.upstream), timeout=5.0
            )
        except Exception:
            app.state.n_ctx = None

        resolved = config
        effective_threshold = resolve_handle_threshold(config, app.state.n_ctx)
        if effective_threshold != config.handle_threshold_tokens:
            resolved = dataclasses.replace(
                config, handle_threshold_tokens=effective_threshold
            )

        # The rewriter is always (re)built here so it wires to the resolved
        # counter/store. Tests that inject a custom rewriter should set it
        # AFTER startup; the request handlers read app.state.rewriter live.
        app.state.rewriter = PromptRewriter(
            resolved, app.state.counter, app.state.store, n_ctx=app.state.n_ctx
        )
        app.state.config = resolved

        try:
            yield
        finally:
            # Shutdown: close only what we own. Injected instances are the
            # caller's responsibility.
            if getattr(app.state, "_owns_upstream", False) and app.state.upstream is not None:
                await app.state.upstream.aclose()
            if getattr(app.state, "_owns_counter", False) and app.state.counter is not None:
                close_counter = getattr(app.state.counter, "close", None)
                if callable(close_counter):
                    close_counter()
            if getattr(app.state, "_owns_store", False) and app.state.store is not None:
                app.state.store.close()

    app = FastAPI(
        title="contextmanager-proxy",
        version="0.1.0",
        lifespan=lifespan,
    )
    # Pre-seed app.state with injected instances so the lifespan knows what to
    # build vs. reuse. Tests may also overwrite these before issuing requests.
    app.state.config = config
    app.state.upstream = upstream
    app.state.store = store
    app.state.counter = counter
    app.state.rewriter = None  # type: ignore[assignment]
    # In-memory observability (Phase 5 §3). Set here (not only in the lifespan)
    # so it exists in injected-test mode too, where ASGITransport skips lifespan.
    app.state.stats = StatsCollector()
    # Serializes the off-event-loop rewrite (run via asyncio.to_thread) so the store's
    # sqlite connections (opened check_same_thread=False) are never used concurrently.
    app.state.rewrite_lock = asyncio.Lock()
    # Upstream true context size (filled at startup by the lifespan probe); None
    # in injected-test mode (lifespan skipped) -> no context propagation/anchoring.
    app.state.n_ctx = None

    # ----------------------------------------------------------- /v1/chat/completions

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception as e:
            return _invalid_request(f"invalid JSON body: {e}")

        if not isinstance(body, dict):
            return _invalid_request("request body must be a JSON object")

        messages = body.get("messages")
        if not isinstance(messages, list):
            return _invalid_request("'messages' must be a list")

        rewriter: PromptRewriter = app.state.rewriter
        # Run the (sync, CPU+IO-bound) rewrite OFF the event loop so a heavy message never
        # freezes the proxy — /healthz, /metrics and other requests stay responsive. The
        # lock serializes store access so the worker thread is the only one touching the
        # sqlite connections at a time.
        async with app.state.rewrite_lock:
            result: RewriteResult = await asyncio.to_thread(
                rewriter.rewrite_outgoing, messages
            )
        payload = {**body, "messages": result.messages}

        # Record prompt-transform stats right after the rewrite (before
        # forwarding), so measurement reflects what the proxy did to the prompt
        # regardless of the upstream outcome. Char counts are free (no tokenize).
        app.state.stats.record(
            messages_in=len(messages),
            messages_handle_ized=len(result.handle_ized_ids),
            messages_rehydrated=len(result.rehydrated_handles),
            slices_recalled=len(result.recalled_handles),
            chars_in=_sum_content_chars(messages),
            chars_out=_sum_content_chars(result.messages),
        )

        upstream_client: UpstreamClient = app.state.upstream

        if body.get("stream") is True:
            # §9.2 (H1): prime the generator's first chunk inside try/except so
            # an UpstreamError raised before any bytes are emitted maps to a
            # proper JSON error response with the correct status BEFORE the
            # 200/event-stream headers are committed. A mid-stream error
            # after the first chunk cannot change the already-sent status;
            # that is acceptable.
            agen = upstream_client.chat_completion_stream(payload)
            first: Optional[bytes]
            try:
                first = await agen.__anext__()
            except StopAsyncIteration:
                first = None
            except UpstreamError as e:
                return _upstream_error_response(e)

            async def _body():
                if first is not None:
                    yield first
                async for chunk in agen:
                    yield chunk

            return StreamingResponse(_body(), media_type="text/event-stream")

        try:
            data = await upstream_client.chat_completion(payload)
        except UpstreamError as e:
            return _upstream_error_response(e)
        return JSONResponse(data, media_type="application/json")

    # ----------------------------------------------------------- /v1/models + /props

    @app.get("/v1/models")
    async def get_models():
        upstream_client: UpstreamClient = app.state.upstream
        try:
            data = await upstream_client.passthrough_get("/v1/models")
        except UpstreamError as e:
            return _upstream_error_response(e)
        data = _apply_model_alias(data, app.state.config.model_alias)
        data = _inject_context_length(data, app.state.n_ctx)
        return JSONResponse(data, media_type="application/json")

    @app.get("/props")
    @app.get("/v1/props")
    async def get_props():
        # Clients using a /v1 base URL probe /v1/props (llama-server serves /props at
        # root); accept both and pass through to the upstream's /props.
        upstream_client: UpstreamClient = app.state.upstream
        try:
            data = await upstream_client.passthrough_get("/props")
        except UpstreamError as e:
            return _upstream_error_response(e)
        return JSONResponse(data, media_type="application/json")

    # ------------------------------------------------- Ollama-style model discovery
    # Some CLIs auto-detect the backend by probing Ollama's native model-list paths
    # (`/api/tags`, `/api/v1/models`, `/api/v1/tags`). Behaviour, by design:
    #   1. FORWARD the probe to the upstream at its ORIGINAL path first. If the
    #      upstream actually implements it (a real Ollama or multi-backend server),
    #      its answer is returned verbatim — we never override a working `/api/*`.
    #   2. ONLY when the upstream returns 404 (llama-server has no `/api/*` routes)
    #      fall back to the upstream's model list (`/v1/models`) so the probe is
    #      still answered (200) and the client stops logging 404s.
    # DISCOVERY ONLY — chat still goes through `/v1/chat/completions`; we do NOT
    # emulate Ollama's `/api/chat`, so the proxy never falsely claims to be a full
    # Ollama chat backend.
    @app.get("/api/tags")
    @app.get("/api/v1/tags")
    @app.get("/api/v1/models")
    async def ollama_discovery(request: Request):
        upstream_client: UpstreamClient = app.state.upstream
        alias = app.state.config.model_alias
        original_path = request.url.path
        try:
            data = await upstream_client.passthrough_get(original_path)
            return JSONResponse(
                _inject_context_length(_apply_model_alias(data, alias), app.state.n_ctx),
                media_type="application/json",
            )
        except UpstreamError as e:
            if e.status_code == 404:
                # Upstream doesn't serve this Ollama path -> answer from its model list.
                try:
                    data = await upstream_client.passthrough_get("/v1/models")
                    return JSONResponse(
                _inject_context_length(_apply_model_alias(data, alias), app.state.n_ctx),
                media_type="application/json",
            )
                except UpstreamError as fallback_error:
                    return _upstream_error_response(fallback_error)
            # Any non-404 upstream error is surfaced transparently.
            return _upstream_error_response(e)

    # ----------------------------------------------------------- /healthz

    @app.get("/healthz")
    async def healthz():
        # Does NOT touch the upstream.
        return JSONResponse({"status": "ok"}, media_type="application/json")

    # ----------------------------------------------------------- /metrics

    @app.get("/metrics")
    async def metrics():
        # Cumulative prompt-transform stats (Phase 5 §3) + retrieval-path counters
        # from the shared store (Phase 7 Stage 1). Does NOT touch upstream.
        snap = app.state.stats.snapshot()
        store = app.state.store
        if store is not None:
            stats_fn = getattr(store, "stats", None)
            if callable(stats_fn):
                try:
                    snap["retrieval"] = stats_fn()
                except Exception:
                    pass
        return JSONResponse(snap, media_type="application/json")

    return app
