"""Shared fixtures and fakes for the Phase 3 proxy test suite.

Binds to ``tasks/phase3-spec.md`` (NORMATIVE). The fakes here are deliberately
deterministic so the proxy invariants can be asserted without any real
llama-server / network.

Reuse from Phase 1 root conftest:
  - ``FakeCounter`` (token = whitespace word count; truncate = first-N-words).

Key design notes (see also ``tasks/phase3-spec.md`` §5, §7):
  - ``httpx.ASGITransport`` does NOT drive the ASGI lifespan, so ``create_app``'s
    startup build of ``app.state.{rewriter,upstream,store,counter}`` does NOT
    run when the app is driven in-process. We therefore INJECT these attributes
    explicitly after ``create_app`` (spec §7: "inject a stub UpstreamClient").
  - The injected ``PromptRewriter`` is wired to a ``FakeCounter`` so
    ``count_text`` / ``truncate_to_tokens`` never touch the network (the real
    ``LlamaServerTokenCounter`` would POST to /tokenize).
"""

from __future__ import annotations

from typing import Any, Optional

import pytest


# ---------------------------------------------------------------------------
# FakeUpstream — a stand-in UpstreamClient that records the payload it received
# and returns a canned response / stream / error. No network.
# ---------------------------------------------------------------------------


class FakeUpstream:
    """In-process stand-in for ``UpstreamClient``.

    Records the last payload seen by both the non-stream and stream paths and
    the call counts, so tests can assert the proxy handle-ized the bulky
    message before forwarding. Raises ``error`` (if set) to simulate upstream
    failure (the app maps ``UpstreamError`` -> HTTP 502).
    """

    def __init__(
        self,
        *,
        response: Optional[dict] = None,
        stream_chunks: Optional[list[bytes]] = None,
        error: Optional[Exception] = None,
        notfound: Optional[set] = None,
    ) -> None:
        self.response: dict = (
            response if response is not None else {"id": "x", "choices": []}
        )
        self.stream_chunks: list[bytes] = list(stream_chunks) if stream_chunks else []
        self.error: Optional[Exception] = error
        # Paths for which passthrough_get raises a 404 UpstreamError (simulates an
        # upstream that doesn't implement that route, e.g. llama-server + /api/*).
        self.notfound: set = set(notfound) if notfound else set()
        self.last_payload: Optional[dict] = None
        self.last_stream_payload: Optional[dict] = None
        self.call_count: int = 0
        self.stream_count: int = 0
        self.get_paths: list[str] = []

    async def chat_completion(self, payload: dict) -> dict:
        self.last_payload = payload
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.response

    async def chat_completion_stream(self, payload: dict):  # type: ignore[no-untyped-def]
        self.last_stream_payload = payload
        self.stream_count += 1
        if self.error is not None:
            raise self.error
        for chunk in self.stream_chunks:
            yield chunk

    async def passthrough_get(self, path: str) -> dict:
        self.get_paths.append(path)
        if self.error is not None:
            raise self.error
        if path in self.notfound:
            from contextmanager.proxy.upstream import UpstreamError

            raise UpstreamError(f"HTTP 404 from {path}", status_code=404)
        return {"ok": True, "path": path}

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Config / app builder fixtures (lazy imports keep collection robust while the
# proxy src lands in parallel; phase-3 modules are imported only when used).
# ---------------------------------------------------------------------------


def _config_kwargs(tmp_path, **over: Any) -> dict:
    """Build a ProxyConfig kwargs dict with low thresholds for deterministic
    handle-ization (a >=10-word message is bulky)."""
    kw: dict[str, Any] = dict(
        upstream_base_url="http://upstream.test",
        store_root=str(tmp_path / "store"),
        upstream_api_key=None,
        listen_host="127.0.0.1",
        listen_port=8900,
        handle_threshold_tokens=10,
        stub_preview_chars=10,
        rehydrate_budget_tokens=4000,
        request_timeout=30.0,
    )
    kw.update(over)
    return kw


@pytest.fixture
def make_config(tmp_path):
    """Return a callable that builds a ``ProxyConfig`` (lazy import)."""
    from contextmanager.proxy.config import ProxyConfig

    def _build(**over: Any):
        return ProxyConfig(**_config_kwargs(tmp_path, **over))

    return _build


@pytest.fixture
def make_app(tmp_path):
    """Return a callable that builds a FastAPI app with everything injected.

    Signature:
        make_app(*, response=None, stream_chunks=None, error=None,
                 config=None, store=None, counter=None, rewriter=None,
                 **cfg_over) -> (app, upstream, store, rewriter)

    A ``FakeUpstream`` is constructed from ``response`` / ``stream_chunks`` /
    ``error``. The rewriter is wired to a ``FakeCounter`` (word-count tokens)
    so NO network is touched. State is injected onto ``app.state`` AFTER
    ``create_app`` (ASGITransport does not run lifespan -> spec §7 injection).
    """
    from contextmanager.proxy.app import create_app
    from contextmanager.proxy.config import ProxyConfig
    from contextmanager.proxy.rewriter import PromptRewriter
    from contextmanager.durable import DurableStore
    from conftest import FakeCounter

    def _build(
        *,
        response: Optional[dict] = None,
        stream_chunks: Optional[list[bytes]] = None,
        error: Optional[Exception] = None,
        notfound: Optional[set] = None,
        config: Optional[ProxyConfig] = None,
        store: Optional[DurableStore] = None,
        counter: Optional[FakeCounter] = None,
        rewriter: Optional[PromptRewriter] = None,
        **cfg_over: Any,
    ):
        if config is None:
            config = ProxyConfig(**_config_kwargs(tmp_path, **cfg_over))
        if store is None:
            store = DurableStore(config.store_root)
        if counter is None:
            counter = FakeCounter()
        if rewriter is None:
            rewriter = PromptRewriter(config, counter, store)
        upstream = FakeUpstream(
            response=response, stream_chunks=stream_chunks, error=error, notfound=notfound
        )

        app = create_app(config)
        # Inject (spec §7). ASGITransport does not run lifespan, so the app's
        # startup build never executes; these stand for the whole test.
        app.state.rewriter = rewriter
        app.state.upstream = upstream
        app.state.store = store
        app.state.counter = counter
        return app, upstream, store, rewriter

    return _build
