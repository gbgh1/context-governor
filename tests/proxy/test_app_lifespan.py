"""Lifespan wiring test (spec §9.3 test_lifespan_startup_wiring, fixes H4).

Exercises the REAL FastAPI lifespan (NOT the inject-after-create_app
shortcut used by the other app tests, where ``httpx.ASGITransport`` does not
run lifespan). Drives it with ``starlette.testclient.TestClient`` used as a
context manager, which enters/exits the ASGI lifespan.

Proves:
  - On startup, ``app.state.rewriter`` is built and wired to the RESOLVED
    injected ``counter`` and ``store`` (the resolved instances, not freshly
    constructed ones).
  - ``app.state.upstream`` is the injected stub upstream (no network).
  - A real request through the lifespan-built app works end-to-end (200),
    closing the previously-untested production-startup seam.

This test is SYNC (``TestClient`` is synchronous) — no ``@pytest.mark.asyncio``
decorator.
"""

from __future__ import annotations

from starlette.testclient import TestClient

# ``FakeCounter`` lives in the root ``tests/conftest.py`` (Phase 1 shared fake,
# importable as the bare ``conftest`` module under pytest's prepend import
# mode). ``FakeUpstream`` lives in ``tests/proxy/conftest.py`` which pytest
# registers as the ``proxy.conftest`` module (rootdir-relative unique name,
# since there is no ``__init__.py`` making ``tests/proxy`` a real package).
from conftest import FakeCounter
from proxy.conftest import FakeUpstream
from contextmanager.durable import DurableStore
from contextmanager.proxy.app import create_app


def test_lifespan_startup_wiring(make_config, tmp_path):
    """Drive the real lifespan; assert wiring + a working request.

    ``make_config`` is the conftest fixture that builds a ``ProxyConfig``
    with low thresholds (handle_threshold_tokens=10, stub_preview_chars=10).
    We inject the same instances ``create_app`` accepts via its kwargs
    (``upstream``/``store``/``counter``), so the lifespan must reuse them
    rather than constructing fresh ones.
    """
    config = make_config()
    counter = FakeCounter()
    store = DurableStore(config.store_root)
    upstream = FakeUpstream(response={"id": "ok", "choices": []})

    app = create_app(config, upstream=upstream, store=store, counter=counter)

    with TestClient(app) as client:
        # Entering the context manager runs the ASGI lifespan startup.
        # The rewriter is built on startup and wired to the resolved
        # injected counter + store.
        assert app.state.rewriter is not None
        assert app.state.rewriter.counter is counter, (
            "rewriter must be wired to the injected counter (resolved instance)"
        )
        assert app.state.rewriter.store is store, (
            "rewriter must be wired to the injected store (resolved instance)"
        )
        assert app.state.upstream is upstream, (
            "upstream must be the injected stub (no network)"
        )
        # The injected instances are NOT owned by the app (shutdown must not
        # close them).
        assert app.state._owns_upstream is False
        assert app.state._owns_counter is False
        assert app.state._owns_store is False

        # A real request flows through the lifespan-built app end-to-end.
        # "hi" is well under the 10-token threshold, so it passes through
        # verbatim; the stub upstream returns {"id":"ok","choices":[]}.
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, f"got {r.status_code}: {r.text}"
        assert r.json() == {"id": "ok", "choices": []}


def test_lifespan_anchors_threshold_on_upstream_n_ctx(make_config, tmp_path):
    """The startup probe reads llama-server's true n_ctx from /props and ANCHORS the
    handle-ization threshold to it (ratio * n_ctx), and caches n_ctx for propagation."""
    config = make_config(handle_threshold_tokens=2000, handle_threshold_ratio=0.04)
    counter = FakeCounter()
    store = DurableStore(config.store_root)

    class PropsUpstream(FakeUpstream):
        async def passthrough_get(self, path):
            if path == "/props":
                return {"default_generation_settings": {"n_ctx": 75776}}
            return await super().passthrough_get(path)

    upstream = PropsUpstream(response={"id": "ok", "choices": []})
    app = create_app(config, upstream=upstream, store=store, counter=counter)

    with TestClient(app):
        assert app.state.n_ctx == 75776                       # probed from /props
        expected = int(75776 * 0.04)                          # anchored to the real window
        assert app.state.config.handle_threshold_tokens == expected
        assert app.state.rewriter.config.handle_threshold_tokens == expected
