"""Tests for ``UpstreamClient`` against an ``httpx.MockTransport`` (spec §4, §7).

NO real network, NO real llama-server. The UpstreamClient is given an injected
``httpx.AsyncClient(transport=httpx.MockTransport(handler))`` so every request
is served (or failed) deterministically by the handler.

Proves:
  - non-stream: parsed JSON returned.
  - stream: chunk bytes yielded in order (concatenated == upstream body).
  - upstream 500 -> UpstreamError (with status_code).
  - transport error (handler raises httpx.ConnectError) -> UpstreamError.
"""

from __future__ import annotations

import pytest
import httpx

from contextmanager.proxy.config import ProxyConfig
from contextmanager.proxy.upstream import UpstreamClient, UpstreamError


def _config(tmp_path) -> ProxyConfig:
    return ProxyConfig(
        upstream_base_url="http://upstream.test",
        store_root=str(tmp_path / "store"),
        handle_threshold_tokens=10,
        request_timeout=10.0,
    )


def _client_with_handler(tmp_path, handler):
    """Build an UpstreamClient backed by a MockTransport handler.

    Returns (client, async_client) so the test can close the injected client.
    """
    cfg = _config(tmp_path)
    async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://upstream.test",
    )
    return UpstreamClient(cfg, client=async_client), async_client


SSE_BODY = b'data: {"x":1}\n\ndata: {"x":2}\n\ndata: [DONE]\n\n'


@pytest.mark.asyncio
async def test_nonstream_returns_parsed_json(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"id": "x", "choices": []})

    client, ac = _client_with_handler(tmp_path, handler)
    try:
        data = await client.chat_completion({"messages": [], "stream": False})
        assert data == {"id": "x", "choices": []}
    finally:
        await ac.aclose()


@pytest.mark.asyncio
async def test_stream_yields_chunks_in_order(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=SSE_BODY,
            headers={"content-type": "text/event-stream"},
        )

    client, ac = _client_with_handler(tmp_path, handler)
    try:
        out: list[bytes] = []
        async for chunk in client.chat_completion_stream(
            {"messages": [], "stream": True}
        ):
            assert isinstance(chunk, bytes)
            out.append(chunk)
        # Concatenated bytes equal the upstream SSE body, in order.
        assert b"".join(out) == SSE_BODY
    finally:
        await ac.aclose()


@pytest.mark.asyncio
async def test_upstream_500_raises_upstream_error_nonstream(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    client, ac = _client_with_handler(tmp_path, handler)
    try:
        with pytest.raises(UpstreamError) as ei:
            await client.chat_completion({"messages": []})
        assert ei.value.status_code == 500
    finally:
        await ac.aclose()


@pytest.mark.asyncio
async def test_upstream_500_raises_upstream_error_stream(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    client, ac = _client_with_handler(tmp_path, handler)
    try:
        with pytest.raises(UpstreamError) as ei:
            async for _ in client.chat_completion_stream({"messages": [], "stream": True}):
                pass
        assert ei.value.status_code == 500
    finally:
        await ac.aclose()


@pytest.mark.asyncio
async def test_transport_error_raises_upstream_error_nonstream(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client, ac = _client_with_handler(tmp_path, handler)
    try:
        with pytest.raises(UpstreamError):
            await client.chat_completion({"messages": []})
    finally:
        await ac.aclose()


@pytest.mark.asyncio
async def test_transport_error_raises_upstream_error_stream(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client, ac = _client_with_handler(tmp_path, handler)
    try:
        with pytest.raises(UpstreamError):
            async for _ in client.chat_completion_stream(
                {"messages": [], "stream": True}
            ):
                pass
    finally:
        await ac.aclose()
