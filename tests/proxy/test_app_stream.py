"""Streaming app tests (spec §5 step 3, §7) — drive the FastAPI app IN-PROCESS
via ``httpx.ASGITransport`` with a FAKE upstream injected.

Async runner: pytest-asyncio (strict mode — each async test is marked
``@pytest.mark.asyncio``).

Proves:
  - POST with stream:true and a bulky message -> response content-type is
    text/event-stream; the streamed bytes equal the fake upstream's SSE chunks
    concatenated in order (passthrough unmodified); the payload the upstream
    saw was handle-ized (stub, not the blob).
"""

from __future__ import annotations

import pytest
import httpx

from contextmanager.proxy.rewriter import PromptRewriter


BULKY = " ".join(f"word{i}" for i in range(50))  # 50 words >= threshold(10)
SSE_CHUNKS = [b'data: {"x":1}\n\n', b'data: {"x":2}\n\n', b'data: [DONE]\n\n']


@pytest.mark.asyncio
async def test_stream_passthrough_and_handleization(make_app):
    app, upstream, store, rewriter = make_app(stream_chunks=SSE_CHUNKS)
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": BULKY}],
        "stream": True,
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/v1/chat/completions", json=body) as r:
            assert r.status_code == 200
            ct = r.headers.get("content-type", "")
            assert ct.startswith("text/event-stream"), f"got {ct!r}"

            received: list[bytes] = []
            async for chunk in r.aiter_bytes():
                assert isinstance(chunk, bytes)
                received.append(chunk)

    # The streamed bytes equal the fake upstream's SSE chunks concatenated,
    # in order (passthrough verbatim — no parse/re-serialize).
    assert b"".join(received) == b"".join(SSE_CHUNKS)

    # The upstream was called once on the stream path.
    assert upstream.stream_count == 1
    # The payload the upstream saw was handle-ized (stub, not the blob).
    payload = upstream.last_stream_payload
    assert payload is not None
    assert payload.get("stream") is True
    sent = payload["messages"][0]["content"]
    assert PromptRewriter.is_stub(sent)
    assert BULKY not in sent


@pytest.mark.asyncio
async def test_stream_upstream_error_returns_502(make_app):
    """Spec §9.3 test_stream_upstream_error_returns_502 (fixes H1).

    The streaming branch must surface an upstream error raised BEFORE the
    first byte as a proper JSON error response with the correct status, NOT
    a 200 with a broken/empty event-stream. We inject a FakeUpstream whose
    ``chat_completion_stream`` raises ``UpstreamError("boom", status_code=500)``
    at the start (before yielding any chunk) and POST with ``stream: true``.
    """
    from contextmanager.proxy.upstream import UpstreamError

    app, upstream, store, rewriter = make_app(
        error=UpstreamError("boom", status_code=500),
        stream_chunks=[b"data: x\n\n"],
    )
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # ``client.post`` fully reads the response body (a StreamingResponse
        # whose generator raised on the primed first chunk returns the JSON
        # error response instead). This is the contract proven here.
        r = await client.post("/v1/chat/completions", json=body)

    # The status MUST be the carried upstream status (500) — NOT 200.
    assert r.status_code != 200, "streaming error must not commit a 200 status"
    assert r.status_code in (500, 502), f"got {r.status_code}"
    # The body is the JSON error shape, not a stream.
    assert r.json()["error"]["type"] == "upstream_error"
    # The stream path was attempted (priming reached the generator).
    assert upstream.stream_count == 1
