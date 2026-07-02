"""/metrics observability tests (Phase 5 §3) — drive the app in-process via
ASGITransport with a fake upstream; assert the proxy-side counter reflects the
prompt transform (and never touches the upstream).
"""

from __future__ import annotations

import httpx
import pytest

BULKY = " ".join(f"word{i}" for i in range(50))  # 50 words >= threshold(10)
FAKE_RESPONSE = {"id": "x", "object": "chat.completion", "choices": []}


async def _post(app, json):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/chat/completions", json=json)


async def _get(app, path):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


@pytest.mark.asyncio
async def test_metrics_counts_handleization_and_wire_shrink(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    await _post(app, {"messages": [{"role": "user", "content": BULKY}]})
    m = (await _get(app, "/metrics")).json()
    assert m["requests"] == 1
    assert m["messages_in"] == 1
    assert m["messages_handle_ized"] >= 1
    # The wire shrank: the bulky blob became a short stub.
    assert m["chars_out"] < m["chars_in"]
    assert m["chars_saved"] == m["chars_in"] - m["chars_out"]
    assert m["chars_saved"] > 0


@pytest.mark.asyncio
async def test_metrics_small_message_no_savings(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    await _post(app, {"messages": [{"role": "user", "content": "hello world"}]})
    m = (await _get(app, "/metrics")).json()
    assert m["requests"] == 1
    assert m["messages_handle_ized"] == 0
    assert m["messages_rehydrated"] == 0
    assert m["chars_saved"] == 0


@pytest.mark.asyncio
async def test_metrics_accumulates_across_requests(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    await _post(app, {"messages": [{"role": "user", "content": BULKY}]})
    await _post(app, {"messages": [{"role": "user", "content": BULKY}]})
    m = (await _get(app, "/metrics")).json()
    assert m["requests"] == 2
    assert m["messages_in"] == 2


@pytest.mark.asyncio
async def test_metrics_does_not_touch_upstream(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    await _get(app, "/metrics")
    assert upstream.call_count == 0
    assert upstream.stream_count == 0


@pytest.mark.asyncio
async def test_metrics_reports_token_estimates_and_summary(make_app):
    # Stage 8.0: /metrics surfaces an approximate token view + a human-readable summary.
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    await _post(app, {"messages": [{"role": "user", "content": BULKY}]})
    m = (await _get(app, "/metrics")).json()
    assert m["tokens_in_est"] == m["chars_in"] // 4
    assert m["tokens_out_est"] == m["chars_out"] // 4
    assert m["tokens_saved_est"] == m["chars_saved"] // 4
    assert m["peak_prompt_tokens_est"] == m["chars_out"] // 4  # one request -> peak == it
    assert isinstance(m["pct_saved"], (int, float))
    assert isinstance(m["summary"], str) and "tokens" in m["summary"]


@pytest.mark.asyncio
async def test_metrics_summary_empty_before_any_request(make_app):
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    m = (await _get(app, "/metrics")).json()
    assert m["summary"] == "no requests yet"
    assert m["tokens_saved_est"] == 0


@pytest.mark.asyncio
async def test_metrics_includes_retrieval_block(make_app):
    # Phase 7 Stage 1: /metrics merges the shared store's retrieval-path counters.
    app, upstream, store, rewriter = make_app(response=FAKE_RESPONSE)
    # One bulky message -> one note paged out -> corpus_size == 1. Since Phase 10
    # the proxy ITSELF searches (Pass-4 auto-recall probes the store each request),
    # so the retrieval counters are now exercised by the hot path.
    await _post(app, {"messages": [{"role": "user", "content": BULKY}]})
    m = (await _get(app, "/metrics")).json()
    assert "retrieval" in m
    r = m["retrieval"]
    assert r["corpus_size"] == 1
    assert r["search_calls"] >= 1          # Pass-4 auto-recall probes the store
    assert "recall_hit_rate" in r
    assert "avg_search_ms" in r
    assert "slices_recalled" in m          # proxy-level recall counter (Phase 10)
    assert upstream.get_paths == []
