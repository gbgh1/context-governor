"""LlamaServerTokenCounter tests against a MOCKED httpx transport.

Binds to `tasks/phase1-spec.md` §7 (NORMATIVE). No real network: every request is
intercepted by `httpx.MockTransport` and answered by a deterministic handler.

Endpoints exercised:
  - POST /tokenize                                   (count_text, add_special toggled)
  - POST /tokenize  {with_pieces: true}              (truncate_to_tokens)
  - POST /v1/chat/completions/input_tokens            (count_messages primary)
  - POST /apply-template                              (count_messages fallback)
  - GET  /props                                       (n_ctx)
  - HTTP 500 -> TokenizerError
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from contextmanager.tokenizer import LlamaServerTokenCounter, TokenizerError
from contextmanager.types import Message


BASE_URL = "http://test-llama"


# ---------------------------------------------------------------------------
# Helpers: build a counter with a mocked client
# ---------------------------------------------------------------------------


def make_counter(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str | None = None,
) -> LlamaServerTokenCounter:
    """Construct a LlamaServerTokenCounter wired to a MockTransport handler."""
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return LlamaServerTokenCounter(
        base_url=BASE_URL,
        api_key=api_key,
        client=client,
    )


def json_resp(data: Any, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=json.dumps(data),
        headers={"content-type": "application/json"},
    )


class CallLog:
    """Records request paths + methods + parsed JSON bodies, in order."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(self, request: httpx.Request) -> dict[str, Any]:
        try:
            body = json.loads(request.content) if request.content else None
        except Exception:
            body = request.content
        entry = {
            "method": request.method,
            "path": request.url.path,
            "url": str(request.url),
            "body": body,
        }
        self.calls.append(entry)
        return entry


# ---------------------------------------------------------------------------
# count_text
# ---------------------------------------------------------------------------


def test_count_text_posts_to_tokenize_and_returns_token_count() -> None:
    log = CallLog()

    def handler(request: httpx.Request) -> httpx.Response:
        log.record(request)
        assert request.method == "POST"
        assert request.url.path == "/tokenize"
        body = log.calls[-1]["body"]
        assert body["content"] == "hello world foo"
        assert body["add_special"] is False
        # 3 tokens for a 3-word sentence.
        return json_resp({"tokens": [10, 20, 30]})

    counter = make_counter(handler)
    assert counter.count_text("hello world foo") == 3


def test_count_text_empty_string_returns_zero_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_resp({"tokens": []})

    counter = make_counter(handler)
    assert counter.count_text("") == 0


# ---------------------------------------------------------------------------
# count_messages — primary path
# ---------------------------------------------------------------------------


def test_count_messages_primary_returns_input_tokens() -> None:
    log = CallLog()

    messages = [
        Message(role="user", content="hi", id="m1"),
        Message(role="assistant", content="hello there", id="m2"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        entry = log.record(request)
        if request.url.path == "/v1/chat/completions/input_tokens":
            assert request.method == "POST"
            assert entry["body"] == {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello there"},
                ]
            }
            return json_resp({"input_tokens": 42})
        # /tokenize should NOT be called on the primary path.
        pytest.fail(f"unexpected call to {request.url.path}")

    counter = make_counter(handler)
    assert counter.count_messages(messages) == 42
    # Path is cached as primary.
    assert counter._messages_path == "primary"

    # Second call must reuse the primary path (no fallback probe).
    n_before = len(log.calls)
    assert counter.count_messages(messages) == 42
    assert len(log.calls) == n_before + 1  # exactly one more /v1/.../input_tokens


# ---------------------------------------------------------------------------
# count_messages — 404 fallback path (then /apply-template + /tokenize)
# ---------------------------------------------------------------------------


def test_count_messages_falls_back_on_404_to_apply_template_then_tokenize() -> None:
    log = CallLog()

    messages = [Message(role="user", content="hi there", id="m1")]
    # The fallback reuses /tokenize with add_special=True on the prompt returned by
    # /apply-template. The prompt here is "rendered prompt body" (3 words).
    apply_template_prompt = "rendered prompt body"

    def handler(request: httpx.Request) -> httpx.Response:
        entry = log.record(request)
        if request.url.path == "/v1/chat/completions/input_tokens":
            # Server signals the route is unavailable.
            return httpx.Response(404, json={"error": "not found"})
        if request.url.path == "/apply-template":
            assert request.method == "POST"
            assert entry["body"] == {
                "messages": [{"role": "user", "content": "hi there"}]
            }
            return json_resp({"prompt": apply_template_prompt})
        if request.url.path == "/tokenize":
            # Fallback path must call count_text with add_special=True.
            assert entry["body"]["content"] == apply_template_prompt
            assert entry["body"]["add_special"] is True
            return json_resp({"tokens": [1, 2, 3]})  # 3 tokens
        pytest.fail(f"unexpected call to {request.url.path}")

    counter = make_counter(handler)
    assert counter.count_messages(messages) == 3
    # The fallback path is now cached; subsequent calls skip the 404 probe.
    assert counter._messages_path == "fallback"

    # Verify call sequence: input_tokens (404) -> apply-template -> tokenize.
    paths = [c["path"] for c in log.calls]
    assert paths[0] == "/v1/chat/completions/input_tokens"
    assert paths[1] == "/apply-template"
    assert paths[2] == "/tokenize"

    # Second call: should go straight to /apply-template (cached), NOT probe 404 again.
    log.calls.clear()
    def handler2(request: httpx.Request) -> httpx.Response:
        entry = log.record(request)
        if request.url.path == "/apply-template":
            return json_resp({"prompt": apply_template_prompt})
        if request.url.path == "/tokenize":
            assert entry["body"]["add_special"] is True
            return json_resp({"tokens": [1, 2, 3]})
        pytest.fail(f"unexpected cached-path call to {request.url.path}")

    counter2 = make_counter(handler2)
    # Re-prime cache to avoid re-probing (simulate post-404 state).
    counter2._messages_path = "fallback"  # type: ignore[attr-defined]
    assert counter2.count_messages(messages) == 3
    assert [c["path"] for c in log.calls] == ["/apply-template", "/tokenize"]


# ---------------------------------------------------------------------------
# truncate_to_tokens
# ---------------------------------------------------------------------------


def test_truncate_to_tokens_returns_text_unchanged_when_already_within_limit() -> None:
    # Short text: count_text returns 2, max_tokens=5 -> no /tokenize?with_pieces call.
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        # count_text first (no with_pieces). Should not be asked for pieces.
        assert request.url.path == "/tokenize"
        body = json.loads(request.content)
        assert body.get("with_pieces", False) is False
        return json_resp({"tokens": [1, 2]})

    counter = make_counter(handler)
    result = counter.truncate_to_tokens("alpha beta", 5)
    assert result == "alpha beta"
    # Only the count_text probe happened; no with_pieces call.
    assert calls == ["/tokenize"]


def test_truncate_to_tokens_reassembles_from_string_pieces() -> None:
    # 5-word text; pieces are detokenized fragments WITH spacing embedded
    # (real llama-server pieces carry leading/trailing spaces).
    pieces = ["alpha ", "beta ", "gamma ", "delta ", "epsilon"]
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        entry = {"path": request.url.path, "body": json.loads(request.content)}
        calls.append(entry)
        if request.url.path == "/tokenize":
            if entry["body"].get("with_pieces") is True:
                # pieces path
                return json_resp({"tokens": list(range(5)), "pieces": pieces})
            # count_text probe: report 5 tokens (over max=3) to force the pieces path.
            return json_resp({"tokens": list(range(5))})
        pytest.fail("unexpected path")

    counter = make_counter(handler)
    result = counter.truncate_to_tokens("alpha beta gamma delta epsilon", 3)
    # First 3 pieces rejoined with NO separator: "alpha " + "beta " + "gamma "
    assert result == "alpha beta gamma "
    # The measured token count of the result is <= max_tokens.
    # (count_text is mocked to count words; "alpha beta gamma " -> 3 words.)
    # Re-verify by re-asking the counter via count_text: 3 words.
    assert result.split() == ["alpha", "beta", "gamma"]


def test_truncate_to_tokens_handles_byte_array_piece_case() -> None:
    # Mix of str pieces and one byte-array piece (list[int] of utf-8 bytes).
    # pieces: "alpha " (str), [98,101,116,97] ("beta" as bytes), " gamma" (str),
    # " delta" (str), " epsilon" (str).
    # Truncating to 2 pieces -> "alpha " + bytes("beta") -> "alpha beta".
    pieces: list[Any] = ["alpha ", [98, 101, 116, 97], " gamma", " delta", " epsilon"]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            body = json.loads(request.content)
            if body.get("with_pieces") is True:
                return json_resp({"tokens": list(range(5)), "pieces": pieces})
            return json_resp({"tokens": list(range(5))})  # count_text probe: 5 tokens
        pytest.fail("unexpected path")

    counter = make_counter(handler)
    result = counter.truncate_to_tokens("alpha beta gamma delta epsilon", 2)
    # First 2 pieces: "alpha " + decode([98,101,116,97]) = "alpha " + "beta" = "alpha beta"
    assert result == "alpha beta"
    assert result.split() == ["alpha", "beta"]


def test_truncate_to_tokens_all_byte_array_pieces() -> None:
    # All-byte-array path: the implementation joins bytes and decodes once.
    pieces: list[Any] = [
        [104, 101, 108, 108, 111],   # "hello"
        [32, 119, 111, 114, 108, 100],  # " world"
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            body = json.loads(request.content)
            if body.get("with_pieces") is True:
                return json_resp({"tokens": list(range(2)), "pieces": pieces})
            return json_resp({"tokens": list(range(2))})
        pytest.fail("unexpected path")

    counter = make_counter(handler)
    result = counter.truncate_to_tokens("hello world", 2)
    assert result == "hello world"


def test_truncate_to_tokens_real_llama_server_shape() -> None:
    # REGRESSION (found LIVE, 2026-07-01): real llama-server returns pieces
    # NESTED PER-TOKEN — {"tokens": [{"id": N, "piece": <str|list[int]>}, ...]} —
    # not a top-level "pieces" list (the shape this client originally assumed
    # and every earlier test mocked). The first live caller of truncate_to_tokens
    # (Phase 10 auto-recall) hit a TokenizerError -> 500 on every request.
    tokens = [
        {"id": 7676, "piece": "alpha "},
        {"id": 5117, "piece": "beta "},
        {"id": 420, "piece": [103, 97, 109, 109, 97]},  # "gamma" as raw bytes
        {"id": 2959, "piece": " delta"},
        {"id": 731, "piece": " epsilon"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            body = json.loads(request.content)
            if body.get("with_pieces") is True:
                return json_resp({"tokens": tokens})  # the real wire shape
            return json_resp({"tokens": list(range(5))})  # count_text probe
        pytest.fail("unexpected path")

    counter = make_counter(handler)
    result = counter.truncate_to_tokens("alpha beta gamma delta epsilon", 3)
    assert result == "alpha beta gamma"
    assert result.split() == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# n_ctx()
# ---------------------------------------------------------------------------


def test_n_ctx_reads_default_generation_settings_n_ctx_from_props() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/props"
        return json_resp({
            "default_generation_settings": {"n_ctx": 8192, "temperature": 0.7},
            "other": "ignored",
        })

    counter = make_counter(handler)
    assert counter.n_ctx() == 8192


# ---------------------------------------------------------------------------
# Error path: HTTP 500 -> TokenizerError
# ---------------------------------------------------------------------------


def test_tokenizer_error_raised_on_http_500_for_count_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal server error")

    counter = make_counter(handler)
    with pytest.raises(TokenizerError):
        counter.count_text("anything")


def test_tokenizer_error_raised_on_http_500_for_n_ctx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    counter = make_counter(handler)
    with pytest.raises(TokenizerError):
        counter.n_ctx()


def test_tokenizer_error_raised_on_http_500_for_apply_template_in_fallback() -> None:
    # primary 404s -> fallback hits /apply-template which 500s -> TokenizerError.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions/input_tokens":
            return httpx.Response(404, json={"error": "no route"})
        if request.url.path == "/apply-template":
            return httpx.Response(500, text="apply-template broken")
        pytest.fail(f"unexpected {request.url.path}")

    counter = make_counter(handler)
    with pytest.raises(TokenizerError):
        counter.count_messages([Message(role="user", content="x", id="m1")])


def test_tokenizer_error_raised_on_http_500_for_input_tokens_non_404() -> None:
    # A 500 (not 404) on the primary route must raise, not fall back.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions/input_tokens":
            return httpx.Response(500, text="primary broken")
        pytest.fail("fallback should not be tried on non-404 errors")

    counter = make_counter(handler)
    with pytest.raises(TokenizerError):
        counter.count_messages([Message(role="user", content="x", id="m1")])


# ---------------------------------------------------------------------------
# Transport error path
# ---------------------------------------------------------------------------


def test_tokenizer_error_raised_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    counter = make_counter(handler)
    with pytest.raises(TokenizerError):
        counter.count_text("anything")


# ===========================================================================
# Round-2 correction §10.5 / §10.7 — malformed JSON raises (fixes H3)
# ===========================================================================
#
# Every resp.json() call must be wrapped so a 200 response with a non-JSON
# body (e.g. an HTML error page from a misconfigured proxy) raises a typed
# TokenizerError naming the offending path, instead of leaking a raw
# json.JSONDecodeError.
# ===========================================================================


HTML_BODY = (
    "<!DOCTYPE html><html><head><title>Bad Gateway</title></head>"
    "<body><h1>502 Bad Gateway</h1><p>upstream returned non-JSON</p></body></html>"
)


def test_tokenizer_malformed_json_raises_for_count_text() -> None:
    """A 200 response with non-JSON body (HTML) on /tokenize must cause
    count_text to raise TokenizerError, not a raw json.JSONDecodeError."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tokenize"
        return httpx.Response(
            200,
            content=HTML_BODY,
            headers={"content-type": "text/html"},
        )

    counter = make_counter(handler)
    with pytest.raises(TokenizerError):
        counter.count_text("hello world")


def test_tokenizer_malformed_json_raises_for_count_messages_primary() -> None:
    """Same contract for the count_messages primary route
    (/v1/chat/completions/input_tokens): a 200 with HTML body must raise
    TokenizerError."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions/input_tokens":
            return httpx.Response(
                200,
                content=HTML_BODY,
                headers={"content-type": "text/html"},
            )
        pytest.fail(f"unexpected call to {request.url.path}")

    counter = make_counter(handler)
    with pytest.raises(TokenizerError):
        counter.count_messages([Message(role="user", content="x", id="m1")])


def test_tokenizer_malformed_json_raises_for_truncate_to_tokens() -> None:
    """truncate_to_tokens's pieces path also parses JSON; a non-JSON 200
    response must raise TokenizerError."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            body = json.loads(request.content)
            if body.get("with_pieces") is True:
                # Overshoot the count probe so the pieces path is exercised.
                return httpx.Response(
                    200,
                    content=HTML_BODY,
                    headers={"content-type": "text/html"},
                )
            # First the count_text probe: report 5 tokens (> max=3) to force
            # the with_pieces path.
            return json_resp({"tokens": list(range(5))})
        pytest.fail(f"unexpected path {request.url.path}")

    counter = make_counter(handler)
    with pytest.raises(TokenizerError):
        counter.truncate_to_tokens("alpha beta gamma delta epsilon", 3)
