from __future__ import annotations

from typing import Any, Optional

import httpx

from .types import Message


class TokenizerError(Exception):
    """Raised on transport or HTTP errors talking to llama-server."""


class _RouteNotFound(Exception):
    """Internal sentinel: the primary /v1/chat/completions/input_tokens route
    returned HTTP 404. Raised inside _count_messages_primary and caught in
    count_messages to trigger the /apply-template fallback. NOT a public API."""


class LlamaServerTokenCounter:
    """TokenCounter implementation backed by llama-server HTTP endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None
        # Which count_messages route works: None=unknown, "primary", "fallback".
        self._messages_path: Optional[str] = None

    # ------------------------------------------------------------------ helpers

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = self._client.post(url, json=body, headers=self._headers())
        except httpx.TransportError as e:
            raise TokenizerError(f"Transport error POST {url}: {e}") from e
        except httpx.HTTPError as e:
            raise TokenizerError(f"HTTP error POST {url}: {e}") from e
        return resp

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = self._client.get(url, headers=self._headers())
        except httpx.TransportError as e:
            raise TokenizerError(f"Transport error GET {url}: {e}") from e
        except httpx.HTTPError as e:
            raise TokenizerError(f"HTTP error GET {url}: {e}") from e
        return resp

    @staticmethod
    def _raise_for_status(resp: httpx.Response, path: str) -> None:
        if resp.status_code >= 400:
            raise TokenizerError(
                f"HTTP {resp.status_code} from {path}: {resp.text[:200]}"
            )

    @staticmethod
    def _json(resp: httpx.Response, path: str) -> Any:
        # §10.5 H3: wrap every resp.json() in try/except -> raise a typed
        # TokenizerError naming the offending route, instead of leaking a bare
        # json.JSONDecodeError (or any other decoding exception) to callers.
        try:
            return resp.json()
        except Exception as e:
            raise TokenizerError(
                f"malformed JSON from {path}: {e}"
            ) from e

    # ----------------------------------------------------------------- count_text

    def _count_text(self, text: str, add_special: bool) -> int:
        resp = self._post(
            "/tokenize",
            {"content": text, "add_special": add_special},
        )
        self._raise_for_status(resp, "/tokenize")
        data = self._json(resp, "/tokenize")
        tokens = data.get("tokens")
        if not isinstance(tokens, list):
            raise TokenizerError(
                f"Unexpected /tokenize response (no tokens list): {data!r}"
            )
        return len(tokens)

    def count_text(self, text: str) -> int:
        return self._count_text(text, add_special=False)

    # ------------------------------------------------------------ count_messages

    def _messages_payload(self, messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _count_messages_primary(self, messages: list[Message]) -> int:
        resp = self._post(
            "/v1/chat/completions/input_tokens",
            {"messages": self._messages_payload(messages)},
        )
        # §10.5 M1: do NOT use a numeric sentinel for the 404 fallback. Raise a
        # private exception caught by count_messages; this keeps the success path
        # (a real int token count, including 0) distinguishable from "route missing".
        if resp.status_code == 404:
            raise _RouteNotFound()
        self._raise_for_status(resp, "/v1/chat/completions/input_tokens")
        data = self._json(resp, "/v1/chat/completions/input_tokens")
        it = data.get("input_tokens")
        if not isinstance(it, int):
            raise TokenizerError(
                f"Unexpected input_tokens response (no int input_tokens): {data!r}"
            )
        return it

    def _count_messages_fallback(self, messages: list[Message]) -> int:
        resp = self._post(
            "/apply-template",
            {"messages": self._messages_payload(messages)},
        )
        self._raise_for_status(resp, "/apply-template")
        data = self._json(resp, "/apply-template")
        prompt = data.get("prompt")
        if not isinstance(prompt, str):
            raise TokenizerError(
                f"Unexpected /apply-template response (no prompt string): {data!r}"
            )
        return self._count_text(prompt, add_special=True)

    def count_messages(self, messages: list[Message]) -> int:
        # Use cached path if known.
        if self._messages_path == "primary":
            return self._count_messages_primary(messages)
        if self._messages_path == "fallback":
            return self._count_messages_fallback(messages)
        # Unknown: try primary; on 404 (_RouteNotFound) fall back to /apply-template.
        # §10.5 M1: only cache the path AFTER the chosen route returns successfully,
        # so a transient failure does not pin us to a broken route.
        try:
            n = self._count_messages_primary(messages)
        except _RouteNotFound:
            n = self._count_messages_fallback(messages)
            self._messages_path = "fallback"
            return n
        self._messages_path = "primary"
        return n

    # ----------------------------------------------------------- truncate_to_tokens

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        # If already within limit, return unchanged.
        if self.count_text(text) <= max_tokens:
            return text
        resp = self._post(
            "/tokenize",
            {"content": text, "with_pieces": True},
        )
        self._raise_for_status(resp, "/tokenize")
        data = self._json(resp, "/tokenize")
        # Piece extraction. REAL llama-server (verified live, 2026-07-01) returns
        #   {"tokens": [{"id": <int>, "piece": <str | list[int]>}, ...]}
        # for with_pieces=true — pieces are nested PER-TOKEN, and a piece that is
        # not valid UTF-8 arrives as a list of byte values. A top-level "pieces"
        # list (the shape this client originally assumed; never observed live) is
        # still accepted as a tolerant fallback for server variants.
        pieces = data.get("pieces")
        if not isinstance(pieces, list):
            tokens = data.get("tokens")
            if isinstance(tokens, list) and all(
                isinstance(t, dict) and "piece" in t for t in tokens
            ):
                pieces = [t["piece"] for t in tokens]
        if not isinstance(pieces, list):
            raise TokenizerError(
                f"Unexpected /tokenize response (no pieces list): {data!r}"
            )
        selected = pieces[:max_tokens]

        # Reassemble: pieces may be byte arrays (list[int]) or strings.
        # §10.5 M2: validate each byte-list element is an int before bytes(...);
        # on a malformed piece raise TokenizerError rather than coercing silently.
        byte_parts: list[bytes] = []
        all_bytes = True
        for piece in selected:
            if isinstance(piece, list):
                for b in piece:
                    if not isinstance(b, int):
                        raise TokenizerError(
                            f"malformed piece (non-int byte element): {piece!r}"
                        )
                byte_parts.append(bytes(int(b) & 0xFF for b in piece))
            else:
                all_bytes = False
                break
        if all_bytes and byte_parts:
            return b"".join(byte_parts).decode("utf-8", errors="replace")

        out: list[str] = []
        for piece in selected:
            if isinstance(piece, list):
                for b in piece:
                    if not isinstance(b, int):
                        raise TokenizerError(
                            f"malformed piece (non-int byte element): {piece!r}"
                        )
                out.append(bytes(int(b) & 0xFF for b in piece).decode("utf-8", errors="replace"))
            elif isinstance(piece, str):
                out.append(piece)
            else:
                raise TokenizerError(f"Unrecognized piece type: {type(piece)!r}")
        return "".join(out)

    # ----------------------------------------------------------------------- n_ctx

    def n_ctx(self) -> int:
        resp = self._get("/props")
        self._raise_for_status(resp, "/props")
        data = self._json(resp, "/props")
        try:
            return int(data["default_generation_settings"]["n_ctx"])
        except (KeyError, TypeError, ValueError) as e:
            raise TokenizerError(
                f"Unexpected /props response (no n_ctx): {data!r}"
            ) from e

    # --------------------------------------------------------------------- cleanup

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "LlamaServerTokenCounter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
