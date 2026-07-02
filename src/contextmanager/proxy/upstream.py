from __future__ import annotations

from typing import Any, AsyncIterator, Optional

import httpx

from .config import ProxyConfig


class UpstreamError(Exception):
    """Raised on transport, HTTP, or decode errors talking to llama-server.

    Carries an optional ``status_code`` (the upstream HTTP status, when the
    failure was an HTTP-level error) so the app layer can map it onto the
    outbound response status.
    """

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class UpstreamClient:
    """Async httpx client to llama-server (stream + non-stream).

    Owns an ``httpx.AsyncClient`` unless one is injected, in which case the
    caller retains ownership (``aclose`` is a no-op then). All errors are
    wrapped in :class:`UpstreamError`; raw ``httpx`` exceptions never leak.
    """

    def __init__(
        self,
        config: ProxyConfig,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._config = config
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(
                base_url=config.upstream_base_url,
                timeout=config.request_timeout,
            )
            self._owns_client = True

    # ------------------------------------------------------------------ helpers

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._config.upstream_api_key:
            h["Authorization"] = f"Bearer {self._config.upstream_api_key}"
        return h

    @staticmethod
    def _raise_for_status(resp: httpx.Response, path: str) -> None:
        if resp.status_code >= 400:
            raise UpstreamError(
                f"HTTP {resp.status_code} from {path}: {resp.text}",
                status_code=resp.status_code,
            )

    @staticmethod
    def _json(resp: httpx.Response, path: str) -> Any:
        try:
            return resp.json()
        except Exception as e:
            raise UpstreamError(f"malformed JSON from {path}: {e}") from e

    # ----------------------------------------------------- chat_completion (non-stream)

    async def chat_completion(self, payload: dict) -> dict:
        """POST /v1/chat/completions (non-stream). Return parsed JSON dict."""
        path = "/v1/chat/completions"
        try:
            resp = await self._client.post(path, json=payload, headers=self._headers())
        except httpx.TransportError as e:
            raise UpstreamError(f"Transport error POST {path}: {e}") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"HTTP error POST {path}: {e}") from e

        self._raise_for_status(resp, path)
        data = self._json(resp, path)
        if not isinstance(data, dict):
            raise UpstreamError(
                f"Unexpected /v1/chat/completions response (not a JSON object): {data!r}"
            )
        return data

    # ----------------------------------------------------- chat_completion_stream

    async def chat_completion_stream(self, payload: dict) -> AsyncIterator[bytes]:
        """POST /v1/chat/completions with stream passthrough.

        The caller is expected to set ``stream=true`` in ``payload``. Yields the
        raw response bytes as they arrive (unbuffered; SSE is forwarded verbatim
        — never parsed or re-serialized). Raises :class:`UpstreamError` on
        transport failure or upstream non-200 (status carried on the error).
        """
        path = "/v1/chat/completions"
        try:
            async with self._client.stream(
                "POST", path, json=payload, headers=self._headers()
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise UpstreamError(
                        f"HTTP {resp.status_code} from {path}: {body.decode('utf-8', errors='replace')}",
                        status_code=resp.status_code,
                    )
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except UpstreamError:
            raise
        except httpx.TransportError as e:
            raise UpstreamError(f"Transport error POST {path}: {e}") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"HTTP error POST {path}: {e}") from e

    # ----------------------------------------------------- passthrough_get

    async def passthrough_get(self, path: str) -> dict:
        """GET ``path`` (e.g. ``/v1/models``, ``/props``); return parsed JSON dict."""
        try:
            resp = await self._client.get(path, headers=self._headers())
        except httpx.TransportError as e:
            raise UpstreamError(f"Transport error GET {path}: {e}") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"HTTP error GET {path}: {e}") from e

        self._raise_for_status(resp, path)
        data = self._json(resp, path)
        if not isinstance(data, dict):
            raise UpstreamError(
                f"Unexpected {path} response (not a JSON object): {data!r}"
            )
        return data

    # ----------------------------------------------------- cleanup

    async def aclose(self) -> None:
        """Close the owned client. No-op when the client was injected."""
        if self._owns_client:
            await self._client.aclose()
