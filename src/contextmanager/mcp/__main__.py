"""``python -m contextmanager.mcp`` — build and run the stdio MCP server.

Config precedence: CLI args > CM_* env vars > defaults. The CLI args matter because
MCP hosts (OpenCode, Hermes) spawn this as a subprocess and do not all forward env
reliably — passing ``--store-root`` in the command guarantees the MCP server shares the
SAME store as the proxy so handles resolve across both surfaces.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Optional

from .config import McpConfig
from .server import build_server


def _env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    return val


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got: {val!r}")


def load_config_from_env() -> McpConfig:
    return McpConfig(
        store_root=_env_str("CM_STORE_ROOT", "./contextstore"),
        upstream_base_url=_env_str("CM_UPSTREAM_BASE_URL"),
        upstream_api_key=_env_str("CM_UPSTREAM_API_KEY"),
        server_name=_env_str("CM_MCP_SERVER_NAME", "context-governor"),
        transport=_env_str("CM_MCP_TRANSPORT", "stdio"),
        default_search_k=_env_int("CM_DEFAULT_SEARCH_K", 5),
        preview_chars=_env_int("CM_PREVIEW_CHARS", 200),
        rehydrate_budget_tokens=_env_int("CM_REHYDRATE_BUDGET_TOKENS", 4000),
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m contextmanager.mcp",
        description="Context Governor — Surface B (cooperative MCP server).",
    )
    p.add_argument("--store-root", help="durable store dir (share with the proxy)")
    p.add_argument("--upstream-base-url", help="llama-server URL for exact tokenization "
                                               "(omit -> offline heuristic counter)")
    p.add_argument("--transport", choices=["stdio", "streamable-http", "sse"],
                   help="MCP transport (default: stdio)")
    return p.parse_args(argv)


def load_config(argv: Optional[list[str]] = None) -> McpConfig:
    """Env-based config with CLI-arg overrides applied on top."""
    config = load_config_from_env()
    args = _parse_args(argv)
    overrides = {}
    if args.store_root is not None:
        overrides["store_root"] = args.store_root
    if args.upstream_base_url is not None:
        overrides["upstream_base_url"] = args.upstream_base_url
    if args.transport is not None:
        overrides["transport"] = args.transport
    return replace(config, **overrides) if overrides else config


def main(argv: Optional[list[str]] = None) -> None:
    config = load_config(argv)
    server = build_server(config)
    try:
        server.run(transport=config.transport)
    finally:
        svc = getattr(server, "_governor_service", None)
        if svc is not None:
            svc.close()


if __name__ == "__main__":
    main()
