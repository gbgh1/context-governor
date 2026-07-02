from __future__ import annotations

import os
from typing import Optional

import uvicorn

from .app import create_app
from .config import ProxyConfig


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


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"{key} must be a float, got: {val!r}")


def load_config_from_env() -> ProxyConfig:
    """Build a ProxyConfig from CM_* env vars, falling back to ProxyConfig
    defaults where a var is unset/empty."""
    return ProxyConfig(
        upstream_base_url=_env_str("CM_UPSTREAM_BASE_URL", "http://127.0.0.1:8080"),
        store_root=_env_str("CM_STORE_ROOT", "./contextstore"),
        upstream_api_key=_env_str("CM_UPSTREAM_API_KEY"),
        listen_host=_env_str("CM_LISTEN_HOST", "127.0.0.1"),
        listen_port=_env_int("CM_LISTEN_PORT", 8900),
        handle_threshold_tokens=_env_int("CM_HANDLE_THRESHOLD_TOKENS", 2000),
        handle_threshold_ratio=_env_float("CM_HANDLE_THRESHOLD_RATIO", 0.02),
        context_budget_ratio=_env_float("CM_CONTEXT_BUDGET_RATIO", 0.50),
        stub_preview_chars=_env_int("CM_STUB_PREVIEW_CHARS", 200),
        rehydrate_budget_tokens=_env_int("CM_REHYDRATE_BUDGET_TOKENS", 4000),
        auto_recall_k=_env_int("CM_AUTO_RECALL_K", 3),
        recall_budget_tokens=_env_int("CM_RECALL_BUDGET_TOKENS", 1500),
        request_timeout=_env_float("CM_REQUEST_TIMEOUT", 300.0),
        model_alias=_env_str("CM_MODEL_ALIAS", "context-governor"),
        diff_min_similarity=_env_float("CM_DIFF_MIN_SIMILARITY", 0.5),
        diff_lookback=_env_int("CM_DIFF_LOOKBACK", 6),
    )


def main() -> None:
    config = load_config_from_env()
    uvicorn.run(
        create_app(config),
        host=config.listen_host,
        port=config.listen_port,
    )


if __name__ == "__main__":
    main()
