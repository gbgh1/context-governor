"""run-governor — the unified launcher + central config front door (Phase 8.1).

One llama-server-style entrypoint that resolves a ProxyConfig from layered sources
(lowest → highest precedence):

    built-in defaults  <  config-file [proxy] table  <  selected provider  <  CLI flags

then runs the proxy. Keeps the system un-scattered: one command instead of a fistful of
CM_* env vars.

    run-governor --provider openai --listen-port 8900
    run-governor --config governor.toml --provider llama
    run-governor --provider ollama --dry-run         # print resolved config, don't run
    run-governor --cli opencode --provider llama      # + print how to wire OpenCode

API keys are NEVER read from flags or the config file body — only from the env var named
by the provider's `api_key_env` (default OPENAI_API_KEY for openai). Anthropic/Claude are
deferred (gated in tasks/plan.md Phase 8).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Optional

from .proxy.config import ProxyConfig
from .wiring import WiringError, apply_wiring, plan_wiring, revert_wiring


class LauncherError(Exception):
    """Bad launcher input — unknown/deferred provider, malformed config, invalid values."""


# Built-in provider profiles (override-able via the config file's [providers.<name>]).
_PROVIDER_DEFAULTS: dict[str, dict] = {
    "llama":  {"upstream_base_url": "http://127.0.0.1:8080",  "api_key_env": None},
    "ollama": {"upstream_base_url": "http://127.0.0.1:11434", "api_key_env": None},
    "openai": {"upstream_base_url": "https://api.openai.com",  "api_key_env": "OPENAI_API_KEY"},
}
# Deferred until the Anthropic adapter + subscription-proxy question are resolved.
_DEFERRED_PROVIDERS = {"anthropic", "claude"}
_KNOWN_CLIS = {"opencode", "hermes"}
_DEFERRED_CLIS = {"claude"}

_PROXY_FIELDS = {f.name for f in dataclasses.fields(ProxyConfig)}


def load_config_file(path: Optional[str]) -> dict:
    """Parse a central TOML config (stdlib ``tomllib`` — no new dependency). Empty dict if
    ``path`` is falsy; raises LauncherError on a missing or malformed file."""
    if not path:
        return {}
    import tomllib

    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise LauncherError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise LauncherError(f"malformed TOML in {path}: {exc}") from exc


def resolve_config(opts: dict, file_cfg: Optional[dict] = None) -> ProxyConfig:
    """Resolve a ProxyConfig from layered sources (see module docstring). The selected
    provider supplies the upstream URL and (from its ``api_key_env``) the API key."""
    file_cfg = file_cfg or {}
    resolved: dict = {}

    # Layer 1: the config file's [proxy] table (known ProxyConfig fields only).
    for key, value in (file_cfg.get("proxy") or {}).items():
        if key in _PROXY_FIELDS:
            resolved[key] = value

    # Layer 2: the selected provider profile (built-in, overlaid by [providers.<name>]).
    provider = opts.get("provider")
    if provider:
        if provider in _DEFERRED_PROVIDERS:
            raise LauncherError(
                f"provider '{provider}' is deferred (Anthropic/Claude not yet supported)"
            )
        if provider not in _PROVIDER_DEFAULTS:
            raise LauncherError(f"unknown provider '{provider}'")
        profile = dict(_PROVIDER_DEFAULTS[provider])
        profile.update((file_cfg.get("providers") or {}).get(provider, {}))
        if profile.get("upstream_base_url"):
            resolved["upstream_base_url"] = profile["upstream_base_url"]
        key_env = profile.get("api_key_env")
        if key_env and os.environ.get(key_env):
            resolved["upstream_api_key"] = os.environ[key_env]

    # Layer 3: explicit CLI flags (any opt whose name is a ProxyConfig field and is set).
    for field in _PROXY_FIELDS:
        if opts.get(field) is not None:
            resolved[field] = opts[field]

    # upstream_base_url is required — default to a local llama-server if nothing set it.
    resolved.setdefault("upstream_base_url", _PROVIDER_DEFAULTS["llama"]["upstream_base_url"])

    try:
        return ProxyConfig(**resolved)
    except (TypeError, ValueError) as exc:
        raise LauncherError(f"invalid configuration: {exc}") from exc


def _redacted_dict(config: ProxyConfig) -> dict:
    data = dataclasses.asdict(config)
    if data.get("upstream_api_key"):
        data["upstream_api_key"] = "***redacted***"  # never print the key
    return data


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run-governor",
        description="Context Governor launcher — one front door for the proxy + config.",
    )
    p.add_argument("--config", help="path to a central TOML config file")
    p.add_argument(
        "--provider",
        choices=sorted(_PROVIDER_DEFAULTS) + sorted(_DEFERRED_PROVIDERS),
        help="upstream provider profile (anthropic/claude deferred)",
    )
    p.add_argument(
        "--cli",
        choices=sorted(_KNOWN_CLIS) + sorted(_DEFERRED_CLIS),
        help="auto-wire this CLI to the governor (backup + insert; reverts on exit)",
    )
    p.add_argument(
        "--cli-config", dest="cli_config",
        help="override the CLI config file to wire (default: per-CLI standard location)",
    )
    p.add_argument("--upstream-url", dest="upstream_base_url", help="override the upstream base URL")
    p.add_argument("--listen-host", dest="listen_host")
    p.add_argument("--listen-port", dest="listen_port", type=int)
    p.add_argument("--store-root", dest="store_root")
    p.add_argument("--handle-threshold-tokens", dest="handle_threshold_tokens", type=int)
    p.add_argument("--handle-threshold-ratio", dest="handle_threshold_ratio", type=float)
    p.add_argument("--context-budget-ratio", dest="context_budget_ratio", type=float)
    p.add_argument("--stub-preview-chars", dest="stub_preview_chars", type=int)
    p.add_argument("--rehydrate-budget-tokens", dest="rehydrate_budget_tokens", type=int)
    p.add_argument("--auto-recall-k", dest="auto_recall_k", type=int)
    p.add_argument("--recall-budget-tokens", dest="recall_budget_tokens", type=int)
    p.add_argument("--request-timeout", dest="request_timeout", type=float)
    p.add_argument("--model-alias", dest="model_alias")
    p.add_argument("--diff-min-similarity", dest="diff_min_similarity", type=float)
    p.add_argument("--diff-lookback", dest="diff_lookback", type=int)
    p.add_argument("--diff-max-chars", dest="diff_max_chars", type=int)
    p.add_argument("--tokenize-max-chars", dest="tokenize_max_chars", type=int)
    p.add_argument("--dry-run", action="store_true", help="print the resolved config and exit")
    p.add_argument("--revert", action="store_true", help="(stage 8.3) undo CLI auto-wiring")
    return p


def main(argv: Optional[list] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    opts = vars(args)
    cli = opts.get("cli")

    if cli in _DEFERRED_CLIS:
        parser.error(f"--cli {cli} is deferred (Claude/Anthropic not yet supported)")

    # --- revert path: restore the CLI config from backup, then exit (no server). ---
    if opts.get("revert"):
        if cli not in _KNOWN_CLIS:
            parser.error("--revert requires --cli {opencode|hermes}")
        reverted = revert_wiring(cli)
        print("run-governor: " + (f"reverted {cli} wiring from backup" if reverted
                                   else f"no wiring state to revert for {cli}"))
        return

    try:
        file_cfg = load_config_file(opts.get("config"))
        config = resolve_config(opts, file_cfg)
    except LauncherError as exc:
        parser.error(str(exc))

    governor_base = f"http://{config.listen_host}:{config.listen_port}/v1"
    store_abs = os.path.abspath(config.store_root)  # the MCP entry needs an absolute store

    if opts.get("dry_run"):
        out = _redacted_dict(config)
        if cli in _KNOWN_CLIS:
            out["_wiring_plan"] = plan_wiring(
                cli, governor_base, store_abs, config_path=opts.get("cli_config")
            )
        json.dump(out, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return

    print(f"run-governor: {config.listen_host}:{config.listen_port} -> {config.upstream_base_url}")

    # PERSISTENT wiring: set the CLI up to use the governor (provider/base_url + MCP) and leave
    # it wired after exit, so "pull repo -> run-governor --cli X" is a one-time setup. Idempotent
    # to re-run; undo with `run-governor --revert --cli X`.
    if cli in _KNOWN_CLIS:
        try:
            state = apply_wiring(cli, governor_base, store_abs, config_path=opts.get("cli_config"))
        except WiringError as exc:
            parser.error(f"wiring {cli} failed: {exc}")
        print(f"wired {cli} (persistent): {state.summary}")
        print(f"  backup: {state.backup_path}   (undo with `run-governor --revert --cli {cli}`)")

    _run(config)


def _run(config: ProxyConfig) -> None:  # pragma: no cover - starts the blocking server
    import uvicorn

    from .proxy.app import create_app

    uvicorn.run(create_app(config), host=config.listen_host, port=config.listen_port)


if __name__ == "__main__":
    main()
