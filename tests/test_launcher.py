"""Phase 8.1 — run-governor launcher + central config resolution.

Tests the pure resolution layers (defaults < config file < provider < flags), provider
profiles, env-only API keys, deferral gating, and the --dry-run / error paths of main().
"""

from __future__ import annotations

import json

import pytest

from contextmanager.launcher import (
    LauncherError,
    load_config_file,
    main,
    resolve_config,
)

_FLAG_FIELDS = [
    "upstream_base_url", "listen_host", "listen_port", "store_root",
    "handle_threshold_tokens", "handle_threshold_ratio", "context_budget_ratio",
    "stub_preview_chars", "rehydrate_budget_tokens", "request_timeout", "model_alias",
    "diff_min_similarity", "diff_lookback", "diff_max_chars", "tokenize_max_chars",
]


def _opts(**over) -> dict:
    base = {k: None for k in (["config", "provider", "cli"] + _FLAG_FIELDS)}
    base["dry_run"] = False
    base["revert"] = False
    base.update(over)
    return base


# --------------------------------------------------------------------------- layering


def test_default_is_local_llama() -> None:
    cfg = resolve_config(_opts(), {})
    assert cfg.upstream_base_url == "http://127.0.0.1:8080"
    assert cfg.listen_port == 8900  # ProxyConfig default flows through


def test_provider_base_urls() -> None:
    assert resolve_config(_opts(provider="llama"), {}).upstream_base_url == "http://127.0.0.1:8080"
    assert resolve_config(_opts(provider="ollama"), {}).upstream_base_url == "http://127.0.0.1:11434"
    assert resolve_config(_opts(provider="openai"), {}).upstream_base_url == "https://api.openai.com"


def test_openai_api_key_only_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    cfg = resolve_config(_opts(provider="openai"), {})
    assert cfg.upstream_api_key == "sk-test-123"


def test_flag_beats_provider_beats_config_beats_default() -> None:
    file_cfg = {"proxy": {"listen_port": 9000, "handle_threshold_tokens": 1234}}
    cfg = resolve_config(
        _opts(provider="llama", upstream_base_url="http://x:1", listen_port=9999), file_cfg
    )
    assert cfg.upstream_base_url == "http://x:1"      # flag beats provider
    assert cfg.listen_port == 9999                    # flag beats config file
    assert cfg.handle_threshold_tokens == 1234        # config file beats default


def test_config_provider_table_overrides_builtin() -> None:
    file_cfg = {"providers": {"llama": {"upstream_base_url": "http://10.0.0.5:8080"}}}
    cfg = resolve_config(_opts(provider="llama"), file_cfg)
    assert cfg.upstream_base_url == "http://10.0.0.5:8080"


def test_unknown_proxy_keys_in_config_are_ignored() -> None:
    cfg = resolve_config(_opts(), {"proxy": {"bogus_key": 1, "listen_port": 8131}})
    assert cfg.listen_port == 8131  # known applied, unknown silently dropped


# --------------------------------------------------------------------------- gating


def test_deferred_provider_raises() -> None:
    with pytest.raises(LauncherError):
        resolve_config(_opts(provider="anthropic"), {})


def test_invalid_value_becomes_launcher_error() -> None:
    with pytest.raises(LauncherError):
        resolve_config(_opts(listen_port=0), {})  # ProxyConfig rejects port 0


# --------------------------------------------------------------------------- config file


def test_load_config_file_missing_raises(tmp_path) -> None:
    with pytest.raises(LauncherError):
        load_config_file(str(tmp_path / "nope.toml"))


def test_load_config_file_parses(tmp_path) -> None:
    p = tmp_path / "g.toml"
    p.write_text("[proxy]\nlisten_port = 8123\n")
    assert load_config_file(str(p))["proxy"]["listen_port"] == 8123


def test_load_config_file_empty_path() -> None:
    assert load_config_file(None) == {}


# --------------------------------------------------------------------------- main()


def test_main_dry_run_prints_redacted_key(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    main(["--provider", "openai", "--dry-run"])
    data = json.loads(capsys.readouterr().out)
    assert data["upstream_base_url"] == "https://api.openai.com"
    assert data["upstream_api_key"] == "***redacted***"   # key never printed


def test_main_deferred_provider_exits() -> None:
    with pytest.raises(SystemExit):
        main(["--provider", "claude", "--dry-run"])


def test_main_deferred_cli_exits() -> None:
    with pytest.raises(SystemExit):
        main(["--cli", "claude", "--dry-run"])
