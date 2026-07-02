"""Entrypoint config precedence: CLI args > env > defaults (spec §6 + CLI override)."""

from __future__ import annotations

from contextmanager.mcp.__main__ import load_config


def test_cli_args_override_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CM_STORE_ROOT", "./from-env")
    cfg = load_config([
        "--store-root", str(tmp_path),
        "--upstream-base-url", "http://up:9",
        "--transport", "stdio",
    ])
    assert cfg.store_root == str(tmp_path)          # arg beats env
    assert cfg.upstream_base_url == "http://up:9"
    assert cfg.transport == "stdio"


def test_env_used_when_no_args(monkeypatch):
    monkeypatch.setenv("CM_STORE_ROOT", "./from-env")
    cfg = load_config([])
    assert cfg.store_root == "./from-env"


def test_defaults_when_nothing_set(monkeypatch):
    monkeypatch.delenv("CM_STORE_ROOT", raising=False)
    cfg = load_config([])
    assert cfg.store_root == "./contextstore"
