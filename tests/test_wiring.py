"""Phase 8.3 — CLI auto-wiring: discovery, provider + MCP insert, backup, revert.

All tests run against SYNTHETIC config files in tmp dirs — never a real CLI config.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from contextmanager.launcher import main
from contextmanager.wiring import (
    WiringError,
    _parse_opencode_paths,
    apply_wiring,
    plan_wiring,
    revert_wiring,
)

GOV = "http://127.0.0.1:8900/v1"
STORE = str(Path("/proj/contextstore"))
MCP_ARGS = ["-m", "contextmanager.mcp", "--store-root", STORE]


# --------------------------------------------------------------------- discovery

def test_parse_opencode_paths() -> None:
    out = (
        "home       C:\\Users\\me\n"
        "config     C:\\Users\\me\\.config\\opencode\n"
        "state      C:\\Users\\me\\.local\\state\\opencode\n"
    )
    assert _parse_opencode_paths(out) == "C:\\Users\\me\\.config\\opencode"
    assert _parse_opencode_paths("no config line here") is None


# --------------------------------------------------------------------- OpenCode (JSON)

def _opencode_cfg(tmp_path: Path, body=None) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(body if body is not None else {"provider": {}}, indent=2))
    return p


def test_opencode_apply_inserts_provider_and_mcp_and_backs_up(tmp_path: Path) -> None:
    cfg = _opencode_cfg(tmp_path, {"provider": {"existing": {"x": 1}}})
    sdir = tmp_path / "state"
    original = cfg.read_text()
    st = apply_wiring("opencode", GOV, STORE, config_path=str(cfg), sdir=sdir)
    data = json.loads(cfg.read_text())
    gov = data["provider"]["contextgovernor"]
    assert gov["options"]["baseURL"] == GOV
    assert "context-governor" in gov["models"]              # a usable model is present
    mcp = data["mcp"]["context-governor"]                   # MCP (Surface B) wired too
    assert mcp["command"] == [sys.executable] + MCP_ARGS
    assert mcp["type"] == "local" and mcp["enabled"] is True
    assert data["provider"]["existing"] == {"x": 1}         # other providers untouched
    assert Path(st.backup_path).read_text() == original     # backup holds the original
    assert (sdir / "opencode.json").is_file()


def test_opencode_apply_is_idempotent(tmp_path: Path) -> None:
    cfg = _opencode_cfg(tmp_path)
    sdir = tmp_path / "state"
    st1 = apply_wiring("opencode", GOV, STORE, config_path=str(cfg), sdir=sdir)
    after1 = cfg.read_text()
    backups1 = sorted((sdir / "backups").glob("*"))
    st2 = apply_wiring("opencode", GOV, STORE, config_path=str(cfg), sdir=sdir)
    assert cfg.read_text() == after1                        # no further change
    assert st2.backup_path == st1.backup_path               # no second backup
    assert sorted((sdir / "backups").glob("*")) == backups1


def test_opencode_revert_restores_original_verbatim(tmp_path: Path) -> None:
    cfg = _opencode_cfg(tmp_path, {"provider": {"existing": {"x": 1}}})
    sdir = tmp_path / "state"
    original = cfg.read_text()
    apply_wiring("opencode", GOV, STORE, config_path=str(cfg), sdir=sdir)
    assert "contextgovernor" in cfg.read_text()
    assert revert_wiring("opencode", sdir=sdir) is True
    assert cfg.read_text() == original                      # verbatim restore (mcp gone too)
    assert not (sdir / "opencode.json").is_file()


def test_revert_with_nothing_to_do(tmp_path: Path) -> None:
    assert revert_wiring("opencode", sdir=tmp_path / "state") is False


def test_apply_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(WiringError):
        apply_wiring("opencode", GOV, STORE, config_path=str(tmp_path / "nope.json"),
                     sdir=tmp_path / "s")


def test_already_wired_without_state_raises(tmp_path: Path) -> None:
    # Both provider AND mcp already match -> "already wired"; with no state file that is
    # ambiguous (can't know the true original) -> refuse rather than back up the wired copy.
    cfg = _opencode_cfg(tmp_path, {
        "provider": {"contextgovernor": {"options": {"baseURL": GOV}}},
        "mcp": {"context-governor": {"command": [sys.executable] + MCP_ARGS}},
    })
    with pytest.raises(WiringError):
        apply_wiring("opencode", GOV, STORE, config_path=str(cfg), sdir=tmp_path / "s")


def test_plan_does_not_modify(tmp_path: Path) -> None:
    cfg = _opencode_cfg(tmp_path)
    before = cfg.read_text()
    text = plan_wiring("opencode", GOV, STORE, config_path=str(cfg))
    assert "opencode" in text and "contextgovernor" in text
    assert cfg.read_text() == before                        # planning never writes


def test_plan_missing_config(tmp_path: Path) -> None:
    assert "would FAIL" in plan_wiring("opencode", GOV, STORE,
                                       config_path=str(tmp_path / "nope.json"))


# --------------------------------------------------------------------- Hermes (YAML)

yaml = pytest.importorskip("yaml")


def _hermes_cfg(tmp_path: Path, body) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def test_hermes_repoints_base_url_and_wires_mcp_and_reverts(tmp_path: Path) -> None:
    cfg = _hermes_cfg(tmp_path, {"model": {"base_url": "http://old:1/v1", "name": "x"},
                                 "other": {"keep": True}})
    sdir = tmp_path / "state"
    original = cfg.read_text()
    apply_wiring("hermes", GOV, STORE, config_path=str(cfg), sdir=sdir)
    data = yaml.safe_load(cfg.read_text())
    assert data["model"]["base_url"] == GOV
    assert data["model"]["name"] == "x"                     # sibling keys preserved
    assert data["other"] == {"keep": True}                  # other sections preserved
    srv = data["mcp_servers"]["context-governor"]           # MCP (Surface B) wired
    assert srv["command"] == sys.executable and srv["args"] == MCP_ARGS
    assert revert_wiring("hermes", sdir=sdir) is True
    assert cfg.read_text() == original                       # verbatim restore


def test_hermes_idempotent(tmp_path: Path) -> None:
    cfg = _hermes_cfg(tmp_path, {"model": {"base_url": "http://old:1"}})
    sdir = tmp_path / "state"
    apply_wiring("hermes", GOV, STORE, config_path=str(cfg), sdir=sdir)
    apply_wiring("hermes", GOV, STORE, config_path=str(cfg), sdir=sdir)  # already -> no-op
    assert yaml.safe_load(cfg.read_text())["model"]["base_url"] == GOV


# --------------------------------------------------------------------- launcher main()

def test_main_revert_via_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CM_WIRING_DIR", str(tmp_path / "wire"))
    cfg = _opencode_cfg(tmp_path)
    apply_wiring("opencode", GOV, STORE, config_path=str(cfg))   # uses CM_WIRING_DIR
    assert "contextgovernor" in cfg.read_text()
    main(["--revert", "--cli", "opencode"])
    assert "contextgovernor" not in cfg.read_text()             # reverted
    assert "reverted opencode" in capsys.readouterr().out


def test_main_revert_no_state_message(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CM_WIRING_DIR", str(tmp_path / "wire"))
    main(["--revert", "--cli", "hermes"])
    assert "no wiring state" in capsys.readouterr().out


def test_main_revert_requires_cli() -> None:
    with pytest.raises(SystemExit):
        main(["--revert"])


def test_main_dry_run_includes_wiring_plan_without_applying(tmp_path: Path, capsys) -> None:
    cfg = _opencode_cfg(tmp_path)
    main(["--cli", "opencode", "--cli-config", str(cfg), "--dry-run"])
    data = json.loads(capsys.readouterr().out)
    assert "opencode" in data["_wiring_plan"]
    assert "contextgovernor" not in cfg.read_text()             # dry-run never writes
