"""CLI auto-wiring (Phase 8.3) — put the governor in between, safely.

Selecting a CLI (`run-governor --cli opencode|hermes`) makes the governor:
  1. locate the CLI's config file,
  2. take a TIMESTAMPED BACKUP,
  3. atomically insert itself (repoint the CLI at the governor),
  4. record a wiring-state file so it can ALWAYS be undone,
and revert on graceful exit. After a crash/BSOD, `run-governor --revert --cli X`
restores the original from the backup recorded in the state file.

Design rules (this edits the user's real config files, so they are strict):
  - **backup-first, atomic** (temp + os.replace), **idempotent** (never double-wrap),
  - **revert restores the backup VERBATIM** (no fragile surgical un-edit),
  - a missing/odd config is a clear error, never a guess.

The per-CLI insert is the only schema-specific part. Hermes = repoint the scalar
`model.base_url` (well-defined). OpenCode = add/repoint a `context-governor`
provider's `options.baseURL` (best-effort; preview with `--dry-run --cli opencode`,
and it's always reverted from backup).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_OC_PROVIDER_KEY = "contextgovernor"
_OC_MODEL_ID = "context-governor"  # must match the proxy's model_alias (what /v1/models serves)
_MCP_KEY = "context-governor"       # the MCP-server entry key, in both CLIs


class WiringError(Exception):
    """Bad/odd CLI config, missing file, or absent driver — never guessed past."""


@dataclass
class WiringState:
    cli: str
    config_path: str
    backup_path: str
    governor_url: str
    applied_at: str
    summary: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "WiringState":
        return cls(**json.loads(text))


# ----------------------------------------------------------------- locations

def _parse_opencode_paths(stdout: str) -> Optional[str]:
    """Pull the `config` directory out of `opencode debug paths` output (lines like
    'config     C:\\Users\\you\\.config\\opencode'). Pure -> unit-testable."""
    for line in stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].strip() == "config":
            return parts[1].strip()
    return None


def discover_opencode_config() -> Optional[Path]:
    """Ask OpenCode itself where its config lives (`opencode debug paths`), so wiring works
    on any OS/layout — not just the Windows default. None if opencode isn't runnable."""
    exe = shutil.which("opencode")
    if not exe:
        return None
    try:
        proc = subprocess.run([exe, "debug", "paths"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    cfg_dir = _parse_opencode_paths(proc.stdout)
    return Path(cfg_dir) / "config.json" if cfg_dir else None


def resolve_cli_config(cli: str) -> Path:
    """The CLI's config file — DISCOVERED where possible, platform-default otherwise."""
    if cli == "opencode":
        found = discover_opencode_config()
        return found if found else Path.home() / ".config" / "opencode" / "config.json"
    if cli == "hermes":
        # Hermes has no reliable path command on a broken install; use the platform default.
        # (Wiring edits config.yaml directly, so it does NOT need the hermes binary.)
        if os.name == "nt":
            return Path.home() / "AppData" / "Local" / "hermes" / "config.yaml"
        return Path.home() / ".config" / "hermes" / "config.yaml"
    raise WiringError(f"no wiring driver for cli '{cli}'")


def _mcp_args(store_root: str) -> list:
    """Args that run THIS project's MCP server (Surface B) against `store_root`."""
    return ["-m", "contextmanager.mcp", "--store-root", store_root]


def state_dir() -> Path:
    """Where wiring-state + backups live. Overridable via CM_WIRING_DIR (used by tests)."""
    override = os.environ.get("CM_WIRING_DIR")
    return Path(override) if override else (Path.home() / ".context-governor" / "wiring")


def _state_path(cli: str, sdir: Path) -> Path:
    return sdir / f"{cli}.json"


# ----------------------------------------------------------------- io helpers

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _backup(src: Path, backups: Path) -> Path:
    backups.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dst = backups / f"{src.name}.{ts}.bak"
    shutil.copy2(src, dst)
    return dst


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".cmtmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                os.unlink(tmp)
        except OSError:
            pass
        raise


def _import_yaml():
    try:
        import yaml  # PyYAML (optional 'wiring' extra)
    except ImportError as exc:  # pragma: no cover - depends on env
        raise WiringError(
            "PyYAML is required to wire Hermes (YAML). Install it: pip install pyyaml "
            "(or `pip install contextmanager[wiring]`), or wire model.base_url manually."
        ) from exc
    return yaml


# ----------------------------------------------------------------- per-CLI insert
# Each returns (new_bytes, already_wired, human_summary).

def _insert_hermes(raw: bytes, base_url: str, store_root: str) -> tuple[bytes, bool, str]:
    yaml = _import_yaml()
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise WiringError("Hermes config is not a YAML mapping")
    model = data.setdefault("model", {})
    if not isinstance(model, dict):
        raise WiringError("Hermes config 'model' is not a mapping")
    servers = data.setdefault("mcp_servers", {})
    if not isinstance(servers, dict):
        raise WiringError("Hermes config 'mcp_servers' is not a mapping")

    desired_args = _mcp_args(store_root)
    srv = servers.get(_MCP_KEY)
    base_ok = model.get("base_url") == base_url
    mcp_ok = (isinstance(srv, dict) and srv.get("command") == sys.executable
              and srv.get("args") == desired_args)
    if base_ok and mcp_ok:
        return raw, True, f"already wired (model.base_url + mcp_servers '{_MCP_KEY}')"

    old = model.get("base_url")
    model["base_url"] = base_url
    servers[_MCP_KEY] = {"command": sys.executable, "args": desired_args}  # Surface B
    new = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).encode("utf-8")
    return new, False, f"model.base_url {old!r}->{base_url!r} + mcp_servers '{_MCP_KEY}'"


def _opencode_provider(base_url: str) -> dict:
    """A COMPLETE OpenAI-compatible provider entry (with a usable model) — OpenCode needs
    a `models` block, not just a baseURL. Limits are sensible defaults; adjust to your n_ctx."""
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Context Governor (Local)",
        "options": {"baseURL": base_url},
        "models": {
            _OC_MODEL_ID: {
                "name": "[LOCAL] ContextGovernor",
                "attachment": True,
                "tool_call": True,
                "reasoning": True,
                "limit": {"context": 131072, "output": 16384},
            }
        },
    }


def _insert_opencode(raw: bytes, base_url: str, store_root: str) -> tuple[bytes, bool, str]:
    data = json.loads(raw.decode("utf-8") or "{}")
    if not isinstance(data, dict):
        raise WiringError("OpenCode config is not a JSON object")
    providers = data.setdefault("provider", {})
    if not isinstance(providers, dict):
        raise WiringError("OpenCode config 'provider' is not an object")
    mcp = data.setdefault("mcp", {})
    if not isinstance(mcp, dict):
        raise WiringError("OpenCode config 'mcp' is not an object")

    desired_cmd = [sys.executable] + _mcp_args(store_root)
    prov = providers.get(_OC_PROVIDER_KEY)
    srv = mcp.get(_MCP_KEY)
    prov_ok = isinstance(prov, dict) and prov.get("options", {}).get("baseURL") == base_url
    mcp_ok = isinstance(srv, dict) and srv.get("command") == desired_cmd
    if prov_ok and mcp_ok:
        return raw, True, f"already wired (provider '{_OC_PROVIDER_KEY}' + mcp '{_MCP_KEY}')"

    # provider: repoint baseURL on an existing entry (keep its models), else a full new one.
    if isinstance(prov, dict):
        merged = dict(prov)
        options = dict(merged.get("options") or {})
        options["baseURL"] = base_url
        merged["options"] = options
    else:
        merged = _opencode_provider(base_url)
    providers[_OC_PROVIDER_KEY] = merged
    # mcp (Surface B): point at THIS project's python + store.
    mcp[_MCP_KEY] = {"type": "local", "command": desired_cmd, "enabled": True}

    new = json.dumps(data, indent=2).encode("utf-8")
    return new, False, f"provider '{_OC_PROVIDER_KEY}' + mcp '{_MCP_KEY}' -> {base_url}"


def _insert(cli: str, raw: bytes, base_url: str, store_root: str) -> tuple[bytes, bool, str]:
    if cli == "hermes":
        return _insert_hermes(raw, base_url, store_root)
    if cli == "opencode":
        return _insert_opencode(raw, base_url, store_root)
    raise WiringError(f"no wiring driver for cli '{cli}'")


# ----------------------------------------------------------------- public API

def plan_wiring(cli: str, base_url: str, store_root: str, *,
                config_path: Optional[str] = None) -> str:
    """Describe what wiring `cli` WOULD do, without touching anything (for --dry-run)."""
    cfg = Path(config_path) if config_path else resolve_cli_config(cli)
    if not cfg.is_file():
        return f"would FAIL: {cli} config not found at {cfg} (pass --cli-config)"
    try:
        _, already, summary = _insert(cli, cfg.read_bytes(), base_url, store_root)
    except WiringError as exc:
        return f"would FAIL: {exc}"
    return f"{cli} @ {cfg}: " + ("already wired" if already else summary)


def apply_wiring(cli: str, base_url: str, store_root: str, *,
                 config_path: Optional[str] = None, sdir: Optional[Path] = None) -> WiringState:
    """Backup + insert the governor (provider/base_url + MCP) into `cli`'s config. Idempotent
    (a no-op if already wired by us). Returns the WiringState that `revert_wiring` consumes."""
    sdir = sdir or state_dir()
    cfg = Path(config_path) if config_path else resolve_cli_config(cli)
    if not cfg.is_file():
        raise WiringError(f"{cli} config not found: {cfg} (pass --cli-config to point at it)")

    new, already, summary = _insert(cli, cfg.read_bytes(), base_url, store_root)
    sp = _state_path(cli, sdir)
    if already:
        if sp.is_file():
            return WiringState.from_json(sp.read_text(encoding="utf-8"))
        raise WiringError(
            f"{cfg} already appears wired but no governor state file exists; "
            f"restore your own backup or remove the wiring manually before re-running"
        )

    backup = _backup(cfg, sdir / "backups")
    _atomic_write_bytes(cfg, new)
    state = WiringState(
        cli=cli, config_path=str(cfg), backup_path=str(backup),
        governor_url=base_url, applied_at=_now(), summary=summary,
    )
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(state.to_json(), encoding="utf-8")
    return state


def revert_wiring(cli: str, *, sdir: Optional[Path] = None) -> bool:
    """Restore `cli`'s config from the recorded backup (verbatim) and clear the state.
    Returns False if there is nothing to revert. Safe to call after a crash/BSOD."""
    sdir = sdir or state_dir()
    sp = _state_path(cli, sdir)
    if not sp.is_file():
        return False
    st = WiringState.from_json(sp.read_text(encoding="utf-8"))
    backup, cfg = Path(st.backup_path), Path(st.config_path)
    if backup.is_file():
        _atomic_write_bytes(cfg, backup.read_bytes())  # verbatim restore
    sp.unlink()
    return True
