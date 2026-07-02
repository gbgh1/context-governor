# run-governor.ps1 — thin wrapper around the `run-governor` Python launcher (Phase 8.1+).
#
# Forwards ALL arguments straight to `python -m contextmanager.launcher`, so the flags are
# exactly the launcher's:
#
#   PS> .\integration\run-governor.ps1 --provider llama --listen-port 8900
#   PS> .\integration\run-governor.ps1 --provider ollama --dry-run
#   PS> .\integration\run-governor.ps1 --cli opencode --dry-run      # preview the wiring
#   PS> .\integration\run-governor.ps1 --cli opencode --provider llama
#   PS> .\integration\run-governor.ps1 --revert --cli opencode
#   PS> .\integration\run-governor.ps1 --help
#
# (No typed params here on purpose — the Python launcher owns flag parsing, config, and help.)

$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "venv python not found at $py — create it: python -m venv .venv; .venv\Scripts\python -m pip install -e ."
}

& $py -m contextmanager.launcher @args
exit $LASTEXITCODE
