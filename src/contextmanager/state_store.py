from __future__ import annotations

import json
import os
from pathlib import Path

try:  # StoreError is defined canonically in note_store.py (spec §1).
    from .note_store import StoreError
except ImportError:  # note_store.py may not yet exist during parallel integration.
    class StoreError(Exception):
        """Raised by stores on malformed persisted data or missing notes."""


class StateStore:
    """Authoritative world/project state as JSON, rewritten in place atomically."""

    def __init__(self, path: str | os.PathLike) -> None:
        # path = the state.json file path. Parent dirs are created on first save.
        self.path: Path = Path(path)

    def load(self) -> dict:
        """Return the parsed dict, or {} if the file does not exist.

        Raise StoreError on malformed JSON (do not silently reset).
        """
        if not self.path.exists():
            return {}
        try:
            text = self.path.read_text(encoding="utf-8")
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StoreError(f"Malformed JSON in {self.path}: {exc}") from exc
        if not isinstance(data, dict):
            raise StoreError(
                f"State file {self.path} did not contain a JSON object"
            )
        return data

    def save(self, state: dict) -> None:
        """ATOMIC write: write to "<path>.tmp" (utf-8, json.dumps sort_keys=True,
        indent=2, ensure_ascii=False), flush + os.fsync the temp file, then
        os.replace(tmp, path). os.replace is atomic on Windows and POSIX for
        same-volume. Create parent dirs if missing.
        """
        parent = self.path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(state, sort_keys=True, indent=2, ensure_ascii=False)
        try:
            # newline="" disables translation so output bytes are deterministic
            # across platforms (spec §8.1).
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(self.path))
        except Exception:
            # Clean up the temp file on any failure so no ".tmp" is left behind.
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise

    def update(self, patch: dict) -> dict:
        """Shallow-merge patch into current load(), save(), return the merged dict."""
        merged = self.load()
        merged.update(patch)
        self.save(merged)
        return merged

    def render(self) -> str:
        """Deterministic text rendering for the engine's state_snapshot tier:
        json.dumps(load(), sort_keys=True, indent=2, ensure_ascii=False).
        "" if state is {}.
        """
        state = self.load()
        if not state:
            return ""
        return json.dumps(state, sort_keys=True, indent=2, ensure_ascii=False)
