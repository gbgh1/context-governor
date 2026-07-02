"""Phase 2 StateStore tests — binds to `tasks/phase2-spec.md` §6 (test_state_store).

Covers:
  - load() returns {} when file absent
  - save() then load() round-trips a nested dict
  - save() is atomic: no "<path>.tmp" left behind; resulting file is valid JSON
  - update() shallow-merges (existing keys preserved, patched keys overwritten)
  - render() is deterministic, sorted, indented; "" for empty state
  - malformed JSON file -> StoreError on load()

StoreError is defined in `contextmanager.note_store` (spec §1) and re-used by
StateStore. We import it from note_store; if a re-export exists on state_store we
fall back to that so the test binds to whichever the implementation exposes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from contextmanager.state_store import StateStore

try:  # spec §1: StoreError lives in note_store
    from contextmanager.note_store import StoreError
except ImportError:  # allow a re-export on state_store
    from contextmanager.state_store import StoreError  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# load() when absent
# ---------------------------------------------------------------------------


def test_load_returns_empty_dict_when_file_absent(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    assert store.load() == {}


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def test_save_then_load_round_trips_nested_dict(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    store = StateStore(path)
    state = {
        "project": "demo",
        "nested": {"a": [1, 2, 3], "b": {"c": True}},
        "scalar": 7,
    }
    store.save(state)
    assert store.load() == state


# ---------------------------------------------------------------------------
# atomic write: no .tmp left behind, file is valid JSON
# ---------------------------------------------------------------------------


def test_save_is_atomic_no_tmp_left_and_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "subdir" / "state.json"  # parent dirs created on first save
    store = StateStore(path)
    store.save({"k": "v"})

    # the .tmp sibling must have been removed by os.replace
    assert not Path(str(path) + ".tmp").exists()
    # and the real file is valid JSON with the expected content
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"k": "v"}


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "x" / "y" / "z" / "state.json"
    StateStore(path).save({"a": 1})
    assert path.exists()


# ---------------------------------------------------------------------------
# update() shallow-merge
# ---------------------------------------------------------------------------


def test_update_shallow_merges_preserving_and_overwriting(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    store.save({"a": 1, "nested": {"x": 1, "y": 2}, "kept": True})
    merged = store.update({"b": 2, "nested": {"z": 3}})

    # patched keys overwritten/added; untouched keys preserved
    assert merged == {
        "a": 1,
        "b": 2,
        "nested": {"z": 3},  # shallow: the whole "nested" value is replaced
        "kept": True,
    }
    # persisted
    assert store.load() == merged


def test_update_on_absent_file_acts_as_save(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    merged = store.update({"fresh": True})
    assert merged == {"fresh": True}
    assert store.load() == {"fresh": True}


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


def test_render_empty_state_returns_empty_string(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    # nothing saved yet -> load == {} -> render == ""
    assert store.render() == ""


def test_render_is_deterministic_sorted_indented(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    # deliberately unsorted keys at insert time
    state = {"zeta": 1, "alpha": [3, 2, 1], "mid": {"k": "v"}}
    store.save(state)

    rendered = store.render()
    expected = json.dumps(state, sort_keys=True, indent=2, ensure_ascii=False)
    assert rendered == expected

    # determinism: repeated renders produce the same bytes
    assert store.render() == store.render()

    # sorted: "alpha" line comes before "mid" comes before "zeta"
    lines = rendered.splitlines()
    keys_in_order = [ln.split(":")[0].strip().strip('"') for ln in lines if ":" in ln and not ln.startswith(" " * 4)]
    # top-level keys appear in sorted order
    assert keys_in_order[:3] == ["alpha", "mid", "zeta"]


# ---------------------------------------------------------------------------
# malformed JSON -> StoreError
# ---------------------------------------------------------------------------


def test_load_malformed_json_raises_store_error(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = StateStore(path)
    with pytest.raises(StoreError):
        store.load()


# ---------------------------------------------------------------------------
# Round-2 (spec §8.6 / §8.3) atomic write on simulated failure
# ---------------------------------------------------------------------------


def test_state_store_atomic_on_simulated_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Spec §8.6 (M6) + §8.3: if os.replace fails mid-save, the ORIGINAL file
    must remain intact (atomic rewrite-in-place must not corrupt existing
    state) and the `<path>.tmp` file must be cleaned up (no orphan left behind).

    We pre-write a valid state file, monkeypatch `os.replace` to raise OSError,
    and assert:
      1. save() raises (the failure propagates — it is not swallowed),
      2. the original file content is byte-for-byte unchanged,
      3. no `<path>.tmp` file remains afterward.
    """
    path = tmp_path / "state.json"
    store = StateStore(path)

    original = {"existing": "state", "n": 1, "nested": {"x": [1, 2]}}
    store.save(original)
    original_bytes = path.read_bytes()  # snapshot exact on-disk bytes

    # Sanity: tmp must not exist before the failing save.
    tmp_path_sibling = Path(str(path) + ".tmp")
    assert not tmp_path_sibling.exists()

    real_os_replace = os.replace

    def _boom(*args, **kwargs):
        raise OSError("simulated atomic-replace failure")

    monkeypatch.setattr(os, "replace", _boom)

    try:
        with pytest.raises(OSError):
            store.save({"this": "should-not-be-written"})
    finally:
        # Restore os.replace so any subsequent teardown/tmp cleanup works.
        monkeypatch.setattr(os, "replace", real_os_replace)

    # (1) original content intact — byte-for-byte unchanged
    assert path.exists(), "original state.json must still exist after failed save"
    assert path.read_bytes() == original_bytes, (
        "original state.json was corrupted by the failed save"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == original

    # (2) no orphan .tmp left behind
    assert not tmp_path_sibling.exists(), (
        "orphan <path>.tmp left behind after failed atomic save"
    )

    # (3) sanity: store still usable after the failed save (writes succeed again)
    monkeypatch.setattr(os, "replace", real_os_replace)
    store.save({"recovered": True})
    assert store.load() == {"recovered": True}
    assert not tmp_path_sibling.exists()
