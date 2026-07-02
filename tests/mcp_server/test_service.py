"""GovernorService — the pure core (spec §4, §7). No MCP runtime, no network."""

from __future__ import annotations

import pytest

from contextmanager.mcp.service import GovernorService, _preview, _slug


# --------------------------------------------------------------------------- store.*


def test_store_save_roundtrip_and_tokens(make_service):
    svc = make_service()
    content = "alpha beta gamma delta epsilon"
    res = svc.store_save(content, role="note")
    assert res["role"] == "note"
    assert res["tokens"] == 5  # FakeCounter = word count
    # full content retrievable by the returned handle, byte-exact
    assert svc.store.get(res["handle"]) == content


def test_store_save_idempotent(make_service):
    svc = make_service()
    content = "same content twice"
    a = svc.store_save(content, role="note")
    b = svc.store_save(content, role="note")
    assert a["id"] == b["id"]
    assert a["handle"] == b["handle"]
    # no duplicate note on disk
    assert svc.store.notes.list_handles() == [a["handle"]]


def test_store_save_explicit_id(make_service):
    svc = make_service()
    res = svc.store_save("body", role="note", id="my-fixed-id")
    assert res["id"] == "my-fixed-id"


def test_store_search_finds_note_with_preview(make_service):
    svc = make_service(preview_chars=10)
    svc.store_save("the lighthouse stands on the rocky shore", role="note")
    svc.store_save("completely unrelated content about pancakes", role="note")
    out = svc.store_search("lighthouse")
    assert out["count"] >= 1
    top = out["results"][0]
    assert "lighthouse" in svc.store.get(top["handle"])
    # preview respects preview_chars (10) -> truncated with marker
    assert top["preview"].startswith("the lighth")
    assert "more chars" in top["preview"]


# --------------------------------------------------------------------------- state.*


def test_state_snapshot_merge_then_replace(make_service):
    svc = make_service()
    svc.state_snapshot({"score": 100, "level": 3})
    merged = svc.state_snapshot({"level": 4})  # shallow-merge
    assert set(merged["keys"]) == {"score", "level"}
    assert svc.state_load()["state"] == {"score": 100, "level": 4}
    # replace drops the old keys
    svc.state_snapshot({"only": 1}, merge=False)
    assert svc.state_load()["state"] == {"only": 1}


def test_state_load_empty(make_service):
    svc = make_service()
    out = svc.state_load()
    assert out["empty"] is True
    assert out["state"] == {}
    assert out["rendered"] == ""


def test_state_snapshot_rejects_non_dict(make_service):
    svc = make_service()
    with pytest.raises(ValueError):
        svc.state_snapshot(["not", "a", "dict"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- context.*


def test_checkpoint_saves_note_and_state(make_service):
    svc = make_service()
    res = svc.context_checkpoint("turn-7", "checkpoint body", state={"turn": 7})
    assert res["state_updated"] is True
    assert svc.store.get(res["handle"]) == "checkpoint body"
    assert svc.state_load()["state"] == {"turn": 7}


def test_checkpoint_same_label_overwrites(make_service):
    svc = make_service()
    a = svc.context_checkpoint("turn-7", "body one")
    b = svc.context_checkpoint("turn-7", "body two")
    assert a["handle"] == b["handle"]
    assert svc.store.get(b["handle"]) == "body two"
    assert svc.store.notes.list_handles() == [a["handle"]]


def test_rehydrate_by_handle(make_service):
    svc = make_service()
    saved = svc.store_save("one two three four five six", role="note")
    out = svc.context_rehydrate(handle=saved["handle"], budget_tokens=100)
    assert out["found"] is True
    assert out["count"] == 1
    assert out["handles"] == [saved["handle"]]
    assert "one two three" in out["text"]


def test_rehydrate_by_handle_truncates_to_budget(make_service):
    svc = make_service()
    saved = svc.store_save("one two three four five six seven eight", role="note")
    out = svc.context_rehydrate(handle=saved["handle"], budget_tokens=3)
    assert out["tokens"] <= 3  # content tokens held to budget


def test_rehydrate_by_query_respects_budget(make_service):
    svc = make_service()
    svc.store_save("retrieval relevant lighthouse keyword " * 10, role="note")
    out = svc.context_rehydrate(query="lighthouse", budget_tokens=5)
    assert out["found"] is True
    assert out["tokens"] <= 5


def test_rehydrate_budget_zero_is_empty(make_service):
    svc = make_service()
    svc.store_save("anything at all here", role="note")
    out = svc.context_rehydrate(query="anything", budget_tokens=0)
    assert out["found"] is False
    assert out["count"] == 0
    assert out["text"] == ""


def test_rehydrate_requires_exactly_one_selector(make_service):
    svc = make_service()
    with pytest.raises(ValueError):
        svc.context_rehydrate()
    with pytest.raises(ValueError):
        svc.context_rehydrate(query="x", handle="y")


def test_rehydrate_unknown_handle_not_found(make_service):
    svc = make_service()
    out = svc.context_rehydrate(handle="does-not-exist")
    assert out["found"] is False
    assert out["count"] == 0


# --------------------------------------------------------------------------- helpers


def test_stable_id_deterministic():
    a = GovernorService.stable_id("note", "user", "hello")
    b = GovernorService.stable_id("note", "user", "hello")
    c = GovernorService.stable_id("note", "user", "different")
    assert a == b and a != c
    assert a.startswith("note-")


def test_preview_and_slug():
    assert _preview("short", 20) == "short"
    assert _preview("abcdefghij", 4) == "abcd…(+6 more chars)"
    assert _slug("Turn #7: Boss Fight!") == "turn-7-boss-fight"
    assert _slug("!!!") and "-" not in _slug("!!!")[:1]  # non-empty fallback
