"""Phase 2 NoteStore tests — binds to `tasks/phase2-spec.md` §6 (test_note_store).

Covers:
  - page_out(Message) writes a note and returns a stable handle; get(handle)
    returns the body EXACTLY, including a body that itself contains a `---` line.
  - handle_for is deterministic and filesystem-safe: ids with spaces, slashes,
    unicode -> safe slug containing only [A-Za-z0-9._-]; same id -> same handle.
  - read_meta round-trips NoteMeta (id, role, tokens, links) for a note written
    via write_note with tokens=42 and links=["a","b"].
  - has() / list_handles() (sorted) / remove() behave; get on missing handle ->
    StoreError.
  - idempotent: page_out same id twice -> same handle and exactly one note file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from contextmanager.note_store import NoteStore, NoteMeta, StoreError
from contextmanager.types import Message


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _note_file_count(root: Path) -> int:
    notes_dir = root / "notes"
    if not notes_dir.exists():
        return 0
    return sum(1 for _ in notes_dir.glob("*.md"))


# ---------------------------------------------------------------------------
# page_out + get round-trip, body preserved exactly (incl. mid-body `---`)
# ---------------------------------------------------------------------------


def test_page_out_returns_stable_handle_and_get_returns_body_exactly(
    tmp_path: Path,
) -> None:
    store = NoteStore(tmp_path)
    body = (
        "first line\n"
        "---\n"  # a horizontal-rule-like separator inside the body
        "second line after rule\n"
        "third\n"
    )
    msg = Message(role="assistant", content=body, id="msg-1")

    handle = store.page_out(msg)
    assert isinstance(handle, str)
    assert handle  # non-empty

    # stable: same id -> same handle
    assert NoteStore.handle_for("msg-1") == handle

    # get returns the body EXACTLY as written, including the embedded `---`
    assert store.get(handle) == body


def test_get_on_missing_handle_raises_store_error(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    with pytest.raises(StoreError):
        store.get("does-not-exist")


# ---------------------------------------------------------------------------
# handle_for: deterministic, filesystem-safe
# ---------------------------------------------------------------------------


_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


@pytest.mark.parametrize(
    "raw_id",
    [
        "simple",
        "msg with spaces",
        "path/with/slashes",
        "uni café ☕",
        "weird!@#$%^&*()",
        "trailing/space ",
    ],
)
def test_handle_for_filesystem_safe_and_deterministic(raw_id: str) -> None:
    h1 = NoteStore.handle_for(raw_id)
    h2 = NoteStore.handle_for(raw_id)
    assert h1 == h2  # deterministic
    assert _SAFE.match(h1), f"handle {h1!r} contains characters outside [A-Za-z0-9._-]"


def test_handle_for_distinct_ids_give_distinct_handles() -> None:
    a = NoteStore.handle_for("alpha")
    b = NoteStore.handle_for("beta")
    assert a != b


def test_handle_for_empty_or_all_unsafe_does_not_produce_empty() -> None:
    # all-unsafe run -> would be empty without the sha1 fallback
    h = NoteStore.handle_for("!!!")
    assert h != ""
    assert _SAFE.match(h)


# ---------------------------------------------------------------------------
# read_meta round-trips NoteMeta
# ---------------------------------------------------------------------------


def test_read_meta_round_trips_note_meta(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    handle = store.write_note(
        id="note-7",
        role="user",
        content="some body text",
        tokens=42,
        links=["a", "b"],
    )
    meta = store.read_meta(handle)
    assert isinstance(meta, NoteMeta)
    assert meta.id == "note-7"
    assert meta.role == "user"
    assert meta.handle == handle
    assert meta.tokens == 42
    assert meta.links == ["a", "b"]


def test_read_meta_tokens_optional_and_links_default_empty(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    handle = store.write_note(id="n2", role="note", content="hello")
    meta = store.read_meta(handle)
    assert meta.tokens is None
    assert meta.links == []


def test_read_meta_missing_handle_raises_store_error(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    with pytest.raises(StoreError):
        store.read_meta("absent")


# ---------------------------------------------------------------------------
# has / list_handles (sorted) / remove
# ---------------------------------------------------------------------------


def test_has_list_handles_and_remove_behave(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    h1 = store.write_note(id="a-1", role="user", content="alpha content")
    h2 = store.write_note(id="b-2", role="assistant", content="beta content")
    h3 = store.write_note(id="c-3", role="tool", content="gamma content")

    assert store.has(h1)
    assert store.has(h2)
    assert store.has(h3)
    assert not store.has("nope")

    handles = store.list_handles()
    assert handles == sorted(handles)  # sorted
    assert set(handles) == {h1, h2, h3}

    store.remove(h2)
    assert not store.has(h2)
    assert store.has(h1)
    assert store.has(h3)
    assert h2 not in store.list_handles()

    # remove absent handle: no error
    store.remove("never-existed")


# ---------------------------------------------------------------------------
# idempotent: page_out same id twice -> same handle, exactly one note file
# ---------------------------------------------------------------------------


def test_page_out_idempotent_same_id_twice_single_note(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    msg = Message(role="user", content="hello world", id="dup-1")

    h1 = store.page_out(msg)
    h2 = store.page_out(msg)
    assert h1 == h2
    assert _note_file_count(tmp_path) == 1
    assert store.get(h1) == "hello world"


# ---------------------------------------------------------------------------
# write_note atomicity (no .tmp left behind)
# ---------------------------------------------------------------------------


def test_write_note_leaves_no_tmp_file(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    handle = store.write_note(id="x", role="user", content="c")
    note_path = tmp_path / "notes" / f"{handle}.md"
    assert note_path.exists()
    assert not Path(str(note_path) + ".tmp").exists()


# ---------------------------------------------------------------------------
# Round-2 (spec §8.6) edge cases
# ---------------------------------------------------------------------------


def test_handle_for_no_collision_across_separator_collapse() -> None:
    """Spec §8.6: ids that differ only by separator kind must NOT collapse to the
    same handle. handle_for must distinguish space/slash/dot/underscore runs so
    that "a b", "a-b", "a/b", "a.b", "a_b", "a b c", "a-b-c" each get a unique
    handle (7 distinct handles for 7 distinct ids).
    """
    ids = ["a b", "a-b", "a/b", "a.b", "a_b", "a b c", "a-b-c"]
    handles = {NoteStore.handle_for(s) for s in ids}
    assert len(handles) == 7, (
        f"expected 7 distinct handles for 7 separator-distinct ids, got {len(handles)}: {handles}"
    )


def test_note_body_verbatim_edge_cases(tmp_path: Path) -> None:
    """Spec §8.1 / §8.6 (H1): the body returned by get() must be byte/char-identical
    to the content written, for ANY content — including:
      * a leading `---` line (frontmatter-like prefix inside the body),
      * an empty body,
      * a body with no trailing newline,
      * a body containing CRLF (`\\r\\n`) — the CR must survive Windows I/O,
      * a body containing a bare `\\r`,
      * a body with an internal `---` separator,
      * a body that is itself a leading `---`-only fragment.

    This locks in the newline-stable I/O fix (write/read with newline="").
    """
    store = NoteStore(tmp_path)
    bodies = [
        "---\nfoo",            # leading `---` inside body
        "",                     # empty body
        "abc",                  # no trailing newline
        "line1\r\nline2\r\n",   # CRLF preserved
        "mid\n---\nbody",       # internal `---` separator
        "lead---\n",            # `---` not on its own line at start
        "no newline end",       # plain, no trailing newline
    ]

    for i, body in enumerate(bodies):
        handle = store.write_note(
            id=f"verbatim-{i}",
            role="user",
            content=body,
        )
        got = store.get(handle)
        assert got == body, (
            f"body #{i} {body!r} did not round-trip verbatim; got {got!r}"
        )
        # Explicit CRLF check (the H1 Windows-corruption regression): if the body
        # contained `\r`, the read-back must still contain `\r`.
        if "\r" in body:
            assert "\r" in got, f"CRLF was stripped from body #{i}: {got!r}"

    # Also exercise the page_out path (engine Store Protocol) for the CRLF case,
    # since page_out is the production write entry-point the engine uses.
    crlf_body = "alpha\r\nbeta\r\ngamma\r\n"
    msg = Message(role="assistant", content=crlf_body, id="crlf-msg")
    h = store.page_out(msg)
    assert store.get(h) == crlf_body
    assert "\r" in store.get(h)


# ---------------------------------------------------------------------------
# Phase 7 Stage 3 — cold-tier compression (gzip at rest, transparent reads)
# ---------------------------------------------------------------------------


def test_compress_round_trips_body_byte_exact(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    body = "alpha\r\nbeta\rgamma\ndelta\n---\nnot-frontmatter\n"
    store.write_note(id="m1", role="user", content=body, tokens=7, links=["x", "y"])
    h = NoteStore.handle_for("m1")

    assert store.compress(h) is True
    assert store.is_compressed(h)
    assert not (tmp_path / "notes" / f"{h}.md").is_file()
    assert (tmp_path / "notes" / f"{h}.md.gz").is_file()

    # get() and read_meta() decompress transparently; body stays byte-exact.
    assert store.get(h) == body
    assert "\r" in store.get(h)
    meta = store.read_meta(h)
    assert meta.tokens == 7 and meta.links == ["x", "y"]

    # has()/list_handles() see the compressed note.
    assert store.has(h)
    assert h in store.list_handles()


def test_compress_then_decompress_restores_plain(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    store.write_note(id="m1", role="user", content="hello cold world")
    h = NoteStore.handle_for("m1")
    store.compress(h)
    assert store.decompress(h) is True
    assert not store.is_compressed(h)
    assert (tmp_path / "notes" / f"{h}.md").is_file()
    assert store.get(h) == "hello cold world"


def test_compress_absent_or_already_compressed_returns_false(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    assert store.compress("never-written") is False
    store.write_note(id="m1", role="user", content="x")
    h = NoteStore.handle_for("m1")
    assert store.compress(h) is True
    assert store.compress(h) is False  # already compressed


def test_remove_deletes_compressed_note(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    store.write_note(id="m1", role="user", content="to be removed")
    h = NoteStore.handle_for("m1")
    store.compress(h)
    store.remove(h)
    assert not store.has(h)
    assert h not in store.list_handles()


# ---------------------------------------------------------------------------
# archive tier (Phase 9 — lossless saving)
# ---------------------------------------------------------------------------


def test_archive_moves_note_out_of_live_set(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    body = "keep me forever\r\ncrlf intact\rbare cr\n"
    store.write_note(id="m1", role="user", content=body)
    h = NoteStore.handle_for("m1")
    assert store.archive(h) is True
    assert not store.has(h)                       # invisible to the live set
    assert h not in store.list_handles()
    assert store.is_archived(h)
    assert store.list_archived() == [h]
    assert (tmp_path / "archive" / f"{h}.md.gz").is_file()
    with pytest.raises(StoreError):
        store.get(h)                              # archived != readable in place


def test_restore_brings_back_byte_exact(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    body = "restore me\r\nwindows crlf\rbare cr\nlf trailing\n"
    store.write_note(id="m1", role="user", content=body)
    h = NoteStore.handle_for("m1")
    store.archive(h)
    assert store.restore(h) is True
    assert store.has(h)
    assert not store.is_archived(h)
    assert store.get(h) == body                   # byte-exact round-trip


def test_archive_of_compressed_note_is_pure_move(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    store.write_note(id="m1", role="user", content="cold then archived")
    h = NoteStore.handle_for("m1")
    store.compress(h)
    assert store.archive(h) is True
    assert store.is_archived(h)
    store.restore(h)
    assert store.get(h) == "cold then archived"


def test_archive_and_restore_absent_return_false(tmp_path: Path) -> None:
    store = NoteStore(tmp_path)
    assert store.archive("never-written") is False
    assert store.restore("never-written") is False
    assert store.list_archived() == []
