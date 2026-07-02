from __future__ import annotations

import gzip
import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .types import Message


class StoreError(Exception):
    """Shared error type for store modules (canonical home: note_store.py per spec §1)."""


@dataclass
class NoteMeta:
    id: str            # source message id (or note id)
    role: str          # "user"|"assistant"|"tool"|"note"|...
    handle: str        # the storage handle (== filename stem)
    created: str       # ISO-8601 UTC timestamp
    tokens: Optional[int]   # token count if known, else None
    links: list[str]   # wikilink targets (handles), may be empty


class NoteStore:
    """Persists content as human-auditable markdown notes with frontmatter.
    Implements the engine's Store Protocol (page_out, get)."""

    _SAFE = re.compile(r"[^A-Za-z0-9._-]+")

    def __init__(self, root: str | os.PathLike) -> None:
        # notes live under <root>/notes/ ; created on first write. Archived
        # (evicted) notes live under <root>/archive/ — outside notes/, so they are
        # invisible to list_handles()/has() but never destroyed (Phase 9).
        self._root = Path(root)
        self._notes_dir = self._root / "notes"
        self._archive_dir = self._root / "archive"

    # ----- paths -----
    def _path(self, handle: str) -> Path:
        return self._notes_dir / f"{handle}.md"

    def _path_gz(self, handle: str) -> Path:
        # Cold notes are gzipped at rest as "<handle>.md.gz" (Phase 7 Stage 3). get()
        # decompresses transparently, so a compressed note is still fully readable and
        # — because the search index is untouched — still findable.
        return self._notes_dir / f"{handle}.md.gz"

    def _archive_path(self, handle: str) -> Path:
        return self._archive_dir / f"{handle}.md.gz"

    def _read_raw(self, handle: str) -> str:
        """Raw note text (frontmatter + body), reading the plain `.md` if present, else
        the gzipped `.md.gz`. ``newline=""`` keeps bodies BYTE-EXACT across both forms."""
        plain = self._path(handle)
        if plain.is_file():
            return open(plain, "r", encoding="utf-8", newline="").read()
        gz = self._path_gz(handle)
        if gz.is_file():
            with gzip.open(gz, "rt", encoding="utf-8", newline="") as fh:
                return fh.read()
        raise StoreError(f"note not found: {handle}")

    # ----- engine Store Protocol -----
    def page_out(self, message: Message) -> str:
        handle = self.handle_for(message.id)
        self.write_note(
            id=message.id,
            role=message.role,
            content=message.content,
            links=[],
        )
        return handle

    def get(self, handle: str) -> str:
        # Reads plain or gzipped form; bodies round-trip BYTE-EXACT either way (spec §8.1).
        raw = self._read_raw(handle)
        _, body = self._split_frontmatter(raw, handle)
        return body

    # ----- general note API -----
    def write_note(self, *, id: str, role: str, content: str,
                   tokens: Optional[int] = None,
                   links: Optional[list[str]] = None) -> str:
        handle = self.handle_for(id)
        links = links if links is not None else []
        created = datetime.now(timezone.utc).isoformat()
        rendered = self._render_note(
            id=id, role=role, handle=handle, created=created,
            tokens=tokens, links=links, content=content,
        )
        self._atomic_write(self._path(handle), rendered)
        return handle

    def read_meta(self, handle: str) -> NoteMeta:
        raw = self._read_raw(handle)  # plain or gzipped
        meta, _ = self._split_frontmatter(raw, handle)
        return meta

    def has(self, handle: str) -> bool:
        return self._path(handle).is_file() or self._path_gz(handle).is_file()

    def mtime(self, handle: str) -> Optional[float]:
        """Last-modified time (epoch seconds) of a note (plain or gzipped), or None if
        absent. Used by the store GC to honor a min-age guard against deleting
        just-written notes that have not yet been indexed."""
        for p in (self._path(handle), self._path_gz(handle)):
            try:
                return p.stat().st_mtime
            except OSError:
                continue
        return None

    def list_handles(self) -> list[str]:
        if not self._notes_dir.is_dir():
            return []
        out: set[str] = {p.stem for p in self._notes_dir.glob("*.md")}
        out.update(p.name[:-6] for p in self._notes_dir.glob("*.md.gz"))  # strip ".md.gz"
        return sorted(out)

    def remove(self, handle: str) -> None:
        for path in (self._path(handle), self._path_gz(handle)):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    # ----- cold-tier compression (Phase 7 Stage 3) -----
    def is_compressed(self, handle: str) -> bool:
        return (not self._path(handle).is_file()) and self._path_gz(handle).is_file()

    def compress(self, handle: str) -> bool:
        """Gzip a note at rest ("<handle>.md" -> "<handle>.md.gz"). Lossless: get() and
        read_meta() decompress transparently and the body stays byte-exact; the search
        index is untouched so the note is still findable. Returns True if it compressed,
        False if the note is absent or already compressed. Atomic (temp + os.replace)."""
        plain = self._path(handle)
        if not plain.is_file():
            return False
        raw = open(plain, "r", encoding="utf-8", newline="").read()
        gz = self._path_gz(handle)
        tmp = gz.with_name(gz.name + ".tmp")
        try:
            with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
                fh.write(raw)
            os.replace(tmp, gz)
        except Exception:
            try:
                if tmp.exists():
                    os.unlink(tmp)
            except OSError:
                pass
            raise
        plain.unlink()  # only after the .gz is durably in place
        return True

    def decompress(self, handle: str) -> bool:
        """Inverse of compress (restore the plain .md, e.g. when a note goes hot again).
        Returns True if it decompressed, False if not compressed/absent."""
        gz = self._path_gz(handle)
        if not gz.is_file():
            return False
        with gzip.open(gz, "rt", encoding="utf-8", newline="") as fh:
            raw = fh.read()
        self._atomic_write(self._path(handle), raw)
        gz.unlink()
        return True

    # ----- archive tier (Phase 9 — lossless saving) -----
    def is_archived(self, handle: str) -> bool:
        return self._archive_path(handle).is_file()

    def list_archived(self) -> list[str]:
        if not self._archive_dir.is_dir():
            return []
        return sorted(p.name[:-6] for p in self._archive_dir.glob("*.md.gz"))

    def archive(self, handle: str) -> bool:
        """Demote a note to the archive tier ("<root>/archive/<handle>.md.gz"): gone
        from the live set (``list_handles``/``has``) but never destroyed —
        ``restore()`` brings it back byte-exact. A plain note is gzipped atomically
        (temp + ``os.replace``) before the live copy is unlinked; an
        already-compressed note is a pure same-volume rename. Returns True if
        archived, False if the note is absent."""
        dest = self._archive_path(handle)
        plain = self._path(handle)
        gz = self._path_gz(handle)
        if plain.is_file():
            raw = open(plain, "r", encoding="utf-8", newline="").read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            try:
                with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
                    fh.write(raw)
                os.replace(tmp, dest)
            except Exception:
                try:
                    if tmp.exists():
                        os.unlink(tmp)
                except OSError:
                    pass
                raise
            plain.unlink()  # only after the archived copy is durably in place
            try:
                gz.unlink()  # drop a stray duplicate compressed form, if any
            except FileNotFoundError:
                pass
            return True
        if gz.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(gz, dest)
            return True
        return False

    def restore(self, handle: str) -> bool:
        """Inverse of ``archive``: bring an archived note back into the live set (as
        "<handle>.md.gz" — reads already decompress transparently). Pure rename,
        byte-exact. Returns True if restored, False if not archived."""
        src = self._archive_path(handle)
        if not src.is_file():
            return False
        self._notes_dir.mkdir(parents=True, exist_ok=True)
        os.replace(src, self._path_gz(handle))
        return True

    @staticmethod
    def handle_for(message_id: str) -> str:
        # Deterministic, filesystem-safe slug from message_id: keep [A-Za-z0-9._-],
        # replace any other run with "-". Stable: same id -> same handle.
        slug = NoteStore._SAFE.sub("-", message_id)
        short = hashlib.sha1(message_id.encode("utf-8")).hexdigest()[:8]
        if not slug:
            # all characters were unsafe -> use short sha1 hex of the original id
            return short
        if set(slug) == {"."}:
            # dot-only slug (e.g. ".", "..") would be a hidden/invalid .md stem
            # (".md", "..md") -> suffix with the short sha1 hex so the handle is
            # a visible, valid filename (spec §8.5).
            return f"{slug}-{short}"
        if slug != message_id:
            # slugging changed the id -> risk of collision across distinct ids that
            # collapse to the same slug; append a short stable sha1 suffix.
            return f"{slug}-{short}"
        return slug

    # ----- internal: rendering & parsing -----
    @staticmethod
    def _render_note(*, id: str, role: str, handle: str, created: str,
                     tokens: Optional[int], links: list[str], content: str) -> str:
        tokens_field = "" if tokens is None else str(tokens)
        links_field = "[" + ", ".join(links) + "]"
        fm = (
            "---\n"
            f"id: {id}\n"
            f"role: {role}\n"
            f"handle: {handle}\n"
            f"created: {created}\n"
            f"tokens: {tokens_field}\n"
            f"links: {links_field}\n"
            "---\n"
            "\n"
        )
        return fm + content

    @staticmethod
    def _split_frontmatter(raw: str, handle: str) -> tuple[NoteMeta, str]:
        # Only the FIRST `---`...`---` block is frontmatter (spec §3).
        # Split on the first two `---` delimiters only, so bodies containing
        # `---` lines are preserved verbatim.
        lines = raw.split("\n")
        if not lines or lines[0] != "---":
            raise StoreError(f"malformed note (missing opening delimiter): {handle}")
        close_idx: Optional[int] = None
        for i in range(1, len(lines)):
            if lines[i] == "---":
                close_idx = i
                break
        if close_idx is None:
            raise StoreError(f"malformed note (missing closing delimiter): {handle}")
        fm_lines = lines[1:close_idx]

        fields = NoteStore._parse_frontmatter_fields(fm_lines, handle)

        # Body = everything after the closing `---` line and its following blank line.
        body_lines = lines[close_idx + 1:]
        if body_lines and body_lines[0] == "":
            body_lines = body_lines[1:]
        body = "\n".join(body_lines)

        meta = NoteMeta(
            id=fields["id"],
            role=fields["role"],
            handle=fields["handle"],
            created=fields["created"],
            tokens=fields["tokens"],
            links=fields["links"],
        )
        return meta, body

    @staticmethod
    def _parse_frontmatter_fields(fm_lines: list[str], handle: str) -> dict:
        parsed: dict[str, object] = {}
        for line in fm_lines:
            if line.strip() == "":
                continue
            if ":" not in line:
                raise StoreError(f"malformed note (bad frontmatter line): {handle}")
            key, _, value = line.partition(":")
            parsed[key.strip()] = value.strip()

        # required scalar fields
        try:
            id_ = parsed["id"]
            role = parsed["role"]
            fm_handle = parsed["handle"]
            created = parsed["created"]
        except KeyError as exc:
            raise StoreError(f"malformed note (missing field {exc!s}): {handle}") from exc
        if not isinstance(id_, str) or not id_:
            raise StoreError(f"malformed note (empty id): {handle}")
        if not isinstance(role, str) or not role:
            raise StoreError(f"malformed note (empty role): {handle}")
        if not isinstance(fm_handle, str) or not fm_handle:
            raise StoreError(f"malformed note (empty handle): {handle}")
        if not isinstance(created, str) or not created:
            raise StoreError(f"malformed note (empty created): {handle}")

        # tokens: int or empty -> None
        tokens_raw = parsed.get("tokens", "")
        if tokens_raw == "" or tokens_raw is None:
            tokens: Optional[int] = None
        else:
            try:
                tokens = int(tokens_raw)
            except ValueError as exc:
                raise StoreError(f"malformed note (bad tokens): {handle}") from exc

        # links: [h1, h2]
        links_raw = parsed.get("links", "")
        links = NoteStore._parse_links(str(links_raw))

        return {
            "id": id_,
            "role": role,
            "handle": fm_handle,
            "created": created,
            "tokens": tokens,
            "links": links,
        }

    @staticmethod
    def _parse_links(field: str) -> list[str]:
        s = field.strip()
        if s == "" or s == "[]":
            return []
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if s.strip() == "":
            return []
        items = [item.strip() for item in s.split(",")]
        return [item for item in items if item != ""]

    # ----- internal: atomic write -----
    def _atomic_write(self, path: Path, data: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        # newline="" disables translation so \n is written as-is, no \r\n
        # (spec §8.1); mirror StateStore.save's temp cleanup so a failed
        # write never leaves an orphan <handle>.md.tmp (spec §8.3).
        try:
            with open(tmp, "w", encoding="utf-8", newline="") as fh:
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
