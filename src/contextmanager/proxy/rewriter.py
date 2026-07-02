"""PromptRewriter — the Phase 3 core: a pure, deterministic transform over an
OpenAI-compatible `messages` list.

Normative per Phase 3 spec §3 (3.1–3.5). Free of any FastAPI/httpx/network
imports: it depends only on the stdlib (`hashlib`, `re`, `dataclasses`) and on
the contextmanager `TokenCounter`/`Message`/`DurableStore` types.

The headline invariant (§3.5) is PREFIX STABILITY / IDEMPOTENCY: a second pass
of `rewrite_outgoing` over its own output is a no-op for handle-ization (stubs
stay byte-identical), and handle-ization of message i depends ONLY on message i
(never on its neighbors). Both hold by construction here because:
  - `stable_id` is a pure function of (role, content);
  - the handle-ization decision (`counter.count_text(content) >= threshold`) is
    per-message;
  - already-stub content is detected and left untouched.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from ..types import Message, TokenCounter
from ..durable import DurableStore
from .recall import extract_query, select_diverse


# Regex matching the opening line of a stub (spec §3.1). The handle is captured;
# role and tokens are matched but not captured.
_HANDLE_RE = re.compile(
    r"\[\[cm:stored handle=(?P<handle>\S+) role=\S+ tokens=\d+\]\]"
)

# The literal prefix that identifies a stub (spec §3.3 is_stub).
_STUB_PREFIX = "[[cm:stored handle="

# Regex matching the opening marker line of a synthetic rehydrated message
# (spec §9.1): `[[cm:rehydrated handle=<H>]]`. The handle is captured. Used by
# `is_rehydrated` (prefix check) and `_parse_rehydrated_handle` (full parse).
_REHYDRATED_RE = re.compile(r"\[\[cm:rehydrated handle=(?P<handle>\S+)\]\]")
_REHYDRATED_PREFIX = "[[cm:rehydrated handle="

# Diff-stub: a delta-compressed handle-ization. A bulky message that is a near-
# duplicate of an already-stored note is replaced by the BASE handle + a unified
# diff instead of a head/tail preview — lossless (full content still paged out
# under `handle`), and tiny + informative for iterative content (file re-reads,
# repeated state dumps). Marker:
#   [[cm:diff handle=<new> base=<base> role=<r> tokens=<n>]]\n<diff>\n[[/cm:diff]]
_DIFF_PREFIX = "[[cm:diff handle="
# Extracts the PRIMARY handle (the full content) from either a stored- or diff-stub.
_PRIMARY_HANDLE_RE = re.compile(r"\[\[cm:(?:stored|diff) handle=(?P<handle>\S+)")

# Auto-recall (Phase 10): ONE synthetic system message per request carrying store
# slices relevant to the LIVE TAIL (anticipatory demand paging — the model cannot
# page-fault on memory it cannot see). Marker:
#   [[cm:recall]]\n[[cm:recalled handle=<h>]]\n<slice>\n…\n[[/cm:recall]]
# Recall blocks are STRIPPED on entry and recomputed fresh each call, so at most one
# exists on the wire at any time and rewrite(rewrite(x)) cannot grow. `[[cm:recalled`
# deliberately does NOT match `_HANDLE_RE` (`[[cm:stored`), so Pass 2 never re-expands
# a recalled slice.
_RECALL_PREFIX = "[[cm:recall]]"

# Handles already VISIBLE on the wire (stored-/diff-stub headers, diff `base=`
# references, rehydrated markers). These are not recall candidates: auto-recall
# targets OFF-wire memory only (content the host CLI compacted away, evicted notes,
# MCP-saved state, prior sessions). Handles are filesystem-safe slugs, hence the
# explicit character class.
_ONWIRE_HANDLE_RE = re.compile(
    r"\[\[cm:(?:stored|diff|rehydrated) handle=(?P<h>[A-Za-z0-9._-]+)"
)
_DIFF_BASE_RE = re.compile(
    r"\[\[cm:diff handle=[A-Za-z0-9._-]+ base=(?P<h>[A-Za-z0-9._-]+)"
)

# Rough chars-per-token used ONLY to estimate the size of content too large to send to
# the tokenizer (the stub's `tokens=` field is informational; the decision to handle-ize
# such content is already certain from its char length).
_EST_CHARS_PER_TOKEN = 4


@dataclass
class RewriteResult:
    """Result of `PromptRewriter.rewrite_outgoing`.

    Attributes:
        messages: the rewritten OpenAI messages to send upstream.
        handle_ized_ids: ids of messages that were replaced by a stub this call.
        rehydrated_handles: handles whose content was paged back in this call.
        recalled_handles: handles auto-recalled into the Pass-4 block this call.
    """

    messages: list[dict]
    handle_ized_ids: list[str]
    rehydrated_handles: list[str]
    recalled_handles: list[str] = field(default_factory=list)


class PromptRewriter:
    """Pure, deterministic rewriter of an OpenAI messages list.

    Constructed with a `ProxyConfig`, a `TokenCounter`, and a `DurableStore`.
    `rewrite_outgoing` produces the rewritten messages plus bookkeeping lists.
    All the static helpers (`stable_id`, `make_stub`, `parse_handles`,
    `is_stub`) are deterministic and free of instance state.
    """

    def __init__(self, config: "ProxyConfig", counter: TokenCounter,
                 store: DurableStore, n_ctx: Optional[int] = None) -> None:
        # Import locally to avoid a circular import at module load time
        # (proxy/__init__. imports both config and rewriter).
        from .config import ProxyConfig
        if not isinstance(config, ProxyConfig):  # defensive, cheap
            raise TypeError("config must be a ProxyConfig")
        self.config = config
        self.counter = counter
        self.store = store
        # The upstream's true context size (from /props), if known. Enables the
        # total-budget windowing pass. None -> windowing disabled.
        self._n_ctx = n_ctx

    # ------------------------------------------------------------------ ids
    @staticmethod
    def stable_id(role: str, content: str) -> str:
        """Deterministic per-message id: same (role, content) -> same id.

        `id = "msg-" + sha1(role + "\\x00" + content).hexdigest()[:16]`.
        The NUL byte separates role from content so that pairs like
        ("ab", "c") and ("a", "bc") do not collide.
        """
        digest = hashlib.sha1((role + "\x00" + content).encode("utf-8")).hexdigest()
        return "msg-" + digest[:16]

    # ---------------------------------------------------------------- stubs
    @staticmethod
    def make_stub(handle: str, role: str, tokens: int, content: str,
                  preview_chars: int) -> str:
        """Render the deterministic stub text (spec §3.1).

        Format (when content length > 2*preview_chars):
            [[cm:stored handle=<handle> role=<role> tokens=<n>]]
            <preview-head>
            …(truncated <m> chars)…
            <preview-tail>
            [[/cm:stored]]

        When content length <= 2*preview_chars, the truncated-line and the
        tail are omitted (just the head, which is the whole content):
            [[cm:stored handle=<handle> role=<role> tokens=<n>]]
            <content>
            [[/cm:stored]]

        `<m>` is the number of characters omitted between head and tail.
        """
        header = f"[[cm:stored handle={handle} role={role} tokens={tokens}]]"
        footer = "[[/cm:stored]]"
        n = len(content)
        if n <= 2 * preview_chars:
            # Omit the truncated-line and the tail; the head IS the whole content.
            return f"{header}\n{content}\n{footer}"
        head = content[:preview_chars]
        # `content[len(content)-preview_chars:]` is the last `preview_chars`
        # chars; this is correct even when preview_chars == 0 (yields "").
        tail = content[n - preview_chars:]
        omitted = n - 2 * preview_chars
        return (
            f"{header}\n{head}\n"
            f"…(truncated {omitted} chars)…\n"
            f"{tail}\n{footer}"
        )

    @staticmethod
    def parse_handles(text: str) -> list[str]:
        """Return all handles referenced by `[[cm:stored handle=…]]` markers in
        `text`, in order of first occurrence, deduplicated."""
        seen: set[str] = set()
        out: list[str] = []
        for m in _HANDLE_RE.finditer(text):
            h = m.group("handle")
            if h not in seen:
                seen.add(h)
                out.append(h)
        return out

    @staticmethod
    def is_stub(content: str) -> bool:
        """True iff `content` (a string) begins, after optional leading
        whitespace, with `[[cm:stored handle=` — i.e. the message IS a stub we
        produced. Non-string input returns False (defensive; callers should
        ensure `content` is a str before relying on the str-typed contract)."""
        if not isinstance(content, str):
            return False
        return content.lstrip().startswith(_STUB_PREFIX)

    @staticmethod
    def is_rehydrated(content: str) -> bool:
        """True iff `content` is a str whose `lstrip()` starts with
        `[[cm:rehydrated handle=` — i.e. the message is one of the synthetic
        rehydrated messages produced by Pass 2 of `rewrite_outgoing`.

        Rehydrated messages are ALREADY-REWRITTEN output: they must never be
        re-handle-ized (Pass 1 leaves them untouched) and must never be
        re-rehydrated (Pass 2 skips handles already present as rehydrated
        markers). This is the §9.1 idempotency fix.
        """
        if not isinstance(content, str):
            return False
        return content.lstrip().startswith(_REHYDRATED_PREFIX)

    @staticmethod
    def is_recall(content: str) -> bool:
        """True iff `content` is a str whose `lstrip()` starts with `[[cm:recall]]`
        — i.e. the message is the Pass-4 auto-recall block. Recall blocks are
        already-rewritten output that must be STRIPPED on entry and recomputed
        (locality shifts every turn), never re-handle-ized or accumulated."""
        if not isinstance(content, str):
            return False
        return content.lstrip().startswith(_RECALL_PREFIX)

    @staticmethod
    def _parse_rehydrated_handle(content: str) -> Optional[str]:
        """Extract the handle from a `[[cm:rehydrated handle=H]]` marker at the
        start of `content` (after optional leading whitespace). Returns the
        handle string, or None if `content` does not start with such a marker.
        """
        if not isinstance(content, str):
            return None
        m = _REHYDRATED_RE.match(content.lstrip())
        return m.group("handle") if m is not None else None

    # ---------------------------------------------------------- diff-encoding
    @staticmethod
    def is_diff_stub(content: str) -> bool:
        """True iff `content` is a diff-stub we produced (already-rewritten output;
        passed through unchanged on re-entry, like a normal stub)."""
        if not isinstance(content, str):
            return False
        return content.lstrip().startswith(_DIFF_PREFIX)

    @staticmethod
    def make_diff_stub(handle: str, base_handle: str, role: str, tokens: int,
                       diff_text: str) -> str:
        """Render a diff-stub. The FULL content is still paged out under `handle`
        (lossless, rehydratable); the wire carries only a unified diff against
        `base_handle`'s stored content."""
        header = (
            f"[[cm:diff handle={handle} base={base_handle} "
            f"role={role} tokens={tokens}]]"
        )
        return f"{header}\n{diff_text}\n[[/cm:diff]]"

    @staticmethod
    def _primary_handle(content: str) -> Optional[str]:
        """Handle of the FULL content behind a stored- or diff-stub, else None."""
        if not isinstance(content, str):
            return None
        m = _PRIMARY_HANDLE_RE.match(content.lstrip())
        return m.group("handle") if m is not None else None

    @staticmethod
    def _unified_diff(base: str, new: str) -> str:
        """Deterministic, lossless line diff base->new (file-header lines dropped
        for compactness)."""
        lines = difflib.unified_diff(base.splitlines(), new.splitlines(), lineterm="", n=1)
        return "\n".join(ln for ln in lines if not ln.startswith(("--- ", "+++ ")))

    def _maybe_diff_stub(self, handle: str, role: str, tokens: int, content: str,
                         recent_stubs: list) -> Optional[str]:
        """Return a diff-stub if a recent same-role stored note is similar enough
        AND the diff comes out smaller than a normal stub; else None (caller falls
        back to ``make_stub``). Deterministic: depends only on prior messages, so it
        preserves the per-message prefix-stability invariant under append."""
        if self.config.diff_min_similarity <= 0.0:
            return None
        # SIZE GUARD (critical): difflib.SequenceMatcher is O(n*m) and pathological on
        # large, repetitive content (log files), and it runs synchronously on the request
        # path — without this cap a single bulky read can freeze the proxy for minutes.
        # Above the cap, fall back to a normal stub (still lossless, just no delta).
        cap = self.config.diff_max_chars
        if cap and len(content) > cap:
            return None
        # Most-recent `diff_lookback` same-role stubs, newest first.
        candidates = [h for (h, r) in recent_stubs if r == role][-self.config.diff_lookback:]
        best_handle: Optional[str] = None
        best_base: Optional[str] = None
        best_ratio = -1.0
        for h in reversed(candidates):
            if h == handle:
                continue
            try:
                base = self.store.get(h)
            except Exception:
                continue
            if cap and len(base) > cap:
                continue  # same O(n*m) guard for an oversized base note
            # autojunk=False (measured): the default heuristic marks "popular"
            # characters as junk on strings >= 200 chars — and every diff-stub
            # candidate is bulky by definition — collapsing a one-line file
            # re-read from a true ~0.999 similarity to a reported ~0.51. With
            # the default, near-duplicates routinely fell below
            # diff_min_similarity and lost their delta encoding entirely.
            ratio = difflib.SequenceMatcher(None, base, content, autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio, best_handle, best_base = ratio, h, base
        if best_handle is None or best_base is None or best_ratio < self.config.diff_min_similarity:
            return None
        diff_stub = self.make_diff_stub(
            handle, best_handle, role, tokens, self._unified_diff(best_base, content)
        )
        normal = self.make_stub(handle, role, tokens, content, self.config.stub_preview_chars)
        return diff_stub if len(diff_stub) < len(normal) else None

    def _count_for_handleization(self, content: str) -> Optional[int]:
        """Token count for the handle-ization decision, AVOIDING a /tokenize round-trip on
        the easy cases:
          - tokens <= chars ALWAYS, so content shorter than the threshold (in chars) can
            never reach the threshold in tokens -> return None (not bulky), no tokenize;
          - content larger than ``tokenize_max_chars`` is bulky-by-size and too big to POST
            to /tokenize (slow + a DoS risk) -> return a char-based ESTIMATE (clamped to at
            least the threshold so it is always handle-ized), no tokenize;
          - otherwise one exact count.
        Returning None means "below threshold, leave it alone".
        """
        n = len(content)
        threshold = self.config.handle_threshold_tokens
        if n < threshold:
            return None
        cap = self.config.tokenize_max_chars
        if cap and n > cap:
            return max(threshold, n // _EST_CHARS_PER_TOKEN)
        return self.counter.count_text(content)

    # -------------------------------------------------------------- rewrite
    def rewrite_outgoing(self, messages: list[dict]) -> RewriteResult:
        """Rewrite an OpenAI `messages` array (spec §3.4), deterministically.

        For each message, in order:
          1. If already a stub (`is_stub(content)`) -> leave as-is.
          2. Else if `content` is a str whose token count is >=
             `config.handle_threshold_tokens` -> page it out via the
             `DurableStore` (id = `stable_id`; `Message(role, content, id)`) and
             replace `content` with `make_stub(...)`; record the id in
             `handle_ized_ids`.
          3. Else pass through unchanged. If `content` is not a plain string
             (e.g. the OpenAI content-parts list) it is passed through
             untouched (never handle-ized).

        After the handle-ization pass, auto-rehydrate: scan the rewritten
        messages for explicit `[[cm:stored handle=H]]` references in NON-stub
        message content; for each referenced handle present in the store and
        not already expanded this call, append a synthetic message
        `{"role": "user", "content": "[[cm:rehydrated handle=H]]\\n<full>" }`
        (role "user" — strict chat templates reject mid-conversation "system")
        immediately AFTER the referencing message, subject to a running token
        budget of `config.rehydrate_budget_tokens` (truncate the last synthetic
        message via `counter.truncate_to_tokens` to fit; never exceed the
        budget). Unknown handles are skipped silently. Stubs themselves do NOT
        trigger auto-rehydration — only explicit references in non-stub
        content do.
        """
        handle_ized_ids: list[str] = []
        rewritten: list[dict] = []
        # (handle, role) of stored-/diff-stubs seen so far, for diff-base lookup.
        # Built in order, so a message's diff base depends only on EARLIER messages
        # (preserves the §3.5 per-message prefix-stability invariant under append).
        recent_stubs: list[tuple[str, str]] = []

        # ---- Pass 0: strip any previous auto-recall block (Phase 10) ----
        # Recall is recomputed fresh each call (the conversation's locality shifts
        # every turn), so a block from a prior pass over this list is dropped BEFORE
        # anything else runs. This is what makes Pass 4 non-accumulating: at most one
        # recall block ever exists, so rewrite(rewrite(x)) cannot grow.
        messages = [
            m for m in messages
            if not (isinstance(m, dict) and self.is_recall(m.get("content")))
        ]

        # ---- Pass 1: handle-ization (per-message deterministic) ----
        for msg in messages:
            # Preserve the message as a shallow copy so we never mutate input.
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            content = msg.get("content") if isinstance(msg, dict) else None

            if isinstance(content, str) and (self.is_stub(content) or self.is_diff_stub(content)):
                # Already a stub (stored or diff): leave byte-identical, but remember
                # its full-content handle as a candidate diff base for later messages.
                h = self._primary_handle(content)
                if h is not None:
                    recent_stubs.append((h, role))
                rewritten.append(dict(msg))
                continue

            if isinstance(content, str) and self.is_rehydrated(content):
                # Already a synthetic rehydrated message from a prior turn:
                # already-rewritten output. Pass through UNCHANGED (never
                # handle-ize, never page out) — §9.1 idempotency fix.
                rewritten.append(dict(msg))
                continue

            if isinstance(content, str):
                # ONE token measurement, and skip the /tokenize round-trip entirely on the
                # easy/dangerous cases (tiny content can't be bulky; huge content is
                # bulky-by-size and unsafe to tokenize). None => below threshold.
                tokens = self._count_for_handleization(content)
                if tokens is not None and tokens >= self.config.handle_threshold_tokens:
                    mid = self.stable_id(role, content)
                    handle = self.store.page_out(  # idempotent: same id -> same handle
                        Message(role=role, content=content, id=mid)
                    )
                    # Delta-compress against a recent near-duplicate of the same role if
                    # one exists and the diff is smaller; else a normal head/tail stub.
                    stub = self._maybe_diff_stub(handle, role, tokens, content, recent_stubs)
                    if stub is None:
                        stub = self.make_stub(
                            handle, role, tokens, content, self.config.stub_preview_chars
                        )
                    rewritten.append({"role": role, "content": stub})
                    handle_ized_ids.append(mid)
                    recent_stubs.append((handle, role))
                    continue

            # Pass through unchanged (covers non-string content too).
            rewritten.append(dict(msg) if isinstance(msg, dict) else msg)

        # ---- Pass 2: auto-rehydration of explicit references ----
        rehydrated_handles: list[str] = []
        if self.config.rehydrate_budget_tokens > 0:
            budget = self.config.rehydrate_budget_tokens
            used = 0
            out: list[dict] = []

            # §9.1 #3: collect handles already expanded as rehydrated markers
            # in the working message list (from prior turns) so we never append
            # a duplicate synthetic message for a handle that is already
            # present inline. This is the idempotency fix for Pass 2.
            already_rehydrated: set[str] = set()
            for msg in rewritten:
                c = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(c, str) and self.is_rehydrated(c):
                    h = self._parse_rehydrated_handle(c)
                    if h is not None:
                        already_rehydrated.add(h)

            for msg in rewritten:
                out.append(msg)
                content = msg.get("content") if isinstance(msg, dict) else None
                if not isinstance(content, str):
                    continue
                if self.is_stub(content):
                    # Stubs do not auto-rehydrate.
                    continue
                handles = self.parse_handles(content)
                for h in handles:
                    # Skip handles we already rehydrated earlier this call or
                    # that are already present as rehydrated markers.
                    if h in rehydrated_handles or h in already_rehydrated:
                        continue
                    remaining = budget - used
                    # M2: stop the loop only when the budget is exhausted.
                    if remaining <= 0:
                        break
                    # Load the full content from the store. Unknown/missing
                    # handle -> skip silently, no crash.
                    try:
                        full = self.store.get(h)
                    except Exception:
                        continue
                    if full is None:
                        continue
                    # L2: the `[[cm:rehydrated handle=H]]` marker line is NEVER
                    # truncated — only the `<full>` body is. This keeps the
                    # marker intact so the message stays detectable next turn.
                    marker = f"[[cm:rehydrated handle={h}]]\n"
                    marker_tokens = self.counter.count_text(marker)
                    # M2: if the marker alone already exceeds the remaining
                    # budget, this handle does not fit — try the next one
                    # (continue) rather than aborting the whole loop.
                    if marker_tokens >= remaining:
                        continue
                    body_budget = remaining - marker_tokens
                    body_tokens = self.counter.count_text(full)
                    if body_tokens <= body_budget:
                        body = full
                    else:
                        body = self.counter.truncate_to_tokens(full, body_budget)
                    synth = marker + body
                    count = self.counter.count_text(synth)
                    # Defensive: never exceed the budget and never emit empty.
                    if count <= 0 or count > remaining:
                        continue
                    # Role "user", NOT "system" (learned live 2026-07-01): strict
                    # chat templates (Qwen: "System message must be at the
                    # beginning") reject a system message anywhere but index 0.
                    # "user" is the only role every template accepts mid-wire;
                    # the [[cm:...]] marker keeps the content distinguishable.
                    out.append({"role": "user", "content": synth})
                    used += count
                    rehydrated_handles.append(h)
                    already_rehydrated.add(h)
                    if used >= budget:
                        break
            rewritten = out

        # ---- Pass 3: total-budget windowing (LOSSLESS pre-compaction) ----
        # Bound the TOTAL wire to context_budget_ratio * n_ctx by paging out the
        # OLDEST non-pinned middle messages (head + recent tail kept verbatim). The
        # paged-out content becomes a retrievable stub -> lossless. This pre-empts the
        # host CLI's own (LOSSY) summarization-compaction so it rarely needs to fire.
        # Prefix-stable: a message switches to a stub at most ONCE (when it crosses
        # the budget boundary) and stays a stub thereafter.
        if self._n_ctx and self.config.context_budget_ratio > 0.0:
            budget = int(self._n_ctx * self.config.context_budget_ratio)
            tail_start = len(rewritten) - self.config.protect_last_n
            # Cheap gate: estimate tokens as chars/4 to avoid per-message tokenizer
            # calls on the common under-budget case; only go precise when plausibly over.
            est_chars = sum(
                len(m["content"]) for m in rewritten
                if isinstance(m, dict) and isinstance(m.get("content"), str)
            )
            if est_chars / 4 > budget and self.config.protect_first_n < tail_start:
                # Count each message ONCE (count_text may be a network call), then
                # maintain a running total as messages are paged out.
                counts = [
                    self.counter.count_text(m["content"])
                    if isinstance(m, dict) and isinstance(m.get("content"), str) else 0
                    for m in rewritten
                ]
                total = sum(counts)
                i = self.config.protect_first_n
                while total > budget and i < tail_start:
                    msg = rewritten[i]
                    content = msg.get("content") if isinstance(msg, dict) else None
                    role = msg.get("role", "") if isinstance(msg, dict) else ""
                    if (
                        isinstance(content, str)
                        and not self.is_stub(content)
                        and not self.is_diff_stub(content)
                        and not self.is_rehydrated(content)
                    ):
                        mid = self.stable_id(role, content)
                        handle = self.store.notes.handle_for(mid)
                        # Minimal stub (no preview) -> the archive marker is tiny.
                        minimal = self.make_stub(handle, role, counts[i], content, 0)
                        stub_tokens = self.counter.count_text(minimal)
                        if stub_tokens < counts[i]:  # never bloat a tiny old message
                            self.store.page_out(Message(role=role, content=content, id=mid))
                            rewritten[i] = {"role": role, "content": minimal}
                            total += stub_tokens - counts[i]
                            counts[i] = stub_tokens
                            handle_ized_ids.append(mid)
                    i += 1

        # ---- Pass 4: auto-recall (anticipatory demand paging, Phase 10) ----
        # The live run proved agents do not ask for their memory back
        # (messages_rehydrated: 0) — and a model cannot page-fault on content it
        # cannot see. So the governor recalls FOR it: derive an implicit query from
        # the live tail (locality), search the store, and inject the top slices of
        # OFF-wire memory as one budgeted, clearly-marked system message right
        # before the final message (KV impact bounded to the tail). Recall flows
        # through store.search(), so retrieval metrics and hotness warming come
        # free — recall feeds the working-set signal that drives eviction.
        recalled_handles: list[str] = []
        if (self.config.auto_recall_k > 0
                and self.config.recall_budget_tokens > 0 and rewritten):
            try:
                block, recalled_handles = self._build_recall_block(rewritten)
            except Exception:
                # Recall is ENRICHMENT, never a dependency: a store/tokenizer
                # hiccup degrades to "no recall this turn" — it must never fail
                # the request. (Learned live 2026-07-01: a /tokenize parse error
                # inside the recall builder 500'd every chat completion.)
                block, recalled_handles = None, []
            if block is not None:
                insert_at = len(rewritten) - 1 if len(rewritten) >= 2 else len(rewritten)
                # Role "user", NOT "system": strict templates reject mid-wire
                # system messages (same live lesson as Pass 2 above).
                rewritten.insert(insert_at, {"role": "user", "content": block})

        return RewriteResult(
            messages=rewritten,
            handle_ized_ids=handle_ized_ids,
            rehydrated_handles=rehydrated_handles,
            recalled_handles=recalled_handles,
        )

    # ---------------------------------------------------------- auto-recall
    def _on_wire_handles(self, messages: list[dict]) -> set:
        """Every handle the current wire already carries — as a stored-/diff-stub,
        a diff ``base=`` reference, a rehydrated marker, OR as verbatim content
        (a message that WOULD map to that handle if paged out). Recall must never
        duplicate what the model can already see."""
        on_wire: set = set()
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, str):
                continue
            for m in _ONWIRE_HANDLE_RE.finditer(content):
                on_wire.add(m.group("h"))
            for m in _DIFF_BASE_RE.finditer(content):
                on_wire.add(m.group("h"))
            if not (self.is_stub(content) or self.is_diff_stub(content)
                    or self.is_rehydrated(content)):
                # Verbatim message: its content lives on the wire even if a copy
                # was paged out in some earlier turn — computing its would-be
                # handle is one local sha1, no I/O.
                role = msg.get("role", "")
                on_wire.add(self.store.notes.handle_for(self.stable_id(role, content)))
        return on_wire

    def _build_recall_block(self, messages: list[dict]) -> tuple[Optional[str], list[str]]:
        """Assemble the Pass-4 recall block: implicit tail query -> store search ->
        off-wire filter -> near-duplicate suppression -> budgeted assembly (the
        marker lines are never truncated, only slice bodies — same discipline as
        Pass 2). Returns ``(block, handles)`` or ``(None, [])`` when there is
        nothing worth recalling (empty/trivial query, no hits, everything already
        on the wire, or nothing fits the budget). Deterministic given the message
        list and the store's state."""
        query = extract_query(messages)
        if not query:
            return None, []
        k = self.config.auto_recall_k
        try:
            slices = self.store.search(query, k=k * 3)  # pool for the filters below
        except Exception:
            return None, []
        if not slices:
            return None, []
        on_wire = self._on_wire_handles(messages)
        candidates = [s for s in slices if s.handle not in on_wire]
        if not candidates:
            return None, []
        keep = select_diverse([s.content for s in candidates])
        candidates = [candidates[i] for i in keep][:k]

        header = "[[cm:recall]]"
        footer = "[[/cm:recall]]"
        budget = self.config.recall_budget_tokens
        used = self.counter.count_text(f"{header}\n{footer}")
        parts: list[str] = [header]
        picked: list[str] = []
        for sl in candidates:
            marker = f"[[cm:recalled handle={sl.handle}]]\n"
            marker_tokens = self.counter.count_text(marker)
            remaining = budget - used
            if marker_tokens >= remaining:
                continue
            body_budget = remaining - marker_tokens
            body_tokens = self.counter.count_text(sl.content)
            body = (sl.content if body_tokens <= body_budget
                    else self.counter.truncate_to_tokens(sl.content, body_budget))
            piece = marker + body
            count = self.counter.count_text(piece)
            if count <= 0 or count > remaining:
                continue
            parts.append(piece)
            picked.append(sl.handle)
            used += count
            if used >= budget:
                break
        if not picked:
            return None, []
        parts.append(footer)
        return "\n".join(parts), picked
