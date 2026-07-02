"""Pure invariant tests for ``PromptRewriter`` — the core of Phase 3.

Binds to ``tasks/phase3-spec.md`` §3 (NORMATIVE). NO network, NO FastAPI/httpx.
Uses the Phase 1 ``FakeCounter`` (token = whitespace word count; truncate =
first-N-words) and a real ``DurableStore(tmp_path)``.

Invariants proven (spec §3.5, §7):
  - bulky message (>= threshold) -> replaced by a stub; full content retrievable
    from the store via the stub's handle; small message passes through verbatim.
  - make_stub / parse_handles / is_stub round-trip.
  - IDEMPOTENCY: rewrite_outgoing(rewrite_outgoing(M).messages).messages == R1
    on a mixed list (small + bulky + already-stub).
  - PER-MESSAGE DETERMINISM / prefix stability: rewriting M then M + [extras]
    yields identical rewritten content for every shared leading message.
  - content as OpenAI content-parts LIST -> passed through untouched (never
    handle-ized), even when "large".
  - auto-rehydration: explicit ``[[cm:stored handle=H]]`` reference -> a
    ``[[cm:rehydrated handle=H]]`` system message appended after the referrer
    with the full content; respects rehydrate_budget_tokens; unknown handle
    skipped without exception.
  - stable_id determinism: same (role, content) -> same id; different -> different.
"""

from __future__ import annotations

import pytest

from conftest import FakeCounter
from contextmanager.durable import DurableStore
from contextmanager.proxy.config import ProxyConfig
from contextmanager.proxy.rewriter import PromptRewriter
from contextmanager.types import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(tmp_path, **over) -> ProxyConfig:
    kw = dict(
        upstream_base_url="http://upstream.test",
        store_root=str(tmp_path / "store"),
        handle_threshold_tokens=10,
        stub_preview_chars=10,
        rehydrate_budget_tokens=4000,
        request_timeout=30.0,
    )
    kw.update(over)
    return ProxyConfig(**kw)


def _rewriter(tmp_path, **over):
    """Return (config, store, counter, rewriter) wired with FakeCounter + real store."""
    cfg = _config(tmp_path, **over)
    store = DurableStore(cfg.store_root)
    counter = FakeCounter()
    return cfg, store, counter, PromptRewriter(cfg, counter, store)


def _bulky_content(n_words: int = 50) -> str:
    """Plain-string content with `n_words` words (>= threshold)."""
    return " ".join(f"word{i}" for i in range(n_words))


# ---------------------------------------------------------------------------
# make_stub / parse_handles / is_stub round-trip (spec §3.1, §3.3)
# ---------------------------------------------------------------------------


class TestStubHelpers:
    def test_make_stub_round_trip_long_content(self):
        handle = "abc-123"
        content = "x" * 100  # > 2 * stub_preview_chars -> full form with head/tail
        stub = PromptRewriter.make_stub(handle, "user", 5, content, 20)
        assert PromptRewriter.parse_handles(stub) == [handle]
        assert PromptRewriter.is_stub(stub) is True

    def test_make_stub_round_trip_short_content(self):
        # content <= 2 * preview -> the truncation line + tail are omitted.
        handle = "short-h"
        stub = PromptRewriter.make_stub(handle, "user", 1, "short body", 20)
        assert PromptRewriter.parse_handles(stub) == [handle]
        assert PromptRewriter.is_stub(stub) is True

    def test_is_stub_false_on_plain_text(self):
        assert PromptRewriter.is_stub("hello world") is False
        assert PromptRewriter.is_stub("") is False

    def test_parse_handles_empty_and_none(self):
        assert PromptRewriter.parse_handles("no handles here") == []
        assert PromptRewriter.parse_handles("") == []

    def test_parse_handles_multiple(self):
        # Two distinct handles in one text -> both, in order.
        txt = (
            "[[cm:stored handle=h1 role=user tokens=5]] "
            "middle "
            "[[cm:stored handle=h2 role=assistant tokens=7]]"
        )
        assert PromptRewriter.parse_handles(txt) == ["h1", "h2"]


# ---------------------------------------------------------------------------
# bulky -> stub + retrievable; small -> passthrough (spec §3.4 steps 2,3)
# ---------------------------------------------------------------------------


class TestHandleIzation:
    def test_bulky_message_replaced_by_stub_and_retrievable(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path)
        bulky = _bulky_content(50)
        res = rw.rewrite_outgoing([{"role": "user", "content": bulky}])

        assert len(res.messages) == 1
        out = res.messages[0]["content"]
        assert isinstance(out, str)
        assert PromptRewriter.is_stub(out)
        handles = PromptRewriter.parse_handles(out)
        assert len(handles) == 1
        # Full original content is retrievable from the store by the stub handle.
        assert store.get(handles[0]) == bulky
        # The original blob is NOT forwarded verbatim (only head/tail preview).
        assert bulky not in out
        # The message id was recorded as handle-ized.
        assert len(res.handle_ized_ids) == 1
        assert res.handle_ized_ids[0].startswith("msg-")
        # No rehydration happened.
        assert res.rehydrated_handles == []

    def test_small_message_passes_through_unchanged(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path)
        small = "hello world"  # 2 words < threshold
        res = rw.rewrite_outgoing([{"role": "user", "content": small}])
        assert res.messages == [{"role": "user", "content": small}]
        assert res.handle_ized_ids == []
        assert res.rehydrated_handles == []

    def test_threshold_is_inclusive(self, tmp_path):
        # count_text == handle_threshold_tokens is bulky (>= per spec §3.4 step 2).
        cfg, store, counter, rw = _rewriter(tmp_path, handle_threshold_tokens=10)
        exactly_threshold = " ".join(f"w{i}" for i in range(10))  # 10 words
        res = rw.rewrite_outgoing([{"role": "user", "content": exactly_threshold}])
        assert PromptRewriter.is_stub(res.messages[0]["content"])
        assert len(res.handle_ized_ids) == 1


# ---------------------------------------------------------------------------
# Idempotency (spec §3.5 headline invariant)
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_rewrite_is_idempotent_on_mixed_list(self, tmp_path):
        """R1 = rewrite_outgoing(M).messages; rewrite_outgoing(R1).messages == R1.

        Mixed list: small + bulky + already-stub. No explicit handle references
        so rehydration does not perturb the second pass.
        """
        cfg, store, counter, rw = _rewriter(tmp_path, stub_preview_chars=10)

        small = "hi there"
        bulky = _bulky_content(50)

        # An already-stub from a pre-existing stored note (different content).
        pre_msg = Message(
            role="user",
            content=" ".join(f"pre{i}" for i in range(30)),
            id="msg-preexisting001",
        )
        pre_handle = store.page_out(pre_msg)
        already_stub = PromptRewriter.make_stub(
            pre_handle, "user", 30, pre_msg.content, 10
        )

        M = [
            {"role": "user", "content": small},
            {"role": "user", "content": bulky},
            {"role": "assistant", "content": already_stub},
        ]
        R1 = rw.rewrite_outgoing(M).messages
        R2 = rw.rewrite_outgoing(R1).messages

        assert R2 == R1, "second rewrite pass must be a no-op on already-rewritten output"

    def test_already_stub_is_not_re_handleized(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path, stub_preview_chars=10)
        pre = Message(role="user", content=" ".join(f"p{i}" for i in range(40)),
                      id="msg-presize001")
        h = store.page_out(pre)
        stub = PromptRewriter.make_stub(h, "user", 40, pre.content, 10)
        res = rw.rewrite_outgoing([{"role": "user", "content": stub}])
        # Output content is byte-identical to the input stub (no re-page-out).
        assert res.messages[0]["content"] == stub
        assert res.handle_ized_ids == []
        assert res.rehydrated_handles == []


# ---------------------------------------------------------------------------
# Per-message determinism / prefix stability (spec §3.5)
# ---------------------------------------------------------------------------


class TestPrefixStability:
    def test_appended_messages_do_not_change_rewritten_prefix(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path, stub_preview_chars=10)
        small = "hello"  # 1 word, plain
        bulky_a = _bulky_content(50)
        bulky_b = " ".join(f"extra{i}" for i in range(60))  # different bulky content

        M1 = [
            {"role": "user", "content": small},
            {"role": "user", "content": bulky_a},
        ]
        M2 = M1 + [{"role": "user", "content": bulky_b}]

        R1 = rw.rewrite_outgoing(M1).messages
        R2 = rw.rewrite_outgoing(M2).messages

        assert len(R1) == 2
        assert len(R2) == 3
        # Every shared leading message is byte-identical (KV-cache prefix stable).
        for i in range(len(R1)):
            assert R1[i] == R2[i], f"prefix diverged at index {i}"

    def test_same_content_same_role_produces_same_stub_bytes(self, tmp_path):
        """Re-sending the same (role, content) yields the SAME handle -> same stub."""
        cfg, store, counter, rw = _rewriter(tmp_path, stub_preview_chars=10)
        bulky = _bulky_content(50)
        r1 = rw.rewrite_outgoing([{"role": "user", "content": bulky}]).messages[0]["content"]
        r2 = rw.rewrite_outgoing([{"role": "user", "content": bulky}]).messages[0]["content"]
        assert r1 == r2
        assert PromptRewriter.parse_handles(r1) == PromptRewriter.parse_handles(r2)


# ---------------------------------------------------------------------------
# content-parts list passthrough (spec §3.4: non-str content -> untouched)
# ---------------------------------------------------------------------------


class TestContentPartsList:
    def test_content_parts_list_is_never_handle_ized(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path)
        # A content-parts list whose embedded text is "large" (well over
        # threshold) must still pass through untouched.
        big_text = " ".join(f"part{i}" for i in range(500))
        parts = [{"type": "text", "text": big_text}]
        res = rw.rewrite_outgoing([{"role": "user", "content": parts}])
        assert res.messages[0]["content"] == parts  # identity by value
        assert res.handle_ized_ids == []
        assert res.rehydrated_handles == []

    def test_content_parts_list_mixed_with_bulky_string(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path, stub_preview_chars=10)
        bulky = _bulky_content(50)
        parts = [{"type": "text", "text": "small"}]
        res = rw.rewrite_outgoing(
            [
                {"role": "user", "content": parts},
                {"role": "user", "content": bulky},
            ]
        )
        # parts pass through; bulky string is handle-ized.
        assert res.messages[0]["content"] == parts
        assert PromptRewriter.is_stub(res.messages[1]["content"])
        assert len(res.handle_ized_ids) == 1


# ---------------------------------------------------------------------------
# Auto-rehydration (spec §3.4 step 4)
# ---------------------------------------------------------------------------


class TestAutoRehydration:
    def test_known_handle_reference_rehydrates_after_referrer(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path)
        stored = " ".join(f"s{i}" for i in range(20))  # 20 words
        H = store.page_out(Message(role="user", content=stored, id="msg-storedhandle01"))

        ref = f"Please explain [[cm:stored handle={H} role=user tokens=5]]"
        res = rw.rewrite_outgoing([{"role": "user", "content": ref}])

        assert res.rehydrated_handles == [H]
        assert len(res.messages) == 2
        # The referrer is unchanged.
        assert res.messages[0]["content"] == ref
        # A synthetic system message is appended right after it.
        rehy = res.messages[1]
        # Role "user", not "system" (live 2026-07-01): strict chat templates
        # (Qwen) reject a system message anywhere but position 0.
        assert rehy["role"] == "user"
        content = rehy["content"]
        assert content.startswith(f"[[cm:rehydrated handle={H}]]")
        # Full content is present (budget 4000 >> 20 words).
        assert stored in content

    def test_rehydrate_budget_truncates_content(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path, rehydrate_budget_tokens=10)
        stored = " ".join(f"s{i}" for i in range(40))  # 40 words > budget
        H = store.page_out(Message(role="user", content=stored, id="msg-budget001"))

        ref = f"see [[cm:stored handle={H} role=user tokens=5]]"
        res = rw.rewrite_outgoing([{"role": "user", "content": ref}])

        assert res.rehydrated_handles == [H]
        rehy = res.messages[1]["content"]
        prefix = f"[[cm:rehydrated handle={H}]]"
        assert rehy.startswith(prefix)
        body = rehy[len(prefix):].lstrip("\n ").rstrip()
        # The body fits within the token budget (word-count tokens).
        assert len(body.split()) <= 10
        # The body is a prefix of the original (truncation keeps the head).
        assert stored.startswith(body)
        # It was actually truncated.
        assert body != stored

    def test_unknown_handle_is_skipped_without_exception(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path)
        ref = "see [[cm:stored handle=does-not-exist-xyz role=user tokens=5]]"
        # Must not raise.
        res = rw.rewrite_outgoing([{"role": "user", "content": ref}])
        assert res.rehydrated_handles == []
        # No rehydrated system message was appended.
        assert all("cm:rehydrated" not in m.get("content", "") for m in res.messages)
        assert len(res.messages) == 1

    def test_stub_messages_do_not_auto_rehydrate(self, tmp_path):
        """A stub (handle-ized message) is NOT an explicit reference: it must
        NOT trigger rehydration (spec §3.4 step 4 — only NON-stub references)."""
        cfg, store, counter, rw = _rewriter(tmp_path, stub_preview_chars=10)
        bulky = _bulky_content(50)
        res = rw.rewrite_outgoing([{"role": "user", "content": bulky}])
        # The bulky became a stub; no rehydrated message should be appended.
        assert res.rehydrated_handles == []
        assert len(res.messages) == 1


# ---------------------------------------------------------------------------
# stable_id determinism (spec §3.3, §3.1)
# ---------------------------------------------------------------------------


class TestStableId:
    def test_same_role_content_same_id(self):
        a = PromptRewriter.stable_id("user", "hello")
        b = PromptRewriter.stable_id("user", "hello")
        assert a == b
        assert a.startswith("msg-")

    def test_different_content_different_id(self):
        assert PromptRewriter.stable_id("user", "hello") != PromptRewriter.stable_id(
            "user", "world"
        )

    def test_different_role_different_id(self):
        assert PromptRewriter.stable_id("user", "hello") != PromptRewriter.stable_id(
            "assistant", "hello"
        )


# ---------------------------------------------------------------------------
# Round-2 corrections (spec §9.3) — explicit-reference idempotency + multi-turn
# no-growth. These pin the §9.1 fix: a rehydrated synthetic message is already-
# rewritten output and must not be re-handle-ized / re-rehydrated, so the
# conversation does not recreate the very livelock the proxy exists to kill.
# ---------------------------------------------------------------------------


class TestRound2Idempotency:
    def test_idempotency_with_explicit_reference(self, tmp_path):
        """Spec §9.3 test_idempotency_with_explicit_reference.

        Build a real handle H in the store, build a messages list whose user
        message EXPLICITLY references H via ``[[cm:stored handle=H ...]]``. A
        first rewrite pass rehydrates H (appends a synthetic system message).
        A second pass over that output MUST be a no-op: the synthetic
        rehydrated message is already-rewritten output (Pass 1 leaves it
        untouched; Pass 2 sees the rehydrated marker and skips re-expanding H).
        """
        cfg = _config(tmp_path)
        store = DurableStore(cfg.store_root)
        counter = FakeCounter()
        rw = PromptRewriter(cfg, counter, store)

        # Page out a bulky Message to get a REAL handle H (deterministic).
        seed = Message(
            role="user",
            content=" ".join(["word"] * 20),  # 20 words >= threshold(10)
            id="seed",
        )
        H = store.page_out(seed)

        M = [
            {
                "role": "user",
                "content": f"please look at [[cm:stored handle={H} role=user tokens=5]] thanks",
            }
        ]
        R1 = rw.rewrite_outgoing(M)
        R2 = rw.rewrite_outgoing(R1.messages)

        assert R2.messages == R1.messages, (
            "second rewrite pass must be a no-op on the already-rewritten "
            "output, including the synthetic rehydrated message"
        )
        assert len(R2.messages) == len(R1.messages)

    def test_multiturn_no_growth(self, tmp_path):
        """Spec §9.3 test_multiturn_no_growth.

        Simulate 4 turns where each turn feeds the previous rewrite output
        back as the next input AND appends a NEW small user message (to model
        the conversation growing). The count of stub+rehydrated messages in
        the carried-over prefix MUST stay constant across turns: the explicit
        reference yields exactly ONE rehydrated message that persists, not
        one-per-turn (which would grow unboundedly — the livelock the proxy
        exists to kill).
        """
        cfg = _config(tmp_path)
        store = DurableStore(cfg.store_root)
        counter = FakeCounter()
        rw = PromptRewriter(cfg, counter, store)

        seed = Message(
            role="user",
            content=" ".join(["word"] * 20),
            id="seed",
        )
        H = store.page_out(seed)

        def _count_stub_rehy(msgs: list[dict]) -> int:
            return sum(
                1
                for m in msgs
                if PromptRewriter.is_stub(m.get("content", ""))
                or PromptRewriter.is_rehydrated(m.get("content", ""))
            )

        # Turn 1 input = the explicit-reference list.
        msgs = [
            {
                "role": "user",
                "content": f"please look at [[cm:stored handle={H} role=user tokens=5]] thanks",
            }
        ]

        counts: list[int] = []
        for _ in range(4):
            result = rw.rewrite_outgoing(msgs)
            counts.append(_count_stub_rehy(result.messages))
            # Next turn: feed the rewritten output back as the prefix, and
            # append one NEW small user message simulating the conversation.
            msgs = result.messages + [{"role": "user", "content": "ok"}]

        # After turn 1 there is exactly one rehydrated message (no stubs).
        assert counts[0] == 1, f"turn 1 count = {counts[0]} (expected 1)"
        # The count is the SAME after turn 2, 3, 4 as after turn 1 — the
        # carried-over prefix does NOT grow turn-over-turn.
        assert counts[1] == counts[0], (
            f"turn 2 count {counts[1]} grew vs turn 1 {counts[0]}"
        )
        assert counts[2] == counts[0], (
            f"turn 3 count {counts[2]} grew vs turn 1 {counts[0]}"
        )
        assert counts[3] == counts[0], (
            f"turn 4 count {counts[3]} grew vs turn 1 {counts[0]}"
        )


# ---------------------------------------------------------------------------
# Diff-encoding (lossless delta compression of near-duplicate content)
# ---------------------------------------------------------------------------


def _file_v(n_lines: int, patched_line: int | None = None) -> str:
    lines = [f"def func_{i}(): return {i}" for i in range(n_lines)]
    if patched_line is not None:
        lines[patched_line] = f"def func_{patched_line}(): return 999  # patched"
    return "\n".join(lines)


class TestDiffEncoding:
    def test_near_duplicate_becomes_diff_stub_and_stays_lossless(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path, stub_preview_chars=200, diff_min_similarity=0.5)
        v1 = _file_v(100)
        v2 = _file_v(100, patched_line=50)  # one line changed
        r1 = rw.rewrite_outgoing([{"role": "user", "content": v1}])
        # feed the v1 stub back (as a real multi-turn flow does) + the re-read v2
        r2 = rw.rewrite_outgoing(r1.messages + [{"role": "user", "content": v2}])
        last = r2.messages[-1]["content"]
        assert PromptRewriter.is_diff_stub(last)
        # the diff is far smaller than the original content
        assert len(last) < len(v2) // 2
        # LOSSLESS: full v2 is still retrievable from the store via the stub's handle
        handle = PromptRewriter._primary_handle(last)
        assert store.get(handle) == v2
        # the diff actually shows the change (informative)
        assert "patched" in last and "+def func_50" in last

    def test_falls_back_to_normal_stub_when_no_prior_base(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path, stub_preview_chars=200, diff_min_similarity=0.5)
        r = rw.rewrite_outgoing([{"role": "user", "content": _file_v(100)}])
        assert PromptRewriter.is_stub(r.messages[0]["content"])
        assert not PromptRewriter.is_diff_stub(r.messages[0]["content"])

    def test_high_threshold_still_diff_encodes_near_duplicate(self, tmp_path):
        # Regression (Phase 10): SequenceMatcher's default autojunk marks "popular"
        # characters as junk on strings >= 200 chars — every diff-stub candidate —
        # collapsing a one-line re-read's TRUE ~0.999 similarity to a reported
        # ~0.51. Under autojunk, this test fails (0.51 < 0.8); with autojunk=False
        # the near-duplicate is seen for what it is and delta-encodes even under a
        # strict threshold.
        cfg, store, counter, rw = _rewriter(
            tmp_path, stub_preview_chars=200, diff_min_similarity=0.8
        )
        v1 = _file_v(100)
        v2 = _file_v(100, patched_line=50)  # one line changed -> truly ~0.999 similar
        r1 = rw.rewrite_outgoing([{"role": "user", "content": v1}])
        r2 = rw.rewrite_outgoing(r1.messages + [{"role": "user", "content": v2}])
        last = r2.messages[-1]["content"]
        assert PromptRewriter.is_diff_stub(last)
        assert store.get(PromptRewriter._primary_handle(last)) == v2  # lossless

    def test_falls_back_when_content_too_different(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path, stub_preview_chars=200, diff_min_similarity=0.8)
        a = _file_v(100)
        b = " ".join(f"totally{i} different{i}" for i in range(120))  # unrelated, bulky
        r1 = rw.rewrite_outgoing([{"role": "user", "content": a}])
        r2 = rw.rewrite_outgoing(r1.messages + [{"role": "user", "content": b}])
        assert not PromptRewriter.is_diff_stub(r2.messages[-1]["content"])

    def test_diff_disabled_when_similarity_zero(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path, stub_preview_chars=200, diff_min_similarity=0.0)
        v1 = _file_v(100)
        v2 = _file_v(100, patched_line=50)
        r1 = rw.rewrite_outgoing([{"role": "user", "content": v1}])
        r2 = rw.rewrite_outgoing(r1.messages + [{"role": "user", "content": v2}])
        assert not PromptRewriter.is_diff_stub(r2.messages[-1]["content"])

    def test_size_guard_skips_diff_for_oversized_content(self, tmp_path):
        # The fix for the log-file brain-lock: with a small diff_max_chars, a near-duplicate
        # re-read must NOT diff-encode (the O(n*m) difflib path is skipped) — it falls back
        # to a normal, lossless stub instead of freezing the proxy.
        cfg, store, counter, rw = _rewriter(
            tmp_path, stub_preview_chars=200, diff_min_similarity=0.5, diff_max_chars=50
        )
        v1 = _file_v(100)                       # ~2500 chars >> 50
        v2 = _file_v(100, patched_line=50)
        r1 = rw.rewrite_outgoing([{"role": "user", "content": v1}])
        r2 = rw.rewrite_outgoing(r1.messages + [{"role": "user", "content": v2}])
        last = r2.messages[-1]["content"]
        assert PromptRewriter.is_stub(last)
        assert not PromptRewriter.is_diff_stub(last)        # guard fired -> no difflib
        assert store.get(PromptRewriter._primary_handle(last)) == v2  # still lossless

    def test_default_cap_still_diff_encodes_normal_files(self, tmp_path):
        # The default cap (20000) sits well above a typical file, so ordinary re-reads
        # still benefit from diff-encoding — the guard only excludes pathological sizes.
        cfg, store, counter, rw = _rewriter(
            tmp_path, stub_preview_chars=200, diff_min_similarity=0.5
        )
        v1 = _file_v(100)
        v2 = _file_v(100, patched_line=50)
        r1 = rw.rewrite_outgoing([{"role": "user", "content": v1}])
        r2 = rw.rewrite_outgoing(r1.messages + [{"role": "user", "content": v2}])
        assert PromptRewriter.is_diff_stub(r2.messages[-1]["content"])

    def test_idempotent_and_prefix_stable_with_diff_stub(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path, stub_preview_chars=200, diff_min_similarity=0.5)
        v1 = _file_v(100)
        v2 = _file_v(100, patched_line=50)
        r1 = rw.rewrite_outgoing([{"role": "user", "content": v1}])
        r2 = rw.rewrite_outgoing(r1.messages + [{"role": "user", "content": v2}])
        # idempotent: a diff-stub is already-rewritten output, passed through
        assert rw.rewrite_outgoing(r2.messages).messages == r2.messages
        # prefix-stable under append: the shared leading messages are unchanged
        assert r2.messages[: len(r1.messages)] == r1.messages


# ---------------------------------------------------------------------------
# Budget-windowing (lossless total-budget bound) — Pass 3
# ---------------------------------------------------------------------------


class TestBudgetWindowing:
    def _msgs(self):
        # 6 messages of 20 words each (each below a high handle threshold so Pass 1
        # leaves them verbatim; Pass 3 windows the middle).
        m = lambda label: {"role": "user", "content": " ".join([label] * 20)}
        return [m("head"), m("mid1"), m("mid2"), m("mid3"), m("tail1"), m("tail2")]

    def _rw(self, tmp_path, n_ctx, **over):
        cfg = _config(tmp_path, handle_threshold_tokens=1000, stub_preview_chars=200,
                      protect_first_n=1, protect_last_n=2, **over)
        store = DurableStore(cfg.store_root)
        return cfg, store, PromptRewriter(cfg, FakeCounter(), store, n_ctx=n_ctx)

    def test_windows_oldest_middle_when_over_budget(self, tmp_path):
        # n_ctx=100, ratio=0.5 -> budget 50 tokens; total is 120 -> windowing fires.
        cfg, store, rw = self._rw(tmp_path, n_ctx=100, context_budget_ratio=0.5)
        msgs = self._msgs()
        out = rw.rewrite_outgoing(msgs).messages
        # pinned head + recent tail are kept verbatim
        assert out[0] == msgs[0]
        assert out[-1] == msgs[-1] and out[-2] == msgs[-2]
        # an old middle message was paged out into a (tiny) stub
        assert PromptRewriter.is_stub(out[1]["content"])
        # LOSSLESS: the full windowed content is recoverable from the store
        handle = PromptRewriter.parse_handles(out[1]["content"])[0]
        assert store.get(handle) == msgs[1]["content"]
        store.close()

    def test_no_windowing_when_under_budget(self, tmp_path):
        # huge window -> budget far above the total -> nothing windowed
        cfg, store, rw = self._rw(tmp_path, n_ctx=100_000, context_budget_ratio=0.70)
        msgs = self._msgs()
        out = rw.rewrite_outgoing(msgs).messages
        assert out == msgs
        store.close()

    def test_windowing_disabled_without_n_ctx(self, tmp_path):
        cfg, store, rw = self._rw(tmp_path, n_ctx=None, context_budget_ratio=0.50)
        msgs = self._msgs()
        assert rw.rewrite_outgoing(msgs).messages == msgs
        store.close()

    def test_windowing_idempotent(self, tmp_path):
        cfg, store, rw = self._rw(tmp_path, n_ctx=100, context_budget_ratio=0.5)
        r1 = rw.rewrite_outgoing(self._msgs())
        assert rw.rewrite_outgoing(r1.messages).messages == r1.messages
        store.close()


class _CountingCounter:
    """Wraps FakeCounter and counts count_text() calls — to prove /tokenize avoidance."""

    def __init__(self):
        self.inner = FakeCounter()
        self.count_text_calls = 0

    def count_text(self, text):
        self.count_text_calls += 1
        return self.inner.count_text(text)

    def truncate_to_tokens(self, text, n):
        return self.inner.truncate_to_tokens(text, n)


class TestTokenizeAvoidance:
    def _rw(self, tmp_path, counter, **over):
        cfg = _config(tmp_path, **over)
        store = DurableStore(cfg.store_root)
        return PromptRewriter(cfg, counter, store)

    def test_bulky_message_tokenized_exactly_once(self, tmp_path):
        # #3: previously count_text ran twice (decision + tokens= field); now once.
        counter = _CountingCounter()
        rw = self._rw(tmp_path, counter, handle_threshold_tokens=10, stub_preview_chars=10)
        content = " ".join(f"word{i}" for i in range(50))  # 50 words, ~330 chars
        r = rw.rewrite_outgoing([{"role": "user", "content": content}])
        assert PromptRewriter.is_stub(r.messages[0]["content"])
        assert counter.count_text_calls == 1

    def test_short_content_skips_tokenize_entirely(self, tmp_path):
        # #4 lower gate: content shorter than the threshold (chars) can't be bulky
        # (tokens <= chars), so it is never sent to the tokenizer.
        counter = _CountingCounter()
        rw = self._rw(tmp_path, counter, handle_threshold_tokens=10)
        r = rw.rewrite_outgoing([{"role": "user", "content": "hi"}])  # 2 chars < 10
        assert counter.count_text_calls == 0
        assert r.messages[0]["content"] == "hi"      # passed through untouched
        assert not r.handle_ized_ids

    def test_huge_content_estimated_without_tokenize(self, tmp_path):
        # #4 upper gate (security seatbelt): content over tokenize_max_chars is handle-ized
        # via a char ESTIMATE — never POSTed to llama-server /tokenize.
        counter = _CountingCounter()
        rw = self._rw(
            tmp_path, counter, handle_threshold_tokens=10,
            tokenize_max_chars=1000, stub_preview_chars=10,
        )
        huge = "x" * 5000  # > tokenize_max_chars
        r = rw.rewrite_outgoing([{"role": "user", "content": huge}])
        assert counter.count_text_calls == 0                 # never tokenized
        assert PromptRewriter.is_stub(r.messages[0]["content"])
        assert r.handle_ized_ids                             # still handle-ized
        assert "tokens=1250" in r.messages[0]["content"]     # 5000 // 4 estimate
