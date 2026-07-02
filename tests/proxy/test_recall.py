"""Phase 10 auto-recall tests — the read path becomes intelligent.

Unit tests for the pure ``recall`` helpers (implicit-query extraction,
near-duplicate suppression) and invariant tests for the rewriter's Pass 4:

  - relevant OFF-wire store content is injected as ONE ``[[cm:recall]]`` system
    message right before the final message; ``recalled_handles`` reports it.
  - handles already on the wire (as stubs or verbatim content) are never recalled.
  - NO-GROWTH: rewriting the rewriter's own output strips the previous block and
    injects a fresh one — message count and block content stay stable, and the
    recalled slices are never re-expanded by Pass 2.
  - the recall block never exceeds ``recall_budget_tokens``; ``auto_recall_k=0``
    disables Pass 4 entirely; an empty store injects nothing.

Same harness as test_rewriter.py: FakeCounter (token = word count) + a real
``DurableStore(tmp_path)``. No network, no FastAPI.
"""

from __future__ import annotations

from conftest import FakeCounter
from contextmanager.durable import DurableStore
from contextmanager.proxy.config import ProxyConfig
from contextmanager.proxy.recall import extract_query, select_diverse
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
    cfg = _config(tmp_path, **over)
    store = DurableStore(cfg.store_root)
    counter = FakeCounter()
    return cfg, store, counter, PromptRewriter(cfg, counter, store)


def _seed(store: DurableStore, mid: str, content: str) -> str:
    """Page an OFF-wire note into the store; returns its handle."""
    return store.page_out(Message(role="user", content=content, id=mid))


def _recall_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages
            if PromptRewriter.is_recall(m.get("content", ""))]


# ---------------------------------------------------------------------------
# extract_query — the implicit tail query
# ---------------------------------------------------------------------------


class TestExtractQuery:
    def test_salient_terms_no_stopwords(self):
        msgs = [{"role": "user",
                 "content": "Fix the quantum flux capacitor and the plasma coil"}]
        q = extract_query(msgs)
        terms = q.split()
        assert "quantum" in terms and "capacitor" in terms and "plasma" in terms
        assert "the" not in terms and "and" not in terms

    def test_recency_weights_newest_higher(self):
        msgs = [{"role": "user", "content": "zebra zebra zebra oldest topic"},
                {"role": "user", "content": "filler middle message here"},
                {"role": "user", "content": "quokka newest topic focus"}]
        q = extract_query(msgs).split()
        # "quokka" (once, newest, weight 2.0) must outrank "zebra"-adjacent
        # single-count old terms; both present, but newest-first ordering shows
        # the weighting (zebra 3x still wins overall — that is real salience).
        assert "quokka" in q and "zebra" in q
        assert q.index("zebra") < q.index("quokka")  # 3 hits beat one weighted hit

    def test_marker_lines_stripped(self):
        stub = ("[[cm:stored handle=zzzuniquehandle role=user tokens=99]]\n"
                "visible preview snippet alpha\n"
                "…(truncated 1234 chars)…\n"
                "visible ending beta\n"
                "[[/cm:stored]]")
        q = extract_query([{"role": "user", "content": stub}])
        terms = q.split()
        assert "zzzuniquehandle" not in terms and "tokens" not in terms
        assert "preview" in terms and "snippet" in terms  # content lines survive

    def test_trivial_or_empty_yields_empty(self):
        assert extract_query([]) == ""
        assert extract_query([{"role": "user", "content": "hello"}]) == ""
        assert extract_query([{"role": "user", "content": "12345 67890"}]) == ""

    def test_output_is_fts5_safe(self):
        msgs = [{"role": "user",
                 "content": 'weird "quoted" AND (parens) OR star* minus-dash!'}]
        q = extract_query(msgs)
        assert q == "" or all(t.replace("_", "a").isalnum() for t in q.split())

    def test_deterministic(self):
        msgs = [{"role": "user", "content": "gamma delta epsilon gamma delta"}]
        assert extract_query(msgs) == extract_query(msgs)


# ---------------------------------------------------------------------------
# select_diverse — near-duplicate suppression
# ---------------------------------------------------------------------------


class TestSelectDiverse:
    def test_identical_suppressed(self):
        text = "same content " * 30
        assert select_diverse([text, text]) == [0]

    def test_distinct_kept(self):
        a = "alpha " * 50
        b = "totally different beta content " * 20
        assert select_diverse([a, b]) == [0, 1]

    def test_near_duplicate_suppressed(self):
        base = " ".join(f"line{i}" for i in range(100))
        near = base.replace("line50", "changed")
        assert select_diverse([base, near]) == [0]


# ---------------------------------------------------------------------------
# Rewriter Pass 4 — auto-recall invariants
# ---------------------------------------------------------------------------


class TestAutoRecall:
    def test_injects_off_wire_slice_before_final_message(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path)
        body = "unicorn banana smoothie recipe with quantum sprinkles"
        h = _seed(store, "old-1", body)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "unicorn banana smoothie recipe quantum?"},
        ]
        res = rw.rewrite_outgoing(messages)
        assert res.recalled_handles == [h]
        blocks = _recall_messages(res.messages)
        assert len(blocks) == 1
        block = blocks[0]["content"]
        assert f"[[cm:recalled handle={h}]]" in block
        assert body in block                       # full slice fits the budget
        assert res.messages[-2] is blocks[0]       # injected before the final msg
        assert res.messages[-1] == messages[-1]    # final message untouched

    def test_skips_handles_already_on_wire(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path)
        bulky = " ".join(f"kumquat{i}" for i in range(50))  # >= threshold
        res = rw.rewrite_outgoing([{"role": "user", "content": bulky}])
        # The bulky message was handle-ized THIS call -> its handle is on-wire
        # (visible as a stub); recalling it would duplicate what the model sees.
        assert len(res.handle_ized_ids) == 1
        assert res.recalled_handles == []
        assert _recall_messages(res.messages) == []

    def test_skips_verbatim_wire_content(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path)
        small = "penguin igloo blueprint schematic"     # below threshold
        # The SAME content was paged out in some earlier turn (its note exists)
        # but the live wire still carries it verbatim -> recall must skip it.
        mid = PromptRewriter.stable_id("user", small)
        store.page_out(Message(role="user", content=small, id=mid))
        res = rw.rewrite_outgoing([{"role": "user", "content": small},
                                   {"role": "user", "content": "penguin igloo?"}])
        assert res.recalled_handles == []
        assert _recall_messages(res.messages) == []

    def test_no_growth_and_no_pass2_reexpansion_on_double_rewrite(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path)
        h = _seed(store, "old-1", "walrus tuba concerto sheet music archive")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "need the walrus tuba concerto notes"},
        ]
        r1 = rw.rewrite_outgoing(messages)
        assert len(_recall_messages(r1.messages)) == 1
        r2 = rw.rewrite_outgoing(r1.messages)
        # Strip-on-entry + fresh injection: still exactly one block, same count.
        assert len(_recall_messages(r2.messages)) == 1
        assert len(r2.messages) == len(r1.messages)
        assert _recall_messages(r2.messages)[0] == _recall_messages(r1.messages)[0]
        # `[[cm:recalled` must NOT look like an explicit stored-reference: Pass 2
        # never re-expands a recalled slice into a rehydrated message.
        assert r2.rehydrated_handles == []
        assert not any("[[cm:rehydrated" in (m.get("content") or "")
                       for m in r2.messages)
        assert r2.recalled_handles == [h]

    def test_budget_respected(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path, recall_budget_tokens=10)
        _seed(store, "old-1", " ".join(f"ocelot{i}" for i in range(50)))
        messages = [{"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "ocelot0 ocelot1 ocelot2 please"}]
        res = rw.rewrite_outgoing(messages)
        blocks = _recall_messages(res.messages)
        assert len(blocks) == 1                    # truncated slice still fits
        assert counter.count_text(blocks[0]["content"]) <= 10

    def test_k_zero_disables_entirely(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path, auto_recall_k=0)
        _seed(store, "old-1", "dragonfruit compass calibration manual")
        messages = [{"role": "user", "content": "dragonfruit compass calibration?"},
                    {"role": "user", "content": "yes the dragonfruit compass"}]
        res = rw.rewrite_outgoing(messages)
        assert res.recalled_handles == []
        assert res.messages == messages            # byte-identical passthrough

    def test_empty_store_no_injection(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path)
        messages = [{"role": "user", "content": "lighthouse gearbox assembly steps"},
                    {"role": "user", "content": "lighthouse gearbox please"}]
        res = rw.rewrite_outgoing(messages)
        assert res.recalled_handles == []
        assert _recall_messages(res.messages) == []

    def test_counter_failure_degrades_to_no_recall(self, tmp_path):
        # REGRESSION (found LIVE, 2026-07-01): a TokenizerError inside the
        # recall builder 500'd every chat completion. Recall is enrichment —
        # any failure while building the block must degrade to "no recall
        # this turn", never break the request.
        class BoomCounter(FakeCounter):
            def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
                raise RuntimeError("tokenizer exploded")

        cfg = _config(tmp_path, recall_budget_tokens=10)
        store = DurableStore(cfg.store_root)
        rw = PromptRewriter(cfg, BoomCounter(), store)
        # Note larger than the budget -> the builder MUST attempt truncation.
        _seed(store, "old-1", " ".join(f"wombat{i}" for i in range(50)))
        messages = [{"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "wombat0 wombat1 wombat2 data"}]
        res = rw.rewrite_outgoing(messages)   # must not raise
        assert res.recalled_handles == []
        assert _recall_messages(res.messages) == []
        assert res.messages == messages       # request passes through intact

    def test_no_synthetic_system_messages_mid_wire(self, tmp_path):
        # REGRESSION (found LIVE, 2026-07-01, second run): Qwen's chat template
        # raises "System message must be at the beginning" for ANY system
        # message past index 0 — the upstream 500'd on every request carrying a
        # recall block. Synthetic messages (Pass-4 recall AND Pass-2
        # rehydration) must use role "user"; only the input's own head system
        # message may carry role "system".
        cfg, store, counter, rw = _rewriter(tmp_path)
        h = _seed(store, "old-1", "gryphon saddle polish instructions manual")
        h2 = _seed(store, "old-2", "gryphon saddle wax alternative recipe")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user",
             "content": (f"see [[cm:stored handle={h} role=user tokens=6]] "
                         "gryphon saddle polish wax?")},
        ]
        res = rw.rewrite_outgoing(messages)
        # Both synthetic paths fired this call...
        assert res.rehydrated_handles == [h]
        assert res.recalled_handles == [h2]
        # ...and neither produced a mid-wire system message.
        system_positions = [i for i, m in enumerate(res.messages)
                            if m.get("role") == "system"]
        assert system_positions == [0]  # only the ORIGINAL head system message

    def test_near_duplicate_slices_suppressed(self, tmp_path):
        cfg, store, counter, rw = _rewriter(tmp_path)
        base = " ".join(f"marmot{i}" for i in range(100))
        near = base.replace("marmot50", "changed")
        distinct = "marmot0 telescope " + " ".join(f"lens{i}" for i in range(80))
        _seed(store, "twin-a", base)
        _seed(store, "twin-b", near)
        h3 = _seed(store, "other", distinct)
        messages = [{"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "marmot0 marmot1 telescope data"}]
        res = rw.rewrite_outgoing(messages)
        # One twin suppressed; the distinct note survives alongside the other twin.
        assert len(res.recalled_handles) == 2
        assert h3 in res.recalled_handles
