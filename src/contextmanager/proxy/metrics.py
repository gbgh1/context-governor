"""Proxy observability (Phase 5 spec §3) — a zero-tokenizer-cost stats surface.

Accumulates per-request prompt-transform stats so a long Hermes session can be
measured before/after the governor (the proxy-side counter that resolves the
"how to measure compaction frequency" open decision). All quantities are derived
from data the rewriter already returns plus free ``len()`` calls — no tokenization.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

# Rough chars-per-token for the FREE (no-tokenizer) wire-savings estimate on /metrics.
# ~4 chars/token matches the project's HeuristicTokenCounter; fields are labeled "_est"
# because they are approximate by design (keeps /metrics zero-tokenizer-cost).
_CHARS_PER_TOKEN = 4


def _human(n: int) -> str:
    """Compact human number: 9_684_702 -> '9.7M', 34_000 -> '34.0K'."""
    a = abs(n)
    if a >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if a >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


@dataclass
class ProxyStats:
    requests: int = 0
    messages_in: int = 0
    messages_handle_ized: int = 0
    messages_rehydrated: int = 0
    slices_recalled: int = 0  # Pass-4 auto-recall (the read-path counterpart)
    chars_in: int = 0
    chars_out: int = 0
    peak_chars_out: int = 0  # single-request high-water of the rewritten wire size

    @property
    def chars_saved(self) -> int:
        # May be negative when rehydration adds more than handle-ization removed;
        # that is real signal (the wire grew this turn), so surface it as-is.
        return self.chars_in - self.chars_out


class StatsCollector:
    """Thread-safe accumulator of ``ProxyStats`` with a JSON-able snapshot."""

    def __init__(self) -> None:
        self._stats = ProxyStats()
        self._lock = Lock()

    def record(
        self,
        *,
        messages_in: int,
        messages_handle_ized: int,
        messages_rehydrated: int,
        chars_in: int,
        chars_out: int,
        slices_recalled: int = 0,
    ) -> None:
        with self._lock:
            s = self._stats
            s.requests += 1
            s.messages_in += messages_in
            s.messages_handle_ized += messages_handle_ized
            s.messages_rehydrated += messages_rehydrated
            s.slices_recalled += slices_recalled
            s.chars_in += chars_in
            s.chars_out += chars_out
            if chars_out > s.peak_chars_out:
                s.peak_chars_out = chars_out

    def snapshot(self) -> dict:
        with self._lock:
            s = self._stats
            requests = s.requests
            chars_in = s.chars_in
            chars_out = s.chars_out
            chars_saved = s.chars_saved
            peak_chars_out = s.peak_chars_out
            base = {
                "requests": requests,
                "messages_in": s.messages_in,
                "messages_handle_ized": s.messages_handle_ized,
                "messages_rehydrated": s.messages_rehydrated,
                "slices_recalled": s.slices_recalled,
                "chars_in": chars_in,
                "chars_out": chars_out,
                "chars_saved": chars_saved,
            }
        # Approximate token view (≈ chars/4, no tokenizer) — the human-readable savings.
        cpt = _CHARS_PER_TOKEN
        tokens_saved = chars_saved // cpt
        peak_tokens = peak_chars_out // cpt
        pct = round(chars_saved / chars_in * 100.0, 1) if chars_in else 0.0
        summary = (
            "no requests yet" if requests == 0 else
            f"saved ~{_human(tokens_saved)} tokens (~{pct:.0f}%) over {requests} "
            f"requests; peak prompt ~{_human(peak_tokens)} tokens"
        )
        base.update({
            "tokens_in_est": chars_in // cpt,
            "tokens_out_est": chars_out // cpt,
            "tokens_saved_est": tokens_saved,
            "pct_saved": pct,
            "peak_prompt_tokens_est": peak_tokens,
            "summary": summary,
        })
        return base
