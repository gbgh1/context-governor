from __future__ import annotations

from typing import Optional

from .types import Message, TokenCounter, Summarizer


def seal_summary(
    messages: list[Message],
    prior_summary: Optional[str],
    counter: TokenCounter,
    summarizer: Summarizer,
    cap_tokens: int,
) -> str:
    """
    1. text = summarizer.summarize(messages, prior_summary, target_tokens=cap_tokens)
    2. MEASURE: counter.count_text(text)
    3. If over cap_tokens -> counter.truncate_to_tokens(text, cap_tokens)
    4. Return text guaranteed counter.count_text(result) <= cap_tokens.
    NEVER trust the model's self-reported / requested length. Always measure+truncate.

    §10.4 (fixes H1): a single truncate call may OVERSHOOT (real tokenizers can
    return slightly more tokens than requested). So after truncation we RE-MEASURE
    and shrink until the result measures within cap. We reuse the same `_enforce_cap`
    semantics as the compactor (imported lazily to avoid a circular module-load
    dependency: compactor.py imports seal_summary from this module at its top level).
    """
    text = summarizer.summarize(messages, prior_summary, target_tokens=cap_tokens)
    # Lazy import: compactor.py imports `seal_summary` from this module at load time,
    # so importing `_enforce_cap` eagerly here would create a circular import.
    from .compactor import _enforce_cap
    return _enforce_cap(text, cap_tokens, counter)
