"""ProxyConfig — configuration for the Phase 3 endpoint proxy.

Normative per Phase 3 spec §2. Frozen dataclass with post-init validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProxyConfig:
    """Immutable configuration for the endpoint proxy.

    Attributes:
        upstream_base_url: llama-server base URL, e.g. "http://127.0.0.1:8080".
        store_root: filesystem root for the DurableStore.
        upstream_api_key: optional API key forwarded to the upstream.
        listen_host: host the proxy listens on.
        listen_port: port the proxy listens on (1..65535).
        handle_threshold_tokens: messages whose token count is >= this get
            handle-ized (their full content paged out to the DurableStore).
        stub_preview_chars: number of head/tail characters kept in the stub
            preview emitted in place of the handle-ized content.
        rehydrate_budget_tokens: maximum tokens paged back in per request via
            auto-rehydration of explicit handle references.
        request_timeout: upstream HTTP request timeout in seconds (generation
            can be long).
        handle_threshold_ratio: when > 0 (default 0.02) AND the upstream's true
            context size (llama-server /props n_ctx) is known at startup, the
            per-message handle-ization threshold is ANCHORED to the real window
            (ratio * n_ctx, floored), making the governor self-tune to whatever
            `-c` the server runs. 0 disables anchoring -> the fixed
            handle_threshold_tokens is always used. llama-server is the source of
            truth for context size, not the CLI.
        context_budget_ratio: when > 0 (default 0.50) AND n_ctx is known, bound the
            TOTAL wire to this fraction of the real window by paging out the oldest
            non-pinned middle messages (lossless — they become retrievable stubs).
            This pre-empts the CLI's own (lossy) compaction so it rarely fires. 0
            disables windowing.
        protect_first_n / protect_last_n: messages at the head (system/spec) and the
            recent tail that budget-windowing never pages out (pinned + recent window).
        model_alias: if set (default "context-governor"), the proxy presents the
            upstream's model under THIS name in /v1/models (and the Ollama discovery
            aliases), inheriting all other fields. Set to None/"" to pass the real
            model name through unchanged. Chat requests forward verbatim — llama-server
            serves the loaded model regardless of the requested name.
        auto_recall_k: max slices auto-recalled per request (Pass 4 anticipatory
            demand paging: an implicit query from the live tail searches the store
            and injects relevant OFF-wire memory as one marked system message).
            0 disables auto-recall entirely.
        recall_budget_tokens: max tokens the Pass-4 recall block may occupy. The
            total wire bound becomes context_budget_ratio*n_ctx + this (~2% of a
            75K window at the default).
    """

    upstream_base_url: str
    store_root: str = "./contextstore"
    upstream_api_key: Optional[str] = None
    listen_host: str = "127.0.0.1"
    listen_port: int = 8900
    handle_threshold_tokens: int = 2000
    stub_preview_chars: int = 200
    rehydrate_budget_tokens: int = 4000
    request_timeout: float = 300.0
    model_alias: Optional[str] = "context-governor"
    diff_min_similarity: float = 0.5
    diff_lookback: int = 6
    # Upper size bound (chars) for diff-encoding. difflib.SequenceMatcher is O(n*m) and
    # pathological on large, repetitive content (log files!), so above this size a bulky
    # message becomes a normal stub instead of freezing the proxy for minutes. 0 = no cap
    # (unsafe; restores the old unbounded behaviour). ~20 KB covers typical file re-reads.
    diff_max_chars: int = 20000
    # Upper size bound (chars) for calling the tokenizer. Content larger than this is
    # definitely bulky AND too big to POST to llama-server /tokenize (slow + a DoS risk),
    # so it is handle-ized using a cheap char-based token ESTIMATE instead. 0 = no cap.
    tokenize_max_chars: int = 100000
    handle_threshold_ratio: float = 0.02
    context_budget_ratio: float = 0.50
    protect_first_n: int = 2
    protect_last_n: int = 6
    auto_recall_k: int = 3
    recall_budget_tokens: int = 1500

    def __post_init__(self) -> None:
        if self.handle_threshold_tokens <= 0:
            raise ValueError(
                f"handle_threshold_tokens must be > 0, got {self.handle_threshold_tokens}"
            )
        if self.stub_preview_chars < 0:
            raise ValueError(
                f"stub_preview_chars must be >= 0, got {self.stub_preview_chars}"
            )
        if self.rehydrate_budget_tokens < 0:
            raise ValueError(
                f"rehydrate_budget_tokens must be >= 0, got {self.rehydrate_budget_tokens}"
            )
        if not (1 <= self.listen_port <= 65535):
            raise ValueError(
                f"listen_port must be in 1..65535, got {self.listen_port}"
            )
        if not (0.0 <= self.diff_min_similarity <= 1.0):
            raise ValueError(
                f"diff_min_similarity must be in 0.0..1.0 (0 disables), got {self.diff_min_similarity}"
            )
        if self.diff_lookback < 0:
            raise ValueError(
                f"diff_lookback must be >= 0, got {self.diff_lookback}"
            )
        if self.diff_max_chars < 0:
            raise ValueError(
                f"diff_max_chars must be >= 0 (0 disables the cap), got {self.diff_max_chars}"
            )
        if self.tokenize_max_chars < 0:
            raise ValueError(
                f"tokenize_max_chars must be >= 0 (0 disables the cap), got "
                f"{self.tokenize_max_chars}"
            )
        if not (0.0 <= self.handle_threshold_ratio <= 1.0):
            raise ValueError(
                f"handle_threshold_ratio must be in 0.0..1.0 (0 = use fixed "
                f"handle_threshold_tokens), got {self.handle_threshold_ratio}"
            )
        if not (0.0 <= self.context_budget_ratio <= 1.0):
            raise ValueError(
                f"context_budget_ratio must be in 0.0..1.0 (0 disables windowing), "
                f"got {self.context_budget_ratio}"
            )
        if self.protect_first_n < 0:
            raise ValueError(f"protect_first_n must be >= 0, got {self.protect_first_n}")
        if self.protect_last_n < 0:
            raise ValueError(f"protect_last_n must be >= 0, got {self.protect_last_n}")
        if self.auto_recall_k < 0:
            raise ValueError(f"auto_recall_k must be >= 0 (0 disables), got {self.auto_recall_k}")
        if self.recall_budget_tokens < 0:
            raise ValueError(
                f"recall_budget_tokens must be >= 0 (0 disables), got {self.recall_budget_tokens}"
            )
