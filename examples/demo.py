"""End-to-end demo of the Context Governor — runs fully offline (no llama-server).

    python examples/demo.py

Shows the headline value of both surfaces over ONE shared store:
  1. Surface A (proxy rewriter) handle-izes a bulky "file dump" -> a tiny stub on the
     wire, full content saved durably.
  2. Surface B (MCP service) snapshots authoritative state, searches, and rehydrates the
     dump back by handle under a token budget.

Uses the offline HeuristicTokenCounter, so no network / model is required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from contextmanager.durable import DurableStore
from contextmanager.mcp.config import McpConfig
from contextmanager.mcp.counter import HeuristicTokenCounter
from contextmanager.mcp.service import GovernorService
from contextmanager.proxy.config import ProxyConfig
from contextmanager.proxy.rewriter import PromptRewriter


def main() -> None:
    root = Path(tempfile.mkdtemp()) / "contextstore"
    counter = HeuristicTokenCounter()

    # A big artifact the agent read into context (think: a 2KB source file).
    file_dump = "def update(state):\n    # ... game logic ...\n" * 60
    print(f"original message: {len(file_dump)} chars (~{counter.count_text(file_dump)} tokens)\n")

    # --- Surface A: handle-ize on the wire ---------------------------------------
    store = DurableStore(str(root))
    rewriter = PromptRewriter(
        ProxyConfig(upstream_base_url="http://llama", store_root=str(root),
                    handle_threshold_tokens=50, stub_preview_chars=40),
        counter, store,
    )
    result = rewriter.rewrite_outgoing([{"role": "user", "content": file_dump}])
    stub = result.messages[0]["content"]
    handle = PromptRewriter.parse_handles(stub)[0]
    print("[Surface A] proxy rewrote the message to a stub:")
    print("  " + stub.replace("\n", "\n  "))
    print(f"  wire shrank: {len(file_dump)} -> {len(stub)} chars; full body stored as {handle!r}\n")

    # --- Surface B: state + recall over the SAME store ---------------------------
    svc = GovernorService(store, counter, McpConfig(store_root=str(root)))
    svc.state_snapshot({"level": 3, "score": 1200, "lives": 2})
    print("[Surface B] authoritative state snapshot:")
    print("  " + svc.state_load()["rendered"].replace("\n", "\n  "))

    found = svc.store_search("update game logic")
    print(f"\n[Surface B] store_search -> {found['count']} hit(s); top handle = "
          f"{found['results'][0]['handle']!r}")

    reh = svc.context_rehydrate(handle=handle, budget_tokens=40)
    print(f"\n[Surface B] context_rehydrate(handle, budget=40 tok) -> "
          f"{reh['tokens']} content tokens (<= budget), found={reh['found']}")
    print("  recovered head: " + reh["text"][:60].replace("\n", " ") + " …")

    store.close()
    print("\nOK — both surfaces, one store, no llama-server.")


if __name__ == "__main__":
    main()
