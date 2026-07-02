"""ContextManager — context governor: Phase 1 core engine + Phase 2 durable store."""

from __future__ import annotations

from .types import Message, ContextState, TokenCounter, Summarizer, Store
from .budget import BudgetConfig
from .sealing import seal_summary
from .compactor import (
    HysteresisCompactor,
    CompactionResult,
    FloorExceedsTargetError,
    ContractError,
    InvariantViolationError,
)
from .tokenizer import LlamaServerTokenCounter, TokenizerError
from .state_store import StateStore
from .note_store import NoteStore, NoteMeta, StoreError
from .retriever import (
    Retriever,
    LexicalRetriever,
    Embedder,
    LlamaServerEmbedder,
    RetrieverError,
)
from .durable import DurableStore, RetrievedSlice

__all__ = [
    # Phase 1 — core engine
    "Message",
    "ContextState",
    "TokenCounter",
    "Summarizer",
    "Store",
    "BudgetConfig",
    "seal_summary",
    "HysteresisCompactor",
    "CompactionResult",
    "FloorExceedsTargetError",
    "ContractError",
    "InvariantViolationError",
    "LlamaServerTokenCounter",
    "TokenizerError",
    # Phase 2 — durable store + retriever
    "StateStore",
    "NoteStore",
    "NoteMeta",
    "StoreError",
    "Retriever",
    "LexicalRetriever",
    "Embedder",
    "LlamaServerEmbedder",
    "RetrieverError",
    "DurableStore",
    "RetrievedSlice",
]
