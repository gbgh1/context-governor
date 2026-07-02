"""Fixtures for the Phase 4 MCP server tests (binds to tasks/phase4-spec.md §7).

Pure, deterministic, no network: a real ``DurableStore(tmp_path)`` + the Phase 1
``FakeCounter`` (token = whitespace word count; truncate = first-N-words).
"""

from __future__ import annotations

from typing import Any

import pytest

from conftest import FakeCounter  # Phase 1 root conftest


@pytest.fixture
def make_config(tmp_path):
    """Callable -> McpConfig with a tmp_path store and small previews."""
    from contextmanager.mcp.config import McpConfig

    def _build(**over: Any):
        kw: dict[str, Any] = dict(
            store_root=str(tmp_path / "store"),
            preview_chars=20,
            default_search_k=5,
            rehydrate_budget_tokens=4000,
        )
        kw.update(over)
        return McpConfig(**kw)

    return _build


@pytest.fixture
def make_service(make_config):
    """Callable -> GovernorService wired to a real DurableStore + FakeCounter."""
    from contextmanager.durable import DurableStore
    from contextmanager.mcp.service import GovernorService

    created: list[GovernorService] = []

    def _build(*, config=None, counter=None, **cfg_over: Any):
        config = config or make_config(**cfg_over)
        counter = counter or FakeCounter()
        svc = GovernorService(DurableStore(config.store_root), counter, config)
        created.append(svc)
        return svc

    yield _build
    for svc in created:
        svc.close()
