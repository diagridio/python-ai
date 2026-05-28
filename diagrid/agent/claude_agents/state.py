# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr-backed memory store for Claude Agent SDK agents."""

import logging
from typing import Any, Optional

from diagrid.agent.core.state import DaprStateStore

logger = logging.getLogger(__name__)


class DaprMemoryStore:
    """Memory store that persists Claude agent conversation context to Dapr.

    Persists agent conversation context between invocations using
    Dapr state store. Session IDs are used directly as keys — callers
    that need to share a state store with other agent frameworks should
    pick session IDs that are already globally unique.

    Example:
        ```python
        from diagrid.agent.claude_agents.state import DaprMemoryStore

        memory = DaprMemoryStore(store_name="agent-memory")

        memory.save_memory("session-123", {"messages": [...]})
        data = memory.load_memory("session-123")
        ```
    """

    def __init__(
        self,
        store_name: str = "statestore",
        consistency: str = "strong",
        state_store: Optional[DaprStateStore] = None,
    ) -> None:
        self._store = state_store or DaprStateStore(
            store_name=store_name, consistency=consistency
        )

    def save_memory(self, session_id: str, data: dict[str, Any]) -> None:
        """Save conversation memory for a session."""
        self._store.save(session_id, data)
        logger.debug("Saved memory session=%s", session_id)

    def load_memory(self, session_id: str) -> Optional[dict[str, Any]]:
        """Load conversation memory for a session."""
        return self._store.get(session_id)

    def delete_memory(self, session_id: str) -> None:
        """Delete conversation memory for a session."""
        self._store.delete(session_id)
        logger.debug("Deleted memory session=%s", session_id)

    def close(self) -> None:
        """Close the underlying state store."""
        self._store.close()
