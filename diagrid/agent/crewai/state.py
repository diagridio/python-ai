# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr-backed memory store for CrewAI agents."""

import logging
from typing import Any, Optional

from diagrid.agent.core.state import DaprStateStore

logger = logging.getLogger(__name__)


class DaprMemoryStore:
    """Memory store that persists CrewAI conversation history to Dapr state.

    Integrates with CrewAI's memory system by persisting conversation
    history between workflow iterations.

    Key format: ``crewai-{session_id}-memory``

    Example:
        ```python
        from diagrid.agent.crewai.state import DaprMemoryStore

        memory = DaprMemoryStore(store_name="statestore")

        # Save conversation memory
        memory.save_memory("session-123", {
            "messages": [...],
            "task_output": "...",
        })

        # Load memory
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

    def _memory_key(self, session_id: str) -> str:
        return f"crewai-{session_id}-memory"

    def save_memory(
        self,
        session_id: str,
        data: dict[str, Any],
    ) -> None:
        """Save conversation memory for a session.

        Args:
            session_id: The session identifier.
            data: Memory data to persist (messages, task outputs, etc.).
        """
        key = self._memory_key(session_id)
        self._store.save(key, data)
        logger.debug("Saved memory session=%s", session_id)

    def load_memory(self, session_id: str) -> Optional[dict[str, Any]]:
        """Load conversation memory for a session.

        Args:
            session_id: The session identifier.

        Returns:
            Memory data dict or None if not found.
        """
        key = self._memory_key(session_id)
        return self._store.get(key)

    def delete_memory(self, session_id: str) -> None:
        """Delete conversation memory for a session.

        Args:
            session_id: The session identifier.
        """
        key = self._memory_key(session_id)
        self._store.delete(key)
        logger.debug("Deleted memory session=%s", session_id)

    def close(self) -> None:
        """Close the underlying state store."""
        self._store.close()
