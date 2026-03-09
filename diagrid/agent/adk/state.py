# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr-backed session store for Google ADK agents."""

import logging
from typing import Any, Optional

from diagrid.agent.core.state import DaprStateStore

logger = logging.getLogger(__name__)


class DaprSessionStore:
    """Session store that persists ADK conversation state to Dapr state.

    Persists conversation history and session state between workflow
    iterations for ADK agents.

    Key format: ``adk-{session_id}-state``

    Example:
        ```python
        from diagrid.agent.adk.state import DaprSessionStore

        session_store = DaprSessionStore(store_name="agent-memory")

        # Save session state
        session_store.save_session("session-123", {
            "messages": [...],
            "user_id": "user-1",
        })

        # Load session
        data = session_store.load_session("session-123")
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

    def _session_key(self, session_id: str) -> str:
        return f"adk-{session_id}-state"

    def save_session(
        self,
        session_id: str,
        data: dict[str, Any],
    ) -> None:
        """Save session state.

        Args:
            session_id: The session identifier.
            data: Session data to persist (messages, user_id, etc.).
        """
        key = self._session_key(session_id)
        self._store.save(key, data)
        logger.debug("Saved session=%s", session_id)

    def load_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Load session state.

        Args:
            session_id: The session identifier.

        Returns:
            Session data dict or None if not found.
        """
        key = self._session_key(session_id)
        return self._store.get(key)

    def delete_session(self, session_id: str) -> None:
        """Delete session state.

        Args:
            session_id: The session identifier.
        """
        key = self._session_key(session_id)
        self._store.delete(key)
        logger.debug("Deleted session=%s", session_id)

    def close(self) -> None:
        """Close the underlying state store."""
        self._store.close()
