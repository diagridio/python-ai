# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr-backed checkpoint saver for LangGraph."""

import logging
from typing import Any, Optional

from diagrid.agent.core.state import DaprStateStore

logger = logging.getLogger(__name__)


class DaprMemoryCheckpointer:
    """LangGraph checkpoint saver backed by Dapr state store.

    Persists LangGraph checkpoints (channel state and version metadata)
    to a Dapr state store, enabling conversation memory across invocations.

    Key format: ``langgraph-{thread_id}-checkpoint-{checkpoint_id}``

    Example:
        ```python
        from diagrid.agent.langgraph.state import DaprMemoryCheckpointer

        checkpointer = DaprMemoryCheckpointer(store_name="agent-memory")

        # Save a checkpoint
        checkpointer.save_checkpoint(
            thread_id="thread-123",
            checkpoint_id="cp-1",
            channel_values={"messages": [...]},
            channel_versions={"messages": 1},
        )

        # Load latest checkpoint
        data = checkpointer.load_checkpoint(
            thread_id="thread-123",
            checkpoint_id="cp-1",
        )
        ```
    """

    def __init__(
        self,
        store_name: str = "agent-memory",
        consistency: str = "strong",
        state_store: Optional[DaprStateStore] = None,
    ) -> None:
        self._store = state_store or DaprStateStore(
            store_name=store_name, consistency=consistency
        )

    def _checkpoint_key(self, thread_id: str, checkpoint_id: str) -> str:
        return f"langgraph-{thread_id}-checkpoint-{checkpoint_id}"

    def _thread_index_key(self, thread_id: str) -> str:
        return f"langgraph-{thread_id}-index"

    def save_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
        channel_values: dict[str, Any],
        channel_versions: dict[str, int],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save a checkpoint to the Dapr state store.

        Args:
            thread_id: The thread identifier.
            checkpoint_id: Unique checkpoint identifier.
            channel_values: Channel state values.
            channel_versions: Channel version counters.
            metadata: Optional metadata to store with the checkpoint.
        """
        key = self._checkpoint_key(thread_id, checkpoint_id)
        data = {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "channel_values": channel_values,
            "channel_versions": channel_versions,
            "metadata": metadata or {},
        }
        self._store.save(key, data)

        # Update thread index with latest checkpoint
        index_key = self._thread_index_key(thread_id)
        index = self._store.get(index_key) or {"checkpoints": []}
        if checkpoint_id not in index["checkpoints"]:
            index["checkpoints"].append(checkpoint_id)
        index["latest"] = checkpoint_id
        self._store.save(index_key, index)

        logger.debug(
            "Saved checkpoint thread=%s checkpoint=%s",
            thread_id,
            checkpoint_id,
        )

    def load_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Load a checkpoint from the Dapr state store.

        Args:
            thread_id: The thread identifier.
            checkpoint_id: Specific checkpoint to load. If None, loads the
                latest checkpoint.

        Returns:
            Checkpoint data dict or None if not found.
        """
        if checkpoint_id is None:
            index_key = self._thread_index_key(thread_id)
            index = self._store.get(index_key)
            if not index or "latest" not in index:
                return None
            checkpoint_id = index["latest"]

        key = self._checkpoint_key(thread_id, checkpoint_id)
        return self._store.get(key)

    def list_checkpoints(self, thread_id: str) -> list[str]:
        """List all checkpoint IDs for a thread.

        Args:
            thread_id: The thread identifier.

        Returns:
            List of checkpoint IDs.
        """
        index_key = self._thread_index_key(thread_id)
        index = self._store.get(index_key)
        if not index:
            return []
        return index.get("checkpoints", [])

    def delete_checkpoint(self, thread_id: str, checkpoint_id: str) -> None:
        """Delete a specific checkpoint.

        Args:
            thread_id: The thread identifier.
            checkpoint_id: The checkpoint to delete.
        """
        key = self._checkpoint_key(thread_id, checkpoint_id)
        self._store.delete(key)

        # Update index
        index_key = self._thread_index_key(thread_id)
        index = self._store.get(index_key)
        if index and checkpoint_id in index.get("checkpoints", []):
            index["checkpoints"].remove(checkpoint_id)
            if index.get("latest") == checkpoint_id:
                index["latest"] = (
                    index["checkpoints"][-1] if index["checkpoints"] else None
                )
            self._store.save(index_key, index)

    def close(self) -> None:
        """Close the underlying state store."""
        self._store.close()
