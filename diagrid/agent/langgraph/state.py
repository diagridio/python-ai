# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dapr-backed checkpoint saver for LangGraph."""

import logging
import random
import time
from typing import Any, Callable, List, Optional, Tuple
from urllib.parse import quote

from diagrid.agent.core.state import DaprStateStore

logger = logging.getLogger(__name__)

#: How long a thread's checkpoints and index are kept after its last save.
DEFAULT_THREAD_TTL_S = 30 * 24 * 3600
#: How many checkpoints a thread keeps: the latest, and the one before it.
DEFAULT_KEEP_CHECKPOINTS = 2
#: Tries at writing a thread's index when other runs write it at the same time.
INDEX_WRITE_ATTEMPTS = 8

#: ``(index after the change, checkpoint ids it no longer lists)``
IndexChange = Callable[[dict[str, Any]], Tuple[dict[str, Any], List[str]]]


class DaprMemoryCheckpointer:
    """LangGraph checkpoint saver backed by Dapr state store.

    Persists LangGraph checkpoints (channel state and version metadata)
    to a Dapr state store, enabling conversation memory across invocations.

    Key format: ``langgraph:{thread_id}:checkpoint:{checkpoint_id}``, and
    ``langgraph:{thread_id}:index`` for the thread's list of checkpoints. Each
    id is path-escaped, so a ``:``, ``|`` or ``#`` in it cannot run into the
    separators (and no key ever holds ``||``, which Dapr reserves). Checkpoints
    saved under the earlier ``langgraph-{thread_id}-checkpoint-{id}`` keys are
    still read.

    A thread keeps its last ``keep_checkpoints`` checkpoints (older ones are
    deleted), and its checkpoints and index expire ``ttl_s`` seconds after the
    thread's last save (``None``: never; the state store must support TTLs).
    The index is written with an etag (first-write concurrency) where the store
    supports it, so concurrent runs on one thread lose none of its entries.

    :class:`~diagrid.agent.langgraph.DaprWorkflowGraphRunner` uses one on the
    app's state store by default, so ``thread_id`` carries a conversation from
    one run to the next.

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
        *,
        ttl_s: Optional[int] = DEFAULT_THREAD_TTL_S,
        keep_checkpoints: int = DEFAULT_KEEP_CHECKPOINTS,
    ) -> None:
        if keep_checkpoints < 1:
            raise ValueError("keep_checkpoints must be at least 1")
        if ttl_s is not None and ttl_s < 1:
            raise ValueError("ttl_s must be a positive number of seconds, or None")
        self._store = state_store or DaprStateStore(
            store_name=store_name, consistency=consistency
        )
        self._ttl_s = ttl_s
        self._keep = keep_checkpoints

    def _checkpoint_key(self, thread_id: str, checkpoint_id: str) -> str:
        return f"langgraph:{_segment(thread_id)}:checkpoint:{_segment(checkpoint_id)}"

    def _thread_index_key(self, thread_id: str) -> str:
        return f"langgraph:{_segment(thread_id)}:index"

    def _metadata(self) -> Optional[dict[str, str]]:
        return None if self._ttl_s is None else {"ttlInSeconds": str(self._ttl_s)}

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
        self._store.save(key, data, self._metadata())
        dropped = self._update_index(
            thread_id, lambda index: _with_latest(index, checkpoint_id, self._keep)
        )
        for old in dropped:
            self._store.delete(self._checkpoint_key(thread_id, old))
        logger.debug(
            "Saved checkpoint thread=%s checkpoint=%s (pruned %d)",
            thread_id,
            checkpoint_id,
            len(dropped),
        )

    def _update_index(self, thread_id: str, change: IndexChange) -> List[str]:
        """Apply ``change`` to the thread's index; the checkpoint ids it dropped.

        With etags, a write that lost to a concurrent one reads the index again
        and re-applies the change (with jittered backoff).
        """
        key = self._thread_index_key(thread_id)
        get_with_etag = getattr(self._store, "get_with_etag", None)
        save_with_etag = getattr(self._store, "save_with_etag", None)
        if not (callable(get_with_etag) and callable(save_with_etag)):
            index, dropped = change(_index_copy(self._store.get(key)))
            self._store.save(key, index, self._metadata())
            return dropped
        for attempt in range(INDEX_WRITE_ATTEMPTS):
            current, etag = get_with_etag(key)
            index, dropped = change(_index_copy(current))
            if save_with_etag(key, index, etag, self._metadata()):
                return dropped
            time.sleep(random.uniform(0, 0.01 * (2**attempt)))
        raise RuntimeError(
            f"The index of thread {thread_id!r} could not be written: "
            f"{INDEX_WRITE_ATTEMPTS} concurrent writes in a row won over this one"
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
        if checkpoint_id is not None:
            found = self._store.get(self._checkpoint_key(thread_id, checkpoint_id))
            return found or self._store.get(
                _legacy_checkpoint_key(thread_id, checkpoint_id)
            )
        index = self._store.get(self._thread_index_key(thread_id))
        if index and index.get("latest"):
            return self._store.get(self._checkpoint_key(thread_id, index["latest"]))
        legacy = self._store.get(_legacy_index_key(thread_id))
        if not legacy or not legacy.get("latest"):
            return None
        return self._store.get(_legacy_checkpoint_key(thread_id, legacy["latest"]))

    def list_checkpoints(self, thread_id: str) -> list[str]:
        """List all checkpoint IDs for a thread.

        Args:
            thread_id: The thread identifier.

        Returns:
            List of checkpoint IDs.
        """
        index = self._store.get(self._thread_index_key(thread_id))
        if not index:
            index = self._store.get(_legacy_index_key(thread_id))
        if not index:
            return []
        return list(index.get("checkpoints", []))

    def delete_checkpoint(self, thread_id: str, checkpoint_id: str) -> None:
        """Delete a specific checkpoint.

        Args:
            thread_id: The thread identifier.
            checkpoint_id: The checkpoint to delete.
        """
        self._store.delete(self._checkpoint_key(thread_id, checkpoint_id))
        self._update_index(thread_id, lambda index: _without(index, checkpoint_id))

    def close(self) -> None:
        """Close the underlying state store."""
        self._store.close()


def _segment(value: str) -> str:
    """``value`` escaped so it is one segment of a ``:``-separated key."""
    return quote(str(value), safe="")


def _legacy_index_key(thread_id: str) -> str:
    """The index key checkpoints were saved under before ``:`` keys (read only)."""
    return f"langgraph-{thread_id}-index"


def _legacy_checkpoint_key(thread_id: str, checkpoint_id: str) -> str:
    return f"langgraph-{thread_id}-checkpoint-{checkpoint_id}"


def _index_copy(index: Any) -> dict[str, Any]:
    """A thread index to change (the stored one is never changed in place)."""
    if not isinstance(index, dict):
        return {"checkpoints": []}
    return {**index, "checkpoints": list(index.get("checkpoints") or [])}


def _with_latest(
    index: dict[str, Any], checkpoint_id: str, keep: int
) -> Tuple[dict[str, Any], List[str]]:
    """``index`` with ``checkpoint_id`` as its latest, trimmed to ``keep`` entries."""
    ids = [c for c in index["checkpoints"] if c != checkpoint_id] + [checkpoint_id]
    return {**index, "checkpoints": ids[-keep:], "latest": checkpoint_id}, ids[:-keep]


def _without(
    index: dict[str, Any], checkpoint_id: str
) -> Tuple[dict[str, Any], List[str]]:
    """``index`` without ``checkpoint_id``; the latest moves to the one before."""
    ids = [c for c in index["checkpoints"] if c != checkpoint_id]
    latest = index.get("latest")
    if latest == checkpoint_id:
        latest = ids[-1] if ids else None
    return {**index, "checkpoints": ids, "latest": latest}, []
