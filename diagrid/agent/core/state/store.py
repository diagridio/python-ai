# Copyright 2025 Diagrid Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable Dapr state store client for agent memory persistence."""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_STORE_NAME = "statestore"


class DaprStateStore:
    """Reusable Dapr state store client for agent memory persistence.

    Wraps the Dapr state management API with JSON serialization and
    lazy client initialization.

    Args:
        store_name: Dapr state store component name.
        consistency: Consistency level ("strong" or "eventual").

    Example:
        ```python
        from diagrid.agent.core.state import DaprStateStore

        store = DaprStateStore(store_name="statestore")
        store.save("my-key", {"messages": ["hello"]})
        data = store.get("my-key")
        store.delete("my-key")
        store.close()
        ```
    """

    def __init__(
        self,
        store_name: str = _DEFAULT_STORE_NAME,
        consistency: str = "strong",
    ) -> None:
        self._store_name = store_name
        self._consistency = consistency
        self._client: Any = None

    @property
    def store_name(self) -> str:
        """The Dapr state store component name."""
        return self._store_name

    def _get_client(self) -> Any:
        """Lazily create and return the DaprClient."""
        if self._client is None:
            from dapr.clients import DaprClient

            self._client = DaprClient()
        return self._client

    def save(
        self,
        key: str,
        value: Any,
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        """Save a value to the state store.

        Args:
            key: The state key.
            value: The value to store (will be JSON-serialized).
            metadata: Optional Dapr state metadata.
        """
        client = self._get_client()
        data = json.dumps(value)
        client.save_state(
            store_name=self._store_name,
            key=key,
            value=data,
            state_metadata=metadata,
        )
        logger.debug("Saved state key=%s store=%s", key, self._store_name)

    def get(self, key: str) -> Any | None:
        """Get a value from the state store.

        Args:
            key: The state key.

        Returns:
            The deserialized value, or None if the key does not exist.
        """
        client = self._get_client()
        response = client.get_state(
            store_name=self._store_name,
            key=key,
        )
        if not response.data:
            return None
        return json.loads(response.data.decode())

    def delete(self, key: str) -> None:
        """Delete a key from the state store.

        Args:
            key: The state key to delete.
        """
        client = self._get_client()
        client.delete_state(
            store_name=self._store_name,
            key=key,
        )
        logger.debug("Deleted state key=%s store=%s", key, self._store_name)

    def save_bulk(
        self,
        items: list[tuple[str, Any]],
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        """Save multiple key-value pairs in a single request.

        Args:
            items: List of (key, value) tuples.
            metadata: Optional Dapr state metadata applied to all items.
        """
        client = self._get_client()
        states = []
        for key, value in items:
            from dapr.clients.grpc._state import StateItem

            states.append(
                StateItem(
                    key=key,
                    value=json.dumps(value),
                    metadata=metadata,
                )
            )
        client.save_bulk_state(store_name=self._store_name, states=states)
        logger.debug("Bulk saved %d keys store=%s", len(items), self._store_name)

    def close(self) -> None:
        """Close the underlying DaprClient connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
