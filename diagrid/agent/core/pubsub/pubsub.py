# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Reusable Dapr pub/sub client for agent event-driven communication."""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DaprPubSub:
    """Dapr pub/sub client for agent event-driven communication.

    Wraps the Dapr pub/sub API for publishing events. Designed for
    agent-to-agent communication and event-driven agent invocation.

    Args:
        pubsub_name: Dapr pub/sub component name.

    Example:
        ```python
        from diagrid.agent.core.pubsub import DaprPubSub

        pubsub = DaprPubSub(pubsub_name="agent-pubsub")
        pubsub.publish("my-topic", {"message": "hello"})
        pubsub.close()
        ```
    """

    def __init__(self, pubsub_name: str = "agent-pubsub") -> None:
        self._pubsub_name = pubsub_name
        self._client: Any = None

    @property
    def pubsub_name(self) -> str:
        """The Dapr pub/sub component name."""
        return self._pubsub_name

    def _get_client(self) -> Any:
        """Lazily create and return the DaprClient."""
        if self._client is None:
            from dapr.clients import DaprClient

            self._client = DaprClient()
        return self._client

    def publish(
        self,
        topic: str,
        data: Any,
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        """Publish an event to a topic.

        Args:
            topic: The topic name to publish to.
            data: The event data (will be JSON-serialized).
            metadata: Optional publish metadata.
        """
        client = self._get_client()
        client.publish_event(
            pubsub_name=self._pubsub_name,
            topic_name=topic,
            data=json.dumps(data),
            data_content_type="application/json",
            publish_metadata=metadata or {},
        )
        logger.debug("Published event topic=%s pubsub=%s", topic, self._pubsub_name)

    def close(self) -> None:
        """Close the underlying DaprClient connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
