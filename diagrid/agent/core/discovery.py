"""Unified Dapr sidecar component discovery.

Queries ``DaprClient.get_metadata()`` once and discovers all known
Dapr components by name/type convention, following the dapr-agents
naming standard (``agent-memory``, ``agent-pubsub``, etc.).

The result is cached at module level — metadata is static for the
process lifetime.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from dapr.clients import DaprClient

logger = logging.getLogger(__name__)

_cached_discovery: Optional[DiscoveredComponents] = None


@dataclass
class DiscoveredComponents:
    """Result of Dapr sidecar component discovery."""

    conversation_name: Optional[str] = None
    """First ``conversation.*`` type component found."""

    configuration_name: Optional[str] = None
    """``configuration`` type component named ``agent-configuration``."""

    memory_store_name: Optional[str] = None
    """``state`` type component named ``agent-memory``."""

    pubsub_name: Optional[str] = None
    """``pubsub`` type component named ``agent-pubsub``."""

    registry_store_name: Optional[str] = None
    """``state`` type component named ``agent-registry``."""

    runtime_conf: dict[str, str] = field(default_factory=dict)
    """Config loaded from ``agent-runtime`` statestore key ``agent_runtime``."""


def discover_components() -> DiscoveredComponents:
    """Discover Dapr sidecar components by convention.

    Calls ``DaprClient.get_metadata()`` and iterates registered components:
    - ``conversation.*`` type -> ``conversation_name`` (first found; warns if multiple)
    - ``configuration`` type + name ``agent-configuration`` -> ``configuration_name``
    - ``state`` type + name ``agent-memory`` -> ``memory_store_name``
    - ``pubsub`` type + name ``agent-pubsub`` -> ``pubsub_name``
    - ``state`` type + name ``agent-registry`` -> ``registry_store_name``
    - ``state`` type + name ``agent-runtime`` -> loads ``agent_runtime`` key into ``runtime_conf``

    Returns ``DiscoveredComponents()`` with empty defaults on sidecar error.
    Result is cached at module level.
    """
    global _cached_discovery
    if _cached_discovery is not None:
        return _cached_discovery

    result = DiscoveredComponents()

    try:
        with DaprClient(http_timeout_seconds=10) as client:
            resp = client.get_metadata()

            conversation_components: list[str] = []

            for component in resp.registered_components:
                ctype = component.type
                cname = component.name

                if ctype.startswith("conversation."):
                    conversation_components.append(cname)
                    if result.conversation_name is None:
                        result.conversation_name = cname

                if "configuration" in ctype and cname == "agent-configuration":
                    result.configuration_name = cname

                if "state" in ctype and cname == "agent-memory":
                    result.memory_store_name = cname

                if "pubsub" in ctype and cname == "agent-pubsub":
                    result.pubsub_name = cname

                if "state" in ctype and cname == "agent-registry":
                    result.registry_store_name = cname

                if "state" in ctype and cname == "agent-runtime":
                    try:
                        raw = client.get_state(
                            store_name=cname,
                            key="agent_runtime",
                        )
                        if raw.data:
                            result.runtime_conf = json.loads(raw.data)
                    except Exception as exc:
                        logger.debug("Failed to load agent-runtime config: %s", exc)

            if len(conversation_components) > 1:
                logger.warning(
                    "Multiple conversation components found: %s; using '%s'",
                    conversation_components,
                    result.conversation_name,
                )

    except Exception as exc:
        logger.debug("Dapr sidecar discovery failed: %s", exc)

    _cached_discovery = result
    return result


def _reset_discovery_cache() -> None:
    """Reset the module-level discovery cache (for testing)."""
    global _cached_discovery
    _cached_discovery = None
