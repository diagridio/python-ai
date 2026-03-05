# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import logging
from typing import Any, Optional

from .metadata import AgentRegistryAdapter
from .introspection import detect_framework

logger = logging.getLogger(__name__)


class AgentRegistryMixin:
    """Mixin to provide agent metadata registration functionality."""

    def _register_agent_metadata(
        self,
        agent: Any,
        framework: Optional[str] = None,
        registry: Optional[Any] = None,
        component_name: Optional[str] = None,
        state_store_name: Optional[str] = None,
    ) -> None:
        """
        Register agent metadata with the registry.

        Args:
            agent: The agent or graph object to register.
            framework: Optional framework name. If None, will be auto-detected.
            registry: Optional registry configuration.
            component_name: Optional Dapr conversation component name resolved at runtime.
            state_store_name: Optional Dapr state store name resolved at runtime.
        """
        try:
            # Avoid duplicate registration for the same agent object in the same process
            if getattr(agent, "_diagrid_registered", False):
                return

            fw = framework or detect_framework(agent)
            if not fw:
                logger.debug("Could not detect framework for agent registry")
                return

            # This will extract metadata and register it if a registry is configured/detected
            AgentRegistryAdapter(
                registry=registry,
                framework=fw,
                agent=agent,
                component_name=component_name,
                state_store_name=state_store_name,
            )

            # Mark as registered
            try:
                setattr(agent, "_diagrid_registered", True)
            except (AttributeError, TypeError):
                # Some objects might not allow setting attributes (e.g. some CompiledStateGraph versions)
                pass

        except Exception as e:
            logger.warning("Failed to register agent metadata: %s", e)
