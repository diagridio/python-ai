# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Optional

from .metadata import AgentRegistryAdapter
from .introspection import detect_framework

logger = logging.getLogger(__name__)


class AgentRegistryMixin:
    """Mixin to provide agent metadata registration functionality."""

    def _register_agent_metadata(
        self, agent: Any, framework: Optional[str] = None, registry: Optional[Any] = None
    ) -> None:
        """
        Register agent metadata with the registry.

        Args:
            agent: The agent or graph object to register.
            framework: Optional framework name. If None, will be auto-detected.
            registry: Optional registry configuration.
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
            AgentRegistryAdapter(registry=registry, framework=fw, agent=agent)

            # Mark as registered
            try:
                setattr(agent, "_diagrid_registered", True)
            except (AttributeError, TypeError):
                # Some objects might not allow setting attributes (e.g. some CompiledStateGraph versions)
                pass

        except Exception as e:
            logger.warning("Failed to register agent metadata: %s", e)