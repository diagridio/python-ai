# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Runner for executing LangChain Deep Agents as Dapr Workflows.

Deep Agents (from the ``deepagents`` package) compile down to standard
LangGraph ``CompiledStateGraph`` objects.  This runner is a thin wrapper
around the existing :class:`DaprWorkflowGraphRunner` that provides a
convenience API matching the Deep Agents harness conventions.
"""

import logging
from typing import Any, Optional, TYPE_CHECKING

from diagrid.agent.core.types import SupportedFrameworks
from diagrid.agent.langgraph.runner import DaprWorkflowGraphRunner

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class DaprWorkflowDeepAgentRunner(DaprWorkflowGraphRunner):
    """Runner that executes LangChain Deep Agents as Dapr Workflows.

    Since ``create_deep_agent()`` returns a compiled LangGraph
    (``CompiledStateGraph``), this runner delegates to the existing
    LangGraph integration.  It adds:

    * An ``agent`` alias accepted by the constructor (maps to ``graph``)
    * Sensible defaults for Deep Agent workloads (higher ``max_steps``)

    Example:
        ```python
        from deepagents import create_deep_agent
        from diagrid.agent.deepagents import DaprWorkflowDeepAgentRunner

        agent = create_deep_agent(model="openai:gpt-4o-mini")
        runner = DaprWorkflowDeepAgentRunner(agent=agent, name="my-agent")
        runner.start()

        result = runner.invoke(
            input={"messages": [{"role": "user", "content": "Hello"}]},
            thread_id="thread-1",
        )
        print(result)
        runner.shutdown()
        ```
    """

    #: Register under the Deep Agents framework identity (not LangGraph), so the
    #: registry and workflow name reflect ``DeepAgents`` and the dedicated
    #: ``DeepAgentsMapper`` is selected for metadata extraction.
    _REGISTRY_FRAMEWORK = SupportedFrameworks.DEEPAGENTS

    def __init__(
        self,
        agent: "CompiledStateGraph",
        *,
        name: str,
        host: Optional[str] = None,
        port: Optional[str] = None,
        max_steps: int = 100,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        registry_config: Optional[Any] = None,
    ):
        """Initialize the runner.

        Args:
            agent: A compiled Deep Agent graph (from ``create_deep_agent()``).
            name: Required name for the workflow.
            host: Dapr sidecar host (default: localhost).
            port: Dapr sidecar port (default: 50001).
            max_steps: Maximum graph steps before stopping (default: 100).
            role: Optional role description for the agent registry.
            goal: Optional goal description for the agent registry.
            registry_config: Optional registry configuration for metadata.
        """
        super().__init__(
            graph=agent,
            name=name,
            host=host,
            port=port,
            max_steps=max_steps,
            role=role,
            goal=goal,
            registry_config=registry_config,
        )

    @property
    def agent(self) -> "CompiledStateGraph":
        """The compiled Deep Agent graph."""
        return self._graph
