# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Runner for executing LangChain Deep Agents as Dapr Workflows.

Deep Agents (from the ``deepagents`` package) compile down to standard
LangGraph ``CompiledStateGraph`` objects.  This runner is a thin wrapper
around the existing :class:`DaprWorkflowGraphRunner` that provides a
convenience API matching the Deep Agents harness conventions.
"""

import logging
from typing import Any, AsyncIterator, Callable, Dict, Optional, TYPE_CHECKING

from langgraph.constants import START, END

from diagrid.agent.langgraph.runner import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.workflow import (
    register_node,
    register_channel_reducer,
    set_serializer,
    clear_registries,
)

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

    # ------------------------------------------------------------------
    # Override: register RunnableCallable wrappers, not raw functions
    # ------------------------------------------------------------------
    def _register_graph_components(self) -> None:
        """Register graph nodes, keeping RunnableCallable wrappers intact.

        Deep Agent middleware nodes (e.g. PatchToolCallsMiddleware) require
        a ``runtime`` argument that is injected by ``RunnableCallable.invoke()``.
        The base class extracts ``bound.func`` (the raw function), which
        bypasses that injection.  Here we register ``bound`` itself so that
        ``execute_node_activity`` can call ``.invoke()`` on it.
        """
        clear_registries()

        graph_nodes = getattr(self._graph, "nodes", {})
        for node_name, node_spec in graph_nodes.items():
            if node_name in (START, END, "__start__", "__end__"):
                continue

            node_obj = None

            if hasattr(node_spec, "bound"):
                # Register the RunnableCallable wrapper, NOT bound.func
                node_obj = node_spec.bound
            elif hasattr(node_spec, "runnable"):
                node_obj = node_spec.runnable
            elif callable(node_spec):
                node_obj = node_spec

            if node_obj:
                register_node(node_name, node_obj)
                logger.info(f"Registered node (deep agent): {node_name}")
            else:
                logger.warning(f"Could not extract callable for node: {node_name}")

        channels = getattr(self._graph, "channels", {})
        for channel_name, channel in channels.items():
            # BinaryOperatorAggregate uses "operator" not "reducer"
            reducer = getattr(channel, "reducer", None) or getattr(
                channel, "operator", None
            )
            if reducer:
                register_channel_reducer(channel_name, reducer)
                logger.info(f"Registered reducer for channel: {channel_name}")

        try:
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

            set_serializer(JsonPlusSerializer())
        except ImportError:
            logger.warning(
                "JsonPlusSerializer not available, using basic serialization"
            )

    @property
    def agent(self) -> "CompiledStateGraph":
        """The compiled Deep Agent graph."""
        return self._graph
