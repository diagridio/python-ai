# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Diagrid Agent Smolagents - Durable execution of smolagents agents using Dapr Workflows.

This extension enables durable execution of Hugging Face `smolagents`
``ToolCallingAgent`` instances using Dapr Workflows. Each model call and each
tool execution runs as a Dapr Workflow activity, providing fault tolerance,
durability, and observability.

Only ``ToolCallingAgent`` is supported (not ``CodeAgent``) — ``CodeAgent``
executes an arbitrary, model-generated Python code blob per step with no
discrete per-tool-call boundary to checkpoint as a Dapr activity.

Example:
    ```python
    from smolagents import ToolCallingAgent, OpenAIServerModel, tool
    from diagrid.agent.smolagents import DaprWorkflowAgentRunner

    @tool
    def search_web(query: str) -> str:
        \"\"\"Search the web for information.

        Args:
            query: The search query.
        \"\"\"
        return f"Results for: {query}"

    agent = ToolCallingAgent(
        model=OpenAIServerModel(model_id="gpt-4o-mini"),
        tools=[search_web],
    )

    runner = DaprWorkflowAgentRunner(agent=agent, name="search-agent")
    runner.start()

    async for event in runner.run_async(
        task="What is the weather in Tokyo?",
        session_id="session-123",
    ):
        print(event)

    runner.shutdown()
    ```

Benefits:
    - Fault tolerance: Agents automatically resume from the last successful
      activity on failure or restart
    - Durability: Agent state is persisted and can survive process restarts
    - Observability: Full visibility into agent execution through Dapr's
      workflow APIs and dashboard
    - Scalability: Workflows can be distributed across multiple instances

Install:
    pip install diagrid[smolagents]
"""

from diagrid.agent.smolagents.runner import DaprWorkflowAgentRunner
from diagrid.agent.smolagents.state import DaprSessionStore
from diagrid.agent.smolagents.models import (
    AgentConfig,
    AgentWorkflowInput,
    AgentWorkflowOutput,
    CallModelInput,
    CallModelOutput,
    ChatEntry,
    ExecuteToolInput,
    ExecuteToolOutput,
    ToolCall,
    ToolDefinition,
)
from .workflow import (
    agent_workflow,
    call_model_activity,
    execute_tool_activity,
    register_model,
    register_tool,
    clear_registries,
)
from diagrid.agent.smolagents.version import __version__

__all__ = [
    # Main runner class
    "DaprWorkflowAgentRunner",
    # State
    "DaprSessionStore",
    # Data models
    "AgentConfig",
    "AgentWorkflowInput",
    "AgentWorkflowOutput",
    "CallModelInput",
    "CallModelOutput",
    "ChatEntry",
    "ExecuteToolInput",
    "ExecuteToolOutput",
    "ToolCall",
    "ToolDefinition",
    # Workflow and activities (for advanced usage)
    "agent_workflow",
    "call_model_activity",
    "execute_tool_activity",
    # Registries (for advanced usage)
    "register_model",
    "register_tool",
    "clear_registries",
    # Version
    "__version__",
]
