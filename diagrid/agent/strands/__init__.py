# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Diagrid Agent Strands - Durable execution of Strands agents using Dapr Workflows.

This extension enables durable execution of Strands agents using Dapr Workflows.
Each agent invocation runs as a Dapr Workflow activity, providing fault tolerance,
durability, and observability.

Example:
    ```python
    from strands import Agent, tool
    from diagrid.agent.strands import DaprWorkflowAgentRunner

    @tool
    def search_web(query: str) -> str:
        \"\"\"Search the web for information.\"\"\"
        return f"Results for: {query}"

    agent = Agent(
        model=my_model,
        tools=[search_web],
    )

    # Create runner and start the workflow runtime
    runner = DaprWorkflowAgentRunner(agent=agent)
    runner.start()

    # Run the agent - each invocation is now durable
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
    pip install diagrid
"""

from diagrid.agent.strands.runner import DaprWorkflowAgentRunner
from diagrid.agent.strands.durable_agent import DurableAgent
from diagrid.agent.strands.workflow import (
    DaprAgentWorkflow,
    dapr_agent_workflow,
    WorkflowInput,
    WorkflowOutput,
)
from diagrid.agent.strands.executor import DaprWorkflowToolExecutor
from diagrid.agent.strands.state import (
    DaprStateSessionManager,
    DaprWorkflowStateManager,
)
from diagrid.agent.strands.hooks import DaprWorkflowHookProvider, DaprRetryHookProvider
from diagrid.agent.strands.activities import (
    DaprToolActivity,
    create_tool_activity,
    register_tool_activities,
    ToolActivityRegistry,
)
from diagrid.agent.strands.version import __version__

__all__ = [
    # Main runner class
    "DaprWorkflowAgentRunner",
    # Simple API
    "DurableAgent",
    # Advanced API
    "DaprAgentWorkflow",
    "dapr_agent_workflow",
    "WorkflowInput",
    "WorkflowOutput",
    # Executor
    "DaprWorkflowToolExecutor",
    # State management
    "DaprStateSessionManager",
    "DaprWorkflowStateManager",
    # Hooks
    "DaprWorkflowHookProvider",
    "DaprRetryHookProvider",
    # Activities
    "DaprToolActivity",
    "create_tool_activity",
    "register_tool_activities",
    "ToolActivityRegistry",
    # Version
    "__version__",
]
