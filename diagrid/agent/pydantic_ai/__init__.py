"""Diagrid Agent Pydantic AI - Durable execution of Pydantic AI agents using Dapr Workflows.

This extension enables durable execution of Pydantic AI agents using Dapr Workflows.
Each tool execution in an agent runs as a separate Dapr Workflow activity,
providing fault tolerance, durability, and observability.

Example:
    ```python
    from pydantic_ai import Agent
    from diagrid.agent.pydantic_ai import DaprWorkflowAgentRunner

    def search_web(query: str) -> str:
        return f"Results for: {query}"

    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt="You help users find accurate information.",
        tools=[search_web],
    )

    runner = DaprWorkflowAgentRunner(agent=agent)
    runner.start()

    async for event in runner.run_async(
        user_message="Search for recent AI news",
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

from diagrid.agent.pydantic_ai.runner import DaprWorkflowAgentRunner
from diagrid.agent.pydantic_ai.state import DaprMemoryStore
from diagrid.agent.pydantic_ai.models import (
    AgentConfig,
    AgentWorkflowInput,
    AgentWorkflowOutput,
    CallLlmInput,
    CallLlmOutput,
    ExecuteToolInput,
    ExecuteToolOutput,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from diagrid.agent.pydantic_ai.workflow import (
    agent_workflow,
    call_llm_activity,
    execute_tool_activity,
    register_tool,
    get_registered_tool,
    clear_tool_registry,
)
from diagrid.agent.pydantic_ai.version import __version__

__all__ = [
    # Main runner class
    "DaprWorkflowAgentRunner",
    # State
    "DaprMemoryStore",
    # Data models
    "AgentConfig",
    "AgentWorkflowInput",
    "AgentWorkflowOutput",
    "CallLlmInput",
    "CallLlmOutput",
    "ExecuteToolInput",
    "ExecuteToolOutput",
    "Message",
    "MessageRole",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    # Workflow and activities (for advanced usage)
    "agent_workflow",
    "call_llm_activity",
    "execute_tool_activity",
    # Tool registry (for advanced usage)
    "register_tool",
    "get_registered_tool",
    "clear_tool_registry",
    # Version
    "__version__",
]
