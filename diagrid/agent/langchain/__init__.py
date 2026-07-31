# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Diagrid Agent LangChain - Durable execution of LangChain agents using Dapr Workflows.

This extension enables durable execution of LangChain (``langchain_core``)
chat models and tools using Dapr Workflows. Each chat-model call and each
tool execution runs as a Dapr Workflow activity, providing fault tolerance,
durability, and observability.

LangChain 1.x's own ``langchain.agents.create_agent`` compiles directly to a
LangGraph graph — already covered by ``diagrid.agent.langgraph``. This
extension instead wraps ``langchain_core`` primitives (a ``BaseChatModel``
and a list of ``BaseTool``) directly and reimplements the tool-calling loop
as a Dapr Workflow.

Example:
    ```python
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from diagrid.agent.langchain import DaprWorkflowAgentRunner

    @tool
    def search_web(query: str) -> str:
        \"\"\"Search the web for information.\"\"\"
        return f"Results for: {query}"

    runner = DaprWorkflowAgentRunner(
        model=ChatOpenAI(model="gpt-4o-mini"),
        tools=[search_web],
        name="search-agent",
    )
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
    pip install diagrid[langchain]
"""

from diagrid.agent.langchain.runner import DaprWorkflowAgentRunner
from diagrid.agent.langchain.state import DaprSessionStore
from diagrid.agent.langchain.models import (
    AgentConfig,
    AgentWorkflowInput,
    AgentWorkflowOutput,
    CallLlmInput,
    CallLlmOutput,
    ExecuteToolInput,
    ExecuteToolOutput,
    Message,
    ToolCall,
    ToolDefinition,
)
from .workflow import (
    agent_workflow,
    call_llm_activity,
    execute_tool_activity,
    register_model,
    register_tool,
    clear_registries,
)
from diagrid.agent.langchain.version import __version__

__all__ = [
    # Main runner class
    "DaprWorkflowAgentRunner",
    # State
    "DaprSessionStore",
    # Data models
    "AgentConfig",
    "AgentWorkflowInput",
    "AgentWorkflowOutput",
    "CallLlmInput",
    "CallLlmOutput",
    "ExecuteToolInput",
    "ExecuteToolOutput",
    "Message",
    "ToolCall",
    "ToolDefinition",
    # Workflow and activities (for advanced usage)
    "agent_workflow",
    "call_llm_activity",
    "execute_tool_activity",
    # Registries (for advanced usage)
    "register_model",
    "register_tool",
    "clear_registries",
    # Version
    "__version__",
]
