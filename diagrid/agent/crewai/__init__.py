# Copyright 2025 Diagrid Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diagrid Agent CrewAI - Durable execution of CrewAI agents using Dapr Workflows.

This extension enables durable execution of CrewAI agents using Dapr Workflows.
Each tool execution in an agent runs as a separate Dapr Workflow activity,
providing fault tolerance, durability, and observability.

Example:
    ```python
    from crewai import Agent, Task
    from crewai.tools import tool
    from diagrid.agent.crewai import DaprWorkflowAgentRunner

    @tool("Search the web for information")
    def search_web(query: str) -> str:
        return f"Results for: {query}"

    # Create your CrewAI agent
    agent = Agent(
        role="Research Assistant",
        goal="Help users find accurate information",
        backstory="An expert researcher with access to various tools",
        tools=[search_web],
        llm="openai/gpt-4o-mini",
    )

    # Define a task
    task = Task(
        description="Research the latest developments in AI agents",
        expected_output="A comprehensive summary of recent AI agent news",
        agent=agent,
    )

    # Create runner and start the workflow runtime
    runner = DaprWorkflowAgentRunner(agent=agent)
    runner.start()

    # Run the agent - each tool call is now durable
    async for event in runner.run_async(task=task, session_id="session-123"):
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

from diagrid.agent.crewai.runner import DaprWorkflowAgentRunner
from diagrid.agent.crewai.models import (
    AgentConfig,
    AgentWorkflowInput,
    AgentWorkflowOutput,
    CallLlmInput,
    CallLlmOutput,
    ExecuteToolInput,
    ExecuteToolOutput,
    Message,
    MessageRole,
    TaskConfig,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from diagrid.agent.crewai.workflow import (
    agent_workflow,
    call_llm_activity,
    execute_tool_activity,
    register_tool,
    get_registered_tool,
    clear_tool_registry,
)
from diagrid.agent.crewai.version import __version__

__all__ = [
    # Main runner class
    "DaprWorkflowAgentRunner",
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
    "TaskConfig",
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
