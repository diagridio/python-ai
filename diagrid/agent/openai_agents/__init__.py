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

"""Diagrid Agent OpenAI Agents SDK - Durable execution of OpenAI agents using Dapr Workflows.

This extension enables durable execution of OpenAI Agents SDK agents using Dapr Workflows.
Each tool execution in an agent runs as a separate Dapr Workflow activity,
providing fault tolerance, durability, and observability.

Example:
    ```python
    from agents import Agent, function_tool
    from diagrid.agent.openai_agents import DaprWorkflowAgentRunner

    @function_tool
    def search_web(query: str) -> str:
        return f"Results for: {query}"

    agent = Agent(
        name="research_assistant",
        instructions="You help users find accurate information.",
        model="gpt-4o-mini",
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

from diagrid.agent.openai_agents.runner import DaprWorkflowAgentRunner
from diagrid.agent.openai_agents.models import (
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
from diagrid.agent.openai_agents.workflow import (
    openai_agents_workflow,
    call_llm_activity,
    execute_tool_activity,
    register_tool,
    get_registered_tool,
    clear_tool_registry,
)
from diagrid.agent.openai_agents.version import __version__

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
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    # Workflow and activities (for advanced usage)
    "openai_agents_workflow",
    "call_llm_activity",
    "execute_tool_activity",
    # Tool registry (for advanced usage)
    "register_tool",
    "get_registered_tool",
    "clear_tool_registry",
    # Version
    "__version__",
]
