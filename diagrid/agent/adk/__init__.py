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

"""Diagrid Agent ADK - Durable execution of Google ADK agents using Dapr Workflows.

This extension enables durable execution of Google ADK agents using Dapr Workflows.
Each tool execution in an agent runs as a separate Dapr Workflow activity.

Example:
    ```python
    from google.adk.agents import LlmAgent
    from google.adk.tools import FunctionTool
    from diagrid.agent.adk import DaprWorkflowAgentRunner

    # Define your ADK agent
    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny."

    agent = LlmAgent(
        name="weather_agent",
        model="gemini-2.0-flash",
        tools=[FunctionTool(get_weather)],
    )

    # Create runner and start
    runner = DaprWorkflowAgentRunner(agent=agent)
    runner.start()

    # Run agent - each tool call is now durable
    async for event in runner.run_async(
        user_message="What's the weather in Tokyo?",
        session_id="session-123",
    ):
        print(event)

    runner.shutdown()
    ```

Install:
    pip install diagrid
"""

from diagrid.agent.adk.runner import DaprWorkflowAgentRunner
from diagrid.agent.adk.models import (
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
from diagrid.agent.adk.workflow import (
    adk_agent_workflow,
    call_llm_activity,
    execute_tool_activity,
    register_tool,
    get_registered_tool,
    clear_tool_registry,
)
from diagrid.agent.adk.version import __version__


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "DaprWorkflowPlugin":
        from diagrid.agent.adk.plugin import DaprWorkflowPlugin

        return DaprWorkflowPlugin
    if name == "PendingToolExecution":
        from diagrid.agent.adk.plugin import PendingToolExecution

        return PendingToolExecution
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    "adk_agent_workflow",
    "call_llm_activity",
    "execute_tool_activity",
    # Tool registry (for advanced usage)
    "register_tool",
    "get_registered_tool",
    "clear_tool_registry",
    # Plugin (for advanced usage)
    "DaprWorkflowPlugin",
    "PendingToolExecution",
    # Version
    "__version__",
]
