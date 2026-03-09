# Diagrid Agent Google ADK Extension

This is the Diagrid Agent extension for Google ADK (Agent Development Kit) using Dapr Workflows.

This extension enables durable execution of Google ADK agents using Dapr Workflows.
Each tool execution in an agent runs as a separate Dapr Workflow activity, providing:

- **Fault tolerance**: Agents automatically resume from the last successful activity on failure
- **Durability**: Agent state is persisted and can survive process restarts
- **Observability**: Full visibility into agent execution through Dapr's workflow APIs

## Community

Have questions, hit a bug, or want to share what you're building? Join the [Diagrid Community Discord](https://diagrid.ws/diagrid-community) to connect with the team and other users.

## Installation

```bash
pip install diagrid
```

## Quick Start

```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from diagrid.agent.adk import DaprWorkflowAgentRunner

# Define your ADK agent as usual
agent = LlmAgent(
    name="my_agent",
    model="gemini-2.0-flash",
    tools=[my_tool],
)

# Create a Dapr workflow runner
runner = DaprWorkflowAgentRunner(
    agent=agent,
    state_store_name="agent-workflow",
)

# Run the agent - each tool execution is now a durable activity
async for event in runner.run_async(
    user_message="Hello, please help me with...",
    session_id="my-session",
):
    print(event)
```

## How It Works

The extension wraps your ADK agent execution in a Dapr Workflow:

1. **Workflow Start**: When you call `run_async()`, a new Dapr Workflow instance is created
2. **LLM Activity**: Each LLM call is executed as a durable activity
3. **Tool Activities**: Each tool execution is a separate durable activity
4. **Checkpointing**: After each activity, the workflow state is checkpointed
5. **Recovery**: On failure, the workflow resumes from the last successful activity

## Architecture

```text
DaprAgentWorkflow (orchestrates the agent loop)
+-- Activity: call_llm()          # Get next action from LLM
+-- Activity: execute_tool_1()    # First tool call
+-- Activity: call_llm()          # LLM processes tool result
+-- Activity: execute_tool_2()    # Second tool call
+-- ... continues until agent completes
```

## Requirements

- Python >= 3.10
- Dapr >= 1.16.0
- Google ADK >= 1.0.0
- A Dapr state store component configured
